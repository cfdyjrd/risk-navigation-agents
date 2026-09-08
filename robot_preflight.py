"""Read-only preflight CLI. No robot client instances or motion command calls.

Consumes an atomically updated JSON snapshot exported by the sensor process.
Local SDK loading and cache health do NOT establish robot network reachability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import importlib.metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
import tempfile
import time

from robot_interface import RobotObservation
from unitree_adapter import UnitreeConfig, load_go1_sdk, validate_unitree_observation

MAX_JSON_BYTES = 1024 * 1024


def write_observation_snapshot(path, observation: RobotObservation):
    """Sensor-side helper: preserve acquisition time and replace JSON atomically."""
    observation.validate()
    rendered = json.dumps(asdict(observation), ensure_ascii=False, allow_nan=False)
    if len(rendered.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("observation exceeds 1 MiB limit")
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=".observation-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(rendered + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    if not stat.S_ISREG(Path(path).stat().st_mode):
        raise ValueError("JSON input must be a regular file")
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds 1 MiB limit")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def validate_config(data, base_dir):
    unknown = set(data) - {"robot", "connection", "observation_file"}
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")
    config = UnitreeConfig(**data["robot"])
    connection = data["connection"]
    if not isinstance(connection, dict):
        raise ValueError("connection must be an object")
    allowed = {"network_interface"} if config.model == "go2" else {"robot_ip", "sdk_extension", "local_port", "remote_port"}
    if set(connection) - allowed:
        raise ValueError("connection fields do not match robot model")
    if config.model == "go2":
        interface = connection.get("network_interface")
        if not isinstance(interface, str) or not interface.strip():
            raise ValueError("set connection.network_interface to the actual Go2 interface")
    else:
        ipaddress.ip_address(connection.get("robot_ip", ""))
        extension = connection.get("sdk_extension")
        if not isinstance(extension, str) or not Path(extension).is_absolute() or not extension.endswith(".so"):
            raise ValueError("set connection.sdk_extension to an absolute robot_interface*.so path")
        for key, default in (("local_port", 8080), ("remote_port", 8082)):
            port = connection.get(key, default)
            if type(port) is not int or not 1 <= port <= 65535:
                raise ValueError(f"invalid {key}")
    snapshot = data.get("observation_file")
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise ValueError("observation_file must point to the sensor process JSON snapshot")
    path = Path(snapshot).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return config, connection, path.resolve()


def sdk_probe(model, extension):
    """Runs only in the subprocess; does not instantiate SDK clients."""
    if model == "go1":
        module = load_go1_sdk(extension)
        for symbol in ("UDP", "HighCmd", "HighState"):
            if not hasattr(module, symbol):
                raise ValueError(f"Go1 SDK missing {symbol}")
        return {"module": str(extension), "version": "legacy extension; verify release on host"}
    channel = importlib.import_module("unitree_sdk2py.core.channel")
    sport = importlib.import_module("unitree_sdk2py.go2.sport.sport_client")
    ChannelFactoryInitialize = getattr(channel, "ChannelFactoryInitialize", None)
    ChannelSubscriber = getattr(channel, "ChannelSubscriber", None)
    SportClient = getattr(sport, "SportClient", None)
    if not all(callable(item) for item in (ChannelFactoryInitialize, ChannelSubscriber, SportClient)):
        raise ValueError("Go2 SDK API mismatch")
    try:
        version = importlib.metadata.version("unitree_sdk2py")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (source installation)"
    return {"module": "unitree_sdk2py", "version": version}


def check_sdk(config, connection, timeout=10):
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--sdk-probe"],
            input=json.dumps({"model": config.model, "extension": connection.get("sdk_extension")}),
            text=True, capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return "fail", {"error": f"SDK import timed out after {timeout}s"}
    if completed.returncode:
        return "fail", {"error": (completed.stderr or completed.stdout)[-2000:],
                        "returncode": completed.returncode}
    try:
        return "pass", json.loads(completed.stdout)
    except ValueError:
        return "fail", {"error": "SDK probe produced an invalid report"}


def check_observations(config, path, duration, interval):
    """Poll snapshots and require advancing source timestamps, not file mtimes."""
    start = time.monotonic()
    deadline = start + duration
    errors = []
    attempts = valid = updates = 0
    first_stamp = last_stamp = None
    while True:
        attempts += 1
        try:
            observation = RobotObservation(**read_json(path))
            validate_unitree_observation(observation, config)
            stamp = datetime.fromisoformat(observation.timestamp.replace("Z", "+00:00"))
            if last_stamp is not None and stamp < last_stamp:
                raise ValueError("sensor timestamp moved backwards")
            valid += 1
            if last_stamp is None or stamp > last_stamp:
                updates += 1
                first_stamp = first_stamp or stamp
                last_stamp = stamp
        except Exception as exc:
            if len(errors) < 20:
                errors.append({"sample": attempts, "error": str(exc)})
        if time.monotonic() >= deadline:
            break
        time.sleep(min(interval, max(0, deadline - time.monotonic())))
    if updates < 2:
        errors.append({"error": "need at least two distinct advancing source timestamps; cache is missing or frozen"})
    span = (last_stamp - first_stamp).total_seconds() if first_stamp else 0
    return ("pass" if not errors else "fail"), {
        "path": str(path), "polls": attempts, "valid_polls": valid,
        "distinct_samples": updates, "observed_update_hz": (updates - 1) / span if span > 0 else 0,
        "duration_s": time.monotonic() - start, "errors": errors,
        "rate_note": "Rate of sampled timestamps; polling may miss intermediate sensor updates.",
    }


def run_preflight(config_path, *, duration=3.0, interval=0.1, skip_sdk=False):
    for value, name, lower, upper in ((duration, "duration", .1, 60), (interval, "interval", .01, 5)):
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be within [{lower}, {upper}]")
    if duration < 2 * interval:
        raise ValueError("duration must be at least twice interval")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_local_and_sensor_cache", "motion_commands_sent": 0,
        "robot_connectivity": "not_verified", "motion_ready": False,
        "host": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
        "checks": [],
    }

    def add(name, status, detail):
        report["checks"].append({"name": name, "status": status, "detail": detail})

    try:
        path = Path(config_path).resolve()
        config, connection, snapshot = validate_config(read_json(path), path.parent)
        report["device_id"], report["model"] = config.device_id, config.model
        add("configuration", "pass", {"config_path": str(path)})
    except Exception as exc:
        add("configuration", "fail", {"error": str(exc)})
        report["status"] = "fail"
        return report

    if config.model == "go2":
        try:
            interfaces = [name for _, name in socket.if_nameindex()]
            present = connection["network_interface"] in interfaces
            add("network_configuration", "pass" if present else "fail", {
                "interface_exists": present, "interfaces": interfaces,
                "note": "Local interface presence only; link, DDS and robot identity not verified."})
        except OSError as exc:
            add("network_configuration", "fail", {"error": str(exc)})
    else:
        add("network_configuration", "pass", {
            "note": "IP and port syntax checked only; no UDP socket opened and no reachability test."})
    if skip_sdk:
        add("sdk_load", "skipped", {"note": "SDK check explicitly skipped"})
    else:
        add("sdk_load", *check_sdk(config, connection))
    add("observation_stream", *check_observations(config, snapshot, duration, interval))
    statuses = [check["status"] for check in report["checks"]]
    report["status"] = "fail" if "fail" in statuses else "incomplete" if "skipped" in statuses else "pass"
    report["limitations"] = [
        "Pass means local prerequisites and sensor-cache checks passed, not robot connection or motion approval.",
        "SDK state subscription, sensor provenance, physical device identity, control authority and independent stop require onsite verification.",
    ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Go1 / Go2-1 只读预检：配置、SDK 加载、JSON 观测更新；不发送控制指令。")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--duration", type=float, default=3, help="观测检查秒数，0.1–60（默认 3）")
    parser.add_argument("--interval", type=float, default=.1, help="轮询间隔秒数（默认 0.1）")
    parser.add_argument("--skip-sdk", action="store_true", help="仅检查配置和观测，报告标记 incomplete")
    parser.add_argument("--output", type=Path, help="另存 JSON 报告（不得覆盖配置或观测文件）")
    parser.add_argument("--sdk-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.sdk_probe:
        try:
            request = json.load(sys.stdin)
            print(json.dumps(sdk_probe(request["model"], request.get("extension"))))
            return 0
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if not args.config:
        parser.error("--config is required")
    try:
        if args.output:
            output = args.output.resolve()
            source = args.config.resolve()
            if output == source:
                raise ValueError("output cannot overwrite config")
            # Check even incomplete configs so report output cannot destroy a live snapshot.
            observation_file = read_json(source).get("observation_file")
            if isinstance(observation_file, str):
                snapshot = Path(observation_file).expanduser()
                if not snapshot.is_absolute():
                    snapshot = source.parent / snapshot
                if output == snapshot.resolve():
                    raise ValueError("output cannot overwrite observation_file")
        report = run_preflight(args.config, duration=args.duration, interval=args.interval, skip_sdk=args.skip_sdk)
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        print(rendered)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if report["status"] == "pass" else 1
    except (ValueError, OSError) as exc:
        print(f"预检工具错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
