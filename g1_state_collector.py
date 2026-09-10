"""Bounded read-only G1 telemetry cache: DDS state + GET_FSM (API 7001).

Never imports a locomotion command client. Raw status is NOT motion-ready
perception. Use g1_lidar_probe.py separately for ROS lidar metadata.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time

from g1_dds_diagnostic import _progress, _run_supervised


def atomic_json(path, data):
    rendered = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".g1-state-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(rendered + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_fsm(code, data):
    if type(code) is not int or code != 0:
        raise ValueError("GET_FSM failed: %r" % code)
    value = json.loads(data) if isinstance(data, str) else data
    if isinstance(value, dict):
        value = value.get("data")
    if type(value) is not int or value < 0:
        raise ValueError("GET_FSM returned invalid data")
    return value


def dds_timestamp(sample):
    # DDS writer timestamp, distinct from local callback/receipt time.
    value = sample.sample_info.source_timestamp
    if type(value) is not int or value <= 0:
        raise ValueError("DDS source timestamp missing")
    return datetime.fromtimestamp(value / 1e9, timezone.utc).isoformat()


def state_freshness(state, now=None):
    now = now or datetime.now(timezone.utc)
    result = {}
    for key in ("lowstate", "odometry", "battery", "fsm"):
        try:
            stamp = datetime.fromisoformat(state[key]["timestamp"].replace("Z", "+00:00"))
            age = (now - stamp).total_seconds()
            result[key] = {"age_s": age, "fresh": 0 <= age <= 1.}
        except (KeyError, TypeError, ValueError, AttributeError):
            result[key] = {"age_s": None, "fresh": False}
    return result


def collect(args):
    _progress("加载 DDS 并订阅 G1 底层状态、里程计、电量……")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.rpc.client import Client
    from unitree_sdk2py.g1.loco.g1_loco_api import LOCO_SERVICE_NAME, LOCO_API_VERSION
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic

    ChannelFactoryInitialize(0, args.interface)
    participant = DomainParticipant(0)
    specs = [("lowstate", "rt/lowstate", LowState_),
             ("odometry", "rt/odommodestate", SportModeState_),
             ("battery", "rt/lf/bmsstate", BmsState_)]
    topics = [Topic(participant, name, kind) for _, name, kind in specs]
    readers = [DataReader(participant, topic) for topic in topics]
    # Register only the read-only GET_FSM API. No velocity/FSM setters exist here.
    rpc = Client(LOCO_SERVICE_NAME, False)
    rpc.SetTimeout(1.)
    rpc._SetApiVerson(LOCO_API_VERSION)
    rpc._RegistApi(7001, 0)
    state, errors = {}, {}
    lock, stop = threading.Lock(), threading.Event()

    def read_fsm():
        while not stop.is_set():
            start = datetime.now(timezone.utc).isoformat()
            try:
                fsm_id = parse_fsm(*rpc._Call(7001, "{}"))
                with lock:
                    state["fsm"] = {"timestamp": start, "fsm_id": fsm_id,
                                    "source": "G1 GET_FSM API 7001",
                                    "timestamp_basis": "request_start_conservative"}
                    errors.pop("fsm", None)
            except Exception as exc:
                with lock:
                    errors["fsm"] = str(exc)
            stop.wait(.25)

    thread = threading.Thread(target=read_fsm, daemon=True)
    thread.start()
    counts = {key: 0 for key, _, _ in specs}
    end, next_print = time.monotonic() + args.duration, time.monotonic()
    _progress("开始采集；仅发送 GET_FSM 查询，不发送运动指令。")
    try:
        while time.monotonic() < end:
            for (key, name, kind), reader in zip(specs, readers):
                for sample in reader.take(64):
                    if not isinstance(sample, kind):
                        continue
                    counts[key] += 1
                    try:
                        record = {"timestamp": dds_timestamp(sample), "source": name,
                                  "timestamp_basis": "dds_writer_source_timestamp",
                                  "received_at": datetime.now(timezone.utc).isoformat()}
                        if key == "lowstate":
                            record.update(tick=sample.tick, rpy_rad=list(sample.imu_state.rpy))
                        elif key == "battery":
                            if type(sample.soc) is not int or not 0 <= sample.soc <= 100:
                                raise ValueError("invalid battery SOC")
                            record["soc"] = sample.soc
                        else:
                            record.update(position_m=list(sample.position), velocity_mps=list(sample.velocity),
                                          yaw_rate_rps=sample.yaw_speed, error_code=sample.error_code)
                        json.dumps(record, allow_nan=False)
                        with lock:
                            if key not in state or record["timestamp"] > state[key]["timestamp"]:
                                state[key] = record
                            errors.pop(key, None)
                    except Exception as exc:
                        with lock:
                            errors[key] = str(exc)
            with lock:
                report = {"collector": "g1_state_collector", "device_id": args.device_id, "interface": args.interface,
                          "collected_at": datetime.now(timezone.utc).isoformat(),
                          "state": dict(state), "errors": dict(errors), "samples": dict(counts),
                          "motion_commands_sent": 0, "motion_ready": False,
                          "missing_inputs": ["validated robot-frame obstacle perception", "measured robot width",
                                             "onsite FSM/tilt limits and data-source identity verification"]}
            report["freshness"] = state_freshness(report["state"])
            report["state_cache_ready"] = all(item["fresh"] for item in report["freshness"].values()) and not report["errors"]
            atomic_json(args.output, report)
            if time.monotonic() >= next_print:
                _progress("状态=%s，电量=%s，FSM=%s，错误=%s" % (
                    counts, report["state"].get("battery", {}).get("soc"),
                    report["state"].get("fsm", {}).get("fsm_id"), report["errors"]))
                if not report["state_cache_ready"]:
                    _progress("缓存尚未就绪：" + str(report["freshness"]))
                next_print = time.monotonic() + 1
            time.sleep(.05)
    finally:
        stop.set()
        thread.join(1.5)
    _progress("采集结束；缓存保留原始时间戳，停止后会自然过期。")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if report["state_cache_ready"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--device-id", default="G1")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interface not in dict(socket.if_nameindex()).values():
        parser.error("network interface does not exist")
    if not 1 <= args.duration <= 300:
        parser.error("duration must be within 1–300 seconds")
    if not args.device_id.strip():
        parser.error("device-id is required")
    args.output = args.output.expanduser().absolute()
    if not args.output.parent.is_dir() or args.output.suffix != ".json":
        parser.error("output must be a .json file in an existing directory")
    if args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
            if existing.get("collector") != "g1_state_collector":
                parser.error("refusing to overwrite a file not owned by this collector")
        except (ValueError, AttributeError):
            parser.error("refusing to overwrite an unrelated output file")
    if not args.worker:
        return _run_supervised([sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                                "--interface", args.interface, "--device-id", args.device_id,
                                "--duration", str(args.duration), "--output", str(args.output)], args.duration + 15)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
