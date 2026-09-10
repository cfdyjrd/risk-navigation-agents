"""Bounded read-only G1 telemetry cache: DDS state + GET_FSM (API 7001).

Never imports a locomotion command client. Raw status is NOT motion-ready
perception. Use g1_lidar_probe.py separately for ROS lidar metadata.
"""
import argparse
from collections import deque
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


LIVENESS_MIN_SAMPLES = 5
LIVENESS_MIN_SPAN_S = .2
LIVENESS_MAX_GAP_S = .5
LIVENESS_MAX_OFFSET_SPREAD_S = .1
LIVENESS_MIN_CLOCK_RATE_RATIO = .8
LIVENESS_MAX_CLOCK_RATE_RATIO = 1.2


def _finite_number(value):
    return type(value) in (int, float) and value == value and abs(value) != float("inf")


class StreamLiveness:
    """Bounded evidence that a DDS writer clock advances with local monotonic time.

    This does not change either clock. It lets a receiver distinguish a stable
    clock offset from an old/frozen sample stream without replacing the original
    DDS source timestamps.
    """

    def __init__(self, window_size=64):
        if type(window_size) is not int or window_size < LIVENESS_MIN_SAMPLES:
            raise ValueError("window_size is too small")
        self.samples = deque(maxlen=window_size)

    def observe(self, source_ns, received_wall_ns=None, received_monotonic_ns=None):
        received_wall_ns = time.time_ns() if received_wall_ns is None else received_wall_ns
        received_monotonic_ns = (time.monotonic_ns() if received_monotonic_ns is None
                                 else received_monotonic_ns)
        if any(type(value) is not int or value <= 0 for value in
               (source_ns, received_wall_ns, received_monotonic_ns)):
            raise ValueError("clock samples must be positive integer nanoseconds")
        self.samples.append((source_ns, received_wall_ns, received_monotonic_ns))
        return self.snapshot()

    def snapshot(self):
        rows = list(self.samples)
        if len(rows) < 2:
            source_span = receipt_span = max_gap = offset_spread = 0.
            rate_ratio = None
            source_regressions = receipt_regressions = 0
        else:
            source_steps = [right[0] - left[0] for left, right in zip(rows, rows[1:])]
            receipt_steps = [right[2] - left[2] for left, right in zip(rows, rows[1:])]
            source_regressions = sum(step <= 0 for step in source_steps)
            receipt_regressions = sum(step <= 0 for step in receipt_steps)
            source_span = (rows[-1][0] - rows[0][0]) / 1e9
            receipt_span = (rows[-1][2] - rows[0][2]) / 1e9
            max_gap = max(receipt_steps) / 1e9
            offsets = [(wall - source) / 1e9 for source, wall, _mono in rows]
            offset_spread = max(offsets) - min(offsets)
            rate_ratio = source_span / receipt_span if receipt_span > 0 else None
        verified = (
            len(rows) >= LIVENESS_MIN_SAMPLES and
            source_span >= LIVENESS_MIN_SPAN_S and
            receipt_span >= LIVENESS_MIN_SPAN_S and
            source_regressions == 0 and receipt_regressions == 0 and
            rate_ratio is not None and
            LIVENESS_MIN_CLOCK_RATE_RATIO <= rate_ratio <= LIVENESS_MAX_CLOCK_RATE_RATIO and
            max_gap <= LIVENESS_MAX_GAP_S and
            offset_spread <= LIVENESS_MAX_OFFSET_SPREAD_S
        )
        offset_estimate = None
        if rows:
            # The minimum observed local-minus-source value has the least
            # contribution from queueing/transport delay.
            offset_estimate = min((wall - source) / 1e9 for source, wall, _mono in rows)
        return {
            "verified": verified,
            "basis": "advancing_dds_source_and_local_monotonic_receipts",
            "sample_count": len(rows),
            "source_span_s": source_span,
            "receipt_span_s": receipt_span,
            "source_clock_rate_ratio": rate_ratio,
            "source_regressions": source_regressions,
            "receipt_regressions": receipt_regressions,
            "max_receipt_gap_s": max_gap,
            "clock_offset_estimate_s": offset_estimate,
            "clock_offset_spread_s": offset_spread,
        }


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


def _parse_timestamp(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("timezone missing")
    return stamp


def _valid_liveness(record, source_stamp, receipt_stamp):
    evidence = record.get("liveness")
    if not isinstance(evidence, dict) or evidence.get("verified") is not True:
        return False
    numeric_limits = {
        "sample_count": (LIVENESS_MIN_SAMPLES, None),
        "source_span_s": (LIVENESS_MIN_SPAN_S, None),
        "receipt_span_s": (LIVENESS_MIN_SPAN_S, None),
        "source_clock_rate_ratio": (LIVENESS_MIN_CLOCK_RATE_RATIO,
                                    LIVENESS_MAX_CLOCK_RATE_RATIO),
        "max_receipt_gap_s": (0., LIVENESS_MAX_GAP_S),
        "clock_offset_spread_s": (0., LIVENESS_MAX_OFFSET_SPREAD_S),
    }
    for key, (minimum, maximum) in numeric_limits.items():
        value = evidence.get(key)
        if not _finite_number(value) or value < minimum or (maximum is not None and value > maximum):
            return False
    if type(evidence.get("sample_count")) is not int:
        return False
    if evidence.get("source_regressions") != 0 or evidence.get("receipt_regressions") != 0:
        return False
    estimate = evidence.get("clock_offset_estimate_s")
    if not _finite_number(estimate):
        return False
    current_offset = (receipt_stamp - source_stamp).total_seconds()
    tolerance = max(LIVENESS_MAX_OFFSET_SPREAD_S,
                    evidence["clock_offset_spread_s"] + .05)
    return abs(current_offset - estimate) <= tolerance


def record_freshness(record, now=None, max_age_s=1.):
    """Return source and effective freshness without rewriting DDS time.

    A synchronized DDS source timestamp is preferred. When clocks differ, a
    local receipt timestamp is accepted only with bounded, advancing-clock
    evidence generated from a sliding monotonic-time window.
    """
    now = now or datetime.now(timezone.utc)
    result = {"age_s": None, "source_age_s": None, "receipt_age_s": None,
              "fresh": False, "basis": "invalid", "effective_timestamp": None,
              "clock_offset_s": None, "liveness_verified": False}
    if not _finite_number(max_age_s) or max_age_s <= 0:
        return result
    try:
        source_stamp = _parse_timestamp(record["timestamp"])
        source_age = (now - source_stamp).total_seconds()
    except (KeyError, TypeError, ValueError, AttributeError):
        return result
    result.update(source_age_s=source_age, age_s=source_age,
                  basis="dds_source_timestamp", effective_timestamp=source_stamp.isoformat())
    if record.get("timestamp_basis") != "dds_writer_source_timestamp":
        result["fresh"] = 0 <= source_age <= max_age_s
        return result
    try:
        receipt_stamp = _parse_timestamp(record["received_at"])
        receipt_age = (now - receipt_stamp).total_seconds()
        current_offset = (receipt_stamp - source_stamp).total_seconds()
    except (KeyError, TypeError, ValueError, AttributeError):
        return result
    result.update(receipt_age_s=receipt_age, clock_offset_s=current_offset)
    if not _valid_liveness(record, source_stamp, receipt_stamp):
        return result
    result["liveness_verified"] = True
    if 0 <= source_age <= max_age_s:
        result.update(fresh=True, basis="verified_dds_source_timestamp")
        return result
    if 0 <= receipt_age <= max_age_s:
        result.update(age_s=receipt_age, fresh=True,
                      basis="verified_local_receipt_with_preserved_dds_source",
                      effective_timestamp=receipt_stamp.isoformat())
    return result


def state_freshness(state, now=None):
    now = now or datetime.now(timezone.utc)
    return {key: record_freshness(state.get(key, {}), now=now)
            for key in ("lowstate", "odometry", "battery", "fsm")}


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
    trackers = {key: StreamLiveness() for key, _name, _kind in specs}
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
                        source_ns = sample.sample_info.source_timestamp
                        source_timestamp = dds_timestamp(sample)
                        received_wall_ns = time.time_ns()
                        received_monotonic_ns = time.monotonic_ns()
                        received_at = datetime.fromtimestamp(received_wall_ns / 1e9, timezone.utc).isoformat()
                        liveness = trackers[key].observe(source_ns, received_wall_ns,
                                                        received_monotonic_ns)
                        record = {"timestamp": source_timestamp,
                                  "source_timestamp_ns": source_ns, "source": name,
                                  "timestamp_basis": "dds_writer_source_timestamp",
                                  "received_at": received_at, "liveness": liveness}
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
                            if key not in state or source_ns > state[key].get("source_timestamp_ns", 0):
                                state[key] = record
                            else:
                                # Preserve the newest payload but expose a duplicate/regression
                                # immediately so freshness cannot remain accidentally trusted.
                                state[key]["liveness"] = liveness
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
    if not 1 <= args.duration <= 1200:
        parser.error("duration must be within 1–1200 seconds")
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
