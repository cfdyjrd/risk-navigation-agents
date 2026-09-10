"""Passive G1 DDS check, compatible with robot Python 3.8; no command clients.

Reports received state only. Does not export RobotObservation: battery,
obstacle perception and an independently sampled locomotion FSM are missing.
"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from xml.sax.saxutils import quoteattr


def _progress(message):
    print(message, file=sys.stderr, flush=True)


def _run_supervised(command, timeout_s):
    """Bound SDK import, initialization, sampling and native DDS teardown."""
    try:
        return subprocess.run(command, timeout=timeout_s, check=False).returncode
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps the child, including its DDS threads.
        _progress("诊断总超时（%g 秒），已结束只读采样进程。请查看最后一条阶段提示。" % timeout_s)
        return 124
    except KeyboardInterrupt:
        _progress("已取消只读诊断。")
        return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--duration", type=float, default=5)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interface not in dict(socket.if_nameindex()).values():
        parser.error("network interface does not exist")
    if not 1 <= args.duration <= 30:
        parser.error("duration must be within 1–30 seconds")

    if not args.worker:
        timeout_s = args.duration + 10
        _progress("开始 G1 只读诊断：网卡 %s，采样 %g 秒，总超时 %g 秒。" %
                  (args.interface, args.duration, timeout_s))
        return _run_supervised(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
             "--interface", args.interface, "--duration", str(args.duration)], timeout_s)

    _progress("[1/3] 加载 DDS / G1 状态消息模块……")
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    config = ('<CycloneDDS><Domain Id="0"><General><Interfaces>'
              '<NetworkInterface name=' + quoteattr(args.interface) + '/>'
              '</Interfaces></General></Domain></CycloneDDS>')
    _progress("[2/3] 初始化 DDS，并创建只读订阅……")
    domain = Domain(0, config)
    participant = DomainParticipant(0)
    specs = [("rt/lowstate", LowState_), ("rt/odommodestate", SportModeState_),
             ("rt/sportmodestate", SportModeState_)]
    topics = [Topic(participant, name, kind) for name, kind in specs]
    readers = [DataReader(participant, topic) for topic in topics]
    counts = {name: 0 for name, _ in specs}
    latest = {}
    _progress("[3/3] 开始接收状态；每秒显示采样计数。")
    end = time.monotonic() + args.duration
    next_progress = time.monotonic() + 1
    while time.monotonic() < end:
        for (name, kind), reader in zip(specs, readers):
            for sample in reader.take(32):
                if not isinstance(sample, kind):
                    continue
                counts[name] += 1
                record = {"rpy_rad": list(sample.imu_state.rpy)}
                if name == "rt/lowstate":
                    record.update(tick=sample.tick, mode_machine=sample.mode_machine)
                else:
                    record.update(position_m=list(sample.position),
                                  velocity_mps=list(sample.velocity),
                                  yaw_rate_rps=sample.yaw_speed,
                                  mode=sample.mode, error_code=sample.error_code)
                latest[name] = record
        if time.monotonic() >= next_progress:
            _progress("已收到：lowstate=%d，odommodestate=%d，sportmodestate=%d" %
                      tuple(counts[name] for name, _ in specs))
            next_progress = time.monotonic() + 1
        time.sleep(.02)
    received = counts["rt/lowstate"] > 0 and any(counts[n] > 0 for n in counts if n != "rt/lowstate")
    print(json.dumps({"interface": args.interface, "samples": counts, "latest": latest,
                      "state_received": received, "motion_commands_sent": 0,
                      "motion_ready": False,
                      "limitations": ["DDS types do not authenticate an individual robot.",
                                      "mode_machine/mode are not a verified locomotion FSM ID.",
                                      "No battery, obstacle clearance or control authority verification."]},
                     indent=2, allow_nan=False), flush=True)
    _progress("采样完成，正在退出只读诊断。")
    return 0 if received else 1


if __name__ == "__main__":
    raise SystemExit(main())
