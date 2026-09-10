"""Operator-authorized, odometry-bounded G1 one-metre forward test.

This hardware diagnostic never changes the FSM, posture, arms, DDS settings,
or system clocks. It requires a separately running g1_state_collector.py and
an operator confirmation that at least 1.5 m of the full-body path is clear
and that an independent emergency stop is immediately available.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from g1_dds_diagnostic import _progress, _run_supervised
from g1_motion_smoke_test import (
    COMMAND_LEASE_S,
    read_report,
    stop_repeatedly,
    validate_report,
    wait_for_stationary_snapshot,
)
from unitree_adapter import G1LocoDriver


CONFIRMATION_TOKEN = "CONFIRM_FORWARD_1M_PATH_CLEAR_ESTOP_READY"
TARGET_DISTANCE_M = 1.0
# Begin stopping slightly before the target to limit gait/lease overrun.
STOP_TRIGGER_M = .95
MIN_FINAL_FORWARD_M = .90
MAX_FINAL_FORWARD_M = 1.10
SPEED_MPS = .15
MAX_MOTION_TIME_S = 12.0
NO_PROGRESS_TIMEOUT_S = 2.0
MIN_PROGRESS_M = .03
MAX_LATERAL_M = .15
MAX_YAW_CHANGE_RAD = .25
MAX_YAW_RATE_RPS = .35
MAX_SPEED_MPS = .40
MAX_POSITION_STEP_M = .10


def _angle_delta(start, end):
    return math.atan2(math.sin(end - start), math.cos(end - start))


def displacement_from(start, current):
    yaw = start.get("yaw_rad")
    if type(yaw) not in (int, float) or not math.isfinite(yaw):
        raise ValueError("initial yaw is missing")
    dx = current["position_m"][0] - start["position_m"][0]
    dy = current["position_m"][1] - start["position_m"][1]
    return {
        "forward_m": dx * math.cos(yaw) + dy * math.sin(yaw),
        "lateral_m": -dx * math.sin(yaw) + dy * math.cos(yaw),
        "xy_m": math.hypot(dx, dy),
    }


def validate_motion_envelope(start, current, previous_position, elapsed_s):
    progress = displacement_from(start, current)
    yaw = current.get("yaw_rad")
    if type(yaw) not in (int, float) or not math.isfinite(yaw):
        raise ValueError("current yaw is missing")
    if abs(_angle_delta(start["yaw_rad"], yaw)) > MAX_YAW_CHANGE_RAD:
        raise ValueError("heading changed beyond the forward-test limit")
    if abs(progress["lateral_m"]) > MAX_LATERAL_M:
        raise ValueError("lateral displacement exceeded the forward-test limit")
    if progress["forward_m"] < -.05:
        raise ValueError("robot moved opposite the requested direction")
    speed = math.sqrt(sum(value * value for value in current["velocity_mps"]))
    if speed > MAX_SPEED_MPS or abs(current["yaw_rate_rps"]) > MAX_YAW_RATE_RPS:
        raise ValueError("measured velocity exceeded the forward-test limit")
    step = math.hypot(current["position_m"][0] - previous_position[0],
                      current["position_m"][1] - previous_position[1])
    if step > MAX_POSITION_STEP_M:
        raise ValueError("odometry position jumped discontinuously")
    if elapsed_s >= NO_PROGRESS_TIMEOUT_S and progress["forward_m"] < MIN_PROGRESS_M:
        raise ValueError("no demonstrated forward progress within two seconds")
    return progress


def execute_distance_motion(driver, provider, interface, *, sleep=time.sleep,
                            monotonic=time.monotonic):
    start = wait_for_stationary_snapshot(provider, interface, sleep=sleep)
    if start.get("yaw_rad") is None:
        raise ValueError("three-axis attitude is required for distance control")
    started = monotonic()
    submissions = 0
    last_odometry_source_ns = start["odometry_source_timestamp_ns"]
    previous_position = start["position_m"]
    peak_forward_m = 0.
    stop_result = None
    try:
        while True:
            elapsed = monotonic() - started
            if elapsed >= MAX_MOTION_TIME_S:
                raise ValueError("one-metre target was not reached before the time limit")
            current = validate_report(provider(), interface, require_stationary=False)
            source_ns = current["odometry_source_timestamp_ns"]
            # Only refresh the finite command lease after a newly sampled
            # state frame. If telemetry freezes, the lease expires naturally.
            if source_ns <= last_odometry_source_ns:
                sleep(.01)
                continue
            progress = validate_motion_envelope(
                start, current, previous_position, elapsed)
            peak_forward_m = max(peak_forward_m, progress["forward_m"])
            previous_position = current["position_m"]
            last_odometry_source_ns = source_ns
            if progress["forward_m"] >= STOP_TRIGGER_M:
                break
            driver.move(SPEED_MPS, 0., 0.)
            submissions += 1
            sleep(driver.period_s)
    finally:
        stop_result = stop_repeatedly(driver, sleep=sleep)

    after = wait_for_stationary_snapshot(
        provider, interface,
        after_source_ns=start["lowstate_source_timestamp_ns"],
        timeout_s=4., sleep=sleep)
    final = displacement_from(start, after)
    fsm_changed = start["fsm_id"] != after["fsm_id"]
    within_target = (MIN_FINAL_FORWARD_M <= final["forward_m"] <=
                     MAX_FINAL_FORWARD_M and
                     abs(final["lateral_m"]) <= MAX_LATERAL_M)
    return {
        "status": "completed" if within_target and not fsm_changed else "failed",
        "motion": "forward",
        "target_distance_m": TARGET_DISTANCE_M,
        "stop_trigger_m": STOP_TRIGGER_M,
        "commanded_speed_mps": SPEED_MPS,
        "measured_forward_m": final["forward_m"],
        "measured_lateral_m": final["lateral_m"],
        "measured_xy_m": final["xy_m"],
        "peak_forward_before_stop_m": peak_forward_m,
        "within_target_tolerance": within_target,
        "nonzero_submissions": submissions,
        "stop": stop_result,
        "fsm_changed": fsm_changed,
        "elapsed_s": monotonic() - started,
        "before": start,
        "after": after,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def worker(args):
    provider = lambda: read_report(args.state_file)
    first = wait_for_stationary_snapshot(provider, args.interface)
    _progress("一米测试前检查通过：电量=%d%%，FSM=%d，roll=%.4f，pitch=%.4f" %
              (first["battery_percent"], first["fsm_id"],
               first["roll_rad"], first["pitch_rad"]))
    driver = G1LocoDriver.connect(args.interface, timeout_s=1.,
                                  command_lease_s=COMMAND_LEASE_S)
    result = execute_distance_motion(driver, provider, args.interface)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if (result["status"] == "completed" and
                 result["stop"]["successful_submissions"] == 3) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--confirm-token", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.confirm_token != CONFIRMATION_TOKEN:
        parser.error("exact operator/path-clear/E-stop confirmation token is required")
    args.state_file = args.state_file.expanduser().absolute()
    if not args.worker:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                   "--interface", args.interface, "--state-file", str(args.state_file),
                   "--confirm-token", args.confirm_token]
        return _run_supervised(command, 25.)
    try:
        return worker(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "motion": "forward", "target_distance_m": TARGET_DISTANCE_M,
                          "commanded_speed_mps": SPEED_MPS}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
