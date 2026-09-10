"""Operator-authorized, fixed low-speed G1 forward motion smoke test.

This is a hardware diagnostic, not an Agent/navigation execution entry point.
It never changes the FSM or DDS configuration. A separately running
g1_state_collector.py must keep the state file fresh throughout the test.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from g1_dds_diagnostic import _progress, _run_supervised
from g1_state_collector import state_freshness
from unitree_adapter import G1LocoDriver


CONFIRMATION_TOKEN = "CONFIRM_FORWARD_0.05_MPS_0.5_S"
SPEED_MPS = .05
DURATION_S = .5
COMMAND_LEASE_S = .2
MIN_BATTERY_PERCENT = 50
ALLOWED_FSM_ID = 500
MAX_TILT_RAD = .2
MAX_STATIONARY_SPEED_MPS = .02
MAX_STATIONARY_YAW_RATE_RPS = .02
MIN_MEASURED_MOVEMENT_M = .03


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid %s" % name)
    return float(value)


def validate_report(report, interface, *, require_stationary):
    if not isinstance(report, dict) or report.get("collector") != "g1_state_collector":
        raise ValueError("state file is not from g1_state_collector")
    if report.get("device_id") != "G1" or report.get("interface") != interface:
        raise ValueError("state identity/interface mismatch")
    if report.get("errors"):
        raise ValueError("state collector reports errors: %r" % report["errors"])
    freshness = state_freshness(report.get("state", {}))
    stale = [key for key, status in freshness.items() if not status["fresh"]]
    if stale:
        raise ValueError("state is not fresh: %s" % stale)
    state = report["state"]
    low, odom, battery, fsm = (state[key] for key in
                               ("lowstate", "odometry", "battery", "fsm"))
    if type(fsm.get("fsm_id")) is not int or fsm["fsm_id"] != ALLOWED_FSM_ID:
        raise ValueError("FSM is not the already-prepared state 500")
    if type(battery.get("soc")) is not int or battery["soc"] < MIN_BATTERY_PERCENT:
        raise ValueError("battery is below the smoke-test minimum")
    rpy = low.get("rpy_rad")
    if not isinstance(rpy, list) or len(rpy) < 2:
        raise ValueError("attitude is missing")
    roll, pitch = _number(rpy[0], "roll"), _number(rpy[1], "pitch")
    yaw = _number(rpy[2], "yaw") if len(rpy) >= 3 else None
    if abs(roll) > MAX_TILT_RAD or abs(pitch) > MAX_TILT_RAD:
        raise ValueError("tilt exceeds smoke-test limit")
    if type(odom.get("error_code")) is not int or odom["error_code"] != 0:
        raise ValueError("odometry error")
    velocity = odom.get("velocity_mps")
    if not isinstance(velocity, list) or len(velocity) != 3:
        raise ValueError("velocity is missing")
    velocity = [_number(value, "velocity") for value in velocity]
    yaw_rate = _number(odom.get("yaw_rate_rps"), "yaw rate")
    if require_stationary and (math.sqrt(sum(value * value for value in velocity)) >
                               MAX_STATIONARY_SPEED_MPS or
                               abs(yaw_rate) > MAX_STATIONARY_YAW_RATE_RPS):
        raise ValueError("robot is not stationary")
    position = odom.get("position_m")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("position is missing")
    position = [_number(value, "position") for value in position]
    source_ns = low.get("source_timestamp_ns")
    if type(source_ns) is not int or source_ns <= 0:
        raise ValueError("lowstate source sequence is missing")
    odometry_source_ns = odom.get("source_timestamp_ns")
    if odometry_source_ns is None:
        # Older test fixtures and non-DDS snapshots may only expose lowstate's
        # sequence. Distance-controlled tests require a real odometry sequence.
        odometry_source_ns = source_ns
    if type(odometry_source_ns) is not int or odometry_source_ns <= 0:
        raise ValueError("odometry source sequence is missing")
    return {"battery_percent": battery["soc"], "fsm_id": fsm["fsm_id"],
            "roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw,
            "position_m": position,
            "velocity_mps": velocity, "yaw_rate_rps": yaw_rate,
            "lowstate_source_timestamp_ns": source_ns,
            "odometry_source_timestamp_ns": odometry_source_ns,
            "freshness": freshness}


def read_report(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def wait_for_stationary_snapshot(provider, interface, *, after_source_ns=0,
                                 timeout_s=3., sleep=time.sleep):
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    last_source_ns = after_source_ns
    last = None
    error = None
    while time.monotonic() < deadline:
        try:
            candidate = validate_report(provider(), interface, require_stationary=True)
            source_ns = candidate["lowstate_source_timestamp_ns"]
            if source_ns > last_source_ns:
                consecutive += 1
                last_source_ns = source_ns
                last = candidate
                if consecutive >= 2:
                    return last
        except Exception as exc:
            consecutive = 0
            error = exc
        sleep(.05)
    raise ValueError("two new stationary snapshots unavailable: %s" % error)


def stop_repeatedly(driver, *, sleep=time.sleep):
    errors = []
    successes = 0
    for _index in range(3):
        try:
            driver.stop()
            successes += 1
        except Exception as exc:
            errors.append(str(exc))
        sleep(.05)
    if not successes:
        raise RuntimeError("all zero-velocity stop submissions failed: %s" % errors)
    return {"attempts": 3, "successful_submissions": successes, "errors": errors}


def execute_motion(driver, provider, interface, *, sleep=time.sleep,
                   monotonic=time.monotonic):
    before = wait_for_stationary_snapshot(provider, interface, sleep=sleep)
    submissions = 0
    stop_result = None
    started = monotonic()
    try:
        deadline = started + DURATION_S
        while monotonic() < deadline:
            validate_report(provider(), interface, require_stationary=False)
            if monotonic() >= deadline:
                break
            driver.move(SPEED_MPS, 0., 0.)
            submissions += 1
            sleep(min(driver.period_s, max(0., deadline - monotonic())))
    finally:
        stop_result = stop_repeatedly(driver, sleep=sleep)
    after = wait_for_stationary_snapshot(
        provider, interface, after_source_ns=before["lowstate_source_timestamp_ns"],
        timeout_s=3., sleep=sleep)
    displacement = math.hypot(after["position_m"][0] - before["position_m"][0],
                              after["position_m"][1] - before["position_m"][1])
    return {"status": "completed", "motion": "forward", "speed_mps": SPEED_MPS,
            "command_duration_s": DURATION_S,
            "theoretical_distance_m": SPEED_MPS * DURATION_S,
            "measured_xy_displacement_m": displacement,
            # A few millimetres can be stance sway or odometry noise on a
            # humanoid and must not be reported as demonstrated walking.
            "movement_detected": displacement >= MIN_MEASURED_MOVEMENT_M,
            "nonzero_submissions": submissions, "stop": stop_result,
            "elapsed_s": monotonic() - started, "before": before, "after": after,
            "fsm_changed": before["fsm_id"] != after["fsm_id"],
            "finished_at": datetime.now(timezone.utc).isoformat()}


def worker(args):
    provider = lambda: read_report(args.state_file)
    # Validate twice before constructing a client that registers command APIs.
    first = wait_for_stationary_snapshot(provider, args.interface)
    _progress("运动前检查通过：电量=%d%%，FSM=%d，roll=%.4f，pitch=%.4f" %
              (first["battery_percent"], first["fsm_id"],
               first["roll_rad"], first["pitch_rad"]))
    driver = G1LocoDriver.connect(args.interface, timeout_s=1.,
                                  command_lease_s=COMMAND_LEASE_S)
    result = execute_motion(driver, provider, args.interface)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0 if result["movement_detected"] and not result["fsm_changed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--confirm-token", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.confirm_token != CONFIRMATION_TOKEN:
        parser.error("exact operator confirmation token is required")
    args.state_file = args.state_file.expanduser().absolute()
    if not args.worker:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                   "--interface", args.interface, "--state-file", str(args.state_file),
                   "--confirm-token", args.confirm_token]
        return _run_supervised(command, 15.)
    try:
        return worker(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc),
                          "motion": "forward", "speed_mps": SPEED_MPS,
                          "command_duration_s": DURATION_S}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
