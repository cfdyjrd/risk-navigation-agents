"""Convert independently sampled G1 state + trusted perception to observations.

No SDK imports or commands. Incomplete raw collector reports cannot be used as
RobotObservation. Source timestamps are preserved, never refreshed on read.
"""
from datetime import datetime
import math

from g1_state_collector import record_freshness
from robot_interface import RobotInterfaceError, RobotObservation
from unitree_adapter import G1Config, validate_unitree_observation


def _fresh(record, max_age):
    status = record_freshness(record, max_age_s=max_age)
    if not status["fresh"]:
        raise RobotInterfaceError("missing/stale timestamp or unverified DDS clock liveness")
    return datetime.fromisoformat(status["effective_timestamp"])


def _vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length or any(
            type(x) not in (float, int) or not math.isfinite(x) for x in value):
        raise RobotInterfaceError("invalid measured motion vector")
    return list(value)


def build_observation(report, perception, config, *, width_m):
    """Perception must come from validated robot-frame obstacle processing.

    A collector status report or raw lidar point count is NOT perception.
    Binding/authenticating each stream to the actual G1 is the caller's duty.
    """
    if not isinstance(config, G1Config):
        raise RobotInterfaceError("G1Config is required")
    if report.get("device_id") != config.device_id:
        raise RobotInterfaceError("state cache belongs to another device")
    if report.get("errors"):
        raise RobotInterfaceError("state collector reports acquisition errors")
    try:
        records = [report["state"][key] for key in ("lowstate", "odometry", "battery", "fsm")]
        stamps = [_fresh(record, config.max_observation_age_s) for record in records]
        stamps.append(_fresh(perception, config.max_observation_age_s))
        low, odom, battery, fsm = records
        low_effective, odom_effective, _battery_effective, fsm_effective = stamps[:4]
        if type(odom.get("error_code")) is not int or odom["error_code"] != 0:
            raise RobotInterfaceError("odometry error_code is missing or nonzero")
        rpy = _vector(low["rpy_rad"], 3)
        velocity = _vector(odom["velocity_mps"], 3)
        position = _vector(odom["position_m"], 3)
        yaw_rate = _vector([odom["yaw_rate_rps"]], 1)[0]
        if perception.get("frame_id") != "base_link" or perception.get("validated") is not True:
            raise RobotInterfaceError("perception must be validated in base_link")
        environment = {key: perception[key] for key in (
            "obstacle_detected", "obstacle_distance_m", "observation_confidence")}
        for key in (
                "corridor_width_m", "corridor_geometry_valid",
                "left_envelope_clearance_m", "right_envelope_clearance_m",
                "minimum_envelope_clearance_m", "clearance_uncertainty_m",
                "corridor_center_offset_m", "corridor_geometry_confidence",
                "corridor_wall_start_m", "corridor_wall_end_m",
                "corridor_heading_error_rad", "critical_wall_bin_x_m"):
            if key in perception:
                environment[key] = perception[key]
        if "clearance_basis" in perception:
            environment["clearance_basis"] = perception["clearance_basis"]
        observation = RobotObservation(
            {"device_id": config.device_id, "type": "g1", "width_m": width_m,
             "battery_percent": battery["soc"]}, environment,
            min(stamps).isoformat(), metadata={
                "g1_state": {"lowstate_timestamp": low_effective.isoformat(),
                             "attitude_timestamp": low_effective.isoformat(),
                             "fsm_timestamp": fsm_effective.isoformat(), "fsm_id": fsm["fsm_id"],
                             "lowstate_source_timestamp": low["timestamp"],
                             "timestamp_basis": record_freshness(
                                 low, max_age_s=config.max_observation_age_s)["basis"],
                             "roll_rad": rpy[0], "pitch_rad": rpy[1],
                             "yaw_rad": rpy[2]},
                "odometry": {"timestamp": odom_effective.isoformat(),
                             "source_timestamp": odom["timestamp"], "position_m": position,
                             "velocity_mps": velocity, "yaw_rate_rps": yaw_rate},
                "sources": {key: report["state"][key].get("source")
                            for key in ("lowstate", "odometry", "battery", "fsm")},
                "perception": {key: perception.get(key) for key in (
                    "producer", "schema_version", "calibration_id", "layout",
                    "source_frame", "frame_id", "source_stamp", "received_at",
                    "timestamp_basis", "source_liveness", "roi_point_count",
                    "wall_support", "wall_estimate", "corridor_geometry_valid")},
                "clock_evidence": {key: record_freshness(
                    report["state"][key], max_age_s=config.max_observation_age_s)
                    for key in ("lowstate", "odometry", "battery", "fsm")}})
    except (KeyError, TypeError) as exc:
        raise RobotInterfaceError("incomplete G1 observation inputs: %s" % exc) from exc
    return validate_unitree_observation(observation, config)


def measured_stationary(observation, *, max_speed_mps=.02, max_yaw_rate_rps=.02,
                        max_age_s=1.):
    """Measured feedback predicate; task loop still requires two new frames."""
    for value in (max_speed_mps, max_yaw_rate_rps, max_age_s):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("stationary limits must be positive and finite")
    odom = observation.metadata.get("odometry", {})
    _fresh(odom, max_age_s)
    velocity = _vector(odom.get("velocity_mps"), 3)
    yaw = _vector([odom.get("yaw_rate_rps")], 1)[0]
    return math.sqrt(sum(v * v for v in velocity)) <= max_speed_mps and abs(yaw) <= max_yaw_rate_rps
