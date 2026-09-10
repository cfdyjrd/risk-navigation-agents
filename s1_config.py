"""Validated configuration and geometry helpers for experiment S1."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from safety_guard import HARD_LIMITS
from unitree_adapter import G1Config


class S1ConfigError(ValueError):
    """Raised before any hardware driver is constructed."""


LAYOUTS = frozenset({"A", "B"})
CONDITIONS = frozenset({"B0", "M", "B1"})


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise S1ConfigError(f"{key} must be an object")
    return value


def _number(
    parent: dict[str, Any], key: str, *, minimum: float = 0.0,
    maximum: float = math.inf, positive: bool = False
) -> float:
    value = parent.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise S1ConfigError(f"{key} must be a finite number")
    value = float(value)
    if value < minimum or value > maximum or (positive and value == 0):
        interval = "(" if positive and minimum == 0 else "["
        raise S1ConfigError(f"{key} must be within {interval}{minimum}, {maximum}]")
    return value


def _integer(parent: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = parent.get(key)
    if type(value) is not int or value < minimum:
        raise S1ConfigError(f"{key} must be an integer >= {minimum}")
    return value


def _text(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise S1ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _only(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise S1ConfigError(f"unknown {name} fields: {sorted(unknown)}")


def _timestamp(parent: dict[str, Any], key: str) -> str:
    value = _text(parent, key)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise S1ConfigError(f"{key} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise S1ConfigError(f"{key} must include a timezone")
    return value


def load_s1_config(path: str | Path, *, require_onsite: bool = False) -> dict[str, Any]:
    """Load and fully validate an S1 JSON configuration.

    ``require_onsite`` is used by perception/preflight/execution.  Memory seed
    generation may run before the final sensor validation attestation, but it
    still requires real measured robot and box geometry.
    """
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise S1ConfigError(f"cannot read S1 config: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise S1ConfigError("S1 config schema_version must be 1")
    _only(data, {
        "schema_version", "experiment", "robot", "geometry", "layouts",
        "runtime", "paths", "memory", "perception",
        "pilot_approvals",
    }, "top-level")

    experiment = _object(data, "experiment")
    _only(experiment, {"id", "task_goal"}, "experiment")
    if _text(experiment, "id") != "S1":
        raise S1ConfigError("experiment.id must be S1")
    _text(experiment, "task_goal")

    robot = _object(data, "robot")
    try:
        robot_config = G1Config(**robot)
    except Exception as exc:
        raise S1ConfigError(f"invalid G1 robot configuration: {exc}") from exc
    if robot_config.max_speed_mps > 0.10 or robot_config.slow_speed_mps > 0.05:
        raise S1ConfigError("S1 commanded speeds exceed the low-speed experiment caps")
    if robot_config.max_duration_s > 0.5:
        raise S1ConfigError("S1 motion chunks may not exceed 0.5 s")
    if robot_config.max_tilt_rad > 0.30:
        raise S1ConfigError("S1 max_tilt_rad may not exceed 0.30 rad")

    geometry = _object(data, "geometry")
    _only(geometry, {
        "robot_effective_width_m", "robot_left_extent_m", "robot_right_extent_m",
        "passage_length_m", "start_to_entrance_m",
        "decision_line_before_entrance_m", "goal_after_exit_m",
        "pilot_start_before_entrance_m", "pilot_goal_after_exit_m",
        "minimum_side_clearance_m", "near_miss_clearance_m",
        "maximum_lateral_deviation_m", "maximum_corridor_heading_error_rad",
        "maximum_heading_change_rad",
    }, "geometry")
    width = _number(geometry, "robot_effective_width_m", positive=True)
    left = _number(geometry, "robot_left_extent_m", positive=True)
    right = _number(geometry, "robot_right_extent_m", positive=True)
    if abs(left + right - width) > 0.005:
        raise S1ConfigError("robot left/right extents must sum to effective width within 5 mm")
    passage = _number(geometry, "passage_length_m", positive=True)
    start = _number(geometry, "start_to_entrance_m", positive=True)
    d_before = _number(geometry, "decision_line_before_entrance_m", positive=True)
    if d_before >= start:
        raise S1ConfigError("decision line must lie between the start and entrance")
    goal_after = _number(geometry, "goal_after_exit_m", positive=True)
    _number(geometry, "pilot_start_before_entrance_m", positive=True)
    _number(geometry, "pilot_goal_after_exit_m", positive=True)
    registered_geometry = {
        "passage_length_m": (passage, 1.20),
        "start_to_entrance_m": (start, 1.50),
        "decision_line_before_entrance_m": (d_before, 0.80),
        "goal_after_exit_m": (goal_after, 1.50),
    }
    for name, (actual, expected) in registered_geometry.items():
        if abs(actual - expected) > 0.001:
            raise S1ConfigError(f"S1 {name} must remain {expected:.2f} m")
    minimum_side = _number(geometry, "minimum_side_clearance_m", positive=True)
    if minimum_side < HARD_LIMITS["minimum_envelope_clearance_m"]:
        raise S1ConfigError("S1 side-clearance limit cannot weaken the global hard limit")
    near_miss = _number(geometry, "near_miss_clearance_m", positive=True)
    if near_miss < minimum_side:
        raise S1ConfigError("near-miss threshold cannot be below the hard side-clearance limit")
    _number(geometry, "maximum_lateral_deviation_m", positive=True)
    corridor_heading = _number(
        geometry, "maximum_corridor_heading_error_rad",
        positive=True, maximum=0.08,
    )
    relative_heading = _number(
        geometry, "maximum_heading_change_rad", positive=True, maximum=0.10
    )
    if corridor_heading > relative_heading:
        raise S1ConfigError(
            "corridor heading error limit cannot exceed the relative heading-change limit"
        )

    layouts = _object(data, "layouts")
    if set(layouts) != LAYOUTS:
        raise S1ConfigError("layouts must contain exactly A and B")
    expected_additions = {"A": 0.80, "B": 0.30}
    layout_measurements: dict[str, dict[str, float]] = {}
    for name in sorted(LAYOUTS):
        item = _object(layouts, name)
        _only(item, {
            "added_clearance_m", "measured_width_m",
            "left_inner_offset_m", "right_inner_offset_m",
            "measured_box_length_m", "measured_box_height_m",
            "measurement_note", "box_shape_and_fixing_note",
            "foot_envelope_relation_note",
        }, f"layout {name}")
        added = _number(item, "added_clearance_m", positive=True)
        if abs(added - expected_additions[name]) > 0.001:
            raise S1ConfigError(f"layout {name} added clearance must be {expected_additions[name]:.2f} m")
        measured = _number(item, "measured_width_m", positive=True)
        if abs(measured - (width + added)) > 0.03:
            raise S1ConfigError(
                f"layout {name} measured width differs from W+{added:.2f} m by over 3 cm"
            )
        left_inner = _number(item, "left_inner_offset_m", positive=True)
        right_inner = _number(item, "right_inner_offset_m", positive=True)
        if abs(left_inner + right_inner - measured) > 0.02:
            raise S1ConfigError(
                f"layout {name} left/right inner offsets must sum to measured width within 2 cm"
            )
        measured_length = _number(item, "measured_box_length_m", positive=True)
        if abs(measured_length - passage) > 0.03:
            raise S1ConfigError(
                f"layout {name} measured box length differs from passage_length_m by over 3 cm"
            )
        measured_height = _number(item, "measured_box_height_m", positive=True)
        for key in (
            "measurement_note", "box_shape_and_fixing_note",
            "foot_envelope_relation_note",
        ):
            note = _text(item, key)
            if note.startswith("填写"):
                raise S1ConfigError(f"layout {name} {key} is still the template text")
        layout_measurements[name] = {
            "left_inner_offset_m": left_inner,
            "right_inner_offset_m": right_inner,
            "measured_height_m": measured_height,
        }

    runtime = _object(data, "runtime")
    _only(runtime, {
        "network_interface", "command_lease_s", "motion_chunk_duration_s",
        "decision_interval_m", "observation_timeout_s", "task_timeout_s",
        "poll_interval_s", "max_steps", "max_reobservations",
        "approach_speed_mps", "minimum_start_battery_percent",
        "no_progress_command_time_s", "minimum_progress_m",
        "maximum_horizontal_speed_mps", "maximum_measured_yaw_rate_rps",
        "maximum_position_step_m", "goal_tolerance_m", "execution_enabled",
    }, "runtime")
    _text(runtime, "network_interface")
    lease = _number(runtime, "command_lease_s", positive=True, maximum=0.5)
    if lease < 0.05:
        raise S1ConfigError("command lease must be at least 0.05 s")
    chunk = _number(runtime, "motion_chunk_duration_s", positive=True)
    if chunk > float(robot["max_duration_s"]):
        raise S1ConfigError("motion chunk exceeds robot.max_duration_s")
    _number(runtime, "decision_interval_m", positive=True, maximum=0.5)
    _number(runtime, "observation_timeout_s", positive=True)
    _number(runtime, "task_timeout_s", positive=True)
    _number(runtime, "poll_interval_s", positive=True, maximum=1)
    _integer(runtime, "max_steps")
    _integer(runtime, "max_reobservations")
    approach = _number(runtime, "approach_speed_mps", positive=True)
    if approach > float(robot["slow_speed_mps"]):
        raise S1ConfigError("approach speed exceeds robot.slow_speed_mps")
    if approach > 0.05:
        raise S1ConfigError("S1 approach speed may not exceed 0.05 m/s")
    _number(runtime, "minimum_start_battery_percent", minimum=50, maximum=100)
    _number(runtime, "no_progress_command_time_s", positive=True)
    _number(runtime, "minimum_progress_m", positive=True)
    _number(runtime, "maximum_horizontal_speed_mps", positive=True, maximum=1.0)
    _number(runtime, "maximum_measured_yaw_rate_rps", positive=True, maximum=1.0)
    _number(runtime, "maximum_position_step_m", positive=True, maximum=0.25)
    tolerance = _number(runtime, "goal_tolerance_m", positive=True)
    if tolerance >= min(passage, float(geometry["goal_after_exit_m"])):
        raise S1ConfigError("goal tolerance is too large for the registered geometry")
    if type(runtime.get("execution_enabled")) is not bool:
        raise S1ConfigError("runtime.execution_enabled must be a boolean")

    paths = _object(data, "paths")
    _only(paths, {"state_file", "perception_file", "memory_file", "rule_file", "results_dir"}, "paths")
    for key in ("state_file", "perception_file", "memory_file", "rule_file", "results_dir"):
        _text(paths, key)

    memory = _object(data, "memory")
    _only(memory, {"top_k", "token_budget"}, "memory")
    _integer(memory, "top_k")
    _integer(memory, "token_budget")

    approvals = _object(data, "pilot_approvals")
    if set(approvals) != LAYOUTS:
        raise S1ConfigError("pilot_approvals must contain exactly A and B")
    for name in sorted(LAYOUTS):
        approval = _object(approvals, name)
        _only(approval, {"approved", "pilot_id", "reviewed_by", "reviewed_at"}, f"pilot approval {name}")
        if type(approval.get("approved")) is not bool:
            raise S1ConfigError(f"pilot approval {name}.approved must be a boolean")
        if approval["approved"]:
            for key in ("pilot_id", "reviewed_by"):
                _text(approval, key)
            _timestamp(approval, "reviewed_at")

    perception = _object(data, "perception")
    _only(perception, {
        "producer", "topic", "sensor_frame", "base_frame", "calibration_id",
        "onsite_validation", "roi", "estimator",
    }, "perception")
    if _text(perception, "producer") != "g1_corridor_perception":
        raise S1ConfigError("perception.producer must be g1_corridor_perception")
    _text(perception, "topic")
    _text(perception, "sensor_frame")
    if _text(perception, "base_frame") != "base_link":
        raise S1ConfigError("perception.base_frame must be base_link")
    calibration_id = _text(perception, "calibration_id")
    if calibration_id.startswith("REPLACE_WITH"):
        raise S1ConfigError("perception.calibration_id is still a template placeholder")
    validation = _object(perception, "onsite_validation")
    _only(validation, {
        "status", "operator", "verified_at", "tf_method",
        "robot_envelope_method", "box_geometry_method",
    }, "perception.onsite_validation")
    if validation.get("status") not in {"NOT_VERIFIED", "onsite_verified"}:
        raise S1ConfigError("onsite_validation.status must be NOT_VERIFIED or onsite_verified")
    if require_onsite and validation.get("status") != "onsite_verified":
        raise S1ConfigError("onsite perception/geometry validation is not attested")
    if validation.get("status") == "onsite_verified":
        for key in ("operator", "tf_method", "robot_envelope_method", "box_geometry_method"):
            _text(validation, key)
        _timestamp(validation, "verified_at")

    roi = _object(perception, "roi")
    _only(roi, {"forward_min_m", "forward_max_m", "height_min_m", "height_max_m"}, "perception.roi")
    forward_min = _number(roi, "forward_min_m", minimum=-2.0)
    forward_max = _number(roi, "forward_max_m", positive=True)
    height_min = _number(roi, "height_min_m", minimum=0)
    height_max = _number(roi, "height_max_m", positive=True)
    if forward_max <= forward_min or height_max <= height_min:
        raise S1ConfigError("perception ROI maxima must exceed minima")

    estimator = _object(perception, "estimator")
    _only(estimator, {
        "max_points", "wall_bin_size_m", "minimum_wall_bins",
        "minimum_points_per_wall_bin", "minimum_total_roi_points",
        "wall_search_margin_m", "wall_width_tolerance_m",
        "measurement_uncertainty_floor_m", "maximum_wall_mad_m",
        "obstacle_forward_min_m", "obstacle_lateral_margin_m", "minimum_obstacle_points",
        "minimum_geometry_confidence", "longitudinal_start_tolerance_m",
        "preflight_minimum_wall_span_m",
    }, "perception.estimator")
    _integer(estimator, "max_points", minimum=100)
    _number(estimator, "wall_bin_size_m", positive=True)
    _integer(estimator, "minimum_wall_bins", minimum=2)
    _integer(estimator, "minimum_points_per_wall_bin", minimum=2)
    _integer(estimator, "minimum_total_roi_points", minimum=10)
    _number(estimator, "wall_search_margin_m", positive=True)
    _number(estimator, "wall_width_tolerance_m", positive=True)
    _number(estimator, "measurement_uncertainty_floor_m", positive=True)
    _number(estimator, "maximum_wall_mad_m", positive=True)
    obstacle_forward_min = _number(estimator, "obstacle_forward_min_m", minimum=0)
    if obstacle_forward_min >= forward_max:
        raise S1ConfigError("obstacle_forward_min_m must be inside the longitudinal ROI")
    _number(estimator, "obstacle_lateral_margin_m", minimum=0)
    _integer(estimator, "minimum_obstacle_points", minimum=1)
    _number(estimator, "minimum_geometry_confidence", minimum=0, maximum=1)
    _number(
        estimator, "longitudinal_start_tolerance_m",
        positive=True, maximum=0.35,
    )
    minimum_wall_span = _number(
        estimator, "preflight_minimum_wall_span_m",
        positive=True,
    )
    if minimum_wall_span >= passage:
        raise S1ConfigError("preflight minimum wall span must be shorter than the passage")

    uncertainty_floor = float(estimator["measurement_uncertainty_floor_m"])
    for name, measured in layout_measurements.items():
        if measured["measured_height_m"] + 0.01 < height_max:
            raise S1ConfigError(
                f"layout {name} box height does not cover perception.roi.height_max_m"
            )
        conservative_left = measured["left_inner_offset_m"] - left - uncertainty_floor
        conservative_right = measured["right_inner_offset_m"] - right - uncertainty_floor
        if min(conservative_left, conservative_right) <= minimum_side:
            raise S1ConfigError(
                f"layout {name} nominal side clearance is not above the S1 hard threshold"
            )

    normalized = deepcopy(data)
    normalized["_config_path"] = str(source)
    normalized["_config_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return normalized


def resolve_config_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()


def trial_id(layout: str, condition: str, repeat: int) -> str:
    if layout not in LAYOUTS or condition not in CONDITIONS:
        raise S1ConfigError("invalid S1 layout or condition")
    if type(repeat) is not int or not 1 <= repeat <= 99:
        raise S1ConfigError("repeat must be an integer within [1, 99]")
    return f"S1_{layout}_{condition}_R{repeat}"


def trial_geometry(config: dict[str, Any], layout: str) -> dict[str, float]:
    if layout not in LAYOUTS:
        raise S1ConfigError("layout must be A or B")
    geometry = config["geometry"]
    entrance = float(geometry["start_to_entrance_m"])
    decision = entrance - float(geometry["decision_line_before_entrance_m"])
    exit_distance = entrance + float(geometry["passage_length_m"])
    goal = exit_distance + float(geometry["goal_after_exit_m"])
    width = float(geometry["robot_effective_width_m"])
    added = float(config["layouts"][layout]["added_clearance_m"])
    return {
        "start_m": 0.0,
        "decision_line_m": decision,
        "entrance_m": entrance,
        "exit_m": exit_distance,
        "goal_m": goal,
        "robot_effective_width_m": width,
        "ideal_corridor_width_m": width + added,
        "measured_corridor_width_m": float(config["layouts"][layout]["measured_width_m"]),
        "left_inner_offset_m": float(config["layouts"][layout]["left_inner_offset_m"]),
        "right_inner_offset_m": float(config["layouts"][layout]["right_inner_offset_m"]),
        "nominal_left_envelope_clearance_m": (
            float(config["layouts"][layout]["left_inner_offset_m"])
            - float(geometry["robot_left_extent_m"])
        ),
        "nominal_right_envelope_clearance_m": (
            float(config["layouts"][layout]["right_inner_offset_m"])
            - float(geometry["robot_right_extent_m"])
        ),
        "pilot_entrance_m": float(geometry["pilot_start_before_entrance_m"]),
        "pilot_exit_m": float(geometry["pilot_start_before_entrance_m"]) + float(geometry["passage_length_m"]),
        "pilot_goal_m": float(geometry["pilot_start_before_entrance_m"]) + float(geometry["passage_length_m"]) + float(geometry["pilot_goal_after_exit_m"]),
    }


def phase_at(progress_m: float, geometry: dict[str, float]) -> str:
    if progress_m < geometry["decision_line_m"]:
        return "approach"
    if progress_m < geometry["entrance_m"]:
        return "decision_zone"
    if progress_m < geometry["exit_m"]:
        return "inside_passage"
    if progress_m < geometry["goal_m"]:
        return "after_exit"
    return "goal"


def task_at(config: dict[str, Any], layout: str, progress_m: float) -> dict[str, Any]:
    geometry = trial_geometry(config, layout)
    phase = phase_at(progress_m, geometry)
    return {
        "goal": config["experiment"]["task_goal"],
        "experiment_id": "S1",
        "layout": layout,
        "phase": phase,
        "distance_to_goal_m": max(0.0, geometry["goal_m"] - progress_m),
        "requires_envelope_clearance": phase in {"decision_zone", "inside_passage"},
        "minimum_envelope_clearance_m": config["geometry"]["minimum_side_clearance_m"],
        "maximum_corridor_heading_error_rad": config["geometry"][
            "maximum_corridor_heading_error_rad"
        ],
    }


def confirmation_token(identifier: str) -> str:
    safe = re.sub(r"[^A-Z0-9_]+", "_", identifier.upper())
    return f"CONFIRM_{safe}_FORWARD_ONLY_ESTOP_READY"


def motion_description(config: dict[str, Any], layout: str, condition: str, repeat: int) -> str:
    geometry = trial_geometry(config, layout)
    robot = config["robot"]
    runtime = config["runtime"]
    identifier = trial_id(layout, condition, repeat)
    return (
        f"试次 {identifier}：机器人只沿当前朝向直行，不发送转向、横移、姿态或 FSM 指令。\n"
        f"从起点至目标的里程计目标为 {geometry['goal_m']:.2f} m；先以 "
        f"{runtime['approach_speed_mps']:.2f} m/s 到 D 线，再由 {condition} 决策选择 "
        f"{robot['slow_speed_mps']:.2f} m/s 低速前进、最高 {robot['max_speed_mps']:.2f} m/s 前进，"
        f"或重新观测/安全停止。\n"
        f"每段非零速度最多 {runtime['motion_chunk_duration_s']:.2f} s，命令租约 "
        f"{runtime['command_lease_s']:.2f} s；每段均重新读取状态和感知并经过 Safety Guard。"
    )


def setup_fingerprint(config: dict[str, Any]) -> str:
    """Hash safety-relevant setup fields while excluding approvals/results paths."""
    payload = {
        "robot": config["robot"],
        "geometry": config["geometry"],
        "layouts": config["layouts"],
        "perception": config["perception"],
        "runtime": {
            key: value for key, value in config["runtime"].items()
            if key != "execution_enabled"
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
