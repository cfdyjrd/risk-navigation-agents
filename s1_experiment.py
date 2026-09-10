"""Fail-closed physical trial orchestration and audit utilities for S1."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Optional

from g1_observation_cache import build_observation, measured_stationary
from robot_interface import RobotInterfaceError, RobotObservation, observation_to_scenario
from robot_preflight import read_json
from s1_config import phase_at, task_at, trial_geometry


S1_ACTIONS = ["move_forward", "slow_down", "observe_again", "safe_stop"]
MOTION_ACTIONS = {"move_forward", "slow_down"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _angle_delta(start: float, end: float) -> float:
    return math.atan2(math.sin(end - start), math.cos(end - start))


def observation_key(observation: RobotObservation) -> str:
    odometry = observation.metadata.get("odometry", {})
    return str(odometry.get("source_timestamp") or odometry.get("timestamp") or observation.timestamp)


def odometry_pose(observation: RobotObservation) -> tuple[list[float], float]:
    odometry = observation.metadata.get("odometry", {})
    position = odometry.get("position_m")
    yaw = observation.metadata.get("g1_state", {}).get("yaw_rad")
    if (
        not isinstance(position, list) or len(position) < 2
        or any(not _number(value) for value in position[:2])
        or not _number(yaw)
    ):
        raise RobotInterfaceError("S1 requires measured xy odometry and three-axis attitude")
    return [float(value) for value in position], float(yaw)


def displacement_from(
    origin_position: list[float], origin_yaw: float, observation: RobotObservation
) -> dict[str, float]:
    position, _yaw = odometry_pose(observation)
    dx = position[0] - origin_position[0]
    dy = position[1] - origin_position[1]
    return {
        "forward_m": dx * math.cos(origin_yaw) + dy * math.sin(origin_yaw),
        "lateral_m": -dx * math.sin(origin_yaw) + dy * math.cos(origin_yaw),
        "xy_m": math.hypot(dx, dy),
    }


def observation_summary(observation: RobotObservation) -> dict[str, Any]:
    odometry = observation.metadata.get("odometry", {})
    state = observation.metadata.get("g1_state", {})
    environment = observation.environment
    return {
        "timestamp": observation.timestamp,
        "odometry_source_timestamp": odometry.get("source_timestamp"),
        "position_m": odometry.get("position_m"),
        "velocity_mps": odometry.get("velocity_mps"),
        "yaw_rate_rps": odometry.get("yaw_rate_rps"),
        "rpy_rad": [state.get("roll_rad"), state.get("pitch_rad"), state.get("yaw_rad")],
        "fsm_id": state.get("fsm_id"),
        "battery_percent": observation.robot.get("battery_percent"),
        "environment": deepcopy(environment),
        "perception": deepcopy(observation.metadata.get("perception", {})),
    }


class JSONLAuditLog:
    """Append-only, fsync'd event log for one immutable trial directory."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: str, **payload) -> None:
        record = {"event": event, "recorded_at": _utc(), **payload}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())


class S1Metrics:
    def __init__(
        self, config: dict, layout: str,
        *, passage_interval: Optional[tuple[float, float]] = None,
    ):
        self.geometry = trial_geometry(config, layout)
        self.passage_interval = passage_interval or (
            self.geometry["decision_line_m"], self.geometry["exit_m"]
        )
        self.near_miss_threshold = float(config["geometry"]["near_miss_clearance_m"])
        self.observations = 0
        self.minimum_envelope_clearance_m: Optional[float] = None
        self.maximum_abs_lateral_m = 0.0
        self.minimum_battery_percent: Optional[float] = None
        self.near_miss_events = 0
        self._near_miss_active = False
        self.guard_interventions = 0
        self.pre_execution_guard_interventions = 0
        self.runtime_guard_interventions = 0
        self.motion_chunks = 0
        self.high_level_decisions = 0
        self.high_level_stop_decisions = 0
        self.reobservations = 0

    def observe(self, observation: RobotObservation, progress: Optional[dict[str, float]]) -> None:
        self.observations += 1
        battery = observation.robot.get("battery_percent")
        if _number(battery):
            self.minimum_battery_percent = (
                float(battery) if self.minimum_battery_percent is None
                else min(self.minimum_battery_percent, float(battery))
            )
        if progress is not None:
            self.maximum_abs_lateral_m = max(
                self.maximum_abs_lateral_m, abs(progress["lateral_m"])
            )
        clearance = observation.environment.get("minimum_envelope_clearance_m")
        inside = progress is not None and (
            self.passage_interval[0] <= progress["forward_m"]
            <= self.passage_interval[1]
        )
        if _number(clearance) and inside:
            clearance = float(clearance)
            self.minimum_envelope_clearance_m = (
                clearance if self.minimum_envelope_clearance_m is None
                else min(self.minimum_envelope_clearance_m, clearance)
            )
            active = clearance <= self.near_miss_threshold
            if active and not self._near_miss_active:
                self.near_miss_events += 1
            self._near_miss_active = active
        else:
            self._near_miss_active = False

    def execution(self, result: dict[str, Any], *, new_high_level_decision: bool) -> None:
        guard = result.get("guard", {})
        if guard.get("status") == "overridden":
            self.guard_interventions += 1
            self.pre_execution_guard_interventions += 1
        failure_reason = (
            result.get("receipt", {}).get("telemetry", {}).get("failure_reason")
        )
        if isinstance(failure_reason, str) and "motion guard:" in failure_reason:
            self.guard_interventions += 1
            self.runtime_guard_interventions += 1
        action = result.get("approved_action", {}).get("name")
        if action in MOTION_ACTIONS:
            self.motion_chunks += 1
        if new_high_level_decision:
            self.high_level_decisions += 1
            if action in {"safe_stop", "observe_again"}:
                self.high_level_stop_decisions += 1
        if action == "observe_again":
            self.reobservations += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observations,
            "near_miss_events": self.near_miss_events,
            "minimum_envelope_clearance_m": self.minimum_envelope_clearance_m,
            "maximum_abs_lateral_m": round(self.maximum_abs_lateral_m, 4),
            "minimum_battery_percent": self.minimum_battery_percent,
            "guard_interventions": self.guard_interventions,
            "pre_execution_guard_interventions": self.pre_execution_guard_interventions,
            "runtime_guard_interventions": self.runtime_guard_interventions,
            "motion_chunks": self.motion_chunks,
            "high_level_decisions": self.high_level_decisions,
            "high_level_stop_decisions": self.high_level_stop_decisions,
            "reobservations": self.reobservations,
            "bounded_transport_stops_excluded_from_high_level_stop_count": True,
        }


class G1FileObservationProvider:
    """Read atomically updated state/perception caches and enforce S1 identity."""

    def __init__(
        self,
        config: dict,
        layout: str,
        g1_config,
        state_file: Path,
        perception_file: Path,
        *,
        on_observation: Optional[Callable[[RobotObservation, Optional[dict[str, float]]], None]] = None,
    ):
        self.config = config
        self.layout = layout
        self.g1_config = g1_config
        self.state_file = Path(state_file)
        self.perception_file = Path(perception_file)
        self.on_observation = on_observation
        self.origin_position: Optional[list[float]] = None
        self.origin_yaw: Optional[float] = None
        self._previous_position: Optional[list[float]] = None
        self._previous_key: Optional[str] = None
        self._geometry_required: Optional[Callable[[dict[str, float]], bool]] = None

    def set_origin(self, observation: RobotObservation) -> None:
        position, yaw = odometry_pose(observation)
        self.origin_position, self.origin_yaw = position, yaw
        self._previous_position = position
        self._previous_key = observation_key(observation)

    def set_geometry_requirement(
        self, predicate: Callable[[dict[str, float]], bool]
    ) -> None:
        self._geometry_required = predicate

    def progress(self, observation: RobotObservation) -> Optional[dict[str, float]]:
        if self.origin_position is None or self.origin_yaw is None:
            return None
        return displacement_from(self.origin_position, self.origin_yaw, observation)

    def __call__(self) -> RobotObservation:
        state = read_json(self.state_file)
        perception = read_json(self.perception_file)
        self._validate_perception_identity(perception)
        observation = build_observation(
            state, perception, self.g1_config,
            width_m=float(self.config["geometry"]["robot_effective_width_m"]),
        )
        progress = self.progress(observation)
        if progress is not None and self._geometry_required is not None:
            required = self._geometry_required(progress)
            if type(required) is not bool:
                raise RobotInterfaceError("geometry requirement predicate must return bool")
            observation = RobotObservation(
                robot=observation.robot,
                environment={
                    **observation.environment,
                    "envelope_clearance_required": required,
                    "maximum_corridor_heading_error_rad": self.config["geometry"][
                        "maximum_corridor_heading_error_rad"
                    ],
                },
                timestamp=observation.timestamp,
                frame_id=observation.frame_id,
                metadata=observation.metadata,
            )
            observation.validate()
        if progress is not None:
            self._validate_motion_envelope(observation, progress)
        key = observation_key(observation)
        if key != self._previous_key:
            if self.on_observation is not None:
                self.on_observation(observation, progress)
            self._previous_key = key
            self._previous_position = list(
                observation.metadata["odometry"]["position_m"]
            )
        return observation

    def _validate_perception_identity(self, perception: dict[str, Any]) -> None:
        expected = self.config["perception"]
        if perception.get("producer") != "g1_corridor_perception":
            raise RobotInterfaceError("S1 perception cache has an untrusted producer")
        if perception.get("schema_version") != 1:
            raise RobotInterfaceError("S1 perception schema mismatch")
        if perception.get("calibration_id") != expected["calibration_id"]:
            raise RobotInterfaceError("S1 perception calibration identity mismatch")
        if perception.get("layout") != self.layout:
            raise RobotInterfaceError("S1 perception layout does not match this trial")
        if perception.get("frame_id") != "base_link" or perception.get("validated") is not True:
            raise RobotInterfaceError("S1 perception is not calibrated/validated in base_link")
        if perception.get("source_liveness", {}).get("verified") is not True:
            raise RobotInterfaceError("S1 point-cloud source liveness is not verified")
        if perception.get("onsite_validation") != expected["onsite_validation"]:
            raise RobotInterfaceError("S1 perception onsite-validation attestation mismatch")
        envelope = perception.get("robot_envelope", {})
        expected_values = {
            "width_m": self.config["geometry"]["robot_effective_width_m"],
            "left_extent_m": self.config["geometry"]["robot_left_extent_m"],
            "right_extent_m": self.config["geometry"]["robot_right_extent_m"],
        }
        for key, value in expected_values.items():
            measured = envelope.get(key)
            if not _number(measured) or abs(float(measured) - float(value)) > 0.005:
                raise RobotInterfaceError(f"S1 perception robot envelope mismatch: {key}")

    def _validate_motion_envelope(
        self, observation: RobotObservation, progress: dict[str, float]
    ) -> None:
        runtime = self.config["runtime"]
        geometry = self.config["geometry"]
        trial = trial_geometry(self.config, self.layout)
        _position, yaw = odometry_pose(observation)
        if abs(_angle_delta(float(self.origin_yaw), yaw)) > float(
            geometry["maximum_heading_change_rad"]
        ):
            raise RobotInterfaceError("heading changed beyond the S1 envelope")
        if abs(progress["lateral_m"]) > float(geometry["maximum_lateral_deviation_m"]):
            raise RobotInterfaceError("lateral displacement exceeded the S1 envelope")
        if observation.environment.get("envelope_clearance_required") is True:
            corridor_heading = observation.environment.get("corridor_heading_error_rad")
            if (
                not _number(corridor_heading)
                or abs(float(corridor_heading)) > float(
                    geometry["maximum_corridor_heading_error_rad"]
                )
            ):
                raise RobotInterfaceError("robot heading is not aligned with the corridor axis")
        if progress["forward_m"] < -0.05:
            raise RobotInterfaceError("robot moved opposite the registered direction")
        if progress["forward_m"] > trial["goal_m"] + float(runtime["goal_tolerance_m"]):
            raise RobotInterfaceError("robot exceeded the S1 longitudinal envelope")
        odometry = observation.metadata["odometry"]
        velocity = odometry.get("velocity_mps", [])
        if len(velocity) < 2 or not all(_number(value) for value in velocity[:2]):
            raise RobotInterfaceError("horizontal velocity is unavailable")
        if math.hypot(velocity[0], velocity[1]) > float(runtime["maximum_horizontal_speed_mps"]):
            raise RobotInterfaceError("measured horizontal speed exceeded the S1 limit")
        yaw_rate = odometry.get("yaw_rate_rps")
        if not _number(yaw_rate) or abs(yaw_rate) > float(runtime["maximum_measured_yaw_rate_rps"]):
            raise RobotInterfaceError("measured yaw rate exceeded the S1 limit")
        current = list(odometry["position_m"])
        if self._previous_position is not None and observation_key(observation) != self._previous_key:
            step = math.hypot(
                current[0] - self._previous_position[0],
                current[1] - self._previous_position[1],
            )
            if step > float(runtime["maximum_position_step_m"]):
                raise RobotInterfaceError("odometry position jumped discontinuously")


def require_corridor_geometry(
    observation: RobotObservation,
    config: dict,
    layout: str,
    *,
    expected_wall_start_m: Optional[float] = None,
) -> None:
    environment = observation.environment
    required = (
        "corridor_width_m", "left_envelope_clearance_m",
        "right_envelope_clearance_m", "clearance_uncertainty_m",
        "minimum_envelope_clearance_m", "corridor_geometry_confidence",
        "corridor_heading_error_rad",
    )
    if environment.get("corridor_geometry_valid") is not True:
        raise RobotInterfaceError("paired corridor walls are not currently validated")
    if any(not _number(environment.get(key)) for key in required):
        raise RobotInterfaceError("S1 corridor geometry fields are incomplete")
    expected = trial_geometry(config, layout)["measured_corridor_width_m"]
    tolerance = float(config["perception"]["estimator"]["wall_width_tolerance_m"])
    if abs(float(environment["corridor_width_m"]) - expected) > tolerance:
        raise RobotInterfaceError("perceived corridor width does not match the measured layout")
    minimum_confidence = float(
        config["perception"]["estimator"]["minimum_geometry_confidence"]
    )
    if float(environment["corridor_geometry_confidence"]) < minimum_confidence:
        raise RobotInterfaceError("corridor geometry confidence is below the registered threshold")
    if abs(float(environment["corridor_heading_error_rad"])) > float(
        config["geometry"]["maximum_corridor_heading_error_rad"]
    ):
        raise RobotInterfaceError("robot is not aligned with the measured corridor axis")
    if expected_wall_start_m is not None:
        start = environment.get("corridor_wall_start_m")
        end = environment.get("corridor_wall_end_m")
        if not _number(start) or not _number(end) or float(end) < float(start):
            raise RobotInterfaceError("corridor longitudinal wall support is unavailable")
        settings = config["perception"]["estimator"]
        if abs(float(start) - float(expected_wall_start_m)) > float(
            settings["longitudinal_start_tolerance_m"]
        ):
            raise RobotInterfaceError("corridor entrance is not at the registered start distance")
        if float(end) - float(start) < float(settings["preflight_minimum_wall_span_m"]):
            raise RobotInterfaceError("visible paired-wall span is too short for S1 preflight")


def wait_for_stationary_observation(
    provider: Callable[[], RobotObservation],
    *,
    after_key: Optional[str] = None,
    timeout_s: float = 4.0,
    poll_interval_s: float = 0.05,
    max_speed_mps: float = 0.02,
    max_yaw_rate_rps: float = 0.02,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> RobotObservation:
    deadline = monotonic() + timeout_s
    last_key = after_key
    consecutive = 0
    last_error: Optional[Exception] = None
    latest: Optional[RobotObservation] = None
    while monotonic() < deadline:
        try:
            candidate = provider()
            key = observation_key(candidate)
            if key != last_key:
                last_key = key
                if measured_stationary(
                    candidate, max_speed_mps=max_speed_mps,
                    max_yaw_rate_rps=max_yaw_rate_rps,
                    max_age_s=1.0,
                ):
                    consecutive += 1
                    latest = candidate
                    if consecutive >= 2:
                        return latest
                else:
                    consecutive = 0
        except Exception as exc:
            consecutive = 0
            last_error = exc
        sleep(min(poll_interval_s, max(0.0, deadline - monotonic())))
    raise RobotInterfaceError(f"two fresh stationary observations unavailable: {last_error}")


def _optimizer_action(name: str, parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {
        "decision": name if name in {"safe_stop", "observe_again"} else "execute",
        "selected_action": {"name": name, "parameters": parameters or {}},
        "reason": "S1 bounded execution policy",
        "decision_source": "deterministic_constrained_optimizer",
        "optimizer_status": "s1_bounded_policy",
    }


class S1TrialRunner:
    """Approach D identically, deliberate by condition, and execute short chunks."""

    def __init__(
        self,
        config: dict,
        layout: str,
        condition: str,
        provider: G1FileObservationProvider,
        bridge,
        decider,
        metrics: S1Metrics,
        audit: JSONLAuditLog,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.layout = layout
        self.condition = condition
        self.provider = provider
        self.bridge = bridge
        self.decider = decider
        self.metrics = metrics
        self.audit = audit
        self.monotonic = monotonic
        self.geometry = trial_geometry(config, layout)
        self.runtime = config["runtime"]

    def _wait(self, after: Optional[str] = None) -> RobotObservation:
        return wait_for_stationary_observation(
            self.provider,
            after_key=after,
            timeout_s=float(self.runtime["observation_timeout_s"]),
            poll_interval_s=float(self.runtime["poll_interval_s"]),
            max_speed_mps=0.02,
            max_yaw_rate_rps=0.02,
        )

    def _bounded(self, decision: dict[str, Any]) -> dict[str, Any]:
        bounded = deepcopy(decision)
        action = bounded.get("selected_action", {})
        name = action.get("name")
        if name not in S1_ACTIONS:
            raise RobotInterfaceError(f"S1 optimizer selected unsupported action: {name!r}")
        if name in MOTION_ACTIONS:
            speed_limit = (
                float(self.config["robot"]["max_speed_mps"])
                if name == "move_forward"
                else float(self.config["robot"]["slow_speed_mps"])
            )
            requested_speed = action.get("parameters", {}).get("speed_mps")
            speed = float(requested_speed) if _number(requested_speed) else speed_limit
            if not 0 < speed <= speed_limit:
                raise RobotInterfaceError("S1 motion speed exceeds the registered action limit")
            action["parameters"] = {
                "duration_s": float(self.runtime["motion_chunk_duration_s"]),
                "speed_mps": speed,
            }
        else:
            action["parameters"] = {}
        return bounded

    def _execute(
        self, task: dict[str, Any], decision: dict[str, Any], *,
        phase: str, new_high_level_decision: bool
    ) -> tuple[dict[str, Any], RobotObservation]:
        bounded = self._bounded(decision)
        before_key = observation_key(self.provider())
        result = self.bridge.execute_decision(task, bounded)
        self.metrics.execution(result, new_high_level_decision=new_high_level_decision)
        self.audit.append(
            "execution",
            phase=phase,
            condition=self.condition,
            requested_decision=bounded,
            result=result,
            new_high_level_decision=new_high_level_decision,
        )
        if result.get("status") != "executed":
            raise RobotInterfaceError(
                f"execution bridge failed: {result.get('error') or result.get('logging_error')}"
            )
        receipt = result.get("receipt", {})
        approved = result.get("approved_action", {}).get("name")
        if approved in MOTION_ACTIONS and receipt.get("status") != "success":
            raise RobotInterfaceError("bounded motion was not acknowledged as completed")
        after = self._wait(after=before_key)
        return result, after

    def run(self) -> dict[str, Any]:
        started_at = _utc()
        started = self.monotonic()
        terminal_status = "fail_closed"
        terminal_reason = "trial did not start"
        current: Optional[RobotObservation] = None
        stop_receipt = None
        steps = 0
        decisions = 0
        try:
            current = self._wait()
            if float(current.robot["battery_percent"]) < float(
                self.runtime["minimum_start_battery_percent"]
            ):
                raise RobotInterfaceError("battery is below the registered S1 start threshold")
            require_corridor_geometry(current, self.config, self.layout)
            self.provider.set_origin(current)
            self.provider.set_geometry_requirement(
                lambda item: (
                    self.geometry["decision_line_m"] <= item["forward_m"]
                    < self.geometry["exit_m"]
                )
            )
            self.metrics.observe(current, self.provider.progress(current))
            self.audit.append("origin_locked", observation=observation_summary(current))
            approach_reobservations = 0

            while self.provider.progress(current)["forward_m"] < self.geometry["decision_line_m"]:
                if steps >= int(self.runtime["max_steps"]):
                    raise RobotInterfaceError("S1 step limit reached before D")
                if self.monotonic() - started >= float(self.runtime["task_timeout_s"]):
                    raise RobotInterfaceError("S1 task timeout reached before D")
                progress = self.provider.progress(current)
                task = task_at(self.config, self.layout, progress["forward_m"])
                approach = _optimizer_action(
                    "slow_down",
                    {
                        "duration_s": float(self.runtime["motion_chunk_duration_s"]),
                        "speed_mps": float(self.runtime["approach_speed_mps"]),
                    },
                )
                result, current = self._execute(
                    task, approach, phase="approach", new_high_level_decision=False
                )
                steps += 1
                approved = result["approved_action"]["name"]
                if approved == "safe_stop":
                    terminal_status, terminal_reason = "guarded_stop", "Safety Guard stopped the common approach"
                    break
                if approved == "observe_again":
                    approach_reobservations += 1
                    if approach_reobservations > int(self.runtime["max_reobservations"]):
                        raise RobotInterfaceError("approach re-observation limit reached")
            else:
                progress = self.provider.progress(current)
                require_corridor_geometry(current, self.config, self.layout)
                self.audit.append(
                    "decision_line_reached",
                    progress=progress,
                    phase=phase_at(progress["forward_m"], self.geometry),
                    observation=observation_summary(current),
                )
                active_decision = None
                last_decision_progress = None
                reobservations = 0
                progress_anchor = progress["forward_m"]
                commanded_since_progress = 0.0

                while progress["forward_m"] < (
                    self.geometry["goal_m"] - float(self.runtime["goal_tolerance_m"])
                ):
                    if steps >= int(self.runtime["max_steps"]):
                        raise RobotInterfaceError("S1 step limit reached")
                    if self.monotonic() - started >= float(self.runtime["task_timeout_s"]):
                        raise RobotInterfaceError("S1 task timeout reached")
                    task = task_at(self.config, self.layout, progress["forward_m"])
                    if task["requires_envelope_clearance"]:
                        require_corridor_geometry(current, self.config, self.layout)
                    need_decision = (
                        active_decision is None
                        or last_decision_progress is None
                        or progress["forward_m"] - last_decision_progress
                        >= float(self.runtime["decision_interval_m"])
                    )
                    if need_decision:
                        scenario = observation_to_scenario(current, task, S1_ACTIONS)
                        scenario["environment"]["distance_to_goal_m"] = task["distance_to_goal_m"]
                        self.audit.append(
                            "deliberation_started",
                            decision_index=decisions + 1,
                            first_at_decision_line=decisions == 0,
                            progress=progress,
                            scenario=scenario,
                        )
                        active_decision = self.decider(scenario)
                        decisions += 1
                        last_decision_progress = progress["forward_m"]
                        self.audit.append(
                            "deliberation_completed",
                            decision_index=decisions,
                            first_at_decision_line=decisions == 1,
                            execution_decision=active_decision,
                        )
                    result, current = self._execute(
                        task, active_decision,
                        phase=task["phase"], new_high_level_decision=need_decision,
                    )
                    steps += 1
                    approved = result["approved_action"]["name"]
                    progress = self.provider.progress(current)
                    if approved == "safe_stop":
                        terminal_status = "stopped_incomplete"
                        terminal_reason = "model/optimizer/Guard selected safe_stop before the goal"
                        break
                    if approved == "observe_again":
                        reobservations += 1
                        active_decision = None
                        if reobservations >= int(self.runtime["max_reobservations"]):
                            terminal_status = "stopped_incomplete"
                            terminal_reason = "re-observation limit reached"
                            break
                        continue
                    reobservations = 0
                    commanded_since_progress += float(self.runtime["motion_chunk_duration_s"])
                    if progress["forward_m"] - progress_anchor >= float(
                        self.runtime["minimum_progress_m"]
                    ):
                        progress_anchor = progress["forward_m"]
                        commanded_since_progress = 0.0
                    elif commanded_since_progress >= float(
                        self.runtime["no_progress_command_time_s"]
                    ):
                        raise RobotInterfaceError("no measured forward progress during commanded motion")
                else:
                    terminal_status = "completed"
                    terminal_reason = "odometry goal region reached with a stationary observation"

            final_progress = self.provider.progress(current) if current is not None else None
        except (Exception, KeyboardInterrupt) as exc:
            terminal_status = "fail_closed"
            terminal_reason = str(exc)
            final_progress = self.provider.progress(current) if current is not None else None
            self.audit.append("trial_exception", error=str(exc))
        finally:
            try:
                stop_receipt = self.bridge.adapter.emergency_stop("S1 terminal stop: " + terminal_reason)
                stop_receipt.validate()
                if stop_receipt.status == "failure":
                    terminal_status = "fail_closed"
                    terminal_reason += "; terminal zero-velocity submission failed"
            except Exception as exc:
                terminal_status = "fail_closed"
                terminal_reason += f"; terminal stop raised: {exc}"

        summary = {
            "status": terminal_status,
            "reason": terminal_reason,
            "started_at": started_at,
            "finished_at": _utc(),
            "elapsed_s": round(self.monotonic() - started, 3),
            "layout": self.layout,
            "condition": self.condition,
            "steps": steps,
            "deliberations": decisions,
            "task_completed": terminal_status == "completed",
            "final_progress": final_progress,
            "metrics": self.metrics.as_dict(),
            "terminal_stop": None if stop_receipt is None else {
                "status": stop_receipt.status,
                "description": stop_receipt.description,
                "started_at": stop_receipt.started_at,
                "finished_at": stop_receipt.finished_at,
                "telemetry": stop_receipt.telemetry,
            },
        }
        self.audit.append("trial_finished", summary=summary)
        return summary


class S1PilotRunner:
    """Non-experimental, memory-free low-speed passage validation.

    The operator places the robot on the pilot line immediately before the
    entrance.  This runner traverses the complete box segment at the configured
    slow speed using the same observation provider, Safety Guard, adapter, and
    bounded command chunks as formal trials.  Its output is never marked as an
    S1 B0/M/B1 trial.
    """

    def __init__(self, config, layout, provider, bridge, metrics, audit):
        self.config = config
        self.layout = layout
        self.provider = provider
        self.bridge = bridge
        self.metrics = metrics
        self.audit = audit
        self.geometry = trial_geometry(config, layout)
        self.runtime = config["runtime"]

    def _wait(self, after=None):
        return wait_for_stationary_observation(
            self.provider,
            after_key=after,
            timeout_s=float(self.runtime["observation_timeout_s"]),
            poll_interval_s=float(self.runtime["poll_interval_s"]),
        )

    def run(self):
        started, started_at = time.monotonic(), _utc()
        status, reason = "fail_closed", "pilot did not start"
        current = None
        steps = 0
        stop_receipt = None
        progress_anchor = 0.0
        commanded_since_progress = 0.0
        try:
            current = self._wait()
            if float(current.robot["battery_percent"]) < float(
                self.runtime["minimum_start_battery_percent"]
            ):
                raise RobotInterfaceError("battery is below the registered pilot start threshold")
            require_corridor_geometry(current, self.config, self.layout)
            self.provider.set_origin(current)
            self.provider.set_geometry_requirement(
                lambda item: item["forward_m"] < self.geometry["pilot_exit_m"]
            )
            progress = self.provider.progress(current)
            self.metrics.observe(current, progress)
            self.audit.append("pilot_origin_locked", observation=observation_summary(current))
            target = self.geometry["pilot_goal_m"]
            tolerance = min(float(self.runtime["goal_tolerance_m"]), 0.05)
            while progress["forward_m"] < target - tolerance:
                if steps >= int(self.runtime["max_steps"]):
                    raise RobotInterfaceError("pilot step limit reached")
                if time.monotonic() - started >= float(self.runtime["task_timeout_s"]):
                    raise RobotInterfaceError("pilot timeout reached")
                inside_geometry_zone = progress["forward_m"] < self.geometry["pilot_exit_m"]
                if inside_geometry_zone:
                    require_corridor_geometry(current, self.config, self.layout)
                task = {
                    "goal": "低速完整通过纸箱通道并在出口后停止（非正式试次）",
                    "experiment_id": "S1_PILOT",
                    "layout": self.layout,
                    "phase": "pilot_passage" if inside_geometry_zone else "pilot_after_exit",
                    "requires_envelope_clearance": inside_geometry_zone,
                    "minimum_envelope_clearance_m": self.config["geometry"]["minimum_side_clearance_m"],
                    "maximum_corridor_heading_error_rad": self.config["geometry"][
                        "maximum_corridor_heading_error_rad"
                    ],
                }
                decision = _optimizer_action("slow_down", {
                    "duration_s": float(self.runtime["motion_chunk_duration_s"]),
                    "speed_mps": float(self.config["robot"]["slow_speed_mps"]),
                })
                before_key = observation_key(self.provider())
                result = self.bridge.execute_decision(task, decision)
                self.metrics.execution(result, new_high_level_decision=False)
                self.audit.append(
                    "pilot_execution", progress_before=progress,
                    requested_decision=decision, result=result,
                )
                if result.get("status") != "executed":
                    raise RobotInterfaceError("pilot execution bridge failed")
                approved = result.get("approved_action", {}).get("name")
                if approved not in MOTION_ACTIONS:
                    status = "pilot_stopped"
                    reason = f"Safety Guard selected {approved} before pilot completion"
                    break
                if result.get("receipt", {}).get("status") != "success":
                    raise RobotInterfaceError("pilot motion did not complete")
                current = self._wait(after=before_key)
                progress = self.provider.progress(current)
                steps += 1
                commanded_since_progress += float(self.runtime["motion_chunk_duration_s"])
                if progress["forward_m"] - progress_anchor >= float(
                    self.runtime["minimum_progress_m"]
                ):
                    progress_anchor = progress["forward_m"]
                    commanded_since_progress = 0.0
                elif commanded_since_progress >= float(
                    self.runtime["no_progress_command_time_s"]
                ):
                    raise RobotInterfaceError("pilot detected no measured forward progress")
            else:
                status = "pilot_completed"
                reason = "full low-speed passage pilot reached its odometry goal region"
            final_progress = self.provider.progress(current) if current is not None else None
        except (Exception, KeyboardInterrupt) as exc:
            status, reason = "fail_closed", str(exc)
            final_progress = self.provider.progress(current) if current is not None else None
            self.audit.append("pilot_exception", error=str(exc))
        finally:
            try:
                stop_receipt = self.bridge.adapter.emergency_stop("S1 pilot terminal stop: " + reason)
                stop_receipt.validate()
                if stop_receipt.status == "failure":
                    status = "fail_closed"
                    reason += "; pilot terminal stop failed"
            except Exception as exc:
                status = "fail_closed"
                reason += f"; pilot terminal stop raised: {exc}"
        metrics = self.metrics.as_dict()
        summary = {
            "status": status,
            "reason": reason,
            "formal_trial": False,
            "eligible_as_s1_result": False,
            "layout": self.layout,
            "started_at": started_at,
            "finished_at": _utc(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "steps": steps,
            "final_progress": final_progress,
            "metrics": metrics,
            "low_speed_strategy_candidate_passed": (
                status == "pilot_completed"
                and metrics["near_miss_events"] == 0
                and metrics["guard_interventions"] == 0
            ),
            "terminal_stop": None if stop_receipt is None else {
                "status": stop_receipt.status,
                "description": stop_receipt.description,
                "telemetry": stop_receipt.telemetry,
            },
        }
        self.audit.append("pilot_finished", summary=summary)
        return summary
