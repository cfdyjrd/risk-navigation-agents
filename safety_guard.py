"""Deterministic safety checks that no language-model decision may bypass."""

from __future__ import annotations

import math
from typing import Any


HARD_LIMITS = {
    "emergency_stop_distance_m": 0.30,
    "minimum_observation_confidence": 0.50,
    "minimum_clearance_m": 0.10,
    # Conservative clearance from either side of the measured dynamic robot
    # envelope.  This is intentionally stricter than the legacy total-width
    # check when side-specific geometry is available.
    "minimum_envelope_clearance_m": 0.10,
    "minimum_battery_percent": 10,
}


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def motion_hard_violations(scenario: dict[str, Any]) -> list[str]:
    """Return deterministic violations that make any motion infeasible.

    Legacy scenarios may contain only ``corridor_width_m``.  An S1 task sets
    ``requires_envelope_clearance`` and therefore fails closed unless the
    calibrated perception stream supplies consistent left/right clearances and
    uncertainty-adjusted minimum clearance.
    """
    robot = scenario.get("robot", {})
    environment = scenario.get("environment", {})
    task = scenario.get("task", {})
    violations: list[str] = []

    obstacle_distance = environment.get("obstacle_distance_m")
    if (
        environment.get("obstacle_detected") is True
        and _number(obstacle_distance)
        and obstacle_distance <= HARD_LIMITS["emergency_stop_distance_m"]
    ):
        violations.append("emergency_obstacle_distance")

    observation_confidence = environment.get("observation_confidence")
    if (
        _number(observation_confidence)
        and observation_confidence < HARD_LIMITS["minimum_observation_confidence"]
    ):
        violations.append("observation_confidence_below_hard_limit")

    battery = robot.get("battery_percent")
    if _number(battery) and battery <= HARD_LIMITS["minimum_battery_percent"]:
        violations.append("battery_below_hard_limit")

    corridor_width = environment.get("corridor_width_m")
    robot_width = robot.get("width_m")
    if _number(corridor_width) and _number(robot_width):
        if corridor_width - robot_width < HARD_LIMITS["minimum_clearance_m"]:
            violations.append("corridor_clearance_below_hard_limit")

    geometry_keys = {
        "left_envelope_clearance_m",
        "right_envelope_clearance_m",
        "clearance_uncertainty_m",
        "minimum_envelope_clearance_m",
    }
    present = geometry_keys.intersection(environment)
    geometry_required = (
        task.get("requires_envelope_clearance") is True
        or environment.get("envelope_clearance_required") is True
    )
    geometry_valid = environment.get("corridor_geometry_valid") is True
    if geometry_required and (present != geometry_keys or not geometry_valid):
        violations.append("corridor_geometry_unavailable")
    elif present:
        if not geometry_valid or present != geometry_keys or not all(
            _number(environment.get(key)) for key in geometry_keys
        ):
            violations.append("corridor_geometry_inconsistent")
        else:
            left = float(environment["left_envelope_clearance_m"])
            right = float(environment["right_envelope_clearance_m"])
            uncertainty = float(environment["clearance_uncertainty_m"])
            reported = float(environment["minimum_envelope_clearance_m"])
            computed = min(left, right) - uncertainty
            tolerance = max(0.005, uncertainty * 0.1)
            width_consistent = True
            if _number(corridor_width) and _number(robot_width):
                reconstructed = left + right + float(robot_width)
                width_consistent = abs(reconstructed - float(corridor_width)) <= max(
                    0.01, uncertainty * 2
                )
            if uncertainty < 0 or abs(computed - reported) > tolerance or not width_consistent:
                violations.append("corridor_geometry_inconsistent")
            else:
                requested = task.get("minimum_envelope_clearance_m")
                threshold = HARD_LIMITS["minimum_envelope_clearance_m"]
                if _number(requested):
                    threshold = max(threshold, float(requested))
                if computed <= threshold:
                    violations.append("minimum_envelope_clearance_below_hard_limit")

    if geometry_required:
        heading = environment.get("corridor_heading_error_rad")
        heading_limit = task.get(
            "maximum_corridor_heading_error_rad",
            environment.get("maximum_corridor_heading_error_rad"),
        )
        if not _number(heading) or not _number(heading_limit) or heading_limit <= 0:
            if "corridor_geometry_unavailable" not in violations:
                violations.append("corridor_geometry_unavailable")
        elif abs(float(heading)) > float(heading_limit):
            violations.append("corridor_heading_misaligned")

    return violations


def apply_safety_guard(
    scenario: dict[str, Any], model_decision: dict[str, Any]
) -> dict[str, Any]:
    """Return the only action that may be passed to a robot executor."""
    available_actions = scenario.get("available_actions", [])
    proposed_action = model_decision.get("selected_action", {})
    proposed_name = proposed_action.get("name")
    violations = motion_hard_violations(scenario)

    if proposed_name not in available_actions:
        violations.append("action_not_available")

    if violations:
        fallback = _safe_fallback(violations, available_actions)
        return {
            "status": "overridden",
            "model_decision": model_decision.get("decision"),
            "model_action": proposed_action,
            "approved_action": {"name": fallback, "parameters": {}},
            "hard_rule_violations": violations,
            "hard_limits": HARD_LIMITS,
        }

    return {
        "status": "approved",
        "model_decision": model_decision.get("decision"),
        "model_action": proposed_action,
        "approved_action": proposed_action,
        "hard_rule_violations": [],
        "hard_limits": HARD_LIMITS,
    }


def _safe_fallback(violations: list[str], available_actions: list[str]) -> str:
    immediate_stop_reasons = {
        "emergency_obstacle_distance",
        "battery_below_hard_limit",
        "corridor_clearance_below_hard_limit",
        "corridor_geometry_unavailable",
        "corridor_geometry_inconsistent",
        "minimum_envelope_clearance_below_hard_limit",
        "corridor_heading_misaligned",
        "action_not_available",
    }
    if immediate_stop_reasons.intersection(violations):
        return "safe_stop" if "safe_stop" in available_actions else ""
    if "observe_again" in available_actions:
        return "observe_again"
    return "safe_stop" if "safe_stop" in available_actions else ""
