"""Deterministic safety checks that no language-model decision may bypass."""

from __future__ import annotations

from typing import Any


HARD_LIMITS = {
    "emergency_stop_distance_m": 0.30,
    "minimum_observation_confidence": 0.50,
    "minimum_clearance_m": 0.10,
    "minimum_battery_percent": 10,
}


def apply_safety_guard(
    scenario: dict[str, Any], model_decision: dict[str, Any]
) -> dict[str, Any]:
    """Return the only action that may be passed to a robot executor."""
    robot = scenario.get("robot", {})
    environment = scenario.get("environment", {})
    available_actions = scenario.get("available_actions", [])
    proposed_action = model_decision.get("selected_action", {})
    proposed_name = proposed_action.get("name")
    violations: list[str] = []

    obstacle_distance = environment.get("obstacle_distance_m")
    if (
        environment.get("obstacle_detected") is True
        and isinstance(obstacle_distance, (int, float))
        and obstacle_distance <= HARD_LIMITS["emergency_stop_distance_m"]
    ):
        violations.append("emergency_obstacle_distance")

    observation_confidence = environment.get("observation_confidence")
    if (
        isinstance(observation_confidence, (int, float))
        and observation_confidence < HARD_LIMITS["minimum_observation_confidence"]
    ):
        violations.append("observation_confidence_below_hard_limit")

    battery = robot.get("battery_percent")
    if (
        isinstance(battery, (int, float))
        and battery <= HARD_LIMITS["minimum_battery_percent"]
    ):
        violations.append("battery_below_hard_limit")

    corridor_width = environment.get("corridor_width_m")
    robot_width = robot.get("width_m")
    if isinstance(corridor_width, (int, float)) and isinstance(
        robot_width, (int, float)
    ):
        clearance = corridor_width - robot_width
        if clearance < HARD_LIMITS["minimum_clearance_m"]:
            violations.append("corridor_clearance_below_hard_limit")

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
        "action_not_available",
    }
    if immediate_stop_reasons.intersection(violations):
        return "safe_stop" if "safe_stop" in available_actions else ""
    if "observe_again" in available_actions:
        return "observe_again"
    return "safe_stop" if "safe_stop" in available_actions else ""

