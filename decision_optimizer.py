"""Deterministic constrained action selection utilities.

The module currently provides feasibility filtering and an explicit baseline
task-utility model. Risk and uncertainty scoring are added in later steps.
"""

from __future__ import annotations

from typing import Any, Iterable

from safety_guard import HARD_LIMITS, motion_hard_violations


MOTION_ACTIONS = {"move_forward", "turn_left", "turn_right", "slow_down"}

# Hand-configured prototype values in [0, 1]. They are intentionally kept in
# one auditable table so later simulator calibration can replace them without
# changing the optimizer logic.
ACTION_TASK_PROFILES = {
    "move_forward": {"progress": 1.00, "time_cost": 0.10, "energy_cost": 0.25},
    "slow_down": {"progress": 0.65, "time_cost": 0.30, "energy_cost": 0.15},
    "turn_left": {"progress": 0.20, "time_cost": 0.25, "energy_cost": 0.10},
    "turn_right": {"progress": 0.20, "time_cost": 0.25, "energy_cost": 0.10},
    "observe_again": {"progress": 0.00, "time_cost": 0.25, "energy_cost": 0.05},
    "ask_human": {"progress": 0.00, "time_cost": 0.80, "energy_cost": 0.00},
    "safe_stop": {"progress": 0.00, "time_cost": 1.00, "energy_cost": 0.00},
}

TASK_UTILITY_WEIGHTS = {
    "progress": 0.65,
    "time_cost": 0.20,
    "energy_cost": 0.15,
}

SCENE_RISK_WEIGHTS = {
    "clearance": 0.35,
    "obstacle_proximity": 0.30,
    "observation_uncertainty": 0.25,
    "battery_scarcity": 0.10,
}

ACTION_RISK_EXPOSURE = {
    "move_forward": 1.00,
    "slow_down": 0.55,
    "turn_left": 0.65,
    "turn_right": 0.65,
    "observe_again": 0.08,
    "ask_human": 0.02,
    "safe_stop": 0.00,
}

RISK_SOURCE_WEIGHTS = {"scene": 0.70, "memory": 0.30}

UNCERTAINTY_WEIGHTS = {
    "observation": 0.45,
    "memory": 0.30,
    "agent": 0.25,
}

OBSERVATION_UNCERTAINTY_EXPOSURE = {
    "move_forward": 1.00,
    "slow_down": 0.70,
    "turn_left": 0.80,
    "turn_right": 0.80,
    "observe_again": 0.10,
    "ask_human": 0.05,
    "safe_stop": 0.00,
}

MEMORY_EVIDENCE_REQUIREMENT = {
    "move_forward": 1.00,
    "slow_down": 0.80,
    "turn_left": 0.70,
    "turn_right": 0.70,
    "observe_again": 0.10,
    "ask_human": 0.10,
    "safe_stop": 0.05,
}

TARGET_MEMORY_SUPPORT = 1.50

OPTIMIZER_LIMITS = {
    "maximum_risk": 0.35,
    "maximum_uncertainty": 0.45,
}


def compute_task_utility(action: str) -> dict[str, Any]:
    """Return an explainable deterministic task-utility score for one action.

    ``utility = progress_weight * progress
               - time_weight * time_cost
               - energy_weight * energy_cost``

    The profiles are baseline engineering assumptions, not learned values.
    """
    if action not in ACTION_TASK_PROFILES:
        raise ValueError(f"missing task-utility profile for action: {action}")

    profile = ACTION_TASK_PROFILES[action]
    contributions = {
        "progress": TASK_UTILITY_WEIGHTS["progress"] * profile["progress"],
        "time_cost": -TASK_UTILITY_WEIGHTS["time_cost"] * profile["time_cost"],
        "energy_cost": -TASK_UTILITY_WEIGHTS["energy_cost"]
        * profile["energy_cost"],
    }
    score = round(sum(contributions.values()), 6)
    return {
        "score": score,
        "profile": dict(profile),
        "weights": dict(TASK_UTILITY_WEIGHTS),
        "contributions": {
            key: round(value, 6) for key, value in contributions.items()
        },
        "parameter_source": "hand_configured_baseline",
    }


def score_task_utilities(
    scenario: dict[str, Any], actions: Iterable[str] | None = None
) -> dict[str, Any]:
    """Score the requested actions, or every available action in the scenario."""
    selected_actions = (
        list(actions)
        if actions is not None
        else list(scenario.get("available_actions", []))
    )
    scores = {
        action: compute_task_utility(action) for action in selected_actions
    }
    ranked_actions = sorted(
        selected_actions,
        key=lambda action: (-scores[action]["score"], action),
    )
    return {
        "items": scores,
        "ranked_actions": ranked_actions,
        "parameter_source": "hand_configured_baseline",
    }


def estimate_action_risk(
    scenario: dict[str, Any],
    action: str,
    memories: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Estimate an explainable risk score from state and matching L1 cards.

    This is a deterministic baseline estimator, not a calibrated probability.
    A memory contributes only when it records the same action being assessed.
    """
    if action not in ACTION_RISK_EXPOSURE:
        raise ValueError(f"missing risk-exposure profile for action: {action}")

    robot = scenario.get("robot", {})
    environment = scenario.get("environment", {})
    envelope_clearance = _as_number(
        environment.get("minimum_envelope_clearance_m")
    )
    if envelope_clearance is not None:
        # Side-specific clearance is the relevant quantity for a robot that is
        # not perfectly centred.  0.20 m per side is the zero-risk reference.
        clearance_risk = _clamp01((0.20 - envelope_clearance) / 0.20)
    else:
        clearance = _numeric_difference(
            environment.get("corridor_width_m"), robot.get("width_m")
        )
        clearance_risk = (
            _clamp01((0.40 - clearance) / 0.40)
            if clearance is not None
            else 0.5
        )

    obstacle_detected = environment.get("obstacle_detected") is True
    obstacle_distance = _as_number(environment.get("obstacle_distance_m"))
    obstacle_risk = 0.0
    if obstacle_detected:
        obstacle_risk = (
            _clamp01((1.50 - obstacle_distance) / 1.50)
            if obstacle_distance is not None
            else 0.5
        )

    confidence = _as_number(environment.get("observation_confidence"))
    confidence_risk = _clamp01(1.0 - confidence) if confidence is not None else 0.5
    battery = _as_number(robot.get("battery_percent"))
    battery_risk = _clamp01(1.0 - battery / 100.0) if battery is not None else 0.5

    scene_factors = {
        "clearance": clearance_risk,
        "obstacle_proximity": obstacle_risk,
        "observation_uncertainty": confidence_risk,
        "battery_scarcity": battery_risk,
    }
    weighted_scene_factors = {
        name: value * SCENE_RISK_WEIGHTS[name]
        for name, value in scene_factors.items()
    }
    context_risk = sum(weighted_scene_factors.values())
    exposure = ACTION_RISK_EXPOSURE[action]
    scene_action_risk = _clamp01(context_risk * exposure)

    memory_evidence: list[dict[str, Any]] = []
    weighted_memory_sum = 0.0
    total_memory_weight = 0.0
    for card in memories:
        if card.get("action", {}).get("name") != action:
            continue
        similarity = _clamp01(
            _as_number(card.get("retrieval", {}).get("score_breakdown", {}).get("similarity"))
            or 0.0
        )
        reliability = _clamp01(
            _as_number(card.get("quality", {}).get("confidence")) or 0.5
        )
        severity = _clamp01(
            (_as_number(card.get("hazard", {}).get("severity")) or 1.0) / 5.0
        )
        status = card.get("outcome", {}).get("status")
        outcome_factor = {"failure": 1.0, "aborted": 0.8, "success": 0.05}.get(
            status, 0.5
        )
        evidence_weight = similarity * reliability
        evidence_risk = severity * outcome_factor
        weighted_memory_sum += evidence_weight * evidence_risk
        total_memory_weight += evidence_weight
        memory_evidence.append(
            {
                "memory_id": card.get("memory_id"),
                "experience_id": card.get("experience_id"),
                "similarity": round(similarity, 6),
                "reliability": round(reliability, 6),
                "severity": round(severity, 6),
                "outcome_factor": round(outcome_factor, 6),
                "evidence_weight": round(evidence_weight, 6),
                "evidence_risk": round(evidence_risk, 6),
            }
        )

    has_memory_evidence = total_memory_weight > 0
    memory_risk = (
        _clamp01(weighted_memory_sum / total_memory_weight)
        if has_memory_evidence
        else None
    )
    if memory_risk is None:
        final_risk = scene_action_risk
        applied_source_weights = {"scene": 1.0, "memory": 0.0}
    else:
        final_risk = (
            RISK_SOURCE_WEIGHTS["scene"] * scene_action_risk
            + RISK_SOURCE_WEIGHTS["memory"] * memory_risk
        )
        applied_source_weights = dict(RISK_SOURCE_WEIGHTS)

    return {
        "score": round(_clamp01(final_risk), 6),
        "scene_action_risk": round(scene_action_risk, 6),
        "memory_risk": round(memory_risk, 6) if memory_risk is not None else None,
        "scene_factors": {
            key: round(value, 6) for key, value in scene_factors.items()
        },
        "weighted_scene_factors": {
            key: round(value, 6) for key, value in weighted_scene_factors.items()
        },
        "action_exposure": exposure,
        "source_weights": applied_source_weights,
        "memory_evidence_count": len(memory_evidence),
        "memory_evidence": memory_evidence,
        "parameter_source": "hand_configured_uncalibrated_baseline",
        "score_semantics": "relative_risk_score_not_probability",
    }


def score_action_risks(
    scenario: dict[str, Any],
    actions: Iterable[str],
    memories: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Score and rank candidate actions from lowest to highest risk."""
    selected_actions = list(actions)
    memory_cards = list(memories)
    scores = {
        action: estimate_action_risk(scenario, action, memory_cards)
        for action in selected_actions
    }
    return {
        "items": scores,
        "ranked_actions": sorted(
            selected_actions,
            key=lambda action: (scores[action]["score"], action),
        ),
        "parameter_source": "hand_configured_uncalibrated_baseline",
    }


def estimate_action_uncertainty(
    scenario: dict[str, Any],
    action: str,
    memories: Iterable[dict[str, Any]] = (),
    agent_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate observation, memory, and role-disagreement uncertainty."""
    if action not in OBSERVATION_UNCERTAINTY_EXPOSURE:
        raise ValueError(f"missing uncertainty profile for action: {action}")

    confidence = _as_number(
        scenario.get("environment", {}).get("observation_confidence")
    )
    raw_observation_uncertainty = (
        _clamp01(1.0 - confidence) if confidence is not None else 1.0
    )
    observation_uncertainty = (
        raw_observation_uncertainty
        * OBSERVATION_UNCERTAINTY_EXPOSURE[action]
    )

    relevant_memory: list[dict[str, Any]] = []
    memory_support = 0.0
    for card in memories:
        if card.get("action", {}).get("name") != action:
            continue
        similarity_value = _as_number(
            card.get("retrieval", {}).get("score_breakdown", {}).get("similarity")
        )
        reliability_value = _as_number(card.get("quality", {}).get("confidence"))
        similarity = _clamp01(similarity_value if similarity_value is not None else 0.0)
        reliability = _clamp01(
            reliability_value if reliability_value is not None else 0.5
        )
        contribution = similarity * reliability
        memory_support += contribution
        relevant_memory.append(
            {
                "memory_id": card.get("memory_id"),
                "experience_id": card.get("experience_id"),
                "support_contribution": round(contribution, 6),
            }
        )
    memory_coverage = _clamp01(memory_support / TARGET_MEMORY_SUPPORT)
    raw_memory_uncertainty = 1.0 - memory_coverage
    memory_uncertainty = (
        raw_memory_uncertainty * MEMORY_EVIDENCE_REQUIREMENT[action]
    )

    reports = agent_reports or {}
    valid_votes: list[dict[str, Any]] = []
    total_vote_weight = 0.0
    action_support = 0.0
    for role, report in reports.items():
        recommended_action = _extract_agent_action(report)
        if recommended_action is None:
            continue
        reported_confidence = _as_number(report.get("confidence"))
        vote_weight = _clamp01(
            reported_confidence if reported_confidence is not None else 0.5
        )
        total_vote_weight += vote_weight
        if recommended_action == action:
            action_support += vote_weight
        valid_votes.append(
            {
                "role": role,
                "action": recommended_action,
                "confidence_weight": round(vote_weight, 6),
            }
        )
    if total_vote_weight > 0:
        support_ratio = _clamp01(action_support / total_vote_weight)
        agent_uncertainty = 1.0 - support_ratio
        agent_evidence_available = True
    else:
        support_ratio = 0.0
        agent_uncertainty = 1.0
        agent_evidence_available = False

    components = {
        "observation": observation_uncertainty,
        "memory": memory_uncertainty,
        "agent": agent_uncertainty,
    }
    weighted_components = {
        name: value * UNCERTAINTY_WEIGHTS[name]
        for name, value in components.items()
    }
    score = _clamp01(sum(weighted_components.values()))
    return {
        "score": round(score, 6),
        "components": {
            key: round(value, 6) for key, value in components.items()
        },
        "weighted_components": {
            key: round(value, 6) for key, value in weighted_components.items()
        },
        "weights": dict(UNCERTAINTY_WEIGHTS),
        "observation": {
            "confidence": confidence,
            "raw_uncertainty": round(raw_observation_uncertainty, 6),
            "action_exposure": OBSERVATION_UNCERTAINTY_EXPOSURE[action],
        },
        "memory": {
            "support": round(memory_support, 6),
            "target_support": TARGET_MEMORY_SUPPORT,
            "coverage": round(memory_coverage, 6),
            "raw_uncertainty": round(raw_memory_uncertainty, 6),
            "action_requirement": MEMORY_EVIDENCE_REQUIREMENT[action],
            "evidence_count": len(relevant_memory),
            "evidence": relevant_memory,
        },
        "agent": {
            "evidence_available": agent_evidence_available,
            "support_ratio": round(support_ratio, 6),
            "votes": valid_votes,
        },
        "parameter_source": "hand_configured_uncalibrated_baseline",
        "score_semantics": "relative_uncertainty_score_not_probability",
    }


def score_action_uncertainties(
    scenario: dict[str, Any],
    actions: Iterable[str],
    memories: Iterable[dict[str, Any]] = (),
    agent_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score and rank candidate actions from lowest to highest uncertainty."""
    selected_actions = list(actions)
    memory_cards = list(memories)
    scores = {
        action: estimate_action_uncertainty(
            scenario, action, memory_cards, agent_reports
        )
        for action in selected_actions
    }
    return {
        "items": scores,
        "ranked_actions": sorted(
            selected_actions,
            key=lambda action: (scores[action]["score"], action),
        ),
        "parameter_source": "hand_configured_uncalibrated_baseline",
    }


def optimize_action(
    scenario: dict[str, Any],
    memories: Iterable[dict[str, Any]] = (),
    matched_rules: Iterable[dict[str, Any]] = (),
    agent_reports: dict[str, dict[str, Any]] | None = None,
    *,
    maximum_risk: float = OPTIMIZER_LIMITS["maximum_risk"],
    maximum_uncertainty: float = OPTIMIZER_LIMITS["maximum_uncertainty"],
) -> dict[str, Any]:
    """Solve constrained action selection over the finite candidate set.

    The objective is maximum task utility subject to hard feasibility, L2
    prohibitions, a relative-risk limit, and a relative-uncertainty limit.
    """
    _validate_unit_interval(maximum_risk, "maximum_risk")
    _validate_unit_interval(maximum_uncertainty, "maximum_uncertainty")
    memory_cards = list(memories)
    rules = list(matched_rules)
    feasibility = filter_safe_actions(scenario, rules)
    feasible_actions = feasibility["feasible_actions"]

    utilities = score_task_utilities(scenario, feasible_actions)["items"]
    risks = score_action_risks(scenario, feasible_actions, memory_cards)["items"]
    uncertainties = score_action_uncertainties(
        scenario, feasible_actions, memory_cards, agent_reports
    )["items"]

    candidates: dict[str, dict[str, Any]] = {}
    eligible_actions: list[str] = []
    for action in scenario.get("available_actions", []):
        hard_assessment = feasibility["action_assessments"][action]
        if not hard_assessment["feasible"]:
            candidates[action] = {
                "eligible": False,
                "status": "rejected_by_hard_or_rule_constraint",
                "constraint_violations": list(
                    hard_assessment["rejection_reasons"]
                ),
                "task_utility": None,
                "risk": None,
                "uncertainty": None,
            }
            continue

        violations: list[str] = []
        risk_score = risks[action]["score"]
        uncertainty_score = uncertainties[action]["score"]
        if risk_score > maximum_risk:
            violations.append("risk_above_limit")
        if uncertainty_score > maximum_uncertainty:
            violations.append("uncertainty_above_limit")
        eligible = not violations
        candidates[action] = {
            "eligible": eligible,
            "status": "eligible" if eligible else "rejected_by_numeric_constraint",
            "constraint_violations": violations,
            "task_utility": utilities[action],
            "risk": risks[action],
            "uncertainty": uncertainties[action],
        }
        if eligible:
            eligible_actions.append(action)

    if eligible_actions:
        selected_action = sorted(
            eligible_actions,
            key=lambda action: (
                -utilities[action]["score"],
                risks[action]["score"],
                uncertainties[action]["score"],
                action,
            ),
        )[0]
        status = "optimized"
        reason = "在所有约束均满足的动作中选择任务收益最高者"
    else:
        available_actions = list(scenario.get("available_actions", []))
        if "safe_stop" in available_actions:
            selected_action = "safe_stop"
        elif "observe_again" in available_actions:
            selected_action = "observe_again"
        else:
            selected_action = ""
        status = "fallback_no_eligible_action"
        reason = "没有动作同时满足安全、风险与不确定性约束，采用保守回退"

    return {
        "status": status,
        "selected_action": {"name": selected_action, "parameters": {}},
        "reason": reason,
        "eligible_actions": eligible_actions,
        "candidates": candidates,
        "constraints": {
            "maximum_risk": maximum_risk,
            "maximum_uncertainty": maximum_uncertainty,
            "limits_source": "hand_configured_uncalibrated_baseline",
            "risk_semantics": "relative_risk_score_not_probability",
            "uncertainty_semantics": "relative_uncertainty_score_not_probability",
        },
        "objective": "maximize_task_utility_subject_to_constraints",
        "decision_source": "deterministic_constrained_optimizer",
    }


def _extract_agent_action(report: dict[str, Any]) -> str | None:
    selected_action = report.get("selected_action")
    if isinstance(selected_action, dict) and isinstance(selected_action.get("name"), str):
        return selected_action["name"]
    recommendation = report.get("recommendation")
    if isinstance(recommendation, str) and recommendation:
        return recommendation
    proposed_action = report.get("proposed_action")
    if isinstance(proposed_action, dict) and isinstance(proposed_action.get("name"), str):
        return proposed_action["name"]
    return None


def _validate_unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1]")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1]")


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _numeric_difference(left: Any, right: Any) -> float | None:
    left_number = _as_number(left)
    right_number = _as_number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def filter_safe_actions(
    scenario: dict[str, Any],
    matched_rules: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Classify every available action before numerical optimization.

    Hard physical constraints and L2 ``prohibited_actions`` are treated as
    feasibility constraints, rather than soft penalties that a high task
    utility could offset.
    """
    available_actions = list(scenario.get("available_actions", []))
    robot = scenario.get("robot", {})
    environment = scenario.get("environment", {})

    global_motion_reasons = motion_hard_violations(scenario)

    prohibited_by_rule: dict[str, list[str]] = {}
    for rule in matched_rules:
        rule_id = str(rule.get("rule_id", "unknown_rule"))
        for action in rule.get("prohibited_actions", []):
            prohibited_by_rule.setdefault(str(action), []).append(rule_id)

    action_assessments: dict[str, dict[str, Any]] = {}
    feasible_actions: list[str] = []
    for action in available_actions:
        reasons: list[str] = []
        if action in MOTION_ACTIONS:
            reasons.extend(global_motion_reasons)
        reasons.extend(
            f"prohibited_by_l2_rule:{rule_id}"
            for rule_id in prohibited_by_rule.get(action, [])
        )
        feasible = not reasons
        action_assessments[action] = {
            "feasible": feasible,
            "rejection_reasons": reasons,
        }
        if feasible:
            feasible_actions.append(action)

    fallback_required = not feasible_actions
    selected_fallback = None
    if fallback_required:
        if "safe_stop" in available_actions:
            selected_fallback = "safe_stop"
        elif "observe_again" in available_actions:
            selected_fallback = "observe_again"

    return {
        "feasible_actions": feasible_actions,
        "action_assessments": action_assessments,
        "fallback_required": fallback_required,
        "selected_fallback": selected_fallback,
        "hard_limits": HARD_LIMITS,
    }
