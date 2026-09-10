"""Layered risk-memory cards and transparent risk-aware retrieval."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


class ExperienceError(ValueError):
    """Raised when an experience record is structurally invalid."""


AgentRole = Literal["advocate", "critic", "decision"]
VALID_ROLES: set[str] = {"advocate", "critic", "decision"}
MEMORY_SCHEMA_VERSION = 2
RETRIEVAL_VERSION = "risk-aware-v1"

REQUIRED_FIELDS = {
    "experience_id",
    "task_goal",
    "robot",
    "environment",
    "action",
    "outcome",
    "risk",
    "decision",
    "lesson",
}

ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    # Advocate needs applicable and reliable success/mitigation evidence.
    "advocate": {
        "similarity": 0.35,
        "severity": 0.05,
        "reliability": 0.20,
        "recency": 0.05,
        "role_fit": 0.35,
    },
    # Critic prioritizes high-consequence failure and stop-condition evidence.
    "critic": {
        "similarity": 0.30,
        "severity": 0.25,
        "reliability": 0.15,
        "recency": 0.05,
        "role_fit": 0.25,
    },
    # Decision receives a balanced evidence set.
    "decision": {
        "similarity": 0.35,
        "severity": 0.20,
        "reliability": 0.20,
        "recency": 0.10,
        "role_fit": 0.15,
    },
}


@dataclass(frozen=True)
class RetrievedExperience:
    score: float
    reasons: list[str]
    experience: dict[str, Any]
    score_breakdown: dict[str, float] = field(default_factory=dict)
    role: str = "decision"
    redundancy_penalty: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "score_breakdown": {
                key: round(value, 4)
                for key, value in self.score_breakdown.items()
            },
            "match_reasons": self.reasons,
            "role": self.role,
            "redundancy_penalty": round(self.redundancy_penalty, 4),
            "experience": self.experience,
        }


class ExperienceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ExperienceError("经验库顶层必须是 JSON 数组")
        seen_ids: set[str] = set()
        for index, item in enumerate(data):
            self._validate(item, index)
            experience_id = item["experience_id"]
            if experience_id in seen_ids:
                raise ExperienceError(f"经验 ID 重复: {experience_id}")
            seen_ids.add(experience_id)
        return data

    def retrieve(
        self,
        scenario: dict[str, Any],
        *,
        top_k: int = 3,
        role: AgentRole = "decision",
    ) -> list[RetrievedExperience]:
        """Return risk-aware, role-specific experiences.

        This keeps the old ``retrieve(scenario, top_k=...)`` interface while
        replacing the old raw-similarity score with an explainable composite.
        """
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        _validate_role(role)
        candidates = [self._score(scenario, item, role) for item in self.load()]
        candidates = [item for item in candidates if item.score_breakdown["similarity"] > 0]
        return _select_diverse(candidates, top_k=top_k)

    def retrieve_memory_cards(
        self,
        scenario: dict[str, Any],
        *,
        role: AgentRole,
        top_k: int = 3,
        token_budget: int = 1200,
    ) -> dict[str, Any]:
        """Build an auditable evidence bundle within a prompt-token budget."""
        if token_budget < 1:
            raise ValueError("token_budget 必须大于 0")
        retrieved = self.retrieve(scenario, top_k=max(top_k * 3, top_k), role=role)
        cards: list[dict[str, Any]] = []
        used_tokens = 0
        for item in retrieved:
            card = build_memory_card(item)
            cost = estimate_tokens(card)
            if len(cards) >= top_k:
                break
            if used_tokens + cost > token_budget:
                continue
            card["retrieval"]["estimated_tokens"] = cost
            cards.append(card)
            used_tokens += cost
        return {
            "role": role,
            "retrieval_version": RETRIEVAL_VERSION,
            "token_budget": token_budget,
            "estimated_tokens_used": used_tokens,
            "items": cards,
        }

    @staticmethod
    def _validate(item: Any, index: int) -> None:
        if not isinstance(item, dict):
            raise ExperienceError(f"第 {index} 条经验必须是对象")
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ExperienceError(
                f"第 {index} 条经验缺少字段: {sorted(missing)}"
            )
        if not isinstance(item["experience_id"], str) or not item["experience_id"]:
            raise ExperienceError(f"第 {index} 条 experience_id 无效")
        if item["outcome"].get("status") not in {"success", "failure", "aborted"}:
            raise ExperienceError(f"第 {index} 条 outcome.status 无效")
        severity = item["risk"].get("severity")
        if not isinstance(severity, int) or not 1 <= severity <= 5:
            raise ExperienceError(f"第 {index} 条 risk.severity 必须为 1 到 5")
        source = item.get("source")
        if source is not None:
            if not isinstance(source, dict):
                raise ExperienceError(f"第 {index} 条 source 必须是对象")
            if (
                source.get("kind") == "human_constructed"
                and source.get("historical_physical_run") is not False
            ):
                raise ExperienceError(
                    f"第 {index} 条人工构造经验必须明确标注 historical_physical_run=false"
                )

    @staticmethod
    def _score(
        scenario: dict[str, Any],
        experience: dict[str, Any],
        role: AgentRole = "decision",
    ) -> RetrievedExperience:
        similarity, reasons = _similarity(scenario, experience)
        components = {
            "similarity": min(similarity / 10.0, 1.0),
            "severity": experience["risk"].get("severity", 1) / 5.0,
            "reliability": _reliability(experience),
            "recency": _recency(experience),
            "role_fit": _role_fit(experience, role),
        }
        weights = ROLE_WEIGHTS[role]
        weighted = {
            key: components[key] * weights[key]
            for key in components
        }
        score = sum(weighted.values())

        severity = experience["risk"].get("severity", 1)
        if severity == 5:
            reasons.append("最高严重度经验进入优先候选")
        reasons.append(f"面向 {role} 角色完成风险感知精排")
        breakdown = {
            **components,
            **{f"weighted_{key}": value for key, value in weighted.items()},
        }
        return RetrievedExperience(
            score=score,
            reasons=reasons,
            experience=experience,
            score_breakdown=breakdown,
            role=role,
        )


def _validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise ValueError(f"role 必须属于 {sorted(VALID_ROLES)}")


def _similarity(
    scenario: dict[str, Any], experience: dict[str, Any]
) -> tuple[float, list[str]]:
    """Transparent scenario similarity, normalized later to 0..1."""
    score = 0.0
    reasons: list[str] = []
    robot = scenario.get("robot", {})
    environment = scenario.get("environment", {})
    old_robot = experience.get("robot", {})
    old_environment = experience.get("environment", {})

    if robot.get("type") == old_robot.get("type"):
        score += 1.5
        reasons.append("机器人类型相同")
    if environment.get("obstacle_detected") == old_environment.get("obstacle_detected"):
        score += 1.0
        reasons.append("障碍物检测状态相同")

    clearance = _clearance(robot, environment)
    old_clearance = _clearance(old_robot, old_environment)
    if clearance is not None and old_clearance is not None:
        difference = abs(clearance - old_clearance)
        if difference <= 0.05:
            score += 3.0
            reasons.append("通行余量高度相似")
        elif difference <= 0.15:
            score += 1.5
            reasons.append("通行余量相近")

    obstacle_distance = environment.get("obstacle_distance_m")
    old_obstacle_distance = old_environment.get("obstacle_distance_m")
    if _number(obstacle_distance) and _number(old_obstacle_distance):
        difference = abs(float(obstacle_distance) - float(old_obstacle_distance))
        if difference <= 0.30:
            score += 2.0
            reasons.append("障碍物距离高度相似")
        elif difference <= 1.0:
            score += 0.75
            reasons.append("障碍物距离相近")

    confidence = environment.get("observation_confidence")
    old_confidence = old_environment.get("observation_confidence")
    if _number(confidence) and _number(old_confidence):
        if abs(float(confidence) - float(old_confidence)) <= 0.15:
            score += 1.5
            reasons.append("观测置信度相近")

    current_goal = scenario.get("task", {}).get("goal", "")
    old_goal = experience.get("task_goal", "")
    if current_goal and old_goal and current_goal == old_goal:
        score += 1.0
        reasons.append("任务目标相同")
    return score, reasons


def _reliability(experience: dict[str, Any]) -> float:
    quality = experience.get("quality", {})
    explicit = quality.get("confidence")
    if _number(explicit):
        return _clamp(float(explicit))
    stats = experience.get("statistics", {})
    successes = stats.get("success")
    failures = stats.get("failure")
    if isinstance(successes, int) and isinstance(failures, int):
        # Beta(1,1) posterior mean prevents one observation from looking certain.
        return (successes + 1) / (successes + failures + 2)
    status = experience.get("outcome", {}).get("status")
    return {"success": 0.85, "failure": 0.75, "aborted": 0.65}.get(status, 0.5)


def _recency(experience: dict[str, Any]) -> float:
    stamp = experience.get("quality", {}).get("last_verified_at")
    if not isinstance(stamp, str) or not stamp:
        return 0.5
    try:
        then = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    age_days = max((datetime.now(timezone.utc) - then).total_seconds() / 86400, 0)
    return math.exp(-age_days / 180.0)


def _role_fit(experience: dict[str, Any], role: AgentRole) -> float:
    status = experience.get("outcome", {}).get("status")
    severity = experience.get("risk", {}).get("severity", 1)
    action = experience.get("action", {}).get("name")
    if role == "advocate":
        return 1.0 if status == "success" else 0.35
    if role == "critic":
        if status in {"failure", "aborted"}:
            return 1.0
        if severity >= 4 or action == "safe_stop":
            return 0.8
        return 0.4
    # Decision values both counterexamples and verified mitigations.
    return 0.9 if status in {"success", "failure"} else 0.7


def _select_diverse(
    candidates: list[RetrievedExperience], *, top_k: int
) -> list[RetrievedExperience]:
    """Greedy MMR-style selection with an explicit redundancy penalty."""
    remaining = list(candidates)
    selected: list[RetrievedExperience] = []
    while remaining and len(selected) < top_k:
        best: RetrievedExperience | None = None
        best_marginal = -1.0
        best_penalty = 0.0
        for item in remaining:
            penalty = max((_redundancy(item, old) for old in selected), default=0.0)
            marginal = item.score - 0.15 * penalty
            if marginal > best_marginal:
                best, best_marginal, best_penalty = item, marginal, penalty
        assert best is not None
        selected.append(
            RetrievedExperience(
                score=max(best_marginal, 0.0),
                reasons=best.reasons,
                experience=best.experience,
                score_breakdown=best.score_breakdown,
                role=best.role,
                redundancy_penalty=best_penalty,
            )
        )
        remaining.remove(best)
    return selected


def _redundancy(a: RetrievedExperience, b: RetrievedExperience) -> float:
    score = 0.0
    if a.experience["risk"].get("type") == b.experience["risk"].get("type"):
        score += 0.5
    if a.experience["outcome"].get("status") == b.experience["outcome"].get("status"):
        score += 0.25
    if a.experience.get("lesson") == b.experience.get("lesson"):
        score += 0.25
    return score


def _clearance(
    robot: dict[str, Any], environment: dict[str, Any]
) -> float | None:
    envelope_clearance = environment.get("minimum_envelope_clearance_m")
    if _number(envelope_clearance):
        return float(envelope_clearance)
    robot_width = robot.get("width_m")
    corridor_width = environment.get("corridor_width_m")
    if _number(robot_width) and _number(corridor_width):
        return float(corridor_width) - float(robot_width)
    return None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_memory_card(item: RetrievedExperience) -> dict[str, Any]:
    """Compress one retrieved record into an auditable L1 Risk Memory Card."""
    experience = item.experience
    robot = experience.get("robot", {})
    environment = experience.get("environment", {})
    risk = experience.get("risk", {})
    outcome = experience.get("outcome", {})
    clearance = _clearance(robot, environment)
    memory_id = experience.get("memory_id", f"rmc_{experience['experience_id']}")
    mitigation = experience.get("mitigations")
    if not isinstance(mitigation, list):
        mitigation = [experience["lesson"]]
    stop_conditions = experience.get("stop_conditions", [])
    if not isinstance(stop_conditions, list):
        stop_conditions = []
    statistics = experience.get("statistics", {})
    if not statistics:
        status = outcome.get("status")
        statistics = {
            "support": 1,
            "success": 1 if status == "success" else 0,
            "failure": 1 if status == "failure" else 0,
            "aborted": 1 if status == "aborted" else 0,
        }
    reliability = item.score_breakdown.get("reliability", _reliability(experience))

    card = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "memory_level": "L1_risk_memory_card",
        "memory_id": memory_id,
        # Kept for backward compatibility with current Agent validators.
        "experience_id": experience["experience_id"],
        "source": experience.get(
            "source",
            {
                "trace_id": f"legacy_{experience['experience_id']}",
                "evidence_span": None,
            },
        ),
        "context": {
            "robot_type": robot.get("type"),
            "platform": robot.get("platform", robot.get("type")),
            "task_goal": experience.get("task_goal"),
        },
        "trigger_conditions": {
            "clearance_m": round(clearance, 3) if clearance is not None else None,
            "corridor_width_m": environment.get("corridor_width_m"),
            "left_envelope_clearance_m": environment.get("left_envelope_clearance_m"),
            "right_envelope_clearance_m": environment.get("right_envelope_clearance_m"),
            "minimum_envelope_clearance_m": environment.get("minimum_envelope_clearance_m"),
            "clearance_uncertainty_m": environment.get("clearance_uncertainty_m"),
            "corridor_heading_error_rad": environment.get("corridor_heading_error_rad"),
            "obstacle_detected": environment.get("obstacle_detected"),
            "obstacle_distance_m": environment.get("obstacle_distance_m"),
            "observation_confidence": environment.get("observation_confidence"),
            "battery_percent": robot.get("battery_percent"),
        },
        # Alias retained so older displays/tests keep working.
        "key_conditions": {
            "clearance_m": round(clearance, 3) if clearance is not None else None,
            "obstacle_detected": environment.get("obstacle_detected"),
            "obstacle_distance_m": environment.get("obstacle_distance_m"),
            "observation_confidence": environment.get("observation_confidence"),
        },
        "hazard": {
            "type": risk.get("type"),
            "severity": risk.get("severity"),
        },
        "action": experience["action"],
        "outcome": outcome,
        "failure_reason": risk.get("failure_reason"),
        "mitigations": mitigation,
        "stop_conditions": stop_conditions,
        "statistics": statistics,
        "quality": {
            "confidence": round(reliability, 4),
            "last_verified_at": experience.get("quality", {}).get("last_verified_at"),
        },
        "rule_links": experience.get("rule_links", []),
        "retrieval": {
            "role": item.role,
            "score": round(item.score, 4),
            "score_breakdown": {
                key: round(value, 4)
                for key, value in item.score_breakdown.items()
                if not key.startswith("weighted_")
            },
            "match_reasons": item.reasons,
            "redundancy_penalty": round(item.redundancy_penalty, 4),
            "retrieval_version": RETRIEVAL_VERSION,
        },
        # Legacy fields for callers that still render the old card shape.
        "relevance": {
            "score": round(item.score, 4),
            "match_reasons": item.reasons,
        },
        "risk": risk,
        "lesson": experience["lesson"],
    }
    return card


def estimate_tokens(value: Any) -> int:
    """Conservative offline estimate for Chinese/English JSON prompt cost."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = len(re.sub(r"[\s\u3400-\u9fff]", "", text))
    return cjk + math.ceil(non_cjk / 4)
