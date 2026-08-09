"""Local risk-experience storage and transparent similarity retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExperienceError(ValueError):
    """Raised when an experience record is structurally invalid."""


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


@dataclass(frozen=True)
class RetrievedExperience:
    score: float
    reasons: list[str]
    experience: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "match_reasons": self.reasons,
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
        self, scenario: dict[str, Any], *, top_k: int = 3
    ) -> list[RetrievedExperience]:
        if top_k < 1:
            raise ValueError("top_k 必须大于 0")
        results = [self._score(scenario, item) for item in self.load()]
        results = [item for item in results if item.score > 0]
        results.sort(
            key=lambda item: (
                item.score,
                item.experience["risk"].get("severity", 0),
            ),
            reverse=True,
        )
        return results[:top_k]

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

    @staticmethod
    def _score(
        scenario: dict[str, Any], experience: dict[str, Any]
    ) -> RetrievedExperience:
        score = 0.0
        reasons: list[str] = []
        robot = scenario.get("robot", {})
        environment = scenario.get("environment", {})
        old_robot = experience.get("robot", {})
        old_environment = experience.get("environment", {})

        if robot.get("type") == old_robot.get("type"):
            score += 1.5
            reasons.append("机器人类型相同")

        if environment.get("obstacle_detected") == old_environment.get(
            "obstacle_detected"
        ):
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
        if isinstance(obstacle_distance, (int, float)) and isinstance(
            old_obstacle_distance, (int, float)
        ):
            difference = abs(obstacle_distance - old_obstacle_distance)
            if difference <= 0.30:
                score += 2.0
                reasons.append("障碍物距离高度相似")
            elif difference <= 1.0:
                score += 0.75
                reasons.append("障碍物距离相近")

        confidence = environment.get("observation_confidence")
        old_confidence = old_environment.get("observation_confidence")
        if isinstance(confidence, (int, float)) and isinstance(
            old_confidence, (int, float)
        ):
            if abs(confidence - old_confidence) <= 0.15:
                score += 1.5
                reasons.append("观测置信度相近")

        current_goal = scenario.get("task", {}).get("goal", "")
        old_goal = experience.get("task_goal", "")
        if current_goal and old_goal and current_goal == old_goal:
            score += 1.0
            reasons.append("任务目标相同")

        return RetrievedExperience(score=score, reasons=reasons, experience=experience)


def _clearance(
    robot: dict[str, Any], environment: dict[str, Any]
) -> float | None:
    robot_width = robot.get("width_m")
    corridor_width = environment.get("corridor_width_m")
    if isinstance(robot_width, (int, float)) and isinstance(
        corridor_width, (int, float)
    ):
        return corridor_width - robot_width
    return None


def build_memory_card(item: RetrievedExperience) -> dict[str, Any]:
    """Compress one retrieved record into an auditable prompt-sized card."""
    experience = item.experience
    robot = experience.get("robot", {})
    environment = experience.get("environment", {})
    clearance = _clearance(robot, environment)
    return {
        "experience_id": experience["experience_id"],
        "relevance": {
            "score": round(item.score, 4),
            "match_reasons": item.reasons,
        },
        "key_conditions": {
            "clearance_m": round(clearance, 3) if clearance is not None else None,
            "obstacle_detected": environment.get("obstacle_detected"),
            "obstacle_distance_m": environment.get("obstacle_distance_m"),
            "observation_confidence": environment.get("observation_confidence"),
        },
        "action": experience["action"],
        "outcome": experience["outcome"],
        "risk": experience["risk"],
        "lesson": experience["lesson"],
    }
