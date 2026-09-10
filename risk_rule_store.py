"""Load and validate compact L2 risk rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RiskRuleError(ValueError):
    """Raised when an L2 risk-rule file is structurally invalid."""


REQUIRED_RULE_FIELDS = {
    "rule_id",
    "memory_level",
    "risk_type",
    "trigger_conditions",
    "rule",
    "recommended_actions",
    "prohibited_actions",
    "source_memory_ids",
    "statistics",
    "confidence",
    "status",
}

SUPPORTED_TRIGGER_FIELDS = {
    "maximum_clearance_m",
    "maximum_observation_confidence",
    "minimum_observation_confidence",
    "maximum_minimum_envelope_clearance_m",
    "minimum_minimum_envelope_clearance_m",
}


class RiskRuleStore:
    """Persistent store for reusable rules distilled from L1 memory cards."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RiskRuleError("风险规则库顶层必须是 JSON 对象")
        if data.get("schema_version") != 1:
            raise RiskRuleError("风险规则库 schema_version 必须为 1")

        rules = data.get("rules")
        if not isinstance(rules, list):
            raise RiskRuleError("风险规则库 rules 必须是数组")

        seen_ids: set[str] = set()
        for index, rule in enumerate(rules):
            self._validate_rule(rule, index)
            rule_id = rule["rule_id"]
            if rule_id in seen_ids:
                raise RiskRuleError(f"风险规则 ID 重复: {rule_id}")
            seen_ids.add(rule_id)
        return rules

    def retrieve_matching(self, scenario: dict[str, Any]) -> list[dict[str, Any]]:
        """Return active L2 rules whose trigger conditions match a scenario."""
        matches: list[dict[str, Any]] = []
        for rule in self.load():
            if rule["status"] != "active":
                continue
            reasons = _match_trigger_conditions(
                scenario, rule["trigger_conditions"]
            )
            if reasons is None:
                continue
            matches.append({
                "rule_id": rule["rule_id"],
                "memory_level": rule["memory_level"],
                "risk_type": rule["risk_type"],
                "rule": rule["rule"],
                "recommended_actions": rule["recommended_actions"],
                "prohibited_actions": rule["prohibited_actions"],
                "mitigations": rule.get("mitigations", []),
                "source_memory_ids": rule["source_memory_ids"],
                "source": rule.get("source"),
                "confidence": rule["confidence"],
                "match_reasons": reasons,
            })
        return sorted(
            matches,
            key=lambda item: (item["confidence"], len(item["source_memory_ids"])),
            reverse=True,
        )

    @staticmethod
    def _validate_rule(rule: Any, index: int) -> None:
        if not isinstance(rule, dict):
            raise RiskRuleError(f"第 {index} 条风险规则必须是对象")

        missing = REQUIRED_RULE_FIELDS - rule.keys()
        if missing:
            raise RiskRuleError(
                f"第 {index} 条风险规则缺少字段: {sorted(missing)}"
            )
        if not isinstance(rule["rule_id"], str) or not rule["rule_id"]:
            raise RiskRuleError(f"第 {index} 条 rule_id 无效")
        if rule["memory_level"] != "L2_risk_rule":
            raise RiskRuleError(f"第 {index} 条 memory_level 必须为 L2_risk_rule")
        if not isinstance(rule["trigger_conditions"], dict):
            raise RiskRuleError(f"第 {index} 条 trigger_conditions 必须是对象")
        conditions = rule["trigger_conditions"]
        unknown = set(conditions) - SUPPORTED_TRIGGER_FIELDS
        if unknown:
            raise RiskRuleError(f"第 {index} 条触发条件不受支持: {sorted(unknown)}")
        if not conditions:
            raise RiskRuleError(f"第 {index} 条 trigger_conditions 不能为空")
        for name, value in conditions.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RiskRuleError(f"第 {index} 条 {name} 必须是数值")
            if name.endswith("observation_confidence") and not 0 <= value <= 1:
                raise RiskRuleError(f"第 {index} 条 {name} 必须在 0 到 1 之间")
        source = rule.get("source")
        if source is not None:
            if not isinstance(source, dict):
                raise RiskRuleError(f"第 {index} 条 source 必须是对象")
            if (
                source.get("kind") == "human_constructed"
                and source.get("historical_physical_run") is not False
            ):
                raise RiskRuleError(
                    f"第 {index} 条人工构造规则必须明确标注 historical_physical_run=false"
                )
        if not isinstance(rule["rule"], str) or not rule["rule"].strip():
            raise RiskRuleError(f"第 {index} 条 rule 不能为空")

        for field in (
            "recommended_actions",
            "prohibited_actions",
            "source_memory_ids",
        ):
            value = rule[field]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise RiskRuleError(f"第 {index} 条 {field} 必须是字符串数组")

        confidence = rule["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise RiskRuleError(f"第 {index} 条 confidence 必须在 0 到 1 之间")
        if rule["status"] not in {"active", "candidate", "retired"}:
            raise RiskRuleError(
                f"第 {index} 条 status 必须是 active、candidate 或 retired"
            )


def _match_trigger_conditions(
    scenario: dict[str, Any], conditions: dict[str, Any]
) -> list[str] | None:
    robot = scenario.get("robot", {})
    environment = scenario.get("environment", {})
    corridor_width = environment.get("corridor_width_m")
    robot_width = robot.get("width_m")
    clearance = None
    if isinstance(corridor_width, (int, float)) and isinstance(
        robot_width, (int, float)
    ):
        clearance = corridor_width - robot_width

    reasons: list[str] = []
    maximum_clearance = conditions.get("maximum_clearance_m")
    if maximum_clearance is not None:
        if clearance is None or clearance > maximum_clearance:
            return None
        reasons.append(
            f"通行总余量 {clearance:.2f}m 不超过 {maximum_clearance:.2f}m"
        )

    maximum_confidence = conditions.get("maximum_observation_confidence")
    if maximum_confidence is not None:
        confidence = environment.get("observation_confidence")
        if not isinstance(confidence, (int, float)) or confidence > maximum_confidence:
            return None
        reasons.append(
            f"观测置信度 {confidence:.2f} 不超过 {maximum_confidence:.2f}"
        )

    minimum_confidence = conditions.get("minimum_observation_confidence")
    if minimum_confidence is not None:
        confidence = environment.get("observation_confidence")
        if not isinstance(confidence, (int, float)) or confidence < minimum_confidence:
            return None
        reasons.append(
            f"观测置信度 {confidence:.2f} 不低于 {minimum_confidence:.2f}"
        )

    envelope_clearance = environment.get("minimum_envelope_clearance_m")
    maximum_envelope = conditions.get("maximum_minimum_envelope_clearance_m")
    if maximum_envelope is not None:
        if (
            not isinstance(envelope_clearance, (int, float))
            or isinstance(envelope_clearance, bool)
            or envelope_clearance > maximum_envelope
        ):
            return None
        reasons.append(
            f"最小包络间距 {envelope_clearance:.3f}m 不超过 {maximum_envelope:.3f}m"
        )

    minimum_envelope = conditions.get("minimum_minimum_envelope_clearance_m")
    if minimum_envelope is not None:
        if (
            not isinstance(envelope_clearance, (int, float))
            or isinstance(envelope_clearance, bool)
            or envelope_clearance < minimum_envelope
        ):
            return None
        reasons.append(
            f"最小包络间距 {envelope_clearance:.3f}m 不低于 {minimum_envelope:.3f}m"
        )

    return reasons


def resolve_rule_conflicts(
    matches: list[dict[str, Any]], available_actions: list[str]
) -> dict[str, Any]:
    """Resolve compatible rules or fail safely when recommendations conflict."""
    if not matches:
        return {
            "status": "no_rule_matched",
            "selected_action": None,
            "requires_human_review": False,
            "matched_rule_ids": [],
        }

    recommendation_sets = [
        set(rule.get("recommended_actions", [])) for rule in matches
    ]
    common_actions = set.intersection(*recommendation_sets)
    prohibited_actions = {
        action
        for rule in matches
        for action in rule.get("prohibited_actions", [])
    }
    usable_common = [
        action
        for action in available_actions
        if action in common_actions and action not in prohibited_actions
    ]
    matched_rule_ids = [rule["rule_id"] for rule in matches]

    if usable_common:
        return {
            "status": "resolved",
            "selected_action": usable_common[0],
            "requires_human_review": False,
            "matched_rule_ids": matched_rule_ids,
            "reason": "所有命中规则存在共同且未被禁止的建议动作",
        }

    fallback = next(
        (
            action
            for action in ("safe_stop", "ask_human", "observe_again")
            if action in available_actions
        ),
        None,
    )
    return {
        "status": "conflict_detected",
        "selected_action": fallback,
        "requires_human_review": True,
        "matched_rule_ids": matched_rule_ids,
        "conflicting_recommendations": {
            rule["rule_id"]: rule.get("recommended_actions", [])
            for rule in matches
        },
        "reason": "命中规则之间没有共同的安全建议动作，采用保守回退",
    }
