"""Task Advocate: proposes a feasible, risk-mitigated navigation action."""

from __future__ import annotations

from typing import Any

from llm_client import LLMError, ZhinaoClient


SYSTEM_PROMPT = """
你是室内移动机器人的 Task Advocate（任务支持 Agent）。
你的职责不是盲目赞成执行，而是在不忽略风险的前提下，寻找完成任务的可行方案。

你必须：
1. 只能从输入的 available_actions 中选择 proposed_action.name；
2. 清楚列出判断所依赖的 assumptions；
3. 为发现的风险提出具体 mitigations；
4. 信息不足或风险过高时，可以建议 observe_again、ask_human 或 safe_stop；
5. 不得虚构传感器数据、地图信息或历史经验；
6. 只输出一个 JSON 对象，不要输出 Markdown 或额外说明。
7. 保持简洁：assumptions、identified_risks、mitigations 最多各 3 项，每项只写一句话。
8. 只有真正使用了输入中的历史经验时，才把其 ID 写入 cited_experience_ids；不得编造 ID。
9. matched_l2_risk_rules 是由多条历史经验归纳出的规则；使用时必须写入 cited_rule_ids。

严格使用以下结构：
{
  "recommendation": "proceed_with_caution | revise_plan | observe_again | ask_human | safe_stop",
  "proposed_action": {
    "name": "必须来自 available_actions",
    "parameters": {}
  },
  "expected_benefit": "字符串",
  "assumptions": ["字符串"],
  "identified_risks": ["字符串"],
  "mitigations": ["字符串"],
  "cited_experience_ids": ["输入中真实存在的 experience_id"],
  "cited_rule_ids": ["输入中真实存在的 rule_id"],
  "confidence": 0.0
}

confidence 必须位于 0 到 1 之间。
""".strip()


REQUIRED_KEYS = {
    "recommendation",
    "proposed_action",
    "expected_benefit",
    "assumptions",
    "identified_risks",
    "mitigations",
    "cited_experience_ids",
    "cited_rule_ids",
    "confidence",
}


def analyze(
    scenario: dict[str, Any],
    client: ZhinaoClient,
    retrieved_experiences: list[dict[str, Any]] | None = None,
    matched_rules: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiences = retrieved_experiences or []
    rules = matched_rules or []
    report, usage = client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_data={
            "scenario": scenario,
            "retrieved_risk_memory_cards": experiences,
            "matched_l2_risk_rules": rules,
        },
        max_tokens=5000,
    )
    validate_report(report, scenario, experiences, rules)
    return report, usage


def validate_report(
    report: dict[str, Any],
    scenario: dict[str, Any],
    retrieved_experiences: list[dict[str, Any]] | None = None,
    matched_rules: list[dict[str, Any]] | None = None,
) -> None:
    missing = REQUIRED_KEYS - report.keys()
    if missing:
        raise LLMError(f"Task Advocate 输出缺少字段: {sorted(missing)}")

    action = report.get("proposed_action")
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise LLMError("proposed_action 必须包含字符串类型的 name")

    allowed = scenario.get("available_actions", [])
    if action["name"] not in allowed:
        raise LLMError(
            f"Task Advocate 输出了未授权动作 {action['name']!r}; 可用动作: {allowed}"
        )

    confidence = report.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise LLMError("confidence 必须是 0 到 1 之间的数字")

    for field in (
        "assumptions",
        "identified_risks",
        "mitigations",
        "cited_experience_ids",
        "cited_rule_ids",
    ):
        if not isinstance(report.get(field), list):
            raise LLMError(f"{field} 必须是数组")

    valid_ids = {
        item.get("experience_id")
        for item in (retrieved_experiences or [])
        if isinstance(item, dict)
    }
    invalid_ids = set(report["cited_experience_ids"]) - valid_ids
    if invalid_ids:
        raise LLMError(f"Task Advocate 引用了不存在的经验 ID: {sorted(invalid_ids)}")

    valid_rule_ids = {
        item.get("rule_id")
        for item in (matched_rules or [])
        if isinstance(item, dict)
    }
    invalid_rule_ids = set(report["cited_rule_ids"]) - valid_rule_ids
    if invalid_rule_ids:
        raise LLMError(f"Task Advocate 引用了不存在的规则 ID: {sorted(invalid_rule_ids)}")
