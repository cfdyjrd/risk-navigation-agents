"""Safety Decision Agent: arbitrates the advocate and critic reports."""

from __future__ import annotations

from typing import Any

from llm_client import LLMError, ZhinaoClient


SYSTEM_PROMPT = """
你是室内移动机器人的 Safety Decision Agent（安全决策主 Agent）。
你将收到原始场景、Task Advocate 报告和 Risk Critic 报告。
你的职责是基于可追溯证据作出最终裁决，而不是简单投票或平均双方意见。

决策原则：
1. 优先使用原始 scenario 中的事实；两份报告中的推测不能覆盖原始事实；
2. 明确区分“已观察事实”“Agent 推断”和“建议阈值”；
3. 未解决的高严重度风险不得被一句笼统理由忽略；
4. 信息不足且可以通过感知补充时，优先选择 observe_again；
5. 风险可以通过调整方案缓解时，选择 revise_plan；
6. 只有证据足够且风险处于可接受范围时，才选择 execute；
7. 输出动作只能来自 scenario.available_actions；
8. 只输出一个 JSON 对象，不要输出 Markdown 或额外说明。
9. 保持简洁：supporting_evidence 最多 3 项，其余数组最多各 3 项；每项只写一句话。
10. 只有真正使用了输入中的历史经验时，才把其 ID 写入 cited_experience_ids；不得编造 ID。

decision 只能是：
execute、revise_plan、observe_again、ask_human、reject、safe_stop。

严格使用以下结构：
{
  "decision": "六种决策之一",
  "selected_action": {
    "name": "必须来自 available_actions",
    "parameters": {}
  },
  "reason": "字符串",
  "supporting_evidence": [
    {
      "source": "scenario | experience | advocate | critic",
      "evidence": "字符串"
    }
  ],
  "accepted_risks": ["字符串"],
  "unresolved_risks": ["字符串"],
  "required_information": ["字符串"],
  "cited_experience_ids": ["输入中真实存在的 experience_id"],
  "confidence": 0.0
}

confidence 必须位于 0 到 1 之间。
""".strip()


ALLOWED_DECISIONS = {
    "execute",
    "revise_plan",
    "observe_again",
    "ask_human",
    "reject",
    "safe_stop",
}
REQUIRED_KEYS = {
    "decision",
    "selected_action",
    "reason",
    "supporting_evidence",
    "accepted_risks",
    "unresolved_risks",
    "required_information",
    "cited_experience_ids",
    "confidence",
}


def decide(
    scenario: dict[str, Any],
    advocate_report: dict[str, Any],
    critic_report: dict[str, Any],
    client: ZhinaoClient,
    retrieved_experiences: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiences = retrieved_experiences or []
    report, usage = client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_data={
            "scenario": scenario,
            "retrieved_risk_experiences": experiences,
            "task_advocate_report": advocate_report,
            "risk_critic_report": critic_report,
        },
        # GLM-5.1 may use a large portion of the budget for reasoning before
        # emitting the visible JSON response.
        max_tokens=5000,
    )
    validate_decision(report, scenario, experiences)
    return report, usage


def validate_decision(
    report: dict[str, Any],
    scenario: dict[str, Any],
    retrieved_experiences: list[dict[str, Any]] | None = None,
) -> None:
    missing = REQUIRED_KEYS - report.keys()
    if missing:
        raise LLMError(f"Safety Decision Agent 输出缺少字段: {sorted(missing)}")

    if report.get("decision") not in ALLOWED_DECISIONS:
        raise LLMError(f"decision 必须属于 {sorted(ALLOWED_DECISIONS)}")

    action = report.get("selected_action")
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise LLMError("selected_action 必须包含字符串类型的 name")
    allowed_actions = scenario.get("available_actions", [])
    if action["name"] not in allowed_actions:
        raise LLMError(
            f"Safety Decision Agent 输出未授权动作 {action['name']!r}; 可用动作: {allowed_actions}"
        )

    confidence = report.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise LLMError("confidence 必须是 0 到 1 之间的数字")

    for field in (
        "supporting_evidence",
        "accepted_risks",
        "unresolved_risks",
        "required_information",
        "cited_experience_ids",
    ):
        if not isinstance(report.get(field), list):
            raise LLMError(f"{field} 必须是数组")

    for index, item in enumerate(report["supporting_evidence"]):
        if not isinstance(item, dict):
            raise LLMError(f"supporting_evidence[{index}] 必须是对象")
        if item.get("source") not in {
            "scenario",
            "experience",
            "advocate",
            "critic",
        }:
            raise LLMError(
                f"supporting_evidence[{index}].source 必须是 scenario、experience、advocate 或 critic"
            )
        if not isinstance(item.get("evidence"), str):
            raise LLMError(f"supporting_evidence[{index}].evidence 必须是字符串")

    valid_ids = {
        item.get("experience_id")
        for item in (retrieved_experiences or [])
        if isinstance(item, dict)
    }
    invalid_ids = set(report["cited_experience_ids"]) - valid_ids
    if invalid_ids:
        raise LLMError(
            f"Safety Decision Agent 引用了不存在的经验 ID: {sorted(invalid_ids)}"
        )
