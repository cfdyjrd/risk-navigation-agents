"""Risk Critic: searches for failure modes and stop conditions."""

from __future__ import annotations

from typing import Any

from llm_client import LLMError, ZhinaoClient


SYSTEM_PROMPT = """
你是室内移动机器人的 Risk Critic（风险反对 Agent）。
你的任务是独立审查当前导航场景，主动寻找可能导致碰撞、卡死、失控、任务失败或人工介入的因素。
你不能读取或猜测 Task Advocate 的报告，也不能为了显得安全而无条件拒绝所有任务。

你必须：
1. 对每个风险分别给出 probability 和 severity；
2. probability 必须位于 0 到 1 之间；severity 必须是 1 到 5 的整数；
3. evidence 只能引用输入中明确存在的数据，不能虚构观测；
4. 明确列出 missing_information 和 stop_conditions；
5. recommendation 只能从输入的 available_actions 中选择；
6. 风险可缓解时应提出 mitigation，而不是直接否决；
7. 只输出一个 JSON 对象，不要输出 Markdown 或额外说明。
8. 保持简洁：identified_risks 最多 3 项；每个 evidence 最多 3 项；
   missing_information 和 stop_conditions 最多各 3 项；每项只写一句话。
9. 只有真正使用了输入中的历史经验时，才把其 ID 写入 cited_experience_ids；不得编造 ID。

严格使用以下结构：
{
  "overall_risk": "low | medium | high | critical",
  "identified_risks": [
    {
      "name": "英文或下划线风险标识",
      "description": "字符串",
      "probability": 0.0,
      "severity": 1,
      "evidence": ["字符串"],
      "mitigation": "字符串"
    }
  ],
  "missing_information": ["字符串"],
  "stop_conditions": ["字符串"],
  "recommendation": "必须来自 available_actions",
  "cited_experience_ids": ["输入中真实存在的 experience_id"],
  "confidence": 0.0
}
""".strip()


REQUIRED_KEYS = {
    "overall_risk",
    "identified_risks",
    "missing_information",
    "stop_conditions",
    "recommendation",
    "cited_experience_ids",
    "confidence",
}
ALLOWED_RISK_LEVELS = {"low", "medium", "high", "critical"}


def analyze(
    scenario: dict[str, Any],
    client: ZhinaoClient,
    retrieved_experiences: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiences = retrieved_experiences or []
    report, usage = client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_data={"scenario": scenario, "retrieved_risk_experiences": experiences},
        max_tokens=5000,
    )
    validate_report(report, scenario, experiences)
    return report, usage


def validate_report(
    report: dict[str, Any],
    scenario: dict[str, Any],
    retrieved_experiences: list[dict[str, Any]] | None = None,
) -> None:
    missing = REQUIRED_KEYS - report.keys()
    if missing:
        raise LLMError(f"Risk Critic 输出缺少字段: {sorted(missing)}")

    if report.get("overall_risk") not in ALLOWED_RISK_LEVELS:
        raise LLMError("overall_risk 必须是 low、medium、high 或 critical")

    recommendation = report.get("recommendation")
    allowed_actions = scenario.get("available_actions", [])
    if recommendation not in allowed_actions:
        raise LLMError(
            f"Risk Critic 输出了未授权动作 {recommendation!r}; 可用动作: {allowed_actions}"
        )

    confidence = report.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise LLMError("confidence 必须是 0 到 1 之间的数字")

    for field in (
        "identified_risks",
        "missing_information",
        "stop_conditions",
        "cited_experience_ids",
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
        raise LLMError(f"Risk Critic 引用了不存在的经验 ID: {sorted(invalid_ids)}")

    for index, risk in enumerate(report["identified_risks"]):
        if not isinstance(risk, dict):
            raise LLMError(f"identified_risks[{index}] 必须是对象")
        probability = risk.get("probability")
        severity = risk.get("severity")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise LLMError(f"identified_risks[{index}].probability 必须在 0 到 1 之间")
        if not isinstance(severity, int) or not 1 <= severity <= 5:
            raise LLMError(f"identified_risks[{index}].severity 必须是 1 到 5 的整数")
        for key in ("name", "description", "evidence", "mitigation"):
            if key not in risk:
                raise LLMError(f"identified_risks[{index}] 缺少字段 {key}")
