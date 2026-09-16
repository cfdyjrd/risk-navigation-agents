"""Single Agent 对照组：单次调用直接选动作，无任何架构设计。

与三 Agent 流程的唯一公平差异是架构本身：输入同样的场景观测，
但没有支持/反对分工、没有经验检索、没有 Safety Guard 硬规则，
模型输出的动作直接执行。
"""

from __future__ import annotations

from typing import Any

from llm_client import LLMError, ZhinaoClient


SYSTEM_PROMPT = """
你是室内移动机器人的控制 Agent。输入是当前场景的 JSON。
请为机器人选择下一步动作。

只输出一个 JSON 对象，不要输出 Markdown 或额外说明：
{
  "action": {
    "name": "必须来自 available_actions",
    "parameters": {}
  },
  "reason": "一句话"
}
""".strip()


def act(
    scenario: dict[str, Any],
    client: ZhinaoClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, usage = client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_data={"scenario": scenario},
        max_tokens=5000,
    )
    validate(report, scenario)
    return report, usage


def validate(report: dict[str, Any], scenario: dict[str, Any]) -> None:
    action = report.get("action")
    if not isinstance(action, dict) or not isinstance(action.get("name"), str):
        raise LLMError("Single Agent 输出必须包含 action.name 字符串")
    allowed = scenario.get("available_actions", [])
    if action["name"] not in allowed:
        raise LLMError(
            f"Single Agent 输出了未授权动作 {action['name']!r}; 可用动作: {allowed}"
        )
