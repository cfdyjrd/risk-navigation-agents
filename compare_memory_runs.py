"""Compare an L1-only run with an L1+L2 layered-memory run offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASELINE_PATH = RESULTS / "baseline_l1_memory.txt"
L2_PATH = RESULTS / "l2_rule_integration.txt"
REPORT_PATH = RESULTS / "l1_vs_l2_comparison.md"

SECTIONS = (
    "Task Advocate",
    "Risk Critic",
    "Safety Decision",
    "Safety Guard",
    "Token Usage",
)


def extract_section(text: str, title: str) -> dict[str, Any]:
    marker = f"=== {title} ==="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"找不到输出区段: {title}")
    json_start = text.find("{", start + len(marker))
    if json_start < 0:
        raise ValueError(f"区段 {title} 后没有 JSON 对象")
    value, _ = json.JSONDecoder().raw_decode(text[json_start:])
    if not isinstance(value, dict):
        raise ValueError(f"区段 {title} 必须是 JSON 对象")
    return value


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return {title: extract_section(text, title) for title in SECTIONS}


def count_items(report: dict[str, Any], field: str) -> int:
    value = report.get(field, [])
    return len(value) if isinstance(value, list) else 0


def fmt_delta(before: float, after: float, digits: int = 2) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{digits}f}"


def build_report(
    baseline: dict[str, dict[str, Any]],
    layered: dict[str, dict[str, Any]],
) -> str:
    base_advocate = baseline["Task Advocate"]
    l2_advocate = layered["Task Advocate"]
    base_critic = baseline["Risk Critic"]
    l2_critic = layered["Risk Critic"]
    base_decision = baseline["Safety Decision"]
    l2_decision = layered["Safety Decision"]
    base_tokens = int(baseline["Token Usage"].get("total_tokens", 0))
    l2_tokens = int(layered["Token Usage"].get("total_tokens", 0))
    token_delta = l2_tokens - base_tokens
    token_percent = (token_delta / base_tokens * 100) if base_tokens else 0.0

    rows = []
    for name, before, after in (
        ("Task Advocate", base_advocate, l2_advocate),
        ("Risk Critic", base_critic, l2_critic),
        ("Safety Decision", base_decision, l2_decision),
    ):
        before_confidence = float(before.get("confidence", 0))
        after_confidence = float(after.get("confidence", 0))
        rows.append(
            "| {name} | {before_confidence:.2f} | {after_confidence:.2f} "
            "| {delta} | {before_exp} | {after_exp} | {rule_count} |".format(
                name=name,
                before_confidence=before_confidence,
                after_confidence=after_confidence,
                delta=fmt_delta(before_confidence, after_confidence),
                before_exp=count_items(before, "cited_experience_ids"),
                after_exp=count_items(after, "cited_experience_ids"),
                rule_count=count_items(after, "cited_rule_ids"),
            )
        )

    rule_id = "未引用"
    cited_rules = l2_decision.get("cited_rule_ids", [])
    if isinstance(cited_rules, list) and cited_rules:
        rule_id = ", ".join(str(item) for item in cited_rules)

    return "\n".join(
        [
            "# L1 与 L1+L2 风险记忆 A/B 对比",
            "",
            "## 实验设置",
            "",
            "- A组：仅使用L1 Risk Memory Card。",
            "- B组：使用L1 Risk Memory Card，并加入匹配的L2 Risk Rule。",
            "- 两组使用相同的窄走廊场景。",
            "",
            "## 结果概览",
            "",
            f"- A组最终决策：`{base_decision.get('decision')}`。",
            f"- B组最终决策：`{l2_decision.get('decision')}`。",
            f"- B组主Agent引用规则：`{rule_id}`。",
            f"- A组Token总量：{base_tokens}。",
            f"- B组Token总量：{l2_tokens}。",
            f"- Token变化：{token_delta:+d}（{token_percent:+.2f}%）。",
            "",
            "| Agent | L1置信度 | L1+L2置信度 | 变化 | L1经验引用数 | L1+L2经验引用数 | L2规则引用数 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 可以得到的结论",
            "",
            "1. 两组都选择 `observe_again`，因此本次实验不能证明L2改变了最终动作。",
            "2. L2组提供了显式规则级证据：主Agent能够说明触发条件、禁止动作和规则来源，决策可追溯性更强。",
            "3. L2组用一条规则概括多条L1经验，说明分层记忆可以把具体案例提升为可复用知识。",
            "4. 加入L2证据带来了额外Token开销，应在后续实验中继续控制规则数量和长度。",
            "",
            "## 实验局限",
            "",
            "- 当前只有一个场景、各运行一次，且语言模型输出具有随机性。",
            "- 置信度变化不能单独归因于L2规则；需要多场景、多次重复实验。",
            "- 下一阶段应加入规则命中、规则不命中和规则冲突三类场景，统计动作正确率、危险动作率、规则引用率与Token成本。",
            "",
        ]
    )


def main() -> None:
    for path in (BASELINE_PATH, L2_PATH):
        if not path.exists():
            raise SystemExit(f"缺少实验输出文件: {path}")
    report = build_report(load_run(BASELINE_PATH), load_run(L2_PATH))
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"对比报告已生成: {REPORT_PATH}")


if __name__ == "__main__":
    main()
