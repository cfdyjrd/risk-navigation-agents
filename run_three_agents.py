"""Run the first end-to-end three-agent deliberation."""

import json
import hashlib
from pathlib import Path

from agents.advocate import analyze as run_advocate
from agents.critic import analyze as run_critic
from agents.decision import decide
from experience_store import ExperienceStore
from llm_client import LLMError, ZhinaoClient
from risk_rule_store import RiskRuleStore, resolve_rule_conflicts
from safety_guard import apply_safety_guard


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
PIPELINE_VERSION = "layered-risk-memory-v4"
ROLE_TOKEN_BUDGET = 1800


def print_section(title: str, value: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def pipeline_cache_dir(scenario: dict, experiences: dict, rules: list[dict]) -> Path:
    cache_input = {
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario,
        "retrieved_experiences": experiences,
        "matched_risk_rules": rules,
    }
    encoded = json.dumps(
        cache_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return CACHE_ROOT / digest


def load_cached(cache_dir: Path, role: str):
    path = cache_dir / f"{role}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    print(f"使用缓存：{role}")
    return saved["report"], saved.get("usage", {})


def save_cached(cache_dir: Path, role: str, report: dict, usage: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{role}.json"
    path.write_text(
        json.dumps(
            {"report": report, "usage": usage},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_rule_conflict_decision(rule_resolution: dict) -> dict:
    """Create a deterministic safe decision without asking an LLM to arbitrate."""
    fallback = rule_resolution.get("selected_action") or "safe_stop"
    rule_ids = list(rule_resolution.get("matched_rule_ids", []))
    return {
        "decision": "safe_stop",
        "selected_action": {"name": fallback, "parameters": {}},
        "reason": "L2风险规则给出互不兼容的建议，确定性安全闸门要求保守停止并人工复核。",
        "supporting_evidence": [
            {
                "source": "rule",
                "evidence": f"冲突规则: {', '.join(rule_ids)}",
            }
        ],
        "accepted_risks": [],
        "unresolved_risks": ["命中的L2风险规则之间存在决策冲突。"],
        "required_information": ["需要人工确认应保留、修订或停用哪条规则。"],
        "cited_experience_ids": [],
        "cited_rule_ids": rule_ids,
        "confidence": 1.0,
        "decision_source": "deterministic_rule_conflict_gate",
    }


def main() -> int:
    scenario_path = ROOT / "scenarios" / "corridor_obstacle.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    store = ExperienceStore(ROOT / "experiences" / "risk_experiences.json")
    evidence_bundles = {
        role: store.retrieve_memory_cards(
            scenario,
            role=role,
            top_k=3,
            token_budget=ROLE_TOKEN_BUDGET,
        )
        for role in ("advocate", "critic", "decision")
    }
    rule_store = RiskRuleStore(ROOT / "experiences" / "risk_rules.json")
    matched_rules = rule_store.retrieve_matching(scenario)
    rule_resolution = resolve_rule_conflicts(
        matched_rules, scenario.get("available_actions", [])
    )
    cache_dir = pipeline_cache_dir(scenario, evidence_bundles, matched_rules)

    if rule_resolution["status"] == "conflict_detected":
        final_decision = build_rule_conflict_decision(rule_resolution)
        print_section("Role-specific Risk Memory", evidence_bundles)
        print_section("Matched L2 Risk Rules", {"items": matched_rules})
        print_section("L2 Rule Resolution", rule_resolution)
        print_section("Safety Decision", final_decision)
        print_section(
            "Safety Guard", apply_safety_guard(scenario, final_decision)
        )
        print("\nL2规则冲突：已跳过三个Agent及API调用。")
        return 0

    try:
        client = ZhinaoClient()
        advocate_cached = load_cached(cache_dir, "advocate")
        if advocate_cached:
            advocate_report, advocate_usage = advocate_cached
        else:
            advocate_report, advocate_usage = run_advocate(
                scenario,
                client,
                evidence_bundles["advocate"]["items"],
                matched_rules,
            )
            save_cached(
                cache_dir, "advocate", advocate_report, advocate_usage
            )

        critic_cached = load_cached(cache_dir, "critic")
        if critic_cached:
            critic_report, critic_usage = critic_cached
        else:
            critic_report, critic_usage = run_critic(
                scenario,
                client,
                evidence_bundles["critic"]["items"],
                matched_rules,
            )
            save_cached(cache_dir, "critic", critic_report, critic_usage)

        decision_cached = load_cached(cache_dir, "decision")
        if decision_cached:
            final_decision, decision_usage = decision_cached
        else:
            final_decision, decision_usage = decide(
                scenario,
                advocate_report,
                critic_report,
                client,
                evidence_bundles["decision"]["items"],
                matched_rules,
            )
            save_cached(
                cache_dir, "decision", final_decision, decision_usage
            )
    except LLMError as exc:
        print(f"三 Agent 流程失败，机器人不得执行动作：{exc}")
        return 1

    print_section("Role-specific Risk Memory", evidence_bundles)
    print_section("Matched L2 Risk Rules", {"items": matched_rules})
    print_section("L2 Rule Resolution", rule_resolution)
    print_section("Task Advocate", advocate_report)
    print_section("Risk Critic", critic_report)
    print_section("Safety Decision", final_decision)
    guarded_action = apply_safety_guard(scenario, final_decision)
    print_section("Safety Guard", guarded_action)

    usages = {
        "advocate": advocate_usage,
        "critic": critic_usage,
        "decision": decision_usage,
    }
    total_tokens = sum(int(item.get("total_tokens", 0)) for item in usages.values())
    print_section("Token Usage", {"by_agent": usages, "total_tokens": total_tokens})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
