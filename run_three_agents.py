"""Run the first end-to-end three-agent deliberation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Callable, Optional

from agents.advocate import analyze as run_advocate
from agents.critic import analyze as run_critic
from agents.decision import decide
from decision_optimizer import optimize_action
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


def load_or_run_agent(
    cache_dir: Path,
    role: str,
    client: Optional[ZhinaoClient],
    invoke: Callable[[ZhinaoClient], tuple[dict, dict]],
) -> tuple[dict, dict, Optional[ZhinaoClient]]:
    """Load a cached report before requiring API credentials.

    The client is created lazily only on a cache miss. This allows fully
    cached, reproducible runs without exposing or reloading an API key.
    """
    cached = load_cached(cache_dir, role)
    if cached:
        report, usage = cached
        return report, usage, client
    active_client = client or ZhinaoClient()
    report, usage = invoke(active_client)
    save_cached(cache_dir, role, report, usage)
    return report, usage, active_client


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


def build_optimizer_execution_decision(
    optimizer_result: dict, model_decision: dict
) -> dict:
    """Adapt a deterministic optimizer result for the final Safety Guard."""
    optimized_action = optimizer_result["selected_action"]
    optimized_name = optimized_action.get("name", "")
    model_action = model_decision.get("selected_action", {})
    model_name = model_action.get("name")
    if optimized_name in {"observe_again", "ask_human", "safe_stop"}:
        decision_label = optimized_name
    else:
        decision_label = "execute"
    return {
        "decision": decision_label,
        "selected_action": optimized_action,
        "reason": optimizer_result["reason"],
        "decision_source": "deterministic_constrained_optimizer",
        "optimizer_status": optimizer_result["status"],
        "model_recommendation": model_action,
        "agrees_with_model": model_name == optimized_name,
    }


def run_post_deliberation_optimizer(
    scenario: dict,
    decision_memories: list[dict],
    matched_rules: list[dict],
    advocate_report: dict,
    critic_report: dict,
    model_decision: dict,
) -> tuple[dict, dict]:
    """Run the auditable deterministic stage after all three Agent reports."""
    agent_reports = {
        "advocate": advocate_report,
        "critic": critic_report,
        "decision": model_decision,
    }
    optimizer_result = optimize_action(
        scenario,
        decision_memories,
        matched_rules,
        agent_reports,
    )
    model_action = model_decision.get("selected_action", {})
    optimized_action = optimizer_result.get("selected_action", {})
    optimizer_result["model_recommendation"] = model_action
    optimizer_result["optimized_action"] = optimized_action
    optimizer_result["agrees_with_model"] = (
        model_action.get("name") == optimized_action.get("name")
    )
    execution_decision = build_optimizer_execution_decision(
        optimizer_result, model_decision
    )
    return optimizer_result, execution_decision


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
        client: Optional[ZhinaoClient] = None
        advocate_report, advocate_usage, client = load_or_run_agent(
            cache_dir,
            "advocate",
            client,
            lambda active_client: run_advocate(
                scenario,
                active_client,
                evidence_bundles["advocate"]["items"],
                matched_rules,
            ),
        )

        critic_report, critic_usage, client = load_or_run_agent(
            cache_dir,
            "critic",
            client,
            lambda active_client: run_critic(
                scenario,
                active_client,
                evidence_bundles["critic"]["items"],
                matched_rules,
            ),
        )

        final_decision, decision_usage, client = load_or_run_agent(
            cache_dir,
            "decision",
            client,
            lambda active_client: decide(
                scenario,
                advocate_report,
                critic_report,
                active_client,
                evidence_bundles["decision"]["items"],
                matched_rules,
            ),
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
    optimizer_result, execution_decision = run_post_deliberation_optimizer(
        scenario,
        evidence_bundles["decision"]["items"],
        matched_rules,
        advocate_report,
        critic_report,
        final_decision,
    )
    print_section("Deterministic Constrained Optimizer", optimizer_result)
    print_section("Execution Decision", execution_decision)
    guarded_action = apply_safety_guard(scenario, execution_decision)
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
