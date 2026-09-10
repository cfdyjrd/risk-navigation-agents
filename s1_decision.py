"""Condition-controlled three-Agent deliberation for the S1 experiment.

The module has no robot or DDS imports.  B0 disables both historical memories
and L2 rules, M performs role-conditioned retrieval, and B1 retrieves one
shared decision-role bundle then gives identical cards to all three Agents.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from typing import Any, Callable, Optional

from robot_interface import RobotInterfaceError


S1_CONDITIONS = frozenset({"B0", "M", "B1"})
S1_ROLES = ("advocate", "critic", "decision")


def _empty_bundle(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "retrieval_mode": "disabled_b0",
        "retrieval_version": "disabled",
        "token_budget": 0,
        "estimated_tokens_used": 0,
        "items": [],
    }


def retrieve_s1_evidence(
    condition: str,
    scenario: dict[str, Any],
    *,
    store=None,
    rule_store=None,
    top_k: int = 3,
    token_budget: int = 1800,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return exactly the evidence allowed by the registered condition."""
    if condition not in S1_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(S1_CONDITIONS)}")
    if condition == "B0":
        return {role: _empty_bundle(role) for role in S1_ROLES}, []
    if store is None or rule_store is None:
        raise ValueError(f"{condition} requires an experience and rule store")

    if condition == "M":
        bundles = {
            role: {
                **store.retrieve_memory_cards(
                    scenario, role=role, top_k=top_k, token_budget=token_budget
                ),
                "retrieval_mode": "role_conditioned_m",
            }
            for role in S1_ROLES
        }
    else:
        shared = store.retrieve_memory_cards(
            scenario, role="decision", top_k=top_k, token_budget=token_budget
        )
        bundle_id = hashlib.sha256(
            json.dumps(
                shared["items"], ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        bundles = {
            role: {
                **deepcopy(shared),
                "role": role,
                "retrieval_mode": "shared_b1",
                "shared_bundle_id": bundle_id,
                "retrieved_as_role": "decision",
            }
            for role in S1_ROLES
        }
    return bundles, rule_store.retrieve_matching(scenario)


def _conflict_execution(resolution: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": "safe_stop",
        "selected_action": {"name": "safe_stop", "parameters": {}},
        "reason": "L2 risk rules conflict; deterministic fail-closed stop",
        "decision_source": "deterministic_constrained_optimizer",
        "optimizer_status": "fallback_rule_conflict",
        "rule_resolution": deepcopy(resolution),
        "model_recommendation": None,
        "agrees_with_model": False,
    }


def make_s1_decider(
    condition: str,
    client,
    *,
    store=None,
    rule_store=None,
    top_k: int = 3,
    token_budget: int = 1800,
    audit_sink: Optional[Callable[[dict[str, Any]], None]] = None,
):
    """Build a fresh, uncached S1 decision callback.

    The callback deliberately does not use ``run_three_agents.py``'s report
    cache because each physical observation and each trial must remain an
    independently auditable decision.
    """
    if condition not in S1_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(S1_CONDITIONS)}")

    from risk_rule_store import resolve_rule_conflicts
    from run_three_agents import (
        decide,
        run_advocate,
        run_critic,
        run_post_deliberation_optimizer,
    )

    def deliberate(scenario: dict[str, Any]) -> dict[str, Any]:
        bundles, matched_rules = retrieve_s1_evidence(
            condition,
            scenario,
            store=store,
            rule_store=rule_store,
            top_k=top_k,
            token_budget=token_budget,
        )
        resolution = resolve_rule_conflicts(
            matched_rules, scenario.get("available_actions", [])
        )
        base_audit = {
            "condition": condition,
            "scenario": deepcopy(scenario),
            "memories": deepcopy(bundles),
            "rules": deepcopy(matched_rules),
            "rule_resolution": deepcopy(resolution),
        }
        if resolution["status"] == "conflict_detected":
            execution = _conflict_execution(resolution)
            if audit_sink is not None:
                audit_sink({
                    **base_audit,
                    "agents_skipped": True,
                    "skip_reason": "conflicting_l2_rules",
                    "execution_decision": deepcopy(execution),
                    "usage": {},
                })
            return execution

        try:
            # Advocate and Critic are intentionally independent.  Running the
            # two I/O-bound API calls together reduces stationary wait time at
            # D without sharing either role's report with the other.
            with ThreadPoolExecutor(max_workers=2) as pool:
                advocate_future = pool.submit(
                    run_advocate,
                    scenario,
                    client,
                    bundles["advocate"]["items"],
                    matched_rules,
                )
                critic_future = pool.submit(
                    run_critic,
                    scenario,
                    client,
                    bundles["critic"]["items"],
                    matched_rules,
                )
                advocate, advocate_usage = advocate_future.result()
                critic, critic_usage = critic_future.result()
            model, decision_usage = decide(
                scenario,
                advocate,
                critic,
                client,
                bundles["decision"]["items"],
                matched_rules,
            )
            optimizer, execution = run_post_deliberation_optimizer(
                scenario,
                bundles["decision"]["items"],
                matched_rules,
                advocate,
                critic,
                model,
            )
        except Exception as exc:
            raise RobotInterfaceError(f"S1 three-Agent deliberation failed: {exc}") from exc

        if audit_sink is not None:
            audit_sink({
                **base_audit,
                "agents_skipped": False,
                "advocate": deepcopy(advocate),
                "critic": deepcopy(critic),
                "decision": deepcopy(model),
                "optimizer": deepcopy(optimizer),
                "execution_decision": deepcopy(execution),
                "usage": {
                    "advocate": deepcopy(advocate_usage),
                    "critic": deepcopy(critic_usage),
                    "decision": deepcopy(decision_usage),
                },
            })
        return execution

    return deliberate
