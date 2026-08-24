"""Offline tests for the deterministic L2 rule-conflict gate."""

import json
import unittest
from pathlib import Path

from risk_rule_store import RiskRuleStore, resolve_rule_conflicts
from run_three_agents import build_rule_conflict_decision
from safety_guard import apply_safety_guard


ROOT = Path(__file__).resolve().parent


class RuleConflictGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = json.loads(
            (ROOT / "scenarios" / "corridor_obstacle.json").read_text(
                encoding="utf-8"
            )
        )

    def test_conflict_bypasses_llm_with_safe_stop_decision(self) -> None:
        matches = RiskRuleStore(
            ROOT / "experiences" / "risk_rules_conflict_fixture.json"
        ).retrieve_matching(self.scenario)
        resolution = resolve_rule_conflicts(
            matches, self.scenario["available_actions"]
        )
        decision = build_rule_conflict_decision(resolution)
        self.assertEqual(resolution["status"], "conflict_detected")
        self.assertEqual(decision["decision"], "safe_stop")
        self.assertEqual(decision["selected_action"]["name"], "safe_stop")
        self.assertEqual(
            decision["decision_source"],
            "deterministic_rule_conflict_gate",
        )
        self.assertEqual(len(decision["cited_rule_ids"]), 2)

        guarded = apply_safety_guard(self.scenario, decision)
        self.assertEqual(guarded["approved_action"]["name"], "safe_stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
