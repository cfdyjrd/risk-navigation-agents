"""Offline tests for registered S1 evidence conditions."""

import unittest
from unittest.mock import Mock, patch

from s1_decision import make_s1_decider, retrieve_s1_evidence


SCENARIO = {
    "task": {"goal": "pass"},
    "robot": {"type": "g1", "width_m": .6, "battery_percent": 80},
    "environment": {
        "corridor_width_m": .9,
        "obstacle_detected": False,
        "obstacle_distance_m": None,
        "observation_confidence": .9,
    },
    "available_actions": ["move_forward", "slow_down", "observe_again", "safe_stop"],
}


class S1DecisionTests(unittest.TestCase):
    def store(self):
        store = Mock()
        store.retrieve_memory_cards.side_effect = lambda scenario, role, **kwargs: {
            "role": role,
            "retrieval_version": "test",
            "token_budget": kwargs["token_budget"],
            "estimated_tokens_used": 1,
            "items": [{"experience_id": f"for_{role}"}],
        }
        return store

    def test_b0_does_not_touch_memory_or_rules(self):
        store, rules = self.store(), Mock()
        bundles, matched = retrieve_s1_evidence(
            "B0", SCENARIO, store=store, rule_store=rules
        )
        self.assertEqual(matched, [])
        self.assertTrue(all(not bundle["items"] for bundle in bundles.values()))
        store.retrieve_memory_cards.assert_not_called()
        rules.retrieve_matching.assert_not_called()

    def test_m_retrieves_independently_for_each_role(self):
        store, rules = self.store(), Mock()
        rules.retrieve_matching.return_value = []
        bundles, _matched = retrieve_s1_evidence(
            "M", SCENARIO, store=store, rule_store=rules
        )
        self.assertEqual(store.retrieve_memory_cards.call_count, 3)
        self.assertEqual(
            [bundles[role]["items"][0]["experience_id"] for role in bundles],
            ["for_advocate", "for_critic", "for_decision"],
        )

    def test_b1_retrieves_once_and_shares_identical_cards(self):
        store, rules = self.store(), Mock()
        rules.retrieve_matching.return_value = []
        bundles, _matched = retrieve_s1_evidence(
            "B1", SCENARIO, store=store, rule_store=rules
        )
        store.retrieve_memory_cards.assert_called_once()
        items = [bundles[role]["items"] for role in ("advocate", "critic", "decision")]
        self.assertEqual(items[0], items[1])
        self.assertEqual(items[1], items[2])
        self.assertEqual(len({bundles[role]["shared_bundle_id"] for role in bundles}), 1)

    def test_rule_conflict_skips_all_models_and_returns_optimizer_stop(self):
        store, rules = self.store(), Mock()
        rules.retrieve_matching.return_value = [
            {"rule_id": "a", "recommended_actions": ["observe_again"], "prohibited_actions": []},
            {"rule_id": "b", "recommended_actions": ["slow_down"], "prohibited_actions": []},
        ]
        audit = Mock()
        with patch("run_three_agents.run_advocate") as advocate, \
             patch("run_three_agents.run_critic") as critic, \
             patch("run_three_agents.decide") as decision:
            decider = make_s1_decider(
                "M", Mock(), store=store, rule_store=rules, audit_sink=audit
            )
            result = decider(SCENARIO)
        advocate.assert_not_called()
        critic.assert_not_called()
        decision.assert_not_called()
        self.assertEqual(result["selected_action"]["name"], "safe_stop")
        self.assertEqual(result["decision_source"], "deterministic_constrained_optimizer")
        self.assertTrue(audit.call_args.args[0]["agents_skipped"])


if __name__ == "__main__":
    unittest.main()
