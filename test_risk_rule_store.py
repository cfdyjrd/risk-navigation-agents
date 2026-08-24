"""Offline tests for L2 risk-rule loading and validation."""

import json
import tempfile
import unittest
from pathlib import Path

from risk_rule_store import RiskRuleError, RiskRuleStore, resolve_rule_conflicts


ROOT = Path(__file__).resolve().parent


class RiskRuleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RiskRuleStore(ROOT / "experiences" / "risk_rules.json")
        self.scenario = json.loads(
            (ROOT / "scenarios" / "corridor_obstacle.json").read_text(
                encoding="utf-8"
            )
        )

    def test_loads_valid_l2_rule(self) -> None:
        rules = self.store.load()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["rule_id"], "rule_narrow_corridor_001")
        self.assertEqual(rules[0]["memory_level"], "L2_risk_rule")
        self.assertEqual(
            rules[0]["source_memory_ids"],
            [
                "rmc_exp_narrow_collision_001",
                "rmc_exp_narrow_success_002",
            ],
        )

    def test_matches_current_narrow_low_confidence_scenario(self) -> None:
        matches = self.store.retrieve_matching(self.scenario)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["rule_id"], "rule_narrow_corridor_001")
        self.assertEqual(matches[0]["recommended_actions"], ["observe_again"])
        self.assertEqual(matches[0]["prohibited_actions"], ["move_forward"])
        self.assertEqual(len(matches[0]["match_reasons"]), 2)

    def test_does_not_match_wide_high_confidence_scenario(self) -> None:
        wide_scenario = json.loads(
            (ROOT / "scenarios" / "wide_corridor_clear.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(self.store.retrieve_matching(wide_scenario), [])

    def test_conflict_fixture_matches_two_different_recommendations(self) -> None:
        fixture_store = RiskRuleStore(
            ROOT / "experiences" / "risk_rules_conflict_fixture.json"
        )
        matches = fixture_store.retrieve_matching(self.scenario)
        self.assertEqual(len(matches), 2)
        recommendations = {
            action
            for rule in matches
            for action in rule["recommended_actions"]
        }
        self.assertEqual(recommendations, {"observe_again", "slow_down"})
        resolution = resolve_rule_conflicts(
            matches, self.scenario["available_actions"]
        )
        self.assertEqual(resolution["status"], "conflict_detected")
        self.assertEqual(resolution["selected_action"], "safe_stop")
        self.assertTrue(resolution["requires_human_review"])

    def test_single_matching_rule_is_resolved(self) -> None:
        matches = self.store.retrieve_matching(self.scenario)
        resolution = resolve_rule_conflicts(
            matches, self.scenario["available_actions"]
        )
        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(resolution["selected_action"], "observe_again")
        self.assertFalse(resolution["requires_human_review"])

    def test_no_matching_rule_needs_no_resolution(self) -> None:
        resolution = resolve_rule_conflicts([], self.scenario["available_actions"])
        self.assertEqual(resolution["status"], "no_rule_matched")
        self.assertIsNone(resolution["selected_action"])

    def test_rejects_invalid_confidence(self) -> None:
        source = json.loads(
            (ROOT / "experiences" / "risk_rules.json").read_text(
                encoding="utf-8"
            )
        )
        source["rules"][0]["confidence"] = 1.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "risk_rules.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(RiskRuleError):
                RiskRuleStore(path).load()


if __name__ == "__main__":
    unittest.main(verbosity=2)
