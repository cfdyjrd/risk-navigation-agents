"""Offline tests for constrained action feasibility filtering."""

import copy
import json
import unittest
from pathlib import Path

from decision_optimizer import (
    compute_task_utility,
    estimate_action_risk,
    estimate_action_uncertainty,
    filter_safe_actions,
    optimize_action,
    score_action_risks,
    score_action_uncertainties,
    score_task_utilities,
)
from experience_store import ExperienceStore


ROOT = Path(__file__).resolve().parent
BASE_SCENARIO = json.loads(
    (ROOT / "scenarios" / "corridor_obstacle.json").read_text(encoding="utf-8")
)


class DecisionOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        store = ExperienceStore(ROOT / "experiences" / "risk_experiences.json")
        cls.memory_cards = store.retrieve_memory_cards(
            BASE_SCENARIO,
            role="decision",
            top_k=4,
            token_budget=5000,
        )["items"]

    def scenario(self) -> dict:
        return copy.deepcopy(BASE_SCENARIO)

    def test_current_scenario_keeps_actions_without_hard_violation(self) -> None:
        result = filter_safe_actions(self.scenario())
        self.assertIn("move_forward", result["feasible_actions"])
        self.assertIn("observe_again", result["feasible_actions"])

    def test_l2_rule_prohibits_forward_only(self) -> None:
        rules = [
            {
                "rule_id": "rule_narrow_corridor_001",
                "prohibited_actions": ["move_forward"],
            }
        ]
        result = filter_safe_actions(self.scenario(), rules)
        self.assertNotIn("move_forward", result["feasible_actions"])
        self.assertIn("slow_down", result["feasible_actions"])
        self.assertEqual(
            result["action_assessments"]["move_forward"]["rejection_reasons"],
            ["prohibited_by_l2_rule:rule_narrow_corridor_001"],
        )

    def test_emergency_obstacle_filters_all_motion_actions(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["obstacle_distance_m"] = 0.20
        result = filter_safe_actions(scenario)
        for action in ("move_forward", "turn_left", "turn_right", "slow_down"):
            self.assertNotIn(action, result["feasible_actions"])
        self.assertIn("observe_again", result["feasible_actions"])
        self.assertIn("safe_stop", result["feasible_actions"])

    def test_low_confidence_filters_motion_but_keeps_observation(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["observation_confidence"] = 0.40
        result = filter_safe_actions(scenario)
        self.assertNotIn("move_forward", result["feasible_actions"])
        self.assertIn("observe_again", result["feasible_actions"])

    def test_low_battery_filters_motion_but_keeps_safe_stop(self) -> None:
        scenario = self.scenario()
        scenario["robot"]["battery_percent"] = 8
        result = filter_safe_actions(scenario)
        self.assertNotIn("move_forward", result["feasible_actions"])
        self.assertIn("safe_stop", result["feasible_actions"])

    def test_insufficient_clearance_filters_motion(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["corridor_width_m"] = 0.60
        result = filter_safe_actions(scenario)
        self.assertNotIn("move_forward", result["feasible_actions"])
        self.assertIn("ask_human", result["feasible_actions"])

    def test_no_available_action_returns_no_fallback(self) -> None:
        scenario = self.scenario()
        scenario["available_actions"] = []
        result = filter_safe_actions(scenario)
        self.assertTrue(result["fallback_required"])
        self.assertIsNone(result["selected_fallback"])

    def test_task_utility_exposes_complete_score_breakdown(self) -> None:
        result = compute_task_utility("move_forward")
        self.assertEqual(result["parameter_source"], "hand_configured_baseline")
        self.assertAlmostEqual(
            result["score"], sum(result["contributions"].values())
        )
        self.assertEqual(
            set(result["contributions"]),
            {"progress", "time_cost", "energy_cost"},
        )

    def test_forward_has_higher_task_utility_than_observation(self) -> None:
        forward = compute_task_utility("move_forward")["score"]
        observation = compute_task_utility("observe_again")["score"]
        self.assertGreater(forward, observation)

    def test_task_utility_ranking_is_deterministic(self) -> None:
        actions = ["observe_again", "slow_down", "move_forward"]
        first = score_task_utilities(self.scenario(), actions)
        second = score_task_utilities(self.scenario(), actions)
        self.assertEqual(first, second)
        self.assertEqual(
            first["ranked_actions"],
            ["move_forward", "slow_down", "observe_again"],
        )

    def test_only_feasible_actions_can_be_scored_after_filtering(self) -> None:
        rules = [
            {
                "rule_id": "rule_narrow_corridor_001",
                "prohibited_actions": ["move_forward"],
            }
        ]
        feasible = filter_safe_actions(self.scenario(), rules)["feasible_actions"]
        result = score_task_utilities(self.scenario(), feasible)
        self.assertNotIn("move_forward", result["items"])
        self.assertEqual(set(result["items"]), set(feasible))

    def test_unknown_action_has_no_silent_default_utility(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing task-utility profile"):
            compute_task_utility("jump")

    def test_risk_score_is_bounded_and_labeled_not_probability(self) -> None:
        result = estimate_action_risk(
            self.scenario(), "move_forward", self.memory_cards
        )
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertEqual(
            result["score_semantics"], "relative_risk_score_not_probability"
        )

    def test_failed_forward_memory_increases_forward_risk(self) -> None:
        without_memory = estimate_action_risk(self.scenario(), "move_forward", [])
        with_memory = estimate_action_risk(
            self.scenario(), "move_forward", self.memory_cards
        )
        self.assertGreater(with_memory["score"], without_memory["score"])
        self.assertGreater(with_memory["memory_evidence_count"], 0)

    def test_unmatched_action_does_not_invent_memory_evidence(self) -> None:
        result = estimate_action_risk(
            self.scenario(), "observe_again", self.memory_cards
        )
        self.assertEqual(result["memory_evidence_count"], 0)
        self.assertIsNone(result["memory_risk"])
        self.assertEqual(result["source_weights"], {"scene": 1.0, "memory": 0.0})

    def test_observation_is_lower_risk_than_forward_motion(self) -> None:
        risks = score_action_risks(
            self.scenario(),
            ["move_forward", "slow_down", "observe_again"],
            self.memory_cards,
        )
        self.assertLess(
            risks["items"]["observe_again"]["score"],
            risks["items"]["move_forward"]["score"],
        )
        self.assertEqual(risks["ranked_actions"][0], "observe_again")

    def test_safe_stop_has_lower_physical_risk_than_observation(self) -> None:
        risks = score_action_risks(
            self.scenario(),
            ["safe_stop", "observe_again"],
            self.memory_cards,
        )
        self.assertLess(
            risks["items"]["safe_stop"]["score"],
            risks["items"]["observe_again"]["score"],
        )

    def test_safer_scene_reduces_forward_scene_risk(self) -> None:
        risky = estimate_action_risk(self.scenario(), "move_forward", [])
        safe_scenario = self.scenario()
        safe_scenario["environment"].update(
            {
                "corridor_width_m": 1.8,
                "obstacle_detected": False,
                "obstacle_distance_m": 5.0,
                "observation_confidence": 0.95,
            }
        )
        safer = estimate_action_risk(safe_scenario, "move_forward", [])
        self.assertLess(safer["scene_action_risk"], risky["scene_action_risk"])

    def test_risk_scoring_is_deterministic(self) -> None:
        actions = ["move_forward", "slow_down", "observe_again"]
        first = score_action_risks(self.scenario(), actions, self.memory_cards)
        second = score_action_risks(self.scenario(), actions, self.memory_cards)
        self.assertEqual(first, second)

    def test_lower_observation_confidence_increases_motion_uncertainty(self) -> None:
        reports = self.consensus_reports("move_forward")
        normal = estimate_action_uncertainty(
            self.scenario(), "move_forward", self.memory_cards, reports
        )
        low_confidence = self.scenario()
        low_confidence["environment"]["observation_confidence"] = 0.20
        uncertain = estimate_action_uncertainty(
            low_confidence, "move_forward", self.memory_cards, reports
        )
        self.assertGreater(uncertain["score"], normal["score"])

    def test_matching_memories_reduce_memory_uncertainty(self) -> None:
        reports = self.consensus_reports("move_forward")
        without_memory = estimate_action_uncertainty(
            self.scenario(), "move_forward", [], reports
        )
        with_memory = estimate_action_uncertainty(
            self.scenario(), "move_forward", self.memory_cards, reports
        )
        self.assertLess(
            with_memory["components"]["memory"],
            without_memory["components"]["memory"],
        )

    def test_agent_consensus_reduces_supported_action_uncertainty(self) -> None:
        consensus = self.consensus_reports("observe_again")
        divided = {
            "advocate": {"recommendation": "observe_again", "confidence": 0.8},
            "critic": {"recommendation": "safe_stop", "confidence": 0.8},
            "decision": {
                "selected_action": {"name": "slow_down"},
                "confidence": 0.8,
            },
        }
        agreed = estimate_action_uncertainty(
            self.scenario(), "observe_again", self.memory_cards, consensus
        )
        disagreed = estimate_action_uncertainty(
            self.scenario(), "observe_again", self.memory_cards, divided
        )
        self.assertLess(agreed["components"]["agent"], disagreed["components"]["agent"])

    def test_missing_agent_reports_are_not_treated_as_consensus(self) -> None:
        result = estimate_action_uncertainty(
            self.scenario(), "observe_again", self.memory_cards, {}
        )
        self.assertFalse(result["agent"]["evidence_available"])
        self.assertEqual(result["components"]["agent"], 1.0)

    def test_safe_stop_ignores_observation_uncertainty(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["observation_confidence"] = 0.0
        result = estimate_action_uncertainty(
            scenario,
            "safe_stop",
            self.memory_cards,
            self.consensus_reports("safe_stop"),
        )
        self.assertEqual(result["components"]["observation"], 0.0)

    def test_uncertainty_scoring_is_deterministic(self) -> None:
        actions = ["move_forward", "slow_down", "observe_again"]
        reports = self.consensus_reports("observe_again")
        first = score_action_uncertainties(
            self.scenario(), actions, self.memory_cards, reports
        )
        second = score_action_uncertainties(
            self.scenario(), actions, self.memory_cards, reports
        )
        self.assertEqual(first, second)

    def test_optimizer_selects_observation_in_current_risky_scenario(self) -> None:
        rules = [
            {
                "rule_id": "rule_narrow_corridor_001",
                "prohibited_actions": ["move_forward"],
            }
        ]
        result = optimize_action(
            self.scenario(),
            self.memory_cards,
            rules,
            self.consensus_reports("observe_again"),
        )
        self.assertEqual(result["status"], "optimized")
        self.assertEqual(result["selected_action"]["name"], "observe_again")
        self.assertEqual(
            result["candidates"]["move_forward"]["status"],
            "rejected_by_hard_or_rule_constraint",
        )

    def test_optimizer_selects_forward_in_clear_supported_scenario(self) -> None:
        scenario = self.scenario()
        scenario["environment"].update(
            {
                "corridor_width_m": 1.8,
                "obstacle_detected": False,
                "obstacle_distance_m": 5.0,
                "observation_confidence": 0.95,
            }
        )
        result = optimize_action(
            scenario,
            [],
            [],
            self.consensus_reports("move_forward"),
        )
        self.assertEqual(result["selected_action"]["name"], "move_forward")

    def test_optimizer_rejects_action_above_numeric_limits(self) -> None:
        result = optimize_action(
            self.scenario(),
            self.memory_cards,
            [],
            self.consensus_reports("observe_again"),
        )
        self.assertFalse(result["candidates"]["move_forward"]["eligible"])
        self.assertIn(
            "risk_above_limit",
            result["candidates"]["move_forward"]["constraint_violations"],
        )

    def test_optimizer_falls_back_when_no_action_meets_zero_limits(self) -> None:
        result = optimize_action(
            self.scenario(),
            self.memory_cards,
            [],
            self.consensus_reports("observe_again"),
            maximum_risk=0.0,
            maximum_uncertainty=0.0,
        )
        self.assertEqual(result["status"], "fallback_no_eligible_action")
        self.assertEqual(result["selected_action"]["name"], "safe_stop")

    def test_optimizer_output_is_deterministic_and_auditable(self) -> None:
        arguments = (
            self.scenario(),
            self.memory_cards,
            [],
            self.consensus_reports("observe_again"),
        )
        first = optimize_action(*arguments)
        second = optimize_action(*arguments)
        self.assertEqual(first, second)
        self.assertEqual(
            first["decision_source"], "deterministic_constrained_optimizer"
        )
        self.assertIn("constraints", first)
        self.assertIn("candidates", first)

    def test_optimizer_rejects_invalid_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum_risk"):
            optimize_action(self.scenario(), maximum_risk=1.1)

    @staticmethod
    def consensus_reports(action: str) -> dict:
        return {
            "advocate": {"recommendation": action, "confidence": 0.8},
            "critic": {"recommendation": action, "confidence": 0.9},
            "decision": {
                "selected_action": {"name": action},
                "confidence": 0.85,
            },
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
