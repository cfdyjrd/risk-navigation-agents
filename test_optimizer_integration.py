"""Offline integration tests for Agent reports -> optimizer -> Safety Guard."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experience_store import ExperienceStore
from risk_rule_store import RiskRuleStore
from run_three_agents import (
    load_or_run_agent,
    run_post_deliberation_optimizer,
    save_cached,
)
from safety_guard import apply_safety_guard


ROOT = Path(__file__).resolve().parent


class OptimizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = json.loads(
            (ROOT / "scenarios" / "corridor_obstacle.json").read_text(
                encoding="utf-8"
            )
        )
        cls.memories = ExperienceStore(
            ROOT / "experiences" / "risk_experiences.json"
        ).retrieve_memory_cards(
            cls.scenario,
            role="decision",
            top_k=4,
            token_budget=5000,
        )["items"]
        cls.rules = RiskRuleStore(
            ROOT / "experiences" / "risk_rules.json"
        ).retrieve_matching(cls.scenario)

    @staticmethod
    def report(action: str, confidence: float = 0.9) -> dict:
        return {"recommendation": action, "confidence": confidence}

    @staticmethod
    def model_decision(action: str, confidence: float = 0.9) -> dict:
        return {
            "decision": "execute",
            "selected_action": {"name": action, "parameters": {}},
            "confidence": confidence,
        }

    def test_consensus_observation_reaches_safety_guard(self) -> None:
        result, execution = run_post_deliberation_optimizer(
            self.scenario,
            self.memories,
            self.rules,
            self.report("observe_again"),
            self.report("observe_again"),
            self.model_decision("observe_again"),
        )
        self.assertTrue(result["agrees_with_model"])
        self.assertEqual(result["optimized_action"]["name"], "observe_again")
        guarded = apply_safety_guard(self.scenario, execution)
        self.assertEqual(guarded["status"], "approved")
        self.assertEqual(guarded["approved_action"]["name"], "observe_again")

    def test_optimizer_corrects_rule_prohibited_model_recommendation(self) -> None:
        result, execution = run_post_deliberation_optimizer(
            self.scenario,
            self.memories,
            self.rules,
            self.report("move_forward"),
            self.report("move_forward"),
            self.model_decision("move_forward"),
        )
        self.assertFalse(result["agrees_with_model"])
        self.assertEqual(
            result["candidates"]["move_forward"]["status"],
            "rejected_by_hard_or_rule_constraint",
        )
        self.assertNotEqual(execution["selected_action"]["name"], "move_forward")
        guarded = apply_safety_guard(self.scenario, execution)
        self.assertEqual(
            guarded["approved_action"]["name"],
            execution["selected_action"]["name"],
        )

    def test_execution_decision_preserves_model_and_optimizer_provenance(self) -> None:
        result, execution = run_post_deliberation_optimizer(
            self.scenario,
            self.memories,
            self.rules,
            self.report("observe_again", 0.8),
            self.report("observe_again", 0.85),
            self.model_decision("observe_again", 0.88),
        )
        self.assertEqual(
            execution["decision_source"], "deterministic_constrained_optimizer"
        )
        self.assertEqual(execution["model_recommendation"]["name"], "observe_again")
        self.assertEqual(execution["selected_action"], result["optimized_action"])

    def test_complete_cache_does_not_initialize_api_client(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            expected_report = {"recommendation": "observe_again"}
            expected_usage = {"total_tokens": 10}
            save_cached(
                cache_dir,
                "advocate",
                expected_report,
                expected_usage,
            )
            with patch(
                "run_three_agents.ZhinaoClient",
                side_effect=AssertionError("API client must not be initialized"),
            ):
                report, usage, client = load_or_run_agent(
                    cache_dir,
                    "advocate",
                    None,
                    lambda _: (_ for _ in ()).throw(
                        AssertionError("Agent must not run on cache hit")
                    ),
                )
            self.assertEqual(report, expected_report)
            self.assertEqual(usage, expected_usage)
            self.assertIsNone(client)


if __name__ == "__main__":
    unittest.main(verbosity=2)
