"""Offline tests for risk-experience validation and retrieval."""

import json
import unittest
from pathlib import Path

from experience_store import (
    MEMORY_SCHEMA_VERSION,
    ExperienceStore,
    build_memory_card,
    estimate_tokens,
)


ROOT = Path(__file__).resolve().parent


class ExperienceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ExperienceStore(
            ROOT / "experiences" / "risk_experiences.json"
        )
        self.scenario = json.loads(
            (ROOT / "scenarios" / "corridor_obstacle.json").read_text(
                encoding="utf-8"
            )
        )

    def test_loads_valid_experiences(self) -> None:
        self.assertEqual(len(self.store.load()), 4)

    def test_narrow_collision_is_most_relevant(self) -> None:
        results = self.store.retrieve(self.scenario, top_k=3)
        self.assertEqual(
            results[0].experience["experience_id"],
            "exp_narrow_collision_001",
        )

    def test_returns_requested_number(self) -> None:
        self.assertEqual(len(self.store.retrieve(self.scenario, top_k=2)), 2)

    def test_memory_card_keeps_evidence_but_removes_redundant_fields(self) -> None:
        result = self.store.retrieve(self.scenario, top_k=1)[0]
        card = build_memory_card(result)
        self.assertEqual(card["experience_id"], "exp_narrow_collision_001")
        self.assertAlmostEqual(card["key_conditions"]["clearance_m"], 0.15)
        self.assertIn("outcome", card)
        self.assertIn("lesson", card)
        self.assertNotIn("robot", card)
        self.assertNotIn("environment", card)
        self.assertNotIn("task_goal", card)

    def test_memory_card_has_layered_risk_fields_and_provenance(self) -> None:
        result = self.store.retrieve(self.scenario, top_k=1, role="critic")[0]
        card = build_memory_card(result)
        self.assertEqual(card["schema_version"], MEMORY_SCHEMA_VERSION)
        self.assertEqual(card["memory_level"], "L1_risk_memory_card")
        self.assertTrue(card["memory_id"].startswith("rmc_"))
        self.assertIn("trace_id", card["source"])
        self.assertIn("trigger_conditions", card)
        self.assertIn("hazard", card)
        self.assertIn("failure_reason", card)
        self.assertIn("mitigations", card)
        self.assertIn("stop_conditions", card)
        self.assertIn("statistics", card)
        self.assertIn("quality", card)
        self.assertIn("score_breakdown", card["retrieval"])

    def test_role_specific_retrieval_uses_different_evidence(self) -> None:
        advocate = self.store.retrieve(self.scenario, top_k=1, role="advocate")[0]
        critic = self.store.retrieve(self.scenario, top_k=1, role="critic")[0]
        self.assertEqual(
            advocate.experience["experience_id"], "exp_narrow_success_002"
        )
        self.assertEqual(
            critic.experience["experience_id"], "exp_narrow_collision_001"
        )
        self.assertEqual(advocate.role, "advocate")
        self.assertEqual(critic.role, "critic")

    def test_risk_aware_score_is_explainable(self) -> None:
        result = self.store.retrieve(self.scenario, top_k=1, role="decision")[0]
        for component in (
            "similarity",
            "severity",
            "reliability",
            "recency",
            "role_fit",
        ):
            self.assertIn(component, result.score_breakdown)
            self.assertGreaterEqual(result.score_breakdown[component], 0)
            self.assertLessEqual(result.score_breakdown[component], 1)

    def test_evidence_bundle_respects_token_budget(self) -> None:
        budget = 1200
        bundle = self.store.retrieve_memory_cards(
            self.scenario,
            role="critic",
            top_k=3,
            token_budget=budget,
        )
        self.assertLessEqual(bundle["estimated_tokens_used"], budget)
        self.assertGreater(len(bundle["items"]), 0)
        for card in bundle["items"]:
            self.assertEqual(card["retrieval"]["role"], "critic")
            self.assertEqual(
                card["retrieval"]["estimated_tokens"],
                estimate_tokens({
                    **card,
                    "retrieval": {
                        key: value
                        for key, value in card["retrieval"].items()
                        if key != "estimated_tokens"
                    },
                }),
            )

    def test_invalid_role_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.retrieve(self.scenario, role="observer")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
