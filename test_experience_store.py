"""Offline tests for risk-experience validation and retrieval."""

import json
import unittest
from pathlib import Path

from experience_store import ExperienceStore, build_memory_card


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
