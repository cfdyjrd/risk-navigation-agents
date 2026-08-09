"""Offline tests for the deterministic safety guard (no API calls)."""

import copy
import json
import unittest
from pathlib import Path

from safety_guard import apply_safety_guard


ROOT = Path(__file__).resolve().parent
BASE_SCENARIO = json.loads(
    (ROOT / "scenarios" / "corridor_obstacle.json").read_text(encoding="utf-8")
)


def decision(action: str) -> dict:
    return {
        "decision": "execute",
        "selected_action": {"name": action, "parameters": {}},
    }


class SafetyGuardTests(unittest.TestCase):
    def scenario(self) -> dict:
        return copy.deepcopy(BASE_SCENARIO)

    def test_safe_observation_is_approved(self) -> None:
        result = apply_safety_guard(self.scenario(), decision("observe_again"))
        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["approved_action"]["name"], "observe_again")

    def test_emergency_obstacle_overrides_forward(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["obstacle_distance_m"] = 0.20
        result = apply_safety_guard(scenario, decision("move_forward"))
        self.assertEqual(result["status"], "overridden")
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn("emergency_obstacle_distance", result["hard_rule_violations"])

    def test_low_confidence_requires_observation(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["observation_confidence"] = 0.40
        result = apply_safety_guard(scenario, decision("move_forward"))
        self.assertEqual(result["status"], "overridden")
        self.assertEqual(result["approved_action"]["name"], "observe_again")

    def test_low_battery_forces_stop(self) -> None:
        scenario = self.scenario()
        scenario["robot"]["battery_percent"] = 8
        result = apply_safety_guard(scenario, decision("move_forward"))
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn("battery_below_hard_limit", result["hard_rule_violations"])

    def test_insufficient_clearance_forces_stop(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["corridor_width_m"] = 0.60
        result = apply_safety_guard(scenario, decision("move_forward"))
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn(
            "corridor_clearance_below_hard_limit",
            result["hard_rule_violations"],
        )

    def test_unavailable_action_forces_stop(self) -> None:
        result = apply_safety_guard(self.scenario(), decision("jump"))
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn("action_not_available", result["hard_rule_violations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

