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

    def test_s1_uses_uncertainty_adjusted_worst_side_clearance(self) -> None:
        scenario = self.scenario()
        scenario["task"].update(
            requires_envelope_clearance=True,
            minimum_envelope_clearance_m=0.10,
            maximum_corridor_heading_error_rad=0.05,
        )
        scenario["robot"]["width_m"] = 0.60
        scenario["environment"].update(
            corridor_width_m=0.90,
            corridor_geometry_valid=True,
            left_envelope_clearance_m=0.15,
            right_envelope_clearance_m=0.15,
            clearance_uncertainty_m=0.02,
            minimum_envelope_clearance_m=0.13,
            corridor_heading_error_rad=0.0,
        )
        approved = apply_safety_guard(scenario, decision("slow_down"))
        self.assertEqual(approved["status"], "approved")

        scenario["environment"].update(
            left_envelope_clearance_m=0.08,
            right_envelope_clearance_m=0.22,
            minimum_envelope_clearance_m=0.06,
        )
        stopped = apply_safety_guard(scenario, decision("slow_down"))
        self.assertEqual(stopped["approved_action"]["name"], "safe_stop")
        self.assertIn(
            "minimum_envelope_clearance_below_hard_limit",
            stopped["hard_rule_violations"],
        )

        scenario["environment"].update(
            left_envelope_clearance_m=0.15,
            right_envelope_clearance_m=0.15,
            minimum_envelope_clearance_m=0.13,
            corridor_heading_error_rad=0.07,
        )
        misaligned = apply_safety_guard(scenario, decision("slow_down"))
        self.assertEqual(misaligned["approved_action"]["name"], "safe_stop")
        self.assertIn("corridor_heading_misaligned", misaligned["hard_rule_violations"])

    def test_s1_missing_or_inconsistent_geometry_fails_closed(self) -> None:
        scenario = self.scenario()
        scenario["task"]["requires_envelope_clearance"] = True
        missing = apply_safety_guard(scenario, decision("move_forward"))
        self.assertIn("corridor_geometry_unavailable", missing["hard_rule_violations"])
        self.assertEqual(missing["approved_action"]["name"], "safe_stop")

        scenario["robot"]["width_m"] = 0.60
        scenario["environment"].update(
            corridor_width_m=0.90,
            corridor_geometry_valid=True,
            left_envelope_clearance_m=0.15,
            right_envelope_clearance_m=0.15,
            clearance_uncertainty_m=0.02,
            minimum_envelope_clearance_m=0.18,
        )
        inconsistent = apply_safety_guard(scenario, decision("move_forward"))
        self.assertIn("corridor_geometry_inconsistent", inconsistent["hard_rule_violations"])

    def test_adapter_level_environment_flag_also_requires_geometry(self) -> None:
        scenario = self.scenario()
        scenario["environment"]["envelope_clearance_required"] = True
        result = apply_safety_guard(scenario, decision("slow_down"))
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn("corridor_geometry_unavailable", result["hard_rule_violations"])

    def test_geometry_values_marked_invalid_force_stop_even_when_optional(self) -> None:
        scenario = self.scenario()
        scenario["environment"].update(
            corridor_geometry_valid=False,
            left_envelope_clearance_m=0.20,
            right_envelope_clearance_m=0.20,
            clearance_uncertainty_m=0.02,
            minimum_envelope_clearance_m=0.18,
        )
        result = apply_safety_guard(scenario, decision("move_forward"))
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn("corridor_geometry_inconsistent", result["hard_rule_violations"])

    def test_nonfinite_required_geometry_fails_closed(self) -> None:
        scenario = self.scenario()
        scenario["task"].update(
            requires_envelope_clearance=True,
            minimum_envelope_clearance_m=0.10,
            maximum_corridor_heading_error_rad=0.05,
        )
        scenario["robot"]["width_m"] = 0.60
        scenario["environment"].update(
            corridor_width_m=0.90,
            corridor_geometry_valid=True,
            left_envelope_clearance_m=float("nan"),
            right_envelope_clearance_m=0.15,
            clearance_uncertainty_m=0.02,
            minimum_envelope_clearance_m=0.13,
            corridor_heading_error_rad=0.0,
        )
        result = apply_safety_guard(scenario, decision("slow_down"))
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertIn("corridor_geometry_inconsistent", result["hard_rule_violations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
