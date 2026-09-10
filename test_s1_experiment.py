"""Offline S1 configuration, provenance, confirmation, and trial-loop tests."""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

from prepare_s1_memories import build_seed_memories
from experience_store import ExperienceStore
from robot_interface import ExecutionReceipt, RobotObservation
from run_s1_trial import require_terminal_confirmation, verify_pilot_approvals
from s1_config import (
    S1ConfigError,
    confirmation_token,
    load_s1_config,
    setup_fingerprint,
    task_at,
    trial_geometry,
    trial_id,
)
from s1_experiment import JSONLAuditLog, S1Metrics, S1PilotRunner, S1TrialRunner


ROOT = Path(__file__).resolve().parent


def completed_config():
    data = json.loads((ROOT / "configs" / "s1.example.json").read_text(encoding="utf-8"))
    data["geometry"].update(
        robot_effective_width_m=.60,
        robot_left_extent_m=.31,
        robot_right_extent_m=.29,
    )
    data["layouts"]["A"]["measured_width_m"] = 1.40
    data["layouts"]["B"]["measured_width_m"] = .90
    for name, offset in (("A", .70), ("B", .45)):
        data["layouts"][name].update(
            left_inner_offset_m=offset,
            right_inner_offset_m=offset,
            measured_box_length_m=1.20,
            measured_box_height_m=1.70,
            measurement_note=f"offline tape fixture {name}",
            box_shape_and_fixing_note="offline rectangular fixed fixture",
            foot_envelope_relation_note="offline asymmetric envelope calculation",
        )
    data["perception"]["calibration_id"] = "cal_s1_test"
    data["perception"]["onsite_validation"].update(
        status="onsite_verified",
        operator="offline-test",
        verified_at="2026-09-10T00:00:00+08:00",
        tf_method="offline fixture",
        robot_envelope_method="offline fixture",
        box_geometry_method="offline fixture",
    )
    data["runtime"]["poll_interval_s"] = .0001
    return data


def load_completed(directory: Path):
    path = directory / "s1.json"
    path.write_text(json.dumps(completed_config()), encoding="utf-8")
    return load_s1_config(path, require_onsite=True), path


def observation(x: float, sequence: int) -> RobotObservation:
    now = datetime.now(timezone.utc).isoformat()
    return RobotObservation(
        robot={"device_id": "G1", "type": "g1", "width_m": .6, "battery_percent": 80},
        environment={
            "corridor_width_m": .9,
            "corridor_geometry_valid": True,
            "left_envelope_clearance_m": .15,
            "right_envelope_clearance_m": .15,
            "clearance_uncertainty_m": .02,
            "minimum_envelope_clearance_m": .13,
            "corridor_center_offset_m": 0.,
            "corridor_geometry_confidence": .9,
            "corridor_heading_error_rad": 0.,
            "obstacle_detected": False,
            "obstacle_distance_m": None,
            "observation_confidence": .9,
        },
        timestamp=now,
        metadata={
            "g1_state": {"yaw_rad": 0., "fsm_id": 500},
            "odometry": {
                "timestamp": now,
                "source_timestamp": f"frame-{sequence}",
                "position_m": [x, 0., 0.],
                "velocity_mps": [0., 0., 0.],
                "yaw_rate_rps": 0.,
            },
        },
    )


class FakeProvider:
    def __init__(self):
        self.x = 0.
        self.sequence = 0
        self.origin = 0.

    def __call__(self):
        self.sequence += 1
        return observation(self.x, self.sequence)

    def set_origin(self, value):
        self.origin = value.metadata["odometry"]["position_m"][0]

    def set_geometry_requirement(self, predicate):
        self.geometry_requirement = predicate

    def progress(self, value):
        x = value.metadata["odometry"]["position_m"][0] - self.origin
        return {"forward_m": x, "lateral_m": 0., "xy_m": abs(x)}


class FakeAdapter:
    def emergency_stop(self, reason):
        now = datetime.now(timezone.utc).isoformat()
        return ExecutionReceipt("aborted", reason, now, now, {"stop_submitted": True})


class FakeBridge:
    def __init__(self, provider):
        self.provider = provider
        self.adapter = FakeAdapter()
        self.actions = []

    def execute_decision(self, task, decision):
        action = decision["selected_action"]
        self.actions.append(action["name"])
        if action["name"] in {"move_forward", "slow_down"}:
            self.provider.x = round(self.provider.x + .10, 10)
            receipt_status = "success"
        elif action["name"] == "observe_again":
            receipt_status = "aborted"
        else:
            receipt_status = "success"
        now = datetime.now(timezone.utc).isoformat()
        return {
            "status": "executed",
            "approved_action": action,
            "guard": {"status": "approved", "hard_rule_violations": []},
            "receipt": {
                "status": receipt_status,
                "description": "offline fake",
                "started_at": now,
                "finished_at": now,
                "telemetry": {},
            },
            "l0_record": {},
        }


class S1ExperimentTests(unittest.TestCase):
    def test_config_geometry_and_onsite_gate(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, _path = load_completed(directory)
            geometry = trial_geometry(config, "B")
            self.assertAlmostEqual(geometry["decision_line_m"], .7)
            self.assertAlmostEqual(geometry["entrance_m"], 1.5)
            self.assertAlmostEqual(geometry["exit_m"], 2.7)
            self.assertAlmostEqual(geometry["goal_m"], 4.2)

            raw = completed_config()
            raw["perception"]["onsite_validation"]["status"] = "NOT_VERIFIED"
            path = directory / "unverified.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(S1ConfigError):
                load_s1_config(path, require_onsite=True)

    def test_config_rejects_untraceable_or_nominally_unsafe_box_geometry(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            raw = completed_config()
            raw["layouts"]["B"]["measured_box_height_m"] = 1.0
            path = directory / "short-box.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(S1ConfigError):
                load_s1_config(path, require_onsite=True)

            raw = completed_config()
            raw["layouts"]["B"].update(
                left_inner_offset_m=.41,
                right_inner_offset_m=.49,
            )
            path = directory / "unsafe-side.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(S1ConfigError):
                load_s1_config(path, require_onsite=True)

    def test_generated_seed_cards_are_never_physical_history(self):
        with tempfile.TemporaryDirectory() as directory_name:
            config, _path = load_completed(Path(directory_name))
            records = build_seed_memories(config)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["source"]["kind"] == "human_constructed" for record in records))
        self.assertTrue(all(record["source"]["historical_physical_run"] is False for record in records))
        self.assertIn("不是真机历史失败", records[1]["outcome"]["description"])
        self.assertAlmostEqual(records[0]["environment"]["left_envelope_clearance_m"], .14)
        self.assertAlmostEqual(records[0]["environment"]["right_envelope_clearance_m"], .16)

    def test_default_seed_retrieval_is_role_differentiated(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config, _path = load_completed(directory)
            records = build_seed_memories(config)
            memory_path = directory / "memories.json"
            memory_path.write_text(json.dumps(records), encoding="utf-8")
            store = ExperienceStore(memory_path)
            scenario = {
                "robot": records[0]["robot"],
                "environment": records[0]["environment"],
                "task": task_at(config, "B", .70),
                "available_actions": [
                    "move_forward", "slow_down", "observe_again", "safe_stop"
                ],
            }
            bundles = {
                role: store.retrieve_memory_cards(
                    scenario,
                    role=role,
                    top_k=config["memory"]["top_k"],
                    token_budget=config["memory"]["token_budget"],
                )
                for role in ("advocate", "critic", "decision")
            }
        advocate_ids = [item["experience_id"] for item in bundles["advocate"]["items"]]
        critic_ids = [item["experience_id"] for item in bundles["critic"]["items"]]
        self.assertIn("s1_constructed_narrow_low_speed_success", advocate_ids)
        self.assertIn("s1_constructed_wide_corridor_distractor", advocate_ids)
        self.assertIn("s1_constructed_insufficient_margin_abort", critic_ids)
        self.assertNotEqual(advocate_ids, critic_ids)

    def test_confirmation_repeats_exact_motion_and_rejects_mismatch(self):
        config = completed_config()
        identifier = trial_id("B", "M", 1)
        token = confirmation_token(identifier)
        output = io.StringIO()
        accepted = require_terminal_confirmation(
            config, "B", "M", 1,
            input_fn=lambda prompt: token,
            output=output,
            require_tty=False,
        )
        self.assertEqual(accepted, token)
        self.assertIn("只沿当前朝向直行", output.getvalue())
        with self.assertRaises(RuntimeError):
            require_terminal_confirmation(
                config, "B", "M", 1,
                input_fn=lambda prompt: "no",
                output=io.StringIO(),
                require_tty=False,
            )

    def test_runner_first_deliberates_only_after_d_and_verifies_goal(self):
        with tempfile.TemporaryDirectory() as directory_name:
            config, _path = load_completed(Path(directory_name))
            provider = FakeProvider()
            bridge = FakeBridge(provider)
            decision_progress = []

            def decide(_scenario):
                decision_progress.append(provider.x - provider.origin)
                return {
                    "decision": "execute",
                    "selected_action": {"name": "slow_down", "parameters": {}},
                    "reason": "offline fake",
                    "decision_source": "deterministic_constrained_optimizer",
                }

            audit = JSONLAuditLog(Path(directory_name) / "events.jsonl")
            metrics = S1Metrics(config, "B")
            summary = S1TrialRunner(
                config, "B", "M", provider, bridge, decide, metrics, audit
            ).run()
            events = [json.loads(line) for line in audit.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["status"], "completed")
        self.assertGreaterEqual(decision_progress[0], .7)
        self.assertGreaterEqual(summary["final_progress"]["forward_m"], 4.12)
        self.assertTrue(any(event["event"] == "decision_line_reached" for event in events))
        self.assertNotIn("turn_left", bridge.actions)
        self.assertNotIn("turn_right", bridge.actions)

    def test_low_speed_pilot_is_non_formal_and_covers_whole_passage(self):
        with tempfile.TemporaryDirectory() as directory_name:
            config, _path = load_completed(Path(directory_name))
            provider = FakeProvider()
            bridge = FakeBridge(provider)
            audit = JSONLAuditLog(Path(directory_name) / "pilot-events.jsonl")
            geometry = trial_geometry(config, "B")
            metrics = S1Metrics(
                config, "B", passage_interval=(0., geometry["pilot_exit_m"])
            )
            summary = S1PilotRunner(
                config, "B", provider, bridge, metrics, audit
            ).run()
        self.assertEqual(summary["status"], "pilot_completed")
        self.assertFalse(summary["formal_trial"])
        self.assertFalse(summary["eligible_as_s1_result"])
        self.assertTrue(summary["low_speed_strategy_candidate_passed"])
        self.assertGreaterEqual(summary["final_progress"]["forward_m"], 1.55)

    def test_formal_trials_require_both_matching_reviewed_pilots(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            raw = completed_config()
            raw["paths"]["results_dir"] = str(directory / "results")
            config_path = directory / "s1.json"
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_s1_config(config_path, require_onsite=True)
            fingerprint = setup_fingerprint(config)
            for layout in ("A", "B"):
                identifier = f"S1_PILOT_{layout}_R1"
                result_dir = directory / "results" / identifier
                result_dir.mkdir(parents=True)
                (result_dir / "summary.json").write_text(json.dumps({
                    "pilot_id": identifier,
                    "layout": layout,
                    "formal_trial": False,
                    "eligible_as_s1_result": False,
                    "low_speed_strategy_candidate_passed": True,
                    "setup_fingerprint": fingerprint,
                }), encoding="utf-8")
                raw["pilot_approvals"][layout] = {
                    "approved": True,
                    "pilot_id": identifier,
                    "reviewed_by": "reviewer",
                    "reviewed_at": "2026-09-10T12:00:00+08:00",
                }
            config_path.write_text(json.dumps(raw), encoding="utf-8")
            approved = load_s1_config(config_path, require_onsite=True)
            evidence = verify_pilot_approvals(approved)
            self.assertEqual(set(evidence), {"A", "B"})
            approved["runtime"]["maximum_horizontal_speed_mps"] = .50
            with self.assertRaises(RuntimeError):
                verify_pilot_approvals(approved)


if __name__ == "__main__":
    unittest.main()
