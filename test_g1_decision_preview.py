from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from run_g1_decision import run


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stamp = datetime.now(timezone.utc).isoformat()
        self.state = {"device_id": "G1", "state": {
            "lowstate": {"timestamp": stamp, "rpy_rad": [0., 0., 0.]},
            "odometry": {"timestamp": stamp, "position_m": [0., 0., .7],
                         "velocity_mps": [0., 0., 0.], "yaw_rate_rps": 0., "error_code": 0},
            "battery": {"timestamp": stamp, "soc": 80},
            "fsm": {"timestamp": stamp, "fsm_id": 500}}}
        self.perception = {"timestamp": stamp, "validated": True, "frame_id": "base_link",
                           "obstacle_detected": True, "obstacle_distance_m": .1,
                           "observation_confidence": .9}
        self.config = {"robot": {"device_id": "G1", "model": "g1", "allowed_fsm_ids": [500],
                                 "max_tilt_rad": .2}, "width_m": .6,
                       "state_file": "state.json", "perception_file": "perception.json",
                       "task": {"goal": "preview"}}
        self.path = self.root / "config.json"

    def write(self):
        for name, data in (("config.json", self.config), ("state.json", self.state),
                           ("perception.json", self.perception)):
            (self.root / name).write_text(json.dumps(data))

    def test_missing_calibration_and_stale_state_skip_model(self):
        self.config["robot"]["allowed_fsm_ids"] = []
        self.config["robot"]["max_tilt_rad"] = None
        self.config["width_m"] = None
        self.config["perception_file"] = None
        self.state["state"]["battery"]["timestamp"] = "2000-01-01T00:00:00+00:00"
        self.write()
        factory = Mock()
        result = run(self.path, decide=True, decider_factory=factory)
        self.assertEqual(result["status"], "blocked")
        self.assertGreaterEqual(len(result["blockers"]), 5)
        factory.assert_not_called()
        self.assertFalse(result["model_called"])

    def test_readiness_never_calls_models_or_drivers(self):
        self.write()
        factory = Mock()
        with patch("unitree_adapter.G1LocoDriver.connect", side_effect=AssertionError("no hardware")):
            result = run(self.path, decider_factory=factory)
        self.assertEqual(result["status"], "ready_for_decision_preview")
        factory.assert_not_called()
        self.assertFalse(result["motion_ready"])

    def test_preview_applies_guard_but_never_executes(self):
        self.write()
        decide = Mock(return_value={"decision_source": "deterministic_constrained_optimizer",
                                    "selected_action": {"name": "move_forward", "parameters": {}}})
        with patch("unitree_adapter.G1LocoDriver.connect", side_effect=AssertionError("no hardware")), \
             patch("execution_bridge.SafeExecutionBridge.execute_decision", side_effect=AssertionError("no execution")):
            result = run(self.path, decide=True, decider_factory=lambda: decide)
        self.assertEqual(result["status"], "decision_preview_only")
        self.assertEqual(result["preview_guard"]["approved_action"]["name"], "safe_stop")
        self.assertEqual(result["motion_commands_sent"], 0)
        self.assertFalse(result["execution_authorized"])
        self.assertNotIn("turn_left", decide.call_args.args[0]["available_actions"])

    def test_moving_or_bad_perception_blocks_before_model(self):
        factory = Mock()
        self.state["state"]["odometry"]["velocity_mps"] = [.2, 0., 0.]
        self.write()
        self.assertEqual(run(self.path, decide=True, decider_factory=factory)["status"], "blocked")
        self.state["state"]["odometry"]["velocity_mps"] = [0., 0., 0.]
        self.perception["frame_id"] = "livox_frame"
        self.write()
        self.assertEqual(run(self.path, decide=True, decider_factory=factory)["status"], "blocked")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
