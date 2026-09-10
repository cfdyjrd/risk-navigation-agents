"""G1 safety and SDK mapping tests; all motion uses fake clients."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from execution_bridge import SafeExecutionBridge
from robot_interface import RobotInterfaceError, RobotObservation
from robot_preflight import sdk_probe, validate_config
from unitree_adapter import G1Config, G1LocoDriver, UnitreeConfig, UnitreeRobotAdapter


def snapshot(**changes):
    stamp = datetime.now(timezone.utc).isoformat()
    state = dict(lowstate_timestamp=stamp, fsm_timestamp=stamp,
                 attitude_timestamp=stamp, fsm_id=500, roll_rad=0., pitch_rad=0.)
    state.update(changes)
    return RobotObservation(
        {"device_id": "G1", "type": "g1", "width_m": .6, "battery_percent": 80},
        {"obstacle_detected": False, "obstacle_distance_m": None, "observation_confidence": .9},
        stamp, metadata={"g1_state": state})


class G1Tests(unittest.TestCase):
    def setUp(self):
        self.config = G1Config("G1", allowed_fsm_ids=(500,), max_tilt_rad=.2)
        self.client = SimpleNamespace(SetVelocity=Mock(return_value=0))
        self.driver = G1LocoDriver(self.client)
        self.adapter = UnitreeRobotAdapter(self.config, self.driver, snapshot)

    def motion(self):
        return self.adapter.execute({"name": "move_forward", "parameters": {"duration_s": .01}})

    def test_bounded_velocity_and_stop_keep_rpc_status(self):
        result = self.motion()
        self.assertEqual(result.status, "success")
        self.assertEqual(self.client.SetVelocity.call_args_list[0].args, (.1, 0., 0., .2))
        self.assertEqual(self.client.SetVelocity.call_args_list[-1].args, (0., 0., 0., .2))
        self.assertFalse(result.telemetry["physical_completion_verified"])

    def test_unverified_g1_configuration_is_rejected(self):
        for make in (lambda: UnitreeConfig("G1", "g1"), lambda: G1Config("G1"),
                     lambda: G1Config("G1", allowed_fsm_ids=(500,)),
                     lambda: G1Config("G1", allowed_fsm_ids=(True,), max_tilt_rad=.2),
                     lambda: replace(self.config, model="go2"),
                     lambda: replace(self.config, max_tilt_rad=float("nan"))):
            with self.assertRaises(RobotInterfaceError):
                make()

    def test_bad_readiness_never_sends_motion(self):
        old = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
        cases = [{"fsm_id": 1}, {"fsm_id": True}, {"roll_rad": .3},
                 {"pitch_rad": float("nan")}, {"roll_rad": False}]
        for key in ("lowstate_timestamp", "fsm_timestamp", "attitude_timestamp"):
            cases.extend([{key: old}, {key: future}, {key: None}])
        for changes in cases:
            with self.subTest(changes=changes):
                self.client.SetVelocity.reset_mock()
                adapter = UnitreeRobotAdapter(self.config, self.driver, lambda: snapshot(**changes))
                result = adapter.execute({"name": "move_forward"})
                self.assertEqual(result.status, "failure")
                self.client.SetVelocity.assert_called_once_with(0., 0., 0., .2)

    def test_missing_readiness_wrong_device_and_dog_state_rejected(self):
        base = snapshot()
        for obs in (replace(base, metadata={}),
                    replace(base, robot={**base.robot, "device_id": "other"}),
                    replace(base, robot={**base.robot, "type": "go2"})):
            adapter = UnitreeRobotAdapter(self.config, self.driver, lambda: obs)
            with self.assertRaises(RobotInterfaceError):
                adapter.observe()

    def test_sdk_failure_and_stop_failure_latch(self):
        for code in (1, None, False):
            self.client.SetVelocity.return_value = code
            with self.assertRaises(RobotInterfaceError):
                self.driver.move(.1, 0, 0)
            with self.assertRaises(RobotInterfaceError):
                self.driver.stop()
        self.assertEqual(self.motion().status, "failure")
        self.client.SetVelocity.return_value = 0
        self.assertEqual(self.motion().status, "failure")
        self.adapter.reset_emergency_stop()
        self.assertEqual(self.motion().status, "success")

    def test_bridge_stops_for_obstacle(self):
        obs = snapshot()
        self.adapter.observation_provider = lambda: replace(obs, environment={
            **obs.environment, "obstacle_detected": True, "obstacle_distance_m": .1})
        bridge = SafeExecutionBridge(self.adapter, available_actions=["move_forward", "safe_stop"])
        result = bridge.execute_decision({"goal": "test"}, {
            "decision_source": "deterministic_constrained_optimizer",
            "selected_action": {"name": "move_forward", "parameters": {}}})
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.client.SetVelocity.assert_called_once_with(0., 0., 0., .2)

    def test_readiness_loss_during_motion_stops(self):
        self.adapter.observation_provider = lambda: snapshot(
            fsm_id=1 if self.client.SetVelocity.called else 500)
        result = self.adapter.execute({"name": "move_forward", "parameters": {"duration_s": .15}})
        self.assertEqual(result.status, "failure")
        self.assertEqual(self.client.SetVelocity.call_count, 2)
        self.assertEqual(self.client.SetVelocity.call_args.args, (0., 0., 0., .2))

    def test_connect_only_initializes_correct_sdk(self):
        client = Mock(spec=["SetTimeout", "Init", "SetVelocity"])
        constructor = Mock(return_value=client)
        channel = SimpleNamespace(ChannelFactoryInitialize=Mock())
        loco = SimpleNamespace(LocoClient=constructor)
        with patch("unitree_adapter.importlib.import_module", side_effect=[channel, loco]) as imports:
            G1LocoDriver.connect("eth0")
        self.assertEqual(imports.call_args_list[-1].args[0], "unitree_sdk2py.g1.loco.g1_loco_client")
        channel.ChannelFactoryInitialize.assert_called_once_with(0, "eth0")
        client.Init.assert_called_once_with()
        client.SetVelocity.assert_not_called()

    def test_invalid_lease_fails_before_dds_initialization(self):
        with patch("unitree_adapter.importlib.import_module") as imports:
            for lease in (0, .01, 1, True, float("nan")):
                with self.assertRaises(RobotInterfaceError):
                    G1LocoDriver.connect("eth0", command_lease_s=lease)
            imports.assert_not_called()

    def test_preflight_selects_g1_without_instantiating_clients(self):
        config, connection, path = validate_config({
            "robot": {"device_id": "G1", "model": "g1", "allowed_fsm_ids": [500], "max_tilt_rad": .2},
            "connection": {"network_interface": "eth0"}, "observation_file": "state.json"}, Path("/tmp"))
        self.assertIsInstance(config, G1Config)
        self.assertEqual(path, Path("/tmp/state.json").resolve())
        constructor = Mock()
        modules = [SimpleNamespace(ChannelFactoryInitialize=Mock(), ChannelSubscriber=Mock()),
                   SimpleNamespace(LocoClient=constructor)]
        with patch("robot_preflight.importlib.import_module", side_effect=modules):
            sdk_probe("g1", None)
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
