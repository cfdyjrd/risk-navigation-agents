"""Offline tests: no vendor SDK import, network connection, or robot motion."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

from execution_bridge import SafeExecutionBridge
from robot_interface import RobotInterfaceError, RobotObservation
from unitree_adapter import Go1UDPDriver, Go2SportDriver, UnitreeConfig, UnitreeRobotAdapter


def observation(device_id="Go2-1", model="go2", age=0, **environment):
    return RobotObservation(
        {"device_id": device_id, "type": model, "width_m": 0.4, "battery_percent": 80},
        {"obstacle_detected": False, "obstacle_distance_m": None,
         "observation_confidence": 0.9, **environment},
        (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat())


class FakeDriver:
    model = "go2"
    period_s = 0.002

    def __init__(self):
        self.calls = []
        self.fail_move = False
        self.fail_stop = False
        self.moving = threading.Event()

    def move(self, *args):
        self.calls.append(("move", args))
        self.moving.set()
        if self.fail_move:
            raise RuntimeError("connection lost")

    def stop(self):
        self.calls.append(("stop",))
        if self.fail_stop:
            raise RuntimeError("stop unavailable")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.driver = FakeDriver()
        self.adapter = UnitreeRobotAdapter(UnitreeConfig("Go2-1", "go2"), self.driver, observation)

    def move(self, name="move_forward", **parameters):
        return self.adapter.execute({"name": name, "parameters": {"duration_s": 0.008, **parameters}})

    def test_move_and_stop_without_claiming_arrival(self):
        result = self.move()
        self.assertEqual(result.status, "success")
        self.assertEqual(self.driver.calls[0], ("move", (0.3, 0, 0)))
        self.assertEqual(self.driver.calls[-1], ("stop",))
        self.assertFalse(result.telemetry["physical_completion_verified"])
        self.assertEqual(result.telemetry["device_id"], "Go2-1")

    def test_turn_directions_and_slow_speed(self):
        for name, expected in (("turn_left", (0, 0, .5)), ("turn_right", (0, 0, -.5)), ("slow_down", (.1, 0, 0))):
            self.driver.calls.clear()
            self.move(name)
            self.assertEqual(self.driver.calls[0], ("move", expected))

    def test_distance_and_angle_units(self):
        self.assertAlmostEqual(self.adapter._motion("move_forward", {"distance_m": .3, "speed_mps": .2})[-1], 1.5)
        self.assertEqual(self.adapter._motion("turn_left", {"angle_rad": .5}), (0, 0, .5, 1))

    def test_slow_observation_cannot_send_an_expired_command(self):
        def delayed_snapshot():
            time.sleep(.015)
            return observation()
        self.adapter.observation_provider = delayed_snapshot
        result = self.move(duration_s=.005)
        self.assertEqual(result.status, "failure")
        self.assertEqual(self.driver.calls, [("stop",)])

    def test_go1_identity_reaches_same_bridge(self):
        self.driver.model = "go1"
        self.adapter = UnitreeRobotAdapter(UnitreeConfig("Go1", "go1"), self.driver,
                                           lambda: observation("Go1", "go1"))
        result = self.move()
        self.assertEqual(result.status, "success")
        self.assertEqual(result.telemetry["model"], "go1")

    def test_bad_parameters_never_move(self):
        for params in ({"speed_mps": float("nan")}, {"speed_mps": True}, {"speed_mps": -1},
                       {"speed_mps": .4}, {"duration_s": 20}, {"duration_s": 0},
                       {"distance_m": 1, "duration_s": 1}, {"speed": .1},
                       {"distance_m": .4, "speed_mps": .01}):
            with self.subTest(params=params), self.assertRaises(RobotInterfaceError):
                self.adapter.execute({"name": "move_forward", "parameters": params})
        self.assertEqual(self.driver.calls, [])

    def test_mismatched_model_rejected(self):
        with self.assertRaises(RobotInterfaceError):
            UnitreeRobotAdapter(UnitreeConfig("Go1", "go1"), self.driver, observation)

    def test_stale_wrong_device_and_future_state_stop(self):
        for kwargs in ({"age": 5}, {"age": -1}, {"device_id": "another-dog"}, {"model": "go1"}):
            self.adapter = UnitreeRobotAdapter(UnitreeConfig("Go2-1", "go2"), self.driver,
                                               lambda: observation(**kwargs))
            self.driver.calls.clear()
            result = self.move()
            self.assertEqual(result.status, "failure")
            self.assertEqual(self.driver.calls, [("stop",)])

    def test_obstacle_appearing_during_motion_stops(self):
        self.adapter.observation_provider = lambda: observation(
            obstacle_detected=bool(self.driver.calls), obstacle_distance_m=.1 if self.driver.calls else None)
        result = self.move()
        self.assertEqual(result.status, "failure")
        self.assertEqual(len([c for c in self.driver.calls if c[0] == "move"]), 1)
        self.assertEqual(self.driver.calls[-1], ("stop",))

    def test_required_side_geometry_loss_during_motion_stops(self):
        def snapshot():
            if self.driver.calls:
                return observation(
                    envelope_clearance_required=True,
                    corridor_geometry_valid=False,
                )
            return observation(
                corridor_width_m=.70,
                corridor_geometry_valid=True,
                envelope_clearance_required=True,
                maximum_corridor_heading_error_rad=.05,
                left_envelope_clearance_m=.15,
                right_envelope_clearance_m=.15,
                clearance_uncertainty_m=.02,
                minimum_envelope_clearance_m=.13,
                corridor_heading_error_rad=0.,
            )
        self.adapter.observation_provider = snapshot
        result = self.move()
        self.assertEqual(result.status, "failure")
        self.assertIn("corridor_geometry_unavailable", result.telemetry["failure_reason"])
        self.assertEqual(len([call for call in self.driver.calls if call[0] == "move"]), 1)
        self.assertEqual(self.driver.calls[-1], ("stop",))

    def test_failed_sdk_call_stops_and_latches(self):
        self.driver.fail_move = True
        result = self.move()
        self.assertEqual(result.status, "failure")
        self.assertEqual(self.driver.calls[-1], ("stop",))
        self.driver.fail_move = False
        self.driver.calls.clear()
        self.assertEqual(self.move().status, "failure")
        self.assertEqual(self.driver.calls, [("stop",)])
        self.adapter.reset_emergency_stop()
        self.assertEqual(self.move().status, "success")

    def test_stop_failure_is_reported(self):
        self.driver.fail_stop = True
        self.assertEqual(self.move().status, "failure")
        result = self.adapter.emergency_stop("operator")
        self.assertEqual(result.status, "failure")
        self.assertFalse(result.telemetry["stop_submitted"])

    def test_emergency_stop_interrupts_and_prevents_later_motion(self):
        results = []
        thread = threading.Thread(target=lambda: results.append(self.move(duration_s=.5)))
        thread.start()
        self.assertTrue(self.driver.moving.wait(1))
        with self.assertRaises(RobotInterfaceError):
            self.move()
        self.adapter.emergency_stop("operator")
        count = len(self.driver.calls)
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertIn(results[0].status, {"aborted", "failure"})
        self.assertTrue(all(call[0] == "stop" for call in self.driver.calls[count:]))

    def test_stationary_requests_do_not_claim_human_response(self):
        for name in ("observe_again", "ask_human"):
            result = self.adapter.execute({"name": name})
            self.assertEqual(result.status, "aborted")
            self.assertEqual(result.telemetry["requested_action"], name)
        self.assertTrue(all(c[0] == "stop" for c in self.driver.calls))

    def test_bridge_uses_adapter_and_guard_override(self):
        self.adapter.observation_provider = lambda: observation(obstacle_detected=True, obstacle_distance_m=.1)
        bridge = SafeExecutionBridge(self.adapter, available_actions=["move_forward", "safe_stop"])
        result = bridge.execute_decision({"goal": "test"}, {
            "decision_source": "deterministic_constrained_optimizer",
            "selected_action": {"name": "move_forward", "parameters": {}}})
        self.assertEqual(result["approved_action"]["name"], "safe_stop")
        self.assertEqual(self.driver.calls, [("stop",)])

    def test_bridge_reports_emergency_stop_exception(self):
        with patch.object(self.adapter, "emergency_stop", side_effect=RuntimeError("offline")):
            result = SafeExecutionBridge(self.adapter, available_actions=["safe_stop"]).execute_decision({}, {})
        self.assertEqual(result["status"], "fail_closed")
        self.assertEqual(result["receipt"]["status"], "failure")
        self.assertFalse(result["receipt"]["telemetry"]["stop_submitted"])

    def test_nonfinite_and_boolean_sensor_values_rejected(self):
        for kwargs in ({"observation_confidence": True}, {"obstacle_distance_m": float("nan")},
                       {"obstacle_detected": "false"}, {"corridor_width_m": float("inf")},
                       {"obstacle_detected": True, "obstacle_distance_m": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RobotInterfaceError):
                observation(**kwargs).validate()
        observation().validate()


class DriverTests(unittest.TestCase):
    def test_go2_return_code(self):
        client = SimpleNamespace(Move=lambda *args: 0, StopMove=lambda: 0)
        driver = Go2SportDriver(client)
        driver.move(.1, 0, 0)
        driver.stop()
        for code in (1, None, False):
            client.Move = lambda *args: code
            with self.assertRaises(RobotInterfaceError):
                driver.move(.1, 0, 0)

    def test_go1_high_level_fields_and_zero_stop(self):
        sent = []
        command = SimpleNamespace()
        udp = SimpleNamespace(InitCmdData=lambda cmd: None,
                              SetSend=lambda cmd: sent.append((cmd.mode, list(cmd.velocity), cmd.yawSpeed)),
                              Send=lambda: None)
        driver = Go1UDPDriver(udp, command)
        driver.move(.2, 0, -.3)
        with patch("unitree_adapter.time.sleep"):
            driver.stop()
        self.assertEqual(sent[0], (2, [.2, 0], -.3))
        self.assertEqual(len(sent), 11)
        self.assertTrue(all(row == (2, [0, 0], 0) for row in sent[1:]))


if __name__ == "__main__":
    unittest.main()
