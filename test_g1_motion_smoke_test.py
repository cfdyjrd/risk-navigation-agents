from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from g1_motion_smoke_test import execute_motion, validate_report


def report(source_ns=1, position=None, velocity=None, fsm=500, battery=95):
    stamp = datetime.now(timezone.utc).isoformat()
    position = [0., 0., .72] if position is None else position
    velocity = [0., 0., 0.] if velocity is None else velocity
    state = {
        "lowstate": {"timestamp": stamp, "source_timestamp_ns": source_ns,
                     "rpy_rad": [0., 0., 0.]},
        "odometry": {"timestamp": stamp, "position_m": position,
                     "velocity_mps": velocity, "yaw_rate_rps": 0., "error_code": 0},
        "battery": {"timestamp": stamp, "soc": battery},
        "fsm": {"timestamp": stamp, "fsm_id": fsm},
    }
    return {"collector": "g1_state_collector", "device_id": "G1",
            "interface": "eth0", "errors": {}, "state": state}


class MotionSmokeTest(unittest.TestCase):
    def test_bad_readiness_rejects_without_driver(self):
        for value in (report(fsm=1), report(battery=49), report(velocity=[.1, 0., 0.])):
            with self.assertRaises(ValueError):
                validate_report(value, "eth0", require_stationary=True)

    def test_fixed_motion_is_stopped_and_measured(self):
        rows = [report(index, position=[0., 0., .72]) for index in range(1, 5)]
        rows += [report(index, position=[.04, 0., .72]) for index in range(5, 30)]
        provider = Mock(side_effect=rows)
        client = SimpleNamespace(SetVelocity=Mock(return_value=0))
        driver = SimpleNamespace(period_s=.05,
                                 move=lambda *args: client.SetVelocity(*args, .2),
                                 stop=lambda: client.SetVelocity(0., 0., 0., .2))
        clock = {"now": 0.}

        def monotonic():
            clock["now"] += .06
            return clock["now"]

        result = execute_motion(driver, provider, "eth0", sleep=lambda _value: None,
                                monotonic=monotonic)
        self.assertTrue(result["movement_detected"])
        nonzero = [call for call in client.SetVelocity.call_args_list if call.args[0] != 0.]
        zero = [call for call in client.SetVelocity.call_args_list if call.args[0] == 0.]
        self.assertGreater(len(nonzero), 0)
        self.assertTrue(all(call.args == (.05, 0., 0., .2) for call in nonzero))
        self.assertEqual(len(zero), 3)

    def test_small_odometry_change_is_not_reported_as_walking(self):
        rows = [report(index, position=[0., 0., .72]) for index in range(1, 5)]
        rows += [report(index, position=[.008, 0., .72]) for index in range(5, 30)]
        provider = Mock(side_effect=rows)
        driver = SimpleNamespace(period_s=.05, move=lambda *_args: None,
                                 stop=lambda: None)
        clock = {"now": 0.}

        def monotonic():
            clock["now"] += .06
            return clock["now"]

        result = execute_motion(driver, provider, "eth0", sleep=lambda _value: None,
                                monotonic=monotonic)
        self.assertFalse(result["movement_detected"])


if __name__ == "__main__":
    unittest.main()
