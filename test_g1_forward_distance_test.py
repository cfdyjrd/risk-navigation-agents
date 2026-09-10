from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from g1_forward_distance_test import (
    execute_distance_motion,
    validate_motion_envelope,
)


def report(source_ns, x=0., y=0., velocity=None, yaw=0.):
    stamp = datetime.now(timezone.utc).isoformat()
    velocity = [0., 0., 0.] if velocity is None else velocity
    state = {
        "lowstate": {"timestamp": stamp, "source_timestamp_ns": source_ns,
                     "rpy_rad": [0., 0., yaw]},
        "odometry": {"timestamp": stamp, "source_timestamp_ns": source_ns,
                     "position_m": [x, y, .72], "velocity_mps": velocity,
                     "yaw_rate_rps": 0., "error_code": 0},
        "battery": {"timestamp": stamp, "source_timestamp_ns": source_ns,
                    "soc": 87},
        "fsm": {"timestamp": stamp, "fsm_id": 500},
    }
    return {"collector": "g1_state_collector", "device_id": "G1",
            "interface": "eth0", "errors": {}, "state": state}


class ForwardDistanceTest(unittest.TestCase):
    def driver(self):
        client = SimpleNamespace(SetVelocity=Mock(return_value=0))
        driver = SimpleNamespace(period_s=.05,
                                 move=lambda *args: client.SetVelocity(*args, .2),
                                 stop=lambda: client.SetVelocity(0., 0., 0., .2))
        return client, driver

    def test_stops_near_one_metre(self):
        rows = [report(1), report(2)]
        positions = (0., .02, .10, .19, .28, .37, .46, .55,
                     .64, .73, .82, .90, .96)
        rows += [report(index, x=x, velocity=[.15, 0., 0.]) for index, x in
                 enumerate(positions, start=3)]
        rows += [report(16, x=1.01), report(17, x=1.01)]
        provider = Mock(side_effect=rows)
        client, driver = self.driver()
        clock = {"now": 0.}

        def monotonic():
            clock["now"] += .01
            return clock["now"]

        result = execute_distance_motion(
            driver, provider, "eth0", sleep=lambda _value: None,
            monotonic=monotonic)
        self.assertEqual(result["status"], "completed")
        self.assertAlmostEqual(result["measured_forward_m"], 1.01)
        nonzero = [call for call in client.SetVelocity.call_args_list
                   if call.args[0] != 0.]
        zero = [call for call in client.SetVelocity.call_args_list
                if call.args[0] == 0.]
        self.assertGreater(len(nonzero), 0)
        self.assertTrue(all(call.args == (.15, 0., 0., .2) for call in nonzero))
        self.assertEqual(len(zero), 3)

    def test_envelope_rejects_lateral_motion_and_no_progress(self):
        start = {"yaw_rad": 0., "position_m": [0., 0., .72]}
        base = {"yaw_rad": 0., "position_m": [0., .16, .72],
                "velocity_mps": [0., 0., 0.], "yaw_rate_rps": 0.}
        with self.assertRaisesRegex(ValueError, "lateral"):
            validate_motion_envelope(start, base, [0., 0., .72], .5)
        base["position_m"] = [.01, 0., .72]
        with self.assertRaisesRegex(ValueError, "no demonstrated"):
            validate_motion_envelope(start, base, [0., 0., .72], 2.1)

    def test_no_progress_fails_closed_with_three_stops(self):
        rows = [report(1), report(2)]
        rows += [report(index, x=0., velocity=[0., 0., 0.])
                 for index in range(3, 9)]
        provider = Mock(side_effect=rows)
        client, driver = self.driver()
        clock = {"now": 0.}

        def monotonic():
            clock["now"] += .5
            return clock["now"]

        with self.assertRaisesRegex(ValueError, "no demonstrated"):
            execute_distance_motion(
                driver, provider, "eth0", sleep=lambda _value: None,
                monotonic=monotonic)
        nonzero = [call for call in client.SetVelocity.call_args_list
                   if call.args[0] != 0.]
        zero = [call for call in client.SetVelocity.call_args_list
                if call.args[0] == 0.]
        self.assertGreater(len(nonzero), 0)
        self.assertEqual(len(zero), 3)


if __name__ == "__main__":
    unittest.main()
