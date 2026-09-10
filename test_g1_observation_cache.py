from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import struct
import unittest

from g1_observation_cache import build_observation, measured_stationary
from g1_state_collector import parse_fsm, dds_timestamp, state_freshness
from g1_lidar_probe import summarize_cloud
from robot_interface import RobotInterfaceError
from unitree_adapter import G1Config


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc).isoformat()
        self.old = (datetime.now(timezone.utc) - timedelta(seconds=.2)).isoformat()
        self.config = G1Config("G1", allowed_fsm_ids=(500,), max_tilt_rad=.2)
        self.state = {"device_id": "G1", "state": {
            "lowstate": {"timestamp": self.now, "rpy_rad": [0., 0., 0.]},
            "odometry": {"timestamp": self.now, "velocity_mps": [0., 0., 0.],
                         "position_m": [0., 0., .7], "yaw_rate_rps": 0., "error_code": 0},
            "battery": {"timestamp": self.old, "soc": 70},
            "fsm": {"timestamp": self.now, "fsm_id": 500}}}
        self.perception = {"timestamp": self.now, "frame_id": "base_link", "validated": True,
                           "obstacle_detected": True, "obstacle_distance_m": 1., "observation_confidence": .8}

    def build(self):
        return build_observation(self.state, self.perception, self.config, width_m=.6)

    def test_oldest_source_preserved_and_motion_measured(self):
        observation = self.build()
        self.assertEqual(observation.timestamp, self.old)
        self.assertEqual(observation.robot["battery_percent"], 70)
        self.assertTrue(measured_stationary(observation))
        self.state["state"]["odometry"]["velocity_mps"] = [.1, 0., 0.]
        self.assertFalse(measured_stationary(self.build()))

    def test_missing_battery_never_gets_default_value(self):
        del self.state["state"]["battery"]
        with self.assertRaises(RobotInterfaceError):
            self.build()

    def test_stale_source_and_wrong_identity_rejected(self):
        self.state["state"]["battery"]["timestamp"] = "2000-01-01T00:00:00+00:00"
        with self.assertRaises(RobotInterfaceError):
            self.build()
        self.state["state"]["battery"]["timestamp"] = self.now
        self.state["device_id"] = "another"
        with self.assertRaises(RobotInterfaceError):
            self.build()

    def test_freshness_reports_clock_offset_without_rewriting(self):
        self.state["state"]["battery"]["timestamp"] = "2000-01-01T00:00:00+00:00"
        result = state_freshness(self.state["state"])
        self.assertFalse(result["battery"]["fresh"])
        self.assertTrue(result["fsm"]["fresh"])
        self.assertEqual(self.state["state"]["battery"]["timestamp"], "2000-01-01T00:00:00+00:00")

    def test_acquisition_error_or_odometry_error_rejects_observation(self):
        self.state["errors"] = {"fsm": "timeout"}
        with self.assertRaises(RobotInterfaceError):
            self.build()
        self.state["errors"] = {}
        self.state["state"]["odometry"]["error_code"] = 1
        with self.assertRaises(RobotInterfaceError):
            self.build()

    def test_raw_lidar_and_wrong_frame_cannot_be_perception(self):
        for key, value in (("validated", False), ("frame_id", "livox_frame")):
            original = self.perception[key]
            self.perception[key] = value
            with self.assertRaises(RobotInterfaceError):
                self.build()
            self.perception[key] = original

    def test_fsm_rpc_rejects_failure_boolean_and_strings(self):
        self.assertEqual(parse_fsm(0, '{"data":500}'), 500)
        for code, data in ((3104, None), (False, 500), (0, True), (0, '{"data":"500"}')):
            with self.assertRaises(ValueError):
                parse_fsm(code, data)

    def test_dds_stamp_uses_writer_time(self):
        sample = SimpleNamespace(sample_info=SimpleNamespace(source_timestamp=1_000_000_000))
        self.assertEqual(dds_timestamp(sample), "1970-01-01T00:00:01+00:00")

    def test_cloud_respects_endianness_and_row_padding(self):
        msg = SimpleNamespace(
            fields=[SimpleNamespace(name=name, offset=i*4, datatype=7, count=1)
                    for i, name in enumerate(("x", "y", "z"))],
            is_bigendian=True, point_step=12, row_step=16, width=1, height=2,
            data=struct.pack(">fff", 3, 4, 0) + b"pad!" + struct.pack(">fff", 0, 0, 2) + b"pad!",
            header=SimpleNamespace(frame_id="lidar", stamp=SimpleNamespace(sec=1, nanosec=2)))
        report = summarize_cloud(msg)
        self.assertEqual(report["finite_points"], 2)
        self.assertEqual(report["nearest_raw_return_m"], 2)
        self.assertFalse(report["validated_obstacle_perception"])
        msg.data = b""
        with self.assertRaises(ValueError):
            summarize_cloud(msg)


if __name__ == "__main__":
    unittest.main()
