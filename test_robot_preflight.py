"""Preflight tests run offline and never load a real vendor SDK."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from robot_preflight import check_sdk, main, run_preflight, validate_config, write_observation_snapshot
from robot_interface import RobotObservation, RobotInterfaceError
from unitree_adapter import UnitreeConfig


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / "config.json"
        self.snapshot = self.root / "state.json"
        self.config = {"robot": {"device_id": "Go2-1", "model": "go2"},
                       "connection": {"network_interface": "test0"}, "observation_file": "state.json"}
        self.path.write_text(json.dumps(self.config))
        self.source = self.make_observation()
        self.snapshot.write_text(json.dumps(self.source))

    def make_observation(self, age=0):
        return {"robot": {"device_id": "Go2-1", "type": "go2", "width_m": .4, "battery_percent": 80},
                "environment": {"obstacle_detected": False, "obstacle_distance_m": None,
                                "observation_confidence": .9},
                "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()}

    def run_check(self, source=None, skip_sdk=False):
        def read(path):
            if Path(path).resolve() == self.path:
                return self.config
            return source() if source else self.make_observation()
        with patch("robot_preflight.read_json", side_effect=read), \
             patch("robot_preflight.socket.if_nameindex", return_value=[(1, "test0")]), \
             patch("robot_preflight.check_sdk", return_value=("pass", {"version": "fake"})), \
             patch("unitree_adapter.Go2SportDriver.connect", side_effect=AssertionError("must not connect")), \
             patch("unitree_adapter.Go1UDPDriver.connect", side_effect=AssertionError("must not connect")):
            return run_preflight(self.path, duration=.1, interval=.02, skip_sdk=skip_sdk)

    def test_passing_preflight_is_not_connection_or_motion_approval(self):
        report = self.run_check()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["robot_connectivity"], "not_verified")
        self.assertFalse(report["motion_ready"])
        self.assertEqual(report["motion_commands_sent"], 0)
        self.assertGreaterEqual(report["checks"][-1]["detail"]["distinct_samples"], 2)

    def test_frozen_but_fresh_timestamp_fails(self):
        report = self.run_check(lambda: self.source)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["checks"][-1]["detail"]["distinct_samples"], 1)

    def test_stale_wrong_identity_and_invalid_values_fail(self):
        for change in ("stale", "identity", "nan", "future"):
            with self.subTest(change=change):
                sample = self.make_observation(age=10 if change == "stale" else -10 if change == "future" else 0)
                if change == "identity":
                    sample["robot"]["device_id"] = "other"
                if change == "nan":
                    sample["robot"]["battery_percent"] = float("nan")
                report = self.run_check(lambda: sample)
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["checks"][-1]["detail"]["valid_polls"], 0)

    def test_timestamp_regression_and_disconnect_fail(self):
        count = 0
        def source():
            nonlocal count
            count += 1
            if count == 1:
                return self.make_observation()
            if count == 2:
                return self.make_observation(age=.1)
            raise FileNotFoundError("sensor stopped")
        result = self.run_check(source)
        errors = str(result["checks"][-1]["detail"]["errors"])
        self.assertIn("backwards", errors)
        self.assertIn("sensor stopped", errors)
        self.assertEqual(result["status"], "fail")

    def test_skip_is_not_pass(self):
        self.assertEqual(self.run_check(skip_sdk=True)["status"], "incomplete")

    def test_missing_interface_fails(self):
        self.config["connection"]["network_interface"] = "missing0"
        self.assertEqual(self.run_check()["status"], "fail")

    def test_go1_config_and_model_separation(self):
        data = {"robot": {"device_id": "Go1", "model": "go1"},
                "connection": {"robot_ip": "192.0.2.1", "sdk_extension": "/tmp/robot_interface.so"},
                "observation_file": "state.json"}
        config, _, path = validate_config(data, self.root)
        self.assertEqual(config.model, "go1")
        self.assertEqual(path, self.snapshot)
        data["connection"]["network_interface"] = "test0"
        with self.assertRaises(ValueError):
            validate_config(data, self.root)

    def test_incomplete_config_yields_report(self):
        self.config["connection"]["network_interface"] = None
        result = self.run_check()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checks"][0]["name"], "configuration")

    def test_missing_snapshot_fails(self):
        with patch("robot_preflight.socket.if_nameindex", return_value=[(1, "test0")]):
            self.snapshot.unlink()
            report = run_preflight(self.path, duration=.1, interval=.02, skip_sdk=True)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["checks"][-1]["detail"]["valid_polls"], 0)

    def test_sdk_subprocess_timeout_and_import_failure(self):
        config = UnitreeConfig("Go2-1", "go2")
        with patch("robot_preflight.subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 10)):
            status, detail = check_sdk(config, {})
            self.assertEqual(status, "fail")
            self.assertIn("timed out", detail["error"])
        with patch("robot_preflight.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "missing SDK")):
            self.assertEqual(check_sdk(config, {})[0], "fail")

    def test_sdk_probe_never_initializes_clients(self):
        from types import ModuleType
        from robot_preflight import sdk_probe
        channel = ModuleType("channel")
        channel.ChannelFactoryInitialize = lambda *a: self.fail("must not initialize DDS")
        channel.ChannelSubscriber = lambda *a: self.fail("must not subscribe")
        sport = ModuleType("sport")
        sport.SportClient = lambda *a: self.fail("must not create control client")
        with patch.dict("sys.modules", {"unitree_sdk2py.core.channel": channel,
                                      "unitree_sdk2py.go2.sport.sport_client": sport}):
            self.assertEqual(sdk_probe("go2", None)["module"], "unitree_sdk2py")
        legacy = ModuleType("legacy")
        legacy.UDP = lambda *a: self.fail("must not create UDP")
        legacy.HighCmd = lambda: self.fail("must not create commands")
        legacy.HighState = lambda: None
        with patch("robot_preflight.load_go1_sdk", return_value=legacy):
            self.assertIn("module", sdk_probe("go1", "/tmp/fake.so"))

    def test_output_cannot_overwrite_inputs(self):
        for target in (self.path, self.snapshot):
            before = target.read_bytes()
            with redirect_stderr(io.StringIO()):
                code = main(["--config", str(self.path), "--output", str(target)])
            self.assertEqual(code, 2)
            self.assertEqual(target.read_bytes(), before)

    def test_cli_report_and_exit_status(self):
        output = self.root / "report.json"
        with patch("robot_preflight.run_preflight", return_value={"status": "incomplete"}), redirect_stdout(io.StringIO()):
            code = main(["--config", str(self.path), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.read_text())["status"], "incomplete")

    def test_invalid_monitor_parameters(self):
        for kwargs in ({"duration": float("nan")}, {"interval": 0}, {"duration": 1, "interval": 1}):
            with self.assertRaises(ValueError):
                run_preflight(self.path, **kwargs)

    def test_atomic_export_preserves_acquisition_time_and_existing_file_on_error(self):
        observation = RobotObservation(**self.source)
        write_observation_snapshot(self.snapshot, observation)
        self.assertEqual(json.loads(self.snapshot.read_text())["timestamp"], self.source["timestamp"])
        before = self.snapshot.read_bytes()
        observation.robot["battery_percent"] = float("nan")
        with self.assertRaises(RobotInterfaceError):
            write_observation_snapshot(self.snapshot, observation)
        self.assertEqual(self.snapshot.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".observation-*")), [])


if __name__ == "__main__":
    unittest.main()
