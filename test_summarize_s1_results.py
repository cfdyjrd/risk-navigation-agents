import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from summarize_s1_results import S1SummaryError, aggregate_rows, load_rows


def _summary(layout, condition, repeat, *, completed=True, near=0, clearance=.12,
             stops=0, guard=0, elapsed=10.0):
    return {
        "trial_id": f"S1_{layout}_{condition}_R{repeat}",
        "layout": layout,
        "condition": condition,
        "repeat": repeat,
        "formal_trial": True,
        "status": "completed" if completed else "stopped_incomplete",
        "task_completed": completed,
        "elapsed_s": elapsed,
        "metrics": {
            "near_miss_events": near,
            "minimum_envelope_clearance_m": clearance,
            "high_level_stop_decisions": stops,
            "guard_interventions": guard,
            "pre_execution_guard_interventions": guard,
            "runtime_guard_interventions": 0,
            "maximum_abs_lateral_m": .01,
            "minimum_battery_percent": 75,
            "reobservations": 0,
            "motion_chunks": 10,
            "high_level_decisions": 2,
        },
    }


def _write_trial(root, payload):
    directory = root / payload["trial_id"]
    directory.mkdir()
    config_bytes = b'{"test_config":true}\n'
    (directory / "config.source.json").write_bytes(config_bytes)
    (directory / "config.snapshot.json").write_text("{}\n")
    condition = payload["condition"]
    inputs = {
        "memory_mode": (
            "disabled" if condition == "B0"
            else "role_conditioned" if condition == "M"
            else "shared_bundle"
        ),
        "rule_mode": "disabled" if condition == "B0" else "enabled",
        "memory_sha256": None,
        "rule_sha256": None,
    }
    if condition != "B0":
        memory_bytes = b"[]\n"
        rule_bytes = b'{"schema_version":1,"rules":[]}\n'
        (directory / "memory.snapshot.json").write_bytes(memory_bytes)
        (directory / "rules.snapshot.json").write_bytes(rule_bytes)
        inputs["memory_sha256"] = hashlib.sha256(memory_bytes).hexdigest()
        inputs["rule_sha256"] = hashlib.sha256(rule_bytes).hexdigest()
    payload["input_hashes"] = inputs
    payload["setup_fingerprint"] = "f" * 64
    manifest = {
        "trial_id": payload["trial_id"],
        "layout": payload["layout"],
        "condition": condition,
        "repeat": payload["repeat"],
        "formal_trial": True,
        "setup_fingerprint": payload["setup_fingerprint"],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "condition_inputs": inputs,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "events.jsonl").write_text(json.dumps({
        "event": "trial_finished", "summary": payload,
    }) + "\n")
    (directory / "summary.json").write_text(json.dumps(payload))
    return directory


class S1SummaryTests(unittest.TestCase):
    def test_complete_primary_protocol_and_pair_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for layout in ("A", "B"):
                for repeat in range(1, 4):
                    for condition in ("B0", "M"):
                        payload = _summary(
                            layout, condition, repeat,
                            near=1 if condition == "B0" and layout == "B" else 0,
                            clearance=.11 if condition == "B0" else .14,
                            stops=1 if condition == "M" and layout == "A" else 0,
                        )
                        _write_trial(root, payload)
            rows = load_rows(root)
            report = aggregate_rows(rows)
        self.assertTrue(report["primary_protocol_complete"])
        self.assertEqual(report["formal_trial_count"], 12)
        self.assertEqual(report["groups"]["B"]["B0"]["near_miss_events_total"], 3)
        self.assertEqual(report["paired_m_minus_b0"]["B"]["pair_count"], 3)
        self.assertAlmostEqual(
            report["paired_m_minus_b0"]["B"]["mean_m_minus_b0_minimum_clearance_m"],
            .03,
        )

    def test_missing_trial_is_explicit(self):
        row = {
            "trial_id": "S1_A_B0_R1", "layout": "A", "condition": "B0",
            "repeat": 1, "status": "completed", "task_completed": True,
            "elapsed_s": 1.0, "near_miss_events": 0,
            "minimum_envelope_clearance_m": .2,
            "high_level_stop_decisions": 0, "guard_interventions": 0,
            "pre_execution_guard_interventions": 0,
            "runtime_guard_interventions": 0,
            "maximum_abs_lateral_m": .01, "minimum_battery_percent": 75,
            "reobservations": 0, "motion_chunks": 1,
            "high_level_decisions": 1, "source": "unused",
        }
        report = aggregate_rows([row])
        self.assertFalse(report["primary_protocol_complete"])
        self.assertIn("S1_B_M_R3", report["missing_primary_trial_ids"])

    def test_directory_and_trial_identity_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "S1_A_B0_R1"
            directory.mkdir()
            payload = _summary("A", "M", 1)
            (directory / "summary.json").write_text(json.dumps(payload))
            with self.assertRaises(S1SummaryError):
                load_rows(root)

    def test_b1_is_excluded_unless_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _summary("B", "B1", 1)
            _write_trial(root, payload)
            self.assertEqual(load_rows(root), [])
            self.assertEqual(len(load_rows(root, include_b1=True)), 1)

    def test_modified_frozen_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = _write_trial(root, _summary("B", "M", 1))
            (directory / "memory.snapshot.json").write_text("[]")
            with self.assertRaises(S1SummaryError):
                load_rows(root)

    def test_negative_intrusion_clearance_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_trial(
                root,
                _summary(
                    "B", "B0", 1, completed=False,
                    near=1, clearance=-.015,
                ),
            )
            rows = load_rows(root)
        self.assertEqual(rows[0]["minimum_envelope_clearance_m"], -.015)


if __name__ == "__main__":
    unittest.main()
