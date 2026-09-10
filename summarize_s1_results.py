"""Validate and aggregate completed S1 formal-trial summaries.

This command never imports the Unitree driver and cannot send robot commands.
It treats B0/M as the preregistered primary comparison; B1 can be included as
an explicitly labelled exploratory condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from g1_state_collector import atomic_json


PRIMARY_CONDITIONS = ("B0", "M")
ALL_CONDITIONS = ("B0", "M", "B1")
LAYOUTS = ("A", "B")
_TRIAL_ID = re.compile(r"S1_(A|B)_(B0|M|B1)_R([1-9][0-9]*)")


class S1SummaryError(ValueError):
    """Raised when a result cannot be safely included in an aggregate."""


def _finite_number(value: Any, field: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise S1SummaryError(f"{field} must be a finite number")
    return float(value)


def _nonnegative_integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise S1SummaryError(f"{field} must be a non-negative integer")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise S1SummaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise S1SummaryError(f"{path}: JSON root must be an object")
    return value


def _validate_artifacts(row: dict[str, Any], summary: dict[str, Any], source: Path) -> None:
    directory = source.parent
    manifest_path = directory / "manifest.json"
    manifest = _read_object(manifest_path)
    for key in ("trial_id", "layout", "condition", "repeat"):
        if manifest.get(key) != row[key]:
            raise S1SummaryError(f"{manifest_path}: {key} disagrees with summary")
    if manifest.get("formal_trial") is not True:
        raise S1SummaryError(f"{manifest_path}: formal_trial must be true")
    setup = manifest.get("setup_fingerprint")
    if not isinstance(setup, str) or not re.fullmatch(r"[0-9a-f]{64}", setup):
        raise S1SummaryError(f"{manifest_path}: setup_fingerprint is invalid")
    if summary.get("setup_fingerprint") != setup:
        raise S1SummaryError(f"{directory}: setup fingerprint is inconsistent")
    row["setup_fingerprint"] = setup

    config_source = directory / "config.source.json"
    config_snapshot = directory / "config.snapshot.json"
    if not config_source.is_file() or not config_snapshot.is_file():
        raise S1SummaryError(f"{directory}: frozen config artifacts are missing")
    expected_config_hash = manifest.get("config_sha256")
    if not isinstance(expected_config_hash, str) or _sha256(config_source) != expected_config_hash:
        raise S1SummaryError(f"{directory}: frozen config SHA-256 mismatch")
    row["config_sha256"] = expected_config_hash

    inputs = manifest.get("condition_inputs")
    if not isinstance(inputs, dict) or summary.get("input_hashes") != inputs:
        raise S1SummaryError(f"{directory}: condition input evidence is inconsistent")
    row["memory_sha256"] = inputs.get("memory_sha256")
    row["rule_sha256"] = inputs.get("rule_sha256")
    memory_snapshot = directory / "memory.snapshot.json"
    rule_snapshot = directory / "rules.snapshot.json"
    if row["condition"] == "B0":
        if inputs.get("memory_mode") != "disabled" or inputs.get("rule_mode") != "disabled":
            raise S1SummaryError(f"{directory}: B0 memory/rule modes must be disabled")
        if inputs.get("memory_sha256") is not None or inputs.get("rule_sha256") is not None:
            raise S1SummaryError(f"{directory}: B0 must have null memory/rule hashes")
        if memory_snapshot.exists() or rule_snapshot.exists():
            raise S1SummaryError(f"{directory}: B0 must not contain memory/rule snapshots")
    else:
        expected_memory_mode = (
            "role_conditioned" if row["condition"] == "M" else "shared_bundle"
        )
        if inputs.get("memory_mode") != expected_memory_mode or inputs.get("rule_mode") != "enabled":
            raise S1SummaryError(f"{directory}: condition evidence mode is incorrect")
        if not memory_snapshot.is_file() or not rule_snapshot.is_file():
            raise S1SummaryError(f"{directory}: M/B1 frozen evidence is missing")
        if _sha256(memory_snapshot) != inputs.get("memory_sha256"):
            raise S1SummaryError(f"{directory}: frozen memory SHA-256 mismatch")
        if _sha256(rule_snapshot) != inputs.get("rule_sha256"):
            raise S1SummaryError(f"{directory}: frozen rule SHA-256 mismatch")

    events = directory / "events.jsonl"
    try:
        lines = [line for line in events.read_text(encoding="utf-8").splitlines() if line]
        final_event = json.loads(lines[-1])
    except (OSError, ValueError, IndexError) as exc:
        raise S1SummaryError(f"{events}: missing or invalid terminal audit event") from exc
    if not isinstance(final_event, dict) or final_event.get("event") != "trial_finished":
        raise S1SummaryError(f"{events}: last audit event must be trial_finished")
    logged_summary = final_event.get("summary")
    if not isinstance(logged_summary, dict):
        raise S1SummaryError(f"{events}: terminal audit event lacks its summary")
    for key in (
        "status", "reason", "started_at", "finished_at", "elapsed_s",
        "layout", "condition", "steps", "deliberations", "task_completed",
        "final_progress", "metrics", "terminal_stop",
    ):
        if logged_summary.get(key) != summary.get(key):
            raise S1SummaryError(f"{directory}: summary disagrees with terminal audit field {key}")


def _validated_row(summary: dict[str, Any], source: Path) -> dict[str, Any]:
    identifier = summary.get("trial_id")
    match = _TRIAL_ID.fullmatch(identifier) if isinstance(identifier, str) else None
    if not match:
        raise S1SummaryError(f"{source}: invalid or missing trial_id")
    layout, condition, repeat_text = match.groups()
    repeat = int(repeat_text)
    if source.parent.name != identifier:
        raise S1SummaryError(f"{source}: directory name must equal trial_id")
    if summary.get("formal_trial") is not True:
        raise S1SummaryError(f"{source}: formal_trial must be true")
    if summary.get("layout") != layout or summary.get("condition") != condition:
        raise S1SummaryError(f"{source}: trial_id disagrees with layout/condition")
    if summary.get("repeat") != repeat:
        raise S1SummaryError(f"{source}: trial_id disagrees with repeat")
    if type(summary.get("task_completed")) is not bool:
        raise S1SummaryError(f"{source}: task_completed must be boolean")
    status = summary.get("status")
    if not isinstance(status, str) or not status:
        raise S1SummaryError(f"{source}: status must be a non-empty string")
    if summary["task_completed"] != (status == "completed"):
        raise S1SummaryError(f"{source}: task_completed disagrees with status")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise S1SummaryError(f"{source}: metrics must be an object")

    elapsed = _finite_number(summary.get("elapsed_s"), "elapsed_s")
    if elapsed < 0:
        raise S1SummaryError(f"{source}: elapsed_s must be non-negative")
    clearance = _finite_number(
        metrics.get("minimum_envelope_clearance_m"),
        "metrics.minimum_envelope_clearance_m",
        optional=True,
    )

    guard = _nonnegative_integer(
        metrics.get("guard_interventions"), "metrics.guard_interventions"
    )
    pre_guard = _nonnegative_integer(
        metrics.get("pre_execution_guard_interventions"),
        "metrics.pre_execution_guard_interventions",
    )
    runtime_guard = _nonnegative_integer(
        metrics.get("runtime_guard_interventions"),
        "metrics.runtime_guard_interventions",
    )
    if guard != pre_guard + runtime_guard:
        raise S1SummaryError(f"{source}: Guard intervention subtotals are inconsistent")
    maximum_lateral = _finite_number(
        metrics.get("maximum_abs_lateral_m"), "metrics.maximum_abs_lateral_m"
    )
    if maximum_lateral < 0:
        raise S1SummaryError(f"{source}: maximum_abs_lateral_m must be non-negative")
    minimum_battery = _finite_number(
        metrics.get("minimum_battery_percent"),
        "metrics.minimum_battery_percent",
        optional=True,
    )
    if minimum_battery is not None and not 0 <= minimum_battery <= 100:
        raise S1SummaryError(f"{source}: minimum battery must be within [0, 100]")

    return {
        "trial_id": identifier,
        "layout": layout,
        "condition": condition,
        "repeat": repeat,
        "status": status,
        "task_completed": summary["task_completed"],
        "elapsed_s": elapsed,
        "near_miss_events": _nonnegative_integer(
            metrics.get("near_miss_events"), "metrics.near_miss_events"
        ),
        "minimum_envelope_clearance_m": clearance,
        "high_level_stop_decisions": _nonnegative_integer(
            metrics.get("high_level_stop_decisions"),
            "metrics.high_level_stop_decisions",
        ),
        "guard_interventions": guard,
        "pre_execution_guard_interventions": pre_guard,
        "runtime_guard_interventions": runtime_guard,
        "maximum_abs_lateral_m": maximum_lateral,
        "minimum_battery_percent": minimum_battery,
        "reobservations": _nonnegative_integer(
            metrics.get("reobservations"), "metrics.reobservations"
        ),
        "motion_chunks": _nonnegative_integer(
            metrics.get("motion_chunks"), "metrics.motion_chunks"
        ),
        "high_level_decisions": _nonnegative_integer(
            metrics.get("high_level_decisions"), "metrics.high_level_decisions"
        ),
        "source": str(source),
    }


def load_rows(results_dir: str | Path, *, include_b1: bool = False) -> list[dict[str, Any]]:
    root = Path(results_dir).expanduser().resolve()
    if not root.is_dir():
        raise S1SummaryError(f"results directory does not exist: {root}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("S1_*_R*/summary.json")):
        summary = _read_object(path)
        if summary.get("formal_trial") is not True:
            continue
        row = _validated_row(summary, path)
        if row["condition"] == "B1" and not include_b1:
            continue
        _validate_artifacts(row, summary, path)
        if row["trial_id"] in seen:
            raise S1SummaryError(f"duplicate trial_id: {row['trial_id']}")
        seen.add(row["trial_id"])
        rows.append(row)
    return rows


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return round(statistics.fmean(materialized), 6) if materialized else None


def _group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clearances = [
        row["minimum_envelope_clearance_m"] for row in rows
        if row["minimum_envelope_clearance_m"] is not None
    ]
    batteries = [
        row["minimum_battery_percent"] for row in rows
        if row["minimum_battery_percent"] is not None
    ]
    completed = sum(row["task_completed"] for row in rows)
    return {
        "trial_count": len(rows),
        "completion_count": completed,
        "completion_rate": round(completed / len(rows), 6) if rows else None,
        "near_miss_events_total": sum(row["near_miss_events"] for row in rows),
        "trials_with_near_miss": sum(row["near_miss_events"] > 0 for row in rows),
        "minimum_clearance_across_trials_m": min(clearances) if clearances else None,
        "mean_trial_minimum_clearance_m": _mean(clearances),
        "high_level_stop_decisions_total": sum(
            row["high_level_stop_decisions"] for row in rows
        ),
        "trials_with_high_level_stop": sum(
            row["high_level_stop_decisions"] > 0 for row in rows
        ),
        "elapsed_s_total": round(sum(row["elapsed_s"] for row in rows), 6),
        "elapsed_s_mean": _mean(row["elapsed_s"] for row in rows),
        "guard_interventions_total": sum(row["guard_interventions"] for row in rows),
        "pre_execution_guard_interventions_total": sum(
            row["pre_execution_guard_interventions"] for row in rows
        ),
        "runtime_guard_interventions_total": sum(
            row["runtime_guard_interventions"] for row in rows
        ),
        "maximum_abs_lateral_across_trials_m": max(
            (row["maximum_abs_lateral_m"] for row in rows), default=None
        ),
        "minimum_battery_across_trials_percent": min(batteries) if batteries else None,
        "reobservations_total": sum(row["reobservations"] for row in rows),
        "motion_chunks_total": sum(row["motion_chunks"] for row in rows),
        "high_level_decisions_total": sum(row["high_level_decisions"] for row in rows),
    }


def aggregate_rows(rows: list[dict[str, Any]], *, expected_repeats: int = 3) -> dict[str, Any]:
    if type(expected_repeats) is not int or expected_repeats < 1:
        raise S1SummaryError("expected_repeats must be a positive integer")
    by_key = {(row["layout"], row["condition"], row["repeat"]): row for row in rows}
    if len(by_key) != len(rows):
        raise S1SummaryError("duplicate layout/condition/repeat combination")

    expected = [
        (layout, condition, repeat)
        for layout in LAYOUTS
        for condition in PRIMARY_CONDITIONS
        for repeat in range(1, expected_repeats + 1)
    ]
    missing = [
        f"S1_{layout}_{condition}_R{repeat}"
        for layout, condition, repeat in expected
        if (layout, condition, repeat) not in by_key
    ]
    unexpected_primary = [
        row["trial_id"] for row in rows
        if row["condition"] in PRIMARY_CONDITIONS
        and not (1 <= row["repeat"] <= expected_repeats)
    ]
    primary_rows = [row for row in rows if row["condition"] in PRIMARY_CONDITIONS]
    setup_fingerprints = sorted({row.get("setup_fingerprint") for row in primary_rows})
    config_hashes = sorted({row.get("config_sha256") for row in primary_rows})
    m_memory_hashes = sorted({
        row.get("memory_sha256") for row in primary_rows
        if row["condition"] == "M"
    })
    m_rule_hashes = sorted({
        row.get("rule_sha256") for row in primary_rows
        if row["condition"] == "M"
    })
    consistent_primary_inputs = all(
        len(values) <= 1
        for values in (setup_fingerprints, config_hashes, m_memory_hashes, m_rule_hashes)
    )

    groups: dict[str, Any] = {}
    for layout in LAYOUTS:
        groups[layout] = {}
        for condition in ALL_CONDITIONS:
            selected = [
                row for row in rows
                if row["layout"] == layout and row["condition"] == condition
            ]
            if selected:
                groups[layout][condition] = _group(selected)

    paired: dict[str, Any] = {}
    for layout in LAYOUTS:
        pairs = []
        for repeat in range(1, expected_repeats + 1):
            b0 = by_key.get((layout, "B0", repeat))
            method = by_key.get((layout, "M", repeat))
            if b0 is None or method is None:
                continue
            clearance_delta = None
            if (
                b0["minimum_envelope_clearance_m"] is not None
                and method["minimum_envelope_clearance_m"] is not None
            ):
                clearance_delta = round(
                    method["minimum_envelope_clearance_m"]
                    - b0["minimum_envelope_clearance_m"], 6
                )
            pairs.append({
                "repeat": repeat,
                "m_minus_b0_completion": (
                    int(method["task_completed"]) - int(b0["task_completed"])
                ),
                "m_minus_b0_near_miss_events": (
                    method["near_miss_events"] - b0["near_miss_events"]
                ),
                "m_minus_b0_minimum_clearance_m": clearance_delta,
                "m_minus_b0_high_level_stops": (
                    method["high_level_stop_decisions"]
                    - b0["high_level_stop_decisions"]
                ),
                "m_minus_b0_elapsed_s": round(
                    method["elapsed_s"] - b0["elapsed_s"], 6
                ),
                "m_minus_b0_guard_interventions": (
                    method["guard_interventions"] - b0["guard_interventions"]
                ),
            })
        clearance_deltas = [
            pair["m_minus_b0_minimum_clearance_m"] for pair in pairs
            if pair["m_minus_b0_minimum_clearance_m"] is not None
        ]
        paired[layout] = {
            "pair_count": len(pairs),
            "expected_pair_count": expected_repeats,
            "pairs": pairs,
            "mean_m_minus_b0_minimum_clearance_m": _mean(clearance_deltas),
            "total_m_minus_b0_near_miss_events": sum(
                pair["m_minus_b0_near_miss_events"] for pair in pairs
            ),
            "total_m_minus_b0_high_level_stops": sum(
                pair["m_minus_b0_high_level_stops"] for pair in pairs
            ),
            "mean_m_minus_b0_elapsed_s": _mean(
                pair["m_minus_b0_elapsed_s"] for pair in pairs
            ),
            "total_m_minus_b0_guard_interventions": sum(
                pair["m_minus_b0_guard_interventions"] for pair in pairs
            ),
        }

    return {
        "schema_version": 1,
        "primary_protocol_complete": (
            not missing and not unexpected_primary and consistent_primary_inputs
        ),
        "expected_repeats": expected_repeats,
        "formal_trial_count": len(rows),
        "missing_primary_trial_ids": missing,
        "unexpected_primary_trial_ids": unexpected_primary,
        "primary_input_consistency": {
            "consistent": consistent_primary_inputs,
            "setup_fingerprints": setup_fingerprints,
            "config_sha256": config_hashes,
            "m_memory_sha256": m_memory_hashes,
            "m_rule_sha256": m_rule_hashes,
        },
        "groups": groups,
        "paired_m_minus_b0": paired,
        "guard_is_reported_separately": True,
        "automatic_claim_decision": None,
        "interpretation_note": (
            "Positive clearance deltas favour M; negative near-miss deltas favour M. "
            "Review raw events, exclusions, video, completion and A-layout over-conservatism "
            "before making any experimental claim."
        ),
        "trials": sorted(rows, key=lambda row: (
            row["layout"], row["repeat"], ALL_CONDITIONS.index(row["condition"])
        )),
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [key for key in rows[0] if key != "source"] if rows else [
        "trial_id", "layout", "condition", "repeat", "status", "task_completed",
        "elapsed_s", "near_miss_events", "minimum_envelope_clearance_m",
        "high_level_stop_decisions", "guard_interventions",
        "pre_execution_guard_interventions", "runtime_guard_interventions",
        "maximum_abs_lateral_m", "minimum_battery_percent", "reobservations",
        "motion_chunks", "high_level_decisions",
    ]
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional aggregate JSON path")
    parser.add_argument("--csv", type=Path, help="optional trial-level CSV path")
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--include-b1", action="store_true")
    args = parser.parse_args()
    try:
        rows = load_rows(args.results_dir, include_b1=args.include_b1)
        report = aggregate_rows(rows, expected_repeats=args.expected_repeats)
        if args.output:
            atomic_json(args.output.expanduser().resolve(), report)
        if args.csv:
            write_csv(args.csv, report["trials"])
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["primary_protocol_complete"] else 2
    except (OSError, S1SummaryError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
