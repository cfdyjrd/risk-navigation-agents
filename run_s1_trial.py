"""Validate, preview, or execute one auditable G1 S1 narrow-corridor trial.

Only ``--mode execute`` can construct a motion driver.  Even then, an attached
TTY must repeat a trial-specific token after the program prints the exact
forward-only motion envelope.  Validate, preflight, and preview never send a
motion command.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import re

from execution_bridge import SafeExecutionBridge
from experience_store import ExperienceStore
from g1_state_collector import atomic_json
from risk_rule_store import RiskRuleStore
from robot_interface import observation_to_scenario
from safety_guard import apply_safety_guard
from s1_config import (
    CONDITIONS,
    LAYOUTS,
    confirmation_token,
    load_s1_config,
    motion_description,
    resolve_config_path,
    task_at,
    trial_geometry,
    trial_id,
    setup_fingerprint,
)
from s1_decision import make_s1_decider
from s1_experiment import (
    G1FileObservationProvider,
    JSONLAuditLog,
    S1_ACTIONS,
    S1Metrics,
    S1TrialRunner,
    observation_summary,
    require_corridor_geometry,
    wait_for_stationary_observation,
)
from unitree_adapter import G1Config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_config(config: dict) -> dict:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def _paths(config: dict) -> dict[str, Path]:
    return {
        key: resolve_config_path(config, value)
        for key, value in config["paths"].items()
    }


def _load_condition_inputs(config: dict, condition: str):
    paths = _paths(config)
    if condition == "B0":
        return None, None, {
            "memory_mode": "disabled",
            "rule_mode": "disabled",
            "memory_sha256": None,
            "rule_sha256": None,
        }
    if not paths["memory_file"].is_file():
        raise ValueError(f"memory file does not exist: {paths['memory_file']}")
    if not paths["rule_file"].is_file():
        raise ValueError(f"rule file does not exist: {paths['rule_file']}")
    store = ExperienceStore(paths["memory_file"])
    rule_store = RiskRuleStore(paths["rule_file"])
    store.load()
    rule_store.load()
    return store, rule_store, {
        "memory_mode": "role_conditioned" if condition == "M" else "shared_bundle",
        "rule_mode": "enabled",
        "memory_sha256": _sha256(paths["memory_file"]),
        "rule_sha256": _sha256(paths["rule_file"]),
    }


def verify_pilot_approvals(config: dict) -> dict:
    """Require operator-reviewed, setup-matching pilots for both layouts."""
    results_root = _paths(config)["results_dir"]
    fingerprint = setup_fingerprint(config)
    evidence = {}
    for layout in sorted(LAYOUTS):
        approval = config["pilot_approvals"][layout]
        if approval.get("approved") is not True:
            raise RuntimeError(f"layout {layout} low-speed pilot is not operator-approved")
        identifier = approval["pilot_id"]
        if not re.fullmatch(rf"S1_PILOT_{layout}_R[1-9][0-9]*", identifier):
            raise RuntimeError(f"layout {layout} pilot_id has an invalid format")
        summary_path = results_root / identifier / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"layout {layout} approved pilot summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("pilot_id") != identifier
            or summary.get("layout") != layout
            or summary.get("formal_trial") is not False
            or summary.get("eligible_as_s1_result") is not False
            or summary.get("low_speed_strategy_candidate_passed") is not True
            or summary.get("setup_fingerprint") != fingerprint
        ):
            raise RuntimeError(f"layout {layout} pilot does not match the current safety setup")
        evidence[layout] = {
            "pilot_id": identifier,
            "summary_sha256": _sha256(summary_path),
            "reviewed_by": approval["reviewed_by"],
            "reviewed_at": approval["reviewed_at"],
            "setup_fingerprint": fingerprint,
        }
    return evidence


def build_provider(config: dict, layout: str, *, on_observation=None):
    paths = _paths(config)
    g1_config = G1Config(**config["robot"])
    provider = G1FileObservationProvider(
        config,
        layout,
        g1_config,
        paths["state_file"],
        paths["perception_file"],
        on_observation=on_observation,
    )
    return provider, g1_config


def preflight(
    config: dict, layout: str, *, start_mode: str = "formal"
) -> tuple[G1FileObservationProvider, object, dict]:
    if start_mode not in {"formal", "pilot"}:
        raise ValueError("start_mode must be formal or pilot")
    paths = _paths(config)
    for key in ("state_file", "perception_file"):
        if not paths[key].is_file():
            raise ValueError(f"{key} does not exist: {paths[key]}")
    provider, g1_config = build_provider(config, layout)
    observation = wait_for_stationary_observation(
        provider,
        timeout_s=float(config["runtime"]["observation_timeout_s"]),
        poll_interval_s=float(config["runtime"]["poll_interval_s"]),
    )
    geometry = trial_geometry(config, layout)
    expected_wall_start = (
        geometry["entrance_m"]
        if start_mode == "formal"
        else geometry["pilot_entrance_m"]
    )
    require_corridor_geometry(
        observation, config, layout,
        expected_wall_start_m=expected_wall_start,
    )
    battery = float(observation.robot["battery_percent"])
    if battery < float(config["runtime"]["minimum_start_battery_percent"]):
        raise ValueError("battery is below the configured S1 start threshold")
    task = task_at(config, layout, geometry["decision_line_m"])
    scenario = observation_to_scenario(observation, task, S1_ACTIONS)
    guard = apply_safety_guard(scenario, {
        "decision": "execute",
        "selected_action": {"name": "slow_down", "parameters": {}},
    })
    if guard["status"] != "approved":
        raise ValueError(f"Safety Guard preflight failed: {guard['hard_rule_violations']}")
    return provider, g1_config, {
        "status": "ready",
        "motion_commands_sent": 0,
        "layout": layout,
        "start_mode": start_mode,
        "expected_corridor_entrance_m": expected_wall_start,
        "geometry": geometry,
        "observation": observation_summary(observation),
        "guard": guard,
    }


def require_terminal_confirmation(
    config: dict,
    layout: str,
    condition: str,
    repeat: int,
    *,
    input_fn=input,
    output=sys.stderr,
    require_tty: bool = True,
) -> str:
    identifier = trial_id(layout, condition, repeat)
    token = confirmation_token(identifier)
    if require_tty and not sys.stdin.isatty():
        raise RuntimeError("physical execution requires an attached interactive TTY")
    print("\n即将允许非零运动指令。请逐项确认场地已清空、纸箱固定、旁站人员手持独立急停：", file=output)
    print(motion_description(config, layout, condition, repeat), file=output)
    print(f"确认后请输入：{token}", file=output)
    answer = input_fn("S1 motion confirmation> ").strip()
    if answer != token:
        raise RuntimeError("confirmation token did not match; no motion driver was constructed")
    return token


def _preview(config: dict, layout: str, condition: str) -> dict:
    provider, _g1_config, report = preflight(config, layout)
    store, rule_store, evidence = _load_condition_inputs(config, condition)
    from llm_client import ZhinaoClient

    audits = []
    decider = make_s1_decider(
        condition,
        ZhinaoClient(),
        store=store,
        rule_store=rule_store,
        top_k=int(config["memory"]["top_k"]),
        token_budget=int(config["memory"]["token_budget"]),
        audit_sink=audits.append,
    )
    observation = provider()
    geometry = trial_geometry(config, layout)
    task = task_at(config, layout, geometry["decision_line_m"])
    scenario = observation_to_scenario(observation, task, S1_ACTIONS)
    scenario["environment"]["distance_to_goal_m"] = task["distance_to_goal_m"]
    decision = decider(scenario)
    return {
        "status": "decision_preview_only",
        "formal_trial": False,
        "execution_authorized": False,
        "motion_commands_sent": 0,
        "preflight": report,
        "condition_inputs": evidence,
        "decision": decision,
        "guard": apply_safety_guard(scenario, decision),
        "decision_audit": audits[-1] if audits else None,
        "note": "Preview observation is treated as the D-line scenario but no physical D-line crossing is claimed.",
    }


def _execute(config: dict, config_path: Path, layout: str, condition: str, repeat: int) -> dict:
    if config["runtime"]["execution_enabled"] is not True:
        raise RuntimeError("runtime.execution_enabled is false")
    interfaces = {name for _index, name in socket.if_nameindex()}
    interface = config["runtime"]["network_interface"]
    if interface not in interfaces:
        raise RuntimeError(f"network interface does not exist: {interface}")
    loaded_config_hash = config.get("_config_sha256")
    if not loaded_config_hash or _sha256(config_path) != loaded_config_hash:
        raise RuntimeError("S1 config changed after it was loaded")

    pilot_evidence = verify_pilot_approvals(config)
    provider, g1_config, preflight_report = preflight(config, layout)
    store, rule_store, evidence = _load_condition_inputs(config, condition)
    # Constructing the API client checks credentials but does not make a call.
    from llm_client import ZhinaoClient
    client = ZhinaoClient()
    identifier = trial_id(layout, condition, repeat)
    paths = _paths(config)
    results_root = paths["results_dir"]
    results_root.mkdir(parents=True, exist_ok=True)
    trial_dir = results_root / identifier
    if trial_dir.exists():
        raise RuntimeError(f"trial result already exists: {trial_dir}")
    token = require_terminal_confirmation(config, layout, condition, repeat)
    if _sha256(config_path) != loaded_config_hash:
        raise RuntimeError("S1 config changed while awaiting confirmation")

    trial_dir.mkdir(exist_ok=False)
    audit = JSONLAuditLog(trial_dir / "events.jsonl")
    manifest = {
        "schema_version": 1,
        "trial_id": identifier,
        "layout": layout,
        "condition": condition,
        "repeat": repeat,
        "formal_trial": True,
        "setup_fingerprint": setup_fingerprint(config),
        "config_sha256": loaded_config_hash,
        "condition_inputs": evidence,
        "approved_low_speed_pilots": pilot_evidence,
        "confirmation_token": token,
        "confirmation_received": True,
        "preflight": preflight_report,
        "motion": motion_description(config, layout, condition, repeat),
    }
    atomic_json(trial_dir / "manifest.json", manifest)
    atomic_json(trial_dir / "config.snapshot.json", _public_config(config))
    shutil.copy2(config_path, trial_dir / "config.source.json")
    if _sha256(trial_dir / "config.source.json") != loaded_config_hash:
        raise RuntimeError("frozen config source does not match the loaded configuration")
    if condition != "B0":
        memory_snapshot = trial_dir / "memory.snapshot.json"
        rule_snapshot = trial_dir / "rules.snapshot.json"
        shutil.copy2(paths["memory_file"], memory_snapshot)
        shutil.copy2(paths["rule_file"], rule_snapshot)
        if _sha256(memory_snapshot) != evidence["memory_sha256"]:
            raise RuntimeError("memory file changed before it could be frozen")
        if _sha256(rule_snapshot) != evidence["rule_sha256"]:
            raise RuntimeError("risk-rule file changed before it could be frozen")
        store = ExperienceStore(memory_snapshot)
        rule_store = RiskRuleStore(rule_snapshot)
        store.load()
        rule_store.load()
    audit.append("trial_created", manifest=manifest)

    metrics = S1Metrics(config, layout)

    def observed(observation, progress):
        metrics.observe(observation, progress)
        audit.append(
            "observation",
            progress=progress,
            observation=observation_summary(observation),
        )

    provider.on_observation = observed
    decision_index = {"value": 0}

    def decision_audit(record):
        decision_index["value"] += 1
        audit.append(
            "decision_audit",
            decision_index=decision_index["value"],
            first_retrieval_at_decision_line=decision_index["value"] == 1,
            record=record,
        )

    decider = make_s1_decider(
        condition,
        client,
        store=store,
        rule_store=rule_store,
        top_k=int(config["memory"]["top_k"]),
        token_budget=int(config["memory"]["token_budget"]),
        audit_sink=decision_audit,
    )

    driver = None
    try:
        # This is the first point at which a command-capable API is registered;
        # it occurs only after the trial-specific terminal confirmation.
        from unitree_adapter import G1LocoDriver, UnitreeRobotAdapter
        driver = G1LocoDriver.connect(
            interface,
            timeout_s=1.0,
            command_lease_s=float(config["runtime"]["command_lease_s"]),
        )
        adapter = UnitreeRobotAdapter(g1_config, driver, provider)
        bridge = SafeExecutionBridge(
            adapter,
            available_actions=S1_ACTIONS,
            experience_sink=lambda record: audit.append("l0_record", record=record),
        )
        summary = S1TrialRunner(
            config, layout, condition, provider, bridge, decider, metrics, audit
        ).run()
    except Exception:
        if driver is not None:
            try:
                driver.stop()
            except Exception:
                pass
        raise
    summary.update({
        "trial_id": identifier,
        "repeat": repeat,
        "formal_trial": True,
        "result_directory": str(trial_dir),
        "input_hashes": evidence,
        "setup_fingerprint": setup_fingerprint(config),
    })
    atomic_json(trial_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("validate", "preflight", "preview", "execute"),
        default="validate",
    )
    args = parser.parse_args()
    motion_capable = args.mode == "execute"
    try:
        if not 1 <= args.repeat <= 99:
            raise ValueError("repeat must be within [1, 99]")
        config_path = args.config.expanduser().resolve()
        config = load_s1_config(
            config_path, require_onsite=args.mode in {"preflight", "preview", "execute"}
        )
        if args.mode == "validate":
            result = {
                "status": "config_valid",
                "trial_id": trial_id(args.layout, args.condition, args.repeat),
                "geometry": trial_geometry(config, args.layout),
                "onsite_verified": config["perception"]["onsite_validation"]["status"] == "onsite_verified",
                "execution_enabled": config["runtime"]["execution_enabled"],
                "pilot_approvals": {
                    layout: config["pilot_approvals"][layout]["approved"]
                    for layout in sorted(LAYOUTS)
                },
                "motion_commands_sent": 0,
            }
        elif args.mode == "preflight":
            _provider, _g1_config, result = preflight(config, args.layout)
        elif args.mode == "preview":
            result = _preview(config, args.layout, args.condition)
        else:
            result = _execute(
                config, config_path, args.layout, args.condition, args.repeat
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        valid_trial_outcomes = {"completed", "stopped_incomplete", "guarded_stop"}
        if args.mode == "execute":
            return 0 if result.get("status") in valid_trial_outcomes else 1
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "failed",
            "error": str(exc),
            "motion_commands_sent": 0 if not motion_capable else "unknown_after_execute_entry",
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
