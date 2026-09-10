"""Run a non-experimental, full-passage low-speed safety pilot for S1.

Place G1 on the marked pilot line 0.20 m before the entrance.  The program uses
the same calibrated perception, Safety Guard, bounded G1 adapter and logging as
formal trials, but no memory or Agent.  A pilot is never counted as B0/M/B1 and
requires an exact interactive confirmation before any command-capable driver
is constructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import socket
import sys

from execution_bridge import SafeExecutionBridge
from g1_state_collector import atomic_json
from run_s1_trial import _paths, _public_config, _sha256, preflight
from s1_config import LAYOUTS, load_s1_config, setup_fingerprint
from s1_experiment import (
    JSONLAuditLog,
    S1_ACTIONS,
    S1Metrics,
    S1PilotRunner,
    observation_summary,
)


def pilot_id(layout: str, repeat: int) -> str:
    return f"S1_PILOT_{layout}_R{repeat}"


def confirmation_text(config: dict, layout: str, repeat: int) -> tuple[str, str]:
    geometry = config["geometry"]
    target = (
        float(geometry["pilot_start_before_entrance_m"])
        + float(geometry["passage_length_m"])
        + float(geometry["pilot_goal_after_exit_m"])
    )
    identifier = pilot_id(layout, repeat)
    token = f"CONFIRM_{identifier}_LOW_SPEED_FORWARD_ESTOP_READY"
    text = (
        f"安全预实验 {identifier}（不计入正式数据）：请把机器人居中放在入口前 "
        f"{geometry['pilot_start_before_entrance_m']:.2f} m 的 P 线。机器人只沿当前朝向以 "
        f"{config['robot']['slow_speed_mps']:.2f} m/s 直行，完整穿过 {geometry['passage_length_m']:.2f} m "
        f"纸箱通道，并在出口后 {geometry['pilot_goal_after_exit_m']:.2f} m 目标区停止；"
        f"总里程计目标 {target:.2f} m。不会发送转向、横移、姿态或 FSM 指令，每段最多 "
        f"{config['runtime']['motion_chunk_duration_s']:.2f} s。"
    )
    return text, token


def require_confirmation(config, layout, repeat, *, input_fn=input, output=sys.stderr, require_tty=True):
    if require_tty and not sys.stdin.isatty():
        raise RuntimeError("physical pilot requires an attached interactive TTY")
    text, token = confirmation_text(config, layout, repeat)
    print("\n即将允许低速非零运动。确认纸箱固定、全程无人无杂物、旁站人员手持独立急停：", file=output)
    print(text, file=output)
    print(f"确认后请输入：{token}", file=output)
    if input_fn("S1 pilot confirmation> ").strip() != token:
        raise RuntimeError("pilot confirmation did not match; no motion driver was constructed")
    return text, token


def execute(config, config_path: Path, layout: str, repeat: int):
    if config["runtime"]["execution_enabled"] is not True:
        raise RuntimeError("runtime.execution_enabled is false")
    interface = config["runtime"]["network_interface"]
    if interface not in {name for _index, name in socket.if_nameindex()}:
        raise RuntimeError(f"network interface does not exist: {interface}")
    loaded_config_hash = config.get("_config_sha256")
    if not loaded_config_hash or _sha256(config_path) != loaded_config_hash:
        raise RuntimeError("S1 config changed after it was loaded")
    provider, g1_config, preflight_report = preflight(
        config, layout, start_mode="pilot"
    )
    paths = _paths(config)
    results_root = paths["results_dir"]
    results_root.mkdir(parents=True, exist_ok=True)
    identifier = pilot_id(layout, repeat)
    trial_dir = results_root / identifier
    if trial_dir.exists():
        raise RuntimeError(f"pilot result already exists: {trial_dir}")
    text, token = require_confirmation(config, layout, repeat)
    if _sha256(config_path) != loaded_config_hash:
        raise RuntimeError("S1 config changed while awaiting pilot confirmation")
    trial_dir.mkdir(exist_ok=False)
    audit = JSONLAuditLog(trial_dir / "events.jsonl")
    fingerprint = setup_fingerprint(config)
    manifest = {
        "schema_version": 1,
        "pilot_id": identifier,
        "layout": layout,
        "repeat": repeat,
        "formal_trial": False,
        "eligible_as_s1_result": False,
        "setup_fingerprint": fingerprint,
        "config_sha256": loaded_config_hash,
        "confirmation_received": True,
        "confirmation_token": token,
        "motion": text,
        "preflight": preflight_report,
    }
    atomic_json(trial_dir / "manifest.json", manifest)
    atomic_json(trial_dir / "config.snapshot.json", _public_config(config))
    shutil.copy2(config_path, trial_dir / "config.source.json")
    if _sha256(trial_dir / "config.source.json") != loaded_config_hash:
        raise RuntimeError("frozen pilot config does not match the loaded configuration")
    audit.append("pilot_created", manifest=manifest)

    geometry = config["geometry"]
    pilot_exit = (
        float(geometry["pilot_start_before_entrance_m"])
        + float(geometry["passage_length_m"])
    )
    metrics = S1Metrics(config, layout, passage_interval=(0.0, pilot_exit))

    def observed(observation, progress):
        metrics.observe(observation, progress)
        audit.append("observation", progress=progress, observation=observation_summary(observation))

    provider.on_observation = observed
    driver = None
    try:
        from unitree_adapter import G1LocoDriver, UnitreeRobotAdapter
        driver = G1LocoDriver.connect(
            interface, timeout_s=1.0,
            command_lease_s=float(config["runtime"]["command_lease_s"]),
        )
        adapter = UnitreeRobotAdapter(g1_config, driver, provider)
        bridge = SafeExecutionBridge(
            adapter,
            available_actions=S1_ACTIONS,
            experience_sink=lambda record: audit.append("l0_record", record=record),
        )
        summary = S1PilotRunner(
            config, layout, provider, bridge, metrics, audit
        ).run()
    except Exception:
        if driver is not None:
            try:
                driver.stop()
            except Exception:
                pass
        raise
    summary.update({
        "pilot_id": identifier,
        "repeat": repeat,
        "setup_fingerprint": fingerprint,
        "result_directory": str(trial_dir),
    })
    atomic_json(trial_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    try:
        if not 1 <= args.repeat <= 99:
            raise ValueError("repeat must be within [1, 99]")
        config_path = args.config.expanduser().resolve()
        config = load_s1_config(config_path, require_onsite=True)
        result = execute(config, config_path, args.layout, args.repeat)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["status"] == "pilot_completed" else 1
    except Exception as exc:
        print(json.dumps({
            "status": "failed", "error": str(exc),
            "motion_commands_sent": "unknown_after_pilot_entry",
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
