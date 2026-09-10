"""Generate provenance-safe S1 seed memories from measured experiment geometry.

Generated records are explicitly human-constructed examples.  They are never
labelled as historical robot failures or verified physical successes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from g1_state_collector import atomic_json
from s1_config import load_s1_config, trial_geometry


def _environment(
    width: float,
    left_clearance: float,
    right_clearance: float,
    uncertainty: float,
    confidence: float,
    center_offset: float,
) -> dict:
    return {
        "corridor_width_m": round(width, 4),
        "corridor_geometry_valid": True,
        "left_envelope_clearance_m": round(left_clearance, 4),
        "right_envelope_clearance_m": round(right_clearance, 4),
        "clearance_uncertainty_m": round(uncertainty, 4),
        "minimum_envelope_clearance_m": round(
            min(left_clearance, right_clearance) - uncertainty, 4
        ),
        "corridor_center_offset_m": round(center_offset, 4),
        "corridor_heading_error_rad": 0.0,
        "corridor_geometry_confidence": confidence,
        "obstacle_detected": False,
        "obstacle_distance_m": None,
        "observation_confidence": confidence,
    }


def build_seed_memories(config: dict) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    width = float(config["geometry"]["robot_effective_width_m"])
    uncertainty = float(
        config["perception"]["estimator"]["measurement_uncertainty_floor_m"]
    )
    layout_a = trial_geometry(config, "A")
    layout_b = trial_geometry(config, "B")
    robot = {
        "type": "g1",
        "platform": "Unitree G1",
        "device_id": config["robot"]["device_id"],
        "width_m": width,
        "battery_percent": 80,
    }

    def source(trace_id: str, note: str) -> dict:
        return {
            "kind": "human_constructed",
            "historical_physical_run": False,
            "trace_id": trace_id,
            "created_at": now,
            "generator": "prepare_s1_memories.py",
            "note": note,
        }

    narrow_environment = _environment(
        layout_b["measured_corridor_width_m"],
        layout_b["nominal_left_envelope_clearance_m"],
        layout_b["nominal_right_envelope_clearance_m"],
        uncertainty,
        0.90,
        (layout_b["left_inner_offset_m"] - layout_b["right_inner_offset_m"]) / 2,
    )
    insufficient_width = width + 0.18
    insufficient_environment = _environment(
        insufficient_width, 0.09, 0.09, uncertainty, 0.88,
        (float(config["geometry"]["robot_left_extent_m"])
         - float(config["geometry"]["robot_right_extent_m"])) / 2,
    )
    wide_environment = _environment(
        layout_a["measured_corridor_width_m"],
        layout_a["nominal_left_envelope_clearance_m"],
        layout_a["nominal_right_envelope_clearance_m"],
        uncertainty,
        0.94,
        (layout_a["left_inner_offset_m"] - layout_a["right_inner_offset_m"]) / 2,
    )
    goal = config["experiment"]["task_goal"]
    slow_speed = float(config["robot"]["slow_speed_mps"])

    return [
        {
            "experience_id": "s1_constructed_narrow_low_speed_success",
            "task_goal": goal,
            "robot": dict(robot),
            "environment": narrow_environment,
            "action": {"name": "slow_down", "parameters": {"speed_mps": slow_speed}},
            "outcome": {
                "status": "success",
                "description": "人工构造的低速居中可行示例；不是历史真机成功。",
            },
            "risk": {
                "type": "narrow_corridor_collision",
                "severity": 3,
                "failure_reason": "人工构造条件中未发生失败；仅用于表达候选低速策略。",
            },
            "decision": "execute",
            "lesson": "仅在实时左右包络间距、置信度和居中状态均满足阈值时考虑低速通过。",
            "mitigations": ["低速短段执行", "每段重新观测", "持续监测两侧最小包络间距"],
            "stop_conditions": ["任一侧保守包络间距不超过硬阈值", "观测失效或机器人偏航"],
            "statistics": {"support": 0, "success": 0, "failure": 0, "aborted": 0},
            "quality": {"confidence": 0.45, "last_verified_at": None},
            "source": source(
                "s1_seed_narrow_success",
                "Hypothetical positive example; must not be reported as a physical trial.",
            ),
        },
        {
            "experience_id": "s1_constructed_insufficient_margin_abort",
            "task_goal": goal,
            "robot": dict(robot),
            "environment": insufficient_environment,
            "action": {"name": "safe_stop", "parameters": {}},
            "outcome": {
                "status": "aborted",
                "description": "人工构造的余量不足告警与安全中止事件；不是真机历史失败。",
            },
            "risk": {
                "type": "narrow_corridor_envelope_intrusion",
                "severity": 5,
                "failure_reason": "人工构造条件下，扣除不确定性后的单侧余量低于安全阈值。",
            },
            "decision": "safe_stop",
            "lesson": "余量不足必须告警并中止，不应用任务收益抵消几何硬约束。",
            "mitigations": ["安全停止", "重新测量通道与动态包络", "由现场人员重新布置"],
            "stop_conditions": ["minimum_envelope_clearance_m <= minimum_side_clearance_m"],
            "statistics": {"support": 0, "success": 0, "failure": 0, "aborted": 0},
            "quality": {"confidence": 0.55, "last_verified_at": None},
            "source": source(
                "s1_seed_insufficient_abort",
                "Constructed safety event; explicitly not a historical collision or failure.",
            ),
        },
        {
            "experience_id": "s1_constructed_wide_corridor_distractor",
            "task_goal": goal,
            "robot": dict(robot),
            "environment": wide_environment,
            "action": {"name": "move_forward", "parameters": {"speed_mps": 0.10}},
            "outcome": {
                "status": "success",
                "description": "人工构造的宽走廊成功干扰卡；不是历史真机成功。",
            },
            "risk": {
                "type": "open_corridor_navigation",
                "severity": 1,
                "failure_reason": "人工构造条件中未发生失败。",
            },
            "decision": "execute",
            "lesson": "宽走廊证据不应覆盖窄通道的单侧包络风险。",
            "mitigations": ["保持前向障碍监测"],
            "stop_conditions": ["出现近距离障碍或感知失效"],
            "statistics": {"support": 0, "success": 0, "failure": 0, "aborted": 0},
            "quality": {"confidence": 0.40, "last_verified_at": None},
            "source": source(
                "s1_seed_wide_distractor",
                "Hypothetical distractor; must not be reported as a physical trial.",
            ),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_s1_config(args.config, require_onsite=False)
        output = args.output.expanduser().resolve()
        if output.exists():
            raise ValueError("output already exists; choose a new file to preserve the frozen memory set")
        output.parent.mkdir(parents=True, exist_ok=True)
        memories = build_seed_memories(config)
        atomic_json(output, memories)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        print(json.dumps({
            "status": "created",
            "output": str(output),
            "sha256": digest,
            "records": len(memories),
            "source_kind": "human_constructed",
            "historical_physical_runs": 0,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
