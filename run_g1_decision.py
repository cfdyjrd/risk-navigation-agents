"""G1 real-observation readiness and optional three-Agent decision preview.

No motion driver is constructed. There is intentionally no execute mode.
--decide may call the configured LLM only after all input checks pass.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from g1_observation_cache import build_observation, measured_stationary
from g1_state_collector import state_freshness
from robot_interface import RobotInterfaceError, observation_to_scenario
from robot_preflight import read_json
from safety_guard import apply_safety_guard
from unitree_adapter import G1Config

ROOT = Path(__file__).resolve().parent
ACTIONS = ["move_forward", "slow_down", "observe_again", "ask_human", "safe_stop"]


def _input_path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be filled in")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def assess(config_path):
    """Collect actionable blockers; never fabricate sensor fields or limits."""
    result = {"mode": "read_only_decision_preview", "motion_commands_sent": 0,
              "motion_ready": False, "model_called": False, "blockers": []}
    blockers = result["blockers"]
    data = read_json(config_path)
    unknown = set(data) - {"robot", "width_m", "state_file", "perception_file", "task"}
    if unknown:
        blockers.append("unknown configuration fields: %s" % sorted(unknown))
    robot = data.get("robot", {})
    if not isinstance(robot, dict):
        robot = {}
    if robot.get("model") != "g1":
        blockers.append("robot.model must be g1")
    if not robot.get("allowed_fsm_ids"):
        blockers.append("allowed_fsm_ids missing: onsite locomotion-state verification required")
    if robot.get("max_tilt_rad") is None:
        blockers.append("max_tilt_rad missing: onsite attitude limit required")
    config = None
    try:
        config = G1Config(**robot)
    except (TypeError, ValueError, RobotInterfaceError) as exc:
        blockers.append("robot configuration: %s" % exc)
    width = data.get("width_m")
    if type(width) not in (int, float) or not math.isfinite(width) or width <= 0:
        blockers.append("width_m missing/invalid: measure current robot passage width")
    base = Path(config_path).resolve().parent
    state, perception = None, None
    try:
        state = read_json(_input_path(data.get("state_file"), base))
        result["state_freshness"] = state_freshness(state.get("state", {}))
        for key, status in result["state_freshness"].items():
            if not status["fresh"]:
                blockers.append("%s missing/stale/future: check acquisition and publisher clock" % key)
        if state.get("device_id") != robot.get("device_id"):
            blockers.append("state device_id does not match config")
        if state.get("errors"):
            blockers.append("state acquisition errors: %s" % state["errors"])
        result["measured_battery_percent"] = state.get("state", {}).get("battery", {}).get("soc")
        result["measured_fsm_id"] = state.get("state", {}).get("fsm", {}).get("fsm_id")
        if config and result["measured_fsm_id"] not in config.allowed_fsm_ids:
            blockers.append("measured FSM is not allowed; operator preparation required")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        blockers.append("state_file: %s" % exc)
    try:
        perception = read_json(_input_path(data.get("perception_file"), base))
        if perception.get("frame_id") != "base_link" or perception.get("validated") is not True:
            blockers.append("perception requires calibrated robot-frame obstacle processing")
    except (OSError, ValueError) as exc:
        blockers.append("perception_file: %s; raw lidar ranges are not validated perception" % exc)
    task = data.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("goal"), str) or not task["goal"].strip():
        blockers.append("task.goal must specify the intended task")
    observation = None
    if not blockers:
        try:
            observation = build_observation(state, perception, config, width_m=width)
            if not measured_stationary(observation, max_age_s=config.max_observation_age_s):
                blockers.append("measured robot motion is not stationary")
        except (ValueError, TypeError, AttributeError, RobotInterfaceError) as exc:
            blockers.append("observation: %s" % exc)
    result["status"] = "blocked" if blockers else "ready_for_decision_preview"
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    return result, observation, task


def run(config_path, *, decide=False, decider_factory=None):
    result, observation, task = assess(config_path)
    if result["blockers"] or not decide:
        return result
    scenario = observation_to_scenario(observation, task, ACTIONS)
    if decider_factory is None:
        from experience_store import ExperienceStore
        from risk_rule_store import RiskRuleStore
        from llm_client import ZhinaoClient
        from robot_task_loop import make_memory_decider

        def decider_factory():
            return make_memory_decider(
                ExperienceStore(ROOT / "experiences/risk_experiences.json"),
                RiskRuleStore(ROOT / "experiences/risk_rules.json"), ZhinaoClient(),
                audit_sink=lambda record: result.update(decision_audit=record))
    decider = decider_factory()
    # Indicates the decision pipeline was invoked, not necessarily a billed request:
    # a rule conflict can fail before LLM invocation.
    result["decision_pipeline_invoked"] = True
    decision = decider(scenario)
    result["model_called"] = True
    result["decision"] = decision
    result["preview_guard"] = apply_safety_guard(scenario, decision)
    result["status"] = "decision_preview_only"
    result["execution_authorized"] = False
    result["note"] = "Preview uses pre-deliberation state; never reuse this result as a live motion command."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decide", action="store_true", help="Run three Agents after checks; may incur API usage; never move")
    args = parser.parse_args()
    try:
        result = run(args.config, decide=args.decide)
    except Exception as exc:
        result = {"status": "error", "error": str(exc), "motion_commands_sent": 0,
                  "motion_ready": False}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] in {"ready_for_decision_preview", "decision_preview_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
