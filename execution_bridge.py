"""Fail-closed bridge from an optimized decision to robot execution."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from robot_interface import (
    ExecutionReceipt,
    RobotAdapter,
    RobotInterfaceError,
    SEMANTIC_ACTIONS,
    observation_to_scenario,
)
from safety_guard import apply_safety_guard


# Runtime type alias must also import on the G1 host's Python 3.8.
ExperienceSink = Callable[[Dict[str, Any]], None]


class SafeExecutionBridge:
    """The only application-level path from a decision to robot motion.

    It samples fresh state, invokes the deterministic Safety Guard, validates
    its output, executes exactly the approved action, and records the outcome.
    Any malformed input or platform error triggers an emergency stop.
    """

    def __init__(
        self,
        adapter: RobotAdapter,
        *,
        available_actions: list[str],
        experience_sink: Optional[ExperienceSink] = None,
    ) -> None:
        self.adapter = adapter
        self.available_actions = list(available_actions)
        self.experience_sink = experience_sink

    def execute_decision(
        self, task: dict[str, Any], execution_decision: dict[str, Any]
    ) -> dict[str, Any]:
        """Guard and execute one optimizer decision; fail closed on any error."""
        try:
            observation = self.adapter.observe()
            scenario = observation_to_scenario(
                observation, task, self.available_actions
            )
            self._validate_decision_source(execution_decision)
            guard_result = apply_safety_guard(scenario, execution_decision)
            approved_action = self._validated_guard_action(guard_result)
            receipt = self.adapter.execute(approved_action)
            receipt.validate()
        except Exception as exc:
            try:
                stop_receipt = self.adapter.emergency_stop(str(exc))
                stop_receipt.validate()
            except Exception as stop_exc:
                now = datetime.now(timezone.utc).isoformat()
                stop_receipt = ExecutionReceipt(
                    "failure", f"Emergency stop failed: {stop_exc}", now, now,
                    {"stop_submitted": False, "failure_reason": str(stop_exc)},
                )
            return {
                "status": "fail_closed",
                "error": str(exc),
                "approved_action": {"name": "safe_stop", "parameters": {}},
                "receipt": _receipt_dict(stop_receipt),
            }

        record = self._build_l0_record(
            scenario, approved_action, receipt, guard_result
        )
        if self.experience_sink is not None:
            try:
                self.experience_sink(record)
            except Exception as exc:
                return {
                    "status": "executed_log_failed",
                    "approved_action": approved_action,
                    "guard": guard_result,
                    "receipt": _receipt_dict(receipt),
                    "l0_record": record,
                    "logging_error": str(exc),
                }
        return {
            "status": "executed",
            "approved_action": approved_action,
            "guard": guard_result,
            "receipt": _receipt_dict(receipt),
            "l0_record": record,
        }

    @staticmethod
    def _validate_decision_source(decision: dict[str, Any]) -> None:
        if decision.get("decision_source") != "deterministic_constrained_optimizer":
            raise RobotInterfaceError(
                "only deterministic optimizer output may enter the execution bridge"
            )
        action = decision.get("selected_action")
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise RobotInterfaceError("selected_action is malformed")

    @staticmethod
    def _validated_guard_action(guard_result: dict[str, Any]) -> dict[str, Any]:
        if guard_result.get("status") not in {"approved", "overridden"}:
            raise RobotInterfaceError("Safety Guard returned an invalid status")
        action = guard_result.get("approved_action")
        if not isinstance(action, dict) or action.get("name") not in SEMANTIC_ACTIONS:
            raise RobotInterfaceError("Safety Guard did not return a valid action")
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            raise RobotInterfaceError("approved action parameters must be an object")
        return {"name": action["name"], "parameters": dict(parameters)}

    @staticmethod
    def _build_l0_record(
        scenario: dict[str, Any],
        action: dict[str, Any],
        receipt: ExecutionReceipt,
        guard_result: dict[str, Any],
    ) -> dict[str, Any]:
        severity = receipt.telemetry.get("risk_severity", 1)
        failure_reason = receipt.telemetry.get("failure_reason", "未发生失败")
        return {
            "experience_id": f"exp_robot_{uuid4().hex[:12]}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "task_goal": scenario["task"]["goal"],
            "robot": scenario["robot"],
            "environment": scenario["environment"],
            "action": action,
            "outcome": {
                "status": receipt.status,
                "description": receipt.description,
            },
            "risk": {
                "type": receipt.telemetry.get("risk_type", "execution_observation"),
                "severity": severity,
                "failure_reason": failure_reason,
            },
            "source": {
                "type": "real_robot",
                "kind": "physical_run_candidate_review_required",
                "observation_timestamp": scenario["observation"]["timestamp"],
                "frame_id": scenario["observation"]["frame_id"],
            },
            "safety_guard": {
                "status": guard_result["status"],
                "hard_rule_violations": guard_result["hard_rule_violations"],
            },
            "telemetry": receipt.telemetry,
        }


def _receipt_dict(receipt: ExecutionReceipt) -> dict[str, Any]:
    return {
        "status": receipt.status,
        "description": receipt.description,
        "started_at": receipt.started_at,
        "finished_at": receipt.finished_at,
        "telemetry": receipt.telemetry,
    }
