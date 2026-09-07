"""Offline contract tests for the real-robot execution boundary."""

import unittest

from execution_bridge import SafeExecutionBridge
from robot_interface import ExecutionReceipt, RobotAdapter, RobotObservation


ACTIONS = [
    "move_forward",
    "turn_left",
    "turn_right",
    "slow_down",
    "observe_again",
    "ask_human",
    "safe_stop",
]


class FakeRobot(RobotAdapter):
    def __init__(self, obstacle_distance: float = 1.0) -> None:
        self.obstacle_distance = obstacle_distance
        self.executed = []
        self.stop_reasons = []

    def observe(self) -> RobotObservation:
        return RobotObservation(
            robot={"type": "mobile_robot", "width_m": 0.55, "battery_percent": 80},
            environment={
                "corridor_width_m": 1.2,
                "obstacle_detected": True,
                "obstacle_distance_m": self.obstacle_distance,
                "observation_confidence": 0.9,
            },
            timestamp="2026-09-07T10:00:00+08:00",
        )

    def execute(self, approved_action: dict) -> ExecutionReceipt:
        self.executed.append(approved_action)
        return ExecutionReceipt(
            "success", "action completed", "10:00:01", "10:00:02"
        )

    def emergency_stop(self, reason: str) -> ExecutionReceipt:
        self.stop_reasons.append(reason)
        return ExecutionReceipt("aborted", reason, "10:00:01", "10:00:01")


def optimizer_decision(action: str) -> dict:
    return {
        "decision": "execute",
        "selected_action": {"name": action, "parameters": {}},
        "decision_source": "deterministic_constrained_optimizer",
    }


class ExecutionBridgeTests(unittest.TestCase):
    def test_executes_exact_guard_approved_action(self) -> None:
        robot = FakeRobot()
        result = SafeExecutionBridge(robot, available_actions=ACTIONS).execute_decision(
            {"goal": "到达会议室"}, optimizer_decision("slow_down")
        )
        self.assertEqual(result["status"], "executed")
        self.assertEqual(robot.executed, [{"name": "slow_down", "parameters": {}}])
        self.assertEqual(result["l0_record"]["source"]["type"], "real_robot")

    def test_guard_override_is_the_action_sent_to_robot(self) -> None:
        robot = FakeRobot(obstacle_distance=0.2)
        result = SafeExecutionBridge(robot, available_actions=ACTIONS).execute_decision(
            {"goal": "前进"}, optimizer_decision("move_forward")
        )
        self.assertEqual(result["guard"]["status"], "overridden")
        self.assertEqual(robot.executed[0]["name"], "safe_stop")

    def test_rejects_direct_model_output_and_stops(self) -> None:
        robot = FakeRobot()
        direct_model_output = {
            "selected_action": {"name": "move_forward", "parameters": {}},
            "decision_source": "safety_decision_agent",
        }
        result = SafeExecutionBridge(robot, available_actions=ACTIONS).execute_decision(
            {"goal": "前进"}, direct_model_output
        )
        self.assertEqual(result["status"], "fail_closed")
        self.assertFalse(robot.executed)
        self.assertEqual(len(robot.stop_reasons), 1)

    def test_malformed_observation_stops(self) -> None:
        robot = FakeRobot()
        robot.obstacle_distance = -1.0
        result = SafeExecutionBridge(robot, available_actions=ACTIONS).execute_decision(
            {"goal": "前进"}, optimizer_decision("move_forward")
        )
        self.assertEqual(result["status"], "fail_closed")
        self.assertEqual(result["approved_action"]["name"], "safe_stop")

    def test_experience_sink_receives_l0_record(self) -> None:
        records = []
        robot = FakeRobot()
        result = SafeExecutionBridge(
            robot, available_actions=ACTIONS, experience_sink=records.append
        ).execute_decision({"goal": "观察"}, optimizer_decision("observe_again"))
        self.assertEqual(result["status"], "executed")
        self.assertEqual(records[0]["action"]["name"], "observe_again")


if __name__ == "__main__":
    unittest.main(verbosity=2)
