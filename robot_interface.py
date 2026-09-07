"""Robot-independent contracts for connecting the method to a real platform.

The core project intentionally has no ROS dependency.  A ROS 2 node only needs
to implement :class:`RobotAdapter`; all decision and safety logic remains here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


SEMANTIC_ACTIONS = frozenset(
    {
        "move_forward",
        "turn_left",
        "turn_right",
        "slow_down",
        "observe_again",
        "ask_human",
        "safe_stop",
    }
)


class RobotInterfaceError(RuntimeError):
    """Raised when robot state or execution feedback violates the contract."""


@dataclass(frozen=True)
class RobotObservation:
    """Minimal real-time state consumed by the existing decision pipeline."""

    robot: dict[str, Any]
    environment: dict[str, Any]
    timestamp: str
    frame_id: str = "base_link"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required_robot = ("type", "width_m", "battery_percent")
        required_environment = (
            "obstacle_detected",
            "obstacle_distance_m",
            "observation_confidence",
        )
        missing = [f"robot.{key}" for key in required_robot if key not in self.robot]
        missing += [
            f"environment.{key}"
            for key in required_environment
            if key not in self.environment
        ]
        if not self.timestamp:
            missing.append("timestamp")
        if missing:
            raise RobotInterfaceError(
                "robot observation is missing required fields: " + ", ".join(missing)
            )

        confidence = self.environment["observation_confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise RobotInterfaceError("observation_confidence must be within [0, 1]")
        distance = self.environment["obstacle_distance_m"]
        if not isinstance(distance, (int, float)) or distance < 0:
            raise RobotInterfaceError("obstacle_distance_m must be non-negative")


@dataclass(frozen=True)
class ExecutionReceipt:
    """Platform feedback used to audit execution and construct an L0 record."""

    status: str
    description: str
    started_at: str
    finished_at: str
    telemetry: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.status not in {"success", "failure", "aborted"}:
            raise RobotInterfaceError(f"invalid execution status: {self.status!r}")
        if not self.started_at or not self.finished_at:
            raise RobotInterfaceError("execution receipt requires start and finish times")


class RobotAdapter(ABC):
    """Hardware boundary to be implemented by a ROS 2 or vendor-specific node."""

    @abstractmethod
    def observe(self) -> RobotObservation:
        """Return the latest synchronized robot and environment observation."""

    @abstractmethod
    def execute(self, approved_action: dict[str, Any]) -> ExecutionReceipt:
        """Execute one semantic action that has already passed Safety Guard."""

    @abstractmethod
    def emergency_stop(self, reason: str) -> ExecutionReceipt:
        """Stop independently of the normal action path."""


def observation_to_scenario(
    observation: RobotObservation,
    task: dict[str, Any],
    available_actions: list[str],
) -> dict[str, Any]:
    """Convert platform state to the scenario schema used by the method."""
    observation.validate()
    if not isinstance(task, dict) or not task.get("goal"):
        raise RobotInterfaceError("task.goal is required")
    if not available_actions or "safe_stop" not in available_actions:
        raise RobotInterfaceError("available_actions must include safe_stop")
    unknown = set(available_actions) - SEMANTIC_ACTIONS
    if unknown:
        raise RobotInterfaceError(f"unsupported semantic actions: {sorted(unknown)}")
    return {
        "task": deepcopy(task),
        "robot": deepcopy(observation.robot),
        "environment": deepcopy(observation.environment),
        "available_actions": list(available_actions),
        "observation": {
            "timestamp": observation.timestamp,
            "frame_id": observation.frame_id,
            "metadata": deepcopy(observation.metadata),
        },
    }
