"""Bounded semantic motion for Go1, Go2 and G1; import never connects hardware.

Observation providers must return a non-blocking snapshot from a sensor cache.
SDK transports acknowledge submission only, not physical arrival or stopping.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.util
import math
from pathlib import Path
import threading
import time
from typing import Callable, Protocol

from robot_interface import ExecutionReceipt, RobotAdapter, RobotInterfaceError, RobotObservation
from safety_guard import apply_safety_guard


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive(value, name: str, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise RobotInterfaceError(f"{name} must be finite and within (0, {maximum}]")
    return float(value)


def load_go1_sdk(sdk_extension: str):
    """Import the vendor extension without initializing UDP or any commands."""
    path = Path(sdk_extension)
    if not path.is_absolute() or not path.is_file() or path.suffix != ".so":
        raise RobotInterfaceError("sdk_extension must name an existing absolute .so path")
    spec = importlib.util.spec_from_file_location("_unitree_go1.robot_interface", path)
    if spec is None or spec.loader is None:
        raise RobotInterfaceError("cannot load Go1 SDK extension")
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    return sdk


@dataclass(frozen=True)
class UnitreeConfig:
    device_id: str
    model: str  # go1, go2 or g1; G1 requires G1Config
    max_speed_mps: float = 0.3
    slow_speed_mps: float = 0.1
    max_yaw_rate_rps: float = 0.5
    max_duration_s: float = 2.0
    max_observation_age_s: float = 1.0

    def __post_init__(self):
        if not isinstance(self.device_id, str) or not self.device_id.strip():
            raise RobotInterfaceError("device_id is required")
        if self.model not in {"go1", "go2", "g1"}:
            raise RobotInterfaceError("model must be go1, go2 or g1")
        if self.model == "g1" and not isinstance(self, G1Config):
            raise RobotInterfaceError("G1 requires G1Config with explicit readiness limits")
        for name in ("max_speed_mps", "slow_speed_mps", "max_yaw_rate_rps", "max_duration_s", "max_observation_age_s"):
            _positive(getattr(self, name), name, math.inf)
        if self.slow_speed_mps > self.max_speed_mps:
            raise RobotInterfaceError("slow_speed_mps exceeds max_speed_mps")


@dataclass(frozen=True)
class G1Config(UnitreeConfig):
    """Provisional bounded speeds; FSM and tilt limits must be supplied onsite."""
    model: str = "g1"
    max_speed_mps: float = 0.1
    slow_speed_mps: float = 0.05
    max_yaw_rate_rps: float = 0.2
    max_duration_s: float = 0.5
    allowed_fsm_ids: tuple[int, ...] = ()
    max_tilt_rad: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.model != "g1":
            raise RobotInterfaceError("G1Config requires model=g1")
        ids = self.allowed_fsm_ids
        if not isinstance(ids, (list, tuple)) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise RobotInterfaceError("set allowed_fsm_ids from verified G1 control configuration")
        object.__setattr__(self, "allowed_fsm_ids", tuple(ids))
        _positive(self.max_tilt_rad, "max_tilt_rad", math.pi / 2)


def _validate_g1_state(observation, config):
    # These must be independently sampled state, not values copied from config.
    state = observation.metadata.get("g1_state")
    if not isinstance(state, dict):
        raise RobotInterfaceError("G1 requires measured metadata.g1_state")
    for key in ("lowstate_timestamp", "fsm_timestamp", "attitude_timestamp"):
        try:
            stamp = datetime.fromisoformat(state[key].replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - stamp).total_seconds()
            if stamp.tzinfo is None or not 0 <= age <= config.max_observation_age_s:
                raise ValueError("stale/future state")
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise RobotInterfaceError(f"G1 {key} must be a fresh source timestamp") from exc
    fsm_id = state.get("fsm_id")
    if type(fsm_id) is not int or fsm_id not in config.allowed_fsm_ids:
        raise RobotInterfaceError("G1 FSM is not an explicitly allowed locomotion state")
    for key in ("roll_rad", "pitch_rad"):
        value = state.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > config.max_tilt_rad:
            raise RobotInterfaceError(f"G1 {key} exceeds measured attitude limit or is invalid")


def validate_unitree_observation(observation: RobotObservation, config: UnitreeConfig) -> RobotObservation:
    """Shared read-only identity, schema and freshness checks."""
    observation.validate()
    if observation.robot.get("device_id") != config.device_id:
        raise RobotInterfaceError("observation device_id does not match target robot")
    if observation.robot.get("type") != config.model:
        raise RobotInterfaceError("observation robot.type does not match model")
    try:
        stamp = datetime.fromisoformat(observation.timestamp.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("timezone required")
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
    except (ValueError, TypeError, AttributeError) as exc:
        raise RobotInterfaceError("observation timestamp must be timezone-aware ISO 8601") from exc
    if not 0 <= age <= config.max_observation_age_s:
        raise RobotInterfaceError(f"observation is stale or from the future: age={age:.3f}s")
    if config.model == "g1":
        _validate_g1_state(observation, config)
    return observation


class MotionDriver(Protocol):
    model: str
    period_s: float

    def move(self, vx: float, vy: float, yaw_rate: float) -> None: ...
    def stop(self) -> None: ...


class Go2SportDriver:
    model = "go2"
    period_s = 0.05

    def __init__(self, sport_client):
        self.client = sport_client

    @classmethod
    def connect(cls, network_interface: str, timeout_s: float = 1.0):
        """One DDS interface/robot per worker process. Does not stand the robot up."""
        if not network_interface or not network_interface.strip():
            raise RobotInterfaceError("explicit Go2 network_interface is required")
        _positive(timeout_s, "timeout_s", 10)
        channel = importlib.import_module("unitree_sdk2py.core.channel")
        sport = importlib.import_module("unitree_sdk2py.go2.sport.sport_client")
        ChannelFactoryInitialize = getattr(channel, "ChannelFactoryInitialize", None)
        SportClient = getattr(sport, "SportClient", None)
        if not callable(ChannelFactoryInitialize) or not callable(SportClient):
            raise RobotInterfaceError("Go2 SDK API mismatch")
        ChannelFactoryInitialize(0, network_interface)
        client = SportClient()
        client.SetTimeout(timeout_s)
        client.Init()
        return cls(client)

    @staticmethod
    def _check(code, operation):
        if type(code) is not int or code != 0:
            raise RobotInterfaceError(f"Go2 {operation} failed: code={code!r}")

    def move(self, vx, vy, yaw_rate):
        self._check(self.client.Move(vx, vy, yaw_rate), "Move")

    def stop(self):
        self._check(self.client.StopMove(), "StopMove")


class G1LocoDriver:
    """G1 LocoClient only. Never changes FSM, posture or control ownership.

    Move/StopMove discard the RPC code in the installed G1 SDK; call
    SetVelocity directly. The finite lease limits a stale velocity request,
    but is not a verified hardware watchdog or a guarantee of physical stop.
    """
    model = "g1"
    period_s = 0.05

    def __init__(self, loco_client, *, command_lease_s: float = 0.2):
        self.command_lease_s = _positive(command_lease_s, "command_lease_s", 0.5)
        if self.command_lease_s < self.period_s:
            raise RobotInterfaceError("command lease must cover at least one send period")
        if not callable(getattr(loco_client, "SetVelocity", None)):
            raise RobotInterfaceError("G1 SDK requires SetVelocity")
        self.client = loco_client

    @classmethod
    def connect(cls, network_interface: str, timeout_s: float = 1.0, *, command_lease_s: float = 0.2):
        if not isinstance(network_interface, str) or not network_interface.strip():
            raise RobotInterfaceError("explicit G1 network_interface is required")
        _positive(timeout_s, "timeout_s", 10)
        _positive(command_lease_s, "command_lease_s", 0.5)
        if command_lease_s < cls.period_s:
            raise RobotInterfaceError("command lease must cover at least one send period")
        channel = importlib.import_module("unitree_sdk2py.core.channel")
        loco = importlib.import_module("unitree_sdk2py.g1.loco.g1_loco_client")
        factory = getattr(channel, "ChannelFactoryInitialize", None)
        client_class = getattr(loco, "LocoClient", None)
        if not callable(factory) or not callable(client_class) or not callable(getattr(client_class, "SetVelocity", None)):
            raise RobotInterfaceError("G1 SDK API mismatch")
        factory(0, network_interface)
        client = client_class()
        client.SetTimeout(timeout_s)
        client.Init()
        return cls(client, command_lease_s=command_lease_s)

    def move(self, vx, vy, yaw_rate):
        self._send(vx, vy, yaw_rate)

    def stop(self):
        # Do not use Damp/ZeroTorque: that is not a walking stop for a humanoid.
        self._send(0.0, 0.0, 0.0)

    def _send(self, vx, vy, yaw_rate):
        code = self.client.SetVelocity(vx, vy, yaw_rate, self.command_lease_s)
        if type(code) is not int or code != 0:
            raise RobotInterfaceError(f"G1 SetVelocity failed: code={code!r}")


class Go1UDPDriver:
    model = "go1"
    period_s = 0.002

    def __init__(self, udp, command):
        self.udp, self.command = udp, command
        udp.InitCmdData(command)

    @classmethod
    def connect(cls, sdk_extension: str, robot_ip: str, *, local_port: int = 8080, remote_port: int = 8082):
        """Load the legacy SDK extension without shadowing our robot_interface.py.

        sdk_extension is the absolute path to the compiled robot_interface*.so
        matching the host's Python ABI and CPU. Network addressing is explicit.
        """
        if not isinstance(robot_ip, str) or not robot_ip.strip():
            raise RobotInterfaceError("Go1 robot_ip is required")
        for port in (local_port, remote_port):
            if type(port) is not int or not 0 < port <= 65535:
                raise RobotInterfaceError("UDP ports must be within [1, 65535]")
        sdk = load_go1_sdk(sdk_extension)
        return cls(sdk.UDP(0xEE, local_port, robot_ip, remote_port), sdk.HighCmd())

    def move(self, vx, vy, yaw_rate):
        cmd = self.command
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [vx, vy]
        cmd.yawSpeed = yaw_rate
        self.udp.SetSend(cmd)
        result = self.udp.Send()
        # Legacy bindings may return None for a void Send(). No delivery ACK.
        if result is not None and (type(result) is not int or result != 0):
            raise RobotInterfaceError(f"Go1 UDP Send failed: {result!r}")

    def stop(self):
        # A burst limits the effect of a single lost UDP packet; no physical ACK.
        for _ in range(10):
            self.move(0.0, 0.0, 0.0)
            time.sleep(self.period_s)


class UnitreeRobotAdapter(RobotAdapter):
    def __init__(self, config: UnitreeConfig, driver: MotionDriver,
                 observation_provider: Callable[[], RobotObservation]):
        if config.model != driver.model:
            raise RobotInterfaceError("configured model does not match driver")
        _positive(driver.period_s, "driver.period_s", 0.1)
        self.config, self.driver = config, driver
        self.observation_provider = observation_provider
        self._execution_lock = threading.Lock()
        self._driver_lock = threading.Lock()
        self._stop_requested = threading.Event()

    def observe(self) -> RobotObservation:
        observation = self.observation_provider()
        return validate_unitree_observation(observation, self.config)

    def _motion(self, name, params):
        turning = name in {"turn_left", "turn_right"}
        allowed = {"duration_s", "angle_rad", "yaw_rate_rps"} if turning else {"duration_s", "distance_m", "speed_mps"}
        if set(params) - allowed:
            raise RobotInterfaceError(f"unsupported parameters: {sorted(set(params) - allowed)}")
        extent_key = "angle_rad" if turning else "distance_m"
        if extent_key in params and "duration_s" in params:
            raise RobotInterfaceError(f"specify either {extent_key} or duration_s")
        limit = self.config.max_yaw_rate_rps if turning else (
            self.config.slow_speed_mps if name == "slow_down" else self.config.max_speed_mps)
        speed_key = "yaw_rate_rps" if turning else "speed_mps"
        speed = _positive(params.get(speed_key, limit), speed_key, limit)
        duration = params.get("duration_s", min(0.5, self.config.max_duration_s))
        if extent_key in params:
            duration = _positive(params[extent_key], extent_key, limit * self.config.max_duration_s) / speed
        duration = _positive(duration, "duration_s", self.config.max_duration_s)
        return (0.0 if turning else speed, 0.0,
                (speed if name == "turn_left" else -speed) if turning else 0.0, duration)

    def _receipt(self, status, description, start, **telemetry):
        return ExecutionReceipt(status, description, start, _utc(), {
            "device_id": self.config.device_id, "model": self.config.model,
            "feedback_basis": "command_submission", "physical_completion_verified": False,
            **telemetry,
        })

    def execute(self, approved_action):
        if not self._execution_lock.acquire(blocking=False):
            raise RobotInterfaceError("another action is already executing")
        start = _utc()
        try:
            name = approved_action.get("name")
            params = approved_action.get("parameters", {})
            if not isinstance(params, dict):
                raise RobotInterfaceError("parameters must be an object")
            if name in {"safe_stop", "observe_again", "ask_human"}:
                if params:
                    raise RobotInterfaceError("stationary actions accept no motion parameters")
                with self._driver_lock:
                    self.driver.stop()
                # The application handles observation/human requests; never pretend completed.
                if name == "safe_stop":
                    return self._receipt("success", "Stop command submitted", start)
                return self._receipt("aborted", f"Stopped; application must handle {name}", start,
                                     requested_action=name)
            if name not in {"move_forward", "slow_down", "turn_left", "turn_right"}:
                raise RobotInterfaceError(f"unsupported action: {name!r}")
            vx, vy, yaw, duration = self._motion(name, params)
            deadline = time.monotonic() + duration
            status, detail = "success", "Bounded velocity command completed; arrival not verified"
            failure = None
            stop_error = None
            submissions = 0
            try:
                while time.monotonic() < deadline:
                    if self._stop_requested.is_set():
                        raise RobotInterfaceError("emergency stop is latched")
                    obs = self.observe()
                    guard = apply_safety_guard({"robot": obs.robot, "environment": obs.environment,
                                                "available_actions": [name, "safe_stop", "observe_again"]},
                                               {"selected_action": approved_action})
                    if guard["status"] != "approved":
                        raise RobotInterfaceError(f"motion guard: {guard['hard_rule_violations']}")
                    with self._driver_lock:
                        if self._stop_requested.is_set():
                            raise RobotInterfaceError("emergency stop is latched")
                        if time.monotonic() >= deadline:
                            raise RobotInterfaceError("command expired before transmission")
                        self.driver.move(vx, vy, yaw)
                        submissions += 1
                    self._stop_requested.wait(min(self.driver.period_s, max(0, deadline - time.monotonic())))
            except Exception as exc:
                failure = str(exc)
                status, detail = "failure", failure
                self._stop_requested.set()
            finally:
                try:
                    with self._driver_lock:
                        self.driver.stop()
                except Exception as exc:
                    stop_error = str(exc)
                    self._stop_requested.set()
            if stop_error:
                status, detail = "failure", f"{detail}; stop submission failed: {stop_error}"
            elif self._stop_requested.is_set() and status == "success":
                status, detail = "aborted", "Interrupted by emergency stop"
            elif submissions == 0 and status == "success":
                status, detail = "failure", "Command expired without a motion submission"
            return self._receipt(status, detail, start, failure_reason=failure or stop_error or (
                                     detail if status == "failure" else "未发生失败"),
                                 stop_error=stop_error, commanded_velocity=[vx, vy, yaw],
                                 requested_duration_s=duration, motion_submissions=submissions)
        finally:
            self._execution_lock.release()

    def emergency_stop(self, reason):
        self._stop_requested.set()
        start = _utc()
        try:
            with self._driver_lock:
                self.driver.stop()
            return self._receipt("aborted", reason, start, stop_submitted=True)
        except Exception as exc:
            return self._receipt("failure", f"{reason}; stop failed: {exc}", start,
                                 stop_submitted=False, failure_reason=str(exc))

    def reset_emergency_stop(self):
        """Explicit operator recovery, only while idle and with valid fresh state."""
        if not self._execution_lock.acquire(blocking=False):
            raise RobotInterfaceError("cannot reset while action is executing")
        try:
            self.observe()
            with self._driver_lock:
                self.driver.stop()
                self._stop_requested.clear()
        finally:
            self._execution_lock.release()
