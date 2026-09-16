"""三形态统一接入（设计文档 §2.1）。Isaac 进程内使用（python.sh，Py3.11）。

Embodiment 约定（指令走物理回调——与官方示例同构，保证 render 与否物理一致）：
  spawn(world, position)         建 prim。car 走 world.scene.add（生命周期归 scene），
                                 policy 机器人直接构造——两套模式类内隔离，不混用。
  on_physics_step(dt)            注册为 world 物理回调：首步 initialize()（不能在
                                 world.reset() 前调），之后每物理步下发当前指令。
  set_cmd(vx, vy, wz)            设定持续速度指令（cfg.cmd_limit 限幅）；由回调消费。
  get_pose() -> (pos3, yaw)      get_world_pose() 实测。仅供宏内环/GTLocalizer/标定，
                                 认知层观测一律走 Localizer（§3.2 注）。
  estop()                        指令记零。

换形态 = make_embodiment(cfg) 一个参数，其余代码零改动。
"""

import math

import numpy as np
import yaml


def load_embodiment_cfg(path, assets_root):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("usd_path", "policy"):
        if isinstance(cfg.get(k), str):
            cfg[k] = cfg[k].replace("{assets}", assets_root)
    return cfg


def _quat_to_yaw_tilt(q):
    """wxyz 四元数 -> (yaw, 倾角°)。倾角 = 机体 +Z 与世界 +Z 的夹角。"""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    up_z = 1 - 2 * (x * x + y * y)  # R[2,2]
    tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    return float(yaw), float(tilt)


class Embodiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self._cmd = np.zeros(3)
        self._ready = False

    # -- 核心接口 --
    def spawn(self, world, position, z_base=0.0, yaw_deg=0.0):
        raise NotImplementedError

    @staticmethod
    def _yaw_quat(yaw_deg):
        h = math.radians(yaw_deg) / 2.0
        return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])   # wxyz，绕 z

    def on_physics_step(self, dt):
        """注册为物理回调：首步 initialize，之后逐步下发当前指令。"""
        if not self._ready:
            self._initialize()
            self._ready = True
        else:
            self._apply(dt)

    def request_reinit(self):
        """world.reset() 之后调用：下个物理步重新 initialize（官方 reset_needed 模式）。"""
        self._ready = False

    def set_cmd(self, vx, vy, wz):
        lim = self.cfg["cmd_limit"]
        self._cmd = np.array([
            np.clip(vx, -lim["vx"], lim["vx"]),
            np.clip(vy, -lim["vy"], lim["vy"]),
            np.clip(wz, -lim["wz"], lim["wz"]),
        ])

    def _initialize(self):
        pass

    def get_pose(self):
        """(pos3, yaw)。yaw 加 cfg.yaw_offset_deg 校正：部分 USD（Jetbot）的本体四元数
        前向轴与实际驱动方向有固定偏置（scripts/diag_yaw.py 实测），不校正会把门走反。"""
        pos, quat = self._articulation().get_world_pose()
        yaw, _ = _quat_to_yaw_tilt(quat)
        off = math.radians(self.cfg.get("yaw_offset_deg", 0.0) or 0.0)
        yaw = (yaw + off + math.pi) % (2 * math.pi) - math.pi
        return np.asarray(pos), yaw

    def estop(self):
        self._cmd = np.zeros(3)

    # -- 辅助 --
    def fallen(self):
        wd = self.cfg.get("fall_watchdog")
        if not wd:
            return False
        pos, quat = self._articulation().get_world_pose()
        _, tilt = _quat_to_yaw_tilt(quat)
        return bool(pos[2] < wd["min_base_z"] or tilt > wd["max_tilt_deg"])

    def link_names(self):
        """机器人 prim 一级子节点名——FPV 挂载 link 用 spawn 后实名确认（§6.1）。"""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path)
        return [c.GetName() for c in prim.GetChildren()]

    def _articulation(self):
        raise NotImplementedError

    def _apply(self, dt):
        raise NotImplementedError


class CarEmbodiment(Embodiment):
    """车 = Jetbot，WheeledRobot + DifferentialController，零 checkpoint。"""

    prim_path = "/World/Robot"

    def spawn(self, world, position, z_base=0.0, yaw_deg=0.0):
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        c = self.cfg["controller"]
        self.robot = world.scene.add(
            WheeledRobot(
                prim_path=self.prim_path,
                name="robot",
                wheel_dof_names=self.cfg["wheel_dof_names"],
                create_robot=True,
                usd_path=self.cfg["usd_path"],
                position=np.array([*position[:2], z_base + self.cfg["spawn_z"]]),
                orientation=self._yaw_quat(yaw_deg),
            )
        )
        self.controller = DifferentialController(
            name="diff",
            wheel_radius=c["wheel_radius"],
            wheel_base=c["wheel_base"],
            max_linear_speed=c["max_linear_speed"],
            max_angular_speed=c["max_angular_speed"],
        )

    def _apply(self, dt):
        # 差速不能横移：vy 恒弃（§2.1）
        self.robot.apply_wheel_actions(self.controller.forward(command=[self._cmd[0], self._cmd[2]]))

    def _articulation(self):
        return self.robot


class _PolicyEmbodiment(Embodiment):
    """狗/人形共用：官方预训练 FlatTerrainPolicy。"""

    prim_path = "/World/Robot"
    policy_cls_name = None

    def spawn(self, world, position, z_base=0.0, yaw_deg=0.0):
        from isaacsim.robot.policy.examples import robots as policy_robots

        cls = getattr(policy_robots, self.policy_cls_name)
        self.policy = cls(
            prim_path=self.prim_path,
            name="robot",
            position=np.array([*position[:2], z_base + self.cfg["spawn_z"]]),
            orientation=self._yaw_quat(yaw_deg),
        )

    def _initialize(self):
        # H1 的 initialize() 不收 physics_sim_view；统一按无参调（官方三示例同）
        self.policy.initialize()

    def _apply(self, dt):
        self.policy.forward(dt, self._cmd)

    def _articulation(self):
        return self.policy.robot


class DogEmbodiment(_PolicyEmbodiment):
    policy_cls_name = "SpotFlatTerrainPolicy"


class HumanoidEmbodiment(_PolicyEmbodiment):
    policy_cls_name = "H1FlatTerrainPolicy"


_REGISTRY = {"car": CarEmbodiment, "dog": DogEmbodiment, "humanoid": HumanoidEmbodiment}


def make_embodiment(cfg):
    return _REGISTRY[cfg["embodiment"]](cfg)
