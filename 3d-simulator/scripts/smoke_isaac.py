#!/usr/bin/env python3
"""M1 冒烟（设计文档 §10 M1）：headless 建场 + Spot 走 2m + 俯视抓帧 + 计时。

用本机构建直跑（保底路径，ISAAC_LAUNCHER=local）：
    ~/IsaacSim/_build/linux-aarch64/release/python.sh scripts/smoke_isaac.py

前置：scripts/mirror_assets.py 已完成（资产在 ~/isaacsim_assets）。
产出：out/smoke/summary.json、out/smoke/god_frame.png。
跑两遍对比 wall_total 即冷/热启动差（shader cache）。
"""

import json
import os
import sys
import time

T0 = time.monotonic()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
T_BOOT = time.monotonic() - T0

import carb
import numpy as np

ASSETS = os.environ.get("FENGWU_ASSETS", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", ASSETS)

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdLux

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "smoke")
os.makedirs(OUT, exist_ok=True)

WALK_TARGET_M = 2.0
CMD_FWD = np.array([1.0, 0.0, 0.0])  # 保守速度，安全上界 2.0 内（§2.1）
SETTLE_STEPS = 60                    # 落地站稳（§2.1 settle）
MAX_WALK_STEPS = 5000                # 1/500 dt 下 10s 仿真时间兜底
FALL_Z = 0.3

timings, result = {"app_boot_s": round(T_BOOT, 2)}, {}

root = get_assets_root_path()
assert root and root.rstrip("/") == ASSETS.rstrip("/"), f"assets root 未指向本地镜像: {root} != {ASSETS}"

t = time.monotonic()
world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 25)
GroundPlane(prim_path="/World/ground", size=40)
dome = UsdLux.DomeLight.Define(world.stage, "/World/dome")  # headless 无默认光源，否则黑帧
dome.CreateIntensityAttr(1000)
spot = SpotFlatTerrainPolicy(prim_path="/World/Spot", name="Spot", position=np.array([0.0, 0.0, 0.8]))
god = Camera(
    prim_path="/World/god_cam",
    position=np.array([0.0, 0.0, 12.0]),
    orientation=euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True),
    resolution=(960, 720),
)
timings["scene_build_s"] = round(time.monotonic() - t, 2)

t = time.monotonic()
world.reset()
god.initialize()  # 必须在 world.reset() 之后（§6.2）
timings["world_reset_s"] = round(time.monotonic() - t, 2)

state = {"first": True, "cmd": np.zeros(3)}


def on_physics_step(dt):
    if state["first"]:
        spot.initialize()
        state["first"] = False
    else:
        spot.forward(dt, state["cmd"])


world.add_physics_callback("physics_step", on_physics_step)

# settle：零指令落地
t = time.monotonic()
for _ in range(SETTLE_STEPS):
    world.step(render=False)
timings["settle_s"] = round(time.monotonic() - t, 2)

# 走 2m：位姿闭环判完成（§3.2 不用指令积分）
start_xy = spot.robot.get_world_pose()[0][:2].copy()
state["cmd"] = CMD_FWD
t = time.monotonic()
steps = 0
dist = 0.0
fell = False
while steps < MAX_WALK_STEPS:
    world.step(render=False)
    steps += 1
    if steps % 50 == 0:
        pos = spot.robot.get_world_pose()[0]
        dist = float(np.linalg.norm(pos[:2] - start_xy))
        if pos[2] < FALL_Z:
            fell = True
            break
        if dist >= WALK_TARGET_M:
            break
state["cmd"] = np.zeros(3)
for _ in range(12):  # dog 落零速站稳（§3.2）
    world.step(render=False)
wall_walk = time.monotonic() - t
sim_walk = steps / 500.0
timings["walk_wall_s"] = round(wall_walk, 2)
timings["walk_sim_s"] = round(sim_walk, 2)
timings["rtf"] = round(sim_walk / wall_walk, 3)

result["walked_m"] = round(dist, 3)
result["fell"] = fell
result["final_z"] = round(float(spot.robot.get_world_pose()[0][2]), 3)

# 抓帧：12 帧 warmup 冲陈旧帧（IsaacLab #6250）+ 判空判维（§6.2）
t = time.monotonic()
for _ in range(12):
    world.step(render=True)
rgba = god.get_rgba()
frame_ok = rgba is not None and getattr(rgba, "ndim", 0) == 3
if frame_ok:
    import imageio.v2 as imageio

    imageio.imwrite(os.path.join(OUT, "god_frame.png"), rgba[:, :, :3])
timings["capture_s"] = round(time.monotonic() - t, 2)
result["frame_ok"] = bool(frame_ok)
result["frame_shape"] = list(rgba.shape) if frame_ok else None

timings["wall_total_s"] = round(time.monotonic() - T0, 2)
ok = result["walked_m"] >= WALK_TARGET_M and not fell and frame_ok
summary = {"ok": ok, "timings": timings, "result": result, "assets_root": root}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("SMOKE_SUMMARY " + json.dumps(summary))

app.close()
sys.exit(0 if ok else 1)
