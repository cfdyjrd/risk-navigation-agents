#!/usr/bin/env python3
"""FPV 朝向诊断：同一位置 4 种设法各渲一帧，红色自发光柱在 +X 方向 2m 处。
正确的"+X 朝前"帧应看到红柱居中。产出 out/diag_fpv/*.png。"""

import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane, VisualCuboid
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.sensors.camera import Camera
from pxr import UsdLux

from isaac.cameras import _bind_emissive, grab

OUT = os.path.join(ROOT, "out", "diag_fpv")
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 25)
GroundPlane(prim_path="/World/ground", size=40)
UsdLux.DomeLight.Define(world.stage, "/World/dome").CreateIntensityAttr(1000)
VisualCuboid(prim_path="/World/pillar", translation=np.array([2.0, 0.0, 1.0]),
             scale=np.array([0.3, 0.3, 2.0]), color=np.array([1.0, 0.0, 0.0]))
_bind_emissive("/World/pillar", (1.0, 0.0, 0.0))
# 左侧参照物：绿色矮柱在 +Y 2m 处
VisualCuboid(prim_path="/World/pillarL", translation=np.array([0.0, 2.0, 0.5]),
             scale=np.array([0.3, 0.3, 1.0]), color=np.array([0.0, 1.0, 0.0]))
_bind_emissive("/World/pillarL", (0.0, 1.0, 0.0))

P = np.array([0.0, 0.0, 0.5])
cams = {}
# a) 构造时不传 orientation（M4 第一版的写法）
cams["a_default"] = Camera(prim_path="/World/cam_a", position=P, resolution=(480, 360))
# b) 构造时传 world 约定 identity（若构造 orientation 按 +X-前约定，这个应正对红柱）
cams["b_ctor_identity"] = Camera(prim_path="/World/cam_b", position=P,
                                 orientation=euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
                                 resolution=(480, 360))
# c) set_world_pose + camera_axes="world"
cams["c_setpose_world"] = Camera(prim_path="/World/cam_c", position=P, resolution=(480, 360))
# d) set_world_pose + camera_axes="usd"（USD 原生 -Z 视轴，预期看脚下/别处）
cams["d_setpose_usd"] = Camera(prim_path="/World/cam_d", position=P, resolution=(480, 360))

world.reset()
for c in cams.values():
    c.initialize()
cams["c_setpose_world"].set_world_pose(P, euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
                                       camera_axes="world")
cams["d_setpose_usd"].set_world_pose(P, euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
                                     camera_axes="usd")

from isaac.cameras import FpvFollowCam

follow = FpvFollowCam(offset_fwd=0.0, offset_z=0.0, alpha=1.0, pitch_deg=8.0, resolution=(480, 360))
follow.initialize()
follow.update(P, 0.0)   # 与 a/b/c 同位同向：应见红柱
cams["e_followcam"] = follow.cam

for _ in range(20):
    world.render()

import imageio.v2 as imageio

for name, c in cams.items():
    f = grab(c)
    if f is not None:
        imageio.imwrite(os.path.join(OUT, f"{name}.png"), f)
        print(f"DIAG {name} saved, center_rgb={f[180, 240].tolist()}", flush=True)
    else:
        print(f"DIAG {name} EMPTY", flush=True)

app.close()
