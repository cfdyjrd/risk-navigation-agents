#!/usr/bin/env python3
"""FPV 诊断二：验证"物理步进后 USD set_world_pose 不再影响渲染（Fabric 接管）"，
并对比两种修法。红色自发光柱在 +X 2m。产出 out/diag_fpv2/*.png + DIAG2 行。

f) 步进物理 50 步后 set_world_pose(camera_axes=world) 指向红柱 → 若灰 = Fabric 接管实锤
g) usdrt Rt.Xformable 直写 Fabric world transform → 若红 = 修法一有效
"""

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

OUT = os.path.join(ROOT, "out", "diag_fpv2")
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 25)
GroundPlane(prim_path="/World/ground", size=40)
UsdLux.DomeLight.Define(world.stage, "/World/dome").CreateIntensityAttr(1000)
VisualCuboid(prim_path="/World/pillar", translation=np.array([2.0, 0.0, 1.0]),
             scale=np.array([0.3, 0.3, 2.0]), color=np.array([1.0, 0.0, 0.0]))
_bind_emissive("/World/pillar", (1.0, 0.0, 0.0))

P = np.array([0.0, 0.0, 0.5])
AWAY = np.array([0.0, 5.0, 0.5])  # 初始故意朝错位置放

cam_f = Camera(prim_path="/World/cam_f", position=AWAY, resolution=(480, 360))
cam_g = Camera(prim_path="/World/cam_g", position=AWAY, resolution=(480, 360))
world.reset()
cam_f.initialize()
cam_g.initialize()

# 关键差异：先步进物理
for _ in range(50):
    world.step(render=False)

Q = euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True)

# f) USD 写
cam_f.set_world_pose(P, Q, camera_axes="world")

# g) usdrt 直写 Fabric（世界变换属性）
try:
    import omni.usd
    from usdrt import Gf as RtGf, Rt, Usd as RtUsd

    stage_rt = RtUsd.Stage.Attach(omni.usd.get_context().get_stage_id())
    prim_rt = stage_rt.GetPrimAtPath("/World/cam_g")
    xf = Rt.Xformable(prim_rt)
    # +X-前 视轴的 USD 相机姿态 = 世界约定 identity 的 USD 等价四元数：
    # 从 cam_f 此刻的 USD 姿态读取（set_world_pose 已把换算好的 USD quat 写进 USD 层）
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdGeom

    usd_prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/cam_f")
    m = UsdGeom.Xformable(usd_prim).ComputeLocalToWorldTransform(0)
    q = m.ExtractRotationQuat()
    xf.SetWorldXformFromUsd()  # 先建立 fabric xform
    xf.GetWorldPositionAttr().Set(RtGf.Vec3d(float(P[0]), float(P[1]), float(P[2])))
    xf.GetWorldOrientationAttr().Set(RtGf.Quatf(q.GetReal(), *[float(v) for v in q.GetImaginary()]))
    print("DIAG2 usdrt write ok", flush=True)
except Exception as e:
    print(f"DIAG2 usdrt FAILED: {e}", flush=True)

for _ in range(6):
    world.render()

import imageio.v2 as imageio

for name, c in (("f_usd_after_stepping", cam_f), ("g_usdrt", cam_g)):
    fr = grab(c)
    if fr is not None:
        imageio.imwrite(os.path.join(OUT, f"{name}.png"), fr)
        print(f"DIAG2 {name} center_rgb={fr[180, 240].tolist()}", flush=True)
    else:
        print(f"DIAG2 {name} EMPTY", flush=True)

app.close()
