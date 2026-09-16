#!/usr/bin/env python3
"""最小渲染冒烟：SimulationApp + World + GroundPlane + Camera + 抓帧。
用于隔离容器内 segfault 是渲染/replicator 链路问题还是机器人加载问题。"""

import json
import time

T0 = time.monotonic()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.sensors.camera import Camera
from pxr import UsdLux

world = World(stage_units_in_meters=1.0, physics_dt=1 / 100, rendering_dt=1 / 25)
GroundPlane(prim_path="/World/ground", size=40)
UsdLux.DomeLight.Define(world.stage, "/World/dome").CreateIntensityAttr(1000)
print("MIN_STAGE built", flush=True)
cam = Camera(
    prim_path="/World/god_cam",
    position=np.array([0.0, 0.0, 12.0]),
    orientation=euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True),
    resolution=(320, 240),
)
world.reset()
print("MIN_RESET ok", flush=True)
cam.initialize()
print("MIN_CAM_INIT ok", flush=True)
for i in range(12):
    world.step(render=True)
print("MIN_RENDER ok", flush=True)
rgba = cam.get_rgba()
ok = rgba is not None and getattr(rgba, "ndim", 0) == 3
print("MIN_SUMMARY " + json.dumps({"ok": bool(ok), "shape": list(rgba.shape) if ok else None,
                                   "wall_s": round(time.monotonic() - T0, 1)}), flush=True)
app.close()
