#!/usr/bin/env python3
"""M4 冒烟（设计文档 §10 M4）：正交投影验证 + TopDown 视觉定位误差 + 双视角视频。

    ~/IsaacSim/.../python.sh scripts/smoke_m4.py --embodiment dog

场景同 M3（两房间 + 1.2m 门洞）+ 机器人顶品红 marker + 四角定标色点 + god/fpv 双相机。
流程：标定单应 → goto B 全程 25fps 采帧出三路 mp4，每帧同时喂 TopDownLocalizer，
逐帧记录 (topdown_pose, gt_pose) 误差。
验收：单应标定成功；开阔区检出率 >90%；位置误差均值 <0.10m（观测取整粒度内）；
朝向误差均值 <10°；三路 mp4 非空。
"""

import argparse
import json
import math
import os
import sys
import time

T0 = time.monotonic()

ap = argparse.ArgumentParser()
ap.add_argument("--embodiment", default="dog", choices=["car", "dog", "humanoid"])
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ASSETS = os.environ.get("FENGWU_ASSETS", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", ASSETS)

import yaml
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from pxr import UsdLux

from isaac.cameras import FpvFollowCam, VideoSink, add_calib_dots, add_marker, grab, make_god_camera
from isaac.embodiments import load_embodiment_cfg, make_embodiment
from isaac.goto_compiler import GotoCompiler, Topology
from isaac.localizer import TopDownLocalizer
from isaac.middleware import MacroExecutor
from isaac.sensors import RaySensor

cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{args.embodiment}.yaml"), ASSETS)
with open(os.path.join(ROOT, "configs", "run.yaml")) as f:
    run_cfg = yaml.safe_load(f)
with open(os.path.join(ROOT, "configs", "localizer.yaml")) as f:
    loc_cfg = yaml.safe_load(f)
OUT = os.path.join(ROOT, "out", "smoke_m4", args.embodiment)
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
GroundPlane(prim_path="/World/ground", size=40)
UsdLux.DomeLight.Define(world.stage, "/World/dome").CreateIntensityAttr(1000)

WALL_H, WALL_T = 0.5, 0.1


def wall(name, cx, cy, lx, ly):
    FixedCuboid(prim_path=f"/World/walls/{name}", position=np.array([cx, cy, WALL_H / 2]),
                scale=np.array([lx, ly, WALL_H]))


wall("div_s", 0.0, -1.3, WALL_T, 1.4)
wall("div_n", 0.0, 1.3, WALL_T, 1.4)
wall("west", -3.0, 0.0, WALL_T, 4.0)
wall("east", 3.0, 0.0, WALL_T, 4.0)
wall("south", 0.0, -2.0, 6.0, WALL_T)
wall("north", 0.0, 2.0, 6.0, WALL_T)

CALIB = {"red": [-2.5, -1.5], "green": [2.5, -1.5], "blue": [2.5, 1.5], "yellow": [-2.5, 1.5]}
add_calib_dots(CALIB)

emb = make_embodiment(cfg)
emb.spawn(world, position=np.array([-1.5, 0.0]))
mount = cfg["fpv_mount"]
marker_z = {"car": 0.12, "dog": 0.30, "humanoid": 0.45}[args.embodiment]
add_marker(f"/World/Robot/{mount['link']}", size_m=cfg["marker"]["size_m"], z_offset=marker_z)
god, is_ortho = make_god_camera(world, center_xy=(0.0, 0.0), extent_m=7.5, resolution=(960, 720))
# 前向偏移必须出机身前缘（0.4m 在 Spot 头里，M4 实测整帧都是自己的头壳）
fpv = FpvFollowCam(offset_fwd=cfg["raycast"]["front_edge_m"] + 0.07,
                   offset_z=mount["translation"][2] + 0.05,
                   alpha=mount.get("alpha", 1.0), resolution=(960, 720))

world.reset()
god.initialize()
fpv.initialize()
world.add_physics_callback("robot", emb.on_physics_step)

# settle + 渲染 warmup（冲陈旧帧）
emb.set_cmd(0.0, 0.0, 0.0)
for _ in range(max(cfg["settle_steps"], 1)):
    world.step(render=False)
for _ in range(16):
    world.render()

rec = {"embodiment": args.embodiment, "orthographic": bool(is_ortho)}

topo = Topology(
    rooms={"A": {"aabb": [-3, -2, 0, 2]}, "B": {"aabb": [0, -2, 3, 2]}},
    doors=[(("A", "B"), {"center": [0.0, 0.0]})],
)
loc = TopDownLocalizer(topo, loc_cfg["topdown"], cfg["marker"]["size_m"], CALIB)
frame0 = grab(god)
rec["calibrated"] = bool(frame0 is not None and loc.calibrate(frame0))
if frame0 is not None:
    import imageio.v2 as imageio

    imageio.imwrite(os.path.join(OUT, "god_calib_frame.png"), frame0)

# goto B 全程采帧 + 定位误差流水
sink = VideoSink(OUT, fps=25)
errors, det = [], {"n": 0, "hit": 0}


def on_frame():
    fpv.update(*emb.get_pose())
    g, f = grab(god), fpv.grab()
    loc.update(g)
    det["n"] += 1
    gt_pos, gt_yaw = emb.get_pose()
    if not loc.stale:
        det["hit"] += 1
        p, y = loc.pose()
        errors.append([
            float(np.linalg.norm(p - gt_pos[:2])),
            abs(math.degrees((y - gt_yaw + math.pi) % (2 * math.pi) - math.pi)),
        ])
    sink.append(g, f)


sensors = RaySensor(emb)
render_every = max(1, round((1 / 25) / cfg["physics_dt"]))
executor = MacroExecutor(world, emb, sensors, run_cfg, render_every=render_every, frame_cb=on_frame)
compiler = GotoCompiler(executor, topo, zone_fn=lambda: topo.zone_of(emb.get_pose()[0][:2]), run_cfg=run_cfg)

t = time.monotonic()
r = compiler.run("B", macro_budget=30)
rec["goto"] = {**r, "wall_s": round(time.monotonic() - t, 1)}
sink.close()

if errors:
    e = np.asarray(errors)
    rec["localizer"] = {
        "frames": det["n"], "detected": det["hit"],
        "det_rate": round(det["hit"] / max(det["n"], 1), 3),
        "pos_err_mean_m": round(float(e[:, 0].mean()), 3),
        "pos_err_p95_m": round(float(np.percentile(e[:, 0], 95)), 3),
        "yaw_err_mean_deg": round(float(e[:, 1].mean()), 1),
        "confidence_last": loc.confidence(),
    }
else:
    rec["localizer"] = {"frames": det["n"], "detected": 0, "det_rate": 0.0}

vids = {v: os.path.getsize(os.path.join(OUT, v)) for v in ("god.mp4", "fpv.mp4", "sbs.mp4")
        if os.path.exists(os.path.join(OUT, v))}
rec["videos_bytes"] = vids
rec["frames_written"] = sink.frames

lz = rec["localizer"]
ok = (
    rec["calibrated"]
    and rec["goto"]["outcome"] == "arrived"
    and lz.get("det_rate", 0) > 0.9
    and lz.get("pos_err_mean_m", 9) < 0.10
    and lz.get("yaw_err_mean_deg", 99) < 10.0
    and len(vids) == 3
    and all(v > 10000 for v in vids.values())
)
rec["ok"] = bool(ok)
rec["wall_total_s"] = round(time.monotonic() - T0, 1)

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(rec, f, indent=2, ensure_ascii=False)
print("SMOKE_M4 " + json.dumps(rec, ensure_ascii=False), flush=True)

app.close()
sys.exit(0 if ok else 1)
