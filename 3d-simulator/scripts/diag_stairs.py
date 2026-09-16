#!/usr/bin/env python3
"""Spot 上楼梯可行性：不同踏步高（riser）的直跑楼梯，升高 RISE m；前进指令 + 航向保持。
    python.sh scripts/diag_stairs.py --risers 0.06,0.10,0.14 --tread 0.35 --rise 2.8
"""
import argparse, math, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--risers", default="0.06,0.10,0.14"); ap.add_argument("--tread", type=float, default=0.35)
ap.add_argument("--rise", type=float, default=2.8); ap.add_argument("--embodiment", default="dog"); a, _ = ap.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaac.embodiments import load_embodiment_cfg, make_embodiment
cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{a.embodiment}.yaml"), os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
results = {}
for riser in [float(x) for x in a.risers.split(",")]:
    world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
    GroundPlane(prim_path="/World/ground", size=80)
    n = int(math.ceil(a.rise / riser)); x0 = 2.0
    for i in range(n):   # 第 i 级：顶面 z=(i+1)*riser，x 从 x0+i*tread 到 x0+(i+1)*tread；做成实心阶梯（每级一个到地面的盒）
        top = (i + 1) * riser
        FixedCuboid(prim_path=f"/World/stairs/s{i}", position=np.array([x0 + (i + 0.5) * a.tread, 0.0, top / 2]),
                    scale=np.array([a.tread, 3.0, top]))
    L = n * a.tread
    FixedCuboid(prim_path="/World/landing", position=np.array([x0 + L + 3.0, 0.0, n * riser / 2]), scale=np.array([6.0, 3.0, n * riser]))
    emb = make_embodiment(cfg); emb.spawn(world, position=np.array([0.0, 0.0])); world.reset()
    world.add_physics_callback("robot", emb.on_physics_step); emb.set_cmd(0, 0, 0)
    hz = round(1 / cfg["physics_dt"])
    for _ in range(max(cfg["settle_steps"], hz)): world.step(render=False)
    v = cfg["macro_speed"]["v_form"] * 0.6
    emb.set_cmd(v, 0, 0)
    fell, top_reached, steps, zmax = False, False, 0, 0.0
    max_steps = int((L + 8) / v * 2.5 * hz)
    while steps < max_steps:
        world.step(render=False); steps += 1
        if steps % 25 == 0:
            pos, yaw = emb.get_pose(); zmax = max(zmax, float(pos[2]))
            emb.set_cmd(v, 0, float(np.clip(-yaw * 1.5, -0.5, 0.5)))
            if emb.fallen(): fell = True; break
            if pos[0] > x0 + L + 1.5 and pos[2] > n * riser - 0.3: top_reached = True; break
    pos, _ = emb.get_pose()
    results[riser] = dict(top=top_reached, fell=fell, x=round(float(pos[0]), 2), z=round(float(pos[2]), 2), zmax=round(zmax, 2), n=n)
    print(f"STAIRS riser={riser:.2f} tread={a.tread}: top={top_reached} fell={fell} final=({pos[0]:.1f},{pos[2]:.2f}) zmax={zmax:.2f} steps={n} len={L:.1f}m sim={steps/hz:.1f}s", flush=True)
    world.clear_instance()
print("STAIRS_SUMMARY", results, flush=True)
app.close()
