#!/usr/bin/env python3
"""Spot 爬坡可行性：不同坡度的直坡道（升高 RISE m），前进指令直走，判定是否登顶不摔。
    python.sh scripts/diag_ramp.py --angles 8,11,14,17 --rise 2.8
"""
import argparse, math, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--angles", default="8,11,14,17"); ap.add_argument("--rise", type=float, default=2.8)
ap.add_argument("--embodiment", default="dog"); a, _ = ap.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaac.embodiments import load_embodiment_cfg, make_embodiment

cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{a.embodiment}.yaml"), os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
results = {}
for deg in [float(x) for x in a.angles.split(",")]:
    world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
    GroundPlane(prim_path="/World/ground", size=80)
    L = a.rise / math.tan(math.radians(deg))            # 坡道水平长度
    slope_len = math.hypot(L, a.rise)
    x0 = 2.0                                            # 坡道起点
    # 坡道：绕 y 轴旋转的长盒（厚 0.2），中心在斜面中点下方 0.1
    cx, cz = x0 + L / 2, a.rise / 2
    FixedCuboid(prim_path="/World/ramp", position=np.array([cx, 0.0, cz - 0.1]),
                orientation=euler_angles_to_quats(np.array([0.0, -deg, 0.0]), degrees=True),
                scale=np.array([slope_len, 3.0, 0.2]))
    FixedCuboid(prim_path="/World/landing", position=np.array([x0 + L + 3.0, 0.0, a.rise - 0.1]), scale=np.array([6.0, 3.0, 0.2]))
    emb = make_embodiment(cfg); emb.spawn(world, position=np.array([0.0, 0.0])); world.reset()
    world.add_physics_callback("robot", emb.on_physics_step); emb.set_cmd(0, 0, 0)
    hz = round(1 / cfg["physics_dt"])
    for _ in range(max(cfg["settle_steps"], hz)): world.step(render=False)
    v = cfg["macro_speed"]["v_form"] * 0.75
    emb.set_cmd(v, 0, 0)
    fell, top, steps = False, False, 0
    max_steps = int((L + 8) / v * 2.0 * hz)
    zmax = 0.0
    while steps < max_steps:
        world.step(render=False); steps += 1
        if steps % 25 == 0:
            pos, yaw = emb.get_pose(); zmax = max(zmax, float(pos[2]))
            # 航向保持：简单 P 控制回正到 yaw=0
            emb.set_cmd(v, 0, float(np.clip(-yaw * 1.5, -0.5, 0.5)))
            if emb.fallen(): fell = True; break
            if pos[0] > x0 + L + 1.5 and pos[2] > a.rise - 0.3: top = True; break
    pos, _ = emb.get_pose()
    results[deg] = dict(top=top, fell=fell, x=round(float(pos[0]), 2), z=round(float(pos[2]), 2), zmax=round(zmax, 2), sim_s=round(steps / hz, 1))
    print(f"RAMP {deg:>4}deg: top={top} fell={fell} final=({pos[0]:.1f},{pos[2]:.2f}) zmax={zmax:.2f} ramp_len={L:.1f}m sim={steps/hz:.1f}s", flush=True)
    world.clear_instance()
print("RAMP_SUMMARY", results, flush=True)
app.close()
