#!/usr/bin/env python3
"""航向偏置诊断：让机器人以 vx>0 直行，比较 get_pose() 报告的 yaw 与实际位移方向。
offset = atan2(dy,dx) - yaw_reported。写入 configs/embodiment/<emb>.yaml 的 yaw_offset_deg。"""
import argparse, math, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--embodiment", required=True); a, _ = ap.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default",
    os.environ.get("FENGWU_ASSETS", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1")))
from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from isaac.embodiments import load_embodiment_cfg, make_embodiment
cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{a.embodiment}.yaml"),
                          os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
cfg["yaw_offset_deg"] = 0.0   # 诊断时强制关掉偏置，量原始值
world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
GroundPlane(prim_path="/World/ground", size=40)
emb = make_embodiment(cfg); emb.spawn(world, position=np.array([0.0, 0.0])); world.reset()
world.add_physics_callback("robot", emb.on_physics_step)
emb.set_cmd(0, 0, 0)
for _ in range(max(cfg["settle_steps"], round(1.0 / cfg["physics_dt"]))): world.step(render=False)
p0, yaw0 = emb.get_pose()
emb.set_cmd(cfg["macro_speed"]["v_form"], 0, 0)
hz = round(1 / cfg["physics_dt"])
for _ in range(int(3.0 * hz)): world.step(render=False)
p1, yaw1 = emb.get_pose()
d = p1[:2] - p0[:2]
motion = math.atan2(d[1], d[0])
off = (motion - yaw1 + math.pi) % (2 * math.pi) - math.pi
print(f"DIAG_YAW {a.embodiment}: yaw_reported={math.degrees(yaw1):.1f}deg motion_dir={math.degrees(motion):.1f}deg "
      f"offset={math.degrees(off):.1f}deg moved={np.linalg.norm(d):.2f}m", flush=True)
app.close()
