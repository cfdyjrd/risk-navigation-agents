#!/usr/bin/env python3
"""射线诊断：加载场景+形态，在出生点打印各方向射线命中的 prim 与距离。"""
import argparse, json, math, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--embodiment", default="car"); a, _ = ap.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
from isaacsim.core.api import World
from isaac.embodiments import load_embodiment_cfg, make_embodiment
from isaac.scene_builder import build
from isaac.sensors import RaySensor
from isaac.cameras import add_marker
cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{a.embodiment}.yaml"), os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
scene = json.load(open(a.scene))
world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
topo, meta = build(scene, world)
emb = make_embodiment(cfg); emb.spawn(world, position=np.array(meta["spawn_xy"]))
add_marker("/World/Robot/" + cfg["fpv_mount"]["link"], size_m=cfg["marker"]["size_m"], tri_m=cfg["marker"]["heading_tri_m"], z_offset={"car":0.12,"dog":0.30,"humanoid":0.45}[a.embodiment])
world.reset(); world.add_physics_callback("robot", emb.on_physics_step); emb.set_cmd(0,0,0)
for _ in range(max(cfg["settle_steps"], round(1.0 / cfg["physics_dt"]))): world.step(render=False)
rs = RaySensor(emb)
pos, yaw = emb.get_pose()
print(f"DIAG_RAY pose={pos.round(2).tolist()} yaw={math.degrees(yaw):.1f}", flush=True)
import carb as _c
for name, off in (("front", 0), ("+15", 15), ("-15", -15), ("left90", 90), ("right90", -90)):
    aa = yaw + math.radians(off); pitch = math.radians(rs.rc.get("down_pitch_deg", 0) or 0)
    d = np.array([math.cos(aa)*math.cos(pitch), math.sin(aa)*math.cos(pitch), math.sin(pitch)])
    start = rs.rc["front_edge_m"] + rs.rc["origin_forward_offset_m"]
    o = np.array([pos[0] + math.cos(yaw)*start, pos[1] + math.sin(yaw)*start, rs.rc["height_m"]])
    hit = rs._q.raycast_closest(_c.Float3(*o.tolist()), _c.Float3(*d.tolist()), rs.max_dist)
    print(f"DIAG_RAY {name}: origin={o.round(2).tolist()} dir={d.round(2).tolist()} hit={hit.get('hit')} dist={hit.get('distance')} prim={hit.get('collision')}", flush=True)
print("DIAG_RAY front_clearance=", rs.front_clearance(), "sides=", rs.side_clearance(), flush=True)
app.close()
