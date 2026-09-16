#!/usr/bin/env python3
"""Spot 在 FixedCuboid 楼板（z=2.8）上行走 vs GroundPlane：有效速度对比（二层实测 0.1 m/s 异常慢）。"""
import os, sys, math
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaac.embodiments import load_embodiment_cfg, make_embodiment
from isaac.sensors import RaySensor
cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", "dog.yaml"), os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
for name, z in (("ground", 0.0), ("slab_2.8", 2.8), ("slab_0.2", 0.2)):
    world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
    GroundPlane(prim_path="/World/ground", size=60)
    if z > 0:
        FixedCuboid(prim_path="/World/slab", position=np.array([5.0, 0.0, z - 0.1]), scale=np.array([30.0, 8.0, 0.2]))
    emb = make_embodiment(cfg); emb.spawn(world, position=np.array([0.0, 0.0]), z_base=z); world.reset()
    world.add_physics_callback("robot", emb.on_physics_step); emb.set_cmd(0, 0, 0)
    hz = round(1 / cfg["physics_dt"])
    for _ in range(hz): world.step(render=False)
    rs = RaySensor(emb)
    p0, _ = emb.get_pose(); fr0 = rs.front_clearance()
    emb.set_cmd(cfg["macro_speed"]["v_form"], 0, 0)
    T = 5.0
    for _ in range(int(T * hz)): world.step(render=False)
    p1, yaw = emb.get_pose()
    print(f"DIAG_SLAB {name}: moved={np.linalg.norm(p1[:2]-p0[:2]):.2f}m in {T}s -> {np.linalg.norm(p1[:2]-p0[:2])/T:.2f} m/s | base_z {p0[2]:.2f}->{p1[2]:.2f} | front_clearance {fr0:.2f} fallen={emb.fallen()}", flush=True)
    world.clear_instance()
app.close()
