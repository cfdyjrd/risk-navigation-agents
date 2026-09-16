#!/usr/bin/env python3
"""Spot 原地转向：GroundPlane vs FixedCuboid 楼板（默认材质 / 高摩擦材质）。二层实测 turn 宏全部超时。"""
import os, sys, math
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, GroundPlane
from isaacsim.core.api.materials import PhysicsMaterial
from isaac.embodiments import load_embodiment_cfg, make_embodiment
cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", "dog.yaml"), os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
from pxr import UsdPhysics, PhysxSchema
import omni.usd
VARIANTS = [("box_patch(default)", {}), ("box_twoDirectional", {"friction": "twoDirectional"}),
            ("box_oneDirectional", {"friction": "oneDirectional"}), ("box_PGS", {"solver": "PGS"}),
            ("box_iters64", {"iters": 64}), ("box_contactOffset", {"contact": True})]
for name, opt in VARIANTS:
    world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
    GroundPlane(prim_path="/World/ground", size=60)
    z = 2.8
    slab = FixedCuboid(prim_path="/World/slab", position=np.array([0.0, 0.0, z - 0.1]), scale=np.array([20.0, 20.0, 0.2]))
    pc = world.get_physics_context()
    stage = omni.usd.get_context().get_stage()
    scene_prim = stage.GetPrimAtPath(pc.prim_path) if hasattr(pc, "prim_path") else None
    if opt.get("friction"):
        api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim) if scene_prim else None
        if api: api.CreateFrictionTypeAttr().Set(opt["friction"])
    if opt.get("solver"):
        pc.set_solver_type(opt["solver"])
    if opt.get("iters"):
        api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim) if scene_prim else None
        if api: api.CreateMinPositionIterationCountAttr().Set(opt["iters"]); api.CreateMaxPositionIterationCountAttr().Set(opt["iters"])
    if opt.get("contact"):
        prim = stage.GetPrimAtPath("/World/slab")
        capi = PhysxSchema.PhysxCollisionAPI.Apply(prim); capi.CreateContactOffsetAttr().Set(0.005); capi.CreateRestOffsetAttr().Set(0.0)
    print("DIAG_TURN setup", name, "scene_prim", pc.prim_path if hasattr(pc, "prim_path") else "?", flush=True)
    emb = make_embodiment(cfg); emb.spawn(world, position=np.array([0.0, 0.0]), z_base=z); world.reset()
    world.add_physics_callback("robot", emb.on_physics_step); emb.set_cmd(0, 0, 0)
    hz = round(1 / cfg["physics_dt"])
    for _ in range(hz): world.step(render=False)
    _, y0 = emb.get_pose()
    w = cfg["macro_speed"]["w_form"]
    emb.set_cmd(0, 0, w)
    acc, prev = 0.0, y0
    for i in range(int(3.0 * hz)):
        world.step(render=False)
        if i % 25 == 0:
            _, y = emb.get_pose(); acc += (y - prev + math.pi) % (2 * math.pi) - math.pi; prev = y
    p, y = emb.get_pose(); acc += (y - prev + math.pi) % (2 * math.pi) - math.pi
    print(f"DIAG_TURN {name:18s}: cmd wz={w} for 3s -> turned {math.degrees(acc):6.1f} deg (expect ~{math.degrees(w*3):.0f}) base_z={p[2]:.2f} fallen={emb.fallen()}", flush=True)
    world.clear_instance()
app.close()
