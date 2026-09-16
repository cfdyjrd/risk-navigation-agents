#!/usr/bin/env python3
"""大场景俯视相机取景诊断：加载 from2d 场景，按不同 extent 建正交相机各渲一帧，统计彩色像素。"""
import json, os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import carb, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
carb.settings.get_settings().set("/persistent/isaac/asset_root/default",
    os.environ.get("FENGWU_ASSETS", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1")))
from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom
from isaac.scene_builder import build, scene_bbox
from isaac.cameras import grab
import imageio.v2 as imageio

scene = json.load(open(sys.argv[1]))
world = World(stage_units_in_meters=1.0, physics_dt=1/100, rendering_dt=1/25)
topo, meta = build(scene, world)
x0, y0, x1, y1 = meta["bbox"]; cx, cy = (x0+x1)/2, (y0+y1)/2
print("DIAG bbox", meta["bbox"], "center", cx, cy, flush=True)
cams = {}
for name, extent, ortho in [("ortho_full", max(x1-x0, y1-y0)+1.5, True), ("ortho_small", 7.5, True), ("persp_high", None, False)]:
    cam = Camera(prim_path=f"/World/cam_{name}", position=np.array([cx, cy, 12.0 if ortho else 30.0]),
                 orientation=euler_angles_to_quats(np.array([0.0, 90.0, 0.0]), degrees=True), resolution=(960, 720))
    if ortho:
        uc = UsdGeom.Camera(world.stage.GetPrimAtPath(f"/World/cam_{name}"))
        uc.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        uc.GetHorizontalApertureAttr().Set(extent*10.0); uc.GetVerticalApertureAttr().Set(extent*0.75*10.0)
        print("DIAG", name, "aperture", uc.GetHorizontalApertureAttr().Get(), "clip", uc.GetClippingRangeAttr().Get(), flush=True)
    cams[name] = cam
world.reset()
for c in cams.values(): c.initialize()
for _ in range(16): world.render()
os.makedirs(os.path.join(ROOT, "out", "diag_god"), exist_ok=True)
for name, c in cams.items():
    f = grab(c)
    if f is None: print("DIAG", name, "EMPTY", flush=True); continue
    imageio.imwrite(os.path.join(ROOT, "out", "diag_god", f"{name}.png"), f)
    r,g,b = f[:,:,0].astype(int), f[:,:,1].astype(int), f[:,:,2].astype(int)
    sat = int(((r>170)&(g<90)&(b<90)).sum()+((r<90)&(g>150)&(b<90)).sum()+((r<90)&(g<110)&(b>160)).sum()+((r>170)&(g>150)&(b<100)).sum())
    dark = int((r<60).sum())
    print(f"DIAG {name}: saturated_px={sat} dark_px(walls)={dark} mean={f.mean():.0f}", flush=True)
app.close()
