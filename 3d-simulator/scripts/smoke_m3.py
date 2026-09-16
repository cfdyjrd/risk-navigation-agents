#!/usr/bin/env python3
"""M3 冒烟：goto 编译器穿 1.2m 门洞 + 障碍 abort 探针（设计文档 §10 M3 的最小验收）。

    ~/IsaacSim/_build/linux-aarch64/release/python.sh scripts/smoke_m3.py [--embodiment dog]

场景：房 A [-3,0]x[-2,2] 与房 B [0,3]x[-2,2]，x=0 隔墙开 1.2m 门洞（y∈[-0.6,0.6]）。
验收：① goto B outcome=arrived（zone 判定 GT AABB）；② 门前净空读数合理；
③ 在面前 0.6m 放墙后 forward_2 以 obstacle_ahead abort。
"""

import argparse
import json
import os
import sys
import time

T0 = time.monotonic()

ap = argparse.ArgumentParser()
ap.add_argument("--embodiment", default="dog", choices=["car", "dog", "humanoid"])
ap.add_argument("--door-width", type=float, default=1.2, help="门洞宽（E17 窄门预演用 0.7）")
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

from isaac.embodiments import load_embodiment_cfg, make_embodiment
from isaac.goto_compiler import GotoCompiler, Topology
from isaac.middleware import MacroExecutor
from isaac.sensors import RaySensor

cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{args.embodiment}.yaml"), ASSETS)
with open(os.path.join(ROOT, "configs", "run.yaml")) as f:
    run_cfg = yaml.safe_load(f)
OUT = os.path.join(ROOT, "out", "smoke_m3")
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
GroundPlane(prim_path="/World/ground", size=40)

WALL_H, WALL_T = 0.5, 0.1


def wall(name, cx, cy, lx, ly):
    FixedCuboid(
        prim_path=f"/World/walls/{name}",
        position=np.array([cx, cy, WALL_H / 2]),
        scale=np.array([lx, ly, WALL_H]),
    )


# x=0 隔墙，门洞居中（默认 1.2m；--door-width 0.7 为 E17 窄门）
hw = args.door_width / 2
seg = 2.0 - hw
wall("div_s", 0.0, -(hw + seg / 2), WALL_T, seg)
wall("div_n", 0.0, hw + seg / 2, WALL_T, seg)
# 外圈（防机器人漂出场景）
wall("west", -3.0, 0.0, WALL_T, 4.0)
wall("east", 3.0, 0.0, WALL_T, 4.0)
wall("south", 0.0, -2.0, 6.0, WALL_T)
wall("north", 0.0, 2.0, 6.0, WALL_T)

emb = make_embodiment(cfg)
emb.spawn(world, position=np.array([-1.5, 0.0]))
world.reset()

world.add_physics_callback("robot", emb.on_physics_step)

# settle：零指令落地
emb.set_cmd(0.0, 0.0, 0.0)
for _ in range(max(cfg["settle_steps"], 1)):
    world.step(render=False)

sensors = RaySensor(emb)
executor = MacroExecutor(world, emb, sensors, run_cfg)
topo = Topology(
    rooms={"A": {"aabb": [-3, -2, 0, 2]}, "B": {"aabb": [0, -2, 3, 2]}},
    doors=[(("A", "B"), {"center": [0.0, 0.0]})],
)
compiler = GotoCompiler(executor, topo, zone_fn=lambda: topo.zone_of(emb.get_pose()[0][:2]), run_cfg=run_cfg)

rec = {"embodiment": args.embodiment, "door_width": args.door_width}

# ① 初始净空：门洞方向应有开阔读数
rec["initial_front_clearance_m"] = round(sensors.front_clearance(), 2)

# ② goto 穿门
t = time.monotonic()
r = compiler.run("B", macro_budget=30)
rec["goto"] = {**r, "wall_s": round(time.monotonic() - t, 1),
               "final_zone": topo.zone_of(emb.get_pose()[0][:2]),
               "final_pos": [round(float(x), 2) for x in emb.get_pose()[0][:2]]}

# ③ 障碍 abort 探针：面前 1.2m 横一堵墙（必须在射线起点 front_edge+offset 之外，
#    否则落入盲区、只能靠 stuck 兜底），forward_2 应 obstacle abort
pos, yaw = emb.get_pose()
front = pos[:2] + 1.2 * np.array([np.cos(yaw), np.sin(yaw)])
wall("probe", float(front[0]), float(front[1]), WALL_T if abs(np.cos(yaw)) > 0.7 else 1.6,
     1.6 if abs(np.cos(yaw)) > 0.7 else WALL_T)
world.step(render=False)
r3 = executor.execute("forward_2")
rec["obstacle_probe"] = r3

ok = (
    rec["goto"]["outcome"] == "arrived"
    and rec["goto"]["final_zone"] == "B"
    and r3["status"] == "aborted"
    and str(r3["reason"]).startswith("obstacle_ahead")
)
rec["ok"] = bool(ok)
rec["wall_total_s"] = round(time.monotonic() - T0, 1)

with open(os.path.join(OUT, f"{args.embodiment}_w{args.door_width}.json"), "w") as f:
    json.dump(rec, f, indent=2, ensure_ascii=False)
print("SMOKE_M3 " + json.dumps(rec, ensure_ascii=False))

app.close()
sys.exit(0 if ok else 1)
