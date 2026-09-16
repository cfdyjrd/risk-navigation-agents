#!/usr/bin/env python3
"""M2 冒烟（设计文档 §10 M2）：单形态直线 2m + 原地 90°，位姿闭环判定 + link 名确认。

    ~/IsaacSim/_build/linux-aarch64/release/python.sh scripts/smoke_m2.py --embodiment dog

产出 out/smoke_m2/<embodiment>.json；exit 0 = 通过。
验收：直线 ≥2m 不摔、转角 90°±15°（实测值进标定记录，门槛软化 §3.5）、link 名打印。
"""

import argparse
import json
import os
import sys
import time

T0 = time.monotonic()

ap = argparse.ArgumentParser()
ap.add_argument("--embodiment", required=True, choices=["car", "dog", "humanoid"])
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ASSETS = os.environ.get("FENGWU_ASSETS", os.path.expanduser("~/isaacsim_assets/Assets/Isaac/5.1"))
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", ASSETS)

from isaacsim.core.api import World
from isaacsim.core.api.objects import GroundPlane
from pxr import UsdLux

from isaac.embodiments import load_embodiment_cfg, make_embodiment

cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{args.embodiment}.yaml"), ASSETS)
OUT = os.path.join(ROOT, "out", "smoke_m2")
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
GroundPlane(prim_path="/World/ground", size=40)
UsdLux.DomeLight.Define(world.stage, "/World/dome").CreateIntensityAttr(1000)

emb = make_embodiment(cfg)
emb.spawn(world, position=np.array([0.0, 0.0]))
world.reset()

world.add_physics_callback("robot", emb.on_physics_step)
state = {"cmd": None}  # 兼容下方 state["cmd"] 写法：转发到 emb.set_cmd


class _S(dict):
    def __setitem__(self, k, v):
        if k == "cmd":
            emb.set_cmd(*v)
        super().__setitem__(k, v)


state = _S(cmd=(0.0, 0.0, 0.0))
emb.set_cmd(0.0, 0.0, 0.0)

HZ = round(1 / cfg["physics_dt"])
rec = {"embodiment": args.embodiment, "links": []}


def run_until(check_every, timeout_s, done_fn):
    """步进物理直到 done_fn 返回真值/摔倒/超时。返回 (结果, 步数, fell)。"""
    steps = 0
    while steps < timeout_s * HZ:
        world.step(render=False)
        steps += 1
        if steps % check_every == 0:
            if emb.fallen():
                return None, steps, True
            r = done_fn()
            if r is not None:
                return r, steps, False
    return None, steps, False


def hold_zero(frames):
    state["cmd"] = (0.0, 0.0, 0.0)
    for _ in range(frames):
        world.step(render=False)


# settle
for _ in range(cfg["settle_steps"]):
    world.step(render=False)

rec["links"] = emb.link_names()

# 阶段 1：直线 2m（位姿闭环）
v = cfg["macro_speed"]["v_form"]
start_xy, _ = emb.get_pose()[0][:2].copy(), None
start_xy = emb.get_pose()[0][:2].copy()
state["cmd"] = (v, 0.0, 0.0)
t = time.monotonic()
dist, steps1, fell1 = run_until(
    check_every=max(HZ // 20, 1),
    timeout_s=20,
    done_fn=lambda: (lambda d: d if d >= 2.0 else None)(float(np.linalg.norm(emb.get_pose()[0][:2] - start_xy))),
)
hold_zero(cfg["zero_hold_frames"])
rec["straight"] = {
    "walked_m": round(dist or float(np.linalg.norm(emb.get_pose()[0][:2] - start_xy)), 3),
    "sim_s": round(steps1 / HZ, 2),
    "wall_s": round(time.monotonic() - t, 2),
    "fell": bool(fell1),
}

# 阶段 2：原地左转 90°（yaw 逐步 unwrap 累加，§3.2）
w = cfg["macro_speed"]["w_form"]
acc = {"prev": emb.get_pose()[1], "sum": 0.0}


def turn_progress():
    yaw = emb.get_pose()[1]
    d = (yaw - acc["prev"] + np.pi) % (2 * np.pi) - np.pi
    acc["sum"] += d
    acc["prev"] = yaw
    return np.degrees(acc["sum"]) if np.degrees(acc["sum"]) >= 90.0 else None


state["cmd"] = (0.0, 0.0, w)
t = time.monotonic()
deg, steps2, fell2 = run_until(check_every=max(HZ // 50, 1), timeout_s=15, done_fn=turn_progress)
hold_zero(cfg["zero_hold_frames"])
# 站稳后的最终转角（含 overshoot，进标定记录）
turn_progress()
rec["turn"] = {
    "target_deg": 90,
    "reached_deg": round(np.degrees(acc["sum"]), 1),
    "sim_s": round(steps2 / HZ, 2),
    "wall_s": round(time.monotonic() - t, 2),
    "fell": bool(fell2),
}

pos, yaw = emb.get_pose()
rec["final"] = {"pos": [round(float(x), 3) for x in pos], "yaw_deg": round(np.degrees(yaw), 1)}
rec["wall_total_s"] = round(time.monotonic() - T0, 2)
ok = bool(
    rec["straight"]["walked_m"] >= 2.0
    and not fell1
    and not fell2
    and 75.0 <= rec["turn"]["reached_deg"] <= 105.0
)
rec["ok"] = ok

with open(os.path.join(OUT, f"{args.embodiment}.json"), "w") as f:
    json.dump(rec, f, indent=2)
print("SMOKE_M2 " + json.dumps(rec, ensure_ascii=False))

app.close()
sys.exit(0 if ok else 1)
