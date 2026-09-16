#!/usr/bin/env python3
"""M3 标定（设计文档 §3.5，实测记录制、不设硬门槛）：
每形态 × 12 宏 × N 次，产出实测位移/转角/耗时（均值±标准差）。

    ~/IsaacSim/.../python.sh scripts/calibrate.py --embodiment dog --n 5

产出 out/calibration/<embodiment>.json；宏语义按标定值重标（进 prompt 与论文附录）。
空旷地面执行，每次宏前把机器人 teleport 回原点并 settle——宏之间互不污染。
"""

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--embodiment", required=True, choices=["car", "dog", "humanoid"])
ap.add_argument("--n", type=int, default=5)
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
from isaacsim.core.api.objects import GroundPlane

from isaac.embodiments import load_embodiment_cfg, make_embodiment
from isaac.middleware import MacroExecutor
from isaac.sensors import RaySensor

cfg = load_embodiment_cfg(os.path.join(ROOT, "configs", "embodiment", f"{args.embodiment}.yaml"), ASSETS)
with open(os.path.join(ROOT, "configs", "run.yaml")) as f:
    run_cfg = yaml.safe_load(f)
OUT = os.path.join(ROOT, "out", "calibration")
os.makedirs(OUT, exist_ok=True)

MACROS = (
    [f"forward_{a}" for a in ("0.3", "0.5", "1", "2")]
    + ["backward_0.5"]
    + [f"turn_{s}_{d}" for s in ("left", "right") for d in (15, 30, 90)]
    + ["stop"]
)

world = World(stage_units_in_meters=1.0, physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
GroundPlane(prim_path="/World/ground", size=60)
emb = make_embodiment(cfg)
emb.spawn(world, position=np.array([0.0, 0.0]))
world.reset()

world.add_physics_callback("robot", emb.on_physics_step)

executor = MacroExecutor(world, emb, RaySensor(emb), run_cfg)


def reset_to_origin():
    """world.reset 回初始状态 + 重新 initialize + settle ≥1s。
    （teleport 对腿式不可用：基座挪走但关节仍是行走姿态，H1 实测连 stop 都摔。）"""
    world.reset()
    emb.request_reinit()
    emb.set_cmd(0.0, 0.0, 0.0)
    for _ in range(max(cfg["settle_steps"], round(1.0 / cfg["physics_dt"]))):
        world.step(render=False)


table = {}
t0 = time.monotonic()
for macro in MACROS:
    runs = []
    for i in range(args.n):
        reset_to_origin()
        if emb.fallen():  # teleport 后没站住，重来一次
            reset_to_origin()
        r = executor.execute(macro)
        runs.append(r)
    ms = [r["executed_m"] for r in runs]
    ds = [r["executed_deg"] for r in runs]
    ts = [r["duration_s"] for r in runs]
    aborted = sum(1 for r in runs if r["status"] == "aborted")
    table[macro] = {
        "n": args.n,
        "m_mean": round(float(np.mean(ms)), 3), "m_std": round(float(np.std(ms)), 3),
        "deg_mean": round(float(np.mean(ds)), 1), "deg_std": round(float(np.std(ds)), 1),
        "wall_s_mean": round(float(np.mean(ts)), 2),
        "aborted": aborted,
        "abort_reasons": sorted({r["reason"] for r in runs if r["reason"]}),
    }
    print(f"CAL {args.embodiment} {macro}: m={table[macro]['m_mean']}±{table[macro]['m_std']} "
          f"deg={table[macro]['deg_mean']}±{table[macro]['deg_std']} abort={aborted}/{args.n}", flush=True)

out = {"embodiment": args.embodiment, "n_per_macro": args.n, "macros": table,
       "wall_total_s": round(time.monotonic() - t0, 1)}
with open(os.path.join(OUT, f"{args.embodiment}.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"CAL_DONE {args.embodiment} -> out/calibration/{args.embodiment}.json", flush=True)

app.close()
