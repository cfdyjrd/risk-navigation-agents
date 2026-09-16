#!/usr/bin/env python3
"""批跑：2D scenario(s) → compile2d → runner(oracle/llm) → host/report.py（2D 同一份指标）。

    python3 scripts/run_batch.py ../2d-simulator/scenarios/*.json --planner oracle --render
    python3 scripts/run_batch.py --glob "../2d-simulator/scenarios/ladder/fam00*_s0.json" --seeds 1

每场景一进程（设计 §8.2）；失败不阻断其它场景；最后汇总 out/batch/<tag>/report.txt。
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("scenarios", nargs="*")
ap.add_argument("--glob")
ap.add_argument("--planner", default="oracle", choices=["oracle", "llm"])
ap.add_argument("--embodiment", help="覆盖场景默认形态（默认按 2D world.embodiment 映射）")
ap.add_argument("--localizer", default="topdown")
ap.add_argument("--render", action="store_true")
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-start", type=int, default=0, help="起始 seed（seed 0 = 无扰动；已跑过 seed 0 时可 --seed-start 1）")
ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M"))
a = ap.parse_args()

files = list(a.scenarios) + (sorted(glob.glob(a.glob)) if a.glob else [])
if not files:
    sys.exit("no scenarios")
batch = ROOT / "out" / "batch" / a.tag
batch.mkdir(parents=True, exist_ok=True)
runs = []
for f in files:
    pre = json.load(open(f))
    if "rooms" in pre and "doors" in pre:          # 已编译的 3D 场景（如 scenes/batch20/*.json）
        scene_path, scene = f, pre
        print(f"{scene['id']}: precompiled ({scene.get('source_2d', {}).get('bucket')}) oracle={len(scene.get('oracle_actions', []))}")
    else:
        comp = subprocess.run([sys.executable, str(ROOT / "scene/compile2d.py"), f] +
                              (["--embodiment", a.embodiment] if a.embodiment else []),
                              capture_output=True, text=True)
        print(comp.stdout.strip() or comp.stderr.strip()[-300:])
        if comp.returncode != 0:
            continue
        scene_path = comp.stdout.strip().split("-> ")[-1]
        scene = json.load(open(scene_path))
    if a.embodiment:
        scene["embodiment"] = a.embodiment
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        out = batch / f"{scene['id']}_{scene['embodiment']}_s{seed}"
        cmd = [sys.executable, str(ROOT / "host/runner.py"), "--scene", scene_path,
               "--embodiment", scene["embodiment"], "--localizer", a.localizer,
               "--planner", a.planner, "--out", str(out), "--horizon", str(scene["horizon"]), "--seed", str(seed),
               "--render" if a.render else "--no-render"]
        t = time.monotonic()
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if l.startswith("[tick") or "summary" in l][-3:]
        print(f"  {out.name}: rc={r.returncode} {round(time.monotonic()-t)}s | " + " | ".join(tail))
        if r.returncode != 0:
            (out / "runner_stderr.txt").parent.mkdir(parents=True, exist_ok=True)
            (out / "runner_stderr.txt").write_text(r.stderr[-4000:])
        runs.append(str(out))

rep = subprocess.run([sys.executable, str(ROOT / "host/report.py")] + runs, capture_output=True, text=True)
(batch / "report.txt").write_text(rep.stdout + rep.stderr)
print(rep.stdout or rep.stderr)
print(f"batch -> {batch}")
