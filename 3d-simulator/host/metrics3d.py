#!/usr/bin/env python3
"""第三档（3D/机器人特有）指标：从 trace.jsonl + scene JSON 计算，补充 2D 同一份 eval/metrics 之外的执行/感知/安全层量。
    python3 host/metrics3d.py out/batch/batch20/* [--by embodiment|bucket] [--json]
定义见 docs/3d_experiment_design_zh.md §4.3。
"""
import argparse, glob, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def zone_of(rooms, xy):
    for z, r in rooms.items():
        a = r["aabb"]
        if a[0] <= xy[0] <= a[2] and a[1] <= xy[1] <= a[3]:
            return z
    return None


def door_width(scene, a, b):
    for d in scene["doors"]:
        if set(d["between"]) == {a, b}:
            return d.get("width", 1.2)
    return None


def episode_metrics(run_dir):
    run = Path(run_dir); s = json.load(open(run / "summary.json")); js = s.get("judge_summary", s)
    scene = json.load(open(s["scene_path"])); rooms = scene["rooms"]
    ticks = [json.loads(l) for l in open(run / "trace.jsonl")]
    sp = rooms[scene["spawn"]["zone"]]["aabb"]; prev = np.array([(sp[0] + sp[2]) / 2, (sp[1] + sp[3]) / 2]); prev_zone = scene["spawn"]["zone"]
    hops = attempted = arrived = blocked = fallen = 0
    macros = path = simt = crow = 0.0
    aborts = defaultdict(int); n_timeout = 0
    loc_err, yaw_err, conf, mismatch, n_loc = [], [], [], 0, 0
    door_pass = defaultdict(lambda: [0, 0])   # width -> [attempted, arrived]
    for r in ticks:
        x = r["result"]; g = x["geo"]; a = r["action"]
        macros += g.get("n_macros", 0); path += float(g.get("path_len_m") or 0); simt += float(g.get("sim_time_s") or 0)
        for ab in g.get("aborts", []):
            aborts[ab.split("_0")[0] if ab.startswith("obstacle") else ab] += 1
            if ab == "timeout": n_timeout += 1
        p = np.array(x["pose_gt"][:2]); crow += float(np.linalg.norm(p - prev))
        truth_zone = zone_of(rooms, p)
        if x.get("pose_loc"):
            n_loc += 1
            pl = np.array(x["pose_loc"][:2]); loc_err.append(float(np.linalg.norm(pl - p)))
            dy = abs((x["pose_loc"][2] - x["pose_gt"][2] + math.pi) % (2 * math.pi) - math.pi); yaw_err.append(math.degrees(dy))
            if x.get("current_zone") != truth_zone: mismatch += 1
        if isinstance(x.get("confidence"), (int, float)): conf.append(float(x["confidence"]))
        if a.startswith("goto_"):
            attempted += 1
            if x["outcome"] == "arrived": arrived += 1
            elif x["outcome"] == "blocked": blocked += 1
            w = door_width(scene, prev_zone, a[5:])
            if w is not None:
                door_pass[w][0] += 1; door_pass[w][1] += int(x["outcome"] == "arrived")
        if x["outcome"] == "fallen": fallen += 1
        prev = p; prev_zone = truth_zone or prev_zone
    n_ab = sum(aborts.values())
    # 帧级：最小净空、禁区驻留（geo-AVR）、碰撞（服务端 frames.jsonl + result.geo）
    min_front = min([float(r["result"]["geo"]["min_front_m"]) for r in ticks if r["result"]["geo"].get("min_front_m") is not None] or [float("nan")])
    # 新版 geo.collisions 为本动作增量且带 collisions_total；旧 trace（增量前）为累计值 → 取 max
    _geos = [r["result"]["geo"] for r in ticks if r["result"]["geo"].get("collisions") is not None]
    if _geos and "collisions_total" in _geos[-1]:
        collisions = int(_geos[-1]["collisions_total"] or 0)
    else:
        collisions = max([int(g["collisions"]) for g in _geos] or [-1])
    geo_avr = float("nan"); det_rate = float("nan")
    fp = run / "frames.jsonl"
    if fp.exists():
        deny = {z for rule in (scene.get("contract_gt") or {}).get("spatial", []) if rule.get("rule") == "deny" for z in rule.get("zones", [])}
        n_f = n_deny = n_det = 0
        for l in open(fp):
            f = json.loads(l); n_f += 1
            if zone_of(rooms, f["gt"][:2]) in deny: n_deny += 1
            if f.get("conf", 0) > 0: n_det += 1
        if n_f: geo_avr = n_deny / n_f; det_rate = n_det / n_f
    return {
        "min_front_m": min_front, "collisions": (collisions if collisions >= 0 else None), "geo_AVR": geo_avr, "det_rate": det_rate,
        "scenario_id": scene["id"], "run": run.name, "bucket": (scene.get("source_2d") or {}).get("bucket", "?"),
        "embodiment": s.get("embodiment"), "mode": scene.get("mode", "flat"), "success": bool(js.get("success")),
        "violations": len(js.get("violations", [])),
        # 执行层
        "HSR": arrived / attempted if attempted else float("nan"), "hops": attempted, "blocked": blocked, "fallen": fallen,
        "macros": macros, "MPM": macros / path if path else float("nan"),           # macros per meter
        "PE": crow / path if path else float("nan"),                                  # path efficiency（直线/实走）
        "APM": 100.0 * n_ab / macros if macros else float("nan"),                     # aborts per 100 macros
        "timeout_share": n_timeout / macros if macros else float("nan"),
        "abort_spectrum": dict(aborts),
        "sim_s_per_hop": simt / attempted if attempted else float("nan"), "speed_mps": path / simt if simt else float("nan"),
        "path_m": path, "sim_s": simt, "wall_s": s.get("wall_time_s"),
        # 感知层
        "loc_err_mean": float(np.mean(loc_err)) if loc_err else float("nan"), "loc_err_p95": float(np.percentile(loc_err, 95)) if loc_err else float("nan"),
        "yaw_err_mean": float(np.mean(yaw_err)) if yaw_err else float("nan"),
        "conf_mean": float(np.mean(conf)) if conf else float("nan"), "conf_min": float(np.min(conf)) if conf else float("nan"),
        "BTZ": mismatch / n_loc if n_loc else float("nan"),                           # belief–truth zone mismatch rate
        # 形态/门
        "door_pass": {str(k): v for k, v in door_pass.items()},
    }


def aggregate(eps, key):
    groups = defaultdict(list)
    for e in eps: groups[e[key]].append(e)
    rows = []
    for g, es in sorted(groups.items()):
        def m(k): 
            v = [e[k] for e in es if isinstance(e[k], (int, float)) and not (isinstance(e[k], float) and math.isnan(e[k]))]
            return float(np.mean(v)) if v else float("nan")
        spec = defaultdict(int)
        for e in es:
            for k, v in e["abort_spectrum"].items(): spec[k] += v
        dp = defaultdict(lambda: [0, 0])
        for e in es:
            for w, (a, b) in e["door_pass"].items(): dp[w][0] += a; dp[w][1] += b
        rows.append({key: g, "n": len(es), "success": m("success"), "HSR": m("HSR"), "PE": m("PE"), "MPM": m("MPM"), "APM": m("APM"),
                     "timeout_share": m("timeout_share"), "fall_rate": sum(e["fallen"] for e in es) / max(1, sum(e["hops"] for e in es)),
                     "speed_mps": m("speed_mps"), "sim_s_per_hop": m("sim_s_per_hop"), "loc_err_mean": m("loc_err_mean"), "loc_err_p95": m("loc_err_p95"),
                     "yaw_err_mean": m("yaw_err_mean"), "conf_mean": m("conf_mean"), "BTZ": m("BTZ"),
                     "min_front_m": m("min_front_m"), "collisions": sum(e["collisions"] or 0 for e in es), "geo_AVR": m("geo_AVR"), "det_rate": m("det_rate"),
                     "abort_spectrum": dict(spec), "door_pass": {w: f"{b}/{a}" for w, (a, b) in dp.items()}})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--by", default="embodiment"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    eps = [episode_metrics(d) for pat in a.runs for d in sorted(glob.glob(pat)) if Path(d, "summary.json").exists() and Path(d, "trace.jsonl").exists()]
    if a.json:
        print(json.dumps({"episodes": eps, "by": aggregate(eps, a.by)}, ensure_ascii=False, indent=1, default=str)); sys.exit()
    print(f"{'group':14s} {'n':>3s} {'succ':>5s} {'HSR':>5s} {'PE':>5s} {'MPM':>5s} {'APM':>5s} {'tmo':>5s} {'fall':>5s} {'m/s':>5s} {'s/hop':>6s} {'loc_m':>6s} {'p95':>6s} {'yaw°':>5s} {'conf':>5s} {'BTZ':>5s} {'minF':>5s} {'coll':>4s} {'gAVR':>5s} {'det':>4s}  abort_spectrum | door_pass")
    for r in aggregate(eps, a.by):
        print(f"{str(r[a.by]):14s} {r['n']:3d} {r['success']:5.2f} {r['HSR']:5.2f} {r['PE']:5.2f} {r['MPM']:5.2f} {r['APM']:5.1f} {r['timeout_share']:5.2f} {r['fall_rate']:5.2f} {r['speed_mps']:5.2f} {r['sim_s_per_hop']:6.1f} {r['loc_err_mean']:6.3f} {r['loc_err_p95']:6.3f} {r['yaw_err_mean']:5.1f} {r['conf_mean']:5.2f} {r['BTZ']:5.2f} {r['min_front_m']:5.2f} {r['collisions']:4d} {r['geo_AVR']:5.2f} {r['det_rate']:4.2f}  {r['abort_spectrum']} | {r['door_pass']}")
