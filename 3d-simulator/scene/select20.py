#!/usr/bin/env python3
"""从 2d-simulator 场景库挑 20 条"较难"场景编译为 3D 批测集（设计文档 §7.2 桶配比）。

难度分：level×3 + horizon/4 + floors×1.5 + zones/12 + 桶权重（unsafe/drift-L3/ambiguous-L3 高）。
硬门槛：compile2d 成功、oracle 计划存在、validate3d 通过（门宽/栅格可达≡拓扑/oracle 路径邻接）、
oracle 路径不经过被丢弃的边。桶配比（20 条）：unsafe-clear 4、drift-L3 5（含 legit/spoof 孪生对）、
drift-L2 2、ambiguous-L3 4、ambiguous-L2 2、safe-clear 3。
    python3 scene/select20.py [--out scenes/batch20]
"""
import argparse, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scene"))
from compile2d import compile_scenario          # noqa: E402
from validate3d import validate, ROBOT_WIDTH   # noqa: E402

SIM2D = ROOT.parent / "2d-simulator" / "scenarios"
QUOTA = {"unsafe-clear": 4, "drift-L3": 5, "drift-L2": 2, "ambiguous-L3": 4, "ambiguous-L2": 2, "safe-clear": 3}
BUCKET_W = {"unsafe-clear": 4, "drift-L3": 5, "drift-L2": 3, "ambiguous-L3": 5, "ambiguous-L2": 3, "safe-clear": 0}


def difficulty(m, w):
    floors = len({z["attrs"].get("floor", 1) for z in w["zones"]})
    return (m.get("level") or 0) * 3 + m["horizon"] / 4 + floors * 1.5 + len(w["zones"]) / 12 + BUCKET_W.get(m.get("bucket"), 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "scenes" / "batch20")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cands = []
    files = sorted(glob.glob(str(SIM2D / "ladder" / "*.json"))) + sorted(glob.glob(str(SIM2D / "generated" / "*.json")))
    for f in files:
        s2d = json.load(open(f)); m = s2d.get("meta") or {}
        if not m.get("horizon") or m.get("bucket") not in QUOTA: continue
        if m["bucket"] == "safe-clear" and "ladder" not in f: continue   # safe 只从新格式 ladder 取
        try:
            sc = compile_scenario(s2d)
        except Exception as e:
            continue
        if sc.get("oracle_status") != "ok" or not sc["oracle_actions"]: continue
        issues = validate(sc, ROBOT_WIDTH.get(sc["embodiment"], 0.5))
        if issues: continue
        cands.append((difficulty(m, s2d["world"]), m["bucket"], m.get("pair_id"), f, sc))
    cands.sort(key=lambda c: -c[0])
    chosen, used_pairs, counts = [], set(), {b: 0 for b in QUOTA}
    # drift-L3 优先成对（legit/spoof 孪生：同 pair_id 的 T/S）
    by_pair = {}
    for c in cands:
        if c[1] == "drift-L3" and c[2]: by_pair.setdefault(c[2], []).append(c)
    for pid, lst in sorted(by_pair.items(), key=lambda kv: -max(x[0] for x in kv[1])):
        if len(lst) >= 2 and counts["drift-L3"] + 2 <= QUOTA["drift-L3"]:
            chosen += lst[:2]; counts["drift-L3"] += 2; used_pairs.add(pid)
    for c in cands:
        b = c[1]
        if counts[b] >= QUOTA[b] or c in chosen: continue
        chosen.append(c); counts[b] += 1
    # 桶配额凑不满时，从剩余候选按难度补足到 20（safe-clear 不补）
    for c in cands:
        if len(chosen) >= 20: break
        if c in chosen or c[1] == "safe-clear": continue
        chosen.append(c); counts[c[1]] += 1
    manifest = []
    for d, b, pid, f, sc in chosen:
        sc["batch_difficulty"] = round(d, 2)
        path = out / f"{sc['id']}.json"
        json.dump(sc, open(path, "w"), ensure_ascii=False, indent=1)
        manifest.append({"id": sc["id"], "bucket": b, "difficulty": round(d, 2), "embodiment": sc["embodiment"],
                         "horizon": sc["horizon"], "n_rooms": len(sc["rooms"]), "oracle_len": len(sc["oracle_actions"]),
                         "amendments": len(sc.get("amendments", [])), "source": f, "scene": str(path)})
    json.dump(manifest, open(out / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"candidates passing all gates: {len(cands)} / files {len(files)}; chosen {len(chosen)}: {counts}")
    for r in manifest:
        print(f"  {r['id']:14s} {r['bucket']:14s} diff={r['difficulty']:5.1f} emb={r['embodiment']:4s} H={r['horizon']:2d} rooms={r['n_rooms']:2d} oracle={r['oracle_len']:2d} amend={r['amendments']}")


if __name__ == "__main__":
    main()
