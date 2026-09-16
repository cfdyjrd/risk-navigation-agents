#!/usr/bin/env python3
"""2D ↔ 3D 配对对照（设计文档 §4.5）：同 scenario_id 配对，用 2D eval.stats.paired_compare（Wilcoxon + bootstrap CI）。
    python3 host/compare2d3d.py --runs "out/batch/batch20/*" --arm2d ladder_llm ladder_single [--label3d oracle]
2D 结果：2d-simulator/results/<arm>/pNNN_<scenario>.json（summary.success / violations / violation_severity_sum / steps）。
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; SIM2D = ROOT.parent / "2d-simulator"
sys.path.append(str(SIM2D))


def load_2d(arm):
    eps = {}
    for f in glob.glob(str(SIM2D / "results" / arm / "p*_fam*.json")):
        d = json.load(open(f)); s = d["summary"]; sid = os.path.basename(f).split("_", 1)[1][:-5]
        eps[sid] = {"scenario_id": sid, "success": bool(s["success"]), "violations": s.get("violations", []),
                    "vss": s.get("violation_severity_sum", 0), "steps": s.get("steps"),
                    "bucket": ((d.get("scenario") or {}).get("meta") or {}).get("bucket") if isinstance(d.get("scenario"), dict) else None}
    return eps


def load_3d(pattern):
    eps = {}
    for d in glob.glob(pattern):
        if not Path(d, "summary.json").exists(): continue
        s = json.load(open(Path(d) / "summary.json")); js = s["judge_summary"]; sc = json.load(open(s["scene_path"]))
        sid = sc["id"]
        eps.setdefault(sid, []).append({"scenario_id": sid, "success": bool(js["success"]), "violations": js["violations"],
                                        "vss": js.get("violation_severity_sum", 0), "steps": js["total_ticks"],
                                        "bucket": (sc.get("source_2d") or {}).get("bucket"), "seed": s.get("seed", 0)})
    return eps


def ssr(e): return 1.0 if e["success"] and not e["violations"] else 0.0
def tsr(e): return 1.0 if e["success"] else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="out/batch/batch20/*"); ap.add_argument("--arm2d", nargs="+", default=["ladder_llm", "ladder_single"])
    ap.add_argument("--label3d", default="3D-oracle"); ap.add_argument("-o", "--out")
    a = ap.parse_args()
    e3 = load_3d(a.runs)
    from eval.stats import paired_compare
    lines = [f"# 2D ↔ 3D 配对对照（{a.label3d} vs 2D 各臂，配对单位 = scenario_id，seed 内先取均值）\n"]
    for arm in a.arm2d:
        e2 = load_2d(arm); common = sorted(set(e2) & set(e3))
        if not common: lines.append(f"## {arm}: 无共同场景\n"); continue
        for sid in common:   # 桶以 3D 场景侧（source_2d.meta）为准
            e2[sid]["bucket"] = e3[sid][0]["bucket"] or e2[sid]["bucket"]
        A = [x for sid in common for x in e3[sid]]; B = [e2[sid] for sid in common]
        by = defaultdict(lambda: [[], []])
        for sid in common:
            b = e2[sid]["bucket"]; by[b][0].append(sum(ssr(x) for x in e3[sid]) / len(e3[sid])); by[b][1].append(ssr(e2[sid]))
        lines.append(f"## {a.label3d} vs 2D {arm}（共同场景 {len(common)}）\n")
        lines.append("| 桶 | n | 3D SSR | 2D SSR | 3D TSR | 2D TSR |"); lines.append("|---|---|---|---|---|---|")
        for b, (x3, x2) in sorted(by.items()):
            t3 = sum(sum(tsr(x) for x in e3[sid]) / len(e3[sid]) for sid in common if e2[sid]["bucket"] == b) / len(x3)
            t2 = sum(tsr(e2[sid]) for sid in common if e2[sid]["bucket"] == b) / len(x2)
            lines.append(f"| {b} | {len(x3)} | {sum(x3)/len(x3):.2f} | {sum(x2)/len(x2):.2f} | {t3:.2f} | {t2:.2f} |")
        for name, fn in (("SSR", ssr), ("TSR", tsr)):
            r = paired_compare(A, B, fn)
            ci = r.get("ci95", [float("nan"), float("nan")])
            lines.append(f"\n**{name}**（3D − 2D）：均值差 {r.get('mean_diff', float('nan')):+.3f}，bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]，Wilcoxon p = {r.get('wilcoxon_p', float('nan')):.3f}（n={r.get('n', len(common))}，{'显著' if r.get('significant') else '不显著'}）")
        lines.append("")
        lines.append("| 场景 | 桶 | 3D success/viol | 2D success/viol |"); lines.append("|---|---|---|---|")
        for sid in common:
            x = e3[sid]; lines.append(f"| {sid} | {e2[sid]['bucket']} | {sum(t['success'] for t in x)}/{len(x)} · {sum(len(t['violations']) for t in x)} | {int(e2[sid]['success'])} · {len(e2[sid]['violations'])} |")
        lines.append("")
    text = "\n".join(lines); print(text[:3000])
    out = a.out or str(ROOT / "out" / "compare2d3d.md"); Path(out).write_text(text, encoding="utf-8"); print("->", out)


if __name__ == "__main__":
    main()
