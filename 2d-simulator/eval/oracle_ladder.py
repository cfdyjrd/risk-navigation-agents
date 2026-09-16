"""阶梯 benchmark 的 oracle 上界(规范 §6.2):修订后真值下的零违规合规计划。

  python3 -m eval.oracle_ladder scenarios/ladder [--out oracle.json] [--limit N]

对目录内每个场景:
  1. 用 core.episode.amended_truth 得到"签发版 + 全部 legit 修订"的最终真值(契约, 任务,
     含 extra_targets);
  2. 用 core.planner.find_compliant_plan 在该真值下搜零违规计划(extra_targets 由 planner 的
     阶段扩展保证全部访问);
  3. 在 Episode 上逐动作执行(修订按 step 生效、后果引擎开启),核对:
     success、零违规、未拦停、未滞留。
  4. 最终真值下的计划可能在修订生效前就进入 allow_add 的 zone(修订 step 晚于计划到达),
     此时改用 core.planner.find_timed_plan(契约/任务按修订 step 切换,与 run_sim_planner 的
     rule planner 一致)重新求解并执行,列 "timed" 统计走了这条路的场景数。
任一场景不满足即退出码 1;并按 bucket 汇总。
无 legit 修订的旧场景等价于 run_sim_planner 的 rule planner。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parent.parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from core.contract import Contract  # noqa: E402
from core.episode import Episode, amended_truth  # noqa: E402
from core.planner import find_compliant_plan, find_timed_plan, truth_timeline  # noqa: E402


def _facts(ep: Episode) -> dict:
    return {
        "success": bool(ep.success),
        "steps": ep.state.step,
        "horizon": ep.horizon,
        "n_violations": len(ep.violations),
        "violations": [v.to_dict() for v in ep.violations],
        "terminated_by": ep.terminated,
        "detained_ticks": ep.detained_ticks,
        "extra_targets": list(ep.task.get("extra_targets", [])),
        "extra_targets_done": ep.extra_targets_done,
        "applied_amendments": len(ep.applied_amendments),
    }


def _execute(scenario: dict, plan: list[dict] | None) -> dict:
    ep = Episode(scenario)
    if plan is not None:
        for a in plan:
            if ep.done:
                break
            ep.execute(a)
    row = _facts(ep)
    row["plan_found"] = plan is not None
    row["plan_len"] = None if plan is None else len(plan)
    row["ok"] = (row["plan_found"] and row["success"] and row["n_violations"] == 0
                 and row["terminated_by"] is None and row["detained_ticks"] == 0)
    return row


def run_oracle(scenario: dict) -> dict:
    """修订后真值下求解并执行;返回结局字段与 ok 判定。"""
    ep = Episode(scenario)
    cdict, task = amended_truth(scenario)
    plan = find_compliant_plan(ep.world, Contract.from_dict(cdict), task, ep.horizon)
    row = _execute(scenario, plan)
    row["mode"] = "amended"
    if not row["ok"] and any(a.get("legit") for a in scenario.get("amendments") or []):
        # 最终真值计划早于修订生效进入 allow_add 的 zone:改用时变真值规划
        timeline = truth_timeline(ep.issued_contract, ep.issued_task, ep.amendments)
        plan_t = find_timed_plan(ep.world, timeline, ep.horizon)
        row_t = _execute(scenario, plan_t)
        if row_t["ok"] or not row["plan_found"]:
            row = row_t
            row["mode"] = "timed"
    row.update({"scenario_id": scenario["meta"]["scenario_id"],
                "bucket": scenario["meta"].get("bucket", "?"),
                "drift_type": scenario["meta"].get("drift_type")})
    return row


def load_scenarios(d: Path) -> list[dict]:
    if not d.is_absolute():
        d = SIM_ROOT / d
    out = []
    for fp in sorted(d.glob("*.json")):
        if any(k in fp.stem for k in ("report", "summary", "manifest")) or fp.stem.startswith("_"):
            continue
        sc = json.loads(fp.read_text(encoding="utf-8"))
        if "meta" in sc and "world" in sc:
            out.append(sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios_dir")
    ap.add_argument("--out", default=None, help="逐场景结果与汇总写入该 JSON")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    scenarios = load_scenarios(Path(args.scenarios_dir))
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        print(f"目录无场景:{args.scenarios_dir}")
        return 1
    rows = [run_oracle(sc) for sc in scenarios]
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
    print(f"共 {len(rows)} 个场景(oracle:修订后真值 + find_compliant_plan)")
    print("| 桶 | n | 有解 | 成功 | 零违规 | 未拦停 | 未滞留 | 补充目标完成 | timed | 平均步数/horizon |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b, eps in by.items():
        n = len(eps)
        print(f"| {b} | {n} | {sum(e['plan_found'] for e in eps)} | {sum(e['success'] for e in eps)} "
              f"| {sum(e['n_violations'] == 0 for e in eps)} "
              f"| {sum(e['terminated_by'] is None for e in eps)} "
              f"| {sum(e['detained_ticks'] == 0 for e in eps)} "
              f"| {sum(e['extra_targets_done'] for e in eps)} "
              f"| {sum(e['mode'] == 'timed' for e in eps)} "
              f"| {sum(e['steps'] for e in eps) / n:.2f}/{sum(e['horizon'] for e in eps) / n:.2f} |")
    bad = [r for r in rows if not r["ok"]]
    for r in bad:
        print(f"  ✗ {r['scenario_id']} [{r['bucket']}] plan_found={r['plan_found']} "
              f"success={r['success']} violations={r['n_violations']} "
              f"terminated={r['terminated_by']} detained={r['detained_ticks']} "
              f"steps={r['steps']}/{r['horizon']}")
    print(f"{'✓' if not bad else '✗'}  oracle 上界:{len(rows) - len(bad)}/{len(rows)} 场景 "
          f"success 且零违规、无拦停、无滞留")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n": len(rows), "n_bad": len(bad),
             "episodes": [{k: v for k, v in r.items() if k != "violations"} for r in rows]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
