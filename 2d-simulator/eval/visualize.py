"""单 episode 可视化复盘(离线录像):楼层平面图 + 逐 tick 回放 + 多 agent 对比。

    python -m eval.visualize --scenario fam0408U_s0 --seed 0 \
        --agents naive_executor,full_system --results results/main --out report/replay_demo.html

实现:把 results/<tag>/trajectories 里落盘的动作序列重新喂给确定性 episode 引擎
逐 tick 重演(core.state_interface.extract_replay,重演结果与原始日志逐 tick 对账),
再交给 eval.html_visualizer 渲染为自包含 HTML(平面图画布 + 契约活性面板 +
Forum 流 + 辩论转录 + 决策历史)。浏览器直接打开,无外部依赖。

交互式驾驶舱(live 模式)见 eval/play.py。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.state_interface import extract_replay

from .html_visualizer import HTMLVisualizer
from .util import read_jsonl_gz


def load_trajectory(run_dir: Path, agent: str, sid: str, seed: int) -> dict | None:
    fp = run_dir / "trajectories" / f"{agent}.jsonl.gz"
    if not fp.exists():
        return None
    for t in read_jsonl_gz(fp):
        if t["scenario_id"] == sid and t["seed"] == seed:
            return t
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="scenario_id,如 fam0408U_s0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", default="naive_executor,full_system")
    ap.add_argument("--scenarios-dir", default="scenarios/")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sc = json.loads((Path(args.scenarios_dir) / f"{args.scenario}.json").read_text())
    run_dir = Path(args.results)
    agent_data = {}
    for a in args.agents.split(","):
        traj = load_trajectory(run_dir, a, args.scenario, args.seed)
        if traj is None:
            print(f"warn: no trajectory for {a} ({args.scenario} seed{args.seed}), skip")
            continue
        rep = extract_replay(sc, traj)
        if rep["summary"]["mismatches"]:
            print(f"warn: {a} 重演与日志有 {rep['summary']['mismatches']} 处不一致")
        agent_data[a] = rep
    if not agent_data:
        raise SystemExit("no trajectories found — 该 run 是否带 --no-trajectory?")
    out = Path(args.out or f"report/replay_{args.scenario}.html")
    HTMLVisualizer(sc).render_to_file(out, agent_data)
    print(f"replay -> {out}")


if __name__ == "__main__":
    main()
