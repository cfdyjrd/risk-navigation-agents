"""在 2D 拓扑仿真中闭环运行 risk-navigation-agents 的 planner。

每个决策步:
  Episode 观测 --bridge--> scenario --> Task Advocate -> Risk Critic
  -> Safety Decision -> Safety Guard --bridge--> 仿真动作 --> Episode.execute

两种 planner:
  --planner rule  (默认,离线零 API):合规规划搜索(core/planner.py),
                  找不到零违规计划时 safe_stop——用于验证 harness 与场景。
  --planner llm   :完整三 Agent 流程逐步决策,每步结果按内容缓存在
                  .cache/ 下,重跑不重复计费。

用法:
  python3 run_sim_planner.py                                   # rule + 默认场景
  python3 run_sim_planner.py --scenario scenarios/hospital_deliver_unsafe.json
  python3 run_sim_planner.py --planner llm                     # 需 ZHINAO_API_KEY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_ROOT.parent))  # agents/ llm_client safety_guard experience_store

from core.episode import Episode
from core.planner import find_compliant_plan

from agents.advocate import analyze as run_advocate
from agents.critic import analyze as run_critic
from agents.decision import decide
from bridge import build_planning_scenario, to_sim_action
from experience_store import ExperienceStore, build_memory_card
from llm_client import LLMError, ZhinaoClient
from safety_guard import apply_safety_guard

CACHE_ROOT = SIM_ROOT / ".cache"
PIPELINE_VERSION = "sim-planner-v1"


# ------------------------------------------------------------------ 逐步缓存
def step_cache_dir(scenario: dict, experiences: list[dict]) -> Path:
    cache_input = {
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario,
        "retrieved_experiences": experiences,
    }
    encoded = json.dumps(
        cache_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return CACHE_ROOT / hashlib.sha256(encoded).hexdigest()[:16]


def cached_call(cache_dir: Path, role: str, fn):
    path = cache_dir / f"{role}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return saved["report"], saved.get("usage", {})
    report, usage = fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"report": report, "usage": usage}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report, usage


# ------------------------------------------------------------------ planner
def plan_step_llm(episode: Episode, client: ZhinaoClient, store: ExperienceStore) -> dict:
    """跑一轮三 Agent 流程,返回 {scenario, advocate, critic, decision, guard, usage}。"""
    scenario = build_planning_scenario(episode)
    experiences = [build_memory_card(item) for item in store.retrieve(scenario, top_k=3)]
    cache_dir = step_cache_dir(scenario, experiences)

    advocate_report, advocate_usage = cached_call(
        cache_dir, "advocate", lambda: run_advocate(scenario, client, experiences)
    )
    critic_report, critic_usage = cached_call(
        cache_dir, "critic", lambda: run_critic(scenario, client, experiences)
    )
    final_decision, decision_usage = cached_call(
        cache_dir,
        "decision",
        lambda: decide(scenario, advocate_report, critic_report, client, experiences),
    )
    guard = apply_safety_guard(scenario, final_decision)
    return {
        "scenario": scenario,
        "advocate": advocate_report,
        "critic": critic_report,
        "decision": final_decision,
        "guard": guard,
        "usage": {
            "advocate": advocate_usage,
            "critic": critic_usage,
            "decision": decision_usage,
        },
    }


def run_llm(episode: Episode) -> tuple[list[dict], list[dict]]:
    """三 Agent planner 闭环。返回 (决策日志, tick 日志)。"""
    client = ZhinaoClient()
    store = ExperienceStore(SIM_ROOT.parent / "experiences" / "risk_experiences.json")
    decision_log: list[dict] = []
    tick_log: list[dict] = []
    while not episode.done:
        step = plan_step_llm(episode, client, store)
        approved = step["guard"]["approved_action"].get("name", "")
        print(
            f"[step {episode.state.step}] zone={episode.state.zone} "
            f"decision={step['decision']['decision']} "
            f"model_action={step['guard']['model_action'].get('name')} "
            f"guard={step['guard']['status']} approved={approved or 'safe_stop'}"
        )
        decision_log.append(
            {
                "at_step": episode.state.step,
                "decision": step["decision"]["decision"],
                "reason": step["decision"].get("reason"),
                "model_action": step["guard"]["model_action"],
                "guard_status": step["guard"]["status"],
                "guard_violations": step["guard"]["hard_rule_violations"],
                "approved_action": approved,
                "usage": step["usage"],
            }
        )
        action = to_sim_action(approved)
        if action is None or step["decision"]["decision"] in ("reject", "safe_stop"):
            print("planner 选择终止（safe_stop / reject），episode 结束。")
            break
        res = episode.execute(action)
        tick_log.extend(res.entries)
        for e in res.entries:
            for v in e["violations"]:
                print(f"  !! 违规 {v['cid']} (severity {v['severity']}) @ {v['zone']}")
    return decision_log, tick_log


def run_rule(episode: Episode) -> tuple[list[dict], list[dict]]:
    """离线合规规划:一次性搜出零违规计划并执行;搜不到则 safe_stop。"""
    plan = find_compliant_plan(
        episode.world, episode.contract, episode.task, episode.horizon
    )
    if plan is None:
        print("合规规划搜索无解：任务无法在契约下零违规完成，safe_stop。")
        return [{"at_step": 0, "decision": "safe_stop",
                 "reason": "no compliant plan", "approved_action": "safe_stop"}], []
    decision_log: list[dict] = []
    tick_log: list[dict] = []
    for action in plan:
        if episode.done:
            break
        label = action["type"] if action["type"] != "goto" else f"goto_{action['zone']}"
        print(f"[step {episode.state.step}] zone={episode.state.zone} action={label}")
        decision_log.append(
            {"at_step": episode.state.step, "decision": "execute",
             "reason": "compliant plan search", "approved_action": label}
        )
        res = episode.execute(action)
        tick_log.extend(res.entries)
    return decision_log, tick_log


# ------------------------------------------------------------------ 主入口
def summarize(episode: Episode) -> dict:
    return {
        "success": episode.success,
        "steps": episode.state.step,
        "horizon": episode.horizon,
        "violations": [v.to_dict() for v in episode.violations],
        "violation_severity_sum": sum(v.severity for v in episode.violations),
        "brs_final": round(episode.brs_curve[-1], 4),
        "final_zone": episode.state.zone,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default=str(SIM_ROOT / "scenarios" / "hospital_deliver_safe.json"),
        help="仿真场景 JSON（含 world / contract_gt / task / forum）",
    )
    parser.add_argument("--planner", choices=("rule", "llm"), default="rule")
    parser.add_argument("--out", help="把运行日志与总结写入该 JSON 文件")
    args = parser.parse_args()

    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    episode = Episode(scenario)
    print(
        f"场景 {scenario['meta']['scenario_id']} ({scenario['meta']['bucket']}), "
        f"任务 {episode.task['type']}, horizon {episode.horizon}, "
        f"起点 {episode.state.zone}, planner={args.planner}"
    )

    try:
        if args.planner == "llm":
            decision_log, tick_log = run_llm(episode)
        else:
            decision_log, tick_log = run_rule(episode)
    except LLMError as exc:
        print(f"三 Agent 流程失败，机器人保持 safe_stop：{exc}")
        return 1

    summary = summarize(episode)
    print("\n=== Episode Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"scenario": scenario["meta"], "planner": args.planner,
                 "decisions": decision_log, "ticks": tick_log, "summary": summary},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"日志已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
