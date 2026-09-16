"""episode 运行器:agent × 场景 × seed -> 完整轨迹 + 轻量 summary。

每个 episode 落盘完整轨迹(每步观测摘要、可见帖、各角色发言、裁决、执行动作、
违规判定结果),供失败分析与录像式复盘。
"""
from __future__ import annotations

import random
import time

from core.episode import Episode

from .gt import GTAdjudicator


def run_episode(scenario: dict, agent, llm_client, seed: int, *,
                stream_forum: bool = True, keep_trajectory: bool = True,
                consequences: bool | None = None) -> dict:
    sid = scenario["meta"]["scenario_id"]
    rng = random.Random(f"{sid}|{agent.name}|{seed}")
    ep = Episode(scenario, stream_forum=stream_forum, consequences=consequences)
    static = ep.static_observation()
    if agent.uses_llm:
        agent.begin_episode(static, rng, contract=scenario["contract_gt"])
    else:
        agent.begin_episode(static, rng, contract=scenario["contract_gt"], scenario=scenario)
    gt = GTAdjudicator(scenario)

    tok0 = llm_client.snapshot() if llm_client else None
    t0 = time.time()
    steps_log, decisions = [], []
    entered_zones: set = set()
    violating_ticks = 0
    total_ticks = 0
    guard = 0
    while not ep.done and guard < scenario["meta"]["horizon"] + 5:
        guard += 1
        obs = ep.observation()
        dec = agent.decide(obs)
        exp = gt.expected_verdict(dec.proposal_target, obs["step"] + 1)
        tclass = gt.target_class(dec.proposal_target)
        res = ep.execute(dec.action)
        for e in res.entries:
            total_ticks += 1
            if e["entered"]:
                entered_zones.add(e["zone"])
            if e["violations"]:
                violating_ticks += 1
        decisions.append({
            "step": obs["step"], "proposal_target": dec.proposal_target,
            "final_target": dec.final_target, "verdict": dec.verdict, "expected": exp,
            "target_class": tclass, "critic_flags": dec.critic_flags,
            "critic_cids": dec.critic_cids, "rounds": dec.rounds,
            "llm_calls": dec.llm_calls, "goal_post_step": dec.goal_post_step,
        })
        if keep_trajectory:
            steps_log.append({
                "obs": {"step": obs["step"], "time": obs["time"],
                        "zone": obs["current_zone"],
                        "visible_posts": [p["step"] for p in obs["forum"]],
                        "carrying": obs["carrying"], "escorted": obs["escorted_human"]},
                "decision": dec.to_dict(),
                "expected": exp, "target_class": tclass,
                "result": {"entries": res.entries, "blocked": res.blocked},
            })
    wall = time.time() - t0
    tok1 = llm_client.snapshot() if llm_client else None

    m = scenario["meta"]
    danger_targets = sorted(gt.hazard_zones | gt.drift_zones)
    danger_entered = sorted(z for z in danger_targets if z in entered_zones)
    summary = {
        "scenario_id": sid, "family_id": m["family_id"], "pair_id": m.get("pair_id"),
        "bucket": m["bucket"], "domain": m["domain"], "agent": agent.name, "seed": seed,
        "success": ep.success, "steps": ep.state.step, "horizon": m["horizon"],
        "long_horizon": bool(m.get("long_horizon")),
        "drift_type": m.get("drift_type"),
        "drift_target_labeled": m.get("drift_target_labeled"),
        "drift_inject_step": m.get("drift_inject_step"),
        "hazard_kind": m.get("hazard_kind"),
        "n_constraints": m.get("n_constraints"),
        "violations": [v.to_dict() for v in ep.violations],
        "vss": sum(v.severity for v in ep.violations),
        "violating_ticks": violating_ticks, "total_ticks": max(1, total_ticks),
        "brs": [round(x, 4) for x in ep.brs_curve],
        "decisions": decisions,
        "danger_targets": danger_targets, "danger_entered": danger_entered,
        "drift_targets": sorted(gt.drift_zones), "hazard_targets": sorted(gt.hazard_zones),
        "ambiguous_targets": sorted(gt.ambiguous),
        "drift_entered": sorted(z for z in gt.drift_zones if z in entered_zones),
        "hazard_entered": sorted(z for z in gt.hazard_zones if z in entered_zones),
        "ambiguous_entered": sorted(z for z in gt.ambiguous if z in entered_zones),
        "ec": {
            "tokens_in": (tok1["tokens_in"] - tok0["tokens_in"]) if tok0 else 0,
            "tokens_out": (tok1["tokens_out"] - tok0["tokens_out"]) if tok0 else 0,
            "llm_calls": (tok1["calls"] - tok0["calls"]) if tok0 else 0,
            "rounds_total": sum(d["rounds"] for d in decisions),
            "action_steps": ep.state.step,
            "wall_time": round(wall, 4),
        },
    }
    traj = {"scenario_id": sid, "agent": agent.name, "seed": seed,
            "steps": steps_log} if keep_trajectory else None
    return {"summary": summary, "trajectory": traj}
