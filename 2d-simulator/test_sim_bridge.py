"""离线测试：bridge 双向映射 + rule planner 闭环 + Safety Guard 对接。不调用 API。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(SIM_ROOT.parent))

from core.episode import Episode
from core.planner import find_compliant_plan

from bridge import build_planning_scenario, to_sim_action
from safety_guard import apply_safety_guard


def load(name: str) -> dict:
    return json.loads((SIM_ROOT / "scenarios" / name).read_text(encoding="utf-8"))


def test_scenario_structure():
    ep = Episode(load("hospital_deliver_safe.json"))
    s = build_planning_scenario(ep)
    for key in ("task", "robot", "environment", "available_actions"):
        assert key in s, f"scenario 缺少 {key}"
    assert s["task"]["goal"]
    assert s["robot"]["current_zone"] == ep.state.zone
    adj = {z["id"] for z in s["environment"]["adjacent_zones"]}
    assert {f"goto_{z}" for z in adj} <= set(s["available_actions"])
    for meta in ("hold", "observe_again", "ask_human", "safe_stop", "return_to_start"):
        assert meta in s["available_actions"]
    assert s["environment"]["authorization_contract"], "契约视图不能为空"
    json.dumps(s, ensure_ascii=False)  # 必须可序列化(缓存/传给 LLM)
    print("test_scenario_structure ok")


def test_action_mapping():
    assert to_sim_action("goto_c1a") == {"type": "goto", "zone": "c1a"}
    assert to_sim_action("return_to_start") == {"type": "return"}
    for name in ("hold", "observe_again", "ask_human"):
        assert to_sim_action(name) == {"type": "hold"}
    assert to_sim_action("safe_stop") is None
    try:
        to_sim_action("fly_away")
    except ValueError:
        pass
    else:
        raise AssertionError("未知动作应当抛 ValueError")
    print("test_action_mapping ok")


def test_safety_guard_integration():
    """Guard 应拦截不在 available_actions 中的动作,并放行合法动作。"""
    ep = Episode(load("hospital_deliver_safe.json"))
    s = build_planning_scenario(ep)
    legal = next(a for a in s["available_actions"] if a.startswith("goto_"))
    ok = apply_safety_guard(s, {"decision": "execute",
                                "selected_action": {"name": legal, "parameters": {}}})
    assert ok["status"] == "approved" and ok["approved_action"]["name"] == legal
    bad = apply_safety_guard(s, {"decision": "execute",
                                 "selected_action": {"name": "goto_nowhere", "parameters": {}}})
    assert bad["status"] == "overridden"
    assert bad["approved_action"]["name"] == "safe_stop"
    print("test_safety_guard_integration ok")


def test_rule_planner_closed_loop():
    ep = Episode(load("hospital_deliver_safe.json"))
    plan = find_compliant_plan(ep.world, ep.contract, ep.task, ep.horizon)
    assert plan is not None, "safe 场景应存在合规计划"
    for action in plan:
        if ep.done:
            break
        ep.execute(action)
    assert ep.success, "合规计划执行后任务应完成"
    assert not ep.violations, f"合规计划不应产生违规: {ep.violations}"
    print("test_rule_planner_closed_loop ok")


def test_scenario_updates_with_state():
    """执行动作后,bridge 生成的 scenario 应反映新状态。"""
    ep = Episode(load("hospital_deliver_safe.json"))
    s0 = build_planning_scenario(ep)
    target = s0["environment"]["adjacent_zones"][0]["id"]
    ep.execute(to_sim_action(f"goto_{target}"))
    s1 = build_planning_scenario(ep)
    assert s1["robot"]["current_zone"] == target
    assert s1["environment"]["step"] == 1
    print("test_scenario_updates_with_state ok")


if __name__ == "__main__":
    test_scenario_structure()
    test_action_mapping()
    test_safety_guard_integration()
    test_rule_planner_closed_loop()
    test_scenario_updates_with_state()
    print("全部通过")
