"""2D 拓扑仿真 <-> 三 Agent planner 的双向桥。

一个方向：把 Episode 的当前观测(zone 图、契约、forum、机器人状态)压成
risk-navigation-agents 三 Agent 流程使用的 scenario 结构
(task / robot / environment / available_actions)。

另一个方向：把 Safety Guard 批准的动作名映射回仿真动作
(goto / hold / return),safe_stop 映射为 None(终止 episode)。
"""
from __future__ import annotations

from core.episode import Episode
from core.state_interface import contract_views

# 仿真动作之外的安全动作:observe_again / ask_human 在拓扑仿真里等价于
# 原地等待一个 tick(时间照常流逝),safe_stop 直接终止 episode。
META_ACTIONS = ("hold", "return_to_start", "observe_again", "ask_human", "safe_stop")


def task_goal_text(task: dict, zone_names: dict[str, str]) -> str:
    nm = lambda z: zone_names.get(z, z)  # noqa: E731
    t = task["type"]
    if t == "deliver":
        return f"把物品 {task['object']} 从 {nm(task['pickup_zone'])} 送到 {nm(task['dropoff_zone'])}"
    if t == "visit":
        return "巡视以下区域：" + "、".join(nm(z) for z in task["targets"])
    if t == "escort":
        return f"把 {task['human']} 从 {nm(task['from_zone'])} 护送到 {nm(task['to_zone'])}"
    raise ValueError(f"unknown task type {t}")


def available_actions(episode: Episode) -> list[str]:
    """goto 只开放相邻 zone,保证 planner 每次决策只推进一跳,便于逐步风险审查。"""
    obs = episode.observation()
    return [f"goto_{z}" for z in obs["adjacent"]] + list(META_ACTIONS)


def build_planning_scenario(episode: Episode) -> dict:
    """Episode 当前状态 -> 三 Agent planner 的 scenario dict。"""
    obs = episode.observation()
    static = episode.static_observation()
    w = episode.world
    zone_names = {z.id: z.name for z in w.zones.values()}

    zones = [
        {
            "id": z.id,
            "name": z.name,
            "kind": z.kind,
            "attrs": {k: v for k, v in z.attrs.items()},
        }
        for z in w.zones.values()
    ]
    edges = [[a, b] for a, b, embs in w.edges if w.embodiment in embs]

    return {
        "task": {
            "goal": task_goal_text(episode.task, zone_names),
            "type": episode.task["type"],
            "detail": dict(episode.task),
            "progress": obs["progress"],
        },
        "robot": {
            "type": f"indoor_{w.embodiment}_robot",
            "embodiment": w.embodiment,
            "current_zone": obs["current_zone"],
            "current_zone_name": zone_names.get(obs["current_zone"], obs["current_zone"]),
            "camera_on": obs["camera_on"],
            "carrying": obs["carrying"],
            "escorted_human": obs["escorted_human"],
        },
        "environment": {
            "kind": "2d_topological_map",
            "time": obs["time"],
            "step": obs["step"],
            "steps_remaining": episode.horizon - obs["step"],
            "zones": zones,
            "traversable_edges": edges,
            "adjacent_zones": [
                {"id": z, "name": zone_names.get(z, z)} for z in obs["adjacent"]
            ],
            "authorization_contract": contract_views(
                episode.scenario["contract_gt"], static["zones"]
            ),
            "forum_posts": obs["forum"],
            "violations_so_far": [v.to_dict() for v in episode.violations],
        },
        "available_actions": available_actions(episode),
    }


def to_sim_action(action_name: str) -> dict | None:
    """Safety Guard 批准的动作名 -> 仿真动作;safe_stop/空动作返回 None(终止)。"""
    if action_name.startswith("goto_"):
        return {"type": "goto", "zone": action_name[len("goto_"):]}
    if action_name == "return_to_start":
        return {"type": "return"}
    if action_name in ("hold", "observe_again", "ask_human"):
        return {"type": "hold"}
    if action_name in ("safe_stop", ""):
        return None
    raise ValueError(f"未知动作名 {action_name!r}")
