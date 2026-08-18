"""合规规划搜索:在 (zone, 任务阶段, step) 状态空间上 BFS,找一条零违规完成任务的动作序列。

三处复用:
  1. 生成器一致性校验(契约在世界图上可满足)
  2. oracle_agent 的执行计划
  3. rewrite 替代动作的构造与验证
允许 hold(等待时间窗),状态数 <= |zones| * |phases| * horizon,毫秒级。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .contract import Contract
from .violations import evaluate_state, make_view
from .world import World


def _task_phases(task: dict):
    """返回 (初始阶段, 终态判断, 进入 zone 时的阶段转移, 该阶段的 carrying/escort 状态)。"""
    t = task["type"]
    if t == "deliver":
        init, final = 0, 2

        def trans(phase, zone):
            if phase == 0 and zone == task["pickup_zone"]:
                return 1
            if phase == 1 and zone == task["dropoff_zone"]:
                return 2
            return phase

        def load(phase):
            return (task["object"] if phase == 1 else None, None)
        return init, final, trans, load
    if t == "visit":
        targets = tuple(sorted(task["targets"]))
        init, final = frozenset(), frozenset(targets)

        def trans(phase, zone):
            return phase | {zone} if zone in targets else phase

        def load(phase):
            return (None, None)
        return init, final, trans, load
    if t == "escort":
        init, final = 0, 2

        def trans(phase, zone):
            if phase == 0 and zone == task["from_zone"]:
                return 1
            if phase == 1 and zone == task["to_zone"]:
                return 2
            return phase

        def load(phase):
            return (None, task["human"] if phase == 1 else None)
        return init, final, trans, load
    raise ValueError(f"unknown task type {t}")


def find_compliant_plan(world: World, contract: Contract, task: dict, horizon: int,
                        start_zone: Optional[str] = None) -> Optional[list[dict]]:
    """返回动作序列(相邻 goto / hold),使任务在 horizon 步内完成且全程零违规;不存在则 None。"""
    init_phase, final, trans, load = _task_phases(task)
    start = start_zone or world.start_zone
    camera_on = bool(world.robot_init.get("camera_on", False))

    _memo: dict = {}

    def ok(zone: str, phase, step: int, entered: bool) -> bool:
        # 判定只依赖 (zone, 负载, step, entered);visit 任务负载恒空,记忆化收益巨大
        carrying, escorted = load(phase)
        key = (zone, carrying, escorted, step, entered)
        hit = _memo.get(key)
        if hit is not None:
            return hit
        view = make_view(zone, world.zones[zone].attrs, world.time_min(step), step=step,
                         camera_on=camera_on, carrying=carrying, escorted_human=escorted,
                         embodiment=world.embodiment, entered=entered)
        res = not evaluate_state(contract, view)
        _memo[key] = res
        return res

    start_phase = trans(init_phase, start)
    if not ok(start, start_phase, 0, True):
        return None
    if start_phase == final:
        return []
    q = deque([(start, start_phase, 0)])
    prev = {(start, start_phase, 0): None}
    while q:
        zone, phase, step = q.popleft()
        if step >= horizon:
            continue
        # hold(等时间窗)
        nxt_states = [(zone, phase, step + 1, False)]
        for nz in world.neighbors(zone):
            nxt_states.append((nz, trans(phase, nz), step + 1, True))
        for nz, nphase, nstep, entered in nxt_states:
            key = (nz, nphase, nstep)
            if key in prev:
                continue
            if not ok(nz, nphase, nstep, entered):
                continue
            prev[key] = (zone, phase, step)
            if nphase == final:
                # 回溯动作序列
                actions: list[dict] = []
                cur = key
                while prev[cur] is not None:
                    pz = prev[cur]
                    actions.append({"type": "hold"} if cur[0] == pz[0]
                                   else {"type": "goto", "zone": cur[0]})
                    cur = pz
                return actions[::-1]
            q.append((nz, nphase, nstep))
    return None


def plan_zone_sequence(actions: list[dict], start: str) -> list[str]:
    """动作序列对应的逐 tick zone 序列(便于校验/调试)。"""
    seq, cur = [], start
    for a in actions:
        if a["type"] == "goto":
            cur = a["zone"]
        seq.append(cur)
    return seq
