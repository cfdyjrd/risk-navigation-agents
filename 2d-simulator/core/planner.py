"""合规规划搜索:在 (zone, 任务阶段, step) 状态空间上 BFS,找一条零违规完成任务的动作序列。

三处复用:
  1. 生成器一致性校验(契约在世界图上可满足)
  2. oracle_agent 的执行计划
  3. rewrite 替代动作的构造与验证
允许 hold(等待时间窗),状态数 <= |zones| * |phases| * horizon,毫秒级。

阶梯版(docs/ladder_spec.md):
  - find_compliant_plan 支持 task["extra_targets"](必访补充目标);
  - find_timed_plan / truth_timeline:契约与任务随修订 step 切换的时变真值规划(oracle 上界),
    与引擎 Episode._tick 的修订时序一致。
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from .contract import Contract
from .violations import evaluate_state, make_view
from .world import World


def _task_phases(task: dict):
    """返回 (初始阶段, 终态判断, 进入 zone 时的阶段转移, 该阶段的 carrying/escort 状态)。

    任务带 extra_targets(阶梯版 add_target 修订追加的必访目标,见 docs/ladder_spec.md E2)时,
    阶段扩展为 (主任务阶段, 已访问的补充目标集合),终态要求补充目标全部访问;
    无 extra_targets 时与原实现完全一致。
    """
    init, final, trans, load = _main_task_phases(task)
    extra = frozenset(task.get("extra_targets") or [])
    if not extra:
        return init, final, trans, load

    def trans_ext(phase, zone):
        main, seen = phase
        return (trans(main, zone), seen | {zone} if zone in extra else seen)

    def load_ext(phase):
        return load(phase[0])
    return (init, frozenset()), (final, extra), trans_ext, load_ext


def _main_task_phases(task: dict):
    """主任务(deliver / visit / escort)的阶段机。"""
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


# ---------------------------------------------------------------- 时变真值(修订按 step 生效)
def truth_timeline(issued_contract: dict, issued_task: dict,
                   amendments: list) -> list[tuple[int, Contract, dict]]:
    """[(生效 step, Contract, task)]:签发版 + 各 legit 修订依次生效(与引擎 _tick 的时序一致)。"""
    from .episode import apply_amendment   # 延迟导入:episode 不依赖 planner,无循环
    import copy
    cdict, task = copy.deepcopy(issued_contract), copy.deepcopy(issued_task)
    timeline = [(0, Contract.from_dict(cdict), task)]
    legit = sorted((a for a in (amendments or []) if a.get("legit")),
                   key=lambda a: int(a.get("step", 0)))
    for a in legit:
        cdict, task = apply_amendment(cdict, task, a)
        timeline.append((int(a.get("step", 0)), Contract.from_dict(cdict), task))
    return timeline


def find_timed_plan(world: World, timeline: list[tuple[int, Contract, dict]], horizon: int,
                    start_zone: Optional[str] = None) -> Optional[list[dict]]:
    """时变真值下的合规规划(oracle 上界):BFS 状态 (zone, 主任务阶段, 已访问目标集, step)。

    与 find_compliant_plan 同构,但契约/任务按 timeline 随 step 切换(修订生效时刻之前
    进入 allow_add 的 zone 仍算违规),并要求各版本任务的 extra_targets 全部访问。
    零违规、horizon 内完成;不存在则 None。timeline 只有签发版时等价于 find_compliant_plan。
    """
    ttype = timeline[0][2]["type"]
    camera_on = bool(world.robot_init.get("camera_on", False))
    tracked: set = set()   # 各版本中出现过的"需访问 zone"并集
    for _, _, tk in timeline:
        tracked |= set(tk.get("extra_targets", []))
        if ttype == "visit":
            tracked |= set(tk.get("targets", []))

    def truth_at(step: int):
        cur = timeline[0]
        for t in timeline:
            if t[0] <= step:
                cur = t
        return cur[1], cur[2]

    def trans(phase: int, zone: str, task: dict) -> int:
        if ttype == "deliver":
            if phase == 0 and zone == task["pickup_zone"]:
                return 1
            if phase == 1 and zone == task["dropoff_zone"]:
                return 2
        elif ttype == "escort":
            if phase == 0 and zone == task["from_zone"]:
                return 1
            if phase == 1 and zone == task["to_zone"]:
                return 2
        return phase

    def load(phase: int, task: dict):
        if ttype == "deliver":
            return (task["object"] if phase == 1 else None, None)
        if ttype == "escort":
            return (None, task["human"] if phase == 1 else None)
        return (None, None)

    def finished(phase: int, seen: frozenset, task: dict) -> bool:
        extra_ok = set(task.get("extra_targets", [])) <= seen
        if ttype == "visit":
            return set(task["targets"]) <= seen and extra_ok
        return phase == 2 and extra_ok

    def ok(zone: str, phase: int, step: int, entered: bool, contract: Contract, task: dict) -> bool:
        carrying, escorted = load(phase, task)
        view = make_view(zone, world.zones[zone].attrs, world.time_min(step), step=step,
                         camera_on=camera_on, carrying=carrying, escorted_human=escorted,
                         embodiment=world.embodiment, entered=entered)
        return not evaluate_state(contract, view)

    start = start_zone or world.start_zone
    c0, t0 = truth_at(0)
    phase0 = trans(0, start, t0)
    seen0 = frozenset({start} & tracked)
    if not ok(start, phase0, 0, True, c0, t0):
        return None
    if finished(phase0, seen0, t0):
        return []
    init = (start, phase0, seen0, 0)
    q = deque([init])
    prev: dict = {init: None}
    while q:
        zone, phase, seen, step = q.popleft()
        if step >= horizon:
            continue
        nstep = step + 1
        contract, task = truth_at(nstep)
        cands = [(zone, False)] + [(nz, True) for nz in world.neighbors(zone)]
        for nz, entered in cands:
            nphase = trans(phase, nz, task) if entered else phase
            nseen = seen | {nz} if (entered and nz in tracked) else seen
            key = (nz, nphase, nseen, nstep)
            if key in prev or not ok(nz, nphase, nstep, entered, contract, task):
                continue
            prev[key] = (zone, phase, seen, step)
            if finished(nphase, nseen, task):
                actions: list[dict] = []
                cur = key
                while prev[cur] is not None:
                    pz = prev[cur]
                    actions.append({"type": "hold"} if cur[0] == pz[0]
                                   else {"type": "goto", "zone": cur[0]})
                    cur = pz
                return actions[::-1]
            q.append(key)
    return None
