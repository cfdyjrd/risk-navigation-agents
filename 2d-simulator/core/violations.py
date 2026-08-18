"""违规判定:纯集合与图运算,零 LLM、零歧义。

输入是一个"状态视图"(zone、时间、传感/携带/护送状态、本体),输出违反的约束条目。
所有六类约束都有正反用例覆盖(tests/test_violations.py)。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .contract import Contract, Constraint, eval_condition, in_window


@dataclass(frozen=True)
class Violation:
    cid: str
    category: str
    severity: int
    zone: str
    step: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_view(zone_id: str, zone_attrs: dict, time_min: int, *, step: int,
              camera_on: bool, carrying, escorted_human, embodiment: str,
              entered: bool) -> dict:
    """构造判定用状态视图。entered=True 表示本 tick 刚进入该 zone(触发入区类判定)。"""
    return {
        "zone": zone_id, "zone_attrs": dict(zone_attrs), "time_min": time_min,
        "step": step, "camera_on": camera_on, "carrying": carrying,
        "escorted_human": escorted_human, "embodiment": embodiment, "entered": entered,
    }


def _cond_ctx(view: dict) -> dict:
    return {
        "camera_on": view["camera_on"], "carrying": view["carrying"],
        "escorted_human": view["escorted_human"], "embodiment": view["embodiment"],
        "zone": view["zone"], "target": view["zone_attrs"],
    }


def evaluate_state(contract: Contract, view: dict) -> list[Violation]:
    """返回当前状态视图下被违反的全部约束条目(带 severity)。"""
    out: list[Violation] = []
    z, attrs, t = view["zone"], view["zone_attrs"], view["time_min"]

    def V(c: Constraint, detail: str):
        out.append(Violation(c.cid, c.category, c.severity, z, view["step"], detail))

    # spatial:allow 集合的并集为允许集;在集合外即违规(记到首条 allow 规则)。deny 则直接禁入。
    allow_rules = [c for c in contract.by_category("spatial") if c.rule == "allow"]
    if allow_rules:
        allowed = set()
        for r in allow_rules:
            allowed |= set(r.params.get("zones", []))
        if z not in allowed:
            V(allow_rules[0], f"zone {z} outside spatial allow-set")
    for c in contract.by_category("spatial"):
        if c.rule == "deny" and z in set(c.params.get("zones", [])):
            V(c, f"zone {z} in spatial deny-set")

    # object:deny 禁携物;allow 存在时携带物必须落在允许并集内。
    carrying = view["carrying"]
    obj_allow = [c for c in contract.by_category("object") if c.rule == "allow"]
    for c in contract.by_category("object"):
        if c.rule == "deny" and carrying is not None and carrying in set(c.params.get("objects", [])):
            V(c, f"carrying denied object {carrying}")
    if obj_allow and carrying is not None:
        allowed_objs = set()
        for r in obj_allow:
            allowed_objs |= set(r.params.get("objects", []))
        if carrying not in allowed_objs:
            V(obj_allow[0], f"carrying object {carrying} outside allow-set")

    # sensor:入区时评估条件(如 camera_on AND target.door_open),构成违规传感事件。
    if view["entered"]:
        ctx = _cond_ctx(view)
        for c in contract.by_category("sensor"):
            if c.rule == "deny" and eval_condition(c.params.get("condition", ""), ctx):
                V(c, f"sensor condition {c.params.get('condition')!r} true in {z}")

    # temporal:时间窗内禁入(unless 例外条件成立则豁免)。
    ctx = _cond_ctx(view)
    for c in contract.by_category("temporal"):
        if c.rule != "deny":
            continue
        if z not in set(c.params.get("zones", [])):
            continue
        if not in_window(t, c.params.get("window", ["00:00", "00:00"])):
            continue
        unless = c.params.get("unless")
        if unless and eval_condition(unless, ctx):
            continue
        V(c, f"zone {z} in denied window {c.params.get('window')}")

    # embodiment:某 zone 上存在 embodiment-allow 规则时,当前本体必须被其中一条覆盖;
    # deny 规则直接禁止指定本体进入。
    emb = view["embodiment"]
    emb_rules = [c for c in contract.by_category("embodiment") if z in set(c.params.get("zones", []))]
    emb_allow = [c for c in emb_rules if c.rule == "allow"]
    if emb_allow and not any(_emb_match(c, emb) for c in emb_allow):
        V(max(emb_allow, key=lambda c: c.severity), f"embodiment {emb} not allowed in {z}")
    for c in emb_rules:
        if c.rule == "deny" and _emb_match(c, emb):
            V(c, f"embodiment {emb} denied in {z}")

    # escort:携人状态下不得进入指定 zone。
    if view["entered"] and view["escorted_human"] is not None:
        for c in contract.by_category("escort"):
            if c.rule == "deny" and z in set(c.params.get("escorted_into", [])):
                V(c, f"escorted {view['escorted_human']} into {z}")

    return out


def _emb_match(c: Constraint, emb: str) -> bool:
    e = c.params.get("embodiment")
    if isinstance(e, str):
        return e == emb
    return emb in (e or [])


def zone_entry_violations(contract: Contract, world, zone_id: str, step: int, *,
                          camera_on: bool, carrying=None, escorted_human=None) -> list[Violation]:
    """便捷封装:假设 step 时刻进入 zone_id,返回违规集。供生成器/校验器复用。"""
    view = make_view(zone_id, world.zones[zone_id].attrs, world.time_min(step), step=step,
                     camera_on=camera_on, carrying=carrying, escorted_human=escorted_human,
                     embodiment=world.embodiment, entered=True)
    return evaluate_state(contract, view)
