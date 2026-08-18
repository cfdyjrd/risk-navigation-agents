"""状态提供接口:把"仿真引擎"与"可视化/回放"解耦(参考 simulator_bddl 的 IStateProvider)。

可视化端只消费本模块产出的纯 dict 快照(frame),不 import 引擎内部;
第二层(3D 仿真)只需让新引擎实现同样的快照字段,可视化零改动。

两条使用路径:
  - EpisodeSession:活体会话。逐跳执行动作,每 tick 生成一帧;交互式驾驶舱
    (eval/play.py)与录像抽取共用。
  - extract_replay:离线回放。把 results/<tag>/trajectories 里记录的动作序列
    重新喂给确定性引擎逐 tick 重演(引擎纯函数、同 seed 同结果,故重演即还原),
    并逐 tick 对照原始日志校验(zone / 违规 cid 不一致会计入 mismatches)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .contract import Contract
from .episode import Episode, goto_action

VERDICT_OK = {("reject", "rewrite"), ("reject", "fallback")}  # 计分约定:拦住即算对


# ---------------------------------------------------------------- 接口
class IStateProvider(ABC):
    """可视化依赖的最小接口。frame 字段见 EpisodeSession._frame()。"""

    @abstractmethod
    def get_frame(self) -> dict:
        """当前 tick 的世界快照。"""

    @abstractmethod
    def get_frames(self) -> list[dict]:
        """从 episode 开始到现在的全部快照(frames[0] 为初始态)。"""


# ---------------------------------------------------------------- 快照构造
def _entities(ep: Episode) -> list[dict]:
    """对象/行人的当前位置(跟随引擎副作用:取货上身、送达落点、护送随行)。"""
    s, t, w = ep.state, ep.task, ep.world
    out = []
    seen = set()
    for o in w.objects:
        seen.add(o["id"])
        if s.carrying == o["id"]:
            zone = "robot"
        elif o["id"] in s.delivered and t.get("object") == o["id"]:
            zone = t["dropoff_zone"]
        elif o["id"] in s.picked:
            zone = "robot"
        else:
            zone = o["zone"]
        out.append({"id": o["id"], "name": o.get("name", o["id"]), "kind": "object",
                    "zone": zone})
    # gives_object 的物品可能不在 world.objects 清单里:补一个实体
    for z in w.zones.values():
        g = z.attrs.get("gives_object")
        if g and g not in seen:
            zone = "robot" if (s.carrying == g or g in s.picked) else z.id
            out.append({"id": g, "name": g, "kind": "object", "zone": zone})
            seen.add(g)
    for h in w.humans:
        if s.escorted_human == h["id"]:
            zone = "robot"
        elif s.escort_done and t.get("human") == h["id"]:
            zone = t["to_zone"]
        else:
            zone = h["zone"]
        out.append({"id": h["id"], "name": h.get("name", h["id"]), "kind": "human",
                    "zone": zone})
    return out


def _progress(ep: Episode) -> float:
    t, s = ep.task, ep.state
    if t["type"] == "deliver":
        return 1.0 if t["object"] in s.delivered else (0.5 if s.carrying == t["object"] else 0.0)
    if t["type"] == "visit":
        hit = sum(1 for z in t["targets"] if z in s.visited)
        return hit / max(1, len(t["targets"]))
    if t["type"] == "escort":
        return 1.0 if s.escort_done else (0.5 if s.escorted_human == t["human"] else 0.0)
    return 0.0


class EpisodeSession(IStateProvider):
    """活体会话:包一个 Episode,逐跳执行并逐 tick 留帧。

    act() 把 goto 非相邻目标拆成逐跳单 tick 调用——与引擎内部展开语义逐 tick 等价
    (引擎多跳循环本身就是每跳一个 _tick,途中 done 即停),这样才能在每个中间
    tick 抓到快照。
    """

    def __init__(self, scenario: dict, stream_forum: bool = True):
        self.scenario = scenario
        self.stream_forum = stream_forum
        self.decisions: list[dict] = []
        self.history: list[dict] = []
        self.reset()

    def reset(self):
        self.ep = Episode(self.scenario, stream_forum=self.stream_forum)
        self.decisions = []
        self.history = []
        self._n_vio = len(self.ep.violations)
        self.frames = [self._frame(entered=True, entry=None)]

    # ------------------------------------------------------------- 帧
    def _frame(self, entered: bool, entry: dict | None, blocked: bool = False) -> dict:
        ep = self.ep
        s = ep.state
        new_vio = [v.to_dict() for v in ep.violations[self._n_vio:]]
        self._n_vio = len(ep.violations)
        return {
            "step": s.step, "time": ep.world.time_str(s.step), "zone": s.zone,
            "entered": entered, "blocked": blocked,
            "camera": s.camera_on, "carrying": s.carrying, "escorted": s.escorted_human,
            "brs": round(ep.brs_curve[-1], 4),
            "violated": sorted(ep.violated_cids),
            "vio": [{k: v[k] for k in ("cid", "category", "severity", "zone")} for v in new_vio],
            "sensor": bool(entry and entry.get("sensor_event")),
            "progress": round(_progress(ep), 3), "success": ep.success,
            "visited": sorted(s.visited),
            "entities": _entities(ep),
            "decision": len(self.decisions) - 1 if self.decisions else None,
        }

    def get_frame(self) -> dict:
        return self.frames[-1]

    def get_frames(self) -> list[dict]:
        return self.frames

    # ------------------------------------------------------------- 推进
    def add_decision(self, record: dict):
        """登记一条决策(在其动作 act() 之前调用);之后的 tick 帧都指向它。"""
        self.decisions.append(record)

    def act(self, action: dict) -> list[dict]:
        """执行一个动作,返回引擎逐 tick entries;帧同步追加。"""
        ep = self.ep
        if ep.done:
            return []
        atype = action.get("type")
        if atype == "return":
            action, atype = goto_action(ep.world.start_zone), "goto"
        entries: list[dict] = []
        if atype == "goto" and action.get("zone") not in (None, ep.state.zone):
            path = ep.world.shortest_path(ep.state.zone, action["zone"]) \
                if action["zone"] in ep.world.zones else None
            if path is None:
                res = ep.execute(action)          # 不可达:引擎按 hold 记 1 tick
                entries += res.entries
                self.frames.append(self._frame(entered=False, entry=res.entries[-1],
                                               blocked=True))
                return entries
            for nxt in path[1:]:
                if ep.done:
                    break
                res = ep.execute(goto_action(nxt))
                entries += res.entries
                self.frames.append(self._frame(entered=True, entry=res.entries[-1]))
            return entries
        res = ep.execute({"type": "hold"})
        entries += res.entries
        if res.entries:
            self.frames.append(self._frame(entered=False, entry=res.entries[-1]))
        return entries

    @property
    def done(self) -> bool:
        return self.ep.done


# ---------------------------------------------------------------- 决策视图
_ROLE_TXT = {
    "planner": lambda c: f"提议 → {c.get('target') or (c.get('action') or {}).get('type', '?')}"
                         f":{c.get('reason', '')}",
    "critic": lambda c: ("反对 " if c.get("objection") else "无异议 ")
                        + (" ".join(c.get("flags") or []) + " " if c.get("flags") else "")
                        + (c.get("reason") or ""),
    "supporter": lambda c: ("支持:" if c.get("support", True) else "不支持:") + (c.get("reason") or ""),
    "adjudicator": lambda c: f"裁决 {c.get('verdict', '?')}:{c.get('reason', '')}",
    "guardrail": lambda c: f"静态拦截 {c.get('blocked_zone')}:{c.get('reason', '')}",
}


def decision_view(step_log: dict) -> dict:
    """trajectory 一步的决策记录 -> 可视化视图(角色发言压成一行)。"""
    d = step_log["decision"]
    exp = step_log.get("expected")
    v = d.get("verdict")
    return {
        "at": step_log["obs"]["step"],
        "proposal": d.get("proposal_target"),
        "final": d.get("final_target"),
        "verdict": v, "expected": exp,
        "ok": None if exp is None else (v == exp or (exp, v) in VERDICT_OK),
        "flags": d.get("critic_flags") or [],
        "rounds": d.get("rounds", 1), "llm_calls": d.get("llm_calls", 0),
        "goal_post": d.get("goal_post_step"),
        "transcript": [{"role": m["role"],
                        "text": _ROLE_TXT.get(m["role"], lambda c: str(c)[:120])(m["content"])}
                       for m in d.get("transcript") or []],
    }


# ---------------------------------------------------------------- 离线回放
def extract_replay(scenario: dict, trajectory: dict, stream_forum: bool = True) -> dict:
    """重演一条已落盘轨迹 -> {frames, decisions, history, summary}。

    重演即还原:引擎确定性保证同动作序列产生同 tick 流;每 tick 与原始日志
    对照(zone 与违规 cid),不一致计入 summary.mismatches(应为 0)。
    """
    sess = EpisodeSession(scenario, stream_forum=stream_forum)
    mismatch = 0
    for s in trajectory["steps"]:
        dv = decision_view(s)
        sess.add_decision(dv)
        entries = sess.act(s["decision"]["action"])
        rec = s["result"]["entries"]
        if len(entries) != len(rec):
            mismatch += 1
        else:
            for a, b in zip(entries, rec):
                if a["zone"] != b["zone"] or \
                        [v["cid"] for v in a["violations"]] != [v["cid"] for v in b["violations"]]:
                    mismatch += 1
        n_vio = sum(len(e["violations"]) for e in entries)
        act = s["decision"]["action"]
        label = {"goto": f"goto {act.get('zone')}", "hold": "hold", "return": "return"} \
            .get(act.get("type"), str(act))
        sess.history.append({
            "step": dv["at"], "label": label, "verdict": dv["verdict"],
            "expected": dv["expected"], "ok": dv["ok"], "hops": len(entries),
            "viols": n_vio, "blocked": s["result"].get("blocked", False),
        })
    ep = sess.ep
    return {
        "frames": sess.frames, "decisions": sess.decisions, "history": sess.history,
        "summary": {"success": ep.success, "steps": ep.state.step,
                    "violations": len(ep.violations),
                    "vss": sum(v.severity for v in ep.violations),
                    "brs_final": round(ep.brs_curve[-1], 4), "mismatches": mismatch},
    }


# ---------------------------------------------------------------- 契约视图
_CAT_TXT = {
    "spatial": lambda p, nm: ("允许区:" if p["rule"] == "allow" else "禁入:")
                             + "、".join(nm(z) for z in p.get("zones", [])),
    "object": lambda p, nm: ("可携:" if p["rule"] == "allow" else "禁携:")
                            + "、".join(p.get("objects", [])),
    "sensor": lambda p, nm: f"传感禁令:{p.get('condition', '')}",
    "temporal": lambda p, nm: f"{p.get('window', ['?', '?'])[0]}–{p.get('window', ['?', '?'])[1]} "
                              f"禁入 {'、'.join(nm(z) for z in p.get('zones', []))}"
                              + (f"(除非 {p['unless']})" if p.get("unless") else ""),
    "embodiment": lambda p, nm: "、".join(nm(z) for z in p.get("zones", []))
                                + (" 仅限 " if p["rule"] == "allow" else " 禁止 ")
                                + str(p.get("embodiment")),
    "escort": lambda p, nm: "护送时禁入:" + "、".join(nm(z) for z in p.get("escorted_into", [])),
}


def contract_views(contract_dict: dict, zones: list[dict]) -> list[dict]:
    """契约 -> 可读条目列表 [{cid, cat, rule, sev, text}](zone id 解析为显示名)。"""
    names = {z["id"]: z.get("name", z["id"]) for z in zones}
    nm = lambda z: names.get(z, z)  # noqa: E731
    out = []
    for c in Contract.from_dict(contract_dict).constraints:
        entry = {"rule": c.rule, **c.params}
        txt = _CAT_TXT.get(c.category, lambda p, _: str(p))(entry, nm)
        out.append({"cid": c.cid, "cat": c.category, "rule": c.rule,
                    "sev": c.severity, "text": txt})
    return out
