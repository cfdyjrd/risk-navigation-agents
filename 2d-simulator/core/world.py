"""拓扑世界:zone 邻接图 + 离散时钟 + 机器人状态。

第一层不做几何/碰撞/渲染,环境步进只做图上集合运算。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .contract import fmt_hhmm, parse_hhmm

ZONE_KINDS = ("corridor", "ward", "icu", "pharmacy", "stair", "checkpoint",
              "retail", "gate", "vip", "lobby")
EMBODIMENTS = ("wheeled", "quadruped")


@dataclass
class Zone:
    id: str
    kind: str
    name: str          # 中文显示名,forum 文本引用它
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "attrs": dict(self.attrs)}


class World:
    """zone 邻接图。边带 traversable_by(允许通行的本体集合)。"""

    def __init__(self, zones: list[Zone], edges: list[tuple], start_zone: str,
                 clock: dict, embodiment: str, robot_init: dict,
                 objects: list[dict] | None = None, humans: list[dict] | None = None):
        self.zones: dict[str, Zone] = {z.id: z for z in zones}
        self.adj: dict[str, dict[str, frozenset]] = {z.id: {} for z in zones}
        self.edges: list[tuple] = []
        for e in edges:
            a, b, embs = e[0], e[1], frozenset(e[2])
            self.adj[a][b] = embs
            self.adj[b][a] = embs
            self.edges.append((a, b, embs))
        self.start_zone = start_zone
        self.clock = {"start": clock["start"], "minutes_per_step": int(clock["minutes_per_step"])}
        self._t0 = parse_hhmm(clock["start"])
        self.embodiment = embodiment
        self.robot_init = dict(robot_init)
        self.objects = [dict(o) for o in (objects or [])]
        self.humans = [dict(h) for h in (humans or [])]

    # ---- 时钟 ----
    def time_min(self, step: int) -> int:
        return (self._t0 + step * self.clock["minutes_per_step"]) % 1440

    def time_str(self, step: int) -> str:
        return fmt_hhmm(self.time_min(step))

    # ---- 图 ----
    def neighbors(self, zid: str, embodiment: Optional[str] = None) -> list[str]:
        emb = embodiment or self.embodiment
        return sorted(b for b, embs in self.adj[zid].items() if emb in embs)

    def shortest_path(self, a: str, b: str, embodiment: Optional[str] = None) -> Optional[list[str]]:
        """按跳数 BFS 最短路(只受本体通行性约束,不考虑契约)。"""
        if a == b:
            return [a]
        emb = embodiment or self.embodiment
        prev: dict[str, str] = {a: a}
        q = deque([a])
        while q:
            cur = q.popleft()
            for nxt in self.neighbors(cur, emb):
                if nxt in prev:
                    continue
                prev[nxt] = cur
                if nxt == b:
                    path = [b]
                    while path[-1] != a:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(nxt)
        return None

    def reachable(self, a: str, b: str, embodiment: Optional[str] = None) -> bool:
        return self.shortest_path(a, b, embodiment) is not None

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "zones": [z.to_dict() for z in self.zones.values()],
            "edges": [[a, b, sorted(embs)] for a, b, embs in self.edges],
            "start_zone": self.start_zone,
            "clock": dict(self.clock),
            "embodiment": self.embodiment,
            "robot_init": dict(self.robot_init),
            "objects": [dict(o) for o in self.objects],
            "humans": [dict(h) for h in self.humans],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        zones = [Zone(z["id"], z["kind"], z.get("name", z["id"]), dict(z.get("attrs", {})))
                 for z in d["zones"]]
        return cls(zones, [(e[0], e[1], e[2]) for e in d["edges"]], d["start_zone"],
                   d["clock"], d["embodiment"], d.get("robot_init", {}),
                   d.get("objects"), d.get("humans"))


@dataclass
class RobotState:
    zone: str
    camera_on: bool = False
    carrying: Optional[str] = None
    escorted_human: Optional[str] = None
    step: int = 0
    visited: set = field(default_factory=set)
    picked: set = field(default_factory=set)      # 已取过的 object id
    delivered: set = field(default_factory=set)   # 已送达的 object id
    escort_done: bool = False
