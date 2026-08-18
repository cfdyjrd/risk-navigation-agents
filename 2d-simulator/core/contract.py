"""授权契约(Authorization Contract)。

契约是六类允许/禁止集,每条约束带 severity(1-5)。

与 Contextual Integrity 五要素的对应关系(schema 注释):
  - actor(信息流动的发起者)        -> embodiment 约束(哪种本体被授权行动)
  - attribute(信息/物的属性)       -> object 约束(可搬运物)、sensor 约束(可感知属性)
  - subject(信息主体)              -> sensor 条件中的 ``target.*``(被感知空间的属性)、
                                       escort 约束中的被护送人
  - recipient(接收者)              -> 第一层退化:接收者固定为机器人本身与 forum 参与者,
                                       不建模第三方转发(见 docs/DESIGN.md 局限)
  - transmission principle(传输原则)-> spatial 允许集、temporal 时间窗(unless 例外)、
                                       escort 的"不得护送进入"规则

违规判定完全是集合与图运算(见 core/violations.py),零歧义、零 LLM。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

CATEGORIES = ("spatial", "object", "sensor", "temporal", "embodiment", "escort")


def parse_hhmm(s: str) -> int:
    """'22:00' -> 分钟数(0-1439)。"""
    h, m = s.split(":")
    return (int(h) % 24) * 60 + int(m)


def fmt_hhmm(t: int) -> str:
    t %= 1440
    return f"{t // 60:02d}:{t % 60:02d}"


def in_window(t_min: int, window: list | tuple) -> bool:
    """时间窗判定,支持跨午夜(如 22:00-06:00)。窗口含起点不含终点。"""
    a, b = parse_hhmm(window[0]), parse_hhmm(window[1])
    t = t_min % 1440
    if a == b:
        return True
    if a < b:
        return a <= t < b
    return t >= a or t < b


# ---------------------------------------------------------------- 条件求值器
# 语法(刻意最小化,保证零歧义):
#   expr    := term ( ("AND"|"OR") term )*     从左向右结合,AND/OR 不混合优先级
#   term    := ["NOT"] atom
#   atom    := name | name "==" literal | name "!=" literal
#   name    := 标识符 或 "target.<attr>"(目标 zone 的属性)
# 上下文键: camera_on / carrying / escorted_human / embodiment / zone / target.*
_TOKEN = re.compile(r"\s+")


def eval_condition(expr: str, ctx: dict) -> bool:
    if not expr or not expr.strip():
        return True
    tokens = _TOKEN.split(expr.strip())
    result: Optional[bool] = None
    op = "AND"
    i = 0
    while i < len(tokens):
        neg = False
        if tokens[i].upper() == "NOT":
            neg = True
            i += 1
        name = tokens[i]
        i += 1
        if i + 1 < len(tokens) and tokens[i] in ("==", "!="):
            cmp_op, literal = tokens[i], tokens[i + 1].strip("'\"")
            i += 2
            val = _resolve(name, ctx)
            v = (str(val) == literal) if cmp_op == "==" else (str(val) != literal)
        else:
            v = bool(_resolve(name, ctx))
        if neg:
            v = not v
        if result is None:
            result = v
        elif op == "AND":
            result = result and v
        else:
            result = result or v
        if i < len(tokens):
            op = tokens[i].upper()
            if op not in ("AND", "OR"):
                raise ValueError(f"bad condition operator {op!r} in {expr!r}")
            i += 1
    return bool(result)


def _resolve(name: str, ctx: dict) -> Any:
    if name.startswith("target."):
        return (ctx.get("target") or {}).get(name[len("target."):])
    return ctx.get(name)


# ---------------------------------------------------------------- 契约对象
@dataclass(frozen=True)
class Constraint:
    cid: str          # 稳定标识,如 "temporal_0"
    category: str     # 六类之一
    rule: str         # "allow" | "deny"
    severity: int     # 1-5
    params: dict      # zones / objects / condition / window / unless / embodiment / escorted_into

    def to_entry(self) -> dict:
        e = {"rule": self.rule, "severity": self.severity}
        e.update(self.params)
        return e


class Contract:
    def __init__(self, constraints: list[Constraint]):
        self.constraints = list(constraints)
        self.by_cid = {c.cid: c for c in self.constraints}

    def __len__(self) -> int:
        return len(self.constraints)

    def by_category(self, cat: str) -> list[Constraint]:
        return [c for c in self.constraints if c.category == cat]

    @classmethod
    def from_dict(cls, d: dict) -> "Contract":
        cons = []
        for cat in CATEGORIES:
            for i, e in enumerate(d.get(cat, []) or []):
                params = {k: v for k, v in e.items() if k not in ("rule", "severity")}
                cons.append(Constraint(f"{cat}_{i}", cat, e["rule"], int(e["severity"]), params))
        return cls(cons)

    def to_dict(self) -> dict:
        out: dict = {cat: [] for cat in CATEGORIES}
        for c in self.constraints:
            out[c.category].append(c.to_entry())
        return {k: v for k, v in out.items() if v}

    # ---- 常用派生集合(供判定与静态规则编译使用) ----
    def spatial_allowed(self) -> Optional[set]:
        """spatial allow 规则的并集;无 allow 规则时返回 None(不限制)。"""
        rules = [c for c in self.by_category("spatial") if c.rule == "allow"]
        if not rules:
            return None
        s: set = set()
        for r in rules:
            s |= set(r.params.get("zones", []))
        return s

    def spatial_denied(self) -> set:
        s: set = set()
        for c in self.by_category("spatial"):
            if c.rule == "deny":
                s |= set(c.params.get("zones", []))
        return s

    def denied_objects(self) -> set:
        s: set = set()
        for c in self.by_category("object"):
            if c.rule == "deny":
                s |= set(c.params.get("objects", []))
        return s
