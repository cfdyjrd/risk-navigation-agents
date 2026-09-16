#!/usr/bin/env python3
"""2D scenario JSON → 3D 场景 JSON 编译器（设计文档 §7.1，M7 核心）。

输入：risk-navigation-agents/2d-simulator 的场景（world/contract_gt/task/forum/meta）。
输出：3d-simulator scenes/schema.md 定义的场景 JSON，附 2D 原字段透传 + oracle 动作序列。

几何：复用 2D 的 eval/spatial_layout.layout_world（确定性楼层平面图），cell→米 乘 CELL；
多层压平为单层：第 i 层沿 x 平移 i×(width+GAPX) cells；跨层边中"走廊↔走廊"的第一条
实现为桥（下层走廊矩形向右延伸到上层走廊左边缘，开一扇门），其余跨层边丢弃并记录
（wheeled 不可爬楼，设计 §4.2）。同层但不共墙的 links 暂不实现（记录并丢弃）。

oracle：在"已实现边"的世界图上跑 2D core.planner.find_compliant_plan，动作序列以
goto_<zone>/hold 字符串写入 scene["oracle_actions"]（runner --planner oracle 消费）。

用法：python3 scene/compile2d.py <2d_scenario.json> [-o out.json] [--cell 1.2]
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # 3d-simulator/
SIM2D = ROOT.parent / "2d-simulator"
if str(SIM2D) not in sys.path:
    sys.path.append(str(SIM2D))

CELL_DEFAULT = 1.2
GAPX = 4                     # 层间桥长（cells）
DOOR_W, DOOR_W_NARROW = 1.2, 0.7
EMB_MAP = {"wheeled": "car", "quadruped": "dog"}


FLOOR_RISE = 2.8             # 两层模式：每层抬高（2.2 墙 + 0.6 楼板/净空）
RAMP_DEG_DEFAULT = 10.0      # 坡道坡度（scripts/diag_ramp.py 实测 Spot 可爬范围内取值）


STAIR_TREAD = 0.35           # 楼梯踏面深度（m）


def compile_scenario(s2d: dict, cell: float = CELL_DEFAULT, embodiment: str | None = None,
                     ramp_deg: float | None = None, stair_riser: float | None = None,
                     stair_tread: float = STAIR_TREAD) -> dict:
    """ramp_deg 非 None → 两层模式：第 i 层放在 z=i*FLOOR_RISE 的楼板上，层间"桥"改为坡道
    （长度按坡度算），机器狗沿坡走上二层（车不可爬，与 2D 的 [quadruped] 边语义一致）。"""
    from eval.spatial_layout import layout_world

    world = s2d["world"]
    L = layout_world(world)
    zones2d = {z["id"]: z for z in world["zones"]}
    floor_of = {z["id"]: int(z["attrs"].get("floor", 1)) for z in world["zones"]}
    floors = [fo["floor"] for fo in L["floors"]]
    # 层偏移按"实际最右边界"累加（layout_world 报告的 width 可能小于真实占位——fam0000 实测
    # 二层与一层同 y 带重叠），保证各层矩形永不重叠
    import math as _m
    gap_cells = GAPX
    if stair_riser is not None:
        ramp_deg = _m.degrees(_m.atan2(stair_riser, stair_tread))   # 楼梯等效坡度，仅用于布局长度
        n_steps = _m.ceil(FLOOR_RISE / stair_riser)
        gap_cells = _m.ceil(n_steps * stair_tread / cell) + 5        # 楼梯长 + 坡脚 1 + 坡顶平台 3.5（对齐门洞的跑道）
    elif ramp_deg is not None:
        gap_cells = _m.ceil(FLOOR_RISE / _m.tan(_m.radians(ramp_deg)) / cell) + 2   # 坡道水平长 + 两端平台
    offset, x_cursor = {}, 0.0
    for fo in L["floors"]:
        offset[fo["floor"]] = x_cursor
        x_cursor += max(z["x"] + z["w"] for z in fo["zones"]) + gap_cells
    floor_z = {f: (i * FLOOR_RISE if ramp_deg is not None else 0.0) for i, f in enumerate(floors)}

    # 1) 房间矩形（cells，含层偏移）。跨层桥的可行性决定是否把上层整层水平镜像：
    #    桥 = 下层"最右走廊"向右延伸到上层"最左走廊"的左边缘。若 2D 的跨层走廊边指向的
    #    上层走廊在最右侧，则镜像上层（x → W - x），使其成为最左；其余情形丢弃该边。
    kind = {zid: z["kind"] for zid, z in zones2d.items()}
    floor_zones = {fo["floor"]: fo for fo in L["floors"]}
    mirror = {f: False for f in floors}

    def corridors_sorted(f, mirrored):
        zs = [z for z in floor_zones[f]["zones"] if kind[z["id"]] == "corridor"]
        if mirrored:
            W = max(z["x"] + z["w"] for z in floor_zones[f]["zones"])
            return sorted(zs, key=lambda z: W - (z["x"] + z["w"]))
        return sorted(zs, key=lambda z: z["x"])

    bridge_plan = {}   # (lo_floor, hi_floor) -> (lo_zone, hi_zone, embs)
    for a, b, embs in world["edges"]:
        fa, fb = floor_of[a], floor_of[b]
        if fa == fb or kind[a] != "corridor" or kind[b] != "corridor":
            continue
        lo, hi = (a, b) if fa < fb else (b, a)
        pair = (floor_of[lo], floor_of[hi])
        if floors.index(pair[1]) != floors.index(pair[0]) + 1 or pair in bridge_plan:
            continue
        lo_cs = corridors_sorted(pair[0], mirror[pair[0]])
        hi_cs = corridors_sorted(pair[1], False)
        if lo_cs[-1]["id"] != lo:          # 下层走廊须是最右（延伸不穿过同层其它走廊）
            continue
        if hi_cs[0]["id"] == hi:
            bridge_plan[pair] = (lo, hi, embs)
        elif hi_cs[-1]["id"] == hi:        # 上层走廊在最右：镜像上层
            mirror[pair[1]] = True
            bridge_plan[pair] = (lo, hi, embs)

    rect = {}
    for fo in L["floors"]:
        f = fo["floor"]; ox = offset[f]
        W = max(z["x"] + z["w"] for z in fo["zones"])
        for z in fo["zones"]:
            x0 = (W - (z["x"] + z["w"])) if mirror[f] else z["x"]
            rect[z["id"]] = [x0 + ox, z["y"], x0 + ox + z["w"], z["y"] + z["h"]]

    # 2) 门（同层共墙边；镜像层的门 x 同步镜像）
    doors, realized = [], set()
    for fo in L["floors"]:
        f = fo["floor"]; ox = offset[f]
        W = max(z["x"] + z["w"] for z in fo["zones"])
        for d in fo["doors"]:
            dx = (W - d["x"]) if mirror[f] else d["x"]
            doors.append({"between": [d["a"], d["b"]], "center": [dx + ox, d["y"]],
                          "orient": d["o"], "width": DOOR_W_NARROW if d["narrow"] else DOOR_W})
            realized.add(frozenset((d["a"], d["b"])))

    dropped = []
    for fo in L["floors"]:
        for lk in fo["links"]:
            dropped.append({"edge": [lk["a"], lk["b"]], "reason": "same-floor link without shared wall"})

    # 3) 跨层桥落地
    bridged = set(); ramps = []
    for pair, (lo, hi, embs) in bridge_plan.items():
        xr = rect[hi][0]
        x_lo_end = rect[lo][2]
        rect[lo][2] = xr
        y_mid = (rect[lo][1] + rect[lo][3]) / 2
        doors.append({"between": [lo, hi], "center": [xr, y_mid], "orient": "v",
                      "width": DOOR_W_NARROW if len(embs) < 2 else DOOR_W})
        realized.add(frozenset((lo, hi))); bridged.add(frozenset((lo, hi)))
        if ramp_deg is not None:   # 层间段占据 [x_lo_end, xr]（cells）：坡道或楼梯，从 lo 层高升到 hi 层高
            ramps.append({"zone": lo, "x0": x_lo_end + 1, "x1": xr - 3.5, "y0": rect[lo][1], "y1": rect[lo][3],
                          "platform_cells": 3.5,
                          "z0": floor_z[pair[0]], "z1": floor_z[pair[1]],
                          "kind": "stairs" if stair_riser is not None else "ramp",
                          "riser": stair_riser, "tread": stair_tread})
    for a, b, embs in world["edges"]:
        if floor_of[a] != floor_of[b] and frozenset((a, b)) not in bridged:
            dropped.append({"edge": [a, b], "reason": "cross-floor edge not bridgeable"})

    # 4) cells → 米
    rooms = {}
    for zid, r in rect.items():
        z = zones2d[zid]
        rooms[zid] = {"aabb": [round(v * cell, 3) for v in r], "kind": z["kind"],
                      "name": z.get("name", zid), "z": floor_z[floor_of[zid]],
                      **{k: v for k, v in z["attrs"].items()}}
    for d in doors:
        d["center"] = [round(v * cell, 3) for v in d["center"]]
        za, zb = floor_z[floor_of[d["between"][0]]], floor_z[floor_of[d["between"][1]]]
        d["z"] = max(za, zb) if ramp_deg is not None else 0.0   # 层间门在上层高度（坡顶）
    for rp in ramps:
        for k in ("x0", "x1", "y0", "y1"):
            rp[k] = round(rp[k] * cell, 3)
        # 顶部平台必须从楼梯实际末端(x0 + n·tread)一直铺到上层门线(x1 + platform_cells·cell = xr)，
        # 否则平台与楼板之间留洞（实测 0.95m 空隙，狗在门前掉落）
        if rp.get("kind") == "stairs":
            n_st = _m.ceil((rp["z1"] - rp["z0"]) / rp["riser"])
            stairs_end = rp["x0"] + n_st * rp["tread"]
            door_x = rp["x1"] + rp.get("platform_cells", 1.5) * cell
            rp["platform_len"] = round(door_x - stairs_end + 0.15, 3)   # +0.15 与楼板搭接
        else:
            rp["platform_len"] = round(rp.get("platform_cells", 1.5) * cell, 3)

    # 5) 实体
    objects = [{"id": o["id"], "zone": o["zone"], "offset": [0.0, 0.0], "name": o.get("name", "")}
               for o in world.get("objects", [])]
    npcs = []
    for h in world.get("humans", []):
        r = rooms[h["zone"]]["aabb"]
        npcs.append({"id": h["id"], "pos": [(r[0] + r[2]) / 2 + 0.6, (r[1] + r[3]) / 2],
                     "role": "human", "name": h.get("name", "")})

    emb3d = embodiment or EMB_MAP.get(world["embodiment"], "dog")
    scene = {
        "id": s2d["meta"].get("scenario_id", "scn"),
        "source_2d": s2d["meta"],
        "rooms": rooms, "doors": doors, "objects": objects, "npcs": npcs,
        "spawn": {"zone": world["start_zone"], "yaw_deg": 0},
        "task": s2d["task"], "goal": dict(s2d["task"]),
        "contract_gt": s2d["contract_gt"], "forum": s2d.get("forum", []),
        "annotations": s2d.get("annotations", {}),            # hazard/drift/ambiguous 目标（report 用）
        "expected_adjudication": s2d.get("expected_adjudication", {}),
        "clock": world["clock"], "robot_init": world.get("robot_init", {}),
        "embodiment": emb3d, "embodiment_2d": world["embodiment"],
        "horizon": int(s2d["meta"].get("horizon", 20)),
        "cell_m": cell, "dropped_edges": dropped,
        "wall_height_m": 2.2,
        "ramps": ramps if ramp_deg is not None else [],
        "floor_z": {str(k): v for k, v in floor_z.items()},
        "mode": ("stairs" if stair_riser is not None else "ramp") if ramp_deg is not None else "flat",
        "macro_budget": 80 if ramp_deg is not None else 40,   # 两层模式：层间段 20m 坡道/楼梯需更多宏
    }

    # 6) oracle：已实现边的世界图上做合规规划
    scene["amendments"] = s2d.get("amendments", [])
    scene["world_staff"] = world.get("staff", [])
    try:
        from core.contract import Contract
        from core.planner import find_compliant_plan, find_timed_plan, truth_timeline
        from core.world import World
        edges_ok = [e for e in world["edges"] if frozenset((e[0], e[1])) in realized]
        w2 = World.from_dict({**world, "edges": edges_ok})
        amends = scene["amendments"]
        if any(a.get("legit") for a in amends) or s2d["task"].get("extra_targets"):
            # 与 2D run_sim_planner.run_rule 同构：合法修订按 step 生效的时变真值 oracle
            plan = find_timed_plan(w2, truth_timeline(s2d["contract_gt"], s2d["task"], amends),
                                   scene["horizon"])
        else:
            plan = find_compliant_plan(w2, Contract.from_dict(s2d["contract_gt"]), s2d["task"],
                                       scene["horizon"])
        scene["oracle_actions"] = ([f"goto_{a['zone']}" if a["type"] == "goto" else "hold"
                                    for a in plan] if plan is not None else [])
        scene["oracle_status"] = "ok" if plan is not None else "no_compliant_plan"
    except Exception as exc:  # 2D 仓库不可用时仍产出几何
        scene["oracle_actions"], scene["oracle_status"] = [], f"error: {exc!r}"
    return scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("-o", "--out")
    ap.add_argument("--cell", type=float, default=CELL_DEFAULT)
    ap.add_argument("--embodiment", choices=["car", "dog", "humanoid"])
    ap.add_argument("--ramp", type=float, default=None, help="两层模式：坡道坡度（度），如 10")
    ap.add_argument("--stairs", type=float, default=None, help="两层模式：楼梯踏步高（m），如 0.06")
    ap.add_argument("--tread", type=float, default=STAIR_TREAD, help="楼梯踏面深度（m）")
    ap.add_argument("--hidden-ramp", action="store_true", help="台阶下加不可见坡道碰撞体（平地 policy 展示兜底，须明示）")
    a = ap.parse_args()
    s2d = json.load(open(a.scenario))
    scene = compile_scenario(s2d, a.cell, a.embodiment, a.ramp, a.stairs, a.tread)
    if a.hidden_ramp:
        scene["stairs_hidden_ramp"] = True
    sub = "from2d_stairs" if a.stairs else ("from2d_ramp" if a.ramp else "from2d")
    out = a.out or str(ROOT / "scenes" / sub / f"{scene['id']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(scene, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{scene['id']}: rooms={len(scene['rooms'])} doors={len(scene['doors'])} "
          f"dropped={len(scene['dropped_edges'])} embodiment={scene['embodiment']} "
          f"oracle={scene['oracle_status']} actions={scene['oracle_actions']} -> {out}")


if __name__ == "__main__":
    main()
