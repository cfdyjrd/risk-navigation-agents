"""确定性楼层平面图合成:zone 邻接图 -> 网格平面图(纯函数,零第三方依赖)。

参考 simulator_bddl 的空间坐标层(assign_spatial_coordinates + spatial_config):
仿真引擎不含任何几何,可视化前把拓扑投影成"建筑平面图"——

  - 每层楼一张网格:走廊为横向脊柱(同层多走廊按 id 依次拼接,图上相邻则共墙),
    病房/登机口贴走廊上沿一字排开,功能区(lobby/药房/ICU/安检/零售/VIP/楼梯)贴下沿;
  - 图上相邻且平面上共享边界的 zone 之间画"门";共层但不共墙的边画细连线;
  - 跨层边(楼梯/电梯)在两端 zone 上标"传送门"徽章,不画长线。

坐标单位是抽象 cell,渲染端乘 cellSize 得像素。布局只依赖 world dict,
同输入同输出(全排序、无随机),可直接进测试断言。
"""
from __future__ import annotations

# 每类 zone 的平面尺寸(cell):房间类贴上沿,功能类贴下沿
DIMS = {"ward": (4, 3), "gate": (4, 3), "icu": (5, 3), "pharmacy": (4, 3),
        "retail": (4, 3), "vip": (4, 3), "checkpoint": (4, 3), "lobby": (5, 3),
        "stair": (3, 3), "corridor": (0, 2)}
TOP_KINDS = ("ward", "gate")          # 贴走廊上沿的 kind
ROW_TOP, ROW_MID, ROW_BOT = 3, 2, 3   # 上排高 / 走廊高 / 下排高
GAP = 1                               # 未连通走廊段之间的空档


def _rects_touch(a: dict, b: dict):
    """两矩形共享边界段则返回门位置 (x, y, orient),否则 None。"""
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    if ax2 == b["x"] or bx2 == a["x"]:                      # 垂直共墙
        x = ax2 if ax2 == b["x"] else bx2
        lo, hi = max(a["y"], b["y"]), min(ay2, by2)
        if hi - lo >= 1:
            return x, (lo + hi) / 2, "v"
    if ay2 == b["y"] or by2 == a["y"]:                      # 水平共墙
        y = ay2 if ay2 == b["y"] else by2
        lo, hi = max(a["x"], b["x"]), min(ax2, bx2)
        if hi - lo >= 1:
            return (lo + hi) / 2, y, "h"
    return None


def layout_world(world: dict) -> dict:
    """world dict -> {floors: [...], portals: [...], width, height_total}。"""
    zones = world["zones"]
    edges = [(e[0], e[1], list(e[2])) for e in world["edges"]]
    adj: dict[str, set] = {z["id"]: set() for z in zones}
    for a, b, _ in edges:
        adj[a].add(b)
        adj[b].add(a)
    floor_of = {z["id"]: int(z["attrs"].get("floor", 1)) for z in zones}
    kind_of = {z["id"]: z["kind"] for z in zones}
    floors = sorted({f for f in floor_of.values()})

    placed: dict[str, dict] = {}          # zid -> {x,y,w,h,floor}
    floor_meta = []
    for f in floors:
        zf = sorted(z["id"] for z in zones if floor_of[z["id"]] == f)
        corridors = [z for z in zf if kind_of[z] == "corridor"]
        rooms = [z for z in zf if z not in corridors]

        # 锚定:每个非走廊 zone 挂到同层某走廊(相邻优先,否则借同层邻居的锚,再否则首条)
        anchor: dict[str, str] = {}
        spine = corridors or ["__spine__"]
        for z in rooms:
            cands = sorted(c for c in corridors if c in adj[z])
            if cands:
                anchor[z] = cands[0]
        for z in rooms:                                     # 二跳兜底
            if z not in anchor:
                near = sorted(n for n in adj[z] if n in anchor)
                anchor[z] = anchor[near[0]] if near else spine[0]

        # 每条走廊段的上/下排成员(id 序即楼号序,保持图相邻的房间平面相邻)
        top: dict[str, list] = {c: [] for c in spine}
        bot: dict[str, list] = {c: [] for c in spine}
        for z in rooms:
            (top if kind_of[z] in TOP_KINDS else bot)[anchor[z]].append(z)
        for c in spine:
            top[c].sort()
            bot[c].sort()

        # 走廊段从左到右拼接;段宽 = max(上排需宽, 下排需宽, 6)
        x = 0.0
        prev = None
        for c in spine:
            need = max(sum(DIMS[kind_of[z]][0] for z in top[c]),
                       sum(DIMS[kind_of[z]][0] for z in bot[c]), 6)
            if prev is not None:
                x += 0 if prev in adj.get(c, set()) else GAP
            for row, y0 in ((top[c], 0), (bot[c], ROW_TOP + ROW_MID)):
                rx = x
                for z in row:
                    w, h = DIMS[kind_of[z]]
                    placed[z] = {"x": rx, "y": y0 if y0 else ROW_TOP - h,
                                 "w": w, "h": h, "floor": f}
                    rx += w
            if c != "__spine__":
                placed[c] = {"x": x, "y": ROW_TOP, "w": need, "h": ROW_MID, "floor": f}
            x += need
            prev = c
        floor_meta.append({"floor": f, "width": x})

    width = max((m["width"] for m in floor_meta), default=6)

    # 门 / 同层连线 / 跨层传送门
    out_floors = []
    for m in floor_meta:
        f = m["floor"]
        zf = [z for z in placed if placed[z]["floor"] == f]
        doors, links = [], []
        for a, b, embs in edges:
            if a not in zf or b not in zf:
                continue
            a, b = min(a, b), max(a, b)
            d = _rects_touch(placed[a], placed[b])
            if d:
                doors.append({"a": a, "b": b, "x": d[0], "y": d[1], "o": d[2],
                              "narrow": len(embs) < 2})
            else:
                links.append({"a": a, "b": b, "narrow": len(embs) < 2})
        out_floors.append({
            "floor": f, "width": width, "height": ROW_TOP + ROW_MID + ROW_BOT,
            "zones": [{"id": z, **{k: placed[z][k] for k in ("x", "y", "w", "h")}}
                      for z in sorted(zf)],
            "doors": doors, "links": links,
        })

    portals = []
    for a, b, embs in edges:
        if floor_of[a] != floor_of[b]:
            for z, o in ((a, b), (b, a)):
                portals.append({"zone": z, "to": o, "to_floor": floor_of[o],
                                "narrow": len(embs) < 2})
    portals.sort(key=lambda p: (p["zone"], p["to"]))

    return {"floors": out_floors, "portals": portals, "width": width,
            "row_heights": [ROW_TOP, ROW_MID, ROW_BOT]}
