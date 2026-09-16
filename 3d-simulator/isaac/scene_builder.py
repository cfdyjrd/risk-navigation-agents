"""场景构建（设计文档 §7.1）：场景 JSON → 墙体/门洞/地贴/道具/NPC/定标点 + Topology。

场景 JSON schema（scenes/schema.md）：
{
  "id": "E01_smoke",
  "rooms":  {"A": {"aabb": [x0, y0, x1, y1], "restricted": false}, ...},
  "doors":  [{"between": ["A", "B"], "center": [x, y], "width": 1.2}, ...],
  "objects": [{"id": "box1", "zone": "B", "offset": [0.3, 0.0]}, ...],
  "npcs":   [{"id": "guard1", "pos": [x, y], "role": "security"}, ...],
  "spawn":  {"zone": "A", "yaw_deg": 0},
  "goal":   {...任务字段，host 侧消费...},
  "macro_budget": 30
}

几何规则：
- 墙 = 各房间 AABB 边界共线区间合并去重（共墙只建一次）→ 减掉门洞区间 → FixedCuboid 段。
- 墙高 0.5m（俯视定位防遮挡，§4.5）、厚 0.1m；门洞宽默认 1.2m、narrow 0.7m。
- restricted 房间铺红色半透地贴（表现层）。
- 定标色点自动放在全场景包围盒四角内缩 0.5m（§4.1）。
几何计算是纯函数（compute_walls），系统 python 即可单测；Isaac 依赖只在 build() 内。
"""

import numpy as np

WALL_H, WALL_T = 2.2, 0.1   # 默认真实层高；场景 JSON "wall_height_m" 可覆盖（正交俯视定位不受墙高影响）
EPS = 1e-6


def _merge_intervals(iv):
    """[(a,b),...] 并集合并。"""
    iv = sorted((min(a, b), max(a, b)) for a, b in iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + EPS:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract(iv, holes):
    """区间列表减去洞列表。"""
    for h0, h1 in holes:
        nxt = []
        for a, b in iv:
            if h1 <= a + EPS or h0 >= b - EPS:
                nxt.append((a, b))
                continue
            if h0 > a + EPS:
                nxt.append((a, h0))
            if h1 < b - EPS:
                nxt.append((h1, b))
        iv = nxt
    return iv


def compute_walls(rooms, doors, wall_h=None):
    """→ [(cx, cy, lx, ly)] 墙段中心+长宽（含厚度）。纯函数。

    竖墙按 x 线分组（区间为 y），横墙按 y 线分组（区间为 x）；
    同线区间并集合并 = 共墙去重；门洞在其所在线上开洞。
    """
    vlines, hlines = {}, {}
    for r in rooms.values():
        x0, y0, x1, y1 = r["aabb"]
        vlines.setdefault(round(x0, 4), []).append((y0, y1))
        vlines.setdefault(round(x1, 4), []).append((y0, y1))
        hlines.setdefault(round(y0, 4), []).append((x0, x1))
        hlines.setdefault(round(y1, 4), []).append((x0, x1))

    vholes, hholes = {}, {}
    for d in doors:
        cx, cy = d["center"]
        w = d.get("width", 1.2)
        kx, ky = round(cx, 4), round(cy, 4)
        o = d.get("orient")  # 显式朝向优先（compile2d 给出）；缺省按墙线归属推断
        if (o == "v") or (o is None and kx in vlines):  # 门在竖墙上：洞是 y 区间
            vholes.setdefault(kx, []).append((cy - w / 2, cy + w / 2))
        elif (o == "h") or (o is None and ky in hlines):
            hholes.setdefault(ky, []).append((cx - w / 2, cx + w / 2))
        else:
            raise ValueError(f"door {d} 不在任何墙线上")

    walls = []
    for x, iv in vlines.items():
        for a, b in _subtract(_merge_intervals(iv), vholes.get(x, [])):
            if b - a > 2 * EPS:
                walls.append((x, (a + b) / 2, WALL_T, b - a))
    for y, iv in hlines.items():
        for a, b in _subtract(_merge_intervals(iv), hholes.get(y, [])):
            if b - a > 2 * EPS:
                walls.append(((a + b) / 2, y, b - a, WALL_T))
    return walls


def _build_stairs(FixedCuboid, ri, rp, hidden_ramp=False):
    """实心台阶：第 i 级为顶面 z0+(i+1)*riser、深 tread 的盒；两侧 0.5m 矮栏；顶部平台。
    hidden_ramp=True 时另加一条不可见的坡道碰撞体（policy 上不了真台阶时的展示兜底，需明示）。"""
    import math as _m
    from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

    riser, tread = rp["riser"], rp["tread"]
    rise = rp["z1"] - rp["z0"]; n = int(_m.ceil(rise / riser)); riser = rise / n
    w = rp["y1"] - rp["y0"]; cy = (rp["y0"] + rp["y1"]) / 2
    for i in range(n):
        # hidden_ramp 时台阶整体降一个踏步（第 i 级顶面 = z0 + i·riser）：坡道从 (x0-tread, z0) 直达 (x1, z1)，
        # 每级踏面后缘触坡、前缘低于坡面 ≤riser，坡顶与楼板齐平——坡顶 5cm 下台阶实测把狗摔在门口
        top = rp["z0"] + (i if hidden_ramp else i + 1) * riser
        if top - rp["z0"] < 1e-6:
            continue
        FixedCuboid(prim_path=f"/World/stairs/s{ri}_{i}",
                    position=np.array([rp["x0"] + (i + 0.5) * tread, cy, (rp["z0"] + top) / 2]),
                    scale=np.array([tread, w, top - rp["z0"]]))
    L = n * tread
    plat = float(rp.get("platform_len", 1.8))
    FixedCuboid(prim_path=f"/World/stairs/s{ri}_top", position=np.array([rp["x0"] + L + plat / 2, cy, rp["z1"] - 0.1]),
                scale=np.array([plat, w, 0.2]))
    for side, yy in (("l", rp["y0"]), ("r", rp["y1"])):
        # 栏杆：走面之上 1.2m（矮栏 0.6 被 Spot 跨过/漂出，实测摔下平台）
        FixedCuboid(prim_path=f"/World/stairs/s{ri}_{side}", position=np.array([rp["x0"] + L / 2, yy, rp["z0"] + rise / 2 + riser + 0.5]),
                    orientation=euler_angles_to_quats(np.array([0.0, -_m.degrees(_m.atan2(rise, L)), 0.0]), degrees=True),
                    scale=np.array([_m.hypot(L, rise) + 0.6, 0.1, 1.4]))
        # 顶部平台两侧墙（到上层门为止，约 3m）
        FixedCuboid(prim_path=f"/World/stairs/s{ri}_{side}_top", position=np.array([rp["x0"] + L + plat / 2, yy, rp["z1"] + 0.7]),
                    scale=np.array([plat + 0.4, 0.1, 1.4]))
    if hidden_ramp:
        deg = _m.degrees(_m.atan2(rise, L))
        # 坡面 = 踏步鼻线 + 一个踏步高：每级踏面后缘刚好触坡、前缘悬空 ≤riser（不能低于踏面，否则台阶
        # 前缘会绊脚——平地 policy 会像真台阶一样摔）
        # 坡面线 z = z0 + riser + (x - x0)·slope 在 x0 处高出地面一个踏步（坡脚 5cm 台阶——实测两次绊倒），
        # 故把坡道向后延伸一个踏面，使其在 x0 - tread 处从地面平滑起坡；顶端到 z1 + riser（5cm 下台阶到楼板，可接受）
        Lx = L + tread                       # 从 x0 - tread（地面）到 x1（楼板高度 z1），无坡顶落差
        deg = _m.degrees(_m.atan2(rise, Lx))
        cx_r = rp["x0"] - tread + Lx / 2
        cz_r = rp["z0"] + rise / 2 - 0.02
        prim = FixedCuboid(prim_path=f"/World/stairs/s{ri}_hidden_ramp",
                           position=np.array([cx_r, cy, cz_r]),
                           orientation=euler_angles_to_quats(np.array([0.0, -deg, 0.0]), degrees=True),
                           scale=np.array([_m.hypot(Lx, rise), w, 0.04]), visible=False)


def scene_bbox(rooms):
    a = np.array([r["aabb"] for r in rooms.values()])
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max())


def calib_points(rooms, inset=0.5, occluded=()):
    """四个标定色点：默认取场景外包框四角内缩 inset；若某角落在带顶棚的 zone（occluded）内，
    俯视相机看不到该点 → 单应求解只剩 3 点失败（E16b 探针即此），改取距该角最近的未遮挡房间角。"""
    x0, y0, x1, y1 = scene_bbox(rooms)
    corners = {"red": (x0, y0, +1, +1), "green": (x1, y0, -1, +1),
               "blue": (x1, y1, -1, -1), "yellow": (x0, y1, +1, -1)}
    occ = [rooms[z]["aabb"] for z in occluded if z in rooms]

    def _hidden(pt):
        return any(a[0] <= pt[0] <= a[2] and a[1] <= pt[1] <= a[3] for a in occ)

    out = {}
    for name, (cx, cy, sx, sy) in corners.items():
        cand = [[cx + sx * inset, cy + sy * inset]]
        for r in rooms.values():
            a = r["aabb"]
            for px, py in ((a[0], a[1]), (a[2], a[1]), (a[2], a[3]), (a[0], a[3])):
                cand.append([px + (inset if px < (a[0] + a[2]) / 2 else -inset),
                             py + (inset if py < (a[1] + a[3]) / 2 else -inset)])
        cand.sort(key=lambda q: (q[0] - cx) ** 2 + (q[1] - cy) ** 2)
        out[name] = next((q for q in cand if not _hidden(q)), cand[0])
    return out


def build(scene, world):
    """把场景 JSON 落到 USD stage。返回 (topology, meta)。"""
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane, VisualCuboid
    from pxr import UsdLux

    from isaac.cameras import add_calib_dots
    from isaac.goto_compiler import Topology

    rooms, doors = scene["rooms"], scene["doors"]
    GroundPlane(prim_path="/World/ground", size=max(40, int(max(scene_bbox(rooms)[2:]) * 3)))
    UsdLux.DomeLight.Define(world.stage, "/World/dome").CreateIntensityAttr(1000)

    wall_h = float(scene.get("wall_height_m", WALL_H))
    zof = {zid: float(r.get("z", 0.0)) for zid, r in rooms.items()}
    # 按层分组建墙（各层墙线独立，层高不同），坡道段的两侧留空（坡道自带矮栏）
    by_z = {}
    for zid, r in rooms.items():
        by_z.setdefault(zof[zid], {})[zid] = r
    ramp_spans = [(rp["x0"], rp["x1"], rp["y0"], rp["y1"]) for rp in scene.get("ramps", [])]
    wi = 0
    for zlev, rz in by_z.items():
        dz = [d for d in doors if d["between"][0] in rz or d["between"][1] in rz]
        for (cx, cy, lx, ly) in compute_walls(rz, dz):
            # 跳过落在坡道 x 区间内的横墙段（坡道走廊两侧不建 2.2m 墙，改为矮栏），竖墙保留
            if any(x0 - 0.05 <= cx <= x1 + 0.05 and lx > ly for (x0, x1, y0, y1) in ramp_spans):
                continue
            FixedCuboid(prim_path=f"/World/walls/w{wi}", position=np.array([cx, cy, zlev + wall_h / 2]),
                        scale=np.array([lx, ly, wall_h])); wi += 1
        if zlev > 0:   # 楼板：该层所有房间包围盒，厚 0.2，顶面 = zlev
            a = np.array([r["aabb"] for r in rz.values()])
            x0, y0, x1, y1 = a[:, 0].min() - 0.15, a[:, 1].min() - 0.15, a[:, 2].max() + 0.15, a[:, 3].max() + 0.15
            FixedCuboid(prim_path=f"/World/slab/z{int(zlev * 100)}",
                        position=np.array([(x0 + x1) / 2, (y0 + y1) / 2, zlev - 0.1]),
                        scale=np.array([x1 - x0, y1 - y0, 0.2]))
    for ri, rp in enumerate(scene.get("ramps", [])):
        import math as _m
        L = rp["x1"] - rp["x0"]; rise = rp["z1"] - rp["z0"]
        if rp.get("kind") == "stairs":
            _build_stairs(FixedCuboid, ri, rp, hidden_ramp=bool(scene.get("stairs_hidden_ramp", False)))
            continue
        deg = _m.degrees(_m.atan2(rise, L)); slen = _m.hypot(L, rise); w = rp["y1"] - rp["y0"]
        cx, cy = (rp["x0"] + rp["x1"]) / 2, (rp["y0"] + rp["y1"]) / 2
        from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
        q = euler_angles_to_quats(np.array([0.0, -deg, 0.0]), degrees=True)
        FixedCuboid(prim_path=f"/World/ramps/r{ri}", position=np.array([cx, cy, rp["z0"] + rise / 2 - 0.1]),
                    orientation=q, scale=np.array([slen, w, 0.2]))
        for side, yy in (("l", rp["y0"]), ("r", rp["y1"])):   # 矮栏 0.5m（不遮俯视）
            FixedCuboid(prim_path=f"/World/ramps/r{ri}_{side}", position=np.array([cx, yy, rp["z0"] + rise / 2 + 0.2]),
                        orientation=q, scale=np.array([slen, 0.1, 0.6]))
        # 坡道前后的平台段（走廊延伸的平地部分）：低层端 z0，高层端 z1
        FixedCuboid(prim_path=f"/World/ramps/r{ri}_top", position=np.array([rp["x1"] + 0.6, cy, rp["z1"] - 0.1]),
                    scale=np.array([1.2, w, 0.2]))

    for name, r in rooms.items():
        if r.get("restricted"):
            x0, y0, x1, y1 = r["aabb"]
            VisualCuboid(prim_path=f"/World/decals/{name}",
                         translation=np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.005]),
                         scale=np.array([x1 - x0 - 0.2, y1 - y0 - 0.2, 0.01]),
                         color=np.array([0.8, 0.1, 0.1]))

    topo = Topology(rooms=rooms, doors=[((d["between"][0], d["between"][1]),
                                         {"center": d["center"], "orient": d.get("orient")}) for d in doors])

    for o in scene.get("objects", []):
        c = topo.zone_center(o["zone"]) + np.array(o.get("offset", [0, 0]))
        DynamicCuboid(prim_path=f"/World/objects/{o['id']}", position=np.array([c[0], c[1], zof.get(o["zone"], 0.0) + 0.15]),
                      scale=np.array([0.2, 0.2, 0.2]), color=np.array([0.9, 0.5, 0.1]))

    for n in scene.get("npcs", []):
        zb = zof.get(n.get("zone", ""), 0.0)
        if n.get("script"):   # E18 动态行人：带碰撞的盒体，isaac_server 按 script.path/speed 逐物理步移动
            FixedCuboid(prim_path=f"/World/npcs/{n['id']}", position=np.array([n["pos"][0], n["pos"][1], zb + 0.85]),
                        scale=np.array([0.45, 0.45, 1.7]), color=np.array([0.9, 0.3, 0.2]))
        else:
            VisualCuboid(prim_path=f"/World/npcs/{n['id']}",
                         translation=np.array([n["pos"][0], n["pos"][1], zb + 0.6]),
                         scale=np.array([0.4, 0.4, 1.2]), color=np.array([0.2, 0.4, 0.9]))

    # E16b 顶棚遮挡：给指定 zone 加一块不透明顶棚（俯视 marker 被遮 → 置信度坠落 → observe/ask 有物理因果）
    for i, oc in enumerate(scene.get("occluders", [])):
        r = rooms[oc["zone"]]["aabb"]; zb = zof.get(oc["zone"], 0.0)
        h = float(oc.get("height_m", 2.4))
        FixedCuboid(prim_path=f"/World/occluders/o{i}", position=np.array([(r[0] + r[2]) / 2, (r[1] + r[3]) / 2, zb + h]),
                    scale=np.array([r[2] - r[0], r[3] - r[1], 0.1]), color=np.array([0.3, 0.3, 0.3]))

    cal = calib_points(rooms, occluded=[oc["zone"] for oc in scene.get("occluders", [])])

    def _z_of(xy):   # 色点所在房间的层高（不在任何房间内则取最近房间）
        z = topo.zone_of(xy)
        if z is None:
            z = min(rooms, key=lambda k: (rooms[k]["aabb"][0] - xy[0]) ** 2 + (rooms[k]["aabb"][1] - xy[1]) ** 2)
        return zof.get(z, 0.0)

    add_calib_dots(cal, z_of=_z_of)

    spawn_zone = scene["spawn"]["zone"]
    meta = {
        "spawn_xy": topo.zone_center(spawn_zone),
        "spawn_z": zof.get(spawn_zone, 0.0),
        "spawn_zone": spawn_zone,
        "calib_points": cal,
        "bbox": scene_bbox(rooms),
        "macro_budget": scene.get("macro_budget", 30),
    }
    return topo, meta
