#!/usr/bin/env python3
"""3D 场景专项校验（设计文档 §4.3）：
① 门宽 ≥ 机器人宽 + 0.2m；② 占据栅格 BFS 可达性与场景拓扑（rooms+doors 图）逐对等价；
③ oracle 动作序列中每一步 goto 目标与当前 zone 相邻（编译期即可发现坏路径）。
    python3 scene/validate3d.py scenes/from2d/*.json [--robot-width 0.5]
"""
import argparse, glob, json, sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaac.scene_builder import compute_walls  # noqa: E402

RES = 0.1  # 栅格分辨率（米）


def occupancy(scene, inflate_m):
    rooms = scene["rooms"]
    xs = [r["aabb"][0] for r in rooms.values()] + [r["aabb"][2] for r in rooms.values()]
    ys = [r["aabb"][1] for r in rooms.values()] + [r["aabb"][3] for r in rooms.values()]
    x0, y0, x1, y1 = min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1
    W, H = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1
    occ = np.zeros((H, W), dtype=bool)
    for cx, cy, lx, ly in compute_walls(rooms, scene["doors"]):
        hx, hy = lx / 2 + inflate_m, ly / 2 + inflate_m
        i0, i1 = int((cy - hy - y0) / RES), int((cy + hy - y0) / RES) + 1
        j0, j1 = int((cx - hx - x0) / RES), int((cx + hx - x0) / RES) + 1
        occ[max(i0, 0):i1, max(j0, 0):j1] = True
    return occ, (x0, y0)


def bfs_reach(occ, origin, start_xy):
    H, W = occ.shape
    si, sj = int((start_xy[1] - origin[1]) / RES), int((start_xy[0] - origin[0]) / RES)
    seen = np.zeros_like(occ)
    if occ[si, sj]:
        return seen
    q = deque([(si, sj)]); seen[si, sj] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and not occ[a, b] and not seen[a, b]:
                seen[a, b] = True; q.append((a, b))
    return seen


ROBOT_WIDTH = {"car": 0.2, "dog": 0.5, "humanoid": 0.5}


def topo_reach(scene, robot_width=None):
    """拓扑可达（本体感知：窄门 = quadruped-only，宽度不足的门对该形态视为不通，与 2D
    world.neighbors(embodiment) 过滤同义）。"""
    adj = {z: set() for z in scene["rooms"]}
    for d in scene["doors"]:
        if robot_width is not None and d["width"] < robot_width + 0.2:
            continue
        a, b = d["between"]; adj[a].add(b); adj[b].add(a)
    start = scene["spawn"]["zone"]
    seen = {start}; q = deque([start])
    while q:
        z = q.popleft()
        for n in adj[z]:
            if n not in seen:
                seen.add(n); q.append(n)
    return seen, adj


def validate(scene, robot_width):
    issues = []
    for d in scene["doors"]:
        if d["width"] < robot_width + 0.2:
            issues.append(f"door {d['between']} width {d['width']} < robot {robot_width}+0.2")
    occ, origin = occupancy(scene, inflate_m=robot_width / 2)
    r = scene["rooms"][scene["spawn"]["zone"]]["aabb"]
    seen = bfs_reach(occ, origin, ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2))
    treach, adj = topo_reach(scene, robot_width)
    for z, room in scene["rooms"].items():
        a = room["aabb"]; cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        geo = bool(seen[int((cy - origin[1]) / RES), int((cx - origin[0]) / RES)])
        if geo != (z in treach):
            issues.append(f"zone {z}: geometric reachable={geo} vs topological={z in treach}")
    cur = scene["spawn"]["zone"]
    for act in scene.get("oracle_actions", []):
        if act.startswith("goto_"):
            tgt = act[5:]
            if tgt not in adj.get(cur, set()):
                issues.append(f"oracle {act} from {cur}: not adjacent in 3D topology")
            cur = tgt
    return issues


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--robot-width", type=float, default=None, help="默认按场景 embodiment：car 0.2 / dog,humanoid 0.5")
    a = ap.parse_args()
    bad = 0
    for pat in a.scenes:
        for f in sorted(glob.glob(pat)):
            s = json.load(open(f))
            rw = a.robot_width or ROBOT_WIDTH.get(s.get("embodiment", "dog"), 0.5)
            iss = validate(s, rw)
            bad += bool(iss)
            print(f"{Path(f).name}: {'OK' if not iss else 'ISSUES'}" + "".join(f"\n   - {i}" for i in iss))
    sys.exit(1 if bad else 0)
