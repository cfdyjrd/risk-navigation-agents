"""goto 编译器：goto_<zone> → 门中点/zone 中心路点 → 宏序列（设计文档 §3.3）。

确定性小策略，~200 行。aborted(stuck/budget) 上抛给上层折算 blocked（2D 收敛语义），
本地 recover（backward_0.5 + 反向 turn_30）最多 recover_max 次。

Topology 由场景 JSON 提供（M5 scene_builder）；这里只依赖三个查询：
    zone_of(xy) / door_center(a, b) / zone_center(z) / neighbors(z)
zone 归属默认用注入的 zone_fn（M4 起接 Localizer.current_zone，之前用 GT AABB）。
"""

import math

import numpy as np


def _bearing_deg(pos, yaw, wp):
    """路点相对机头的方位角（度，左正右负）。"""
    a = math.atan2(wp[1] - pos[1], wp[0] - pos[0]) - yaw
    return math.degrees((a + math.pi) % (2 * math.pi) - math.pi)


class Topology:
    """场景 JSON 的 rooms/doors 视图（M5 完整版之前的最小实现）。

    rooms: {zone: {"aabb": [x0, y0, x1, y1]}}
    doors: {frozenset({a, b}): {"center": [x, y]}}
    """

    def __init__(self, rooms, doors):
        self.rooms = rooms
        self._doors = {frozenset(k): v for k, v in doors}   # v: {"center":[x,y], "orient": "h"|"v"|None}

    def door_orient(self, a, b):
        """门朝向：'v' 门在竖墙上（两室左右相邻），'h' 门在横墙上；缺省按两室中心相对位置推断。"""
        d = self._doors.get(frozenset({a, b})) or {}
        if d.get("orient") in ("h", "v"):
            return d["orient"]
        ca, cb = self.zone_center(a), self.zone_center(b)
        return "v" if abs(ca[0] - cb[0]) >= abs(ca[1] - cb[1]) else "h"

    def door_waypoints(self, cur, nxt, standoff=0.9):
        """穿门三段路点：门前预点（cur 侧法线偏移）→ 门中点 → 门后点（nxt 侧）。
        先对齐门洞再直穿，避免斜向撞门框/把落在墙线上的门中点当到点目标（M7 实测）。"""
        c = self.door_center(cur, nxt)
        if c is None:
            return []
        o = self.door_orient(cur, nxt)
        axis = np.array([1.0, 0.0]) if o == "v" else np.array([0.0, 1.0])   # 墙法线方向
        sign = np.sign(np.dot(self.zone_center(nxt) - c, axis)) or 1.0
        return [c - sign * standoff * axis, c, c + sign * standoff * axis]

    def zone_of(self, xy):
        for z, r in self.rooms.items():
            x0, y0, x1, y1 = r["aabb"]
            if x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1:
                return z
        return None

    def zone_center(self, z):
        x0, y0, x1, y1 = self.rooms[z]["aabb"]
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def door_center(self, a, b):
        d = self._doors.get(frozenset({a, b}))
        return np.array(d["center"]) if d else None

    def neighbors(self, z):
        return sorted({b for k in self._doors for b in k if z in k} - {z})

    def zone_path(self, start, goal):
        """BFS 最短跳数路径 start→goal（含两端，如 ["A","B","C"]）。

        start==goal 返回 [start]；任一端不在 rooms 或不可达返回 None。
        供多跳 goto / return_to_start 逐跳展开（isaac_server 消费）。
        """
        from collections import deque

        if start not in self.rooms or goal not in self.rooms:
            return None
        if start == goal:
            return [start]
        prev = {start: None}
        q = deque([start])
        while q:
            z = q.popleft()
            for nb in self.neighbors(z):
                if nb in prev:
                    continue
                prev[nb] = z
                if nb == goal:
                    path = [goal]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(nb)
        return None


class GotoCompiler:
    def __init__(self, executor, topology, zone_fn, run_cfg):
        self.ex = executor
        self.topo = topology
        self.zone_fn = zone_fn        # () -> zone（Localizer 或 GT 注入）
        self.cfg = run_cfg

    def run(self, target_zone, macro_budget):
        """当前 zone → 相邻 target_zone。返回 §3.1 goto 结果。"""
        cur = self.zone_fn()
        if cur == target_zone:
            return {"outcome": "arrived", "aborts": [], "n_macros": 0}
        door_wps = self.topo.door_waypoints(cur, target_zone) if cur else []
        waypoints = door_wps + [self.topo.zone_center(target_zone)]
        n_door = len(door_wps)

        aborts, n, recovers, obst = [], 0, 0, 0
        for wi, wp in enumerate(waypoints):
            is_door_leg = wi < n_door                      # 三段门路点全程过门豁免
            leg_start = self.ex.emb.get_pose()[0][:2].copy()
            while True:
                pos, yaw = self.ex.emb.get_pose()
                dist = float(np.linalg.norm(wp - pos[:2]))
                reach = 0.2 if is_door_leg else self.cfg["goto"]["reach_radius_m"]   # 门段 0.2m：对齐门洞
                if dist < reach:
                    break
                # 已入目标 zone：视为到达，不必再走到 zone 中心。但门后段要先离开门洞
                # ≥0.4 m（post 路点距门 0.9 m，dist<0.5）——否则 zone 在墙线一翻就停在门框里，
                # 下一次原地转向时 H1 肩宽 + 0.19 m 转向漂移直接顶到门柱（fam0121 人形两跑同点摔倒）。
                if self.zone_fn() == target_zone and (wi >= n_door or (wi == n_door - 1 and dist < 0.5)):
                    return {"outcome": "arrived", "aborts": aborts, "n_macros": n}
                if n >= macro_budget:
                    return {"outcome": "blocked", "aborts": aborts + ["macro_budget_exhausted"], "n_macros": n}
                # 过门豁免：接近门中点（<1.2 x 门豁免半宽）时收窄射线（§3.4）
                self.ex.set_door_exempt(
                    is_door_leg and dist < 1.2 * self.cfg["watchdog"]["door_exempt_halfwidth_m"]
                )
                macro = self._pick_macro(pos, yaw, self._pursuit_point(leg_start, wp, pos, lookahead=min(4.0, max(1.5, dist))), dist)
                r = self.ex.execute(macro)
                n += 1
                if r["status"] == "aborted":
                    aborts.append(r["reason"])
                    if r["reason"] == "fell_over":
                        self.ex.set_door_exempt(False)
                        return {"outcome": "fallen", "aborts": aborts, "n_macros": n}
                    if r["reason"] == "timeout":
                        continue   # 上坡/楼梯有效速度减半导致的超时：不后退，直接重新瞄准继续
                    if r["reason"].startswith("stuck"):
                        if recovers >= self.cfg["goto"]["recover_max"]:
                            self.ex.set_door_exempt(False)
                            return {"outcome": "blocked", "aborts": aborts, "n_macros": n}
                        recovers += 1
                        n += self._recover(wp)
                    if r["reason"].startswith("obstacle_ahead"):
                        # 连续撞同一障碍：先侧转再试；两次无效即后退+转向恢复（防预算空耗）
                        obst += 1
                        if obst % 3 == 0:
                            n += self._recover(wp)
                        else:
                            pos, yaw = self.ex.emb.get_pose()
                            b = _bearing_deg(pos, yaw, wp)
                            self.ex.execute("turn_left_30" if b >= 0 else "turn_right_30")
                            n += 1
        self.ex.set_door_exempt(False)
        ok = self.zone_fn() == target_zone
        return {"outcome": "arrived" if ok else "blocked", "aborts": aborts, "n_macros": n}

    def _pursuit_point(self, a, b, pos, lookahead=4.0):
        """pure pursuit：瞄准"当前位置在腿 a→b 上的投影 + 前视距离"处的点，而不是远端路点本身。
        长走廊/坡道上横向漂移 0.5m 对 20m 外目标只有 1.5° 方位差（低于 10° 转向阈值，永不纠正），
        对 4m 前视点则是 7°——漂移被及时纠回，且不产生中间到点/转向事件（坡脚转向实测会绊倒）。"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ab = b - a; L = float(np.linalg.norm(ab))
        if L < 1e-6:
            return b
        t = float(np.clip(np.dot(pos[:2] - a, ab) / (L * L), 0.0, 1.0))
        s = min(L, t * L + lookahead)
        return a + ab * (s / L)

    def _pick_macro(self, pos, yaw, wp, dist):
        """§3.3：先对准（90/30/15 三档转向），再前进（按剩余距离与前方净空选档）。"""
        b = _bearing_deg(pos, yaw, wp)
        if abs(b) > 60:
            return "turn_left_90" if b > 0 else "turn_right_90"
        if abs(b) > 25:
            return "turn_left_30" if b > 0 else "turn_right_30"
        if abs(b) > 10:
            return "turn_left_15" if b > 0 else "turn_right_15"
        d = min(dist, self.ex.sensors.front_clearance() - 0.4)
        for name, m in (("forward_2", 2.0), ("forward_1", 1.0), ("forward_0.5", 0.5)):
            if d >= m:
                return name
        return "forward_0.3"

    def _recover(self, wp):
        """stuck 本地恢复：后退半米 + 朝路点侧转 30。返回消耗宏数。"""
        self.ex.execute("backward_0.5")
        pos, yaw = self.ex.emb.get_pose()
        b = _bearing_deg(pos, yaw, wp)
        self.ex.execute("turn_left_30" if b > 0 else "turn_right_30")
        return 2
