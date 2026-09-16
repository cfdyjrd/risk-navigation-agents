"""前向射线传感（设计文档 §3.4 raycast 实现）。Isaac 进程内使用。

- 起点从机身前缘（cfg.raycast.front_edge_m）再外扩 origin_forward_offset_m，
  防腿式摆腿/俯仰时射线命中自身；命中路径仍以机器人 prim 前缀过滤兜底。
- 正前 + ±15° 共 3 条取最小值 = front_clearance；±90° 各 1 条供 observe 的左右净空。
- car 额外按 down_pitch_deg 下倾（E20 台阶对水平射线不可见）。
"""

import math

import numpy as np

CLEAR_M = 50.0  # 无命中/自身命中时的"无障碍"读数上限


class RaySensor:
    def __init__(self, emb, robot_prim_path="/World/Robot", max_dist=20.0):
        from omni.physx import get_physx_scene_query_interface

        self._q = get_physx_scene_query_interface()
        self.emb = emb
        self.rc = emb.cfg["raycast"]
        self.robot_prefix = robot_prim_path
        self.max_dist = max_dist

    def _cast(self, yaw_offset_deg, pitch_deg=0.0, ignore_ground=False):
        import carb

        pos, yaw = self.emb.get_pose()
        a = yaw + math.radians(yaw_offset_deg)
        pitch = math.radians(pitch_deg)
        d = np.array([math.cos(a) * math.cos(pitch), math.sin(a) * math.cos(pitch), math.sin(pitch)])
        start = self.rc["front_edge_m"] + self.rc["origin_forward_offset_m"]
        # 起点高度相对基座：height_m 是平地站立时的绝对高度，减去平地站立基座高 standing_base_z
        # 再加当前基座 z（楼梯/二层上基座 z 变化，绝对高度会打到楼板或悬空）
        z_rel = self.rc["height_m"] - float(self.emb.cfg.get("standing_base_z", 0.0))
        o = np.array([
            pos[0] + math.cos(yaw) * start,
            pos[1] + math.sin(yaw) * start,
            float(pos[2]) + z_rel,
        ])
        hit = self._q.raycast_closest(
            carb.Float3(*o.tolist()), carb.Float3(*d.tolist()), self.max_dist
        )
        if not hit or not hit.get("hit"):
            return CLEAR_M
        coll = str(hit.get("collision", ""))
        if coll.startswith(self.robot_prefix):
            return CLEAR_M  # 自身命中兜底过滤
        if ignore_ground and coll.startswith("/World/ground"):
            return CLEAR_M  # 下倾探针打到地面不算障碍（car 实测 0.06m 高 -20° 在 0.175m 处触地）
        # 距离换算回机身前缘系：加上外扩起点偏移
        return float(hit["distance"]) + self.rc["origin_forward_offset_m"]

    def front_clearance(self):
        """正前 + ±15° 三条水平射线取最小（§3.4 第 1 层障碍 abort 用）；
        car 再并入下倾"低矮障碍探针"（E20 台阶），探针命中地面则忽略。"""
        d = min(self._cast(0), self._cast(15), self._cast(-15))
        pitch = self.rc.get("down_pitch_deg", 0) or 0
        if pitch:
            d = min(d, self._cast(0, pitch_deg=pitch, ignore_ground=True))
        return d

    def front_center(self):
        """仅正前 1 条（过门豁免时用，阈值放宽 §3.4）。"""
        return self._cast(0)

    def side_clearance(self):
        """(left, right) ±90° 各 1 条，observe 观测用。"""
        return self._cast(90), self._cast(-90)
