"""定位层 Localizer（设计文档 §4）：GT / TopDown 两档（FPV 档 M7 后按需加）。

TopDownLocalizer：消费上帝视角相机帧检测机器人顶部品红 marker。
- 检测：纯 numpy 通道阈值分割（品红 R高B高G低）→ 最大连通近似（质心+面积）；
  白色朝向三角在 marker 质心邻域内检测，向量 = 朝向。零 cv2 依赖。
- 像素→世界：场景四角预置 4 个定标色点（红/绿/蓝/黄，世界坐标已知），
  启动时从一帧解 DLT 单应——正交投影下退化为仿射同样成立，
  透视高位相机下按 marker 高度平面近似（设计 §4.1 双保险路线的统一实现）。
- confidence = 检测面积/期望面积（截断 [0,1]，量化 0.05）；无检出 pose 保持上次值
  + stale 标记 + confidence=0。

认知层（zone 判定、LLM 观测）只消费本层输出；宏内环 500Hz 闭环用本体真值（§4.2）。
"""

import numpy as np

QUANT = 0.05

# 定标点颜色掩码：名字 -> (帧 RGB bool 掩码函数, 检测顺序无关)
_CAL_MASKS = {
    "red": lambda r, g, b: (r > 170) & (g < 90) & (b < 90),
    "green": lambda r, g, b: (r < 90) & (g > 150) & (b < 90),
    "blue": lambda r, g, b: (r < 90) & (g < 110) & (b > 160),
    "yellow": lambda r, g, b: (r > 170) & (g > 150) & (b < 100),
}


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, 0
    return np.array([xs.mean(), ys.mean()]), len(xs)


def solve_homography(px, wd):
    """DLT：像素 (N,2) -> 世界 (N,2)，N>=4。返回 3x3 H。"""
    A = []
    for (u, v), (x, y) in zip(px, wd):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=float))
    return vt[-1].reshape(3, 3)


def apply_homography(H, uv):
    p = H @ np.array([uv[0], uv[1], 1.0])
    return p[:2] / p[2]


class GTLocalizer:
    """真值档：get_world_pose 直读，confidence 恒 1。"""

    def __init__(self, emb, topology):
        self.emb = emb
        self.topo = topology

    def update(self, frame=None):
        pass

    def pose(self):
        pos, yaw = self.emb.get_pose()
        return pos[:2].copy(), yaw

    def current_zone(self):
        return self.topo.zone_of(self.pose()[0])

    def confidence(self):
        return 1.0


class TopDownLocalizer:
    """视觉主线档。update(frame) 喂最新 god 帧（HxWx3/4 uint8）。"""

    def __init__(self, topology, loc_cfg, marker_size_m, calib_points):
        """calib_points: {"red": [x, y], "green": ..., "blue": ..., "yellow": ...} 世界坐标。"""
        self.topo = topology
        self.cfg = loc_cfg
        self.expected_area_px = None      # 标定后按 H 尺度估计
        self.marker_size_m = marker_size_m
        self.calib_world = calib_points
        self.H = None
        self._pose = (np.zeros(2), 0.0)
        self._conf = 0.0
        self.stale = True

    def seed(self, xy, yaw):
        """用出生位姿做初值（机器人知道自己从哪出发）；检测失效时保持而非退到原点。"""
        self._pose = (np.asarray(xy, dtype=float).copy(), float(yaw))

    # -- 标定 --
    def calibrate(self, frame):
        """从一帧解出像素→世界单应。四点齐全才成功，返回 bool。"""
        r, g, b = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int), frame[:, :, 2].astype(int)
        px, wd = [], []
        found = {}
        for name, fn in _CAL_MASKS.items():
            c, area = _centroid(fn(r, g, b))
            if c is None or area < 6:
                continue
            found[name] = (c, area)
            px.append(c)
            wd.append(np.asarray(self.calib_world[name], dtype=float))
        if len(px) < 4:
            return False
        self.H = solve_homography(px, wd)
        # 期望 marker 面积：用单应局部尺度（米/像素）折算
        c0 = px[0]
        s = np.linalg.norm(apply_homography(self.H, c0 + np.array([1, 0])) - apply_homography(self.H, c0))
        self.expected_area_px = (self.marker_size_m / max(s, 1e-9)) ** 2
        return True

    # -- 逐帧更新 --
    def update(self, frame):
        if frame is None or self.H is None:
            self._conf = 0.0
            self.stale = True
            return
        r, g, b = frame[:, :, 0].astype(int), frame[:, :, 1].astype(int), frame[:, :, 2].astype(int)
        magenta = (r > 170) & (b > 170) & (g < 120)
        c, area = _centroid(magenta)
        if c is None or area < self.cfg.get("min_area_px", 30):
            self._conf = 0.0
            self.stale = True
            return
        xy = apply_homography(self.H, c)
        # 朝向：marker 邻域内的白色三角
        rad = int(2.5 * np.sqrt(max(area, 1)))
        y0, y1 = max(0, int(c[1]) - rad), int(c[1]) + rad
        x0, x1 = max(0, int(c[0]) - rad), int(c[0]) + rad
        sub = (r[y0:y1, x0:x1] > 215) & (g[y0:y1, x0:x1] > 215) & (b[y0:y1, x0:x1] > 215)
        cw, area_w = _centroid(sub)
        if cw is not None and area_w >= 3:
            tip = apply_homography(self.H, np.array([x0, y0]) + cw)
            d = tip - xy
            yaw = float(np.arctan2(d[1], d[0]))
            self._yaw_src = "tip"
        else:
            # 白色朝向块丢失：用位移方向估朝向（位移 ≥0.08m 才可信），否则保持上次
            # （首批指标实测部分桶 yaw_err 18–39°，均来自"保持上次"回退）
            mv = xy - self._pose[0]
            if float(np.linalg.norm(mv)) >= 0.08:
                yaw = float(np.arctan2(mv[1], mv[0])); self._yaw_src = "motion"
            else:
                yaw = self._pose[1]; self._yaw_src = "hold"
        self._pose = (xy, yaw)
        self._conf = round(min(1.0, area / max(self.expected_area_px, 1.0)) / QUANT) * QUANT
        self.stale = False

    def pose(self):
        return self._pose[0].copy(), self._pose[1]

    def current_zone(self):
        return self.topo.zone_of(self._pose[0])

    def confidence(self):
        return self._conf
