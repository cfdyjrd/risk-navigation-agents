"""双视角相机与视频落盘（设计文档 §6）。Isaac 进程内使用。

- god_cam：场景正上方，优先正交投影（M4 首日验证；失败自动留在透视——
  Localizer 的单应两种投影都成立）。身兼定位输入与出片双职（§4.5）。
- fpv_cam：car/dog 刚性挂载到机身 link；humanoid 的 EMA 跟随 M7 出片时再加。
- 帧落盘：imageio 三路 mp4（god/fpv/sbs），warmup 冲陈旧帧 + 判空判维（§6.2）。
"""

import os

import numpy as np


def god_framing(bbox, resolution=(960, 720), pad_m=1.5):
    """按场景包围盒选相机 yaw 与口径：长轴对齐图像长边（yaw=0 时世界 +X 映射到图像竖边，
    宽场景会被竖边截掉——from2d 场景 38m 走廊实测定标点全在画面外）。
    返回 (extent_m, yaw_deg)。"""
    x0, y0, x1, y1 = bbox
    dx, dy = x1 - x0, y1 - y0
    aspect = resolution[1] / resolution[0]           # 竖/横
    if dx >= dy:   # 长轴是 X：转 90° 让 X 落到图像横边
        return max(dx, dy / aspect) + pad_m, 90.0
    return max(dy, dx / aspect) + pad_m, 0.0


def make_god_camera(world, center_xy, extent_m, resolution=(960, 720), height=12.0, yaw_deg=0.0):
    """建正上方相机；尝试正交投影。返回 (camera, is_orthographic)。extent_m 为图像横边覆盖米数。"""
    from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom

    cam = Camera(
        prim_path="/World/god_cam",
        position=np.array([center_xy[0], center_xy[1], height]),
        orientation=euler_angles_to_quats(np.array([0.0, 90.0, yaw_deg]), degrees=True),
        resolution=resolution,
    )
    ortho = False
    try:
        usd_cam = UsdGeom.Camera(world.stage.GetPrimAtPath("/World/god_cam"))
        usd_cam.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        # USD 正交 aperture 单位是"场景单位的十分之一"：width_m x10
        aspect = resolution[1] / resolution[0]
        usd_cam.GetHorizontalApertureAttr().Set(extent_m * 10.0)
        usd_cam.GetVerticalApertureAttr().Set(extent_m * aspect * 10.0)
        ortho = usd_cam.GetProjectionAttr().Get() == UsdGeom.Tokens.orthographic
    except Exception:
        ortho = False
    return cam, ortho


class FpvFollowCam:
    """FPV 跟随相机：不 parent 到机身 link（链接系旋转不可控，M4 实测视轴翻车），
    每渲染帧按机器人位姿重设世界位姿。alpha=1 刚性跟随（car/dog）；
    humanoid 用 alpha<1 低通抑躯干晃动（§6.1）。自由相机默认 +X 朝前（已诊断验证）。"""

    def __init__(self, offset_fwd=0.4, offset_z=0.1, alpha=1.0, pitch_deg=8.0, resolution=(960, 720)):
        import omni.usd
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        self.cam = Camera(prim_path="/World/fpv_cam", position=np.array([0.0, 0.0, 1.0]),
                          resolution=resolution)
        # 默认 ~60° 水平 FOV 在室内看不到门柱/地标（M4 实测整段视频零特征）；
        # focal 10.5mm + aperture 20.955mm ≈ 90° 水平 FOV
        UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath("/World/fpv_cam")) \
            .GetFocalLengthAttr().Set(10.5)
        self.offset_fwd = offset_fwd
        self.offset_z = offset_z
        self.alpha = alpha
        self.pitch_deg = pitch_deg
        self._smooth = None

    def initialize(self):
        self.cam.initialize()

    def update(self, base_pos, yaw):
        from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

        target = np.array([
            base_pos[0] + np.cos(yaw) * self.offset_fwd,
            base_pos[1] + np.sin(yaw) * self.offset_fwd,
            base_pos[2] + self.offset_z,
        ])
        if self._smooth is None or self.alpha >= 1.0:
            self._smooth = target
        else:
            self._smooth = self.alpha * target + (1 - self.alpha) * self._smooth
        self.cam.set_world_pose(
            self._smooth,
            euler_angles_to_quats(np.array([0.0, self.pitch_deg, np.degrees(yaw)]), degrees=True),
            camera_axes="world",  # 不传默认按 USD 轴（-Z 视轴）解释，实测又是俯视地面
        )

    def grab(self):
        return grab(self.cam)


def _bind_emissive(prim_path, rgb):
    """给定位用色块绑自发光材质——检测不依赖场景光照（实测反射色暗到无法分割）。"""
    import omni.usd
    from pxr import Gf, Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat = UsdShade.Material.Define(stage, f"{prim_path}_mat")
    sh = UsdShade.Shader.Define(stage, f"{prim_path}_mat/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(mat)


def add_marker(stage_path_body, size_m=0.4, tri_m=0.16, z_offset=0.25):
    """机器人顶部品红定位板 + 前缘白色朝向块（§4.1）。纯视觉、无碰撞、自发光。"""
    from isaacsim.core.api.objects import VisualCuboid

    VisualCuboid(
        prim_path=f"{stage_path_body}/loc_marker",
        translation=np.array([0.0, 0.0, z_offset]),
        scale=np.array([size_m, size_m, 0.02]),
        color=np.array([1.0, 0.0, 1.0]),
    )
    _bind_emissive(f"{stage_path_body}/loc_marker", (1.0, 0.0, 1.0))
    VisualCuboid(
        prim_path=f"{stage_path_body}/loc_marker_tip",
        translation=np.array([size_m / 2 + tri_m / 2, 0.0, z_offset]),
        scale=np.array([tri_m, tri_m, 0.022]),
        color=np.array([1.0, 1.0, 1.0]),
    )
    _bind_emissive(f"{stage_path_body}/loc_marker_tip", (1.0, 1.0, 1.0))


def add_calib_dots(points_world, size_m=0.35, z_of=None):
    """场景四角定标色点（红/绿/蓝/黄，自发光），Localizer.calibrate 消费（§4.1）。
    z_of(xy)->z：两层模式下色点放在所在层的地面/楼板上（正交投影下单应与 z 无关，
    但放在楼板下面会被楼板遮住——楼梯示例实测标定失败）。"""
    from isaacsim.core.api.objects import VisualCuboid

    colors = {"red": [1, 0, 0], "green": [0, 1, 0], "blue": [0, 0, 1], "yellow": [1, 1, 0]}
    for name, xy in points_world.items():
        z = (z_of(xy) if z_of else 0.0) + 0.01
        VisualCuboid(
            prim_path=f"/World/calib/{name}",
            translation=np.array([xy[0], xy[1], z]),
            scale=np.array([size_m, size_m, 0.02]),
            color=np.array(colors[name], dtype=float),
        )
        _bind_emissive(f"/World/calib/{name}", tuple(float(c) for c in colors[name]))


def grab(cam):
    """判空判维取一帧 RGB（HxWx3 uint8）；无效返回 None。"""
    rgba = cam.get_rgba()
    if rgba is None or getattr(rgba, "ndim", 0) != 3:
        return None
    return rgba[:, :, :3]


class VideoSink:
    """三路 mp4 落盘：god / fpv / sbs（同高 hstack 直拼，§6.2）。"""

    def __init__(self, out_dir, fps=25):
        import imageio.v2 as imageio

        os.makedirs(out_dir, exist_ok=True)
        kw = dict(fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
        self.god = imageio.get_writer(os.path.join(out_dir, "god.mp4"), **kw)
        self.fpv = imageio.get_writer(os.path.join(out_dir, "fpv.mp4"), **kw)
        self.sbs = imageio.get_writer(os.path.join(out_dir, "sbs.mp4"), **kw)
        self.frames = 0

    def append(self, god_rgb, fpv_rgb):
        if god_rgb is not None:
            self.god.append_data(god_rgb)
        if fpv_rgb is not None:
            self.fpv.append_data(fpv_rgb)
        if god_rgb is not None and fpv_rgb is not None and god_rgb.shape[0] == fpv_rgb.shape[0]:
            self.sbs.append_data(np.hstack([god_rgb, fpv_rgb]))
        self.frames += 1

    def close(self):
        for w in (self.god, self.fpv, self.sbs):
            w.close()
