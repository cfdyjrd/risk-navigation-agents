#!/usr/bin/env python3
"""Isaac 进程侧入口（设计文档 §8.1）。经 IsaacSim python.sh 运行（Py3.11）：

    ~/IsaacSim/.../python.sh isaac/isaac_server.py

协议见 isaac/protocol.py docstring：stdin 逐行 JSON 命令，stdout 回 @FW 前缀
JSON-lines（kit 日志与协议行靠前缀区分，host/ipc.py 过滤）。

命令：init / execute_action / observe / snapshot / shutdown。
- 物理只在 execute_action 内推进（事件驱动，逻辑时钟在 host）；
  init 后除 settle/warmup 不再步进；observe/snapshot 只 render 不 step。
- 任何命令异常回 {ok:false, error}，主循环不因单条消息崩溃；stdin EOF 视同 shutdown。
"""

import json
import math
import os
import sys
import traceback

# -- SimulationApp 必须最先创建（在任何 isaacsim.* / omni.* import 之前）--
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ASSETS = os.path.expanduser(
    os.environ.get("FENGWU_ASSETS", "~/isaacsim_assets/Assets/Isaac/5.1"))
carb.settings.get_settings().set("/persistent/isaac/asset_root/default", ASSETS)

import numpy as np
import yaml
from isaacsim.core.api import World

from isaac import protocol
from isaac.cameras import FpvFollowCam, VideoSink, add_marker, grab, make_god_camera
from isaac.embodiments import load_embodiment_cfg, make_embodiment
from isaac.goto_compiler import GotoCompiler
from isaac.localizer import GTLocalizer, TopDownLocalizer
from isaac.middleware import MacroExecutor
from isaac.scene_builder import build as build_scene
from isaac.sensors import RaySensor

MARKER_Z = {"car": 0.12, "dog": 0.30, "humanoid": 0.45}   # 形态表：marker 挂高
GOD_EXTENT_PAD_M = 1.5
MARKER_MIN_PX = 5           # 俯视图中 marker 最小边长（px）：batch20 实测 4.7px（108m 场景）conf=1.0、4.2px（120m）时好时坏、3.7px（137m 三层楼梯）全程失效；只在 px/m<12.5 的宽场景放大 marker

WARMUP_RENDER_FRAMES = 16



def rooms_z(scene, zone):
    return float((scene.get("rooms", {}).get(zone) or {}).get("z", 0.0))


class TrackingExecutor(MacroExecutor):
    """MacroExecutor + 单次 execute_action 的 geo 统计（宏数/路径长/物理步）。

    path_len_m 按宏间位姿差累计（§3.1 geo 语义）；物理步用 _gstep 差值，
    含 ramp/zero_hold 收尾步，比逐宏 sim_steps 求和更准。
    """

    def begin_action(self):
        self._track = {"n": 0, "path_m": 0.0, "g0": self._gstep}

    def execute(self, macro):
        p0, _ = self.emb.get_pose()
        r = super().execute(macro)
        p1, _ = self.emb.get_pose()
        t = getattr(self, "_track", None)
        if t is not None:
            t["n"] += 1
            t["path_m"] += float(np.linalg.norm(np.asarray(p1[:2]) - np.asarray(p0[:2])))
        return r

    def action_stats(self):
        t = self._track
        return {"n": t["n"], "path_m": t["path_m"], "steps": self._gstep - t["g0"]}


class IsaacServer:
    def __init__(self):
        self.world = None          # init 前所有状态为 None
        self.cfg = None            # embodiment yaml
        self.run_cfg = None
        self.emb = None
        self.topo = None
        self.meta = None           # scene_builder meta
        self.god = None
        self.fpv = None
        self.loc = None
        self.sensors = None
        self.ex = None
        self.compiler = None
        self.sink = None
        self.sim_time_s = 0.0      # 物理步计秒累计（settle + 各 action）

    # -- 分发 ----------------------------------------------------------------

    def handle(self, msg):
        cmd = msg.get("cmd")
        if cmd == "init":
            return self.cmd_init(msg)
        if self.world is None:
            return {"ok": False, "error": f"not initialized (cmd={cmd})"}
        if cmd == "execute_action":
            return self.cmd_execute(msg)
        if cmd == "observe":
            return self.cmd_observe(msg)
        if cmd == "snapshot":
            return self.cmd_snapshot(msg)
        return {"ok": False, "error": f"unknown cmd: {cmd}"}

    # -- init ----------------------------------------------------------------

    def cmd_init(self, msg):
        if self.world is not None:
            return {"ok": False, "error": "already initialized（每进程一次 init）"}
        scene = msg["scene"]
        emb_name = msg.get("embodiment", "car")
        loc_kind = msg.get("localizer", "gt")
        profile = msg.get("profile") or {}
        spawn = msg.get("spawn") or {}
        video_dir = msg.get("video_dir")

        cfg = load_embodiment_cfg(
            os.path.join(ROOT, "configs", "embodiment", f"{emb_name}.yaml"), ASSETS)
        with open(os.path.join(ROOT, "configs", "run.yaml")) as f:
            run_cfg = yaml.safe_load(f)
        with open(os.path.join(ROOT, "configs", "localizer.yaml")) as f:
            loc_cfg = yaml.safe_load(f)

        world = World(stage_units_in_meters=1.0,
                      physics_dt=cfg["physics_dt"], rendering_dt=cfg["rendering_dt"])
        topo, meta = build_scene(scene, world)

        emb = make_embodiment(cfg)
        xy = spawn.get("xy") or [float(meta["spawn_xy"][0]), float(meta["spawn_xy"][1])]
        emb.spawn(world, position=np.asarray(xy, dtype=float), z_base=float(meta.get("spawn_z", 0.0)),
                  yaw_deg=float(spawn.get("yaw_deg", 0.0) or 0.0))

        links = emb.link_names()
        mount = cfg["fpv_mount"]
        marker_link = mount.get("link") or next(
            (l for l in ("pelvis", "torso", "base", "body", "chassis") if l in links),
            links[0] if links else None)
        x0, y0, x1, y1 = meta["bbox"]
        resolution = tuple(run_cfg["video"]["resolution"])
        from isaac.cameras import god_framing
        _extent, _yaw = god_framing((x0, y0, x1, y1), resolution, GOD_EXTENT_PAD_M)

        # 宽场景（三层楼梯展平 137m）下 0.4m marker 只剩 ~3px，低于 min_area_px → 定位层全程 conf=0
        # （batch_s2/fam0119 实测）。按俯视像素密度放大 marker，保证边长 ≥ MARKER_MIN_PX。
        _px_per_m = resolution[0] / max(_extent, 1e-6)
        _min_size = MARKER_MIN_PX / _px_per_m
        if _min_size > cfg["marker"]["size_m"]:
            _k = _min_size / cfg["marker"]["size_m"]
            cfg["marker"]["size_m"] = round(_min_size, 3)
            cfg["marker"]["heading_tri_m"] = round(cfg["marker"]["heading_tri_m"] * _k, 3)
            print(f"[server] wide scene extent={_extent:.1f}m ({_px_per_m:.1f}px/m): marker -> {cfg['marker']['size_m']}m",
                  file=sys.stderr)
        if marker_link:
            add_marker(f"/World/Robot/{marker_link}",
                       size_m=cfg["marker"]["size_m"],
                       tri_m=cfg["marker"]["heading_tri_m"],
                       z_offset=MARKER_Z[emb_name])
        god, is_ortho = make_god_camera(
            world, center_xy=((x0 + x1) / 2, (y0 + y1) / 2),
            extent_m=_extent, resolution=resolution, yaw_deg=_yaw)

        # humanoid 无 rigid translation（ema_follow）：退 0.55m 近似头高偏置
        tr = mount.get("translation")
        fpv = FpvFollowCam(
            offset_fwd=cfg["raycast"]["front_edge_m"] + 0.07,
            offset_z=(tr[2] if tr else 0.55) + 0.05,
            alpha=mount.get("alpha", 1.0),
            resolution=resolution)

        # PhysX 求解器：默认 TGS 在盒碰撞体（楼板/平台/坡道）上足端摩擦失真，Spot 原地转向只剩 30%；
        # PGS 恢复到与 GroundPlane 一致（scripts/diag_turn.py 实测 44° vs 154.9°）。须在 reset 前设置。
        # 求解器按形态选：轮式 Jetbot 在 PGS 下前进会侧偏 ~25°/m、转 90° 过冲到 106°（b20_seeds 车臂
        # 28 次 21 败即此），轮式保持 TGS；足式用 PGS。可用环境变量 FW_SOLVER 覆盖做对照。
        _solver = os.environ.get("FW_SOLVER") or cfg.get("solver") or "PGS"
        world.get_physics_context().set_solver_type(_solver)
        print(f"[server] physx solver = {_solver}", file=sys.stderr)
        world.reset()
        god.initialize()
        fpv.initialize()
        world.add_physics_callback("robot", emb.on_physics_step)

        # settle（首物理步触发 emb initialize）+ 渲染 warmup 冲陈旧帧（§6.2）
        emb.set_cmd(0.0, 0.0, 0.0)
        settle = max(cfg["settle_steps"], 1)
        for _ in range(settle):
            world.step(render=False)
        for _ in range(WARMUP_RENDER_FRAMES):
            world.render()
        self.sim_time_s = settle * cfg["physics_dt"]

        if loc_kind == "gt":
            loc = GTLocalizer(emb, topo)
            calibrated = True
        else:
            loc = TopDownLocalizer(topo, loc_cfg["topdown"],
                                   cfg["marker"]["size_m"], meta["calib_points"])
            f0 = grab(god)
            calibrated = bool(f0 is not None and loc.calibrate(f0))
            _p0, _y0 = emb.get_pose()
            loc.seed(_p0[:2], _y0)          # 出生位姿种子：检测失效时不退到原点

        fps = profile.get("fps") or run_cfg["video"]["fps"]
        sink = VideoSink(video_dir, fps=fps) if video_dir else None
        render_every = (max(1, round((1 / fps) / cfg["physics_dt"]))
                        if (profile.get("render") or loc_kind == "topdown") else 0)

        # E18：脚本行人在路径点间往返（每物理步 teleport 静态盒体；射线可见 → 反射急停/障碍 abort）
        self._npc_state = []
        for n in scene.get("npcs", []):
            sc = n.get("script")
            if not sc or not sc.get("path"):
                continue
            from isaacsim.core.prims import SingleXFormPrim
            self._npc_state.append({"prim": SingleXFormPrim(f"/World/npcs/{n['id']}"), "path": [np.asarray(p, float) for p in sc["path"]],
                                    "speed": float(sc.get("speed", 0.5)), "i": 0, "dir": 1,
                                    "z": float(rooms_z(scene, n.get("zone", ""))) + 0.85})

        def npc_step(dt):
            for st in self._npc_state:
                pos, _ = st["prim"].get_world_pose()
                tgt = st["path"][st["i"]]
                d = tgt - np.asarray(pos[:2]); L = float(np.linalg.norm(d))
                step = st["speed"] * dt
                if L <= step:
                    nxt = st["i"] + st["dir"]
                    if nxt < 0 or nxt >= len(st["path"]):
                        st["dir"] *= -1; nxt = st["i"] + st["dir"]
                    st["i"] = nxt; newp = tgt
                else:
                    newp = np.asarray(pos[:2]) + d / L * step
                st["prim"].set_world_pose(np.array([newp[0], newp[1], st["z"]]))

        if self._npc_state:
            world.add_physics_callback("npcs", npc_step)

        sensors = RaySensor(emb)
        log_dir = video_dir or msg.get("log_dir")
        self._frames_fh = open(os.path.join(log_dir, "frames.jsonl"), "w") if log_dir else None
        self._frame_i = 0
        self._min_front = float("inf")

        def frame_cb():   # 同 smoke_m4：fpv 跟随 + 定位喂帧 + 视频落盘 + 逐帧日志（§4.3 min clearance / geo-AVR）
            pos, yaw = emb.get_pose()
            fpv.update(pos, yaw)
            g = grab(god)
            loc.update(g)
            if sink:
                sink.append(g, fpv.grab())
            front = sensors.front_clearance()
            self._min_front = min(self._min_front, front)
            if self._frames_fh:
                lp, ly = loc.pose()
                self._frames_fh.write(json.dumps({
                    "i": self._frame_i, "t": round(self.sim_time_s + self.ex.action_stats()["steps"] * cfg["physics_dt"], 3)
                    if hasattr(self, "ex") else 0.0,
                    "gt": [round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3), round(float(yaw), 3)],
                    "loc": [round(float(lp[0]), 3), round(float(lp[1]), 3), round(float(ly), 3)],
                    "conf": round(float(loc.confidence()), 2), "front": round(float(front), 2),
                    "yaw_src": getattr(loc, "_yaw_src", "gt"), "collisions": self._collisions}) + "\n")
                self._frame_i += 1

        # 碰撞计数：PhysX contact report（机器人任一刚体 vs 场景静态几何），失败则 collisions=None
        self._collisions = 0
        self._coll_by = {}            # 接触对象分类计数（walls / npcs / objects ...），累计
        self._contact_sub = None
        try:
            self._contact_sub = self._setup_contact_report(world)
        except Exception as exc:  # noqa: BLE001
            print(f"[isaac_server] contact report unavailable: {exc!r}", file=sys.stderr, flush=True)
            self._collisions = None
        ex = TrackingExecutor(world, emb, sensors, run_cfg,
                              render_every=render_every, frame_cb=frame_cb)
        compiler = GotoCompiler(ex, topo, zone_fn=loc.current_zone, run_cfg=run_cfg)

        self.world, self.cfg, self.run_cfg = world, cfg, run_cfg
        self.emb, self.topo, self.meta = emb, topo, meta
        self.god, self.fpv, self.loc = god, fpv, loc
        self.sensors, self.ex, self.compiler, self.sink = sensors, ex, compiler, sink

        return {"ok": True, "meta": {
            "spawn_zone": meta["spawn_zone"],
            "orthographic": bool(is_ortho),
            "calibrated": calibrated,
            "links": links,
            "marker_link": marker_link,
            "macro_budget": meta["macro_budget"],
        }}

    # -- execute_action ------------------------------------------------------

    def cmd_execute(self, msg):
        action = msg["action"]
        budget = msg.get("macro_budget") or self.meta["macro_budget"]
        self.ex.begin_action()
        aborts = []
        _coll0 = self._collisions or 0
        _by0 = dict(self._coll_by)

        if action.startswith("goto_") or action == "return_to_start":
            target = (self.meta["spawn_zone"] if action == "return_to_start"
                      else action[len("goto_"):])
            outcome, aborts = self._goto_multi(target, budget)
        elif action == "hold":
            self._run_macros(["stop"] * 3, aborts)
            outcome = "held"
        elif action == "observe_again":
            self._run_macros(["turn_left_90"] * 4, aborts)
            outcome = "observed"
        elif action == "ask_human":
            self._run_macros(["stop"], aborts)
            outcome = "asked"
        elif action == "safe_stop":
            self.emb.estop()
            self._run_macros(["stop"], aborts)
            outcome = "stopped"
        else:
            return {"ok": False, "error": f"unknown action: {action}"}

        # 摔倒任何时刻上抛（§3.4 第 3 层）
        if outcome != "fallen" and ("fell_over" in aborts or self.emb.fallen()):
            outcome = "fallen"

        stats = self.ex.action_stats()
        action_time = stats["steps"] * self.cfg["physics_dt"]
        _min_front_now, self._min_front = self._min_front, float("inf")
        self.sim_time_s += action_time
        pos, yaw = self.emb.get_pose()
        return {"ok": True, "result": {
            "action": action,
            "outcome": outcome,
            "current_zone": self.loc.current_zone(),
            "confidence": float(self.loc.confidence()),
            "pose_gt": [round(float(pos[0]), 4), round(float(pos[1]), 4),
                        round(float(yaw), 4)],
            "pose_loc": self._pose_loc(),
            "geo": {
                "n_macros": stats["n"],
                "aborts": aborts,
                "sim_time_s": round(action_time, 3),
                "path_len_m": round(stats["path_m"], 3),
                "min_front_m": (round(_min_front_now, 3) if _min_front_now != float("inf") else None),
                "collisions": (None if self._collisions is None
                               else self._collisions - _coll0),   # 本动作内新增接触数
                "collisions_total": self._collisions,
                "collisions_by": {k: v - _by0.get(k, 0) for k, v in self._coll_by.items() if v - _by0.get(k, 0) > 0},
            },
        }}

    def _setup_contact_report(self, world):
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        robot = stage.GetPrimAtPath("/World/Robot")
        n = 0
        for prim in [robot] + list(__import__("pxr").Usd.PrimRange(robot)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
                n += 1
        if n == 0:
            raise RuntimeError("no rigid body under /World/Robot")
        from pxr import PhysicsSchemaTools
        # 只计"不该碰"的几何：墙 / 物体 / NPC。楼板、楼梯、坡道是可行走支撑面，足端接触是正常步态
        # （两层场景实测每回合 ~1000 次 slab/stairs 接触，全是脚），不计入碰撞。
        scene_prefixes = ("/World/walls", "/World/objects", "/World/npcs", "/World/occluders")

        def on_contact(headers, data):
            for h in headers:
                if int(h.type) != 0:      # 0 = CONTACT_FOUND
                    continue
                a = str(PhysicsSchemaTools.intToSdfPath(h.actor0)); b = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                other = None
                if a.startswith("/World/Robot") and b.startswith(scene_prefixes):
                    other = b
                elif b.startswith("/World/Robot") and a.startswith(scene_prefixes):
                    other = a
                if other is not None:
                    self._collisions = (self._collisions or 0) + 1
                    kind = other.split("/")[2] if other.count("/") >= 2 else other   # walls/npcs/objects/...
                    self._coll_by[kind] = self._coll_by.get(kind, 0) + 1
        return get_physx_simulation_interface().subscribe_contact_report_events(on_contact)

    def _run_macros(self, macros, aborts):
        """元动作用：顺序执行宏，收集 abort，摔倒即停。"""
        for m in macros:
            r = self.ex.execute(m)
            if r["status"] == "aborted":
                aborts.append(r["reason"])
                if r["reason"] == "fell_over":
                    return

    def _goto_multi(self, target, budget):
        """goto/return_to_start：BFS 展开逐跳（Topology.zone_path），budget 共享。"""
        if target not in self.topo.rooms:
            return "blocked", [f"unknown_zone_{target}"]
        cur = self.loc.current_zone()
        if cur == target:
            return "arrived", []
        if cur is None:
            hops = [target]      # 定位丢失：直奔目标一跳（compiler 无门中点路径）
        else:
            path = self.topo.zone_path(cur, target)
            if not path:
                return "blocked", [f"no_path_{cur}_to_{target}"]
            hops = path[1:]
        aborts, used = [], 0
        for hop in hops:
            r = self.compiler.run(hop, macro_budget=max(budget - used, 0))
            used += r["n_macros"]
            aborts.extend(r["aborts"])
            if r["outcome"] != "arrived":
                return r["outcome"], aborts
        return "arrived", aborts

    def _pose_loc(self):
        if isinstance(self.loc, TopDownLocalizer) and self.loc.stale:
            return None
        p, yaw = self.loc.pose()
        return [round(float(p[0]), 4), round(float(p[1]), 4), round(float(yaw), 4)]

    # -- observe -------------------------------------------------------------

    def cmd_observe(self, msg):
        # topdown：只渲染刷新一帧喂定位（不推进物理）
        if isinstance(self.loc, TopDownLocalizer):
            self.world.render()
            self.loc.update(grab(self.god))
        p, yaw = self.loc.pose()
        front = self.sensors.front_clearance()
        left, right = self.sensors.side_clearance()
        q = protocol.quantize
        return {"ok": True, "obs": {
            "position": [q(float(p[0]), protocol.QUANT_POS_M),
                         q(float(p[1]), protocol.QUANT_POS_M)],
            "heading_deg": q(math.degrees(yaw), protocol.QUANT_DEG),
            "current_zone": self.loc.current_zone(),
            "observation_confidence": q(float(self.loc.confidence()),
                                        protocol.QUANT_CONF),
            "front_obstacle_m": q(float(front), protocol.QUANT_DIST_M),
            "left_clearance_m": q(float(left), protocol.QUANT_DIST_M),
            "right_clearance_m": q(float(right), protocol.QUANT_DIST_M),
            "sim_time_s": round(self.sim_time_s, 2),
        }}

    # -- snapshot ------------------------------------------------------------

    def cmd_snapshot(self, msg):
        path = msg["path"]
        self.world.render()
        frame = grab(self.god)
        if frame is None:
            return {"ok": False, "error": "god camera frame is empty"}
        import imageio.v2 as imageio

        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        imageio.imwrite(path, frame)
        return {"ok": True, "path": path}

    # -- shutdown ------------------------------------------------------------

    def close_sink(self):
        if self.sink is not None:
            try:
                self.sink.close()
            except Exception:
                traceback.print_exc(file=sys.stderr)
            self.sink = None


def main():
    srv = IsaacServer()
    print("[isaac_server] ready", flush=True)   # 无前缀：落 host 日志，不算协议行
    for line in sys.stdin:                       # EOF 自然退出 = shutdown
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            protocol.send({"ok": False, "error": f"bad json: {e}"}, sys.stdout)
            continue
        if not isinstance(msg, dict):
            protocol.send({"ok": False, "error": "message must be a JSON object"},
                          sys.stdout)
            continue
        if msg.get("cmd") == "shutdown":
            try:
                srv.close_sink()
            finally:
                protocol.send({"ok": True}, sys.stdout)
            break
        try:
            reply = srv.handle(msg)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            protocol.send(reply, sys.stdout)
        except Exception as e:                   # 应答不可序列化等：兜底回错误
            traceback.print_exc(file=sys.stderr)
            protocol.send({"ok": False, "error": f"send failed: {e}"}, sys.stdout)
    srv.close_sink()
    app.close()


if __name__ == "__main__":
    main()
