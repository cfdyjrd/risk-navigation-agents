"""宏执行器 + 四级看门狗 + 反射急停（设计文档 §3.2–§3.4）。Isaac 进程内使用。

MacroExecutor.execute(macro_name) 阻塞执行一条宏，返回：
    {"status": "done"|"aborted",
     "reason": None | "obstacle_ahead_<d>m" | "stuck_no_progress" | "fell_over" | "timeout",
     "executed_m": float, "executed_deg": float, "duration_s": float}

宏词表（12 个，configs/run.yaml macros）：
    forward_{0.3,0.5,1,2} / backward_0.5 / turn_{left,right}_{15,30,90} / stop

执行语义闭环于 get_world_pose() 实测（不用指令积分）；humanoid 宏起止按
cfg.ramp_s 线性升/降速（失稳高发于急加减速切换，§2.1）。

安全层（自下而上，全部独立于 LLM）：
    0 反射急停  前向射线 < reflex_stop_m → 指令立即清零（每 REFLEX_EVERY 物理步查一次）
    1 障碍 abort 前向射线 < obstacle_abort_m 且前进中 → 宏中止（door_zone 豁免见 set_door_exempt）
    2 进度 stuck 每 stuck_window_s 位移增量 < stuck_min_progress_m 连续 stuck_strikes 次
    3 摔倒      emb.fallen() → episode 级终止信号
    4 步数上限  timeout_factor x 名义时长
"""

import math
import time

import numpy as np

REFLEX_HZ = 60    # 反射急停检查频率（按时间，不随形态 physics_dt 变）
WATCH_HZ = 50     # 完成度/摔倒/障碍检查频率——转向精度上界 ≈ w_form/WATCH_HZ
                  # （car w=1.0 rad/s → 每检查间隔 ~1.1°，此前按 25 步计在 50Hz 物理下
                  #   一个间隔转 28°+，标定实测 30° 宏过冲到 58.6°，故改按时间）


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MacroExecutor:
    def __init__(self, world, emb, sensors, run_cfg, render_every=0, frame_cb=None):
        """render_every>0 时每 N 物理步 world.render()（只画不步进物理，§6.2）并回调
        frame_cb——视频开关不改变物理轨迹。指令经 emb.set_cmd 由物理回调逐步消费。"""
        self.world = world
        self.emb = emb
        self.sensors = sensors
        self.cfg = run_cfg
        self.hz = round(1 / emb.cfg["physics_dt"])
        self.reflex_every = max(1, round(self.hz / REFLEX_HZ))
        self.watch_every = max(1, round(self.hz / WATCH_HZ))
        self.door_exempt = False  # 过门豁免：仅中央射线 + 放宽阈值（§3.4 第 1 层）
        self.render_every = render_every
        self.frame_cb = frame_cb
        self._gstep = 0

    def set_door_exempt(self, on):
        self.door_exempt = on

    # -- 宏解析 --
    def _parse(self, macro):
        v = self.emb.cfg["macro_speed"]["v_form"]
        w = self.emb.cfg["macro_speed"]["w_form"]
        if macro == "stop":
            return dict(kind="stop", nominal_s=self.cfg["macros"]["stop_hold_s"], cmd=(0, 0, 0))
        kind, _, arg = macro.partition("_")
        if kind == "forward":
            m = float(arg)
            vx = 0.5 * v if m <= 0.3 else v
            return dict(kind="forward", target_m=m, nominal_s=m / vx, cmd=(vx, 0, 0))
        if kind == "backward":
            m = float(arg)
            return dict(kind="backward", target_m=m, nominal_s=m / (0.6 * v), cmd=(-0.6 * v, 0, 0))
        if kind == "turn":
            side, deg = arg.split("_")
            deg = float(deg)
            wz = (0.6 * w if deg <= 15 else w) * (1 if side == "left" else -1)
            return dict(kind="turn", target_deg=deg, nominal_s=math.radians(deg) / abs(wz), cmd=(0, 0, wz))
        raise ValueError(f"unknown macro: {macro}")

    # -- 执行 --
    def execute(self, macro):
        spec = self._parse(macro)
        ramp_s = self.emb.cfg.get("ramp_s", 0) or 0
        timeout_steps = int((spec["nominal_s"] * self.cfg["macro_timeout_factor"] + 2 * ramp_s) * self.hz)
        t0 = time.monotonic()
        start_pos, start_yaw = self.emb.get_pose()
        acc = {"prev": start_yaw, "sum": 0.0}
        watch = {"last_check_pos": start_pos[:2].copy(), "strikes": 0, "next_s": self.cfg["watchdog"]["stuck_window_s"]}
        reason = None
        steps = 0
        translating = spec["kind"] in ("forward", "backward")

        while steps < timeout_steps:
            frac = min(1.0, (steps / self.hz) / ramp_s) if ramp_s > 0 else 1.0
            cmd = tuple(c * frac for c in spec["cmd"])
            self._step(cmd)
            steps += 1

            if steps % self.reflex_every == 0 and translating and spec["kind"] == "forward":
                if self._front() < self.cfg["watchdog"]["reflex_stop_m"]:
                    reason = f"obstacle_ahead_{self._front():.2f}m"
                    break

            if steps % self.watch_every == 0:
                if self.emb.fallen():
                    reason = "fell_over"
                    break
                if spec["kind"] == "forward":
                    d = self._front()
                    thr = (
                        self.cfg["watchdog"]["door_exempt_threshold_m"]
                        if self.door_exempt
                        else self.cfg["watchdog"]["obstacle_abort_m"]
                    )
                    if d < thr:
                        reason = f"obstacle_ahead_{d:.2f}m"
                        break
                pos, yaw = self.emb.get_pose()
                acc["sum"] += _wrap(yaw - acc["prev"])
                acc["prev"] = yaw
                if spec["kind"] == "forward" and np.linalg.norm(pos[:2] - start_pos[:2]) >= spec["target_m"]:
                    break
                if spec["kind"] == "backward" and np.linalg.norm(pos[:2] - start_pos[:2]) >= spec["target_m"]:
                    break
                if spec["kind"] == "turn" and abs(math.degrees(acc["sum"])) >= spec["target_deg"]:
                    break
                # stuck 看门狗（仅平移宏）
                if translating and steps / self.hz >= watch["next_s"]:
                    prog = np.linalg.norm(pos[:2] - watch["last_check_pos"])
                    watch["last_check_pos"] = pos[:2].copy()
                    watch["next_s"] += self.cfg["watchdog"]["stuck_window_s"]
                    if prog < self.cfg["watchdog"]["stuck_min_progress_m"]:
                        watch["strikes"] += 1
                        if watch["strikes"] >= self.cfg["watchdog"]["stuck_strikes"]:
                            reason = "stuck_no_progress"
                            break
                    else:
                        watch["strikes"] = 0
        else:
            if spec["kind"] != "stop":
                reason = "timeout"

        # 收尾：ramp 降速 + 落零速站稳（§2.1/§3.2）
        if ramp_s > 0 and reason != "fell_over":
            down_steps = int(ramp_s * self.hz)
            for i in range(down_steps):
                frac = 1.0 - (i + 1) / down_steps
                self._step(tuple(c * frac for c in spec["cmd"]))
        for _ in range(self.emb.cfg["zero_hold_frames"]):
            self._step((0.0, 0.0, 0.0))

        end_pos, end_yaw = self.emb.get_pose()
        acc["sum"] += _wrap(end_yaw - acc["prev"])
        if reason is None and self.emb.fallen():
            reason = "fell_over"
        # 逐宏日志 → stderr（host 侧落到 isaac.log），排障用（二层 timeout 风暴实测无日志可查）
        import sys as _sys
        print(f"[macro] {macro:14s} {('done' if reason is None else 'ABORT:' + str(reason)):28s} "
              f"m={float(np.linalg.norm(end_pos[:2] - start_pos[:2])):.2f} deg={math.degrees(acc['sum']):6.1f} "
              f"steps={steps} pos=({end_pos[0]:.2f},{end_pos[1]:.2f},{end_pos[2]:.2f}) yaw={math.degrees(end_yaw):.0f} "
              f"front={self._front():.2f}", file=_sys.stderr, flush=True)
        return {
            "status": "done" if reason is None else "aborted",
            "reason": reason,
            "executed_m": round(float(np.linalg.norm(end_pos[:2] - start_pos[:2])), 3),
            "executed_deg": round(math.degrees(acc["sum"]), 1),
            "duration_s": round(time.monotonic() - t0, 2),
            "sim_steps": steps,
        }

    def _step(self, cmd):
        self.emb.set_cmd(*cmd)
        self.world.step(render=False)
        self._gstep += 1
        if self.render_every and self._gstep % self.render_every == 0:
            self.world.render()
            if self.frame_cb:
                self.frame_cb()

    def _front(self):
        return self.sensors.front_center() if self.door_exempt else self.sensors.front_clearance()
