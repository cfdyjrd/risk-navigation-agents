"""3d-simulator 主 runner：读场景 -> 拉起 isaac 子进程 -> 逐 tick observe/decide/execute
-> judge 记录 -> trace/decisions/summary 落盘（设计文档 §8）。

示例：
    python host/runner.py --scene scenes/E00_smoke.json --embodiment dog \
        --localizer topdown --planner scripted --actions "goto_B" \
        --render --out out/E00 --horizon 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

FENGWU_ROOT = Path(__file__).resolve().parents[1]
if str(FENGWU_ROOT) not in sys.path:
    sys.path.insert(0, str(FENGWU_ROOT))

from host.ipc import IsaacCrashed, IsaacProc, protocol  # noqa: E402
from host.obs_schema import adjacency_from_doors  # noqa: E402


# -- planner ----------------------------------------------------------------

class ScriptedPlanner:
    """按给定动作序列出牌；耗尽后 safe_stop。"""

    name = "scripted"

    def __init__(self, actions: list[str]):
        self._actions = list(actions)
        self._i = 0

    def decide(self, obs: dict, available_actions: list[str]) -> str:
        if self._i < len(self._actions):
            action = self._actions[self._i]
            self._i += 1
            return action
        return "safe_stop"


def make_planner(args, scene: dict):
    if args.planner == "oracle":
        # 2D core.planner.find_compliant_plan 的零违规计划（scene/compile2d.py 写入场景）
        actions = list(scene.get("oracle_actions") or [])
        if not actions:
            print(f"[runner] 警告: 场景无 oracle_actions（oracle_status={scene.get('oracle_status')}），首 tick 即 safe_stop",
                  file=sys.stderr)
        return ScriptedPlanner(actions)
    if args.planner == "scripted":
        actions = [a.strip() for a in (args.actions or "").split(",") if a.strip()]
        if not actions:
            print("[runner] 警告: --planner scripted 未给 --actions，首 tick 即 safe_stop",
                  file=sys.stderr)
        return ScriptedPlanner(actions)

    # llm
    if not os.environ.get("ZHINAO_API_KEY"):
        sys.exit("[runner] 错误: --planner llm 需要环境变量 ZHINAO_API_KEY，"
                 "当前未设置。export ZHINAO_API_KEY=... 后重试。")
    try:
        from host.planner_bridge import LLMPlanner
    except Exception as e:
        sys.exit(f"[runner] 错误: 无法导入 host.planner_bridge.LLMPlanner: {e!r}")

    llm = LLMPlanner(scene, horizon=args.horizon)

    class _LLMAdapter:
        name = "llm"

        def decide(self, obs: dict, available_actions: list[str]) -> str:
            return llm.decide(obs, available_actions)

        @property
        def last_transcript(self):
            return llm.last_transcript

        def feed_result(self, result: dict, violations=None) -> None:
            llm.feed_result(result, violations)

    return _LLMAdapter()


# -- judge ------------------------------------------------------------------

class _FallbackJudge:
    """host/judge.py 不可用时的最小兜底：只做 goal 到达 / horizon 判定。"""

    def __init__(self, scene: dict, horizon: int):
        self._horizon = horizon
        goal = scene.get("goal") or {}
        self._target = goal.get("target") if goal.get("type") == "goto" else None
        self._ticks = 0
        self._success = False
        self._last_outcome = None

    def record(self, tick: int, obs: dict, action: str, result: dict) -> None:
        self._ticks = tick
        self._last_outcome = result.get("outcome")
        if (self._target is not None
                and result.get("outcome") == "arrived"
                and result.get("current_zone") == self._target):
            self._success = True

    def done(self) -> bool:
        return self._success or self._ticks >= self._horizon

    def finalize(self) -> dict:
        return {
            "judge": "fallback",
            "success": self._success,
            "ticks": self._ticks,
            "horizon": self._horizon,
            "goal_target": self._target,
            "last_outcome": self._last_outcome,
        }


def make_judge(scene: dict, horizon: int):
    try:
        from host.judge import Judge
    except Exception as e:
        print(f"[runner] host/judge.py 不可用（{e!r}），使用内置 fallback judge",
              file=sys.stderr)
        return _FallbackJudge(scene, horizon)
    return Judge(scene, horizon)


# -- main -------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="3d-simulator host runner")
    p.add_argument("--scene", required=True, help="场景 JSON 路径")
    p.add_argument("--embodiment", default="dog", choices=["car", "dog", "humanoid"])
    p.add_argument("--localizer", default="topdown", choices=["topdown", "gt"])
    p.add_argument("--planner", default="scripted", choices=["scripted", "oracle", "llm"])
    p.add_argument("--seed", type=int, default=0, help="出生位姿扰动种子：0=无扰动；>0 → xy ±0.2m、yaw ±10°（设计 §4.5 pass^k）")
    p.add_argument("--actions", default="",
                   help="scripted planner 的逗号分隔动作序列，如 'goto_B,hold'")
    p.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out", required=True, help="输出目录（trace/summary/日志/视频）")
    p.add_argument("--horizon", type=int, default=20, help="最大逻辑 tick 数")
    return p.parse_args(argv)


def available_actions_for(scene: dict, current_zone: str | None = None) -> list[str]:
    """与 2D bridge.available_actions 同构：goto 只开放相邻 zone，每次决策只推进一跳（judge 逐跳可见）。"""
    if current_zone is None:
        zones = sorted((scene.get("rooms") or {}).keys())
    else:
        zones = adjacency_from_doors(scene).get(current_zone, [])
    return [f"goto_{z}" for z in zones] + list(protocol.META_ACTIONS)


def main(argv=None) -> int:
    args = parse_args(argv)

    scene_path = Path(args.scene)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    planner = make_planner(args, scene)
    judge = make_judge(scene, args.horizon)
    spawn = scene.get("spawn") or {}
    macro_budget = scene.get("macro_budget")

    trace_fh = open(out / "trace.jsonl", "w", encoding="utf-8")
    decisions_fh = open(out / "decisions.jsonl", "w", encoding="utf-8")
    transcript_fh = (open(out / "llm_transcript.jsonl", "w", encoding="utf-8")
                     if planner.name == "llm" else None)

    def jline(fh, obj):
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()

    t_start = time.monotonic()
    summary: dict = {
        "scene": scene.get("id", scene_path.stem),
        "scene_path": str(scene_path.resolve()),   # host/report.py 回读 source_2d/annotations
        "embodiment": args.embodiment,
        "localizer": args.localizer,
        "planner": args.planner,
        "horizon": args.horizon,
    }
    exit_code = 0
    ticks_run = 0

    print(f"[runner] scene={summary['scene']} embodiment={args.embodiment} "
          f"localizer={args.localizer} planner={args.planner} render={args.render}")

    try:
        with IsaacProc(log_path=out / "isaac.log") as proc:
            import random as _rnd
            _rng = _rnd.Random(args.seed)
            _xy_jit = [_rng.uniform(-0.2, 0.2), _rng.uniform(-0.2, 0.2)] if args.seed else None
            _yaw_jit = _rng.uniform(-10.0, 10.0) if args.seed else 0.0
            summary["seed"] = args.seed
            _spawn_xy = None
            if _xy_jit:
                _r = scene["rooms"][scene["spawn"]["zone"]]["aabb"]
                _spawn_xy = [(_r[0] + _r[2]) / 2 + _xy_jit[0], (_r[1] + _r[3]) / 2 + _xy_jit[1]]
            init_msg = {
                "cmd": "init",
                "scene": scene,
                "embodiment": args.embodiment,
                "localizer": args.localizer,
                "profile": {"render": bool(args.render), "fps": 25},
                "spawn": {"xy": _spawn_xy,
                          "yaw_deg": float(spawn.get("yaw_deg", 0.0)) + _yaw_jit},
        "log_dir": str(out.resolve()),
                "video_dir": str(out.resolve()) if args.render else None,
            }
            init_resp = proc.request(init_msg, timeout_s=600.0)  # 首次含 kit 启动+资产加载
            if not init_resp.get("ok"):
                raise RuntimeError(f"init 失败: {init_resp.get('error')}")
            summary["init_meta"] = init_resp.get("meta", {})
            print(f"[runner] init ok, meta={summary['init_meta']}")

            for tick in range(1, args.horizon + 1):
                obs_resp = proc.request({"cmd": "observe"})
                if not obs_resp.get("ok"):
                    raise RuntimeError(f"observe 失败: {obs_resp.get('error')}")
                obs = obs_resp.get("obs", {})

                action = planner.decide(obs, available_actions_for(scene, obs.get("current_zone")))
                jline(decisions_fh, {"tick": tick, "planner": planner.name,
                                     "obs": obs, "action": action})
                transcript = getattr(planner, "last_transcript", None)
                if transcript_fh is not None and transcript is not None:
                    jline(transcript_fh, transcript)

                exec_resp = proc.request({"cmd": "execute_action",
                                          "action": action,
                                          "macro_budget": macro_budget})
                if not exec_resp.get("ok"):
                    raise RuntimeError(
                        f"execute_action({action}) 失败: {exec_resp.get('error')}")
                result = exec_resp.get("result", {})

                rec = judge.record(tick, obs, action, result) or {}
                if hasattr(planner, "feed_result"):
                    planner.feed_result(result, rec.get("violations"))
                jline(trace_fh, {"tick": tick, "obs": obs,
                                 "action": action, "result": result})

                outcome = result.get("outcome")
                zone = result.get("current_zone")
                ticks_run = tick
                print(f"[tick {tick:>2}/{args.horizon}] action={action:<16} "
                      f"outcome={outcome:<8} zone={zone}")

                if judge.done() or outcome in ("stopped", "fallen"):
                    break

            summary["ticks_run"] = ticks_run
            summary.update({"judge_summary": judge.finalize()})
    except IsaacCrashed as e:
        print(f"[runner] IsaacCrashed:\n{e}", file=sys.stderr)
        summary["crashed"] = True
        summary["error"] = str(e).splitlines()[0]
        summary["ticks_run"] = ticks_run
        try:
            summary["judge_summary"] = judge.finalize()
        except Exception:
            pass
        exit_code = 1
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"[runner] 中止: {e!r}", file=sys.stderr)
        summary["error"] = str(e)
        summary["ticks_run"] = ticks_run
        try:
            summary["judge_summary"] = judge.finalize()
        except Exception:
            pass
        exit_code = 1
    finally:
        trace_fh.close()
        decisions_fh.close()
        if transcript_fh is not None:
            transcript_fh.close()
        summary["wall_time_s"] = round(time.monotonic() - t_start, 2)
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    print(f"[runner] summary -> {out / 'summary.json'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
