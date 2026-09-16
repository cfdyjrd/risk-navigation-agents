"""三 Agent LLM planner 桥：3d-simulator obs -> scenario -> Advocate/Critic/Decision
-> Safety Guard -> approved 动作名。

编排与 risk-navigation-agents/2d-simulator/run_sim_planner.py 的 plan_step_llm
同构（去掉经验库/规则库层：本桥第一版不接 ExperienceStore / RiskRuleStore，
retrieved_experiences / matched_rules 传空表）。

步级缓存：sha256(scenario 规范化序列化 + 模型名 + 版本) 为目录名，
每个角色一个 json（advocate/critic/decision），重跑不重复计费——
与 2D 版 step_cache_dir/cached_call 同策略，缓存目录在 3d-simulator/out/llm_cache/。

三 Agent 原文通过 self.last_transcript 暴露，runner 负责落盘。

运行在系统 Python 3.12，纯 stdlib + 2D 仓库源码；不 import isaac 模块。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

try:                                     # runner 以 host.planner_bridge 导入时
    from host.obs_schema import (DEFAULT_QUESTIONS, REPLY_DELAY, build_scenario,
                                 origin_author, reply_text)
except ImportError:                      # 直接从 host/ 目录运行时
    _HOST_DIR = str(Path(__file__).resolve().parent)
    if _HOST_DIR not in sys.path:
        sys.path.append(_HOST_DIR)
    from obs_schema import (DEFAULT_QUESTIONS, REPLY_DELAY, build_scenario,  # noqa: F401
                            origin_author, reply_text)

FENGWU_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = FENGWU_ROOT / "out" / "llm_cache"
PIPELINE_VERSION = "3d-simulator-planner-v2"

# 2D 仓库两级目录：repo 根（agents/ llm_client safety_guard），2d-simulator（core/ 等，
# judge 侧也会用到；这里一并挂上保持两模块 sys.path 状态一致）。
RNA_ROOT = str(Path(__file__).resolve().parents[2])  # 3d-simulator/host -> 仓库根
RNA_SIM = f"{RNA_ROOT}/2d-simulator"
for _p in (RNA_ROOT, RNA_SIM):
    if _p not in sys.path:
        sys.path.append(_p)

from agents.advocate import analyze as run_advocate     # noqa: E402
from agents.critic import analyze as run_critic         # noqa: E402
from agents.decision import decide as run_decision      # noqa: E402
from llm_client import LLMError, ZhinaoClient           # noqa: E402
from safety_guard import apply_safety_guard             # noqa: E402


class LLMPlanner:
    """每步：build_scenario -> advocate -> critic -> decision -> safety guard。

    decide() 返回 approved 动作名（必属于 available_actions，越界回退 hold）。
    """

    def __init__(self, scene: dict, horizon: int | None = None):
        if not os.getenv("ZHINAO_API_KEY"):
            raise RuntimeError(
                "缺少环境变量 ZHINAO_API_KEY（360 智脑 API key）。设置方法：\n"
                "  export ZHINAO_API_KEY=sk-...\n"
                "可选：export ZHINAO_BASE_URL=https://api.360.cn/v1 、"
                "export ZHINAO_MODEL=z-ai/glm-5.1 、ZHINAO_MAX_TOKENS 调思维链额度。"
            )
        self.scene = scene
        self.horizon = horizon
        self.client = ZhinaoClient()
        self.tick = 0                       # 逻辑时钟：每次 decide +1
        self.history: list[dict] = []       # 供 build_scenario 算进度/已发生违规
        # 2D 引擎 E4 问答同构：配额 1，回复 REPLY_DELAY tick 后进入论坛
        self.questions_remaining = DEFAULT_QUESTIONS
        self.pending_replies: list[dict] = []
        self.dynamic_posts: list[dict] = []
        self.last_transcript: dict[str, Any] | None = None   # 三 Agent 原文，runner 落盘

    # -------------------------------------------------------------- 缓存
    def _cache_dir(self, scenario: dict) -> Path:
        cache_input = {
            "pipeline_version": PIPELINE_VERSION,
            "model": os.getenv("ZHINAO_MODEL"),   # 不同模型的输出不可互相复用
            "scenario": scenario,
        }
        encoded = json.dumps(cache_input, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return CACHE_ROOT / hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _cached_call(cache_dir: Path, role: str, fn, retries: int = 2):
        """成功输出按内容缓存；结构校验失败(LLMError)整体重试（与 2D 版同策略）。"""
        path = cache_dir / f"{role}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return saved["report"], saved.get("usage", {})
        last_error: LLMError | None = None
        for _ in range(retries + 1):
            try:
                report, usage = fn()
                break
            except LLMError as exc:
                last_error = exc
        else:
            raise last_error
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"report": report, "usage": usage},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report, usage

    # -------------------------------------------------------------- 决策
    def decide(self, obs: dict, available_actions: list[str]) -> str:
        """一步决策。obs 为 isaac observe 应答的 obs 字段；available_actions 为
        runner 权威给定的可用动作集（覆盖 scenario 自推的邻接集）。"""
        available = list(available_actions)
        due = [p for p in self.pending_replies if p["step"] <= self.tick]
        self.pending_replies = [p for p in self.pending_replies if p["step"] > self.tick]
        self.dynamic_posts.extend(due)
        scenario = build_scenario(self.scene, obs, self.tick,
                                  self.history, horizon=self.horizon,
                                  dynamic_posts=self.dynamic_posts,
                                  questions_remaining=self.questions_remaining)
        scenario["available_actions"] = available   # runner 为准
        cache_dir = self._cache_dir(scenario)

        usage: dict[str, Any] = {}
        error: str | None = None
        advocate_report = critic_report = final_decision = guard = None
        try:
            advocate_report, usage["advocate"] = self._cached_call(
                cache_dir, "advocate",
                lambda: run_advocate(scenario, self.client, [], []))
            critic_report, usage["critic"] = self._cached_call(
                cache_dir, "critic",
                lambda: run_critic(scenario, self.client, [], []))
            final_decision, usage["decision"] = self._cached_call(
                cache_dir, "decision",
                lambda: run_decision(scenario, advocate_report, critic_report,
                                     self.client, [], []))
            guard = apply_safety_guard(scenario, final_decision)
            approved = (guard.get("approved_action") or {}).get("name") or "safe_stop"
        except LLMError as exc:
            # 整链失败：保守降级 safe_stop（可用集里没有再回退 hold）
            error = str(exc)
            approved = "safe_stop" if "safe_stop" in available else "hold"

        if approved not in available:               # 越界回退 hold
            approved = "hold" if "hold" in available else (available[0] if available else "hold")
        asked = None
        if approved == "ask_human" and self.questions_remaining > 0:
            self.questions_remaining -= 1
            asked = {"step": self.tick + 1 + REPLY_DELAY, "author": origin_author(self.scene),
                     "text": reply_text(self.scene, self.tick), "role": "reply"}
            self.pending_replies.append(asked)

        self.last_transcript = {
            "tick": self.tick,
            "cache_dir": str(cache_dir),
            "scenario_sha256": cache_dir.name,
            "scenario": scenario,
            "advocate": advocate_report,
            "critic": critic_report,
            "decision": final_decision,
            "guard": guard,
            "usage": usage,
            "approved": approved,
            "asked": asked,
            "error": error,
        }
        self.history.append({
            "tick": self.tick,
            "current_zone": obs.get("current_zone"),
            "action": approved,
        })
        self.tick += 1
        return approved

    # -------------------------------------------------------------- 记录回灌
    def feed_result(self, result: dict, violations: list[dict] | None = None) -> None:
        """可选：runner 执行动作后把 result（与 judge 判出的违规）回灌进历史，
        使下一步 scenario 的 progress / violations_so_far 反映真实执行结果。"""
        if self.history:
            self.history[-1]["outcome"] = (result or {}).get("outcome")
            self.history[-1]["current_zone"] = (result or {}).get(
                "current_zone", self.history[-1].get("current_zone"))
            if violations:
                self.history[-1]["violations"] = list(violations)
