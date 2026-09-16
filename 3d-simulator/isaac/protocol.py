"""host <-> isaac 双进程 IPC 协议（设计文档 §8.1）。纯 stdlib，双端共享 import。

传输：JSON-lines。host→isaac 走子进程 stdin（每行一条 JSON）。
isaac→host 走 stdout，但 kit 会往 stdout 倾倒日志——协议行一律加前缀
`PREFIX`（"@FW "），host 侧按前缀过滤；其余 stdout 行按日志转存。

host→isaac 消息（"cmd" 字段区分）：
  init            {cmd, scene:{场景JSON}, embodiment:"car|dog|humanoid",
                   localizer:"gt|topdown", profile:{render:bool, fps:int},
                   spawn:{xy:[x,y]|null, yaw_deg:float}, video_dir:str|null}
                  → {ok, meta:{spawn_zone, orthographic, calibrated, links:[...]}}
  execute_action  {cmd, action:"goto_<zone>|hold|observe_again|ask_human|
                   return_to_start|safe_stop", macro_budget:int|null}
                  → {ok, result:{action, outcome, current_zone, confidence,
                     pose_gt:[x,y,yaw_rad], pose_loc:[x,y,yaw_rad]|null,
                     geo:{n_macros, aborts:[...], sim_time_s, path_len_m}}}
                  outcome ∈ arrived|blocked|held|observed|asked|stopped|fallen
  observe         {cmd} → {ok, obs:{position:[x,y], heading_deg, current_zone,
                   observation_confidence, front_obstacle_m,
                   left_clearance_m, right_clearance_m, sim_time_s}}
                  数值离散化（QUANT_*）在 isaac 侧完成，保 LLM 输入可缓存。
  snapshot        {cmd, path:str} → {ok, path}       # 存一帧 god png
  shutdown        {cmd} → {ok}，随后进程退出（VideoSink close 在此时发生）

错误统一 {ok:false, error:"..."}；任何未知 cmd 返回错误不退进程。
"""

import json

PREFIX = "@FW "

# 观测离散化（§5.2/§5.3：缓存一致性）
QUANT_POS_M = 0.1
QUANT_DEG = 5.0
QUANT_CONF = 0.05
QUANT_DIST_M = 0.1

META_ACTIONS = ["hold", "observe_again", "ask_human", "return_to_start", "safe_stop"]
OUTCOMES = ["arrived", "blocked", "held", "observed", "asked", "stopped", "fallen"]


def dumps(msg: dict) -> str:
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def send(msg: dict, stream) -> None:
    """isaac 侧发协议行（带前缀 + flush）。"""
    stream.write(PREFIX + dumps(msg) + "\n")
    stream.flush()


def parse_line(line: str):
    """host 侧解析子进程 stdout 一行：协议行返回 dict，其余返回 None。"""
    if not line.startswith(PREFIX):
        return None
    return json.loads(line[len(PREFIX):])


def quantize(value: float, step: float) -> float:
    return round(round(value / step) * step, 6)
