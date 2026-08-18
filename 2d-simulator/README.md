# 2D 拓扑仿真

`core/` ：zone 邻接图 + 离散时钟 +
机器人状态的 2D 拓扑仿真。环境步进是纯图上集合运算（无几何、无碰撞、无渲染），
单 tick 开销毫秒级；违规判定（六类授权契约约束）零 LLM、零歧义。

本目录在此基础上把仿真接入本仓库的三 Agent planner
（Task Advocate → Risk Critic → Safety Decision → Safety Guard）。

## 目录

```
core/                 仿真引擎（world / episode / contract / violations /
                      planner 合规规划搜索 / state_interface 可视化解耦接口）
scenarios/            示例场景（world + contract_gt + task + forum 四件套）
bridge.py             Episode 观测 <-> planner scenario 的双向映射
run_sim_planner.py    闭环主循环（rule / llm 两种 planner）
test_sim_bridge.py    离线测试（不调用 API）
```

## 运行

```bash
# 离线测试（不调用 API）
python3 test_sim_bridge.py

# rule planner：合规规划搜索直接求解，验证场景与 harness（不调用 API）
python3 run_sim_planner.py
python3 run_sim_planner.py --scenario scenarios/hospital_deliver_unsafe.json

# llm planner：每个决策步跑完整三 Agent 流程（需先设置 ZHINAO_API_KEY 等，
# 见上级目录 README）；每步结果按内容缓存在 .cache/，重跑不重复计费
python3 run_sim_planner.py --planner llm

# 把决策日志、逐 tick 记录和总结写入 JSON
python3 run_sim_planner.py --planner llm --out run_log.json
```

## 接入方式

每个决策步（`bridge.py`）：

1. `build_planning_scenario` 把 Episode 当前观测压成三 Agent 使用的
   scenario 结构：`task`（目标 + 进度）、`robot`（本体 / 位置 / 携带 / 护送）、
   `environment`（zone 图、可读契约视图、流式 forum、已发生违规）、
   `available_actions`（相邻 zone 的 `goto_<zone>` + `hold` / `return_to_start` /
   `observe_again` / `ask_human` / `safe_stop`）。goto 只开放相邻 zone，
   planner 每次决策只推进一跳，逐步接受风险审查。
2. 三 Agent 输出经过各自的结构化校验，最终决策再过 `safety_guard.py`
   的确定性硬规则（含"动作必须在 available_actions 内"）。
3. `to_sim_action` 把 `Safety Guard.approved_action` 映射回仿真动作：
   `goto_<zone>` → goto；`hold`/`observe_again`/`ask_human` → 原地等待一个
   tick（时间照常流逝）；`return_to_start` → 返回起点；`safe_stop` → 终止
   episode。只有 approved_action 会进入 `Episode.execute`。

Episode 结束后输出总结：任务成功与否、步数、违规明细与严重度总和、
BRS（契约保持率）。

## 场景

| 文件 | 桶 | 任务 |
|---|---|---|
| hospital_deliver_safe.json | safe-clear | deliver |
| hospital_deliver_unsafe.json | unsafe-clear | deliver（forum 中的提议不安全，存在合规替代路线） |
| hospital_escort_safe.json | safe-clear | escort |

更多场景可用 pku_icra/longsafe-l1 的生成器批量生成后放入 `scenarios/`：

```bash
cd ../../pku_icra/longsafe-l1
python -m generator.build --n 10 --seeds 1 --out /tmp/scen --seed 42
```
