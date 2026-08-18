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
run_sim_planner.py    闭环主循环（rule / llm 两种 planner，--html 生成回放）
spatial_layout.py     zone 邻接图 -> 楼层平面图的确定性布局
html_visualizer.py    自包含 HTML 回放页渲染器（只消费帧快照，不碰引擎内部）
test_sim_bridge.py    离线测试（不调用 API）
```

## 运行

### 环境准备

- Python 3.11+；离线部分（测试与 rule planner）**零第三方依赖**，
  不需要 pip install 任何东西。
- 只有 `--planner llm` 需要 API。先在当前终端会话设置（不要写进代码或 Git）：

```bash
export ZHINAO_API_KEY='你的 API Key'
export ZHINAO_BASE_URL='https://api.360.cn/v1'   # 可省略，默认即此值
export ZHINAO_MODEL='z-ai/glm-5.1'               # 可省略，默认即此值
```

- 在本目录内运行，或在上级目录用 `python3 2d-simulator/run_sim_planner.py`
  运行均可，脚本内部按自身位置解析路径。

### 第一步：离线测试（不调用 API，几秒内完成）

```bash
python3 test_sim_bridge.py
```

依次验证 6 件事：scenario 结构完整且可 JSON 序列化、动作名双向映射、
Safety Guard 能拦截不在 available_actions 里的动作、rule planner 闭环
零违规完成任务、执行动作后 scenario 随状态更新、HTML 回放页能正确内嵌
帧与决策转录（含 guard 覆盖场景）。预期最后一行输出 `全部通过`；
任何断言失败都会直接抛出。

### 第二步：rule planner（不调用 API）

用 `core/planner.py` 的合规规划搜索一次性求解零违规计划并逐步执行，
用来验证场景可解、引擎与指标工作正常——它是 oracle 上界，不经过三 Agent：

```bash
python3 run_sim_planner.py                                              # 默认 safe 配送场景
python3 run_sim_planner.py --scenario scenarios/hospital_deliver_unsafe.json
python3 run_sim_planner.py --scenario scenarios/hospital_escort_safe.json
```

预期输出形如：

```text
场景 fam0001_s0 (safe-clear), 任务 deliver, horizon 9, 起点 lobby, planner=rule
[step 0] zone=lobby action=goto_c1a
[step 1] zone=c1a action=goto_c2a
...
=== Episode Summary ===
{
  "success": true,
  "steps": 5,
  ...
  "brs_final": 1.0
}
```

若场景在契约下无解，会打印
`合规规划搜索无解：任务无法在契约下零违规完成，safe_stop。`——这是
正确行为，不是报错。

### 第三步：llm planner（三 Agent 闭环，会计费）

```bash
python3 run_sim_planner.py --planner llm
python3 run_sim_planner.py --planner llm --scenario scenarios/hospital_deliver_unsafe.json
```

每个决策步依次调用 Task Advocate → Risk Critic → Safety Decision
（**每步 3 次模型调用**，一个 episode 通常 5–15 个决策步），最终决策再过
Safety Guard 硬规则，批准动作进仿真执行。每步打印一行：

```text
[step 2] zone=c2a decision=execute model_action=goto_pharmacy2 guard=approved approved=goto_pharmacy2
```

- `decision`：Safety Decision Agent 的裁决
  （execute / revise_plan / observe_again / ask_human / reject / safe_stop）；
- `model_action` 与 `approved`：模型提议的动作 vs Guard 实际放行的动作，
  两者不同说明触发了硬规则覆盖（此时 `guard=overridden`）;
- 执行中若产生违规会打印 `!! 违规 <cid> (severity n) @ <zone>`；
- planner 给出 safe_stop / reject 时 episode 提前终止。

**计费与缓存**：每步三个角色的成功输出都按
（scenario 内容 + 检索到的经验 + 流程版本）哈希缓存在 `.cache/` 下，
中途失败重跑只会重试尚未成功的角色，同一状态不重复计费。场景或经验库
一变，缓存键随之改变，不会错误复用旧决策。想强制全新决策：`rm -rf .cache`。

任一角色最终失败（网络、额度、输出不合法）时，程序打印
`三 Agent 流程失败，机器人保持 safe_stop：<原因>` 并以退出码 1 结束，
机器人不会执行任何动作。

### 保存运行日志

```bash
python3 run_sim_planner.py --planner llm --out run_log.json
```

`run_log.json` 三段内容：

- `decisions`：每个决策步的裁决、理由、模型动作、guard 状态与
  token 用量（llm 模式）；
- `ticks`：逐 tick 记录（时间、zone、是否新进入、传感事件、违规明细），
  AVR 等指标从这里算；
- `summary`：success、步数、违规明细、severity 总和、`brs_final`、终点 zone。

### 可视化回放（--html）

任何一次运行都可以同时生成一个**自包含 HTML 回放页**——单文件、零外部
依赖、不需要起服务器，浏览器直接双击打开（rule 模式同样可用，不调 API）：

```bash
python3 run_sim_planner.py --html replay.html
python3 run_sim_planner.py --planner llm --html replay.html
open replay.html        # macOS；或直接在浏览器打开
```

实现方式：`run_sim_planner.py` 用 `core/state_interface.py` 的
`EpisodeSession` 包住引擎，每个 tick 抓一帧世界快照；`spatial_layout.py`
把 zone 邻接图确定性合成楼层平面图（走廊为横向脊柱、房间贴墙、图上相邻
则平面共墙、共墙即门、跨层边为传送门徽章）；`html_visualizer.py` 把
帧 + 决策 + 历史内嵌为页面里的 JSON，客户端 Canvas 渲染。渲染器只消费
快照字段、不 import 引擎内部，所以将来换 3D 引擎时可视化零改动。

页面上能看到什么：

- **楼层平面图**：每层一张画布，机器人按本体显示（🤖 轮式 / 🐕 四足），
  逐 tick 移动；发生违规的 tick 机器人红脉冲；物品与行人实体随取放/
  护送跟随机器人移动；
- **回放控制**：拖动进度条、播放/暂停，键盘 ←→ 逐 tick、空格播放；
  URL 深链 `#agent=<name>&tick=<n>`，可以把"第 7 tick 出事那一刻"直接
  发给别人；
- **任务进度条 + BRS 条**：任务完成度与契约保持率随 tick 实时变化，
  BRS 掉下 1.0 的位置就是边界开始失守的位置；
- **契约面板**：六类约束逐条列出，实时活性——当前 tick 正被违反的
  约束变红；
- **Forum 流**：流式帖子按可见时间点亮，能看出 planner 决策时到底
  看到了哪些指令；
- **决策历史与辩论转录**：每个决策步一条记录（提议动作 → 判决 →
  实际执行 → 消耗 tick 数 → 违规数），点击跳转到对应 tick；llm 模式下
  每条记录附三 Agent 转录——Advocate 的提议与置信度、Critic 的分项
  风险（含概率/严重度）、Decision 的裁决理由，若 Safety Guard 触发
  硬规则覆盖会多一行 guardrail 记录（`模型动作 → 实际放行动作`）；
- 明暗主题自适应系统设置。

rule 模式的回放没有辩论转录（每步只有"合规规划搜索"一行），主要用来
检查场景布局和计划路线；llm 模式的回放才是完整的"决策过程 + 执行后果"
复盘。交互式驾驶舱（手动驾驶 + agent 托管混合的 live 模式）尚未移植。

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scenario` | `scenarios/hospital_deliver_safe.json` | 场景 JSON，需含 world / contract_gt / task / forum / meta |
| `--planner` | `rule` | `rule` 离线合规搜索；`llm` 三 Agent 闭环 |
| `--out` | 不写文件 | 把 decisions / ticks / summary 写入该 JSON |
| `--html` | 不生成 | 把本次运行渲染为自包含回放页（与 `--out` 可同时使用） |

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
