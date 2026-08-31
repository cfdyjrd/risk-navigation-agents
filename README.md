# Risk Navigation Agents

面向室内移动机器人的风险感知三 Agent 决策原型。系统使用 360 智脑 API
运行 Task Advocate、Risk Critic 和 Safety Decision Agent。模型报告进入确定性
约束优化器，最终动作再由 Safety Guard 复核。

项目当前重点实现两项机制：

1. 分层风险记忆：从历史经历生成 L1 Risk Memory Card，再归纳为可复用的
   L2 Risk Rule。
2. 风险感知检索：根据场景、风险严重度、经验可靠性、时效性和 Agent 角色，
   为三个角色构造不同的证据包。

## 系统流程

```text
历史经历（L0）
    ↓ 压缩并保留来源
Risk Memory Card（L1）
    ↓ 归纳
Risk Rule（L2）
    ↓ 场景匹配与冲突检查
Task Advocate + Risk Critic
    ↓
Safety Decision Agent
    ↓
确定性约束优化器
    ↓
Safety Guard
    ↓
机器人执行层（尚未接入）
```

L2规则目前作为可追溯证据进入三个Agent。若多条L2规则给出互不兼容的建议，
确定性前置闸门会跳过大模型调用，生成 `safe_stop` 并要求人工复核。

## 三个Agent的职责

- Task Advocate：寻找完成任务的可行方案，优先使用成功经验和有效缓解措施。
- Risk Critic：主动寻找碰撞、卡死和任务失败等风险，优先使用失败和高严重度经验。
- Safety Decision Agent：综合场景事实、正反报告、L1卡片和L2规则作出最终裁决。

Safety Decision Agent 的结果是模型建议，不直接进入机器人执行层。确定性约束
优化器根据任务收益、相对风险和相对不确定性重新选择可行动作，并同时保留
`model_recommendation`、`optimized_action` 和 `agrees_with_model` 供审计。

三个Agent均须通过 `cited_experience_ids` 和 `cited_rule_ids` 显式声明实际使用的
历史经验与规则。代码会拒绝模型编造的ID。

## 环境配置

项目仅使用Python标准库，不需要安装第三方SDK。建议使用Python 3.10或更高版本。

复制环境变量示例并编辑本地 `.env`：

```bash
cp .env.example .env
nano .env
```

本地文件内容应为：

```text
ZHINAO_API_KEY=你的真实API_Key
ZHINAO_BASE_URL=https://api.360.cn/v1
ZHINAO_MODEL=z-ai/glm-5.1
```

每次打开新终端后加载：

```bash
set -a
source .env
set +a
```

检查Key是否加载（不会显示Key内容）：

```bash
python3 -c 'import os; print("API Key 已加载" if os.getenv("ZHINAO_API_KEY") else "API Key 未加载")'
```

`.env` 已被 `.gitignore` 忽略。不要把真实API Key写入代码、README、
`.env.example`或Git。

## API连通性测试

```bash
python3 test_api.py
```

预期输出包含“API连接成功”。该命令会调用360 API并产生Token用量。

## 运行完整三Agent流程

```bash
python3 run_three_agents.py 2>&1 | tee results/latest_three_agents.txt
```

程序执行以下步骤：

1. 为三个角色分别检索L1证据包，每个证据包受1800个估算Token的预算约束。
2. 匹配当前场景适用的L2规则并检查冲突。
3. 无规则冲突时运行三个Agent。
4. 校验每份JSON输出、动作集合、置信度及引用ID。
5. 使用确定性约束优化器逐动作计算任务收益、相对风险和相对不确定性。
6. 排除违反硬规则、L2禁止规则或数值约束的动作，在剩余动作中最大化任务收益。
7. 使用Safety Guard对优化后的动作进行执行前复核。

成功的Agent报告缓存在 `.cache/`。缓存键包含场景、证据包、规则和流程版本，
相同输入再次运行时可以避免重复调用和重复计费。

## 确定性约束动作选择

`decision_optimizer.py` 将动作选择显式实现为：

```text
maximize task_utility(action)
subject to:
  action通过硬安全与L2禁止规则过滤
  relative_risk(action) <= 0.35
  relative_uncertainty(action) <= 0.45
```

任务收益由任务推进、时间成本和能量成本组成。相对风险综合当前场景因素与
同动作历史记忆；相对不确定性综合观测置信度、记忆证据覆盖度和三个Agent的
置信度加权分歧。若没有动作满足约束，系统保守回退到 `safe_stop`。

风险、不确定性权重及阈值当前均标记为
`hand_configured_uncalibrated_baseline`。它们是可解释的工程基线，而非已学习或
经过概率校准的参数。因此输出明确使用 `relative_*_score_not_probability`，
不能解释为碰撞概率或统计置信区间。

## 分角色风险感知检索

历史经历保存在 `experiences/risk_experiences.json`。检索评分包含：

- 场景相似度
- 风险严重度
- 经验可靠性
- 经验时效性
- Agent角色适配度
- 重复证据惩罚

所有评分分解保存在 `score_breakdown` 中。Advocate、Critic和Decision会获得排序
不同的证据包。

离线查看检索结果：

```bash
python3 run_retrieval.py
```

## L1 Risk Memory Card

`experience_store.py` 将完整历史经历压缩为L1卡片，主要字段包括：

- `memory_id` 与原始轨迹来源
- 场景和触发条件
- 危险类型与严重度
- 动作、结果和失败原因
- 缓解措施与停止条件
- 统计、可靠性和检索评分
- 指向L2规则的链接

完整原始经历仍保存在经验库中，避免在每个Agent请求中反复发送长文本。

## L2 Risk Rule

正式规则库位于 `experiences/risk_rules.json`。当前窄通道规则由一条失败经历和
一条成功经历归纳而来：通行总余量不超过0.2米且观测置信度不超过0.8时，
禁止直接前进，应先重新观测。

查看正式规则的匹配结果：

```bash
python3 run_rule_retrieval.py scenarios/corridor_obstacle.json
```

验证规则不会错误匹配宽走廊场景：

```bash
python3 run_rule_retrieval.py scenarios/wide_corridor_clear.json
```

预期 `matched_rule_count` 为0。

## L2规则冲突实验

`experiences/risk_rules_conflict_fixture.json` 是专用测试夹具，不属于正式规则库。
其中故意加入一条过度宽泛的规则，用于制造 `observe_again` 与 `slow_down` 的冲突：

```bash
python3 run_rule_retrieval.py scenarios/corridor_obstacle.json \
  --rules experiences/risk_rules_conflict_fixture.json
```

预期结果：

```json
{
  "status": "conflict_detected",
  "selected_action": "safe_stop",
  "requires_human_review": true
}
```

冲突时系统不会按置信度随意选择某条学习规则，而是采用保守回退。

## A/B对比

已有两份相同场景下的运行记录：

- `results/baseline_l1_memory.txt`：仅使用L1。
- `results/l2_rule_integration.txt`：使用L1与L2。

生成离线对比报告：

```bash
python3 compare_memory_runs.py
```

报告写入 `results/l1_vs_l2_comparison.md`。当前单次实验中，两组均选择
`observe_again`；L2组增加了显式规则证据和决策可追溯性，但Token增加931
（约6.36%）。单次运行不能证明L2提高了准确率，后续需要多场景、多次重复实验。

## 自动化测试

运行全部离线测试：

```bash
python3 -m unittest discover -v
```

当前共有62项测试，覆盖：

- L1经验加载、压缩、检索、Token预算和角色差异
- L2规则加载、字段校验、命中和不命中
- Agent规则引用与虚构ID拦截
- L2规则冲突检测和确定性 `safe_stop` 闸门
- Safety Guard紧急障碍、低置信度、低电量、过窄通道和非法动作
- 候选动作硬约束与L2禁止规则过滤
- 任务收益、场景风险、历史记忆风险和三类不确定性计算
- 约束优化、无可行动作回退和结果可重复性
- 三Agent报告到优化器再到Safety Guard的离线集成

这些测试不调用API，不产生模型费用。

## 主要文件

```text
agents/                            三个Agent的提示词、调用与输出校验
experiences/risk_experiences.json L0历史经历
experiences/risk_rules.json       正式L2规则库
experience_store.py               L1卡片与风险感知检索
risk_rule_store.py                L2规则加载、匹配和冲突处理
run_three_agents.py               完整主流程与确定性冲突闸门
decision_optimizer.py             安全过滤、评分与确定性约束优化
safety_guard.py                   模型之外的硬安全规则
run_retrieval.py                  L1检索演示
run_rule_retrieval.py             L2规则匹配演示
compare_memory_runs.py            L1与L1+L2离线对比
scenarios/                        实验场景
results/                          实验输出和对比报告
test_*.py                         离线自动化测试
docs/method_draft_zh.md           与当前代码对应的Method中文初稿
```

## 当前边界

- 尚未连接ROS、Nav2或真实机器人SDK。
- L2规则目前手工归纳并校验，尚未实现从大量L1卡片自动聚类和自动更新。
- 学习规则不能替代Safety Guard的硬安全边界。
- 当前实验规模较小，需要增加场景数量、重复次数和量化指标。
- 任务收益、相对风险、相对不确定性的权重与阈值尚未通过仿真开发集校准，
  当前分数不能解释为真实事件概率。
