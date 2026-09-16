# 架构级对照 baseline（可实现为本 benchmark 的对照臂，全部有公开代码）

按设计轴分组：每个 baseline 回答"我们架构的哪个组件有价值"。标注移植成本 = 接入
run_sim_planner 作为新 planner 臂的工作量。

## 轴 1：审议结构（对照我们的三 Agent 分工）

| 架构 | 出处 / 代码 | 机制 | 移植成本 | 对照价值 |
|---|---|---|---|---|
| Multi-Agent Debate (Du et al.) | ICML'24 · github.com/composable-models/llm_multiagent_debate | N 个同构 agent 互评多轮 + 多数票 | 低（每步 N×R 次调用） | "同构辩论"vs 我们的"角色分工"：分工是否优于人多 |
| Self-Refine | NeurIPS'23 · github.com/madaan/self-refine | 单模型自我反馈迭代 | 低 | 单 agent + 自批评：不加人数能否达到 Critic 效果 |
| MADRA | arXiv 2511.21460（无代码） | 辩论 + 批判评估者打分 | 中（按论文复现） | 最接近的已发表架构；无代码正好由我们实现并对比 |
| SAFER | arXiv 2503.15707（无代码） | 规划 LLM + 安全 LLM + LLM judge | 中 | LLM judge vs 我们的程序化判定 |

## 轴 2：护栏（对照我们的 Safety Guard / 拓扑 Guard）

| 架构 | 出处 / 代码 | 机制 | 移植成本 | 对照价值 |
|---|---|---|---|---|
| RoboGuard | arXiv'25 · github.com/KumarRobotics/RoboGuard | 落地安全规则 → 时序逻辑合成两段护栏 | 中（LTL 规则需映射到契约） | 形式化护栏 vs 审议：与 single+guard 臂直接可比 |
| GuardAgent | ICML'25 · github.com/guardagent/code | 护栏 agent 把安全要求编译成可执行检查代码 | 中 | "生成检查代码"vs 我们手写确定性 Guard |
| LlamaGuard 类内容护栏 | Meta（开源权重） | 输入输出内容分类 | 低 | 证明内容级护栏对授权语义无效（预期全放行） |

## 轴 3：不确定时提问（对照我们的 ask_human 机制）

| 架构 | 出处 / 代码 | 机制 | 移植成本 | 对照价值 |
|---|---|---|---|---|
| KnowNo | CoRL'23 oral · robot-help.github.io | 保形预测校准何时问 | 中（需 top-k 候选打分接口） | 校准式提问 vs 三 Agent 的审议式提问：ambiguous-L3 主战场 |
| CLARA | RA-L'24 · github.com/jeongeun980906/CLARA-SaGC | 清晰/歧义/不可行分类 + 对话澄清 | 低-中 | 恰好对应 ambiguous L1/L2/L3 三级 |

## 轴 4：经验记忆（对照我们的 L1/L2 分层记忆，learning_spec 用）

| 架构 | 出处 / 代码 | 机制 | 移植成本 | 对照价值 |
|---|---|---|---|---|
| Reflexion | NeurIPS'23 · github.com/noahshinn/reflexion | 失败的言语反思进下轮 prompt | 低（≈single+raw 臂） | 记忆通道基线 |
| ExpeL | AAAI'24 · github.com/LeapLabTHU/ExpeL | 轨迹→insight 提炼 + 检索 | 中（≈我们 L1 卡） | 结构化记忆对照 |
| Voyager | 2305.16291 · github.com/MineDojo/Voyager | 技能库归纳复用 | 高 | L2 规则归纳的思想源头，叙述性对比即可 |

## 推荐的论文对照臂组合（按优先级）

1. single / single+guard / 3A（已有）
2. + Multi-Agent Debate（同构 N=3, R=2）——回答"分工 vs 人多"，成本低
3. + Self-Refine——回答"一个人自我批评够不够"
4. + CLARA 式两段（分类→澄清）——ambiguous 桶的方法级对照
5. + RoboGuard 式形式化护栏（把契约编译为检查器）——护栏轴的强对照
6. 学习实验期：+ Reflexion（=single+raw）与 ExpeL（=L1 检索）
