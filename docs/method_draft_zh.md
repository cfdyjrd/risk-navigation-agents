# 3 Method（中文初稿）

> 本文档与当前代码实现对应。任务收益、风险、不确定性权重和阈值仍是未经
> 数据校准的工程基线；在完成仿真实验前，不应将相对分数表述为事件概率。

## 3.1 Problem Formulation

本文研究具身机器人在环境风险与感知不确定性共存条件下的安全动作选择问题。
在决策时刻 $t$，场景表示为

$$
x_t=(g_t,r_t,o_t,\mathcal A_t),
$$

其中 $g_t$ 为任务目标，$r_t$ 为机器人状态，$o_t$ 为环境观测，
$\mathcal A_t$ 为候选动作集合。机器人状态包括机器人类型、尺寸和剩余电量；
环境观测包括目标距离、通道宽度、障碍物状态、障碍物距离和观测置信度。

当前原型的动作集合为

$$
\mathcal A_t=\{\texttt{move\_forward},\texttt{turn\_left},
\texttt{turn\_right},\texttt{slow\_down},\texttt{observe\_again},
\texttt{ask\_human},\texttt{safe\_stop}\}.
$$

为利用长期交互经验，系统维护分层风险记忆

$$
\mathcal M_t=(\mathcal M_t^{L0},\mathcal M_t^{L1},\mathcal M_t^{L2}),
$$

其中 L0 保存完整交互经历，L1 保存压缩且可追溯的风险记忆卡，L2 保存从多个
案例中归纳的风险规则。

系统的动作选择目标为

$$
a_t^*=\arg\max_{a\in\mathcal A_t}U_{\mathrm{task}}(a\mid x_t)
$$

满足

$$
a\in\mathcal A_t^{safe},\qquad
R(a\mid x_t,\mathcal M_t)\le R_{\max},\qquad
U_{\mathrm{uncertainty}}(a)\le U_{\max}.
$$

若不存在满足全部约束的动作，系统执行保守回退：

$$
a_t^*=\texttt{safe\_stop}.
$$

## 3.2 Framework Overview

框架首先根据当前场景分别为 Task Advocate、Risk Critic 和 Safety Decision
Agent 检索角色相关的 L1 风险记忆，并匹配适用的 L2 风险规则。Advocate 侧重
任务推进与成功缓解经验，Critic 侧重失败案例和高严重度风险，Decision Agent
综合场景事实及正反证据生成模型建议。

模型建议不直接进入执行层。确定性约束优化器对全部候选动作进行安全过滤和
数值评估，输出最终优化动作。该动作最后由独立 Safety Guard 进行执行前复核。
若 L2 规则在 Agent 调用前已经发生不可调和冲突，系统跳过大模型并直接生成
`safe_stop`，要求人工检查规则。

## 3.3 Hierarchical Risk Memory

L0 经验记录环境、机器人、任务、动作、动作结果、失败原因、风险严重度和安全
教训。L1 风险记忆卡压缩 L0 内容，同时保留 `memory_id`、`experience_id` 和
来源轨迹，使输入大模型的证据可追溯。L2 规则包含触发条件、推荐动作、禁止动作、
缓解措施、来源记忆和规则置信度。

当前实现从经验库动态构造 L1 卡片；L2 规则经过人工归纳和验证。自动聚类、规则
归纳、长期置信度更新和规则退役尚未实现。

## 3.4 Role-conditioned Risk-aware Retrieval

角色 $k$ 对记忆 $m_i$ 的检索分数为

$$
S_k(m_i,x_t)=\alpha_kS_{sim}+\beta_kS_{sev}+\eta_kS_{rel}
+\lambda_kS_{rec}+\phi_kS_{role}-\omega_kS_{red},
$$

分别对应场景相似度、风险严重度、可靠性、时效性、角色适配度和冗余惩罚。
不同角色使用不同权重，并在证据数量和估算 Token 预算内选择记忆卡。输出保留
完整 `score_breakdown`，三个 Agent 必须通过 `cited_experience_ids` 和
`cited_rule_ids` 声明实际使用的证据。

## 3.5 Deterministic Constrained Action Selection

### 3.5.1 安全动作集合

首先构造

$$
\mathcal A_t^{safe}=\{a\in\mathcal A_t\mid H(x_t,a)=1,
a\notin A_{L2}^{-}\},
$$

其中 $H$ 包含紧急障碍距离、最低观测置信度、最低电量和最低通行余量等硬约束，
$A_{L2}^{-}$ 为命中规则禁止的动作集合。

### 3.5.2 任务收益

当前工程基线定义为

$$
U_{\mathrm{task}}(a)=0.65P_{progress}(a)-0.20C_{time}(a)
-0.15C_{energy}(a).
$$

动作属性来自显式配置表，所有贡献项随结果输出，以支持复核和后续校准。

### 3.5.3 相对风险

动作风险由当前场景和历史记忆共同构成：

$$
R(a)=0.70R_{scene}(a)+0.30R_{memory}(a).
$$

$R_{scene}$ 综合通行余量、障碍物接近程度、观测不确定性和电量不足程度，再乘以
动作风险暴露系数。$R_{memory}$ 只使用记录相同动作的 L1 卡片，并根据相似度、
可靠性、严重度和历史结果进行加权。没有匹配证据时，系统仅采用场景风险，且
明确记录历史证据数量为零。

### 3.5.4 相对不确定性

$$
U_{\mathrm{uncertainty}}(a)=0.45U_{obs}(a)+0.30U_{memory}(a)
+0.25U_{agent}(a).
$$

$U_{obs}$ 来自观测置信度并按动作暴露程度调整；$U_{memory}$ 根据同动作历史证据
的数量、相似度和可靠性计算；$U_{agent}$ 根据三个角色对该动作的置信度加权支持
程度计算。缺少有效 Agent 报告时，系统将该部分设为高不确定性，而不是将缺失
误判为共识。

### 3.5.5 约束求解

当前基线取 $R_{max}=0.35$、$U_{max}=0.45$。系统先排除违反硬约束、L2规则和
数值阈值的动作，再在剩余动作中确定性选择任务收益最高者。并列时依次比较风险、
不确定性和动作名称，以保证相同输入得到相同输出。

## 3.6 Safety Verification and Traceability

系统同时保存 Safety Decision Agent 的 `model_recommendation`、优化器的
`optimized_action` 以及 `agrees_with_model`。这使实验能够统计模型与优化器的
一致率，并分析优化器纠正模型建议的情况。优化动作进入 Safety Guard 后再次
接受硬规则检查，只有 `approved_action` 才允许传给未来的机器人执行接口。

当前系统尚未连接 ROS、Nav2 或真实机器人 SDK，因此所有结果均为决策输出，
不会直接驱动物理设备。

## 3.7 Implementation and Current Scope

三个角色通过 360 智脑 API 调用 `z-ai/glm-5.1`，采用受约束的 JSON 输出。成功
报告按场景、记忆、规则和流程版本缓存。确定性模块仅使用 Python 标准库。项目
目前具有 62 项离线单元与集成测试，不调用 API，也不产生模型费用。

目前权重和阈值尚未经过仿真开发集校准，风险与不确定性输出是相对分数而非概率。
后续实验应在训练/开发/测试场景分离的条件下调参，报告碰撞率、任务成功率、
不安全动作率、人工介入率和决策一致性，并开展 L1、L2、角色检索、约束优化器和
Safety Guard 的消融实验。
