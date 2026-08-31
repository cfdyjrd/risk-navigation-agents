# 3 Method

> 中文论文初稿。本章严格对应当前代码版本 `0cd8ecf`。文中的任务收益、风险、
> 不确定性权重及判定阈值均为未经数据校准的工程基线；风险和不确定性是用于动作
> 排序与约束的相对分数，而非真实事件发生概率。

## 3.1 Problem Formulation

本文研究具身机器人在环境风险、感知不确定性与有限历史经验共同存在时的安全
动作选择问题。在决策时刻 $t$，将机器人面对的场景表示为

$$
x_t=(g_t,r_t,o_t,\mathcal A_t),
$$

其中，$g_t$ 表示任务目标，$r_t$ 表示机器人自身状态，$o_t$ 表示当前环境观测，
$\mathcal A_t$ 表示当前允许执行的候选动作集合。机器人状态包含机器人类型、几何
尺寸和剩余电量；环境观测包含目标距离、通道宽度、障碍物检测结果、障碍物距离
以及观测置信度。当前原型采用如下离散动作空间：

$$
\mathcal A_t=\{\texttt{move\_forward},\texttt{turn\_left},
\texttt{turn\_right},\texttt{slow\_down},\texttt{observe\_again},
\texttt{ask\_human},\texttt{safe\_stop}\}.
$$

系统维护随历史交互积累的分层风险记忆

$$
\mathcal M_t=(\mathcal M_t^{L0},\mathcal M_t^{L1},\mathcal M_t^{L2}),
$$

其中 $\mathcal M_t^{L0}$ 为完整情景经验，$\mathcal M_t^{L1}$ 为压缩且可追溯的
风险记忆卡，$\mathcal M_t^{L2}$ 为跨案例复用的风险规则。给定场景与检索到的
风险记忆，系统需要选择既能推进任务、又满足安全约束的动作：

$$
a_t^*=\arg\max_{a\in\mathcal A_t}U_{\mathrm{task}}(a\mid x_t),
$$

满足

$$
a\in\mathcal A_t^{\mathrm{safe}},\qquad
R(a\mid x_t,\mathcal M_t)\le \tau_R,\qquad
Q(a\mid x_t,\mathcal M_t,\mathcal D_t)\le \tau_Q,
$$

其中 $\mathcal A_t^{\mathrm{safe}}$ 是通过硬安全约束和 L2 规则过滤后的动作集合，
$R(\cdot)$ 是相对风险分数，$Q(\cdot)$ 是相对不确定性分数，$\mathcal D_t$ 表示
三个角色 Agent 的审议报告，$\tau_R$ 和 $\tau_Q$ 分别为风险与不确定性上限。
若不存在同时满足全部约束的候选动作，则采用保守回退：

$$
a_t^*=\texttt{safe\_stop}.
$$

因此，本文并不让大语言模型直接控制机器人，而是将其作为结构化风险推理组件，
最终执行动作由确定性约束优化器与独立安全守卫共同决定。

## 3.2 Framework Overview

整体框架包含五个连续阶段。

1. **场景结构化。** 将任务目标、机器人状态、环境观测和可用动作编码为统一的
   JSON 场景表示 $x_t$。
2. **风险记忆检索。** 针对 Advocate、Critic 和 Decision 三种角色分别检索 L1
   风险卡，并根据当前场景匹配适用的 L2 风险规则。
3. **多角色审议。** Task Advocate 寻找风险可控的任务推进方案，Risk Critic
   搜索失败模式与停止条件，Safety Decision Agent 基于场景事实和双方证据给出
   结构化模型建议。
4. **确定性约束选优。** 优化器先排除不安全动作，再计算剩余动作的任务收益、
   相对风险和相对不确定性，并在满足阈值的动作中选择任务收益最高者。
5. **执行前复核。** 独立 Safety Guard 再次检查最终动作。只有通过复核的动作
   才能作为未来机器人接口的输入。

模型建议不直接进入执行层。系统同时保存模型建议与优化动作，从而既利用语言模型
的语义推理能力，又避免将安全性完全交给不可确定的生成过程。当命中的 L2 规则
给出互不相容的安全建议时，冲突门控在调用大模型前直接输出 `safe_stop` 并要求
人工检查，避免矛盾规则继续传播到下游决策。

## 3.3 Hierarchical Risk Memory

### 3.3.1 L0：完整情景经验

L0 记忆保存一次交互的完整事实，记为

$$
e_i=(g_i,r_i,o_i,a_i,y_i,f_i,z_i,l_i),
$$

其中 $a_i$ 为实际动作，$y_i$ 为动作结果，$f_i$ 为失败原因，$z_i$ 为风险类型和
严重度，$l_i$ 为经验教训。每条经验具有唯一 `experience_id`，并保留任务、环境、
动作与结果字段。L0 的目标是最大限度保存原始证据，而不是直接输入全部长文本。

### 3.3.2 L1：风险记忆卡

为降低长上下文中的冗余，系统将检索到的 L0 经验动态转换为 L1 风险记忆卡：

$$
m_i=C(e_i),
$$

其中 $C(\cdot)$ 为确定性的结构化压缩函数。风险卡保留任务目标、触发条件、动作、
结果、风险、失败原因、安全教训、质量信息和来源标识，同时删除与风险判断无关的
重复字段。每张卡同时保存 `memory_id`、`experience_id`、模式版本及检索得分分解，
使 Agent 引用的压缩结论能够回溯到原始经验。

相较于直接拼接完整历史，L1 卡片有三个作用：第一，在固定上下文预算内提供更多
有效证据；第二，将环境、动作、结果与风险显式绑定；第三，保留成功缓解案例和
失败案例，使系统既能学习“什么不能做”，也能学习“在什么条件下可以安全执行”。

### 3.3.3 L2：可复用风险规则

L2 将多个 L1 案例中的稳定关系表示为条件规则

$$
q_j=(c_j,A_j^+,A_j^-,B_j,S_j,\rho_j),
$$

其中 $c_j$ 为触发条件，$A_j^+$ 为推荐动作集合，$A_j^-$ 为禁止动作集合，$B_j$
为缓解措施，$S_j$ 为来源风险卡 ID 集合，$\rho_j$ 为规则置信度。规则状态属于
`active`、`candidate` 或 `retired`；当前决策阶段只匹配 `active` 规则。

当前实现中的 L1 卡由 L0 动态构造，L2 规则由人工归纳与验证后写入规则库。
自动聚类、自动规则归纳、长期置信度更新以及规则退役机制尚未实现，因此不作为
本章已完成贡献进行表述。

### 3.3.4 可追溯性

系统通过两条引用链保证证据可审计：

$$
q_j\rightarrow\{m_i\}_{i\in S_j}\rightarrow\{e_i\}_{i\in S_j}.
$$

三个 Agent 只能在 `cited_experience_ids` 和 `cited_rule_ids` 中引用输入证据包里
真实存在的 ID。程序在接受模型输出前验证这些 ID，拒绝模型虚构的经验或规则。

## 3.4 Role-conditioned Risk-aware Retrieval

### 3.4.1 可解释相关性评分

对当前场景 $x_t$、候选经验 $e_i$ 和角色 $k$，系统计算

$$
S_k(e_i,x_t)=w_{k,s}S_{sim}+w_{k,v}S_{sev}+w_{k,r}S_{rel}
+w_{k,t}S_{rec}+w_{k,f}S_{role}.
$$

$S_{sim}$ 表示场景相似度，比较机器人类型、任务目标、障碍物状态、通行余量、
障碍物距离和观测置信度；$S_{sev}$ 表示风险严重度；$S_{rel}$ 表示经验可靠性；
$S_{rec}$ 表示时效性；$S_{role}$ 表示经验与 Agent 职责的适配程度。各项均归一化
到 $[0,1]$，最终输出保存完整 `score_breakdown`。

角色权重设置为：

$$
\begin{aligned}
w_{adv}&=(0.35,0.05,0.20,0.05,0.35),\\
w_{cri}&=(0.30,0.25,0.15,0.05,0.25),\\
w_{dec}&=(0.35,0.20,0.20,0.10,0.15).
\end{aligned}
$$

因此，Advocate 更重视与任务推进相关的可靠成功或缓解经验；Critic 更重视高严重度
失败证据；Decision 获得相对均衡的证据集合。这些权重目前为可审计的手工基线，
尚未通过数据学习。

### 3.4.2 多样性与上下文预算

系统先扩大候选池，再根据相关性和候选间冗余进行多样化选择，防止多个高度重复的
案例占满上下文。随后将候选经验转换为 L1 卡片，并估算序列化后的 Token 成本。
在最多返回 $K$ 张卡片的同时满足

$$
\sum_{m_i\in\mathcal B_k}\widehat{T}(m_i)\le B_k,
$$

其中 $\mathcal B_k$ 为角色 $k$ 的证据包，$\widehat{T}$ 为近似 Token 数，$B_k$
为提示预算。当前流程取 $K=3$、$B_k=1800$。这一机制直接针对长上下文中历史
经验难以持续保留的问题：系统优先传入少量、角色相关、互补且可追溯的风险证据。

### 3.4.3 L2 规则匹配与冲突处理

对于每条活跃规则 $q_j$，系统使用确定性条件匹配器判断当前场景是否满足 $c_j$，
并返回具体的 `match_reasons`。若多条命中规则的推荐动作存在非空交集，则保留共同
安全动作；若其推荐集合没有交集，则判定

$$
\bigcap_{q_j\in\mathcal Q_t}A_j^+=\varnothing,
$$

触发规则冲突门控，跳过三 Agent 调用并输出 `safe_stop`。该处理优先保证冲突状态
可见且可审查，而不使用语言模型临时解释相互矛盾的规则。

## 3.5 Multi-role Risk Deliberation

给定场景、角色相关 L1 证据包和匹配的 L2 规则，系统依次获得三份结构化报告。

**Task Advocate。** Advocate 的目标是在承认风险的基础上提出可行方案。其报告
包含建议动作、预期收益、假设、已识别风险、缓解措施、证据引用与置信度。该角色
不得选择场景动作集合之外的动作，信息不足时允许建议重新观测、询问人工或停止。

**Risk Critic。** Critic 独立搜索碰撞、卡死、失控和任务失败等失效模式，并报告
风险描述、严重度、模型估计的发生可能性、事实证据、缓解措施、缺失信息和停止
条件。Critic 不读取 Advocate 报告，从而减少双方相互迎合。

**Safety Decision Agent。** Decision Agent 接收场景事实及两份角色报告，基于
证据完成仲裁，而不是进行简单多数投票。它优先采用原始场景事实，区分观察事实、
Agent 推断和建议阈值，并显式列出已接受风险、未解决风险和所需信息。

所有模型输出必须满足预定义 JSON 模式。系统验证必需字段、动作合法性、数值范围
和引用 ID；解析失败、输出截断、动作越权或证据引用无效时，本轮流程失败且不得
执行机器人动作。模型报告仅产生建议 $\hat a_t^{LLM}$，不具有最终执行权限。

## 3.6 Deterministic Constrained Action Selection

### 3.6.1 硬约束与规则过滤

优化器首先构造可行动作集合

$$
\mathcal A_t^{safe}=\{a\in\mathcal A_t\mid H(x_t,a)=1,
a\notin\cup_{q_j\in\mathcal Q_t}A_j^-\},
$$

其中 $H(x_t,a)$ 检查紧急障碍距离、最低观测置信度、最低电量、最小通行余量及
动作是否得到授权。对每个被排除动作，系统保存具体拒绝原因；L2 规则中的禁止
动作在数值评分之前生效。

### 3.6.2 任务收益

对可行动作 $a$，任务收益定义为

$$
U_{\mathrm{task}}(a)=0.65P(a)-0.20C_T(a)-0.15C_E(a),
$$

其中 $P(a)$ 为任务进展，$C_T(a)$ 为时间成本，$C_E(a)$ 为能量成本。三个属性由
显式动作配置表给出，均位于 $[0,1]$。系统同时输出每一项对最终收益的贡献，便于
复核和后续使用仿真数据校准。

### 3.6.3 场景与记忆联合风险

当前场景风险由通行余量、障碍物接近程度、观测不确定性和电量不足构成：

$$
R_{ctx}=0.35R_{clr}+0.30R_{obs}+0.25R_{per}+0.10R_{bat}.
$$

考虑不同动作对风险的暴露程度 $\gamma_a$，得到

$$
R_{scene}(a)=\gamma_aR_{ctx}.
$$

历史风险只使用记录了相同动作的 L1 卡片。对记忆 $m_i$，其证据权重与相似度和
可靠性成正比，证据风险由严重度与结果因子共同确定：

$$
\pi_i=S_{sim}(m_i,x_t)S_{rel}(m_i),\qquad
h_i=S_{sev}(m_i)F_{out}(m_i),
$$

$$
R_{memory}(a)=\frac{\sum_{i:a_i=a}\pi_i h_i}
{\sum_{i:a_i=a}\pi_i}.
$$

联合相对风险为

$$
R(a)=0.70R_{scene}(a)+0.30R_{memory}(a).
$$

当不存在同动作历史证据时，系统不虚构记忆风险，而是令 $R(a)=R_{scene}(a)$，
并将记忆证据数量记为零。

### 3.6.4 多源不确定性

动作不确定性综合感知、记忆覆盖和 Agent 分歧：

$$
Q(a)=0.45Q_{obs}(a)+0.30Q_{mem}(a)+0.25Q_{agent}(a).
$$

$Q_{obs}$ 由 $1-$观测置信度得到，并根据动作暴露程度调整；$Q_{mem}$ 根据同动作
历史证据的数量、相似度与可靠性计算，证据越充分则不确定性越低；$Q_{agent}$
根据三个角色对动作的置信度加权支持比例计算。若没有有效 Agent 报告，则令
$Q_{agent}=1$，避免将缺失信息误判为角色共识。

### 3.6.5 约束选优与确定性回退

当前工程基线采用 $\tau_R=0.35$、$\tau_Q=0.45$。满足以下条件的动作构成最终
候选集合：

$$
\mathcal E_t=\{a\in\mathcal A_t^{safe}\mid
R(a)\le\tau_R\land Q(a)\le\tau_Q\}.
$$

若 $\mathcal E_t\neq\varnothing$，系统选择任务收益最大的动作；收益并列时依次
比较更低风险、更低不确定性和动作名称，以保证相同输入产生相同输出。若
$\mathcal E_t=\varnothing$，则优先回退到 `safe_stop`，其次为 `observe_again`。

最终输出包含每个动作的可行性、约束违反原因、收益分解、风险证据、不确定性证据、
合格动作集合和决策来源，从而使动作选择过程可重放、可比较、可审计。

## 3.7 Safety Verification and Decision Traceability

系统保留三类动作：Safety Decision Agent 给出的
`model_recommendation`、确定性优化器输出的 `optimized_action`，以及 Safety
Guard 复核后的 `approved_action`。同时记录

$$
I_{agree}=\mathbb{1}[\hat a_t^{LLM}=a_t^*],
$$

以便实验统计模型与优化器的一致率，并分析约束模块纠正模型建议的案例。

Safety Guard 与大语言模型相互独立，对优化动作再次应用紧急距离、观测置信度、
通行余量、电量和动作授权等硬规则。若动作违反任意硬规则，守卫将其覆盖为保守
动作，并输出 `hard_rule_violations`；只有 `approved_action` 才允许交给未来的
机器人执行适配器。因此，系统形成“生成式建议—确定性选优—独立安全复核”的
三级防护链路。

## 3.8 Overall Decision Procedure

算法 1 总结了所提出框架从场景输入到安全动作输出的整体决策流程。最终论文的
LaTeX 导言区需要引入 `algorithm` 和 `algpseudocode` 宏包。

```tex
\begin{algorithm}[t]
\caption{Overall Risk-aware Decision Procedure}
\label{alg:overall-decision}
\begin{algorithmic}[1]
\Require Scenario $x_t$, L0 experience store $\mathcal{M}^{L0}$,
         L2 rule store $\mathcal{M}^{L2}$
\Ensure Approved action $a_t^{\mathrm{app}}$ and decision trace $\mathcal{T}_t$

\For{$k \in \{\mathrm{Advocate},\mathrm{Critic},\mathrm{Decision}\}$}
    \State $\mathcal{B}_k \gets
    \Call{RetrieveRiskMemory}{x_t,\mathcal{M}^{L0},k}$
\EndFor

\State $\mathcal{Q}_t \gets
\Call{MatchRiskRules}{x_t,\mathcal{M}^{L2}}$
\State $c_t \gets \Call{ResolveRuleConflicts}{\mathcal{Q}_t}$

\If{$c_t=\mathrm{conflict}$}
    \State $\hat{a}_t^{\mathrm{LLM}} \gets \texttt{safe\_stop}$
    \State $a_t^* \gets \texttt{safe\_stop}$
\Else
    \State $d_t^{\mathrm{adv}} \gets
    \Call{TaskAdvocate}{x_t,\mathcal{B}_{\mathrm{adv}},\mathcal{Q}_t}$
    \State $d_t^{\mathrm{cri}} \gets
    \Call{RiskCritic}{x_t,\mathcal{B}_{\mathrm{cri}},\mathcal{Q}_t}$
    \State $d_t^{\mathrm{dec}} \gets
    \Call{SafetyDecision}{x_t,\mathcal{B}_{\mathrm{dec}},\mathcal{Q}_t,
    d_t^{\mathrm{adv}},d_t^{\mathrm{cri}}}$
    \State $\hat{a}_t^{\mathrm{LLM}} \gets
    d_t^{\mathrm{dec}}.\mathrm{selected\_action}$

    \State $\mathcal{A}_t^{\mathrm{safe}} \gets
    \Call{FilterUnsafeActions}{x_t,\mathcal{Q}_t}$
    \ForAll{$a\in\mathcal{A}_t^{\mathrm{safe}}$}
        \State $u_a\gets\Call{TaskUtility}{a}$
        \State $r_a\gets
        \Call{ActionRisk}{x_t,a,\mathcal{B}_{\mathrm{dec}}}$
        \State $q_a\gets
        \Call{ActionUncertainty}{x_t,a,\mathcal{B}_{\mathrm{dec}},
        d_t^{\mathrm{adv}},d_t^{\mathrm{cri}},d_t^{\mathrm{dec}}}$
    \EndFor

    \State $\mathcal{E}_t\gets
    \{a\in\mathcal{A}_t^{\mathrm{safe}}\mid
    r_a\le\tau_R\land q_a\le\tau_Q\}$
    \If{$\mathcal{E}_t\neq\emptyset$}
        \State $a_t^*\gets
        \arg\max_{a\in\mathcal{E}_t}u_a$
    \Else
        \State $a_t^*\gets\texttt{safe\_stop}$
    \EndIf
\EndIf

\State $a_t^{\mathrm{app}}\gets
\Call{SafetyGuard}{x_t,a_t^*}$
\State $\mathcal{T}_t\gets
\Call{BuildDecisionTrace}{\mathcal{B},\mathcal{Q}_t,
\hat{a}_t^{\mathrm{LLM}},a_t^*,a_t^{\mathrm{app}}}$
\State \Return $a_t^{\mathrm{app}},\mathcal{T}_t$
\end{algorithmic}
\end{algorithm}
```

成功的 Agent 报告按照场景、证据包、规则和流程版本生成缓存键。缓存只复用相同
输入下已通过结构验证的报告，用于重复实验和节省 API 调用；确定性模块不依赖
外部模型服务，可在离线条件下测试。

## 3.9 Implementation and Current Scope

三个角色通过 360 智脑 OpenAI-compatible API 调用 `z-ai/glm-5.1`。模型响应采用
受约束 JSON，确定性记忆、规则、优化和安全模块均使用 Python 标准库实现。
当前版本具有 62 项离线单元与集成测试，覆盖经验加载、L1 构造、角色检索、Token
预算、L2 规则匹配与冲突、证据引用、动作过滤、收益/风险/不确定性评分、约束选优、
缓存运行和 Safety Guard。

当前原型尚未接入 ROS、Nav2 或真实机器人 SDK，因此 `approved_action` 仍是决策
输出，不会直接驱动物理设备。L2 规则尚未自动归纳，风险及不确定性参数尚未使用
仿真数据校准，Critic 报告中的 `probability` 也只是语言模型判断，不作为优化器的
真实概率输入。

后续实验应在训练、开发和测试场景分离的条件下校准参数，并报告任务成功率、碰撞
率、不安全动作率、人工介入率、平均任务时间和模型—优化器一致率。消融实验应分别
移除 L1 风险卡、L2 规则、角色条件检索、约束优化器与 Safety Guard，以验证各模块
对安全性、任务效率和可解释性的独立贡献。
