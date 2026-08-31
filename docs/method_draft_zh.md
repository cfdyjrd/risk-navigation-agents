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
$\mathcal A_t$ 表示当前场景允许执行的候选动作集合。机器人状态包含机器人类型、
几何尺寸和剩余电量；环境观测包含目标距离、通道宽度、障碍物检测结果、障碍物
距离以及观测置信度。

设 $\mathcal A$ 为系统支持的全局离散动作词表：

$$
\mathcal A=\{\mathtt{move\_forward},\mathtt{turn\_left},
\mathtt{turn\_right},\mathtt{slow\_down},\mathtt{observe\_again},
\mathtt{ask\_human},\mathtt{safe\_stop}\}.
$$

当前可用动作由场景给出，并满足 $\mathcal A_t\subseteq\mathcal A$。这种区分使系统
能够根据机器人能力或环境限制动态禁用部分动作。本文假设
$\mathtt{safe\_stop}\in\mathcal A_t$，以保证系统始终存在保守回退动作。

为利用历史交互中的风险证据，系统维护分层风险记忆

$$
\mathcal M_t=(\mathcal M_t^{L0},\mathcal M_t^{L1},\mathcal M_t^{L2}),
$$

其中 $\mathcal M_t^{L0}$ 为完整情景经验，$\mathcal M_t^{L1}$ 为压缩且可追溯的
风险记忆卡，$\mathcal M_t^{L2}$ 为跨案例复用的风险规则。三个角色 Agent 基于
当前场景与角色相关证据形成审议结果

$$
\mathcal D_t=
\left(d_t^{\mathrm{adv}},d_t^{\mathrm{cri}},d_t^{\mathrm{dec}}\right),
$$

其中 $d_t^{\mathrm{adv}}$、$d_t^{\mathrm{cri}}$ 和 $d_t^{\mathrm{dec}}$ 分别表示
Task Advocate、Risk Critic 和 Safety Decision Agent 的结构化报告。Agent 的
输出仅作为语义推理证据和模型建议，不直接获得动作执行权限。

在此基础上，系统先通过硬安全约束与命中的 L2 禁止规则构造可行动作集合
$\mathcal A_t^{\mathrm{safe}}$，再定义同时满足风险与不确定性阈值的合格动作集合：

$$
\mathcal E_t=
\left\{a\in\mathcal A_t^{\mathrm{safe}}\,\middle|\,
R(a\mid x_t,\mathcal M_t)\le\tau_R,\;
Q(a\mid x_t,\mathcal M_t,\mathcal D_t)\le\tau_Q
\right\}.
$$

其中，$R(\cdot)$ 和 $Q(\cdot)$ 分别为相对风险分数与相对不确定性分数，
$\tau_R$ 和 $\tau_Q$ 为对应上限。二者用于确定性约束与动作排序，并不表示真实
事件概率。系统在合格动作中最大化任务收益：

$$
a_t^*=
\begin{cases}
\displaystyle\arg\max_{a\in\mathcal E_t}
U_{\mathrm{task}}(a\mid x_t), & \mathcal E_t\neq\varnothing,\\[6pt]
\mathtt{safe\_stop}, & \mathcal E_t=\varnothing.
\end{cases}
$$

最后，独立 Safety Guard $G$ 对优化动作进行执行前复核，得到唯一允许传递给未来
机器人执行接口的批准动作

$$
a_t^{\mathrm{app}}=G(x_t,a_t^*).
$$

因此，本文的目标不是让大语言模型直接控制机器人，而是在历史风险证据与多角色
推理的辅助下，由确定性约束优化器和独立安全守卫共同完成安全动作选择。

## 3.2 Framework Overview

所提出框架将**语义风险推理**与**确定性安全约束**分离。语言模型负责结合场景、
历史风险证据与角色职责形成结构化建议；规则冲突门控、约束优化器和 Safety Guard
则负责对建议进行确定性筛选与执行前验证。因此，任一 Agent 的生成结果都不能直接
成为机器人控制指令。框架的整体数据流可概括为：

```text
                                  ┌─ conflict ─> safe_stop ───────────────┐
scene x_t ─> L2 rule matching ─> conflict gate                           │
    │                             └─ no conflict                          │
    │                                      │                              │
    └─> role-conditioned L1 retrieval ─> Advocate + Critic                │
                                               │                          │
                                               v                          │
                                         Decision Agent                  │
                                               │                          │
                                               v                          │
                                  constrained action optimizer           │
                                               │                          │
                                               v                          v
                                         Safety Guard ─────────> approved action
                                               │
                                               └───────────────> decision trace
```

具体而言，系统按以下六个阶段运行。

1. **场景结构化。** 系统读取任务目标、机器人状态、环境观测以及当前可用动作，
   并将其编码为统一场景表示 $x_t$。该表示是记忆检索、规则匹配、Agent 推理和
   确定性验证所共享的事实基础。
2. **分角色记忆检索与规则匹配。** 系统分别为 Advocate、Critic 和 Decision
   构造受 Token 预算约束的 L1 风险证据包，并从 L2 规则库中检索满足当前场景
   条件的风险规则。角色相关的检索结果允许不同 Agent 关注任务收益、失败模式或
   决策证据，而 L2 规则提供可显式检查的行为约束。
3. **规则冲突门控。** 在调用语言模型之前，系统首先检查命中的 L2 规则是否给出
   相容的安全建议。若规则之间没有共同的安全动作，系统绕过全部 Agent 推理，
   确定性地产生 `safe_stop`，标记需要人工复核，并将该动作交给 Safety Guard。
   该旁路阻止矛盾规则继续传播到生成式决策阶段。
4. **多角色风险审议。** 在不存在规则冲突时，Task Advocate 与 Risk Critic
   分别从任务推进和风险质疑角度生成结构化报告；随后 Safety Decision Agent
   综合场景事实、双方报告、角色化 L1 证据及命中的 L2 规则，输出模型建议
   $a_t^{\mathrm{LLM}}$。两份前置报告在逻辑上相互独立，当前实现按顺序调用。
5. **确定性约束动作选择。** 优化器不直接接受模型建议作为最终动作，而是在当前
   可用动作集合上执行硬约束过滤，并综合任务收益、场景与记忆风险以及多源不确定性
   进行约束选优。若不存在满足条件的候选动作，系统回退到 `safe_stop`。
6. **安全复核与轨迹记录。** Safety Guard 根据执行时硬规则复核优化动作，并输出
   最终批准动作 $a_t^{\mathrm{app}}$。系统同时保留角色化证据包、命中规则及其匹配
   原因、三角色报告、模型建议、优化评分、Guard 结果以及缓存和 Token 用量，形成
   可审计的完整决策轨迹。

由此，正常路径为“检索—审议—约束选优—安全复核”，规则冲突路径则为“冲突检测—
保守停止—安全复核”。第 3.3--3.7 节将分别给出各模块的记忆表示、检索机制、角色
职责、优化目标与安全验证方法。

## 3.3 Hierarchical Risk Memory

为缓解完整交互历史随任务持续增长而带来的上下文膨胀，系统采用由案例到规则逐级
抽象的三层风险记忆：L0 保存可复查的完整情景经验，L1 将相关经验压缩为适合输入
Agent 的风险记忆卡，L2 则表示跨案例复用的条件化风险规则。三层记忆可写为

$$
\mathcal M_t=
\left(\mathcal M_t^{L0},\mathcal M_t^{L1},\mathcal M_t^{L2}\right),
$$

其中，信息密度和复用范围从 L0 到 L2 逐步提高，而实例细节逐步减少。为避免抽象
过程切断证据来源，每一层均保留指向下层记忆的稳定标识。

### 3.3.1 L0：完整情景经验

L0 是持久化的案例层，保存一次历史交互的场景条件、实际动作和观测结果。第 $i$
条经验表示为

$$
e_i=
\left(g_i,r_i,o_i,a_i,y_i,z_i,d_i,l_i\right),
$$

其中，$g_i$ 为任务目标，$r_i$ 为机器人状态，$o_i$ 为环境观测，$a_i$ 为实际动作，
$y_i$ 为结果及其状态，$z_i$ 为风险类型、严重度和失败原因，$d_i$ 为当时的决策，
$l_i$ 为由该案例得到的安全教训。每条记录具有唯一 `experience_id`，结果状态必须
属于 success、failure 或 aborted，风险严重度限定为 1 至 5。加载时，系统检查
必需字段、标识唯一性和取值范围，从而阻止结构不完整的案例进入检索过程。

L0 同时保留成功、失败和中止案例。失败案例提供危险动作与失败模式之间的反例证据；
成功案例则记录风险被何种观测、减速或停止策略有效缓解。L0 的目标是保留可供复查
和重新压缩的事实基础，而不是在每次推理时将全部历史直接拼接进提示词。

### 3.3.2 L1：风险记忆卡

为降低长上下文中的冗余，系统仅对检索到的 L0 候选经验动态构造 L1 风险记忆卡：

$$
m_i=C(e_i,s_i^{(k)},k),
$$

其中，$C(\cdot)$ 为确定性的结构化压缩函数，$s_i^{(k)}$ 为经验 $e_i$ 面向角色
$k\in\{\mathrm{adv},\mathrm{cri},\mathrm{dec}\}$ 的检索得分。该过程不是由语言模型
自由摘要，而是通过固定字段映射生成，因此相同输入能够得到可重复的卡片结构。

每张 L1 卡包含以下信息：

- **身份与来源：**模式版本、`memory_id`、原始 `experience_id` 及 `source`；
- **适用上下文：**机器人类型、任务目标、通行余量、障碍物状态与距离、观测置信度
  和剩余电量；
- **风险证据：**风险类型、严重度、执行动作、结果、失败原因、缓解措施与停止条件；
- **质量信息：**支持次数、成功/失败/中止统计、可靠性以及最近验证时间；
- **检索元数据：**目标角色、总分、分项得分、匹配原因、冗余惩罚和检索器版本。

因此，L1 是面向推理上下文的**有损但可追溯压缩**：它删除 L0 中与当前风险判断
关系较弱的重复表示，但保留“触发条件—动作—结果—风险—缓解措施”的因果证据链。
L1 不替代 L0；当卡片结论需要审计或重新解释时，仍可利用 `experience_id` 返回原始
案例。当前实现按请求即时生成 L1 卡，而不是将其作为独立文件长期存储。

相较于直接拼接完整历史，L1 卡片有三个作用：第一，在固定上下文预算内提供更多
有效证据；第二，将环境、动作、结果与风险显式绑定；第三，保留成功缓解案例和
失败案例，使系统既能学习“什么不能做”，也能学习“在什么条件下可以安全执行”。

### 3.3.3 L2：可复用风险规则

L2 是持久化的规则层，用于将多个 L1 案例中反复出现的条件—风险—动作关系表示为
可执行检查的条件规则：

$$
q_j=
\left(c_j,A_j^+,A_j^-,B_j,S_j,n_j,\rho_j,h_j\right),
$$

其中 $c_j$ 为触发条件，$A_j^+$ 为推荐动作集合，$A_j^-$ 为禁止动作集合，$B_j$
为缓解措施，$S_j$ 为来源风险卡 ID 集合，$n_j$ 为支持、成功和失败统计，$\rho_j$
为规则置信度，$h_j$ 为规则状态。规则状态属于 active、candidate 或 retired；
在线决策阶段只匹配 active 规则。

与自然语言经验教训不同，$c_j$ 被编码为可由程序直接判断的结构化阈值。例如，
`maximum_clearance_m` 和 `maximum_observation_confidence` 分别约束通行总余量和观测
置信度。只有全部触发条件均满足时规则才被命中，系统同时返回实际数值与阈值构成的
`match_reasons`。命中规则按照置信度及来源案例数量排序，后续冲突门控再检查其推荐
动作是否存在共同且未被禁止的安全选项。

当前实现中的 L1 卡由 L0 动态构造，L2 规则由人工归纳与验证后写入规则库。
自动聚类、自动规则归纳、长期置信度更新以及规则退役机制尚未实现，因此不作为
本章已完成贡献进行表述。

### 3.3.4 可追溯性

系统通过稳定 ID 建立由规则到案例的证据链：

$$
q_j
\xrightarrow{\mathtt{source\_memory\_ids}}
\{m_i\}_{i\in S_j}
\xrightarrow{\mathtt{experience\_id}}
\{e_i\}_{i\in S_j}.
$$

L1 卡片还保存模式版本、来源描述以及检索器版本，使后续实验能够区分记忆内容变化
与检索策略变化。三个 Agent 通过 `cited_experience_ids` 和 `cited_rule_ids` 显式
声明其使用的历史证据；程序在接受模型输出前，将这些引用与实际输入给该 Agent 的
经验和规则 ID 集合进行比对。若输出包含不存在或未提供的 ID，则报告验证失败，
机器人不得依据该输出执行动作。该机制只能证明“引用对象存在且曾被提供”，不能单独
证明模型对证据的解释必然正确；因此，引用校验仍需与后续确定性优化及 Safety Guard
共同使用。

## 3.4 Role-conditioned Risk-aware Retrieval

给定当前场景 $x_t$，检索器不是为三个角色返回同一组历史案例，而是根据各角色在
审议过程中的职责分别构造证据包。对角色
$k\in\{\mathrm{adv},\mathrm{cri},\mathrm{dec}\}$，该过程输出

$$
\mathcal B_t^{(k)}=
\operatorname{Retrieve}_{k}
\left(x_t,\mathcal M^{L0};K,B_k\right),
$$

其中，$K$ 为最多返回的 L1 卡片数量，$B_k$ 为角色提示词的记忆 Token 预算。
检索过程依次执行候选过滤、角色化精排、多样性选择和预算装包，所有得分及匹配原因
均随证据包返回。

### 3.4.1 可解释相关性评分

对当前场景 $x_t$、候选经验 $e_i$ 和角色 $k$，系统计算

$$
S_k(e_i,x_t)=w_{k,s}S_{sim}+w_{k,v}S_{sev}+w_{k,r}S_{rel}
+w_{k,t}S_{rec}+w_{k,f}S_{role}.
$$

$S_{sim}$ 表示场景相似度，$S_{sev}$ 表示风险严重度，$S_{rel}$ 表示经验可靠性，
$S_{rec}$ 表示时效性，$S_{role}$ 表示经验与 Agent 职责的适配程度。各项均归一化
到 $[0,1]$，且每个角色的权重之和为 1。

场景相似度由六类可解释匹配组成：机器人类型相同记 1.5 分，障碍物检测状态相同
记 1.0 分，任务目标相同记 1.0 分；通行余量差不超过 0.05 m 或 0.15 m 时分别记
3.0 或 1.5 分；障碍物距离差不超过 0.30 m 或 1.0 m 时分别记 2.0 或 0.75 分；
观测置信度差不超过 0.15 时记 1.5 分。原始相似度最高为 10 分，并通过

$$
S_{sim}=\min\left(\frac{S_{sim}^{raw}}{10},1\right)
$$

归一化。$S_{sim}=0$ 的经验不会进入后续候选池。风险严重度按照
$S_{sev}=z_i^{sev}/5$ 归一化。若经验显式提供质量置信度，系统直接将其截断到
$[0,1]$；若仅提供成功与失败统计，则采用带 $\operatorname{Beta}(1,1)$ 先验的
后验均值

$$
S_{rel}=\frac{n_i^{succ}+1}
{n_i^{succ}+n_i^{fail}+2}.
$$

当上述信息均不存在时，系统根据 success、failure 和 aborted 状态分别使用
0.85、0.75 和 0.65 作为工程回退值。对于带有最近验证时间的经验，时效性按照

$$
S_{rec}=\exp\left(-\frac{\Delta t_i}{180}\right)
$$

衰减，其中 $\Delta t_i$ 为距当前时间的天数；缺少有效时间戳时取 0.5。上述取值是
当前原型中的显式工程设定，并不被解释为经过统计校准的真实概率。

角色权重设置为：

| 角色 $k$ | $w_{k,s}$ | $w_{k,v}$ | $w_{k,r}$ | $w_{k,t}$ | $w_{k,f}$ |
|---|---:|---:|---:|---:|---:|
| Advocate | 0.35 | 0.05 | 0.20 | 0.05 | 0.35 |
| Critic | 0.30 | 0.25 | 0.15 | 0.05 | 0.25 |
| Decision | 0.35 | 0.20 | 0.20 | 0.10 | 0.15 |

因此，Advocate 更重视与任务推进相关的可靠成功或缓解经验；Critic 更重视高严重度
失败证据与停止案例；Decision 同时重视成功缓解和失败反例。具体地，Advocate 对
成功案例的角色适配分为 1.0、对其他案例为 0.35；Critic 对 failure 或 aborted
案例赋 1.0，对严重度不低于 4 或执行 `safe_stop` 的案例赋 0.8，其余为 0.4；
Decision 对 success 或 failure 案例赋 0.9，其余为 0.7。最终输出保留
`score_breakdown` 和自然语言 `match_reasons`。权重与角色适配值目前均为可审计的
手工基线，尚未通过训练数据学习或校准。

### 3.4.2 多样性与上下文预算

仅按 $S_k$ 取前 $K$ 个案例可能使相同风险类型和相同结果的重复案例占满上下文。
因此，系统首先建立最多 $3K$ 个候选的池，并采用贪心 MMR 式准则逐个选择：

$$
e^*=\arg\max_{e_i\in\mathcal C}
\left[
S_k(e_i,x_t)-\lambda
\max_{e_j\in\mathcal B_t^{(k)}}D(e_i,e_j)
\right],
$$

其中 $\lambda=0.15$。若两条经验风险类型相同，冗余度 $D$ 增加 0.5；结果状态相同
增加 0.25；安全教训完全相同再增加 0.25。因此 $D\in[0,1]$。该机制保留高相关
证据，同时鼓励成功案例、失败反例和不同风险模式共同进入角色证据包。

完成多样性选择后，系统将候选经验依次转换为 L1 卡片，并估算其 JSON 序列化后的
Token 成本。当前离线估计器将每个中日韩字符计为一个 Token，并将其他非空白字符
按每四个字符约一个 Token 估算。证据包必须同时满足

$$
|\mathcal B_t^{(k)}|\le K,
\qquad
\sum_{m_i\in\mathcal B_t^{(k)}}\widehat{T}(m_i)\le B_k,
$$

其中 $\widehat{T}$ 为近似 Token 数。若加入某张完整卡片会超过预算，系统跳过该卡，
而不是截断其内部证据字段。当前三种角色均设置 $K=3$、$B_k=1800$。预算值控制的
只是风险记忆部分，而不是整个 API 请求的精确 Token 数；其主要用途是在长历史条件下
提供确定、可测试的上下文上限。

### 3.4.3 L2 规则匹配与冲突处理

L2 检索不复用 L1 的加权排名，而是采用确定性条件匹配。对于每条 active 规则
$q_j$，仅当场景满足其全部触发条件 $c_j$ 时，才将其加入命中集合

$$
\mathcal Q_t=\{q_j\in\mathcal M^{L2}\mid c_j(x_t)=1\}.
$$

匹配器同时返回构成判断的实际场景数值和规则阈值，并按照规则置信度和来源记忆数量
降序排列结果。令 $A_j^+$ 和 $A_j^-$ 分别为规则 $q_j$ 的推荐与禁止动作集合，则
多规则共同允许的候选集合为

$$
\mathcal U_t=
\left(
\mathcal A_t\cap
\bigcap_{q_j\in\mathcal Q_t}A_j^+
\right)
\setminus
\bigcup_{q_j\in\mathcal Q_t}A_j^-.
$$

当没有规则命中时，系统继续正常的三角色决策流程；当 $\mathcal U_t\ne\varnothing$
时，规则被标记为可协调，命中规则及其推荐动作作为显式证据进入后续流程；当
$\mathcal U_t=\varnothing$ 时，系统将其判定为规则冲突，记录所有冲突规则 ID 和
各自建议，跳过三 Agent 的 API 调用并执行保守回退。回退顺序为 `safe_stop`、
`ask_human`、`observe_again` 中第一个当前可用的动作；在本文假设 `safe_stop` 始终
可用的动作空间下，冲突结果即为 `safe_stop`。这种设计使规则矛盾显式暴露给人工
审查，而不要求语言模型临时解释或消解相互冲突的安全约束。

## 3.5 Multi-role Risk Deliberation

多角色审议的目标不是通过增加模型数量进行多数投票，而是将任务推进、风险反驳和
证据仲裁分解为具有不同信息需求的三个职责。当前实现使用同一个底层语言模型，通过
三组独立系统提示词和三次结构化 API 调用实例化三个角色。给定场景 $x_t$、角色化
证据包 $\mathcal B_t^{(k)}$ 和命中规则 $\mathcal Q_t$，审议过程表示为

$$
\begin{aligned}
d_t^{adv}
&=F_{adv}\left(x_t,\mathcal B_t^{(adv)},\mathcal Q_t\right),\\
d_t^{cri}
&=F_{cri}\left(x_t,\mathcal B_t^{(cri)},\mathcal Q_t\right),\\
d_t^{dec}
&=F_{dec}\left(
x_t,d_t^{adv},d_t^{cri},\mathcal B_t^{(dec)},\mathcal Q_t
\right).
\end{aligned}
$$

其中，$F_{adv}$、$F_{cri}$ 和 $F_{dec}$ 分别表示受角色提示约束的语言模型调用。
Advocate 和 Critic 均直接基于原始场景独立生成报告，互不读取对方输出；Decision
在两份前置报告均通过验证后再进行仲裁。因此，该拓扑保留了任务支持观点与风险反例
之间的显式差异，同时为最终建议提供单一、可追踪的综合入口。

### 3.5.1 Task Advocate

Task Advocate 在不忽略风险的前提下寻找任务可行性。其输出
$d_t^{adv}$ 包含推荐类型、候选动作、预期任务收益、显式假设、已识别风险、缓解
措施、使用的经验与规则 ID，以及自报置信度。该角色必须从 $\mathcal A_t$ 中选择
候选动作；当信息不足或风险过高时，可以选择 `observe_again`、`ask_human` 或
`safe_stop`，而不是被要求始终推动任务执行。

Advocate 的作用是形成“在何种假设和缓解措施下仍可推进”的正向论证。其输出不是
安全证明：假设可能不成立，自报置信度也不是经过校准的成功概率，因此后续 Critic、
Decision 和确定性约束模块仍需独立审查该建议。

### 3.5.2 Risk Critic

Risk Critic 主动搜索碰撞、卡死、失控、任务失败和需要人工介入等失效模式。对于
每个风险，其报告 $d_t^{cri}$ 给出风险名称、描述、模型估计的发生可能性、1 至 5
级严重度、场景证据与缓解措施，并进一步列出缺失信息、停止条件、建议动作、证据
引用和自报置信度。风险证据只能来自输入场景或提供的历史记忆，不允许虚构传感器
观测。

Critic 不接收 $d_t^{adv}$，从而避免直接围绕 Advocate 的措辞进行迎合式修正。
同时，Critic 也不被设定为无条件拒绝任务：当风险可以通过补充观测、减速或改变
计划缓解时，应输出相应措施。其 `probability` 字段表示模型用于比较风险的主观评分，
当前尚未通过真实事故频率进行概率校准。

### 3.5.3 Safety Decision Agent

Safety Decision Agent 接收原始场景、两份前置报告、Decision 专用 L1 证据包及
命中的 L2 规则。其职责是基于证据完成仲裁，而不是对 Advocate 与 Critic 的动作
进行简单投票。提示词规定原始场景事实优先于 Agent 推断，并要求区分已观察事实、
推断结论与建议阈值；未解决的高严重度风险不能被笼统理由忽略。

输出 $d_t^{dec}$ 包含决策类别、所选动作、理由、带来源类型的支持证据、已接受风险、
未解决风险、仍需获取的信息、引用 ID 和自报置信度。支持证据的来源标签被限制为
scenario、experience、rule、advocate 或 critic；决策类别被限制为 execute、
revise_plan、observe_again、ask_human、reject 或 safe_stop。由此得到的所选动作
记为模型建议 $a_t^{\mathrm{LLM}}$，它仍需经过第 3.6 节的确定性约束选优。

### 3.5.4 结构化输出与失效关闭

每次模型调用只接受一个 JSON 对象作为有效输出。系统对三份报告执行以下检查：

1. **语法检查：**去除可选 JSON 代码围栏后进行严格解析，并要求顶层为对象；
2. **结构检查：**确认各角色所需字段、列表和嵌套对象存在且类型正确；
3. **数值与枚举检查：**检查置信度位于 $[0,1]$、Critic 风险严重度位于 1 至 5，
   并验证风险等级、Decision 决策类型和证据来源标签；
4. **动作授权检查：**确认 Advocate、Critic 和 Decision 提出的动作均属于
   当前场景的 $\mathcal A_t$；
5. **引用完整性检查：**确认 `cited_experience_ids` 和 `cited_rule_ids` 分别属于
   实际输入给该角色的经验和规则 ID 集合。

若任一调用发生网络错误、响应字段缺失、JSON 截断或解析失败，或者报告未通过上述
验证，整个三角色流程以错误状态结束，机器人不得执行该轮动作。该失效关闭策略防止
不完整的上游报告继续进入 Decision 或优化器。需要注意的是，结构验证只能保证输出
形式、动作权限和引用存在性，不能保证自然语言推理本身正确；这也是后续确定性约束
选择与 Safety Guard 仍然必要的原因。

## 3.6 Deterministic Constrained Action Selection

三个 Agent 完成审议后，系统不直接执行 $a_t^{\mathrm{LLM}}$，而是在有限动作集合
$\mathcal A_t$ 上重新求解一个确定性约束选择问题。该模块以 Decision 角色的 L1
证据包、命中的 L2 规则及三份 Agent 报告作为辅助证据，对所有可用动作使用同一组
显式函数进行评估。给定相同输入和参数，优化器始终产生相同结果。

### 3.6.1 硬约束与规则过滤

优化器首先在数值评分之前构造硬约束可行集

$$
\mathcal A_t^{safe}=
\left\{
a\in\mathcal A_t\,\middle|\,
H(x_t,a)=1
\land
a\notin\bigcup_{q_j\in\mathcal Q_t}A_j^-
\right\}.
$$

其中 $H(x_t,a)$ 表示物理硬规则。对于移动动作 move_forward、slow_down、turn_left
和 turn_right，以下任一条件成立即令 $H(x_t,a)=0$：检测到的障碍物距离不超过
0.30 m、观测置信度低于 0.50、电量不超过 10%，或通行总余量低于 0.10 m。
非移动动作不会因这些移动风险被过滤，但仍必须属于 $\mathcal A_t$，并且不能出现
在任何命中 L2 规则的禁止动作集合中。

硬约束不是可由高任务收益抵消的软惩罚。系统为每个动作记录
`emergency_obstacle_distance`、`battery_below_hard_limit` 或
`prohibited_by_l2_rule:<rule_id>` 等拒绝原因。只有保留在
$\mathcal A_t^{safe}$ 中的动作才进入后续收益、风险和不确定性计算。

### 3.6.2 任务收益

对于 $a\in\mathcal A_t^{safe}$，任务收益定义为

$$
U_{\mathrm{task}}(a)=0.65P(a)-0.20C_T(a)-0.15C_E(a),
$$

其中 $P(a)$ 为任务进展，$C_T(a)$ 为时间成本，$C_E(a)$ 为能量成本。三个属性由
以下显式动作配置表给出，均位于 $[0,1]$：

| 动作 | $P(a)$ | $C_T(a)$ | $C_E(a)$ |
|---|---:|---:|---:|
| move_forward | 1.00 | 0.10 | 0.25 |
| slow_down | 0.65 | 0.30 | 0.15 |
| turn_left / turn_right | 0.20 | 0.25 | 0.10 |
| observe_again | 0.00 | 0.25 | 0.05 |
| ask_human | 0.00 | 0.80 | 0.00 |
| safe_stop | 0.00 | 1.00 | 0.00 |

系统同时输出各属性对最终收益的贡献。表中参数是便于复核的手工工程基线，尚未使用
真实导航时间、能耗或任务完成率进行拟合。

### 3.6.3 场景与记忆联合风险

令 $\operatorname{clip}(u)=\min(1,\max(0,u))$，通行总余量为
$c_t=w_t^{corridor}-w_t^{robot}$。系统从当前场景计算四项风险因子：

$$
\begin{aligned}
R_{clr}&=\operatorname{clip}\left(\frac{0.40-c_t}{0.40}\right),\\
R_{obs}&=
\mathbb{1}[o_t^{det}]
\operatorname{clip}\left(\frac{1.50-d_t^{obs}}{1.50}\right),\\
R_{per}&=\operatorname{clip}(1-p_t^{obs}),\\
R_{bat}&=\operatorname{clip}\left(1-\frac{b_t}{100}\right).
\end{aligned}
$$

其中，$o_t^{det}$ 表示是否检测到障碍物，$d_t^{obs}$ 为障碍物距离，$p_t^{obs}$
为观测置信度，$b_t$ 为剩余电量。若相应数值缺失，除“明确未检测到障碍物”对应
$R_{obs}=0$ 外，当前工程实现对缺失风险因子使用 0.5。上下文风险为

$$
R_{ctx}=0.35R_{clr}+0.30R_{obs}+0.25R_{per}+0.10R_{bat}.
$$

考虑不同动作对环境风险的暴露程度 $\gamma_a$，得到

$$
R_{scene}(a)=\gamma_aR_{ctx}.
$$

其中 move_forward、slow_down、turn_left/right、observe_again、ask_human 和
safe_stop 的 $\gamma_a$ 分别为 1.00、0.55、0.65、0.08、0.02 和 0.00。

历史风险只使用记录了相同动作的 Decision 角色 L1 卡片。对记忆 $m_i$，其证据
权重与相似度和可靠性成正比，证据风险由严重度与结果因子共同确定：

$$
\pi_i=S_{sim}(m_i,x_t)S_{rel}(m_i),\qquad
h_i=S_{sev}(m_i)F_{out}(m_i),
$$

$$
R_{memory}(a)=\frac{\sum_{i:a_i=a}\pi_i h_i}
{\sum_{i:a_i=a}\pi_i}.
$$

结果因子 $F_{out}$ 对 failure、aborted 和 success 分别取 1.0、0.8 和 0.05，
未知状态取 0.5。成功经验仍保留一个较小的残余风险，而不会被解释为绝对安全。
场景风险与记忆风险的组合为

$$
R(a)=0.70R_{scene}(a)+0.30R_{memory}(a).
$$

当不存在同动作历史证据时，系统不虚构记忆风险，而是令 $R(a)=R_{scene}(a)$，
并将实际来源权重记录为 scene 1.0、memory 0.0。$R(a)$ 是用于动作间比较的相对风险
分数，不是事故发生概率。

### 3.6.4 多源不确定性

动作不确定性综合感知、记忆覆盖和 Agent 报告一致性：

$$
Q(a)=0.45Q_{obs}(a)+0.30Q_{mem}(a)+0.25Q_{agent}(a).
$$

观测不确定性为

$$
Q_{obs}(a)=(1-p_t^{obs})\eta_a,
$$

其中缺少观测置信度时令原始不确定性为 1。动作暴露系数 $\eta_a$ 对
move_forward、slow_down、turn_left/right、observe_again、ask_human 和 safe_stop
分别取 1.00、0.70、0.80、0.10、0.05 和 0.00。

记忆支持度及记忆不确定性定义为

$$
L_a=\sum_{i:a_i=a}S_{sim}(m_i,x_t)S_{rel}(m_i),
$$

$$
Q_{mem}(a)=
\left[1-\operatorname{clip}\left(\frac{L_a}{1.5}\right)\right]\kappa_a,
$$

其中证据目标支持度为 1.5；$\kappa_a$ 对 move_forward、slow_down、turn_left/right、
observe_again、ask_human 和 safe_stop 分别取 1.00、0.80、0.70、0.10、0.10 和
0.05。因而，动作越依赖历史支持而同动作记忆越稀缺，其记忆不确定性越高。

Agent 分歧根据报告中可解析的首选标签及其自报置信度计算。令 $v_k$ 为角色 $k$
的首选标签，$c_k\in[0,1]$ 为其置信度，则

$$
Q_{agent}(a)=
1-
\frac{\sum_k c_k\mathbb{1}[v_k=a]}
{\sum_k c_k}.
$$

若报告缺少可解析标签，则不将其计入分母；若完全没有有效报告，则令
$Q_{agent}=1$，避免将缺失信息误判为角色共识。对于不是 $\mathcal A_t$ 中具体动作
的决策类别，它不会为任何候选动作提供支持。该项衡量的是报告对动作标签的相对一致
程度，并不等价于多个独立模型之间的统计分歧。最终的 $Q(a)$ 同样是未校准的相对
不确定性分数，而非概率。

### 3.6.5 约束选优与确定性回退

当前工程基线采用 $\tau_R=0.35$、$\tau_Q=0.45$。满足以下条件的动作构成最终
候选集合：

$$
\mathcal E_t=\left\{a\in\mathcal A_t^{safe}\,\middle|\,
R(a)\le\tau_R\land Q(a)\le\tau_Q\right\}.
$$

若 $\mathcal E_t\neq\varnothing$，系统选择任务收益最大的动作；收益并列时依次
比较更低风险、更低不确定性和动作名称，即

$$
a_t^*=\operatorname*{lexmin}_{a\in\mathcal E_t}
\left(-U_{\mathrm{task}}(a),R(a),Q(a),\operatorname{name}(a)\right),
$$

从而保证相同输入产生相同输出。若 $\mathcal E_t=\varnothing$，则优先回退到
`safe_stop`，其次为 `observe_again`；若两者均不在 $\mathcal A_t$ 中，当前实现
返回空动作名称，该结果不能形成有效机器人动作。在本文假设 `safe_stop` 始终可用
时，保守回退始终具有定义。

最终输出包含每个动作的可行性、约束违反原因、收益分解、风险证据、不确定性证据、
合格动作集合、模型原始建议、优化动作、二者是否一致以及决策来源，从而使动作选择
过程可重放、可比较、可审计。阈值、权重、动作画像和暴露系数均被标记为
hand-configured、uncalibrated baseline；因此，本节给出的是可检验的确定性原型，
而不是已经完成真实机器人数据校准的最优控制器。

## 3.7 Safety Verification and Decision Traceability

确定性优化器之后设置独立于语言模型的 Safety Guard，作为动作进入执行接口之前的
最后一道检查。该模块不重新解释自然语言理由，也不依赖 Agent 的自报置信度，而是
仅依据当前结构化场景、候选动作和固定硬阈值作出批准或覆盖决定。

### 3.7.1 三阶段动作谱系

正常决策路径保留三个语义不同的动作：

| 动作 | 来源 | 含义 |
|---|---|---|
| $a_t^{\mathrm{LLM}}$ | Safety Decision Agent | 生成式模型基于审议证据给出的建议 |
| $a_t^*$ | 确定性约束优化器 | 在硬约束、风险和不确定性限制下选出的动作 |
| $a_t^{\mathrm{app}}$ | Safety Guard | 唯一可提交给未来执行适配器的批准动作 |

模型建议与优化动作的一致性记录为

$$
I_{agree}=\mathbb{1}\left[a_t^{\mathrm{LLM}}=a_t^*\right].
$$

该变量用于统计语言模型与确定性优化器的一致率，并定位优化器纠正模型建议的案例。
系统不会为了保持一致而优先选择 $a_t^{\mathrm{LLM}}$；模型建议只通过 Agent 一致性
项间接影响不确定性评分。执行决策还保存 `decision_source` 和 `optimizer_status`，
以区分正常优化、无合格动作回退以及规则冲突旁路。

### 3.7.2 Safety Guard 复核

令 $G$ 表示 Safety Guard，则正常路径的最终动作满足

$$
a_t^{\mathrm{app}}=G(x_t,a_t^*).
$$

Guard 再次检查障碍物紧急距离、观测置信度、剩余电量、通行总余量和动作授权。
其阈值与第 3.6 节的硬约束保持一致：紧急障碍物距离不超过 0.30 m、观测置信度
低于 0.50、电量不超过 10%、通行余量低于 0.10 m，或者动作不属于
$\mathcal A_t$，均会产生相应的 `hard_rule_violations`。该重复检查构成纵深防护：
即使上游优化器的输出被错误构造或未来被其他模块替换，执行入口仍保留硬规则复核。

若不存在违反项，Guard 返回 `status=approved` 并保持候选动作不变。若存在违反项，
则返回 `status=overridden` 并采用确定性回退：紧急障碍物、低电量、通行余量不足或
未授权动作触发 `safe_stop`；若唯一问题是观测置信度过低，则优先选择
`observe_again`，不可用时再选择 `safe_stop`。在本文动作空间假设下，这两种保守
动作至少有一种可用。

L2 规则冲突形成一条更短的路径。此时三个 Agent 和第 3.6 节优化器均被绕过，冲突
门控直接构造带 `decision_source=deterministic_rule_conflict_gate` 的保守决策，
随后仍由同一个 Safety Guard 复核。因而，不论动作来自正常优化还是冲突回退，
都不存在绕过最终执行检查的路径。

### 3.7.3 决策轨迹

系统将一次正常运行的决策轨迹组织为

$$
\mathcal T_t=\left(
\{\mathcal B_t^{(k)}\}_k,
\mathcal Q_t,
\rho_t^{rule},
d_t^{adv},d_t^{cri},d_t^{dec},
\Omega_t,a_t^{\mathrm{app}},\nu_t
\right),
$$

其中 $\rho_t^{rule}$ 为 L2 规则协调结果，$\Omega_t$ 为优化器对全部动作的评估记录，
$\nu_t$ 为各 Agent 的 Token 用量。具体输出包括角色化记忆卡及检索分解、命中规则
及匹配原因、规则冲突状态、三份 Agent 报告与引用 ID、每个动作的硬约束拒绝原因、
收益/风险/不确定性分解、合格动作集合、模型建议、优化动作、Guard 状态、覆盖原因
与批准动作。规则冲突路径不产生 Agent 报告和优化评分，但会保留冲突规则、保守
决策、人工复核要求及 Guard 结果。

当前命令行程序以分节 JSON 的形式输出上述轨迹，Agent 报告及其 Token 用量还可由
基于输入摘要的缓存复用。完整轨迹目前并未自动写入事务化审计数据库；实验中需要
通过输出重定向或上层运行器保存。因此，本节所称“可追溯”是指系统已生成稳定 ID、
分项评分和动作来源链，而不是声称已经具备不可篡改的生产级日志基础设施。

### 3.7.4 验证边界

Safety Guard 是确定性运行时检查器，而非对连续机器人动力学的形式化安全证明。
它基于单个决策时刻的场景快照，尚未处理观测到执行之间的状态变化、制动距离模型、
定位误差传播或控制器动态响应。当前系统也尚未接入 ROS、Nav2 或具体机器人 SDK；
`approved_action` 只是未来执行适配器允许接收的动作。因此，在真实机器人部署前，
仍需将 Guard 与实时传感器、底层碰撞检测、控制频率约束和硬件急停共同验证。

## 3.8 End-to-end Decision Algorithm

本节中的“端到端”表示从场景输入到批准动作输出的完整决策链路，而非端到端训练
的神经网络。算法 1 将第 3.3--3.7 节连接为一个完整过程，并显式表示规则冲突旁路、
Agent 报告验证、无合格动作回退和最终 Safety Guard。最终论文的 LaTeX 导言区需要
引入 `algorithm` 和 `algpseudocode` 宏包。

```tex
\begin{algorithm}[t]
\caption{Overall Risk-aware Decision Procedure}
\label{alg:overall-decision}
\begin{algorithmic}[1]
\Require Scenario $x_t$, L0 store $\mathcal{M}^{L0}$,
         L2 store $\mathcal{M}^{L2}$, limits $\tau_R,\tau_Q$
\Ensure Approved action $a_t^{\mathrm{app}}$ or $\bot$,
        and decision trace $\mathcal{T}_t$

\For{$k \in \{\mathrm{adv},\mathrm{cri},\mathrm{dec}\}$}
    \State $\mathcal{B}_t^{(k)} \gets
    \Call{RetrieveRiskMemory}{x_t,\mathcal{M}^{L0},k,K,B_k}$
\EndFor

\State $\mathcal{Q}_t \gets
\Call{MatchRiskRules}{x_t,\mathcal{M}^{L2}}$
\State $\rho_t^{rule} \gets
\Call{ResolveRuleConflicts}{\mathcal{Q}_t,\mathcal{A}_t}$

\If{$\rho_t^{rule}.\mathrm{status}=\mathrm{conflict}$}
    \State $d_t^{conf} \gets
    \Call{BuildConflictDecision}{\rho_t^{rule}}$
    \State $a_t^{\mathrm{app}} \gets
    \Call{SafetyGuard}{x_t,d_t^{conf}.\mathrm{action}}$
    \State $\mathcal{T}_t \gets
    \Call{BuildConflictTrace}{\mathcal{B}_t,\mathcal{Q}_t,
    \rho_t^{rule},d_t^{conf},a_t^{\mathrm{app}}}$
    \State \Return $a_t^{\mathrm{app}},\mathcal{T}_t$
\EndIf

\State $d_t^{\mathrm{adv}} \gets
\Call{TaskAdvocate}{x_t,\mathcal{B}_t^{(\mathrm{adv})},\mathcal{Q}_t}$
\If{$\neg\Call{ValidReport}{d_t^{\mathrm{adv}}}$}
    \State \Return $\bot,\mathrm{error}_{adv}$
\EndIf
\State $d_t^{\mathrm{cri}} \gets
\Call{RiskCritic}{x_t,\mathcal{B}_t^{(\mathrm{cri})},\mathcal{Q}_t}$
\If{$\neg\Call{ValidReport}{d_t^{\mathrm{cri}}}$}
    \State \Return $\bot,\mathrm{error}_{cri}$
\EndIf
\State $d_t^{\mathrm{dec}} \gets
\Call{SafetyDecision}{x_t,\mathcal{B}_t^{(\mathrm{dec})},\mathcal{Q}_t,
d_t^{\mathrm{adv}},d_t^{\mathrm{cri}}}$
\If{$\neg\Call{ValidReport}{d_t^{\mathrm{dec}}}$}
    \State \Return $\bot,\mathrm{error}_{dec}$
\EndIf
\State $a_t^{\mathrm{LLM}} \gets d_t^{\mathrm{dec}}.\mathrm{action}$

\State $\mathcal{A}_t^{\mathrm{safe}} \gets
\Call{FilterUnsafeActions}{x_t,\mathcal{Q}_t}$
\ForAll{$a\in\mathcal{A}_t^{\mathrm{safe}}$}
    \State $u_a\gets\Call{TaskUtility}{a}$
    \State $r_a\gets\Call{ActionRisk}{x_t,a,\mathcal{B}_t^{(\mathrm{dec})}}$
    \State $q_a\gets\Call{ActionUncertainty}{x_t,a,
    \mathcal{B}_t^{(\mathrm{dec})},d_t^{\mathrm{adv}},d_t^{\mathrm{cri}},
    d_t^{\mathrm{dec}}}$
\EndFor
\State $\mathcal{E}_t\gets\{a\in\mathcal{A}_t^{\mathrm{safe}}\mid
r_a\le\tau_R\land q_a\le\tau_Q\}$
\If{$\mathcal{E}_t\neq\emptyset$}
    \State $a_t^*\gets\Call{DeterministicBest}{\mathcal{E}_t,u,r,q}$
\Else
    \State $a_t^*\gets\Call{ConservativeFallback}{\mathcal{A}_t}$
\EndIf
\State $a_t^{\mathrm{app}}\gets
\Call{SafetyGuard}{x_t,a_t^*}$
\State $\mathcal{T}_t\gets
\Call{BuildDecisionTrace}{\mathcal{B}_t,\mathcal{Q}_t,\rho_t^{rule},
d_t^{\mathrm{adv}},d_t^{\mathrm{cri}},d_t^{\mathrm{dec}},
a_t^{\mathrm{LLM}},a_t^*,a_t^{\mathrm{app}}}$
\State \Return $a_t^{\mathrm{app}},\mathcal{T}_t$
\end{algorithmic}
\end{algorithm}
```

其中 `DeterministicBest` 按 $(-u_a,r_a,q_a,\operatorname{name}(a))$ 的字典序
选择动作，`ConservativeFallback` 按 `safe_stop`、`observe_again` 的顺序回退。
`ValidReport` 概括第 3.5.4 节的 JSON、字段、范围、动作和引用检查；API 错误或验证
失败同样返回 $\bot$，不产生可执行动作。当前命令行实现在失败时输出错误消息并以
非零状态退出，尚未构造与成功路径等量的结构化失败轨迹。

成功的 Agent 报告按照流程版本、场景、角色证据包和匹配规则的规范化 JSON 计算
SHA-256 摘要，并存入摘要对应的缓存目录。缓存文件只在一次 Agent 调用完成并通过
验证后写入，可用于重复实验和节省 API 调用；但当前读取缓存时不会再次执行报告验证
或完整性校验，因此缓存目录应被视为受信任的本地实验状态，而非安全边界。确定性
检索、规则、优化与 Guard 模块本身不依赖外部模型服务，可在离线条件下测试。

## 3.9 Implementation and Current Scope

本研究实现了一个以 Python 为主的可执行原型，用于验证前述风险记忆、角色审议、
确定性选优与安全复核能否形成闭合决策链。该实现强调模块边界和中间证据的可检查
性，而非端到端训练或直接控制真实机器人。

### 3.9.1 Software Architecture

表 1 给出方法组件与实现文件之间的对应关系。三个语言模型角色位于 `agents/`，其余
核心组件均为确定性 Python 模块。`run_three_agents.py` 负责组装一次完整决策，
但不负责执行机器人运动。

| 方法组件 | 主要实现 | 当前职责 |
|---|---|---|
| L0 经验、L1 风险卡与角色化检索 | `experience_store.py` | 加载完整经验，构造带来源信息的风险卡，并在 Token 预算下生成三类证据包 |
| L2 风险规则与冲突协调 | `risk_rule_store.py` | 校验和匹配规则，解释命中条件，并在建议不相容时触发保守回退 |
| 多角色风险审议 | `agents/advocate.py`、`agents/critic.py`、`agents/decision.py` | 构造角色提示、调用模型、校验结构化报告及其证据引用 |
| 确定性动作选择 | `decision_optimizer.py` | 执行硬约束过滤，计算收益、联合风险与多源不确定性，并确定性选优 |
| 最终安全复核 | `safety_guard.py` | 复查固定安全阈值，对不安全或未授权动作执行覆盖 |
| 流程编排与缓存 | `run_three_agents.py` | 串联检索、规则、三角色、优化器和 Guard，并输出决策轨迹与 Token 用量 |

经验、规则与场景分别使用 JSON 文件表示，使场景输入、历史证据和规则库可以独立
替换。命令行辅助程序提供 L1 检索、L2 匹配、记忆运行对比和完整三角色流程入口，
从而可以分别检查单个模块和整体链路。

### 3.9.2 Model Interface and Reproducibility

当前三个角色通过 360 智脑的 OpenAI-compatible Chat Completions 接口调用
`z-ai/glm-5.1`。API 密钥、服务地址和模型名分别由 `ZHINAO_API_KEY`、
`ZHINAO_BASE_URL` 和 `ZHINAO_MODEL` 注入；密钥不写入源码或版本库。当前默认配置
使用温度 0.2、单次请求 120 s 超时和每个角色最多 5000 个生成 Token。角色化风险
证据包另设 1800 Token 的近似预算，以控制历史记忆进入上下文的长度。模型通信仅
承担结构化风险分析，记忆匹配、规则协调、动作评分和安全复核均不依赖外部模型服务。

为支持重复运行，系统以流程版本、场景、角色证据包和命中规则的规范化内容计算
SHA-256 缓存键。完全命中缓存时可以在不加载 API 密钥的情况下复现已有 Agent
报告；缓存未命中时才创建模型客户端。该机制减少重复调用造成的成本与随机差异，
但缓存不是不可篡改的实验记录，读取时也尚未重新验证内容完整性。因此，正式实验
仍应保存代码提交、配置、场景、原始响应和完整决策轨迹。

### 3.9.3 Verification and Simulation Support

当前主工程包含 62 项离线单元与集成测试。测试覆盖 L0 经验加载与校验、L1 分层
字段和来源信息、角色条件检索、风险感知评分、多样性及 Token 预算、L2 规则加载与
匹配、规则冲突处理、Agent 证据引用、动作硬约束、任务收益、场景与记忆联合风险、
多源不确定性、确定性选优、冲突旁路以及 Safety Guard 覆盖行为。在线接口连通性由
`test_api.py` 单独检查，不计入上述离线测试；因此离线测试通过并不代表外部服务
可用，也不构成真实机器人安全性的经验证明。

仓库还包含离散时钟、拓扑地图式的 2D 仿真环境，可用于场景桥接、闭环步进、违规
记录和 HTML 回放。该仿真器不是连续几何或动力学仿真，也不模拟定位误差、制动过程
和控制延迟。当前仿真中的 LLM planner 连接的是较早的“三角色审议—Safety Guard”
路径，尚未完整接入本文主流程新增的 L2 冲突门控与确定性动作优化器。因此，现阶段
仿真结果只能验证接口与离散任务逻辑，不能被表述为对完整方法或物理部署的验证。

### 3.9.4 Current Scope and Limitations

当前实现存在以下范围边界。第一，系统尚未连接 ROS、Nav2 或具体机器人 SDK，
`approved_action` 是结构化决策输出，而非已发送给底层控制器的速度或轨迹命令。
第二，L2 规则目前由人工整理，尚未实现从新增 L0 经验中自动聚类、归纳、置信度更新
和失效规则淘汰；L1 风险卡也在检索时动态构造，而非持久化的长期记忆实体。第三，
风险权重、不确定性权重、动作画像和阈值是手工配置的未校准基线。Critic 输出中的
`probability` 是语言模型的主观比较量，不是事故频率估计，也不作为优化器中的真实
发生概率。第四，当前场景和历史经验规模较小且主要由人工设计，尚不足以证明跨环境
泛化。第五，完整轨迹目前依赖命令行输出或上层程序保存，尚未接入事务化审计存储。

这些限制界定了本文原型的主张：它实现并验证了一个可追踪、失效关闭的风险决策
软件流程，但尚未证明连续动力学条件下的机器人安全性，也尚未完成面向真实部署的
统计校准、系统集成和大规模实验。

### 3.9.5 Planned Evaluation

后续实验不涉及模型训练集，而应将场景划分为规则与参数开发集、校准集和独立测试
集，避免用测试场景反复调整阈值。主要指标包括任务成功率、碰撞或违规率、不安全
动作提议率、Guard 覆盖率、人工介入率、平均完成时间、API Token 消耗以及
$I_{agree}$ 所定义的模型—优化器一致率。还应报告规则冲突频率、保守回退频率和
在不同观测噪声水平下的性能变化。

消融实验应分别移除 L1 风险卡、L2 风险规则、角色条件检索、Critic、确定性约束
优化器与 Safety Guard，并设置无记忆、仅 L1、L1+L2 和完整系统等递增配置。参数
敏感性分析需要改变风险阈值、未知信息惩罚和角色检索权重。最终应在完整优化链已
接入仿真器后进行闭环对比，再通过 ROS/Nav2 适配器、低速受控场地和硬件急停逐级
推进到真实机器人验证。
