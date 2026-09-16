# 可对比论文清单（按与本项目的对比维度分组，优先有公开代码）

本项目定位：三 Agent 审议（Advocate/Critic/Decision）+ 确定性护栏 + 分层风险记忆，
在带授权契约、后果引擎与问答机制的 2D 拓扑仿真上做阶梯评测（drift/ambiguous 三级）。

## 1. 具身/Agent 安全 benchmark（我们的 benchmark 与谁比）

| 论文 | 出处 | 代码 | 与本项目的对比点 |
|---|---|---|---|
| SafeAgentBench | arXiv 2412.13178 | ✅ github.com/shengyin1224/SafeAgentBench | 750 个危险任务、AI2-THOR 环境；测"拒绝危险任务"，无授权契约/后果引擎/难度分级。我们的 drift-L1≈其显式危险，L3 授权孪生是其没有的维度 |
| Agent-SafetyBench | arXiv 2412.14470 | ✅ github.com/thu-coai/Agent-SafetyBench | 通用 agent 安全 349 环境；无具身、无逐步闭环。可对比安全失败分类法 |
| MobileSafetyBench | arXiv 2410.17520 | ✅ github.com/jylee425/mobilesafetybench | 安卓设备控制安全；有"日常任务 vs 风险任务"对照，理念接近我们的 safe/unsafe 桶 |
| ToolEmu | ICLR'24 | ✅ github.com/ryoungj/ToolEmu | LM 模拟沙箱里测工具 agent 风险；我们用确定性引擎替代 LM 模拟器，判定零歧义 |
| AgentDojo | NeurIPS'24 | ✅ github.com/ethz-spylab/agentdojo | 提示注入攻防动态环境；其"注入指令"≈我们的 drift 帖，但无权威推理（spoof/legit 孪生）与物理后果 |
| InjecAgent | ACL'24 Findings | ✅ github.com/uiuc-kang-lab/InjecAgent | 间接提示注入 1054 例；同上，静态、无具身 |

## 2. 多 Agent 审议 / 安全规划架构（我们的三 Agent 与谁比）

| 论文 | 出处 | 代码 | 对比点 |
|---|---|---|---|
| MADRA | arXiv 2511.21460 | ❌ 未见公开 | 多 Agent 辩论评估指令风险 + 批判评估者，与我们最接近；其重点是降误拒，我们量化了"误拒 vs 被骗"的双向权衡与配对显著性 |
| SAFER | arXiv 2503.15707 | ❌ 未见公开 | 多 LLM 安全规划 + LLM-as-a-Judge；我们用程序化判定替代 LLM judge，指标零主观 |
| Safe Planner | arXiv（2025） | 未确认 | 安全感知任务规划；可作 baseline 叙述 |
| MobileSafetyBench 的 SCoT | 同上 | ✅ | 安全提示法（单 agent + 提示）——正好是我们 single 臂的强化版对照 |

## 3. 歧义澄清 / 主动提问（我们的 ambiguous 桶与谁比）

| 论文 | 出处 | 代码 | 对比点 |
|---|---|---|---|
| KnowNo | CoRL'23 (oral) | ✅ robot-help.github.io（含代码） | 保形预测校准"何时该问"；我们的 ambiguous-L3 是行为学版本（问才有解），可引它论证 ask 指标合理性 |
| CLARA | RA-L'24 | ✅ github.com/jeongeun980906/CLARA-SaGC | 分类"清晰/歧义/不可行"并对话澄清；对应我们 L1（忽略）/L2（识别）/L3（澄清）三级 |
| Learning to Ask (AwN) | arXiv 2409.00557 | ✅（论文附） | 不清晰指令下主动问；与我们"提问有代价+配额"的设计互补 |
| Robots That Ask For Help | arXiv 2307.01928 | ✅ | KnowNo 同系 |

## 4. 失败后学习 / 经验记忆（我们的 learning_spec 与谁比）

| 论文 | 出处 | 代码 | 对比点 |
|---|---|---|---|
| Reflexion | NeurIPS'23 | ✅ github.com/noahshinn/reflexion | 言语自反思跨 trial 学习；对应我们 single+raw 臂（原始失败进 prompt） |
| ExpeL | AAAI'24 | ✅ github.com/LeapLabTHU/ExpeL | 从成败轨迹提炼 insight 检索复用；对应我们的 L1 记忆卡 + 检索（3A-full 臂） |
| Voyager | arXiv 2305.16291 | ✅ github.com/MineDojo/Voyager | 技能库终身学习；对应我们的 L2 规则归纳（跨楼 transfer） |
| CLIN | arXiv 2310.10134 | ✅ | 持续因果记忆跨 episode 适应；同坑二踩率指标可与其对齐 |

## 5. 引用建议（写论文时的定位句）

- benchmark 定位：相对 SafeAgentBench/AgentDojo，我们贡献的是**授权语义 + 权威推理孪生 +
  分级难度 + 违规后果 + 可问询环境**的组合，且判定全程序化（无 LLM judge）。
- 架构定位：相对 MADRA/SAFER，我们给出**同模型配对对照（p=0.004）**并把"误拒/被骗"拆成
  方向性指标；且两篇均未开源，我们全栈可复现。
- 学习定位：Reflexion/ExpeL/Voyager 在游戏/文本环境验证经验学习，我们把它放进
  带授权约束的具身闭环，并用"同坑二踩率/记忆归因避让率"防止把碰巧当学习。
