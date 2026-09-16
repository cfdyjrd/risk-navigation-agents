# 实验补齐 Checklist（按优先级；✅=已有 ☐=待做）

## 一、已完成的
- ✅ ladder 160 双臂全量（三Agent / single，同模型，McNemar p=0.004）
- ✅ 四个架构 baseline 臂 × priority 100（debate / refine / clara / single_guard）
- ✅ SafeAgentBench 规划层 600 任务双臂（拒绝率 79% vs 71.7%，p=2e-4）
- ✅ 旧 benchmark 三方对照（v1/v2/single，307）、重设计诊断、阶梯验收探针

## 二、实验补齐（高优先级）
- ☐ 四个 baseline 臂补跑 ladder 其余 60 场景(safe/drift-L1/ambiguous-L1)到 160 全量
     ——排除"只跑判别桶"的选择偏差（约 1500 万 token）
- ☐ 六臂共同集 93→100：补齐各臂零星失败场景
- ☐ **方差量化**：判别桶每场景 3 次采样（同模型非确定性未量化，p 值都基于单次）；
     报告置信区间 + 多重比较校正（六臂两两比较膨胀）
- ☐ **三 Agent 的 legit 过度拒绝修复实验**：decision prompt 加"权限表采信规则"，
     重跑 drift-L3（当前 14/20 采信 vs debate 18/20——这是三 Agent 唯一明显短板）
- ☐ 三 Agent 内部消融：no-critic / no-advocate / decision-only（分工中谁贡献大）
- ☐ ask 配额敏感性（0/1/2）与后果引擎 on/off 消融
- ☐ 恢复指标 A1 + recovery-amb 桶 A2（learning_spec 已设计，零 LLM 部分半天可做）
- ☐ 学习课程 Phase 0/1（learning_spec B：经验写回 + 同坑二踩率，约 2000 万 token）
- ☐ **跨模型验证**：第二个模型（glm-5.1 或 mimo）跑 三Agent/single/debate 三臂 ×
     判别桶——证明"debate 追平三Agent"不是 deepseek 特有

## 三、baseline 补齐
- ☐ **三Agent + 拓扑护栏组合臂**：single_guard 显示纯护栏在 legit 上全灭(0/20)，
     审议+护栏是否互补是自然的下一问
- ☐ debate 的 N/R 扫描（当前只有 N=3,R=2 一个点；N=5、R=1 的成本-效果曲线）
- ☐ KnowNo 式校准提问（API 无 logprob 则用多数投票近似置信度）
- ☐ MADRA 复现（辩论+评估者打分，与朴素 debate 差异）；Reflexion/ExpeL 臂（随学习课程）

## 四、benchmark 补齐
- ☐ SafeAgentBench 剩余子集：abstract(100) + long-horizon(50)；
     三 Agent 计划格式 −20pp 修复后重测 safe 300；执行层需装 AI2-THOR（或声明只报规划层）
- ☐ 外部第二基准（二选一）：AgentDojo（注入/权威方向最贴）或 SafeMind
- ☐ 判别桶扩容：ambiguous-L2/L3 每桶 20→40+（L3 p=0.062 差一步显著）；多 seed 措辞变体
- ☐ boundary_forgetting 式长程一致性桶（旧 benchmark 的结构性 0/0 问题仍未解决）

## 五、统计与呈现
- ☐ 全臂对照表统一为：代价比曲线 + 配对 2x2 + 分桶帕累托（token/干净成功）
- ☐ 每个结论标注:单次采样/多次采样、场景 n、检验方法
