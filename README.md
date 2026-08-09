# Risk Navigation Agents

当前阶段已包含 360 智脑 API 连通性测试和第一个 Task Advocate。

## Mac 上运行

在终端进入本目录，然后仅对当前终端会话设置 API Key：

```bash
export ZHINAO_API_KEY='你的 API Key'
export ZHINAO_BASE_URL='https://api.360.cn/v1'
export ZHINAO_MODEL='z-ai/glm-5.1'
python3 test_api.py
```

预期输出包含：

```text
模型：z-ai/glm-5.1
回复：API连接成功
```

GLM-5.1 会使用一部分输出额度进行内部推理，因此测试为它预留了
256 个输出 token。如果接口返回了用量但没有可见回复，通常说明输出
额度被推理 token 耗尽，并不代表鉴权或网络连接失败。

不要把真实 API Key 写入代码、README、`.env.example` 或 Git。

## 运行 Task Advocate

完成上述环境变量配置后：

```bash
python3 run_advocate.py
```

程序会读取 `scenarios/corridor_obstacle.json`，调用模型生成支持方报告，
并检查字段完整性、置信度范围以及动作是否属于允许动作集合。

## 运行 Risk Critic

```bash
python3 run_critic.py
```

Risk Critic 独立分析同一场景，输出分项风险、发生概率、严重程度、
证据、缓解措施、缺失信息和停止条件。它不会读取 Task Advocate 的结果。

## 运行完整三 Agent 流程

```bash
python3 run_three_agents.py
```

程序依次运行 Task Advocate、Risk Critic 和 Safety Decision Agent，验证每份
结构化输出，并汇总三个角色的 token 用量。每个成功报告会按场景内容缓存在
`.cache/` 中；某个角色失败后再次运行，只会重试尚未成功的角色及其下游，
避免重复调用和重复计费。最终模型决策还会经过 `safety_guard.py` 的确定性
硬规则检查；只有 `Safety Guard.approved_action` 可以交给机器人执行层。

## 离线测试 Safety Guard

```bash
python3 test_safety_guard.py
```

该测试不调用 API，也不产生模型费用。它验证紧急障碍、低观测置信度、
低电量、过窄通道和非法动作都能覆盖模型提出的不安全动作。

## 风险经验存储与检索

`experiences/risk_experiences.json` 使用统一结构保存任务、机器人、环境、
动作、结果、风险、决策和经验结论。第一版检索器采用透明的规则相似度，
根据机器人类型、通行余量、障碍物状态与距离、观测置信度和任务目标排序。

```bash
python3 run_retrieval.py
python3 test_experience_store.py
```

这两个命令均不调用 API。独立验证检索排序可以减少错误历史经验影响机器人
决策的风险。

完整三 Agent 流程现在会先检索最相关的 3 条经验，将其同时提供给三个角色，
并要求每份报告通过 `cited_experience_ids` 明确引用经验。缓存键同时包含场景、
经验内容和流程版本，因此经验库变化后不会错误复用旧决策。

检索结果进入模型前会被压缩为 `Memory Card`，仅保留经验 ID、相关性依据、
关键环境条件、动作、结果、风险和结论。完整原始经验仍保存在经验库中，
但不会在每个 Agent 请求中重复发送，以降低 prompt token 消耗。
