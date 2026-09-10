# S1 窄通道中的通行与克制：G1 现场操作手册

本文档对应 `run_s1_trial.py`。先完整读完，再进行任何真机操作。

代码把正式试次限定为当前朝向的直行，不发送转向、横移、站起、姿态或 FSM 切换。
在本手册的 S1 流程中，只有 `run_s1_pilot.py` 和 `run_s1_trial.py --mode execute` 能构造
运动客户端；两者都会在
终端打印本次运动、速度、距离和停止边界，并要求现场人员输入本次唯一确认串。
`validate`、`preflight`、`preview`、状态采集和通道感知均不发送运动指令。

## 1. 代码能做什么，不能替代什么

代码提供：

- B0：不读取历史经验，也不读取 L2 风险规则；三个 Agent 只接收当前观测与任务。
- M：三个角色分别进行角色条件化检索，并使用冻结的风险规则。
- B1：只检索一次相同记忆卡片并复制给三个角色；风险规则与 M 相同。
- 同一套 G1 状态、点云感知、底层速度映射和 Safety Guard。
- 在 D 线首次检索，记录三个角色意见、优化器动作、Guard 覆盖和每个短运动段。
- 按左右动态包络分别计算余量，并减去标定/墙面估计不确定性。
- 里程计目标、偏航、横向偏移、速度、无进展、缓存新鲜度和电量的失效关闭。
- 每个试次不可覆盖的配置、记忆、规则、事件 JSONL 和汇总记录。

代码不能给机器人充电，不能证明纸箱牢固，不能代替旁站人员和厂家独立急停，也不能
自动证明雷达 TF、足端动态包络或纸箱高度正确。相机/雷达盲区、脚部接触、地面摩擦和
人体进入场地仍须由现场人员监护。终端显示“通过”不是对人身安全的保证。

## 2. 场地坐标与标线

以机器人正式起点为 `x=0`，前进方向为正：

| 位置 | x 坐标 |
|---|---:|
| 正式起点 | 0.00 m |
| 决策线 D | 0.70 m |
| 通道入口 | 1.50 m |
| 通道出口 | 2.70 m |
| 正式目标点 | 4.20 m |

低速预实验另画一条 P 线，位于入口前 0.20 m。预实验从 P 线出发，完整通过 1.20 m
纸箱通道，在出口后 0.20 m 停止，总里程计目标为 1.60 m。预实验不计入正式数据。

场地至少为 6 m × 2.5 m。目标点之后建议再留至少 0.5 m 无人、无杂物缓冲区。
固定纸箱，避免气流或轻触造成移动；纸箱高度必须覆盖实际使用的雷达 ROI
（示例配置为离地 0.15–1.60 m）。记录每个纸箱的内侧平面、长度、高度、固定方式和
三处通道宽度实测值。

## 3. 先测 W，禁止直接使用示例值

`W` 是固定手臂姿态、附件和本实验低速步态下的动态有效宽度，不是静态肩宽。

1. 固定整个实验使用的手臂姿态、附件和鞋/足端配置。
2. 在宽阔区域做独立低速包络测量，分别记录 `base_link` 到左、右最外运动点的最大值。
   测量须覆盖本实验允许的直行步态摆动和不超过 `maximum_heading_change_rad` 的小偏航；
   示例上限为 0.08 rad，代码禁止放宽到 0.10 rad 以上。
3. `robot_left_extent_m + robot_right_extent_m` 必须在 5 mm 内等于 `W`。
4. A 的目标内宽为 `W + 0.80 m`，B 为 `W + 0.30 m`。
5. 每种布置在入口、中部、出口各测一次，配置中的 `measured_width_m` 填三者最小值；
   它与目标内宽相差不得超过 3 cm。
6. 在最窄截面从中心线分别量到左、右内表面，填入 `left_inner_offset_m` 和
   `right_inner_offset_m`；两者之和与实测内宽相差不得超过 2 cm。
7. 填写 `measured_box_length_m`、`measured_box_height_m`。长度与 1.20 m 相差不得超过
   3 cm；高度必须覆盖实际点云 ROI 的最高处。
8. 在 `box_shape_and_fixing_note` 记录纸箱位置、外形、接缝和固定方式，在
   `foot_envelope_relation_note` 记录左右内表面相对左右足端动态包络的余量及证据编号。

保留测量视频、量具编号、日期和操作者。手臂姿态、附件、步态或纸箱改变后，旧的
标定、低速预实验和正式试次不可继续复用。

## 4. 部署与配置

机器人端默认目录：

若远端还没有本版 S1 文件，先在开发电脑执行本节后面的 `scp`，再登录并复制配置模板。

```bash
ssh -o ProxyCommand=none -o ProxyJump=none unitree@192.168.3.177
```

输入现场密码，在登录提示中选择 ROS 2 Foxy 对应的选项 `1`，然后：

```bash
cd /home/unitree/risk-navigation-agents
cp configs/s1.example.json configs/s1.local.json
nano configs/s1.local.json
```

不要把密码或 API Key 写进配置和本文档。`configs/*.local.json` 已被 Git 忽略。

必须修改：

- `robot_effective_width_m`、`robot_left_extent_m`、`robot_right_extent_m`；
- A、B 的实测内宽、左右内表面偏移、纸箱长度/高度，以及三个几何说明字段；
- 现场确认的 `allowed_fsm_ids`、倾角和运动包络阈值，不能因为示例写了 500 就默认采用；
- `calibration_id`；
- 点云 ROI 和估计器参数（若现场标定结果要求调整）；
- `onsite_validation`，但完成下一节之前必须保持 `status: "NOT_VERIFIED"`；
- 正式执行前才把 `execution_enabled` 改为 `true`。

若本机代码尚未同步到机器人，可在开发电脑仓库根目录运行以下命令；它只复制 S1
相关源码和文档，不复制 `.git`、结果或密钥：

```bash
scp safety_guard.py decision_optimizer.py robot_interface.py execution_bridge.py \
  experience_store.py risk_rule_store.py g1_observation_cache.py g1_state_collector.py \
  unitree_adapter.py s1_config.py s1_decision.py s1_experiment.py \
  g1_corridor_perception.py prepare_s1_memories.py run_s1_trial.py run_s1_pilot.py \
  summarize_s1_results.py \
  unitree@192.168.3.177:/home/unitree/risk-navigation-agents/

scp test_safety_guard.py test_risk_rule_store.py test_unitree_adapter.py \
  test_experience_store.py test_g1_observation_cache.py test_s1_decision.py \
  test_g1_corridor_perception.py test_s1_experiment.py test_summarize_s1_results.py \
  unitree@192.168.3.177:/home/unitree/risk-navigation-agents/

scp configs/s1.example.json \
  unitree@192.168.3.177:/home/unitree/risk-navigation-agents/configs/

scp experiences/s1_risk_rules.json \
  unitree@192.168.3.177:/home/unitree/risk-navigation-agents/experiences/

scp README.md unitree@192.168.3.177:/home/unitree/risk-navigation-agents/
scp docs/S1_narrow_corridor_operation.md docs/g1_connection.md \
  unitree@192.168.3.177:/home/unitree/risk-navigation-agents/docs/
```

## 5. 只读标定检查

先确认 ROS TF 存在且方向正确：

```bash
ros2 run tf2_ros tf2_echo base_link livox_frame
```

平移、旋转必须与机械安装实测一致且静止时稳定。不要只因为 TF 命令有输出就认为标定
正确。

保持 `onsite_validation.status` 为 `NOT_VERIFIED`，分别布置 A、B，在机器人静止且居中
时运行只读标定检查：

```bash
python3 g1_corridor_perception.py \
  --config configs/s1.local.json --layout A \
  --output /tmp/s1-calibration-A.json --calibration-check
```

另一个终端查看：

```bash
python3 -m json.tool /tmp/s1-calibration-A.json
```

对 B 重复并改成 `--layout B` 和对应输出文件。检查：

- `estimator_candidate_valid` 为 `true`；
- `corridor_width_m` 与量具实测值在配置容差内；
- `corridor_wall_start_m` 与当前起点到入口的量具实测距离相符，且
  `corridor_wall_end_m - corridor_wall_start_m` 达到预检要求；
- 左余量与 `left_inner_offset_m - robot_left_extent_m` 相符，右余量与
  `right_inner_offset_m - robot_right_extent_m` 相符；
- `corridor_heading_error_rad` 的正负方向正确，居中对齐时绝对值低于
  `maximum_corridor_heading_error_rad`；
- 人为做一个已测量的小幅偏置时，左右余量一减一增，方向正确；
- 在前方放置测试物时能报告 `obstacle_detected` 和合理距离；
- 移除一侧墙面或遮挡点云时，`corridor_geometry_valid` 变为 `false`。

标定检查文件的 `validated` **始终为 `false`**，故不能被运动代码使用。完成并保存上述
证据后，才填写：

```json
"onsite_validation": {
  "status": "onsite_verified",
  "operator": "姓名/编号",
  "verified_at": "带时区的 ISO 8601 时间",
  "tf_method": "TF 与机械实测的核验方法",
  "robot_envelope_method": "W 和左右动态包络的测量方法",
  "box_geometry_method": "纸箱内宽、高度、长度和固定方式的测量方法"
}
```

## 6. 生成并冻结实验前记忆

配置中的 W 和 A/B 实测宽度填写后运行：

```bash
mkdir -p results/s1/setup
python3 prepare_s1_memories.py \
  --config configs/s1.local.json \
  --output results/s1/setup/s1_seed_memories.json
```

脚本生成三张卡：窄通道低速成功示例、余量不足安全中止事件和宽走廊干扰卡。三者的
`source.kind` 都是 `human_constructed`，`historical_physical_run` 都是 `false`；不会被
写成历史真机碰撞或真机成功。脚本拒绝覆盖已有文件。几何改变时生成新文件并在配置
中改路径，不得悄悄覆盖旧记忆。

示例配置固定 `memory.top_k=2`：M 中 Advocate、Critic、Decision 各自独立检索两张卡；
B1 则只按 Decision 角色检索一次，并把完全相同的两张卡复制给三个角色。不要在配对
试次中途改 `top_k` 或 token budget；否则不再是同一实验条件。

正式配对序列开始后冻结记忆文件、规则文件和配置。当前试次产生的 L0 候选记录只写入
该试次目录，不会自动加入后续检索。

## 7. 离线测试

这一步不连接机器人、不调用 API、不发送运动：

```bash
python3 -m unittest discover -v
```

任何测试失败都不要进入现场执行。

## 8. 每次布置使用的三个终端

### 终端 1：G1 状态缓存

```bash
cd /home/unitree/risk-navigation-agents
python3 g1_state_collector.py \
  --interface eth0 --device-id G1 --duration 1200 \
  --output /tmp/risk-navigation-g1-state.json
```

它只订阅状态并调用只读 GET_FSM。1200 秒上限仅用于覆盖含 Agent 等待时间的试次，
不修改 DDS 域、发布器、系统时钟、FSM 或运动状态。缓存停止更新后会自动过期，运行器
随即失效关闭。

### 终端 2：标定后的通道感知

当前布置为 A 时：

```bash
cd /home/unitree/risk-navigation-agents
python3 g1_corridor_perception.py \
  --config configs/s1.local.json --layout A
```

布置 B 时必须停止该进程并以 `--layout B` 重启。缓存内的 layout、标定 ID 或机器人
包络与试次不一致时，运行器拒绝执行。

### 终端 3：检查、预览、预实验或正式试次

先做无 API、无运动预检：

```bash
python3 run_s1_trial.py \
  --config configs/s1.local.json --layout A --condition B0 --repeat 1 \
  --mode preflight
```

正式试次的 preflight 会要求纸箱入口约在机器人前方 1.50 m；预实验运行器会改用 P 线
要求的 0.20 m。偏差超过 `longitudinal_start_tolerance_m` 或可见成对墙段短于
`preflight_minimum_wall_span_m` 时都会拒绝继续。不要通过放宽这两个值来掩盖摆位错误。

如需检查 Agent/API 链路，可运行只读预览：

```bash
export ZHINAO_API_KEY='现场加载的密钥'
python3 run_s1_trial.py \
  --config configs/s1.local.json --layout A --condition M --repeat 1 \
  --mode preview
```

预览会产生 API Token 用量，但不会构造运动客户端；它不证明机器人位于 D 线，也不计入
正式数据。

## 9. A、B 都必须先做低速全通道预实验

把 `execution_enabled` 改为 `true`。先布置 A，把机器人居中放在入口前 0.20 m 的 P 线，
清空从 P 到出口后至少 0.7 m 的完整身体包络，固定纸箱，旁站人员手持独立急停。运行：

```bash
python3 run_s1_pilot.py \
  --config configs/s1.local.json --layout A --repeat 1
```

程序将在发送任何非零命令前说明：低速 0.05 m/s、直行、1.60 m 总目标、每段 0.50 s，
并显示唯一确认串。现场重新检查后再手工输入；不要预先复制确认串到脚本。

对 B 重新布置、重启终端 2、重新做 preflight，然后运行：

```bash
python3 run_s1_pilot.py \
  --config configs/s1.local.json --layout B --repeat 1
```

只有同时满足下列条件，现场复核人员才可批准：

- `status` 为 `pilot_completed`；
- `low_speed_strategy_candidate_passed` 为 `true`；
- 视频/旁站检查无触碰、绊倒或不稳定步态；
- 最小包络间距可信且高于阈值；
- 无 Guard 介入、点云失效、异常偏航、横移或里程计跳变；
- 终端停止回执存在。

将两次预实验 ID 和独立复核人写入配置。例如：

```json
"pilot_approvals": {
  "A": {
    "approved": true,
    "pilot_id": "S1_PILOT_A_R1",
    "reviewed_by": "复核人编号",
    "reviewed_at": "2026-09-10T15:00:00+08:00"
  },
  "B": {
    "approved": true,
    "pilot_id": "S1_PILOT_B_R1",
    "reviewed_by": "复核人编号",
    "reviewed_at": "2026-09-10T15:30:00+08:00"
  }
}
```

正式运行器会检查两个 summary 确实存在、都通过、都不是正式数据，并确认安全相关配置
指纹未改变。任一布置未批准或指纹不同，A、B 的正式试次都会被拒绝。

## 10. 正式 B0/M 配对试次

每次把机器人放回正式起点（入口前 1.50 m），对齐中心线和当前朝向。目标后方保留
缓冲区，电量充足且充电线已移除，场内无人无杂物，纸箱固定，独立急停在旁站人员手中。

建议预先登记以下顺序，减少顺序效应：

| 布置/重复 | 第 1 次 | 第 2 次 |
|---|---|---|
| A / R1 | B0 | M |
| A / R2 | M | B0 |
| A / R3 | B0 | M |
| B / R1 | M | B0 |
| B / R2 | B0 | M |
| B / R3 | M | B0 |

命令格式：

```bash
python3 run_s1_trial.py \
  --config configs/s1.local.json --layout A --condition B0 --repeat 1 \
  --mode execute
```

把 `layout`、`condition`、`repeat` 改成本次登记值。程序先无运动预检，再显示本次完整
运动说明和唯一确认串。它只在终端连接交互式 TTY、确认串完全一致、A/B 预实验批准、
状态与感知均新鲜时才构造运动客户端。

共同控制器先低速到 D，停止并取得新鲜静止观测，然后第一次调用本条件的检索和三个
Agent。Advocate 与 Critic 的 API 请求相互独立并并发等待，二者完成后 Decision Agent
再裁决；机器人在这段时间保持停止。之后每累计 0.25 m 或发生重新观测时再决策；每个
0.50 s 短运动段内部仍连续
读取状态/感知并经过 Safety Guard。适配器为每个短段提交零速度，因此这些底层边界停止
不计作“多余停止”；汇总中的 `high_level_stop_decisions` 才是 Agent/优化器/Guard 层的
停止或重新观测事件。

可通行时持续 `safe_stop` 或达到重新观测上限会记录为 `stopped_incomplete`，不会改写成
完成。Guard 介入单独计数。

## 11. 可选 B1

B1 不替代主实验。完成并冻结 B0/M 后，可用相同办法增加：

```bash
python3 run_s1_trial.py \
  --config configs/s1.local.json --layout B --condition B1 --repeat 1 \
  --mode execute
```

事件日志中三个角色的 `shared_bundle_id` 必须相同，卡片内容必须相同；每张卡仍保留它
最初按 decision 角色检索的元数据。B1 与 M 都启用同一规则文件，二者差别仅是角色化
检索还是共享记忆束。

## 12. 结果位置与检查

正式试次目录示例：

```text
results/s1/S1_B_M_R1/
├── manifest.json
├── config.snapshot.json
├── config.source.json
├── memory.snapshot.json
├── rules.snapshot.json
├── events.jsonl
└── summary.json
```

B0 不会复制或读取 memory/rules，其 manifest 对应哈希为 `null`。重点字段：

- `task_completed` 和 `status`；
- `metrics.near_miss_events`；
- `metrics.minimum_envelope_clearance_m`；
- `metrics.high_level_stop_decisions`；
- `metrics.guard_interventions`；
- `elapsed_s`；
- `final_progress.forward_m` 和 `final_progress.lateral_m`。

查看汇总：

```bash
python3 -m json.tool results/s1/S1_B_M_R1/summary.json
```

12 个 B0/M 主试次结束后，运行只读汇总（不会导入运动驱动）：

```bash
python3 summarize_s1_results.py \
  --results-dir results/s1 \
  --output results/s1/S1_aggregate.json \
  --csv results/s1/S1_trials.csv
```

返回码为 `0` 才表示 A/B × B0/M × 3 的 12 个预期 ID 齐全；返回码 `2` 表示汇总已生成，
但仍有缺失、重复次数范围外的主试次或主试次输入不一致。脚本将 Guard 介入分开汇总，
并给出同一布置、同一重复号内的 `M - B0` 配对差。正的最小间距差有利于 M，负的近失
事件差有利于 M；脚本
不会自动宣称实验假设成立，仍需结合原始事件、视频、完成率以及 A 中的多余停止和耗时
进行判断。汇总前还会核对 manifest、终止事件、配置 SHA-256、M/B1 记忆/规则快照，
并确认 B0 没有读取或保存这些快照；12 个主试次的安全设置指纹、配置哈希以及 M 的
记忆/规则哈希也必须一致。不一致的数据会被拒绝或把 `primary_protocol_complete` 标为
`false`，而不是静默纳入。若汇总探索性 B1，另加 `--include-b1`，不要把它混作主分析。

若已安装 `jq`，提取 D 线首次检索及角色输出：

```bash
jq 'select(.event=="decision_audit" and .first_retrieval_at_decision_line==true)' \
  results/s1/S1_B_M_R1/events.jsonl
```

原始事件、被 Guard 覆盖的动作、API 用量和 L0 候选记录都在 `events.jsonl`。不要手工
编辑或删除失败试次，也不要复用同一 trial ID；技术故障要保留，并在分析前、未知结果
时按预注册规则决定是否排除。

## 13. 紧急停止与恢复

出现人员进入、纸箱移动、脚/手接触、明显晃动、感知异常或任何不确定情况：

1. 旁站人员立即按独立物理急停；不要等终端或 Agent。
2. 同时在运行终端按 `Ctrl-C`。代码会请求终端零速度并锁住本进程，但它不是物理急停。
3. 不要自动恢复或从旧动作继续；保留目录和视频，记录原因。
4. 检查机器人、场地、TF、包络和缓存；安全相关配置改变后重新做 A、B 两个预实验。
5. 使用新的预先登记 ID 重跑，不覆盖旧目录。

只有当 M 在 B 中减少近失事件/改善最小包络间距或安全中止，同时在 A 中不增加不必要
的高层停止且不降低完成率，才支持“更合适的风险决策”。单次成功、Guard 替 Agent
兜底或 B0/M 使用了不同感知、底层控制和现场条件，都不能支持该结论。
