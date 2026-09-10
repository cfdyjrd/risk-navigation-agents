# G1 人形机器人接入

本适配器使用 G1 `LocoClient.SetVelocity`，不复用 Go2 `SportClient`。
导入模块、运行测试、被动 DDS 诊断均不发送运动或停止指令。

## 已核对的现场环境（2026-09-10）

- SSH：`unitree@192.168.3.177`，Ubuntu 20.04.6 / aarch64 / Python 3.8.10。
- 无线管理网卡 `wlan0`；内部网卡 `eth0` 为 `192.168.123.164/24`。
- SDK 源码：`/home/unitree/unitree_sdk2_python`，包版本报告 `1.0.1`。
- 5 秒被动采样收到 `rt/lowstate`（`unitree_hg` 类型）223 帧、
  `rt/odommodestate`（`unitree_go.SportModeState_` 类型）222 帧。
  `rt/sportmodestate` 未收到数据。轮询计数不代表传感器实际发布频率。
- 最后一个里程计样本速度非零；不能认为现场已停稳。
- 已有 `/home/unitree/icra_framework/g1/g1_peer/g1_sdk.py` 可作为接口参考；
  本次没有启动、修改其控制服务。

IP/网卡需随现场网络重新核对。DDS 类型和 SSH 主机名不能唯一认证某一台机器人；
应隔离内部控制网络并现场核对设备。G1 与之前的 Go2 主机使用相同 SSH 主机公钥，
不能用这个公钥区分型号。

## 立即可用的只读检查

在机器人仓库目录运行：

```bash
python3 g1_dds_diagnostic.py --interface eth0 --duration 5
python3 -m unittest test_g1_adapter test_unitree_adapter test_robot_preflight test_execution_bridge test_robot_task_loop test_safety_guard
```

诊断程序仅创建 DDS 数据读取器，没有运动客户端；结果始终标记
`motion_ready=false`、`motion_commands_sent=0`。它兼容机器人系统 Python 3.8。
阶段提示和每秒计数输出到 stderr，最终 JSON 输出到 stdout。`--duration` 是
采样时长；整个子进程（包含 SDK 初始化与退出）另有“采样时长 + 10 秒”的总超时，
超时会结束该只读进程并返回 124。默认 5 秒采样的总超时是 15 秒。
它不生成完整观测文件，因为这些消息没有提供已核实的电量、障碍距离和运控 FSM。
不能把 `mode_machine` 或里程计 `mode` 当作运控 `fsm_id`。

## G1 运动接口

`G1LocoDriver` 实现共享适配器所需的 `move(vx, vy, yaw_rate)` / `stop()`。
现场 SDK 的 `Move()` / `StopMove()` 包装函数丢弃返回码，因此直接使用
`SetVelocity(vx, vy, yaw_rate, duration)` 检查返回值必须是整数 0。
每个请求默认携带 0.2 秒有效期，约每 0.05 秒重新发送；不使用持续运动模式。
请求有效期不是实测制动距离，也不是经过验证的硬件看门狗。

停止发送零速度，不调用 `Damp`、`ZeroTorque`、站起或模式切换。
机器人须由现场操作员先置于允许高层速度控制的状态。
SDK 结果只代表命令提交，不能证明实际移动或停止。

## 必须提供的真实观测

`read_sensor_snapshot()` 必须快速读取同步缓存并返回 `RobotObservation`，包含
现有契约中的真实电量、机身通行宽度、障碍测距、置信度、采样时间，
`robot.type="g1"`，`robot.device_id` 与配置一致。

G1 还强制要求 `metadata.g1_state`：

| 字段 | 来源与用途 |
| --- | --- |
| `lowstate_timestamp` | 当前绑定设备的 `unitree_hg LowState_` 新样本时间 |
| `fsm_timestamp` | 独立读取运控 FSM 的采样时间 |
| `fsm_id` | 实际运控 FSM，必须在现场核定的允许列表内 |
| `attitude_timestamp` | 实测姿态采样时间 |
| `roll_rad`、`pitch_rad` | IMU 横滚、俯仰，单位弧度 |

三个时间戳均要求带时区、不过期、不来自未来；不能读取旧缓存时刷新时间。
顶层时间戳仍用最旧必需数据采样时间，主机与传感器须同步。
若只能得到接收时间，需先核实传输延迟和数据源更新机制，不能把回放或积压数据
当作新采样。状态类型仅是平台一致性证据，不是设备身份认证。

G1 状态缺失、FSM 不符、倾斜超限时拒绝运动，并在运动循环中持续检查。
这些检查不替代厂家平衡控制、现场独立急停或侧后方和台阶感知。
现有 Safety Guard 的距离、通行余量等阈值仍是原型基线，需要按 G1 实测核定。

## 配置与接线

复制 `configs/g1.preflight.example.json` 为本地配置并填写：

- `allowed_fsm_ids`：现场控制配置核定的列表，不自动假定 200 或 500。
- `max_tilt_rad`：现场核定的姿态限制。
- `observation_file`：真实传感器导出的同步 JSON。
- 网卡、设备编号、速度等按实际情况核对。

模板故意留空必要项，直接运行会失败。G1 默认前进上限 0.1 m/s、低速
0.05 m/s、转向 0.2 rad/s、单次最长 0.5 秒，均为待验证的工程设置。

```bash
python3 robot_preflight.py --config configs/g1.preflight.local.json --duration 5
```

预检通过只说明本机 SDK 与观测缓存检查通过，不证明控制权和运动准备完成。

完整传感器接入后，应用启动入口按以下结构接线（变量由现场应用提供）：

```python
from execution_bridge import SafeExecutionBridge
from robot_interface import SEMANTIC_ACTIONS
from robot_task_loop import RobotTaskLoop, make_memory_decider
from unitree_adapter import G1Config, G1LocoDriver, UnitreeRobotAdapter

config = G1Config(**validated_robot_config)
driver = G1LocoDriver.connect(verified_interface)
adapter = UnitreeRobotAdapter(config, driver, read_sensor_snapshot)
bridge = SafeExecutionBridge(adapter, available_actions=sorted(SEMANTIC_ACTIONS),
                             experience_sink=save_execution_record)
loop = RobotTaskLoop(bridge, make_memory_decider(store, rules, client),
                     is_stationary=measured_stationary,
                     goal_reached=verified_goal_reached)
try:
    result = loop.run(task)
finally:
    adapter.emergency_stop("shutdown")
```

上述代码会在调用后发送控制指令，只有现场运动准备完成后才可执行。
不要通过直接调用 driver 绕过观测、Guard 和优化器路径。
完整感知回调、停稳/到达判断、模型配置尚未接通，当前没有一键行走入口。
仓库根目录的决策与执行模块已针对系统 Python 3.8 修复运行时类型注解兼容问题；
这不表示 `2d-simulator` 的 Python 3.11+ 要求改变。

## 当前验证边界

已验证 SDK 源码/API 存在、实际状态接收、离线的驱动映射和故障停止逻辑。
未执行真实 `SetVelocity`，未验证电量/障碍感知来源、允许 FSM、运动控制权、
实际位移、制动或到达。API 密钥未复制到机器人。

## 电量、FSM 与观测缓存（2026-09-10 后续接入）

新增 `g1_state_collector.py`，在独立、有总超时的进程中订阅：

- `rt/lowstate`：G1 姿态与 tick。
- `rt/odommodestate`：位置、速度、角速度和状态错误码。
- `rt/lf/bmsstate`：`unitree_hg.BmsState_` 的真实 `soc` 电量。
- `sport` 服务只读 API `7001`：运控 FSM。仅注册 GET 接口，不加载运动客户端。

```bash
cd /home/unitree/risk-navigation-agents
python3 g1_state_collector.py --interface eth0 --duration 10 \
  --output /tmp/risk-navigation-g1-state.json
python3 g1_lidar_probe.py --duration 5
```

状态文件以原子替换方式更新；只有标记属于此采集器的 JSON 才能被后续运行覆盖。
采集时长默认 10 秒、最多 300 秒，父进程总超时为采集时长加 15 秒。
采集结束后源时间戳不再更新，缓存自然过期；这不是后台常驻服务。
`state_cache_ready` 仅检查四种状态的时间有效性和采集错误，
`motion_ready` 始终为 false。缺失、未来或超过一秒的时间戳导致缓存未就绪，
退出码 1；0 也不表示可以运动。

实测采集到了电量 **81%** 和 FSM **1**（当时的状态，不是持续保证）。
现场 SDK 的 `Damp()` 写入状态 1；本次未进行任何模式切换。
DDS 写入端时间戳比本机接收时间落后约 **24.7 秒**。主机 `timedatectl`
显示 NTP 同步，但这不能证明 DDS 发布端时钟一致，也不能仅凭此区分时钟偏差和延迟。
需要核查发布端时间与链路。采集器明确报告每个源的 `age_s/fresh`，不通过加偏移或
刷新为当前时间规避适配器的时效要求。

雷达 `/utlidar/cloud_livox_mid360` 在 5 秒内收到 50 帧，坐标系为
`livox_frame`；一帧约 20160 个有效点。最近原始回波约 0.099 米，可能包含
机身、地面等，**不能把这个值当作前方障碍距离**。探针按 PointCloud2 的
字段类型、大小端和行跨度解析，只用于检查数据，不宣称完成障碍感知。

`g1_observation_cache.build_observation()` 将四种状态与外部可信感知转换为
现有 `RobotObservation`：所有源必须新鲜；保留最旧源时间；没有电量或感知
不会补默认值；要求无采集/里程计错误，并复用 G1 FSM/姿态校验。
`measured_stationary()` 根据新鲜速度与角速度判断，任务循环仍需两帧新观测。

用户确认目前没有实测通行宽度和雷达到机身的安装标定。因此以下项仍阻断运动：

1. 测量当前姿态下的通行宽度（含手臂、附件及所需余量）。
2. 核定雷达到 `base_link` 的变换、地面/自身回波过滤与感知覆盖。
3. 核对 DDS 发布端与主机时间及传输延迟。
4. 由现场操作员核定允许的运控状态、姿态限制、独立停止和目标到达判据。

不能通过将当前 FSM 1 加入允许列表来代替运动准备，也不能用点云数量推断
置信度或“无障碍”。本阶段仅完成真实状态采集和受约束的观测转换接口。

## 三 Agent 只决策入口

`run_g1_decision.py` 默认汇总实际观测与现场配置的阻断项，不调用模型，也不
创建运动驱动。模板中的测量值故意留空，可先直接运行，查看还缺什么：

```bash
python3 run_g1_decision.py --config configs/g1.runtime.example.json
```

现场配置完成后，复制模板为 `configs/g1.runtime.local.json`，填写真实的
`width_m`、允许 FSM、姿态限制，以及感知文件路径。相对路径以配置文件目录为准。
状态文件由另一终端持续采集；已停止采集的文件会过期。感知文件结构须满足
`g1_observation_cache.build_observation()`，且 `validated` 标记必须来自实际核验流程，
不能仅手动改为 true。

```bash
# 完整输入通过检查后才会调用三 Agent，可能产生 API 用量；仍不发送运动指令。
python3 run_g1_decision.py --config configs/g1.runtime.local.json --decide
```

该入口只提供前进、低速前进、重新观察、人工请求和停止候选，不开放缺少侧向
感知支持的转向。输出决策、证据审计与 Guard 预览；所有输出标记不能直接执行。
预览使用决策前的观测，可能在模型返回时已经过期，不能缓存后直接下发。
真实执行仍须重新读状态、独立停稳/到达判据和 `SafeExecutionBridge`。
当前缺少物理标定，因此没有启用运动模式或编造到达判据。

时间检查补充：开发电脑使用 systemd-timesyncd，同步服务器报告偏差约 -19 毫秒；
这不解释 DDS 端约 24.7 秒差距。内部邻居可见 `192.168.123.161`，但尚未验证
其与 DDS 发布者的对应关系，未登录或修改它的时间。现有 MID360 配置中的外参
全为零，不能据此认为雷达与机身坐标系重合。

## FSM 与时钟差异进一步核查

后续同日检查：`sport` 服务 GET 7001 返回 500、GET 7002 返回 0；
`loco` 服务 GET 查询返回 3102（发送失败）。这与现机 SDK 的服务名 `sport`
一致。先前 FSM 1 是当时查询结果，后续已发生改变；没有由本仓库切换模式。

新增有总超时的 `g1_clock_probe.py`，仅构造 API 7001 GET 请求，使用随机唯一
请求 ID 与响应逐一关联，依据发送时间、接收时间与响应的 DDS 写入时间求偏差区间：

```bash
python3 g1_clock_probe.py --interface eth0
```

三次实测往返约 5.6–5.8 毫秒，响应写入端时钟落后开发电脑约
24.724–24.730 秒，因此该响应的差值是时钟偏差证据，不是 24 秒网络延迟。
脚本也输出实际写入者及 DDS 参与者 GUID。观察到：

- `sport` 响应使用一个参与者；
- `lowstate` 与 `lf/bmsstate` 共享另一个参与者；
- `odommodestate` 使用第三个参与者。

不同参与者不证明它们位于不同电脑，但不足以证明时钟必然相同。
因此不能把 GET 响应的校时区间自动应用到全部状态流。内部邻居
`192.168.123.161` 的 SSH 22 端口明确拒绝连接，未取得其系统时钟配置。
未修改任何时钟、姿态或运动模式；下一步需要通过实际运控端管理入口确认
状态发布进程所在主机及其同步配置。现场宽度、雷达标定等缺项仍保留。
