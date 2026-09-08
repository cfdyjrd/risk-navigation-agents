# Go1 / Go2-1 真机接口

设备型号与编号分开配置：Go1 使用 `model="go1"`，Go2-1 使用
`model="go2", device_id="Go2-1"`。Go1 的实际编号可以通过 `device_id` 设置。
`littleRobots-main/g1_*.py` 是 G1 人形机器人的代码，不用于 Go1。

## 已实现的边界

```text
传感器缓存 → RobotObservation → SafeExecutionBridge → Safety Guard
                                            ↓
                                  UnitreeRobotAdapter
                                   ↙              ↘
                         Go1UDPDriver       Go2SportDriver
                         legacy UDP          SDK2 DDS
```

`unitree_adapter.py` 不包含 Agent、VLM 或 WebSocket 服务。上层继续通过现有
`SafeExecutionBridge.execute_decision()` 下发确定性优化器的结果；可以从机器人本机
进程调用，也可以由已有服务在校验消息后调用。导入模块不会加载 SDK 或连接硬件。

参考库的 Go2 `Move` / `StopMove` 接法已接入；Go1 使用独立高层 UDP 驱动。
此处不使用 G1 接口，不自动站立、不切换运动模式控制权、不实现低层关节控制。
机器人应由现场操作员先置于适合高层速度控制的状态。

## 安装与网络

核心接口及离线测试仅使用 Python 标准库。真机工作进程另需安装对应宇树 SDK：

- Go2：[unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)。
  明确传入与该机器人相连的网卡名。DDS 初始化属于进程级配置，建议每台机器人
  独立进程和网络连接。`device_id` 是应用层标识，不是 DDS 目标地址；同一 DDS
  网络存在多台机器时必须在部署层隔离，不能只改编号来路由。
- Go1：[unitree_legged_sdk 的 go1 分支](https://github.com/unitreerobotics/unitree_legged_sdk/tree/go1)。
  需要匹配实际 Linux 架构、Python ABI 的 `robot_interface*.so` 及其动态库依赖。
  将完整绝对路径交给 `Go1UDPDriver.connect()`。通过隔离名称加载该扩展，避免与
  本项目的 `robot_interface.py` 重名。必须显式提供实际机器人 IP；本地和远端
  UDP 端口默认 8080 / 8082，可覆盖。

SDK 版本、网卡名、Go1 IP 与现机控制权限尚未在硬件上验证；没有安装 SDK 或启动连接。

## 提供观测

适配器接收 `observation_provider` 回调，返回 `RobotObservation`。将你们现有的
电量、激光/深度或其他可靠测距、感知置信度结果合并为一个同步快照。回调必须快速、
非阻塞地读取缓存，不要在里面拍照、调用 VLM 或等待网络回复。

观测的必需字段：

| 字段 | 含义 |
| --- | --- |
| `robot.device_id` | 与当前适配器目标编号完全一致 |
| `robot.type` | `go1` 或 `go2` |
| `robot.width_m` | 实测机身通行宽度（米），正数 |
| `robot.battery_percent` | 电量 0–100，不能用固定默认值代替 |
| `environment.obstacle_detected` | 真正的 JSON/Python 布尔值 |
| `environment.obstacle_distance_m` | 有障碍时为有限非负米数；无障碍允许 `None` |
| `environment.observation_confidence` | 有限的 0–1 数值 |
| `timestamp` | 带时区 ISO 8601 采样时间，例如 UTC `...+00:00` |

走廊场景可另提供 `environment.corridor_width_m`。时间戳必须代表底层数据采样时间，
不能在读取旧缓存时更新为当前时间；合并不同传感器时使用最旧必需数据的采样时间，
并做好时钟同步。设备编号必须来自正确绑定的数据源。接口默认拒绝超过 1 秒的观测、
未来时间、型号或编号不匹配、NaN/Infinity 和用布尔值冒充数值。

本次没有实现摄像头、Go1 HighState 或 Go2 状态主题的自动订阅与传感器融合。
原参考库的图像与 VLM 输出不能直接当作完整的测距和电量观测。传感器驱动应在独立
线程/进程更新缓存；Go1 的 `Recv/GetRecv` 也不要放进运动发送回路。

## 接入现有流水线

下面是接线函数，可由你们的启动入口调用。`read_sensor_snapshot` 和
`optimizer_decision` 必须来自已有感知与优化器，不应从模型原始输出直接执行。

```python
from execution_bridge import SafeExecutionBridge
from unitree_adapter import UnitreeConfig, UnitreeRobotAdapter, Go2SportDriver, Go1UDPDriver

ACTIONS = ["move_forward", "turn_left", "turn_right", "slow_down",
           "observe_again", "ask_human", "safe_stop"]


def connect_go2(network_interface, read_sensor_snapshot):
    driver = Go2SportDriver.connect(network_interface, timeout_s=1.0)
    adapter = UnitreeRobotAdapter(
        UnitreeConfig(device_id="Go2-1", model="go2"),
        driver, read_sensor_snapshot,
    )
    return adapter, SafeExecutionBridge(adapter, available_actions=ACTIONS)


def connect_go1(sdk_extension, robot_ip, read_sensor_snapshot):
    driver = Go1UDPDriver.connect(sdk_extension, robot_ip)
    adapter = UnitreeRobotAdapter(
        UnitreeConfig(device_id="Go1", model="go1"),
        driver, read_sensor_snapshot,
    )
    return adapter, SafeExecutionBridge(adapter, available_actions=ACTIONS)


def execute_one(bridge, task, optimizer_decision):
    return bridge.execute_decision(task, optimizer_decision)
```

应用退出时在自己的 `finally` 中调用 `adapter.emergency_stop("shutdown")`。
接口不注册全局信号处理器，由主进程统一管理退出与遥控急停。

## 动作参数与回执

所有参数显式带单位；未知参数会被拒绝，不会默默忽略。

| 动作 | 参数 | 默认行为 |
| --- | --- | --- |
| `move_forward` | `speed_mps`；`duration_s` 或 `distance_m` 二选一 | 0.3 m/s，0.5 秒 |
| `slow_down` | 同上，但速度不超过 `slow_speed_mps` | 0.1 m/s，0.5 秒 |
| `turn_left` / `turn_right` | `yaw_rate_rps`；`duration_s` 或 `angle_rad` 二选一 | 0.5 rad/s，0.5 秒 |
| `safe_stop` | 空参数 | 下发停止 |
| `observe_again` | 空参数 | 先停止，回传需要重新观测 |
| `ask_human` | 空参数 | 先停止，回传需要人工处理 |

默认单次最长 2 秒。向左转的角速度为正，向右为负。Go2 约每 50 ms 发一次速度，
Go1 约每 2 ms 发一次，Python 调度不保证硬实时频率。每轮发运动命令前检查新鲜观测
与 Safety Guard；任何拒绝或 SDK 异常都会尝试停止。距离/角度仅换算速度控制时长，
不是定位闭环，也不保证走过精确距离。

`slow_down` 的定义是执行一段受限的低速前行，不是异步修改之前仍在执行的动作。
一个适配器同一时间只执行一个动作。急停可以从其他线程调用，会锁存并阻止后续运动；
现场确认恢复后显式调用 `reset_emergency_stop()`，恢复时仍要求有效观测。

回执 `success` 只表示命令发送流程正常完成，`physical_completion_verified=False`
明确表示没有核验实际位移/到达。Go2 Move 的返回码不证明运动已发生；Go1 UDP 不提供
送达确认，停止发送 10 个零速包也不能保证断网情况下真机已停。软件急停可能等待当前
SDK 调用结束，不替代遥控器或底盘的独立急停/看门狗。

`observe_again` / `ask_human` 返回 `aborted`，并在 `telemetry.requested_action` 中标记
请求，主应用据此触发感知或人工交互。不要将其标为“已经完成观察/获得人工答复”。

当 `SafeExecutionBridge` 返回 `status="executed"` 时，仍须查看 `receipt.status` 判断
成功、失败或中止；顶层状态表示已经调用执行层。停止异常也会保留为失败回执，不会伪装
为停止成功。L0 记录包含编号、型号、指令速度、计划时长及停止错误（如有）。

## 离线验证

```bash
python3 -m unittest test_unitree_adapter test_execution_bridge test_safety_guard -v
```

这些测试使用假的 SDK 客户端，不导入真机 SDK，不连接网络。

实现核对来源：
[Go2 SportClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py)、
[Go1 高层行走例程](https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/example_py/example_walk.py)。

## 连接前只读预检工具

`robot_preflight.py` 检查本机配置、SDK 是否可加载，以及传感器 JSON 快照是否持续更新。
工具不实例化机器人客户端，不初始化 DDS，不打开 Go1 UDP，不发送停止或运动命令。
SDK 在有 10 秒超时的独立进程中加载，避免扩展加载故障卡住主检查。

先复制对应配置模板并填写：

- `configs/go1.preflight.example.json`：填写真实 Go1 IP、SDK `.so` 的绝对路径，以及观测文件路径。
- `configs/go2-1.preflight.example.json`：填写真实网卡名，以及观测文件路径。

模板里的 `null` 是待填写项目，直接检查模板会失败，不会自动选择网卡或猜测 IP。
相对 `observation_file` 路径以配置文件所在目录为基准。

```bash
# 示例：在项目根目录，先复制并编辑副本
cp configs/go2-1.preflight.example.json configs/go2-1.preflight.local.json

# 填好配置、启动你们的传感器快照导出后，持续检查 5 秒
python3 robot_preflight.py --config configs/go2-1.preflight.local.json --duration 5 --output output/go2-1-preflight.json

# 没装 SDK 的电脑可以只检查配置与观测，结果明确标记 incomplete
python3 robot_preflight.py --config configs/go2-1.preflight.local.json --skip-sdk
```

`--output` 的父目录需已存在；省略时报告直接输出到终端。不会允许报告覆盖当前配置
或观测文件。退出码 0 表示所选本机预检项目通过，1 表示检查失败或不完整，2 表示
命令参数或报告写入错误。

传感器进程每次产生同步快照时调用：

```python
from robot_preflight import write_observation_snapshot

# observation 是你们感知层生成的 RobotObservation，包含真实数据和采样时间。
write_observation_snapshot("/实际共享目录/go2-1-state.json", observation)
```

该函数原子替换 JSON，避免读取到半份文件；不重写采样时间。Go1 使用另一份观测文件。
应从收到真实新数据的回调中调用，不能定时给旧数据换时间戳。快照格式与上述
`RobotObservation` 相同，顶层为 `robot`、`environment`、`timestamp`，可选
`frame_id` 和 `metadata`。程序读取的是传感器进程已经导出的数据；传感器 SDK
订阅本身仍需你们接入。

报告检查：

- 型号/编号、IP/端口格式、Go2 本地网卡是否存在、SDK 导入与所需 API 是否存在。
- 观测结构与数值、设备一致性、采样时间是否过期/超前/倒退。
- 检查窗口内至少出现两个时间戳不同的新样本；即使文件一直能读，时间戳不变也失败。
- 文件缺失、采集停止导致数据过期、非法 JSON 都记录失败；最多保留 20 条采样错误。
- `observed_update_hz` 是轮询采到的源时间戳变化速率，可能漏掉中间帧，不等于传感器实际频率。

报告始终标记 `robot_connectivity="not_verified"`、`motion_ready=false`。
即使 `status="pass"`，也只说明本机与观测缓存检查通过；不证明 IP 可达、DDS 已连上
指定机器人、数据确实来自真机，或控制权/现场急停可用。工具不会主动探测这些事项。
模拟或回放数据也可能通过格式检查，数据来源需现场核对。

工具离线测试：

```bash
python3 -m unittest test_robot_preflight test_unitree_adapter test_execution_bridge test_safety_guard
```
