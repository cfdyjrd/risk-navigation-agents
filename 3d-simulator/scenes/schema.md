# 场景 JSON schema（v1，设计文档 §7.1）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 场景 ID（E01…E20 或冒烟场景） |
| `rooms` | {zone: {aabb, restricted?}} | zone AABB `[x0,y0,x1,y1]`（米）；`restricted: true` 铺红色地贴（表现层，判定在 host 权威侧） |
| `doors` | [{between, center, width?}] | 门洞：两 zone 名、门中点（必须落在共享墙线上）、宽度默认 1.2（narrow 0.7） |
| `objects` | [{id, zone, offset?}] | 0.2m DynamicCuboid，锚在 zone 中心 + 偏移 |
| `npcs` | [{id, pos, role, script?}] | 胶囊 NPC；`script` 为路径点列表（M7 脚本驱动：行人往返/安保拦截） |
| `spawn` | {zone, yaw_deg} | 机器人出生 zone（中心）与朝向 |
| `goal` | 任意 | host 侧任务定义原样透传（planner/judge 消费，Isaac 进程不读） |
| `macro_budget` | int | 每次 goto 的宏预算（M7 起 = 每场景 oracle 实测 ×1.8，§3.5） |

## 几何规则（scene_builder.compute_walls）

- 墙 = 所有房间 AABB 边界按共线区间**并集合并**（共墙只建一次）再减去门洞区间；
  墙高 0.5m（俯视定位防遮挡 §4.5）、厚 0.1m。
- 定标色点（红/绿/蓝/黄）自动放在全场景包围盒四角内缩 0.5m，TopDownLocalizer
  开机标定单应用（§4.1）。
- 多层拓扑在 2D→3D 编译期压平为单层（portal→连接走廊），进 M7 与
  `risk-navigation-agents/2d-simulator` 的 `spatial_layout.layout_world` 对接。
