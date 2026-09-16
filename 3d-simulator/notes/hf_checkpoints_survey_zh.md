# HF locomotion checkpoint 调研（2026-09-02）

## 结论

- **官方 Spot/H1 flat policy 不必替换**：NVIDIA 官方 legged policy 唯一发布渠道就是
  Isaac Sim 官方 asset S3（Spot/H1/ANYmal-C 三个，我们已用其二）；HF 上 nvidia /
  unitreerobotics / isaac-lab org 均无 locomotion checkpoint，LeRobot 无 locomotion。
- **若加第四形态 Go2（可选，≈半天）**：`yanshi-robotics/yanshi-unitree-go2-default`
  —— Isaac Lab 原版 Isaac-Velocity-Flat-Unitree-Go2-v0 导出（rsl_rl .pt + onnx +
  完整 env.yaml），48 维 obs 与 PolicyController 拼接同构，[vx,vy,wz] 指令，README 称 MIT。
  或自己在 Isaac Lab 跑 1500 iter（约 20 分钟 GPU 时）。
- **若 H1 不够稳（备胎，1–2 天适配）**：unitree_rl_gym（GitHub, BSD-3）
  `deploy/pre_train/h1/motion.pt` —— 唯一实机验证过的 H1 velocity policy（torch.jit,
  41 维 obs），需 obs 缩放 + sin/cos 步态相位时钟 + MuJoCo→USD 关节序映射。
  （注：M3 标定显示官方 H1 policy 在 world.reset 复位下零摔倒，目前不需要。）

## 其他备选（记录）

- `diasAiMaster/unitree-go2-velocity-flat`：BSD-3，deploy.yaml 文档最全（45 维无
  lin_vel），MuJoCo 训练 sim2sim，1–2 天。
- `Kyu3224/quadruped-locomotion-policy`：go1/go2/spot/anymal flat+rough 全家桶，
  **无 license**，只能内部实验不能分发。
- `josabb/G1-humanoid-6dof-hands-locomotion-rl`：Apache-2.0，G1 形态，MuJoCo 关节序，2–3 天。
- rough 系 policy（235 维含 height scan）：Isaac Sim 侧需自建 187 维高度扫描拼接，3–5 天，
  暂无需求。
- 不可用：rambo-go2（CC-BY-NC）、AMP/目标条件类（非 velocity 指令）、绑 gym env 的
  SB3/skrl 导出、零文档大 zip。
