> 历史实验文档。当前研究状态以[根 README](../../README.md)和[实验协议](../research/cooperative_handoff/experiments/00_control_consistency/protocol.md)为准。

# G1 当前 GPU 训练架构

这是 `soccer_sim/gpu_receive/` 的实际入口，区别于保留的 30 轮 CPU 实验。先验收单次面对面 A→B 接球；连续互传尚未完成。

```mermaid
flowchart TB
  subgraph REAL[真实物理采样 · GPU1 · 4096 个并行世界]
    SIM[两台 31 关节 G1 + 同一个自由足球]
    S[160 维根位姿、速度、关节与球状态]
    N[冻结的 Joga 原生策略<br/>A：步态与传球 / B：接球]
    PI[可训练接球策略<br/>12 个腿关节残差，最大 ±0.15 rad]
    Q[原生目标角 + 残差<br/>电机速度与力矩限幅 → 62 维实际关节目标]
    PD[200 Hz PD 与 MuJoCo Warp 积分<br/>50 Hz 策略]
    SIM --> S
    S --> N
    S --> PI
    N --> Q
    PI --> Q
    Q --> PD --> SIM
  end
  S --> DATA[真实状态、实际关节目标、后继状态<br/>另存确定性原生技能历史]
  Q --> DATA
  DATA --> FIT[每轮采样32步，再做8步自回归拟合<br/>batch 2048 / 2048×2048×2048 MLP]
  subgraph LEARN[SuperMimic 思路的直接策略学习]
    FIT --> WM[冻结 WM 参数，保留输入梯度]
    DATA --> INIT[真实状态与原生技能历史初始化]
    INIT --> ROLL[32 步想象，0.64 秒]
    WM --> ROLL
    ROLL --> FK[预测根部与关节 → 可微相机链 FK<br/>显式更新接球策略观测与历史]
    FK --> FROZEN[冻结原生接球网络 + 当前腿部残差]
    FROZEN --> ROLL
    ROLL --> OBJ[近球、减速、稳定和平滑奖励<br/>λ-return critic 的末端价值]
    OBJ --> BPTT[BPTT 更新接球策略<br/>有效 batch 2048，微批 128]
    BPTT --> PI
  end
  PI --> EVAL[独立环境真实仿真验收，不重置训练轨迹<br/>有序触球 → 相向停球 ≥0.8 秒 → 最终仍保持]
  EVAL --> COMPARE[冻结原生 B0 / 同接口 PPO B1 / WM 策略 M<br/>开发集选模，独立初态最终测试，CPU 后端确认]
```

原生技能历史由控制器维护，不是 WM 的预测目标。WM 接收真正执行的关节目标；没有规划教师或 OPD。发送方在想象窗口内保持当前关节目标，真实采样时继续执行原生策略，这是当前近似。

复用的是固定版本社区 SuperMimic 的 ActorCritic、RunningNorm、FK 与多步模型/BPTT/critic 思路；不声称完整官方复现。策略使用仿真真值，视觉闭环、接到球后的角色交换、双方反复互传均是后续独立验收项。

`prediction.py` 另保存不进入 replay 的原生技能固定轨迹，以已知真实关节目标比较 WM 和恒速预测器，报告球、根部、身体和双脚的误差。该测试用于分离预测能力与控制能力，不能替代真实接球成功率。

```mermaid
stateDiagram-v2
  [*] --> A传球B接球
  A传球B接球 --> B稳定持球: 有序真实触球、面对面、低球速并保持
  B稳定持球 --> B传球A接球: 后续：技能交接和角色互换
  B传球A接球 --> A稳定持球: 后续：同样的严格持球条件
  A稳定持球 --> A传球B接球: 后续：保持球的连续物理状态
```

第二张图是循环传接的目标状态机，当前训练仅覆盖第一条传接及持球判断；仿真 reset 不算完成角色交换。
