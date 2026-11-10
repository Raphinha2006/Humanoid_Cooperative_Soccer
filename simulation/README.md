# 仿真模块

当前研究入口、进展与 G1 GPU 安装见[根 README](../README.md)。本目录保留 K1 场地原型、预训练策略接口和历次世界模型实现；旧文档中的结果属于历史实验。

## K1 场地与接触基线

与 G1 分开建立 Python 3.12 环境：

```bash
cd simulation
python3.12 -m venv .venv-k1
.venv-k1/bin/pip install -e '.[test,policy]'
.venv-k1/bin/python tools/fetch_assets.py
.venv-k1/bin/soccer-sim build --out runs/field
MUJOCO_GL=egl .venv-k1/bin/soccer-sim pass --out runs/contact_demo --video
.venv-k1/bin/soccer-sim probe --out runs/calibration
```

`build` 输出 MJCF、解析配置、连杆/关节参数和来源；`probe` 输出球滚动及落球数据。`pass` 是固定骨盆的接触基线，不能用它证明自由行走、平衡或稳定接球。

场地/球门、尺寸来源、球质量和惯量、摩擦单位、控制参数及修改接口见 [PARAMETERS](docs/PARAMETERS.md)。默认 14 × 9 m、六台 K1；没有六机器人自主比赛逻辑。草地是等效接触平面，网是刚性替代物。

```bash
.venv-k1/bin/soccer-sim pass --out runs/variant \
  --set 'contacts.ball_turf.friction=[0.75,0.75,0.002,0.001,0.001]' \
  --set ball.mass=0.34
```

## 外部 K1 策略

```bash
.venv-k1/bin/python tools/fetch_policies.py
OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl .venv-k1/bin/soccer-policy \
  --kind booster_mjlab --scenario pass --config configs/hsl_m_pretrained_pass.yaml \
  --seconds 12 --video --out runs/k1_policy
```

Booster_mjlab、SONIC-K1 和独立足球技能是不同接口，不能把一种控制权重当成另一种机器人策略。来源、下载版本、输入输出与已有失败边界见 [POLICIES](docs/POLICIES.md)。

## G1 当前研究

- `soccer_sim/gpu_receive/`：并行真实交互、物理 WM、训练、审计与控制器一致性。
- `configs/supermimic_g1_receive.yaml`：当前低层接口与任务配置。
- [G1_PARAMETERS](docs/G1_PARAMETERS.md)：G1 模型、接触和覆盖接口。
- [当前协议](research/cooperative_handoff/experiments/00_control_consistency/protocol.md)：对照条件与预算。
- [历史文档](docs/)：旧版本方法与结果；以根 README 和当前协议判断正在使用的架构。

策略运行中不重设球位置或球速；初态只由 reset 设置。当前传球方技能冻结，接收方学习残差。双方共同学习、角色交换和稳定连续互传仍是待完成任务。
