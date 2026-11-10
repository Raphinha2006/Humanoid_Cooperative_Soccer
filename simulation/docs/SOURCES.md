# 来源、版本与选型

核查日期：2026-11-08（Asia/Shanghai）。

| 资料 | 本项目用途 | 版本与边界 |
|---|---|---|
| [Booster 官方 booster_assets](https://github.com/BoosterRobotics/booster_assets) | K1_22dof.xml、STL、惯性、关节范围、电机力矩 | 固定 commit `12a97516f57469078707afb587a5f84c1cb110d1`，BSD-3-Clause，下载时保留 LICENSE 并核验 Git blob SHA-1 |
| [HSL 官方 environment.tex](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/06c44d4b0ea2fd3cd4fdcc2a7cb907f6ace400dc/rules/environment.tex) | M-Field 尺寸、草高、球门范围、中型组允许用球 | 固定 commit `06c44d4b0ea2fd3cd4fdcc2a7cb907f6ace400dc`；表格明确为示例尺寸，比赛具体场地需按当届公告确认 |
| [HSL 仓库声明](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/06c44d4b0ea2fd3cd4fdcc2a7cb907f6ace400dc/README.md) | 规则状态 | 仓库仍声明 2026 内容在开发中、非规范发布，不能称为最终赛事认证场地 |
| [MuJoCo 3.3.7 接触模型](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#contact-parameters) | 显式接触 pair、摩擦维数与混合规则、solref/solimp | 引擎固定为 3.3.7；使用 pair 避免误以为修改地面 geom 后会覆盖球体摩擦 |
| [NUbots/NUSim](https://github.com/NUbots/NUSim) | 开源 K1 足球仿真选型参考 | 有 DDS、相机、裁判接入；README 的多机器人模式说明额外机器人为 PD 站立，不能直接当作六机器人自主传球策略。本项目未复制其代码 |
| [Booster 官方 robocup_demo](https://github.com/BoosterRobotics/robocup_demo) | 当前工程的视觉/行为/裁判背景 | 不是完整 MuJoCo 动力学环境，也未提供本项目所用踢球策略 |
| [booster_mjlab](https://intelligentroboticslab.github.io/booster_mjlab/) | 已训练 AMP 速度行走模型及配套 MJCF | Git commit `b6d7639c69410a247f74ba61d0034bfc277a3922`；使用网站实际运行的 `policy.bin`、`scene.json`、`k1_web.xml`，不是自行训练或猜测权重 |
| [Kyk0/sonic-k1](https://github.com/Kyk0/sonic-k1) | K1 动作跟踪的 ONNX 推理、关节映射、配套模型、参考动作 | Git commit `47920074916699148abc722f94193d2859739755`；代码/资产各自许可见上游 LICENSE/NOTICE |
| [Kyry10/sonic-k1 权重](https://huggingface.co/Kyry10/sonic-k1) | `policy_g1.onnx` 和 `k1_plant.json` | HF commit `1b8092be48e99a1b3c202e5ffa807536a3abeaec`，按 SHA-256 核验；权重 CC BY-NC 4.0 |
| [MSL 现行规则页](https://msl.robocup.org/rules) | 区分联赛 | MSL 2026 沿用 2025 v26.0，正式场地 22×14 m、5号球；用户确认人形 K1/T1 后，本项目使用 HSL，而非 MSL |

场地生成器、固定骨盆控制轨迹、传球判定和实验工具在 `soccer_sim/`。固定骨盆基线不调用预训练模型；新增的 `soccer-policy` 明确调用上述已发布权重及 SONIC 仓库附带参考动作。它们没有包含本项目的足球决策，详细接口与许可边界见 [POLICIES.md](POLICIES.md)。

HSL 中型组允许 3 或 4 号球；本例选 3 号球。规则源码没有给出这一具体球的半径、质量、惯量、气压或草地摩擦实测值，因此这些项目标为工程初值。实际用球应称重、测周长，再用 r=C/(2π) 设置半径。
