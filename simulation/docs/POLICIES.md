> 历史实验文档。当前研究状态以[根 README](../../README.md)和[实验协议](../research/cooperative_handoff/experiments/00_control_consistency/protocol.md)为准。

# 已发布 K1 策略的接入与验证

核查与实测日期：2026-11-08。服务器：`ssh <server-alias>`，目录 `<project-root>`。本地代码：`simulation/`。

本文记录早期 Booster/SONIC 六机器人测试。2026-11-09 新增的面对面传接球与 WM 实验使用独立 B-Human 步态/软踢球网络，当前入口和结果见 [WORLD_MODEL.md](WORLD_MODEL.md)。

## 结论

两套现有权重均已接入同一个 HSL M-Field 六机器人场景，并完成服务器复测。所有 K1 都保留根部六自由度，无固定骨盆、无扶持力、无逐帧重设关节位置。网络以50 Hz闭环推理，MuJoCo以1 kHz积分。SONIC共享一份ONNX会话，六台各自保存独立观测历史。

它们能用作底层运动控制器，不能直接充当“看球—选队友—传球”的决策模型。当前推荐以 booster_mjlab 作为接近球/转向/站立的基础，SONIC 用于测试全身动作跟踪。没有训练或假冒一套已学会足球的策略。

## 来源与精确接口

| 项目 | 实际下载的模型 | 输入 → 输出 | 版本 |
|---|---|---|---|
| [booster_mjlab](https://intelligentroboticslab.github.io/booster_mjlab/) / [代码](https://github.com/IntelligentRoboticsLab/booster_mjlab) | 网站演示的 `policy.bin`、`scene.json`、`k1_web.xml` | 75维观测 → 22关节动作；速度指令 vx、vy、wz | Git `b6d7639c69410a247f74ba61d0034bfc277a3922` |
| [SONIC-K1](https://github.com/Kyk0/sonic-k1) / [权重](https://huggingface.co/Kyry10/sonic-k1) | 融合推理 `policy_g1.onnx`、配套 `k1_plant.json`、原仓库 K1 MJCF | 1220维：未来参考动作、相对朝向、本体感知历史 → 22关节动作 | Git `47920074916699148abc722f94193d2859739755`；HF `1b8092be48e99a1b3c202e5ffa807536a3abeaec` |

Booster观测顺序为角速度3、机体坐标重力3、相对默认关节角22、关节速度22、上一动作22、速度指令3。按照 `scene.json` 的均值/标准差及75→512→256→128→22、ELU网络逐层执行；不是用随机权重替代。输出转换为默认角+逐关节scale×action，原MJCF的仿射执行器实现原PD，避免重复叠加PD。默认头部kp/kd=4/0.25、手臂10/1、髋膝80/4、踝50/2。

SONIC直接复用固定版本 `deploy_k1_booster.py` 的 Plant、Reference、Policy 推理实现，未实例化硬件后端。1220维由未来10帧关节参考440、朝向参考60、本体感知10帧历史720组成；历史按观测项展开。使用上游 IsaacLab ↔ MuJoCo 关节排列映射、默认角、action scale、armature、PD和力矩限制。输出动作限幅±20，PD在每个物理步施加并按plant限幅。ONNX文件名虽带g1，实际下载的是该仓库发布的K1权重，非Unitree G1策略。

SONIC依赖参考动作，不接受 vx/vy/wz 直接导航。测试使用仓库附带 `squat.npz` 和 `crouch_walk.npz`；仅初始状态从第一帧设置，此后由神经网络输出目标角并积分动力学，不是把NPZ逐帧写入qpos。参考结束会截断仿真，避免循环跳接制造假象。

适配代码：`soccer_sim/pretrained.py`、`policy_scene.py`、`policy_runner.py`。映射按名称处理，每台使用独立前缀 A1/…B3/。两套策略分别保留自己的配套模型，避免把不同关节定义/执行器/默认姿态当成同一套控制接口。球场、球、空气和接触参数保持共用。Booster网站原本用外部GLB显示，本地录像直接显示其MJCF的第3组碰撞外形；这是一套较粗的外形，物理模型和网站导出一致。

默认模型质量也有差别：Booster网站配套模型合计 **19.7969 kg**（手部末端球形body的质量由geom自动推导），SONIC及基础官方资产为 **19.666 kg**。保留各自原模型，未为统一数字擅改预训练模型的质量。`robot_parameters.json` 和完整 `scene.xml` 可审查实际参数。

## 服务器测试结果

运行 `bash tools/run_policy_checks.sh` 生成 `runs/policy_delivery/`。以下均为六台浮动基座，统一工程接触初值；没有使用 `--native-contacts`。

| 场景 | 仿真时长 | 观测结果 | 跌倒 / MuJoCo警告 |
|---|---:|---|---|
| Booster站立 | 8 s | A1水平漂移约3 cm | 0 / 0 |
| Booster前进，vx=0.4 | 6 s | A1沿x前进约2.315 m，其他五台站立 | 0 / 0 |
| Booster行走碰球 | 6 s | A1前进约2.302 m，发生脚球接触 | 0 / 0 |
| Booster A1→A2 | 12 s | A1、A2先后碰球；约7.10 s满足0.3 s接球窗口；随后球弹离，最终稳定传接判失败 | 0 / 0 |
| SONIC下蹲 | 7.88 s | 六台完成当前参考动作，无跌倒 | 0 / 0 |
| SONIC蹲走 | 4.82 s | A1位移约[-2.217,-0.631] m，六台跟踪同一参考 | 0 / 0 |

11项测试通过，包括球体惯量、接触传球/未踢球反例、滚动摩擦影响、越界、自由基座、参数继承、修改质量/阻尼、保存场景重载和SONIC各机器人历史隔离。`summary.json`、各场景 `result.json`、轨迹、视频、版本清单保留原始结果。有限时长的无跌倒测试不等于任意指令、随机初态或整场比赛都可靠。

跌倒定义：根高度<0.25 m，或躯干竖直方向余弦<0.5持续超过0.4 s。接球事件要求有>0.01 N的发送方和接收方接触、先后顺序正确、球水平移动>0.5 m，并在接收方0.45 m内以<0.25 m/s保持0.3 s。`receive_event_success`记录曾满足此条件；`stable_receive_at_end`记录结尾仍满足；`pass_success`要求两者为真且无人跌倒。

本次实验的上层控制只有：A1接收[0.4,0,0]速度指令，球移动0.7 m后切换零速，其余机器人接收零速。接收方初始横向位置由−0.0962改为−0.0262 m以对齐该步态出球方向，已完整写入 `configs/hsl_m_pretrained_pass.yaml`。没有修改草地摩擦来制造接球成功。该固定布置仍不能稳定停球，不提供任意方向的传球泛化能力。

## 复现与修改

```bash
cd <project-root>
.venv/bin/pip install -e '.[test,policy]'
.venv/bin/python tools/fetch_policies.py
bash tools/run_policy_checks.sh
```

下载脚本固定Git/HF版本，GitHub文件校验Git blob SHA-1，ONNX及plant校验SHA-256。ONNX为54,794,351字节，SHA-256 `5166065099985c0de4fe791a489e7177d3a8574f7aaa3c5dce086d0013c30781`。缺失才下载，不读取`.pt` pickle权重；不需要安装训练栈、IsaacLab或Booster SDK。当前ONNX在CPU推理，4090用于EGL渲染也足够；没有在本任务中重新训练。

单独运行与换参考：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/soccer-policy --kind booster_mjlab \
  --scenario walk --command 0.3 0 0.2 --seconds 6 --out runs/turn

OPENBLAS_NUM_THREADS=1 .venv/bin/soccer-policy --kind sonic_k1 \
  --scenario motion --seconds 5 \
  --motion policies/sonic_k1/examples/k1/motions/crouch_walk.npz --out runs/crouch
```

视频加 `--video`；Linux离屏使用 `MUJOCO_GL=egl`。新机器先按README安装并抓取官方资产。脚本使用根目录下的`.venv`；本地另一个环境可通过 `PYTHON=/absolute/path/to/python bash tools/run_policy_checks.sh` 指定。

每次输出 `scene.xml`（包含运行时执行器设置和`policy_initial`关键帧）、`initial_state.npz`、`resolved_config.yaml`、逐连杆/关节 `robot_parameters.json`、`provenance.json`、`source.json`、`result.json`、50 Hz `trajectory.csv`；录制时另有 `policy.mp4`。复制服务器场景到本地不能直接加载其中服务器的绝对网格路径，应在本机重建。重新加载初态需显式 `mj_resetDataKeyframe(model,data,model.key('policy_initial').id)`。

物理参数和控制倍率修改入口见 [PARAMETERS.md](PARAMETERS.md)“预训练控制参数”。固定骨盆的 `passing.backswing/strike/receiver_pose` 不应用于预训练模式。模型质量、惯量、力矩限幅变化可能使既有权重失稳，应保留原始测试作为对照。

## 许可与下一层接口

SONIC仓库代码使用Apache-2.0，Booster来源模型保留BSD归属；具体文件归属见其LICENSE/NOTICE。当前下载的SONIC权重及参考动作标明 **CC BY-NC 4.0，限非商业用途**，不是无限制商用权重。Booster网站演示资产保留来源清单；在固定版本中未发现仓库根LICENSE，部署到产品或再分发前需要确认其授权。下载的资产和权重均在被git忽略的目录，未作为本项目自有权重重新发布。

后续可把比赛状态送入上层控制器，上层输出速度命令或选择参考技能；底层输出22个目标角/力矩。要实现稳定传球，还需补充球相对位置闭环、瞄准、专门的踢球/停球技能与步态切换，再测试不同球位、距离及接触参数。当前两个网络的输入不包含足球/队友/球门状态，因此不能仅靠装入权重获得足球决策。
