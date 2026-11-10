# G1 原生策略验证参数

本配置用于验证两机器人传接和训练架构，采用 [Joga 的公开部署资源](https://joga.csail.mit.edu/demo/)，不是对 HSL 比赛球动力学的标定。原 K1 场地规则、14×9 m M-Field、球门和可改尺寸仍见 [PARAMETERS.md](PARAMETERS.md)。G1 验证场景只有场地边线和两台参与者，没有完整六机器人比赛逻辑。

## 来源和默认值

下载入口 `tools/fetch_joga.py`，逐文件 SHA-256 在 `configs/joga_sources.json`。机器人是带双自由度颈部的 G1，31 个受控关节，每台模型质量合计 33.9277 kg；这些是部署模型数值，不作为宇树全部 G1 型号的硬件规格。身体尺寸、各连杆质量和惯量保留原 MJCF 网格和模型。

| 参数 | 原生默认 | 修改入口 |
|---|---|---|
| 物理 / 技能频率 | 200 / 50 Hz | `physics_overrides.option.timestep` / `control_dt` |
| 数值积分 | implicitfast、Newton、pyramidal、10 次迭代、容差 1e-8 | `physics_overrides.option`，枚举用 MuJoCo 整数值 |
| 重力 | [0,0,-9.81] m/s² | `physics_overrides.option.gravity` |
| 球半径、质量 | 0.11 m、0.43 kg | `physics_overrides.geom.soccerball_geom.size` / `body.soccerball.mass` |
| 球主惯量 | 每轴 0.00346867 kg·m²，约 2/3·m·r² | `physics_overrides.body.soccerball.inertia` |
| 地面 geom 摩擦 | [1,0.005,0.0001] | `physics_overrides.geom.floor.friction` |
| 鞋底 geom 摩擦 | [1,0.005,0.0001] | 逐个 `A1/left_foot_collision` 等的 `friction` |
| 球 geom 摩擦 | [0.75,0.05,0.02] | `physics_overrides.geom.soccerball_geom.friction` |
| 球 geom solref | [0.055,0.18] | 同上 `solref` |
| 脚球显式 pair 摩擦 | [0.75,0.75,0.05,0.02,0.02] | `physics_overrides.foot_ball_pairs.friction` |
| 脚球 pair solref / solimp | [0.055,0.09] / [0.85,0.95,0.001,0.5,2] | 同上 `solref` / `solimp` |
| 非根关节 armature | 0.01 | `physics_overrides.joint.<完整关节名>.armature` |
| PD 和电机限制 | 原生 31 维 kp、kd、max_torque、max_vel | `assets/joga/skills.json`；或 `actuation_scale` 各标量/31 维缩放 |
| 初始机器人位置 | 发送方 [5,0]，接收方 [0,0] m，相向 | `sender_xy` / `receiver_xy` |
| 接收方初态扰动 | ±[0.15,0.10] m | `receiver_xy_noise` |
| 发送方球位置 | 局部 [0.45,0] m，扰动 ±[0.02,0.02] m | `ball_xy` / `ball_xy_noise` |
| 传球指令 | 4.75 m/s，方向补偿 +17° | `kick_speed` / `pass_direction_bias_deg`，均为策略指令，不是直接设球速 |
| 接球腿部残差 | 12 维，最大动作 0.6；膝最大修正 0.072 rad | `receiver_joint_residual_scale` / `action_limit` |
| 严格接球 | 根部水平距 <0.55 m，球速 <0.5 m/s，连续 0.8 s，面对面 ≤30° | `capture_radius/speed/hold` / `facing_tolerance_deg` |

geom 的三项摩擦分别是滑动（无量纲）、扭转（m）、滚动（m）；pair 的五项分别是滑动 x/y、扭转、滚动 x/y。显式脚球 pair 优先于 geom，因此只改球 geom 摩擦不会改变脚球 pair 摩擦。球地接触还受 geom 的 priority、condim、solref 和 solimp 选择影响。这里的“摩擦”不是单个统一的草地摩擦系数，均未做真实草地实验标定。

## 如何修改

把 `configs/joga_physics_overrides.example.yaml` 中需要的键放进复制的 `configs/supermimic_g1_receive.yaml`，用 `direct_train --config 新配置.yaml` 运行。支持按模型完整名称修改 geom、body 和 joint 属性。修改质量和半径时应同步修改惯量；修改物理步长时必须整除 `control_dt`。原生策略依赖 50 Hz 的历史接口，应保持 `control_dt=0.02`。修改接触和增益后必须重新验证、采样和训练，已有 WM 不会自动适应。WM 的三个物理条件输入仅含地面滑动摩擦、球质量、半径，未覆盖完整接触参数空间。

原生部署屏蔽机器人间和自身碰撞；本次沿用这一设置。模型浮动基座，运行时不使用外部支撑力；只在 reset 中放球，传球开始后不覆盖球速度或位置。Joga 网页队友演示的独立仿真间镜像/修正足球方式没有移入本场景。

每次训练保存实际 `settings.yaml`、`runtime_scene.xml`、`runtime_actuation.json` 和代码快照；数值核对文件为 `research/source_audit/g1_native_parameters_mj312.json`。本次服务器独立环境为 `.venv-joga312`，借用现有只读 Torch 等依赖，未升级原 `.venv-wm`。

新机器可安装 `requirements-g1-validation.txt`，通过 `PYTHONPATH=.` 运行；原 `pyproject.toml` 仍为旧 K1 实验固定 MuJoCo 3.3.7。
