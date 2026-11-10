# 参数表与修改接口

全部长度 m、质量 kg、时间 s、角度 rad；惯量 kg·m²，力矩 N·m。主入口 `configs/hsl_m_3v3.yaml`。所有参数均可在复制的 YAML 中修改，已有字段也可用重复的 `--set dotted.key=YAML值` 覆盖。每次运行保存完整 resolved_config，避免视频与参数不对应。

新增 G1 原生策略验证使用独立的 Joga 参数配置，见 [G1_PARAMETERS.md](G1_PARAMETERS.md)。不要把其 0.11 m / 0.43 kg 球与本页 K1 场地的 size3 工程球混为同一配置。

面对面 WM 实验默认入口为 `configs/wm_pass_facing_gentle.yaml`，转向、接球保持、教师和训练参数见 [WORLD_MODEL.md](WORLD_MODEL.md)。它沿用本页物理参数，但 kp/kd 从开源策略接口加载，`robot.kp/kd` 不控制这些预训练技能；使用 `skill_kp_scale` / `skill_kd_scale` 调整，实际数组导出到 `runtime_actuation.json`。

## 场地与球门

来源：[HSL environment.tex 固定版本](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules/blob/06c44d4b0ea2fd3cd4fdcc2a7cb907f6ace400dc/rules/environment.tex) 的 `Field surface`、`Dimensions`、`Goals`、`The Ball`。当前是官方仓库开发源码，不是赛事最终发布。

| YAML 参数 | 当前值 | 规则含义/范围 |
|---|---:|---|
| field.length / width | 14 / 9 | M-Field 示例；允许约长9–14、宽6–9 |
| field.line_width | 0.05 | M 示例0.05；一般标线0.05–0.12 |
| field.goal_area_depth / width | 1 / 4 | M 示例 |
| field.penalty_area_depth / width | 3 / 6 | M 示例 |
| field.penalty_mark_distance | 2 | 从球门线外边到点中心 |
| field.center_circle_radius | 1.5 | 示例直径3，取白色圆环外缘半径 |
| field.corner_arc_radius | 0.5 | M 角球弧 |
| field.center_mark_diameter / penalty_mark_diameter | 0.1 / 0.1 | M 示例点尺寸 |
| field.runoff | 1 | 至少1的边界缓冲区 |
| field.turf_pile_height | 0.025 | 中型组草高约0.020–0.030；记录用途，未逐根建草 |
| goal.width | 2.4 | 净开口，Middle允许2.4–2.6 |
| goal.height | 1.6 | 横梁底到地面，Middle允许1.5–1.9 |
| goal.depth | 0.8 | Middle允许0.7–1.2 |
| goal.post_diameter | 0.1 | 球柱/横梁宽度0.07–0.12，本例圆截面 |
| goal.net_collision / net_thickness | true / 0.015 | 刚性替代网板碰撞及厚度，工程设置 |
| field.safety_wall | false | HSL并未要求MSL式挡墙，默认不加 |
| field.wall_height / wall_thickness | 0.12 / 0.05 | 可选挡墙工程参数，仅开关true时生效 |

坐标：原点在场地中心，x 沿长边、y 沿宽边、z 向上。地表接触面z=0。边界外缘为x=±7、y=±4.5；边线中心分别为±6.975、±4.475。球门柱中心位于球门线中心，横向为±(净宽/2+柱半径)，横梁中心高为净高+柱半径。

标线是无碰撞视觉网格，不会在地面形成凸起。明暗草纹是视觉装饰；所有区域使用同一接触面参数。没有自动生成技术席、观众、照度测量或完整竞赛裁判逻辑。

## 球与空气

| 参数 | 当前值 | 解释 |
|---|---:|---|
| ball.size_label | size_3 | HSL Middle允许size3或size4，本例选择size3 |
| ball.radius | 0.095 | 直径0.19、周长约0.597；工程名义值，需要实测比赛球 |
| ball.mass | 0.32 | 工程初值，需要称重 |
| ball.inertia_factor | 2/3 | 薄球壳近似 I=k·m·r²；单轴约0.0019253 kg·m² |
| ball.pressure_kpa | 80 | 记录气压初值，不直接施加压力；通过solref/solimp近似接触变形 |
| ball.initial_position | [-1.72,0,0.096] | 世界坐标，仅reset写入 |
| ball.initial_velocity | [0,0,0] | 世界坐标线速度，仅reset写入 |
| physics.air_density | 1.225 | 工程初值，kg/m³ |
| physics.air_drag_coefficient | 0.47 | 球体阻力近似，未测足球表面/速度依赖 |

空气阻力 F=−½ρCdπr²|v|v；不模拟马格努斯力、复杂流场、球壳有限元或充气变化。改球半径后也需检查初始球高和踢球姿态。惯量由质量、半径和inertia_factor自动重算。

## 接触动力学：没有统一官方摩擦系数

这些参数是**未标定的工程初值**，不能称为官方标准。真实值取决于球皮、球压、草高、填充物、鞋底、温湿度和速度。

本项目采用MuJoCo显式 `contact/pair`。数组顺序为 `[滑动x, 滑动y, 绕法线扭转, 滚动x, 滚动y]`：前两项无量纲；后三项单位m。滚动参数是约束中力矩/法向力的比例尺度，不等于无量纲滚动阻力系数。

| contacts下的组 | friction五元组 | condim | solref [时间常数s,阻尼比] |
|---|---|---:|---|
| ball_turf | [0.65,0.65,0.002,0.0008,0.0008] | 6 | [0.012,0.5] |
| foot_turf | [0.8,0.8,0.003,0.0001,0.0001] | 6 | [0.008,1.0] |
| ball_foot | [0.6,0.6,0.002,0.0001,0.0001] | 6 | [0.015,1.2] |
| ball_goal | [0.5,0.5,0.002,0.0001,0.0001] | 6 | [0.008,0.8] |

各组 `solimp=[0.9,0.95,0.001,0.5,2]`；含义依次为dmin、dmax、width(m)、midpoint、power，控制阻抗随穿透的变化。`solref`正值形式控制约束弹簧阻尼，不能直接当作恢复系数e。condim=6才包含全部转动摩擦项；改为3后滚动项即使非零也不生效。摩擦初值不会自动随草高变化。

例如热更新：

```python
env.set_contact('ball_turf', friction=[0.7,0.7,0.002,0.001,0.001], solref=[0.012,0.6])
```

其他机器人碰撞用MuJoCo常规geom接触默认 `friction=[0.6,0.002,0.0001]`、solref `[0.012,1]`。每台机器人有独立碰撞位，启用机器人间碰撞；本基础版本关闭同一机器人的自碰撞，避免简化碰撞体相交影响传球实验。

## K1模型、惯性与控制

[Booster 官方产品页](https://www.booster.tech/booster-k1?target=en) 标示 K1 站高约 **0.95 m**、质量约 **19.5 kg**、22 自由度（双腿各6、双臂各4、头部2）。[官方K1_22dof.xml](https://github.com/BoosterRobotics/booster_assets/blob/12a97516f57469078707afb587a5f84c1cb110d1/robots/K1/K1_22dof.xml) 的单台模型总质量为 **19.666 kg**；这是MJCF连杆质量之和，不是对用户真机的称重结果。官方的网格和连杆坐标决定形状尺寸；本项目没有缩放网格。脚碰撞盒全尺寸0.18×0.07×0.036 m。模型是串联关节近似，踝关节不展开真实并联机构。官网硬件峰值力矩与仿真文件中的理想关节力矩限制并非同一个指标，不将二者混用。

| 参数 | 当前值 | 作用 |
|---|---:|---|
| robot.asset_dir / xml | assets/booster_assets/robots/K1 / K1_22dof.xml | 模型入口；当前控制命名针对K1，T1需额外适配 |
| robot.base_mode | fixed | 去掉根自由关节、固定骨盆；free保留浮动基座 |
| robot.base_height | 0.572 | 世界坐标骨盆高度，固定基座实验设置，不是机器人身高 |
| robot.joint_damping | 0.25 | 每个非根关节粘性阻尼 N·m·s/rad |
| robot.kp / kd | 180 / 6 | τ=kp(qtarget−q)−kd·qdot；工程初值 |
| robot.torque_scale | 1 | 乘于官方MJCF各关节forcelimit |
| robot.mass_scale | 1 | 同比例乘各连杆质量及惯量 |
| robot.inertia_scale | 1 | 在mass_scale之后额外乘惯量 |
| robot.link_overrides | {} | 以原连杆名称逐项覆盖mass、diaginertia，优先于scale |
| robot.home | YAML字典 | 站立目标角rad |
| robot.instances | 六台位置/朝向/队色 | 平面位置m、偏航rad；可修改数量及部署点 |

官方电机限制摘要（正负对称，N·m）：头6、手臂14、髋俯仰68、髋滚转43、髋偏航38.3、膝112、踝38.3。详细关节角上下限、armature、每段连杆质量/惯量及惯性坐标系均自动导出到 `robot_parameters.json`。

```bash
soccer-sim build --out runs/heavier --set robot.mass_scale=1.05
soccer-sim build --out runs/trunk --set 'robot.link_overrides={trunk: {mass: 6.8}}'
```

直接改质量而不改惯量表示人为更改惯性性质；如果保持几何和质量分布，应使用mass_scale，或同时提供合理惯量。`step(joint_torques=...)`接受132维动作（六台），索引由actuator_lookup查询，仍按力矩限制裁剪。

## 踢球与接球策略

本节仅适用于 `soccer-sim pass`，源码 `soccer_sim/passing.py`；手写轨迹，无RL训练、无运动捕捉回放。只调节发送方右髋俯仰、右膝、右踝俯仰；三次smoothstep插值形成后摆、击球、恢复，关节运动由力矩积分产生。已下载权重的 `soccer-policy` 使用另一套控制入口，见下节。

| passing参数 | 当前值 |
|---|---|
| sender / receiver | A1 / A2 |
| kick_start_time | 0.7 s |
| backswing_duration / strike_duration / recover_duration | 0.35 / 0.16 / 0.6 s |
| backswing 右髋/膝/踝 | 0.25 / 0.45 / −0.65 rad |
| strike 右髋/膝/踝 | −0.65 / 0.25 / 0.3 rad |
| receiver_pose 右踝 | −0.75 rad（抬脚尖缓冲） |
| receive_radius / receive_speed / receive_hold | 0.45 m / 0.25 m/s / 0.3 s |
| timeout | 12 s |

发送/接收角色及初始球位置可改；这条轨迹针对当前固定间距和朝向，改变场景后可能需要重调，不提供任意目标自动瞄准。示例中只有两台参与传接，其他四台保持站姿。

## 预训练控制参数

`soccer-policy` 的场地、球、空气、接触参数与上述配置共享。机器人保留各策略配套的 MJCF、初始姿态、增益、关节顺序和 armature，强制浮动基座；不套用固定骨盆的 `robot.home/kp/kd/joint_damping`。配置的 `robot.base_height` 会在输出中改记上游初始化使用的实际根高度，它不是站高。逐关节实际数值导出至 `robot_parameters.json`，初始状态另存 `initial_state.npz` 和场景 `policy_initial` keyframe。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| pretrained.command | [0.4,0,0] | Booster 的机体坐标速度指令 vx、vy（m/s）、yaw rate（rad/s）；CLI `--command` 可覆盖 |
| pretrained.release_distance | 0.7 m | 简单传球实验中球移动超过该距离后发送方改为零速度指令 |
| pretrained.kp_scale / kd_scale | 1 / 1 | 乘各策略原有逐关节PD增益 |
| pretrained.action_scale | 1 | 乘原有神经网络输出到目标角的比例，不改变观测中的历史原始动作 |
| pretrained.joint_damping | null | 保留原模型关节阻尼，或设置统一非负阻尼 |
| robot.mass_scale / inertia_scale / link_overrides | 1 / 1 / {} | 两种策略场景均生效；覆盖名称按各自导出的连杆名填写 |
| robot.torque_scale | 1 | Booster乘原执行器力矩限幅；SONIC乘上游plant的显式PD力矩限幅 |

预训练策略固定50 Hz，物理步长需整除0.02 s；默认1 ms。使用 `--native-contacts` 可对比原策略的足地接触设置，默认仍用本项目的 `contacts.foot_turf`。两种设置均保留可编辑的球地、脚球接触。策略网络本身没有足球状态输入，不能根据球的位置自主选传球对象。配套模型名称/碰撞近似与固定骨盆官方资产有所不同，不能只把22个输出按索引直接套用。

```bash
soccer-policy --kind booster_mjlab --scenario walk --seconds 6 --out runs/variant \
  --set 'pretrained.command=[0.3,0,0.1]' --set pretrained.kp_scale=1.05 \
  --set 'contacts.foot_turf.friction=[0.75,0.75,0.003,0.0001,0.0001]'
```

## 数值与渲染设置

`physics.timestep=0.001`、gravity=[0,0,−9.81]、integrator=implicitfast、solver=Newton、iterations=80、tolerance=1e−8、cone=elliptic。步长需≤0.005 s，并建议≤0.001–0.002 s做脚球接触。

render.width/height=1280/720、fps=30，camera_distance=10、camera_azimuth=95°、camera_elevation=−35°、camera_lookat=[0,0,0.3]；相机角度使用MuJoCo的度数。完整球场俯视可将distance改为17、elevation改为−60。

## 实测标定建议

1. 测球周长、质量及气压，先固定球参数。
2. 无机器人滚动实验记录不同初速度下的x(t)、v(t)及停球距离；用 `probe` 轨迹对比，先匹配滚动摩擦，再考虑空气阻力。
3. 滑移/斜坡实验约束滑动摩擦，避免用停止距离同时拟合所有参数。
4. 记录不同高度落球的首次反弹高度和接触时间，拟合ball_turf的solref/solimp。没有直接的e字段。
5. 再测脚球碰撞和不同关节轨迹的出球速度，标定ball_foot、PD增益和电机限制；最后解除固定骨盆接入平衡策略。

`probe`的滚动通道位于y=2，避免球门挡板；接触地面是无限平面，轨迹可越出可视球场。probe是复现实验工具，不会自行把工程初值声明为实测标定结果。
