# 来源与许可

本仓库的仿真、研究适配代码和文档以 Apache-2.0 发布，见 LICENSE。下载到本机的第三方模型、网格和策略权重仍按各自上游条件使用；仓库主许可不覆盖它们。

| 来源 | 复用内容 | 记录方式 |
|---|---|---|
| [SuperMimic-mjlab](https://github.com/YHFone/supermimic-mjlab) | 部分网络与可微 FK 实现；不是官方完整复现 | `simulation/soccer_sim/supermimic_core/SOURCE.json` 与原 LICENSE；本地四元数工具替换见源码 |
| [Joga](https://joga.csail.mit.edu/) | G1 公开演示的部署模型、技能权重和接口 | `simulation/configs/joga_sources.json` 的 SHA-256；资源由 `fetch_joga.py` 获取，未随仓库再分发 |
| [Booster assets](https://github.com/BoosterRobotics/booster_assets) | 官方 K1 MJCF 与网格 | `fetch_assets.py` 固定 revision，下载并保留上游许可 |
| [booster_mjlab](https://github.com/IntelligentRoboticsLab/booster_mjlab) | K1 步态部署资源 | `fetch_policies.py` 固定 revision；不包含其训练系统 |
| [SONIC-K1](https://github.com/Kyk0/sonic-k1) | K1 部署接口和运动资源 | `fetch_policies.py` 固定代码/权重 revision，保留上游 LICENSE/NOTICE |
| [B-Human](https://github.com/bhuman/BHumanCodeRelease) | 独立步态/踢球导出资源 | `bhuman_skill_sources.json`；下载保留 License.md |
| [NUbots](https://github.com/NUbots/NUbots_K1) | 早期 K1 技能接口资源 | `skill_sources.json` 固定 revision/校验和 |
| [HSL Rules](https://github.com/RoboCup-HumanoidSoccerLeague/HSL-Rules) | 场地与规则参数依据 | 配置固定源码 revision，并注明开发稿身份 |

This product includes software developed by B-Human (http://www.b-human.de).

Joga 公开部署资源的可访问性不等于完整训练代码或权重的再许可；本仓库只提供本地获取和校验工具。外部资源版本改变时校验会失败，应先核对来源再更新清单。

SuperMimic 适配由网络/FK 组件与独立编写的足球环境、物理 WM、控制器及训练器组成。任务设计和训练结果由本项目负责，未暗示上游作者认可本研究。
