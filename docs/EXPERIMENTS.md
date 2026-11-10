# 实验运行说明

先按根 README 安装 G1 GPU 环境并下载 Joga 部署资源。命令从 `simulation/` 执行，设置 `PYTHONPATH=.`。K1 的 MuJoCo 3.3.7 环境与 G1 的 MuJoCo 3.12.0 环境分开安装。

## 新机器能运行什么

- 根 README 的原生技能初始化训练及 CPU/GPU 检查。
- 自己生成的检查点的固定初态重复评估。
- 下载模型后的 K1 场地构建、球滚动/落球标定探针。

历史续训研究依赖未发布的起点/峰值/晚期检查点；仓库包含协议与实现，未声称已提供这些权重。新初始化实验与历史续训的初始条件不同，结果要分别报告。

## 固定条件审计

```bash
CUDA_VISIBLE_DEVICES=0 .venv-g1/bin/python -m soccer_sim.gpu_receive.audit \
  --out runs/my_audit --checkpoint start=runs/my_run/best.pt \
  --num-envs 512 --repeats 5 --controller-kinematics current
```

保存检查点 SHA-256、精确 qpos/qvel、每环境结果和重复评估汇总。相同初态的重复只衡量数值/评估波动，不算独立训练种子。

## 历史续训队列

准备独立目录 `<experiment>/`，其中 `source/` 是本仓库的 `simulation/` 快照，`runs/` 为空。将已取得并核验的三个历史检查点路径写入 `<experiment>/checkpoint_paths.json`：

```json
{"start": "/absolute/path/start.pt", "peak": "/absolute/path/peak.pt", "late": "/absolute/path/late.pt"}
```

```bash
.venv-g1/bin/python tools/run_cooperation_pilot.py \
  --root /absolute/path/experiment --gpu 0 \
  --python /absolute/path/simulation/.venv-g1/bin/python --preflight-only

# 补做正常十次 WM 拟合的三组检查，通过后继续匹配预算队列
.venv-g1/bin/python tools/complete_cooperation_preflight.py \
  --root /absolute/path/experiment --gpu 0 \
  --python /absolute/path/simulation/.venv-g1/bin/python
```

队列一次只运行一个本项目 GPU 子进程，保存状态、命令、日志和源文件哈希；检查失败、显存不足、磁盘不足或预算不完整会停止。`--skip-preflight` 不会绕过已通过的源代码校验。源代码改变后应创建新版本并重新验证。

## 记录口径

- 2048 个训练环境，512 个独立评估环境，batch1024、micro128。
- 三组共享 H32 合法窗口、当前状态 FK 修复、相同优化器和奖励。
- `recorded` 使用未来真实控制，是诊断上界输入，不是机器人能获得的部署观测。
- 种子 104/105/106 在共同历史检查点后重新初始化 RNG，是独立续训重复。
- 每组追加 512 次更新 × 32 步 × 2048 环境 = 33,554,432 条训练交互；warmup 另计。
- 开发集用于分析；seed240000/1024 预留最终验收，当前开发过程不使用。

完整条件以[预注册协议](../simulation/research/cooperative_handoff/experiments/00_control_consistency/protocol.md)为准。
