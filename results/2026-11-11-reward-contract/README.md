# 奖励公式初检与计划修订

2026-11-11。构造物理状态上的CPU代数检查；**没有运行新奖励训练、没有用这些状态证明物理可行性或实际策略利用漏洞**。没有占用GPU，没有使用独立验收集。

| 构造状态 | 当前瞬时奖励 | 可以判断什么 |
|---|---:|---|
| 身前静止球，有接触历史语义 | 1.0000 | 当前奖励偏好身前且低速 |
| 相同物理状态，无接触历史语义 | 1.0000 | 奖励接口没有接触历史输入，无法区分这两类历史 |
| 背离队友，球仍在身体前方静止 | 1.0000 | 当前奖励没有直接检查面对队友 |
| 身前来球仍以1 m/s移动 | 0.2147 | 速度项降低了瞬时分数；这本身合理，不能当作完整接球质量排名 |

前两行的历史只是反事实语义标签，脚本输入物理状态完全相同；不是两条真实MuJoCo轨迹。完整合理性必须进一步检查相同时限/折扣的物理轨迹、各奖励项的累计预算和独立任务结果。

下一步已纳入[研究计划](../../docs/RESEARCH_PLAN.md)、[奖励验证](../../docs/REWARD_VALIDATION.md)与[协议/工程参数](../../simulation/research/cooperative_handoff/experiments/01_reception_metrics_reward/protocol.md)。评估器和新奖励尚未实现。

## 复现

从仓库simulation目录、已安装G1依赖的环境执行：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python ../results/2026-11-11-reward-contract/probe.py --out /tmp/reward_contract.json
```

[summary.json](summary.json)保存原奖励模块哈希与结果。[scheduling.json](scheduling.json)记录GPU1原对照恢复快照；[interruption.json](interruption.json)记录中断检查点与排除规则。原失败尝试归档保留，完整重跑不改变九组科学配置。新任务按评估正确性→可行性→奖励审计→短训练→WM与协作的顺序放行。
