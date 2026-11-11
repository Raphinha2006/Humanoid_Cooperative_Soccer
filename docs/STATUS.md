# 当前进展

北京时间2026-11-11 10:16：**完成3/9组，GPU1正在重跑第4组seed105_recorded**，日志已到iteration155（共同起点141）。随后五组串行排队。本进程采样约4.58 GiB，未见新错误。

原第4组在2026-11-11 09:36:58因运行中显存不足退出，最后日志iteration230、最新检查点228。检查点缺replay，故整组/日志已单独归档，从共同起点重新跑完整预算；失败尝试不混入匹配对照。恢复措施见[记录](../simulation/research/cooperative_handoff/experiments/00_control_consistency/retry_20261111.md)。

## seed104 三种处理均完成

每组追加512次更新和33,554,432条训练交互，同一512环境开发集、current观测模式：

| 发送方处理 | 起点 | 新训练峰值（不含起点） | 完成时 |
|---|---:|---:|---:|
| held | 5.66% | 8.59% | 4.69% |
| recorded | 5.66% | 4.69% | 2.54% |
| closed_loop | 5.86% | 2.15% | 0.39% |

**三个处理在这个种子下都未稳定提升，闭环组退化最明显。** 这不是多种子结论；seed105/106尚在执行，不能确定总体排名或退化主因。尚未实现稳定连续互传。

[本轮三组曲线、预测误差与原始日志](../results/2026-11-11-seed104-comparison/README.md)已公开。最终回报依次16.53/21.41/27.70，更高回报未对应更高严格接球率。误差来自各自策略的轨迹，不能等同于固定数据集模型排名。

## 显存与队列

recorded完整进程采样最高4,706 MiB（约4.60 GiB），closed_loop为7,062 MiB（约6.90 GiB）。held首组未进行完整进程采样。这些是每5秒采样的最大值，非精确瞬时峰值。

启动门槛为held/recorded 7 GiB、closed_loop 8.5 GiB；环境2048、batch1024、micro128、WM2048、H32、优化器、奖励和预算保持不变。每30分钟归档检查点，资源不够时最多等待12小时；失败则停队列保留产物。

原队列曾在11月11日00:51因统一10,000MiB门槛等待超时退出，发生在第二组启动前，没有训练OOM。已恢复并完成seed104其余两组。操作记录见[恢复记录](../simulation/research/cooperative_handoff/experiments/00_control_consistency/recovery_20261111.md)和[显存修订](../simulation/research/cooperative_handoff/experiments/00_control_consistency/memory_admission_20261111.md)。

## 新计划与当前结论

严格成功率继续作为完整任务验收；下一阶段分别测停球耗时（越短越好）与连续控球保持时长（越长越好），不凭联合保持量下降直接推断停球变慢。物理事件、失败/删失、固定观察窗、能力图和奖励反例纳入[修订计划](RESEARCH_PLAN.md)及[奖励验证](REWARD_VALIDATION.md)。评估器和新奖励尚未实现。

CPU构造状态检查已完成：当前奖励对相同物理状态的不同接触历史、以及球仍在身体前方但机器人背离队友的状态，均可给出同样的瞬时奖励。它说明奖励缺少部分任务区分，不证明真实策略利用漏洞。[检查结果](../results/2026-11-11-reward-contract/README.md)。

前置CPU/GPU一致性与冻结预测检查见[历史报告](../results/2026-11-10-control-audit/README.md)。独立验收集未用，尚未实现稳定连续互传。
