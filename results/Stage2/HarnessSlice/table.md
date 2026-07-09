# Harness action-only 垂直切片：frozen vs updating（第一张 Harness 表）

> frozen 臂=账本重放（不真执行，声明）；updating 臂全 uid 真执行 overlay；grounded utility=S2 nested L_test（offline-gym 语义，声明）
> 纪律：允许更新=Router 参数/κ；Pattern=P0 冻结、动作池冻结、无模型轴、无 LLM

| blk | family | n | frozen | updating | Δ(f−u) | diverged | 事件 |
|---|---|---|---|---|---|---|---|
| 0 | S_season | 84 | 0.077 | 0.077 | +0.000 | 0 | accept κ=0.5 |
| 1 | S_trend | 84 | 1.368 | 1.571 | -0.203 | 0 | ROLLBACK, accept κ=2.0 |
| 2 | S_both | 84 | 0.646 | 0.729 | -0.083 | 0 | ROLLBACK, accept κ=0.5 |
| 3 | S_ar | 84 | 0.073 | 0.083 | -0.009 | 0 | reject |
| 4 | S_multiseason | 84 | 0.203 | 0.205 | -0.001 | 0 | reject |
| 5 | S_hetero | 84 | 0.066 | 0.059 | +0.006 | 0 | reject |
| 6 | S_intermittent | 84 | 0.082 | 0.081 | +0.001 | 0 | reject |
| 7 | S_regime | 84 | 0.441 | 0.515 | -0.073 | 0 | reject |

**cumulative regret**：frozen=0.3695 updating=0.4149 （Δ=-0.0454）
updates：accepted=3 rejected=5 rollbacks=2；evidence：672 条（routing 672）
