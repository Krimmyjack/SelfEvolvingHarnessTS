# updater v3 = response-aware support（16 半块 × 5 预锁排列；prereg_updater3.md）

| arm | cum regret (5 排列均值 [min,max]) | 首遇 harm(max 均值) | 复现增益 | TTR | accepts | canary-rej | rollbacks | coverage | cov首遇/复现 |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 0.3677 [0.368,0.368] | +0.000 | +0.0000 | 0.0 | 0 | 0 | 0 | 0.00 | — |
| updater_v2 | 0.3775 [0.361,0.401] | +0.198 | +0.0033 | 14.4 | 18 | 3 | 7 | 0.68 | — |
| updater_v3 | 0.4033 [0.364,0.464] | +0.284 | -0.0194 | 15.4 | 19 | 1 | 14 | 0.91 | 0.91/0.92 |

判据：{"g1_cum_not_worse_than_frozen": false, "g2_first_unseen_harm_controlled": false, "g3_recurrence_beats_frozen": false, "g4_coverage_floor": true, "g5_fewer_failures_than_v2": false, "g6_probe_budget_respected": true}
**判决**：FAIL：响应前提存疑，回操纵检查证据分析（不调阈值）

操纵检查（非门控）：1-NN 族分类 签名空间 0.350 vs P0 空间 0.896（n=666，无效签名 6）
探针预算（独立预算行，prereg §0.3）：4 动作 × 1 切点 × 3 维签名/uid；无 regret 单位折算（目标函数无成本项，声明）。
