# updater v2 三臂（16 半块 × 5 预锁排列；prereg §3）

| arm | cum regret (5 排列均值 [min,max]) | 首遇 harm(max 均值) | 复现增益 | TTR | accepts | canary-rej | rollbacks | false-acc | coverage |
|---|---|---|---|---|---|---|---|---|---|
| frozen | 0.3677 [0.368,0.368] | +0.000 | +0.0000 | 0.0 | 0 | 0 | 0 | 0 | 0.00 |
| updater_v1 | 0.4012 [0.362,0.460] | +0.337 | -0.0141 | 15.4 | 19 | 0 | 15 | 3 | 1.00 |
| updater_v2 | 0.3775 [0.361,0.401] | +0.198 | +0.0033 | 14.4 | 18 | 3 | 7 | 0 | 0.68 |

判据：{"c1_cum_not_worse_than_frozen": false, "c2_first_unseen_harm_controlled": false, "c3_fewer_failures_than_v1": true, "c4_nonzero_coverage": true, "c5_recurrence_beats_frozen": true, "c6_ttr_shorter_than_v1": true}
**判决**：PARTIAL(4/6)：按分支细则判
