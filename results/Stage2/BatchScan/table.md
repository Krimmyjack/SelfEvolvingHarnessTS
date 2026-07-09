# Track A exploratory 批键扫描（**描述性/发现集/非转正证据**；主指标=oracle 一致率↑）

| batch 键 | #batches | oracle 一致率 | 批内 response 方差 | family purity(旁证) |
|---|---|---|---|---|
| legacy_cell | 4 | 0.222 | 6.2423 | 0.231 |
| P0_kmeans | 8 | 0.333 | 4.0923 | 0.500 |
| P1b_kmeans | 8 | 0.329 | 4.4216 | 0.506 |

**方向读数（非门控）**：oracle 一致率最高键 = **P0_kmeans**（0.333）。响应同质性越高 = 批越"处理响应相似"。

> 边界：发现集描述性扫描，**不锁 confirmatory**；新 namespace S2R1_scan_20260707 扫描（须重建 L_test）为早上 turnkey 项。K=8 固定=族数（非调参）。
