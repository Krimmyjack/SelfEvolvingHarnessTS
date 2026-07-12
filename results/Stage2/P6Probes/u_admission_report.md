# P6 U 域准入探针报告（u_admission）

- 日期：2026-07-10；seed=20260710；探针 n=24/config（item_id 排序后抽样，uid 落盘并排除出最终 U 集）
- 冻结判官协议：L_WIN=48, H=48, MIN_LEN=144（evaluators/base.py, data/load_real.py:43）
- 性质：只读准入探针，**不属于 P6 实验本体**；不裁量修改 L/H。

## electricity_hourly

**下载失败**：ValueError: no parquet shards found for electricity_hourly/test

## traffic_hourly

**① 可用性** 加载 60，长度≥144 共 60；长度 min/median/max = 1024/1024/1024

**② period 估计**（分桶容差 ±10%）

| estimator | ≈24 | ≈168 | none | other |
|---|---|---|---|---|
| legacy_fft_v0 (P0 感知端) | 1.00 | 0.00 | 0.00 | 0.00 |
| robust_v1 (算子端) | 1.00 | 0.00 | 0.00 | 0.00 |

**③ ACF** acf24 mean/median = 0.7178/0.7458；acf168 = 0.6579/0.6712；|acf24|>|acf168| 占比 = 0.92

**④ 48-lookback 判官**（pool 5304 窗，stride 4）

| λ | judge mean nRMSE | snaive24 mean | judge/sn24 | 胜率 vs sn24 | snaive168 mean | judge/sn168 | 胜率 vs sn168 |
|---|---|---|---|---|---|---|---|
| 0.001 | 0.4014 | 0.4696 | 0.855 | 0.79 | 0.4251 | 0.944 | 0.50 |
| 1 | 0.4014 | 0.4696 | 0.855 | 0.79 | 0.4251 | 0.944 | 0.50 |

**⑤ 截断影响**（last-4096 vs last-1024，n=24） legacy 分桶变化占比 = 0.00；robust = 0.00；tail-1024 字节不一致 0 条（NaN 填充边界效应）

**准入判决：`PASS_HEADLINE_U`**（主导周期≤48 占比 1.00；168 主导占比 0.00）

## 结论与首选推荐

- **首选 headline-U 候选：`traffic_hourly`**（主导周期≤48 占比 1.00，λ=1 判官 vs snaive24 胜率 0.79）。
- `electricity_hourly` **无法经项目 pinned loader 获取**（ValueError: no parquet shards found for electricity_hourly/test）——在不改数据路径的前提下不可作为 U 候选（改路径属 P6 决策，非本探针裁量）。
- 被探针 uid（各 config 24 条）见对应 JSON `excluded_uids_for_final_U`，**必须排除出最终 U 集**。
- 不建议修改 L/H（不在裁量范围）。
