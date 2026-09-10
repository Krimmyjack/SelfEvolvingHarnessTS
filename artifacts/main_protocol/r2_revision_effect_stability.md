# R2 · 修订效应稳定性读数（描述性；0 fit）

地位：development 描述性读数，**不做显著性主张**。
问题：同一谱系里子程序相对祖先的差值 `d` 是否比祖先自身的效应 `g` 在跨单元上更稳定。

## 数据来源

| 源 | 路径 | 行数 / 覆盖 |
| --- | --- | --- |
| (a) 聚合 `per_unit` | `artifacts/main_protocol/dev_auto1_blocked_candidates__run2.json` | 3 个策略 × 16 单元 × 2 面 |
| (b) 逐序列预测库 | `_scratch/m_r0k_prediction_store.json` | 263 条；键 `program|position|face` |

预测库覆盖：ANCESTOR / W1 / W2 / W3 各 21 位置 × 2 面；两个 CAND 合计 31+31 条。
逐序列增益 `g = raw − program`（正 = 程序有益）。

## 缺失处理

a face is paired only when both child and ANCESTOR are READ and carry aggregate_gain; WINDOW_VERIFIER_REJECTED, SERVING_CONTEXT_DEGENERATE, NO_STORED_READING, NOT_READABLE and UNKNOWN stay missing and are never stored as 0。
方差：样本方差 `ddof=1`。符号一致率：`max(正占比, 负占比)`，零计入分母。

已观测的非 READ 状态（原样保留为缺失，不记 0）：
`WINDOW_VERIFIER_REJECTED`、`SERVING_CONTEXT_DEGENERATE`、`NO_STORED_READING`。
本批 `per_unit` 未出现字面 `NOT_READABLE` / `UNKNOWN`；若出现同样保留为缺失。

## 字段核验（一条完整可读记录的键，未猜测）

- 单元记录键：`policy, position, unit, side, exposure, faces`
- 可读 face 键：`origin, resolved, ancestor_resolved, reading, ancestor_reading, gate, ancestor_gate, served_denominator_violations, behaviour_identical, series_scored_differently, delta_aggregate_gain, delta_harmed_series, delta_max_single_series_harm, delta_treated, exclusion_account`
- reading 键：`status, identity, treated, served, per_series_gain, aggregate_gain, harmed_fraction, harmed_series, max_single_series_harm, mean_smase, static_mean_smase`
- 预测库样本键 `ANCESTOR|0|support_face` 字段：`program, position, face, origin, verifier_passed, eval_uids, raw_per_view, program_per_view, degenerate_uids, physical_fits`

## 读数（6 位小数）

CAND 的聚合层来自 `per_unit`；W1/W2/W3 的聚合层来自预测库按单元均值 （`per_unit` 不含这些程序）。逐序列层全部来自预测库。

| child | face | 层 | n | 正占比(d) | 负占比(d) | 符号一致率(d) | 符号一致率(g) | mean(d) | std(d) | mean(g) | std(g) | Var(d)/Var(g) | 受损数差均值 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CAND_outlier_iqr({})>hampel_filter({})` | support_face | per_unit_aggregate | 12 | 0.583333 | 0.416667 | 0.583333 | 0.916667 | 0.015763 | 0.077780 | 0.151091 | 0.142861 | 0.296424 | 0.500000 |
| `CAND_outlier_iqr({})>hampel_filter({})` | delayed_face | per_unit_aggregate | 14 | 0.571429 | 0.428571 | 0.571429 | 0.857143 | 0.003750 | 0.118534 | 0.155353 | 0.178081 | 0.443051 | 1.142857 |
| `CAND_outlier_iqr({})>winsorize({})` | support_face | per_unit_aggregate | 12 | 0.833333 | 0.166667 | 0.833333 | 0.916667 | 0.075791 | 0.105704 | 0.151091 | 0.142861 | 0.547469 | -0.916667 |
| `CAND_outlier_iqr({})>winsorize({})` | delayed_face | per_unit_aggregate | 14 | 0.785714 | 0.214286 | 0.785714 | 0.857143 | 0.044090 | 0.067530 | 0.155353 | 0.178081 | 0.143800 | -0.214286 |
| `W1_hampel_filter` | support_face | store_unit_aggregate | 21 | 0.047619 | 0.952381 | 0.952381 | 0.857143 | -0.276782 | 0.190573 | 0.245427 | 0.197427 | 0.931773 | 4.666667 |
| `W1_hampel_filter` | delayed_face | store_unit_aggregate | 21 | 0.142857 | 0.857143 | 0.857143 | 0.904762 | -0.179176 | 0.164381 | 0.224645 | 0.196580 | 0.699238 | 3.476190 |
| `W2_pmc_then_outlier_mad` | support_face | store_unit_aggregate | 21 | 0.619048 | 0.380952 | 0.619048 | 0.857143 | 0.026891 | 0.177055 | 0.245427 | 0.197427 | 0.804268 | 1.047619 |
| `W2_pmc_then_outlier_mad` | delayed_face | store_unit_aggregate | 21 | 0.476190 | 0.523810 | 0.523810 | 0.904762 | -0.039643 | 0.191374 | 0.224645 | 0.196580 | 0.947736 | 0.809524 |
| `W3_outlier_mad_then_pmc` | support_face | store_unit_aggregate | 21 | 0.000000 | 0.000000 | 0.000000 | 0.857143 | 0.000000 | 0.000000 | 0.245427 | 0.197427 | 0.000000 | 0.000000 |
| `W3_outlier_mad_then_pmc` | delayed_face | store_unit_aggregate | 21 | 0.000000 | 0.000000 | 0.000000 | 0.904762 | 0.000000 | 0.000000 | 0.224645 | 0.196580 | 0.000000 | 0.000000 |

### 逐序列层

| child | face | n 配对序列 | 正占比(d_i) | 负占比(d_i) | 符号一致率(d_i) | Var(d_i)/Var(g_anc_i) | 单元内 d_i>0 占比均值 | 该占比标准差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `CAND_outlier_iqr({})>hampel_filter({})` | support_face | 240 | 0.491667 | 0.508333 | 0.508333 | 0.533886 | 0.491667 | 0.139534 |
| `CAND_outlier_iqr({})>hampel_filter({})` | delayed_face | 280 | 0.485714 | 0.514286 | 0.514286 | 0.410592 | 0.485714 | 0.143351 |
| `CAND_outlier_iqr({})>winsorize({})` | support_face | 240 | 0.654167 | 0.345833 | 0.654167 | 0.384033 | 0.654167 | 0.199384 |
| `CAND_outlier_iqr({})>winsorize({})` | delayed_face | 280 | 0.578571 | 0.421429 | 0.578571 | 0.217974 | 0.578571 | 0.175098 |
| `W1_hampel_filter` | support_face | 420 | 0.302381 | 0.697619 | 0.697619 | 1.154265 | 0.302381 | 0.090106 |
| `W1_hampel_filter` | delayed_face | 420 | 0.352381 | 0.647619 | 0.647619 | 0.932199 | 0.352381 | 0.140958 |
| `W2_pmc_then_outlier_mad` | support_face | 420 | 0.509524 | 0.490476 | 0.509524 | 0.781368 | 0.509524 | 0.160950 |
| `W2_pmc_then_outlier_mad` | delayed_face | 420 | 0.454762 | 0.545238 | 0.545238 | 0.595278 | 0.454762 | 0.196153 |
| `W3_outlier_mad_then_pmc` | support_face | 420 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `W3_outlier_mad_then_pmc` | delayed_face | 420 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### 逐序列受损计数差（g<0 重算）对照聚合层 harmed_series 差

| child | face | 聚合层受损数差均值 | 逐序列 g<0 计数差均值 |
| --- | --- | ---: | ---: |
| `CAND_outlier_iqr({})>hampel_filter({})` | support_face | 0.500000 | 0.833333 |
| `CAND_outlier_iqr({})>hampel_filter({})` | delayed_face | 1.142857 | 1.214286 |
| `CAND_outlier_iqr({})>winsorize({})` | support_face | -0.916667 | -1.250000 |
| `CAND_outlier_iqr({})>winsorize({})` | delayed_face | -0.214286 | -0.785714 |
| `W1_hampel_filter` | support_face | 4.666667 | 4.666667 |
| `W1_hampel_filter` | delayed_face | 3.476190 | 3.476190 |
| `W2_pmc_then_outlier_mad` | support_face | 1.047619 | 1.047619 |
| `W2_pmc_then_outlier_mad` | delayed_face | 0.809524 | 0.809524 |
| `W3_outlier_mad_then_pmc` | support_face | 0.000000 | 0.000000 |
| `W3_outlier_mad_then_pmc` | delayed_face | 0.000000 | 0.000000 |

## 缺失单元清单（CAND × per_unit）

### `CAND_outlier_iqr({})>hampel_filter({})` / support_face（4 个缺失）

| position | ancestor_status | child_status |
| ---: | --- | --- |
| 5 | READ | WINDOW_VERIFIER_REJECTED |
| 6 | READ | WINDOW_VERIFIER_REJECTED |
| 14 | READ | SERVING_CONTEXT_DEGENERATE |
| 23 | READ | NO_STORED_READING |

### `CAND_outlier_iqr({})>hampel_filter({})` / delayed_face（2 个缺失）

| position | ancestor_status | child_status |
| ---: | --- | --- |
| 5 | READ | WINDOW_VERIFIER_REJECTED |
| 6 | READ | WINDOW_VERIFIER_REJECTED |

### `CAND_outlier_iqr({})>winsorize({})` / support_face（4 个缺失）

| position | ancestor_status | child_status |
| ---: | --- | --- |
| 5 | READ | WINDOW_VERIFIER_REJECTED |
| 6 | READ | WINDOW_VERIFIER_REJECTED |
| 14 | READ | SERVING_CONTEXT_DEGENERATE |
| 23 | READ | NO_STORED_READING |

### `CAND_outlier_iqr({})>winsorize({})` / delayed_face（2 个缺失）

| position | ancestor_status | child_status |
| ---: | --- | --- |
| 5 | READ | WINDOW_VERIFIER_REJECTED |
| 6 | READ | WINDOW_VERIFIER_REJECTED |

## 一句话

W3 is identically the ancestor on both faces (d=0 on every paired unit; trivial stability, not a revision). Among actual revisions, no child×face jointly has the highest sign consistency and the smallest Var(d)/Var(g): highest sign consistency is W1_hampel_filter/support_face (0.952381, mostly negative); smallest Var ratio is CAND_outlier_iqr({})>winsorize({})/delayed_face (0.143800). Var(d)<Var(g) on 8/8 non-zero pairs, so d is generally smaller in variance than g, but sign(d) is as concentrated as sign(g) on only 1/8 pairs. Descriptive only.

## 边界

fits=0，LLM=0，不编辑既有文件，不提交 git，不新增 SHA/Hash。

复跑：`python -m evaluation.main_protocol_p4.audit_r2_revision_effect_stability`

