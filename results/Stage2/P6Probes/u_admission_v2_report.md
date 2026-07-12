# P6 U 域全宇宙复检探针报告（u_admission_v2）

- 日期：2026-07-11；capability seed=20260711；性质：只读复检探针（外部 GPT 审查 NO-GO 意见 #10 兑现），**不属于 P6 实验本体**；不裁量修改 L/H。
- 相对首轮（u_admission_report.md）的两点整改：①结构统计从前 60 条扩到**过滤后全宇宙逐条**；②judge-capability 改调 **canonical 判官** `p6/judge_closed_form.py`（history-only z-score；series_weight="equal"、λ=1e-3、stride=4、window_cap=None），对照基线统一为原始尺度 RMSE ÷ history std（与判官同尺度）。
- 冻结协议：L_WIN=48, H=48, MIN_LEN=144；判官 `dlinear_closed_form_v1`；双路对拍 atol=1e-09。

## ① 全宇宙可用性（loader min_len=64 → 再按 ≥144 过滤）

- 加载 **862** 条（预期 862：✔ 一致）；长度≥144：**862** 条；重复 item_id：0。
- 长度分布（过滤后）：min/median/max = 1024/1024/1024；直方图 {'1024': 862}

## ② period 估计（全宇宙 n=862；分桶容差 ±10%）

| estimator | ≈24 | ≈168 | none | other |
|---|---|---|---|---|
| legacy_fft_v0 (P0 感知端) | 0.9803 | 0.0012 | 0.0012 | 0.0174 |
| robust_v1 (算子端) | 0.9919 | 0.0012 | 0.0070 | 0.0000 |

（robust_v1 计数：≈24 855、≈168 1、none 6、other 0）

## ③ ACF（全宇宙 n=862）

- acf24 mean/median = 0.7162/0.7387；acf168 = 0.6454/0.6747；**|acf24|>|acf168| 占比 = 0.9037**

## ④ judge-capability（canonical 判官；n=32；排除首轮 24 条后从 838 条均匀抽样）

- 双路对拍（fit_domain vs fit_domain_rebuild，atol 1e-09）：评估量级（per-series RMSE + utility）**PASS**（rmse max|Δ|=2.46e-14，utility |Δ|=3.33e-16）；W 严格级 **FAIL**（W max|Δ|=1.64e-08，相对 max|W|=3.59e-08）；pooled 7072 窗（stride 4，series_weight=equal，λ=0.001）。
- ⚠ 发现（供 P6 签发前决策）：真实 U 尺度（7072 窗 pooled）下两条代数等价路径的浮点累积使 W 级 |Δ| ≈ 1e-8 > 1e-9（toy 级单测通过），而承载判决的 RMSE/utility 一致到 ~1e-14——正式 runner 的双路对拍若按 W 级 atol=1e-9 实现将在 U 尺度 technical abort，须按评估量或 W 相对容差实现（本探针不裁量）。

| judge mean nRMSE | judge median | sn24 mean | judge/sn24 | 胜率 vs sn24 | sn168 mean | judge/sn168 | 胜率 vs sn168 |
|---|---|---|---|---|---|---|---|
| 0.3620 | 0.3179 | 0.4276 | 0.847 | 0.66 | 0.4096 | 0.884 | 0.56 |

- 口径：judge = canonical per-series RMSE（history-only z 空间，数学上 = raw-RMSE ÷ history std）；sn24/sn168 = 原始尺度 RMSE ÷ zscore_state(history).std —— 同尺度可比。
- 首轮参考值（不同口径不直接可比，仅方向参考：判官镜像实现 + 全数组 z-score、pooled 无 series 等权、n=24、前 60 条内）：judge/sn24=0.855、胜率 0.79；judge/sn168=0.944、胜率 0.50。

## ⑤ 准入判决（判则与首轮逐字一致；过滤后全宇宙 n=862（首轮仅前 60 条中抽 24））

- share_dominant_period_le48 = **0.9954**；share_168_dominant = 0.0000
- **判决：`PASS_HEADLINE_U`**（首轮 `PASS_HEADLINE_U` → **维持**）

## ⑥ 探针消费与最终 U 排除集

- 首轮 24 条 + 本轮 capability 32 条，重叠 0 → `all_probe_consumed_item_ids` 共 **56** 条（见 JSON 同名字段）——**最终 U 抽取的排除集**。
- capability 子样本 32 条 item_id + content sha256（NaN 填充后、z-score 前 float64 字节）见 JSON `capability_manifest`。

## ⑦ 确定性（幂等重跑 diff）

- 新进程全量重算，canonical diff（json sort_keys；剔除 `_volatile`/`determinism` 墙钟字段后逐字节比较）：**PASS**
- payload sha256：run1 = `532ad8b3aa5f2c305fd25b1def0b5bc428bda904082892c84381f013a2238ad2`，run2 = `532ad8b3aa5f2c305fd25b1def0b5bc428bda904082892c84381f013a2238ad2`
