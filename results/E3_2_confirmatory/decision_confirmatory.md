# E-3.2 Confirmatory（seeds 20–39 一次性打开）— 判决

日期：2026-07-04/05　协议 = A-40（科学冻结）+ A-41（实现门禁，评审第十七轮"有条件绿灯"）
freeze：`confirmatory_freeze.json` config_sha=`d27acaf5f6c5bf7e`、router artifact sha256=`90d8e0dfae87d1e0…`
（**先于任何 holdout 读取落盘**；九守卫 12 测 + 全库 279 测全过；dev smoke lt/ci/reporter 三链路全通后打开）。
语料 = 基底 seeds 20–39（320）+ A38C 补齐（+240，12/12 可行槽至 N=40，带界=A31e dev 预锁值，
snrHigh×S_ar 0/2000 再证结构性 infeasible）；primary（common-support，排 S_ar）n=480。
**本批 holdout 已消耗：无论下文结果，不得再调策略并复用（A-41⑦/A-32d）。**

---

## 0. 一句话判决

> **判官口径的核心命题通过 confirmatory（C1–C5 全过，主 estimand=locked transfer，统计=grouped
> full-refit B=1000）：dev 冻结的 dp_abstain 原样搬到从未见过的 holdout，仍显著胜 global/
> D-only 查表/连续-D，且 F0 点名的 season 结构伤害被消除（season 全子群 Δ 均值为正、最差
> full-refit 5% 分位 −0.039 > −δ_safe）。**
> **但 C6 独立报告器门失败（5/6）：两报告器一致显示 trend 剂量增益不迁移（dlinear 显著反向），
> 而 season 安全性迁移成立 → "Pattern 条件化路由的下游模型无关增益"未确立；确立的是
> "判官口径增益 + 报告器稳健的安全修复"。**

## 1. 布尔判据表（freeze["criteria"]，主 estimand=locked transfer，primary 口径）

| 门 | 规格 | 结果 | 数字 |
|---|---|:--:|---|
| C1 vs global | 点差<−ε ∧ full-refit CI 上界<0 | ✅ | −0.181；boot −0.179 CI[−0.273,**−0.093**] |
| C2 vs d_lookup | 同上 | ✅ | −0.158；boot −0.148 CI[−0.216,**−0.075**] |
| C3 vs d_gbdt（连续-D） | 同上 | ✅ | −0.178；boot −0.178 CI[−0.248,**−0.107**] |
| C4 season worst LCB | min full-refit q05(Δ vs incumbent) > −δ_safe | ✅ | **−0.039** > −0.05（4 子群 q05：+0.016/−0.016/−0.039/−0.021；点均值全正 +0.023~+0.036）|
| C5 trend retention | ≥50%（vs 冻结 d_lookup） | ✅ | **113%** |
| C6 reporter panel | 两报告器方向一致 ∧ 无显著反向 | **❌** | 见 §3 |

**gates_passed = False（5/6）**；C1–C5 的科学内容=判官口径命题 confirmatory 确认，C6 失败的
科学内容=该增益的**测量口径依赖性**（非机制否证，见 §3 解读）。

## 2. locked transfer 主结果（判官口径，真 out-of-sample）

| 臂（全部 dev 冻结） | regret | 备注 |
|---|--:|---|
| global | 0.4547 | |
| d_lookup | 0.4317 | **season harm 在 holdout 复现**（S_season Δ：snrLow −0.643/−0.900）|
| d_gbdt | 0.4516 | |
| true_d_gbdt | 0.4105 | dp 胜 true-D：−0.136 CI[−0.209,−0.061]（报告） |
| p_gbdt | 0.2738 | P 单独已达大部分增益（与 dev 一致，报告）|
| dp_gbdt | 0.2460 | dp_abstain vs dp_gbdt +0.030 CI[−0.004,+0.067]（报告：abstain 在 holdout 均值略亏、CI 跨 0）|
| **dp_abstain（候选）** | **0.2740** | abstain 18%，集中 snrHigh S_both/S_season（38/33/30/28%）|
| oracle_struct（诊断）| 0.1697 | dev 冻结 cell×origin 表迁移仍近上界 |

paired-uid CI（次要参照）四比较 pos% 全 0.000。
dev→confirmatory 的效应缩水：vs global −0.256→−0.181、vs d_lookup −0.239→−0.158（同向决定性，
幅度 ~70%——正常迁移损耗）。**原始分布与 cell-equal 两聚合几乎重合**（语料平衡设计使然；
cell-equal CI 上界 −0.091/−0.071/−0.109，结论不变）。

**Replication（次要 estimand，confirmatory 内重新 nested cross-fit）**：dp_abstain 0.260 vs
global 0.475 / d_lookup 0.415 / d_gbdt 0.421；retention 139%；season 子群全正；paired CI 全不
跨 0——**学习程序可重复**。（(v) abstain worst-LCB −0.187 vs dp_gbdt −0.180 差一线，如实记录。）

## 3. C6 报告器门（FAIL）——结构化失败，非噪声

per-series nRMSE 配对（Δ=dp−base，负=dp 好）：

| | vs global | vs d_lookup | S_season 子群 | S_trend 子群 |
|---|---|---|---|---|
| dlinear_scratch | **+0.107 CI[+0.063,+0.151] 显著反向** | **+0.083 CI[+0.023,+0.145] 显著反向** | −0.035 / −0.365（dp 优）| **+0.214 / +0.580（dp 劣）** |
| chronos | +0.036 CI[−0.060,+0.130]（方向×，不显著）| **−0.151 CI[−0.224,−0.074]（dp 显著优）** | −0.125 / −0.515（dp 优）| +0.128 / +0.112（dp 劣）|

- **两报告器子群方向完全一致**：S_season 上 dp 全面优于两基线（安全修复**迁移成立**，chronos
  下 d_lookup 的 season 伤害同样巨大）；S_trend 上 dp 全面更差——**判官（Ridge/FrozenProbe）
  给出的重 median 剂量 trend 增益不被独立模型认可，甚至反向**。dlinear 5/5 训练种子配对同向
  （非训练噪声）；dlinear 均值层面被 trend 项主导 → 显著反向 → C6 失败。
- **解读边界**：判官标签本身合法（真 held-out nRMSE），dp 确实改善了该模型类的预测；C6 说明
  该改善**不是模型无关的**——重平滑对 Ridge-head 类是增益、对 DLinear/Chronos 类是损伤。
  F0"剂量维有价值"须改写为"剂量维价值是下游模型类条件化的"。
- **协议纪律**：判据在打开前冻结（A-40⑥/A-41④），不重挑报告器、不重加权、不改口径。

## 4. 只报告项

- overall worst-group（locked transfer）= snrLow|full|S_trend，full-refit q05 −0.410（**均值
  +0.215 为正**——重尾方差而非 season 式结构损伤；按 A-40⑦ 不得写"全局安全"）。
- all_data 附录（含 S_ar，full-refit B=1000）：dp 0.310 vs global 0.523 / d_lookup 0.388 /
  d_gbdt 0.467，retention 94.5%；boot 比较 vs global −0.215 CI[−0.311,−0.120]、vs d_lookup
  −0.072 CI[−0.138,**−0.001**]（贴零——S_ar 混入使 vs D-only 优势变薄，与 dev 稳定性发现互恰）、
  vs d_gbdt −0.158 CI[−0.223,−0.088]；season q05 全 >−0.05（−0.035 最差）；overall worst=
  snrLow|full|S_trend q05 −0.164。定性结论全保持。
- oracle_struct（dev 冻结 cell×origin 表迁移诊断）：见 lt_point / replication（oracle 0.172）。

## 5. 声明（正式措辞）

1. **确认（confirmatory，判官口径，common-support，level-1 合成语料）**：可观测 Pattern 特征
   相对 degradation-only 与连续-D 基线携带可实现决策价值；structure-conditioned 剂量+abstain
   同时消除 F0 季节性结构伤害（season 全子群非劣于 incumbent 且均值为正）并保留 trend 判官
   口径增益；该策略作为**冻结产物**跨样本迁移成立、其**学习程序**可重复。
2. **未确立**：①下游模型无关的全面增益（C6：trend 剂量收益判官特异）；②全部结构子群全局
   非劣（overall worst q05 −0.41，trend 重尾）；③跨真实 domain 泛化（level-2 未做）；
   ④LLM/self-evolution 必要性（八臂全零-LLM，D-3.2a/b 未答）。
3. **报告器稳健的部分**：season 安全修复（两报告器、两基线全同向）；d_lookup 的 season 伤害
   在判官与 chronos 下均复现 → "degradation-only 路由结构不安全"为测量口径稳健结论。

## 6. 产物清单

`confirmatory_freeze.json` / `frozen_arms.joblib`（+SHA）/ `results/A38C/`（protocol→manifest→audit）/
`locked_transfer/`（records_locked_{scope}.jsonl、lt_point_{scope}.json、fullrefit_{scope}.json、
reporter_panel_primary_no_Sar.json、ckpt）/ `replication/`（records+replication_report.json）/
`confirmatory_report.json`（布尔表）。守卫=`tests/test_confirmatory.py`（12 测）。
