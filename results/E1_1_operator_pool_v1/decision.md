# E-1.1R2 — 良定性 gate + degradation/pattern 分开估计（Stage 1 首实验，最终锁定口径）

日期：2026-07-03
脚本：`run_variance_decomp.py --n-seeds 20 --boot 1000 --perm 1000`（judge=frozen；OOF grouped-5fold Ridge，**bootstrap 内重拟合 + 重数加权训练**；**6 动作良性池**）
语料：方差匹配结构族（S_season/S_trend/S_both/S_ar）× 退化网格（hi/lo SNR × full/miss）× 20 seed = **320 forecast series** → 4 cells，全非 LOWCONF（n=72–88）。

> **本文取代 E-1.1（预实验）与 E-1.1R**。三轮迭代 = 建立事实（cell_id 只按 SNR×missing → cell oracle ≡ degradation router）→ 修 3 实现 bug（A-27）→ 修 2 残留 bootstrap bug + 动作池缺陷（A-28）。
> **A-28 修正后的内部一致性验证**：L2 权重改用重采样重数后，bootstrap share（76.1/23.9）**与点估计（75.4/24.6）吻合**——旧版权重不一致时 bootstrap 给 94.4/5.6，修后归位 → 坐实 A-28a 正确。且 **L0≥L1≥L2 逐 replicate 断言全程通过**。

## 三层 ORACLE 分解（nRMSE，cell 等权，B=1000 CI；L0/L1/L2 **均为数据内 oracle 选择值**）

| 层级 | 定义 | nRMSE (95%CI) | 增益 (95%CI) | 判定 |
|---|---|---|---|---|
| L0 | global-oracle（单一最优动作 v_median） | 1.5102 [1.315, 1.711] | — | — |
| **L1** | **degradation-conditioning oracle**（每 cell 最优动作：snrHigh→v_median / snrLow→v_stl） | 1.4576 [1.269, 1.651] | **+0.0526 [+0.003, +0.120]** | **可靠正** |
| L2 | structure-aware oracle（**in-sample 信息上界**，乐观） | 1.4410 [1.259, 1.631] | +0.0166 [+0.006, +0.033] | 正但小、**上界** |

- **degradation conditioning 显示稳定的 oracle value（+0.053，CI 可靠>0）。**
- **cell 内 structure 残差（乐观上界）小（+0.017）**：修不变量后 CI 已不含负（[+0.006,+0.033]），但这只是"in-sample 每-origin oracle"的上界；**可实现决策价值未确立**。
- 描述性 share（**非 headline**，A-28c）：degradation 76.1% / pattern(UB) 23.9%——与点估计 75/25 一致。近零量转百分比不稳定，不作论断。
- **L0/L1 本身是 oracle 选择（在数据上选 cell-best action）**；可实现的 degradation router 仍须 **E-3.2 held-out Lookup** 验证。

## 预注册判据裁决

### D-1.1a/b 良定性 gate — **"function"（50% 阈值，marginal）**
| cell | n | winner | win_rate | gap CI（固定 top1/top2，重数加权重拟合） | well-defined |
|---|---:|---|---:|---|:--:|
| snrHigh\|full | 88 | v_median | 1.00 | [+0.245, +0.586] | ✅ 稳固 |
| snrHigh\|miss | 78 | v_median | 0.98 | [+0.034, +0.500] | ✅ 稳固 |
| snrLow\|full  | 72 | v_stl    | 0.94 | **[−0.001, +0.293]** | ⚠ marginal（win_rate 过、gap CI 下界擦 0） |
| snrLow\|miss  | 82 | v_stl    | 0.73 | [−0.137, +0.227] | ❌ |

- **2/4 稳固良定（50% = 阈值）→ D-1.1b = "function"（勉强）**。**稳健白名单 = 2 个 snrHigh cell**；snrLow\|full 可用但须标 marginal；snrLow\|miss 排除；真 Monash 全 LOWCONF 排除。
- 相对 E-1.1R（3/4）：R2 补上训练重数方差（A-28b）后 CI 略宽，snrLow\|full 的 gap 下界从 +0.052 掉到 −0.001 → 更保守、更诚实。**N 是良定功效变量**（低 SNR cell 少数结构样本少）。

### D-1.1c 响应维度（**6 动作良性池**）— **rank ≈ 2.3（非一维轴，稳固）**
- median 首奇异值方差占比 **0.60**（<0.90）、median 有效秩 **2.34**、rank-2 重构误差 0.15–0.21。
- **动作池强敏感（A-28d 关键旁证）**：`v_wavelet` 病态（pywt db4 soft-threshold 过收缩，nRMSE≈2×v_none）——**含它** first-SV→0.93/rank→1.1（伪 single-axis）、**剔它** rank≈2.3。**证明 SVD 秩是当前动作池的性质、非真实处理空间维度**（坐实 D-1.1c 解释边界）→ wavelet 算子缺陷 + 秩的池依赖性一并交 **E-3.3 供给侧**。

### structure×action 交互（SNR **粗**分层置换控混淆）— **交互真实、决策价值未确立**
| cell | 各结构 winner | 交互占比 | perm p（普通） | perm p（**SNR 分层**） |
|---|---|---:|---:|---:|
| snrHigh\|full | season→**winsor**, trend/both→median | 0.21 | 0.001 | **0.003** |
| snrHigh\|miss | season→**stl**, trend/both→median | 0.13 | 0.073 | 0.252 |
| snrLow\|full  | ar/both/season/trend→大多 **v_stl** | 0.57 | 0.001 | **0.001** |
| snrLow\|miss  | ar→**winsor**, 其余 stl/median | 0.31 | 0.209 | 0.290 |

- **2/4 cell（snrHigh\|full, snrLow\|full）交互显著且存活 coarse SNR 分层置换** → 那部分结构效应非纯 SNR 伪影。
- **但"交互真实"≠"路由有用"**：cell 统一最优动作对每结构近乎一样好（near-tie 0.23–0.42）→ L1→L2 增益仅 +0.017（上界）。**核心区分不变**：结构携带"哪个动作最优"的可测信息，利用它买到的效用增益小且仅为上界。

## 归因谨慎（A-28e）
- S_ar 整族落 snrLow（SNR≈−6.8 vs 其余 3–9）；snrLow\|full origin 分布极不均衡（S_ar 40 / S_both 9 / S_season 18 / S_trend 5）→ 三分位 SNR 分层是**粗**控。可写 "interaction survives **coarse** within-cell SNR-stratified permutation"，**不可**写"已与 SNR 完全解耦"。后续用连续 SNR residualization / 匹配加强。

## 综合裁决（方向经三轮修正稳定）
1. **degradation router = 确立的主力**：oracle value +0.053 可靠>0，2 个高 SNR cell 稳固良定。可实现版仍须 E-3.2 held-out Lookup。
2. **pattern 的可实现决策价值 = 未确立**：in-sample 上界仅 +0.017（占比 ~24%，描述性），交互真实但决策增益小、且只是上界 → **必须 E-3.2 held-out policy regret 裁**。
3. **点估计方向经全部修正不变**（deg 主导 ~76%、pattern ~24% 上界）→ **满足进入 E-3.2 的前提**（评审判据："若三层点估计方向不变则进 E-3.2"）。

## 下一步
1. **E-3.2 决胜**（推荐）：degradation-only Lookup vs full-pattern GBDT policy 的 **held-out policy regret**（arm-in / arm-out LODO）。E-1.1R2 预测：pattern 优势小、可能显著于 snrHigh\|full / snrLow\|full（交互存活的 2 cell），跨域多半不过。
2. **E-3.3 供给侧**：修/换 wavelet 算子（现过收缩）、加库外算子看 rank 是否升——秩的池依赖性已证。
3. 补低 SNR 少数结构样本（S_trend/S_both 在 snrLow 仅 5–11）以收紧 L2 上界与 snrLow\|full 良定性。
4. Stage-2 报告基建 A-25/A-26（非阻塞）。

## 产物
`results/E1_1/report.json`（全统计 + 三层 CI + SVD bootstrap + 不变量断言）、`results/E1_1/response_matrix.csv`（完整长表 cell,uid,origin,snr,fold,action,oof_nrmse，可独立复算）。6 动作良性池；wavelet 病态列已剔并记档。
