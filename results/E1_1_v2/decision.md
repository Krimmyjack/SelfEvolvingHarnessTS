# E-1.1R2 / operator_pool_v2 — 算子身份修复后重跑（v1↔v2 对照）

日期：2026-07-03
脚本：`run_variance_decomp.py --n-seeds 20 --boot 300 --perm 500 --out results/E1_1_v2`（operator_pool_v2，7 语义真实动作）
对照 v1（`results/E1_1_operator_pool_v1/`，B=1000，6 动作、含病态/静默回退算子——**冻结存档不覆盖**）。
前置：S0.7 Operator Integrity Gate（修 `_guess_period`/`denoise_wavelet`/`impute_fft` + provenance 台账，116 测过）。

> **B 说明**：v2 用 B=300（环境限制长任务；点估计与方向 B 无关，CI 略宽于 v1 的 B=1000）。已向量化 bootstrap 内层。

> **⚠ 追录警示（2026-07-03，A-31b/S0.7-8，详见文末追录）**：`denoise_median` 的零填充 medfilt 边界缺陷被证实**压低了 v_median**（诊断实测修复后均值 OOF 1.4755→1.3131）。本文件所有含 v_median 的量（L0/L1/headroom/witness 对比/良定 gap）已被 **v2.1 完整重跑取代（`results/E1_1_v2_1/decision.md`，2026-07-03）**：仅 v_median 列变（320/320，其余 bit 级一致），v_median 4/4 全胜、良定 3/4、routing_gain(deg)=+0.0043 CI 上界 0.0283<ε、processing_gain +0.4607、witness 缩为 snrLow|full 单 cell。**E-3.3 基线以 v2.1 为准，本文件转历史对照。**

## 算子身份修复实证（provenance 台账）
- `denoise_stl → {真 STL: 109, savgol 回退: 211}`：**此台账是 v2 运行的记录**（provenance 模块 S0.7 才建）。v2 中 v_stl 只在 ~34% 有真实季节的序列上跑真 STL，其余**显式**回退 savgol。**v1 无台账、无 git，具体 fallback 比例不可追溯**（等式重建也无效：OOF 损失经共享 Ridge 头耦合整列，逐 uid 相等性不能反推算子相等性——A-29a）。从行为看 v1 的 v_stl 在全部 320 uid 上与 v2 不同、且在 snrLow 表现为激进平滑，与"garbage-period STL 广泛真跑（而非回退）"一致，但这是推断非台账实证。
- `denoise_wavelet → {wavelet: 320}`：修复后真跑（v1 因 pywt 缺失 320/320 静默 == savgol）。
- deps 指纹：numpy 2.2.6 / scipy 1.15.2 / statsmodels 0.14.6 / pywt 1.8.0 / sklearn 1.7.2。semantic-dup：无。

## v1 → v2 关键对照

| 量 | v1（算子破损） | **v2（算子诚实）** | 变化 |
|---|---|---|---|
| **gain L0→L1（退化路由 oracle）** | **+0.0526**，CI[+0.003,+0.120] 可靠正 | **+0.0078**，CI[+0.000,+0.040] | **塌到≈0**（CI 下界触 0） |
| gain L1→L2（pattern in-sample 上界） | +0.0166，CI[+0.006,+0.033] | +0.0196，CI[+0.009,+0.034] | 基本不变（仍是上界） |
| eff_rank（响应维度） | 2.34（6 动作） | 2.71（7 真动作） | 真 wavelet 加一维 |
| 良定 cell | 2/4（snrHigh×2） | 2/4（snrHigh×2） | **稳定** |
| snrLow\|full point winner | v_stl (1.541) | **v_winsor_savgol (1.689)**，win_rate 0.49 | winner 变、nRMSE 更高 |

## 机制（为什么退化路由价值塌了）
- v1 的退化路由 gap(+0.053) 来自 **snrHigh→v_median / snrLow→v_stl** 两 regime。
- 但 v1 的 `v_stl` 在 snrLow 上 = **破损 STL**（周期估计被趋势劫持 → garbage-period STL / savgol），它在高噪数据上**碰巧激进平滑 → nRMSE 反而更低(1.541)**，制造出"snrLow 偏好 stl"的假象。
- 修好后 `v_stl` 尊重季节性、对非季节高噪序列少平滑 → 不再在 snrLow 取胜；v_median/winsor_savgol 接管，且各 cell winner 趋同(v_median 广赢) → **跨 cell winner 差异消失 → 退化路由 gap→0**。
- **结论：v1 的退化路由价值实质是一个算子 bug 的产物。** anchor 的 +0.033、v1 的 +0.053 都受此污染。

## 修复后判据裁决（以 v2 为准）

### D-1.1a/b 良定性 — "function"（2/4，marginal，且**跨 v1/v2 稳定**）
| cell | winner | win_rate | gap CI | WD |
|---|---|---:|---|:--:|
| snrHigh\|full | v_median | 1.00 | [+0.125, +0.697] | ✅ |
| snrHigh\|miss | v_median | 0.99 | [+0.057, +0.502] | ✅ |
| snrLow\|full  | v_winsor_savgol | 0.49 | [−0.067, +0.157] | ❌（winner 掷硬币） |
| snrLow\|miss  | v_median | 0.91 | [−0.017, +0.179] | ❌ |
- 稳健白名单 = 2 个 snrHigh cell（v_median），跨算子版本稳定。两 snrLow cell 修复后**更**不良定（winner 换、win_rate 0.49）。

### D-1.1c 维度 — rank ≈ 2.71（7 真动作，非一维）
- 但 SVD 秩**强依赖动作池**（wavelet 病态→1.1 / 重复→虚高 / 修好→2.71 已证）→ 秩是池的性质、非真实处理空间维度。

### 三层分解 — **routable oracle 价值在所有方向都很小**
- L0 1.5168 → L1 1.5090（deg gain **+0.0078**，CI[0.000,0.040]）→ L2-UB 1.4894（pat gain **+0.0196**，CI[+0.009,+0.034]）。
- **两个 gain 都很小（~0.008 / 0.020 nRMSE），总 routable oracle 价值 L0→L2 仅 ~0.027。**
- 描述性 share 28.5%/71.5%（deg/pat）**严禁作 headline**：两个近零量之比，不稳定且误导（评审 A-28c）。且 **L2≤L1 由构造成立 → pat gain CI 下界>0 不是显著性证据**，只是"in-sample 上界非零"。

### structure×action 交互 — 2/4 存活 coarse SNR 分层（同 v1）
- season 偏好真 STL、trend/both 偏 median/winsor_savgol、ar 偏 median/winsor——交互真实但**决策增益仍是上界**。

## 综合裁决（operator_pool_v2，诚实口径；A-29 修订措辞）
1. **旧退化路由增益不再复现（"未建立"，非"证伪为零"）**：修复算子后 deg-conditioning oracle gain +0.008。注意 **L0≥L1 由构造成立** → gain 非负是机械事实，CI 下界触 0 不是标准零效应检验（与 L2≤L1 的既有 caveat 对称，A-29b）；且 CI 上界 0.040 > ε=0.03 → 当前功效下也**不能断言其为零**。可靠的说法：v1/anchor 的退化条件化证据（+0.053/+0.033）**主要由破损 STL 制造，已失效**；v2 下没有可靠证据支持有实际意义的 degradation routing value。
2. **良定性稳定**：2 个高 SNR cell 稳固良定（v_median），跨算子版本不变；低 SNR cell 不良定。注意：2/4 良定证明的是 **v_median 在这两个 cell 稳定胜出**（H* 局部稳定），**不构成"不同 cell 需要不同策略"的证据**——恰相反，v2 下条件异质性很弱，global v_median 已近 oracle。
3. **pattern 决策价值仍未确立**：结构残差上界 +0.020（in-sample、乐观），可实现值须 E-3.2 held-out regret。
4. **供给不足 = 最强候选解释（假设，待 E-3.3 判定；但已有构造性 witness）**：在诚实动作池里，"永远 v_median" 已接近 per-cell/per-struct oracle，任何路由的 oracle 头顶只有 ~0.02–0.03。**witness（来自冻结 v1 矩阵，同判官同 OOF 机器）**：v1 的 v_stl 在 snrLow|full 达 **1.5406，比 pool_v2 最优动作(1.6888) 好 0.148**、snrLow|miss 1.6752 vs 1.7089 (+0.034)——**池外存在对低 SNR 特别有效的确定性变换**。其机制="garbage-period STL 的激进平滑"系**推断**（修复时观察过旧 `_guess_period` 行为，但 v1 无代码/台账可重放，A-30e）；确证的只是该响应列的存在与表现。→ E-3.3 检验诚实版强平滑（先剂量扫描，A-30b）能否实现它，判据用 nested held-out Δ_supply（A-30a，防 action-set expansion winner's curse——实测模拟 K=10 纯噪声动作即可机械通过旧 in-sample gate）。若合并池 Δ_supply 仍≈0，转查弱形式备择（强 forecaster/真实数据/判官平滑偏好/跨任务性，A-30e）。

## 下一步（据 v2 重排优先级）
1. **E-3.3 供给（升为首要）**：加/修强去噪与结构感知算子（LOWESS、robust-STL、period-aware completion、variance stabilization），看 per-cell oracle 与 rank 是否上移——**routable 价值小是不是因为池太弱**。
2. **E-3.2（降级但仍做）**：Global / D-only / P-only / D+P GBDT 的 held-out policy regret（评审 6 臂设计）——此时是检验"~0.02 上界残差可否实现"，预期收益很小。
3. **补低 SNR 少数结构样本**（S_trend/S_both 仅 5–11）+ 连续 SNR residualization 强化归因。
4. Stage-2 报告基建 A-25/A-26；S0.7-6（fill_gaps 去重 / EMA 正名 / task 级 destructive 硬过滤）在 anomaly/Stage-2 前修。

## 补录（2026-07-03 分析，A-29）：v1↔v2 逐动作 diff + 供给 headroom 定量

**逐动作 diff（同 320 uid，冻结矩阵直算）**：v_none/v_median/v_savgol/v_winsor/v_winsor_savgol **bit 级一致（0/320 变化）**；仅 v_stl 变化（320/320，max|Δ|=2.58）→ 翻案的全部变化**唯一归因于 STL 修复**，因果链闭合。逐 cell v_stl：snrHigh|full 1.837→1.746（**修好后更好**，真 STL 帮季节序列）、snrHigh|miss 1.545→1.501、**snrLow|full 1.541→1.762、snrLow|miss 1.675→1.884**（破损版=激进平滑在高噪碰巧赢）。

**供给 headroom（v2 原始 OOF 点估计，cell 等权；A-30e 措辞）**：
| 量 | 值 | 含义 |
|---|---|---|
| v_none → L1(cell-best) | **+0.303** | **相对最低处理基线 v_none(=impute_linear，非纯 raw) 的附加处理价值** = 10×ε；且限定于 frozen probe+合成语料条件 |
| v_none → v_median(全局单动作) | +0.307(uid 等权) | 且几乎全部被单一全局动作捕获 |
| v_none → per-series oracle(池内) | +0.435 | 含选择噪声、乐观 |
| L0 → L1（路由价值） | +0.004(原始)/+0.008(bootstrap) | ≈0 |
| 逐 cell none→L1 | snrHigh +0.55/+0.46 vs **snrLow +0.11/+0.09** | 池在 snrLow 最弱（witness 所在地） |

**三个直接推论**：①"数据不需要预处理/判官对处理不敏感/预处理无下游价值"三个备择解释被**强否定**（+0.30 的响应）；②但绝对 nRMSE 全 >1（substrate 弱，编码器不破 seasonal-naive）→ 预处理价值在强 forecaster 下可能缩水，E-3.3 须带 independent reporter 方向核验；③**供给与路由价值耦合**：把 v1-STL 行为等价物（诚实命名的激进平滑）加回 pool_v2，L1 从 1.4712 → 1.4257（**ΔL1=+0.046 > ε**），且 deg-routing gain 从 +0.004 **回升到 +0.050**（winners 变 median/median/smoother/smoother）——**v1 的路由价值本身可能是真的（高噪→重平滑是合理物理），假的只是动作的语义标签**（此耦合计算用的是 v1 真实响应列回填，数字精确；"激进平滑"机制标签系推断，A-30e）。E-3.3 预注册预测 P1（A-30d 方向性版）：强平滑族成为 ≥1 个 snrLow cell 的 held-out 赢家、Δ_supply 过 gate、deg-Lookup>global、独立 reporter 同向、snrHigh 无回退；deg gain 回升 +0.03~+0.06 仅作期望区间。

**语料备注**：snrLow cell 内 per-origin SNR 不均（S_ar ≈ −6.8dB vs 其他 ≈ +3dB）→ cell 内 SNR 同质假设不完美，补样/连续 SNR residualization 时一并处理。

## 追录 2（2026-07-03，评审第十一轮 / A-31）：边界伪影 + nested 实现口径

**S0.7-8 边界语义缺陷（追溯污染本文件数字）**：`denoise_median` 用零填充 `scipy.signal.medfilt`（末端 2 点被拉向 0=z-score 下拉向均值，但实际效果是**破坏编码器末窗输入**）；`moving_average` 用 `np.convolve(mode="same")` 同类。诊断实测（`diagnostics/boundary_diag.py`，reflect 修复对照，320 series，**中段 bit 级一致 → 变化纯由边界产生**）：

| cell | v_median 当前(零填充) | v_median 修复(reflect) | Δ | v_none |
|---|---:|---:|---:|---:|
| snrHigh\|full | 1.3642 | **1.0146** | +0.3496 | 1.9104 |
| snrHigh\|miss | 1.1228 | **0.9076** | +0.2152 | 1.5874 |
| snrLow\|full | 1.7060 | 1.6823 | +0.0237 | 1.7999 |
| snrLow\|miss | 1.7089 | **1.6478** | +0.0611 | 1.7976 |
| **mean** | 1.4755 | **1.3131** | **+0.1624** | 1.7738 |

- **方向出乎意料**：零填充不是让 v_median"占便宜"，而是一直**拖累**它——修复版单动作均值 1.3131 已好于当前全池 L1 1.4712。
- **对本文件的影响**：所有含 v_median 的量（L0/L1/deg gain/headroom +0.303/witness 对比/良定 gap）**待 v2.1 重验**。初判方向：v_median 更强 → headroom 变大、deg gain 更接近 0、供给假设定性不翻转。
- **witness 初判（诊断点估计）**：snrLow|full **存活**（v1-STL 1.5406 vs 修复池最优 ≈1.682，仍差 ~0.14）；snrLow|miss **翻转**（修复 v_median 1.6478 已优于 v1-STL 1.6752）→ witness 缩为单 cell，P1 以 v2.1 为基准。
- **对 E-3.3 的含义（关键）**：若不修就跑 E-3.3，新平滑动作会对着被压低的 v_median 基线**虚报 Δ_supply**——边界修复 + v2.1 重验是 E-3.3 硬前置。

**A-31a nested 实现口径**：本目录 `response_matrix.csv` 的 OOF loss 经共享 Ridge 头跨 fold 耦合（每 loss 的头在全部其他 fold uid 上训练）→ **不能事后重切成 nested held-out**；只可用于探索性分析与模拟。正式 Δ_supply 须重算：inner grouped CV 选动作 → outer-train 重拟合头 → 仅 outer-test 评估（缓存 PhiX/Y/PhiTest/future/obs 后为廉价 Ridge 重拟合）。

## 产物
`results/E1_1_v2/{report.json,response_matrix.csv,decision.md}`（provenance=依赖指纹+fallback 台账+semantic-dup 守卫）。v1 冻结存 `results/E1_1_operator_pool_v1/`。诊断脚本（补录用，已入库可复查）：`SelfEvolvingHarnessTS/diagnostics/diff_v1_v2.py`（v1↔v2 diff+headroom，只读冻结矩阵）、`diagnostics/boundary_diag.py`（边界伪影定量，现算不落 results/；修复合入后改为显式对比人为零填充版与当前版）。
