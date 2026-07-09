# E-3.3 Family 0 剂量扫描（exploratory）— 全 4 cell 完成

> ⚠ **已被正式收尾修正（评审第十三轮，见 `results/E1_1_family0_final/`）**。本 exploratory 的
> 两处结论须按 final 读：①「无一 cell 受损 / 全局安全」**只成立于 cell 层**——cell×origin 分解显示
> `snrLow|full` 内 **S_season 子群被 F0 扩池重伤 ≈ −0.67**（窗 25 ≈ 主周期 24 抹平季节性），
> degradation-only 路由**结构上不安全**；②per-cell B=100/300 是 smoke，final 用 B=1000。
> 数字保留原样，判决以 final 为准。

日期：2026-07-04
脚本：`run_family0.py`（`family0_actions.F0_DOSAGE_GRID`：median 9/15/25、savgol 21/31、MA 9/15/25）
base = operator_pool_v2.1 的 7 动作；expanded = base + 8 剂量动作；两池共用 outer folds（paired）。
Δ_supply = nested held-out（`nested_supply`）；**正式 CI = full-refit group bootstrap（A-33c）**。
decisive cell `snrLow|full` 用 B=300；harm cell（snrHigh×2 + snrLow|miss）用 B=100（per-cell n_boot 见字段）。
**confirmatory seeds 20–39 未开**（A-32d 分支：见 §3）。产物 `results/E1_1_family0/report.json`。

## 1. 决定性 cell `forecast|snrLow|full`（witness 所在，72 uids）

**问题**：v2.1 池最优 v_median 1.6823，池外 v1-STL 1.5406（差 0.1417）——更强平滑剂量能否在
held-out 上补上这道供给缺口？

### 1a. 方向/机制：决定性证实（池缺强剂量维）
in-sample col_mean（诊断口径，不判决）：

| 动作 | col_mean | 备注 |
|---|---:|---|
| v_median（在任冠军） | 1.6823 | v2.1 池最优 |
| **f0_median_w25** | **1.4897** | **低于 v1-STL witness 1.5406** |
| f0_ma_w25 | 1.5508 | ≈ witness |
| f0_median_w15 | 1.6253 | |
| f0_median_w9 | 1.6459 | |
| f0_savgol_w31 | 1.9005 | 全池最差（见 §1c） |

- 8 剂量动作中 **5 个** in-sample 击败 v_median；
- **held-out 逐 fold 选择：f0_median_w25 五折全票**；
- bootstrap 选择频率：f0_median_w25 **59.3%** + f0_ma_w25 **23.1%** = 82% 强剂量；v_median 仅 11.1%。

→ "**供给缺 dosage 维度**"假设结构上成立：补进强 median 剂量后，held-out 选择一致偏好它，
把该 cell 的 held-out loss 从 base **1.7497** 拉到 expanded **1.5916**（witness 量级）。

### 1b. 幅度：诚实 CI 不显著（CI 升级翻转裁决）
| CI 口径 | Δ_supply | 95% CI | pos% | 过 D-3.3v3.1 门② |
|---|---:|---|---:|:--:|
| 单次-nested（test-uid，便宜） | +0.2055 | [+0.0076, +0.4866] | — | ✅（会误判显著） |
| **full-refit grouped（A-33c，诚实）** | **+0.1581** | **[−0.1086, +0.5773]** | 0.737 | ❌（下界<0） |

- 点 Δ +0.1581 **超过** witness gap 0.1417；median +0.148。
- 但纳入①选择不稳定②头重拟合③fold 划分④组内相关四类方差后，效应在 72-uid 单 cell 上
  **95% 不显著**（74% replicate 为正）。
- **方法学结论**：便宜的 test-uid CI 会误判显著（过门②），A-33c 的 full-refit group bootstrap
  纠正了它——这是 winner's-curse/选择不稳定性在真实数据上的实证，兑现了评审第十二轮的 CI 升级要求。
- 不显著根源是**小样本**（72 uids + 选择方差 → CI 宽），非无效应 → 指向 A-31e 补低 SNR 样本
  （收窄本 cell 的最干净路径，且为 E-3.2 硬前置）。

### 1c. savgol@31 被选择器正确拒绝（gate 预测兑现）
`f0_savgol_w31`（窗 31 > 语料主周期 24、polyorder 3）端点多项式过冲 → 全池最差（1.9005）、
从未被任何 fold 选中。池对过量平滑自我调节——A-33d 预注册的"剂量质量真实信号（非零填充伪影）"
在数据上兑现。

## 2. 全 4 cell 结果 — 无 harm + SNR→剂量单调路由

| cell | uids | 在任冠军 v_median | **held-out 选择** | Δ_supply grouped | 95% CI | pos% | B |
|---|--:|--:|---|--:|---|--:|--:|
| snrHigh\|full（SNR 最高） | 88 | 1.0146 | f0_median_**w9** ×5 | +0.0345 | [−0.067,+0.138] | 0.75 | 100 |
| snrHigh\|miss | 78 | 0.9076 | f0_median_**w15** ×5 | +0.1238 | [−0.039,+0.255] | 0.92 | 100 |
| snrLow\|miss | 82 | 1.6478 | f0_median_**w25/15/9** | +0.0981 | [−0.076,+0.372] | 0.76 | 100 |
| snrLow\|full（SNR 最低） | 72 | 1.6823 | f0_median_**w25** ×5 | +0.1581 | [−0.109,+0.577] | 0.74 | 300 |

**cell 等权 Δ_supply = +0.1036**（点，>ε=0.03）。三个关键事实：

1. **每个 cell 都欠剂量 + SNR→强度单调路由**：所有 cell 的在任冠军都是 v_median@w5（默认窗），
   而 held-out 选择在每个 cell 都换成**更重的 median 剂量**，且窗宽随 SNR 下降单调递增
   （w9→w15→w15/25→w25）。per-cell router 选出**随 degradation 分级**的剂量——这正是
   "**dosage 维缺失且产生可路由异质性**"的形态，且单调可解释（非噪声）。
2. **无一 cell 受损**：4/4 点 Δ_supply>0；无 cell 显著为负（snrHigh 最差 CI 下界 −0.067、点 +0.034）。
   → f0 剂量动作**全局安全可加**（分支不落"伤高 SNR 的专才"）。
3. **剂量家族已分出胜负**：median 是赢家；`f0_ma_*`（滑动均值）在每个匹配窗被 median 支配
   （如 snrHigh|miss：ma_w15 1.35 vs median_w15 0.796）；`f0_savgol_*` 在每个 cell 最差或近最差
   （savgol_w31：1.90/1.86/2.06），从未被选中——§1c 的端点过冲预测跨 cell 稳健兑现。

**幅度/显著性**：per-cell 诚实 CI 全部跨 0（无单 cell 过门②）；snrHigh|miss 最接近（pos% 0.92、
single CI 下界 +0.011）。方向一致性 4/4 正 + 单调 + 无 harm 是**强定性证据**；缺一个**严谨的
cell 等权聚合 CI**（joint stratified bootstrap；当前只有 per-cell CI 与点聚合，未存 bootstrap 数组
无法事后合成）。

## 3. 分支判定（A-32d / A-30d）

**A-30b 预注册逻辑兑现**：剂量扫描**复现了 witness**（snrLow|full held-out 由 base 1.750 恢复到
expanded 1.592≈witness 1.54；点 Δ 超过 witness gap 0.142）→ 结论 "**缺 dosage 维度**" 成立，
**F1–F5 机制族降为补充**（LOWESS/robust-STL/period-imputation/wavelet/Hampel 不再是主线）。

对 A-30d 五条方向判据：①强平滑成为两个 snrLow cell 的 held-out selected action ✅；
②Δ_supply(cell 等权)>ε 点成立、per-cell 2SE 未过 ⚠；③deg-conditioned 路由>全局单动作（SNR→剂量
单调）✅；④≥1 independent reporter 同向 —— 未算（confirmatory）；⑤snrHigh/worst-group 无显著回退 ✅。
→ **①③⑤ 明确达成，②点达成/CI 未达成，④待办**。

**裁决**：F0 **机制/方向决定性坐实**，但 **magnitude 未达 per-cell 显著**（小 n）→ 既非"无效"（不转
纯 F1–F5），也非"四门齐可冻结开 confirmatory"。**不打开 seeds 20–39**。收尾形式：
> "动作池缺少平滑**强度**维度，补入后 per-cell router 选出随 SNR 单调的剂量、无 harm、cell 等权
>  held-out 增益 +0.10（点）；单 cell 显著性受 72–88 uid 小样本限制。"

**通向可冻结的最干净两步（均待你拍板，未执行）**：
- (a) **A-31e 补低 SNR 样本** → 收窄两个 snrLow cell 最宽的 CI（本就是 E-3.2 硬前置）；
- (b) 给 runner 加 **cell 等权聚合 CI**（joint stratified bootstrap）→ 直接测 A-30d②（聚合 Δ_supply>max(ε,2SE)）；
- 之后把 **F0 候选池剪枝为 median 剂量族**（+f0_median_w9/w15/w25，弃被支配的 ma_* 与有害的 savgol_*，
  减选择噪声）并**在 confirmatory seeds 20–39 上一次性确认**（A-30c/A-31d：数据驱动的网格剪枝须在
  从未参与选择的 holdout 上确认）。

## 4. 解释边界（评审强调）
F0 成功只证 **degradation/SNR 条件化的剂量路由**有价值——selection 单位仍 cell=SNR×missing。
是否由**内部 Pattern**（而非 degradation）预测该剂量维，须留 E-3.2（P-only/D-only/D+P/cell 内结构/LODO）。
**注（final 补充）**：cell×origin 分解显示，在固定 degradation cell 内**最优剂量随结构翻号**
（S_season 抗重 median、S_trend 嗜重 median）——这本身就是 Pattern 携带 degradation 之外决策价值的
直接证据，把 E-3.2 从"待验证"提为"被 F0 数据点名"。
