# E-3.2 六臂 + 诊断臂（level-1 平衡 dev 语料）— 判决

日期：2026-07-04　runner：`run_e32.py --dev`（A-37 协议 + A-39 修复包，freeze sha=`f1b6c1b75a4e975c`）
语料 = dev(20) + A-31e 补样（560 uid；primary 口径排除 S_ar → 480）；动作池 = 剪枝 10 核心
（pool_v2.1 + f0_median_w9/15/25）；评估器 = **标签生成嵌入 policy outer fold**（`e32_nested.py`，
A-39① P0 修复，六守卫 9 测全过）；总用时 44 min；**confirmatory seeds 20–39 未触碰**。

---

## 0. 一句话判决

> **D-3.2e 七项全过（primary 口径）**：structure-conditioned 剂量 + abstain **同时**显著胜过
> global 与 degradation-only 路由（paired bootstrap CI 全部不跨 0）、**消除 F0 点名的 S_season
> 结构 harm 并显著改善总体 worst-group（−0.427→−0.083）——但未证明所有结构子群全局非劣**
> （评审第十六轮措辞修正，见 §5.2）、trend 增益不仅保留而且放大（154%）、P 增量过连续 SNR
> residualization（perm p=0.01=最小可能值）**且不能仅由当前定义的生成器 noise/missing 参数
> 解释（true-D 对照）**。**F0 的可证伪靶全中——可观测 Pattern 特征在 degradation 之外携带
> 可实现决策价值（development、平衡合成语料、common-support 口径，level-1）。**

## 1. D-3.2e 判据（primary_no_Sar，n=480）

| 判据 | 结果 | 数字 |
|---|:--:|---|
| (i) regret(D+P) < Global − ε | ✅ | 0.263 vs 0.492；CI[−0.301,−0.160]，pos% 0.000 |
| (ii) regret(D+P) < D-only Lookup − ε | ✅ | 0.263 vs 0.475；CI[−0.264,−0.158] |
| (iii) S_season worst-group LCB > −δ_safe | ✅ | dp_abstain 全 season 子群 LCB ≥ −0.0103（对照 d_lookup：−0.90/−0.63） |
| (iv) trend 保留 ≥50% D-only 增益 | ✅ | **154%**（dp_abstain Δ_trend +0.855 vs d_lookup +0.554）|
| (v) abstain 不劣化 worst-group | ✅ | worst LCB −0.427(dp) → **−0.083**(abstain)；mean regret 还降 −0.028 |
| (vi) 过连续 SNR residualization | ✅ | vs d_gbdt(连续 D)：CI[−0.262,−0.135]；**分层置换 p=0.01**（T=+0.199, null −0.005）|

## 2. 臂表（arm-in，mean regret / worst-group LCB / abstain 率）

| 臂 | regret | worst LCB | 备注 |
|---|--:|--:|---|
| global | 0.4915 | −0.310 | |
| d_lookup | 0.4745 | **−0.900** | **F0 season harm 在 policy 空间复现**（S_season −0.846/−0.537）|
| d_gbdt（连续 D）| 0.4619 | −0.538 | (vi) 对照 |
| p_gbdt | 0.2752 | −0.201 | P 单独已达大部分增益 |
| dp_gbdt | 0.2633 | −0.427 | 弱点=snrLow\|full\|S_trend 方差 |
| **dp_abstain（候选策略）** | **0.2352** | **−0.083** | abstain 21%，**集中于 S_season（42/38/20/18%）** |
| oracle_struct（诊断上界）| 0.1393 | −0.007 | dp_abstain 覆盖 d_lookup→oracle 差距的 **71%** |
| true_d_gbdt（诊断）| 0.3737 | −0.427 | **D+P 显著胜 true-D**（CI[−0.185,−0.035]）→ Pattern 增量**不能仅由当前定义的生成器 noise/missing 参数解释**（不排除未编码的退化因素：尺度/结构相关有效噪声比/局部异常等，评审十六轮降级措辞）|

全部主比较 CI = paired-uid bootstrap B=2000（预注册 caveat：条件于已拟合头/router，
未含重拟合方差；confirmatory 升级 grouped full-refit，A-40⑤）。

**补充（评审第十六轮，`supplement_dp_abstain_cis.json`，零重训）——最终候选策略 dp_abstain
自己的直接 paired CI**（原 report 判据 (i)(ii)(vi) 用 dp_gbdt）：

| 比较（primary） | mean | 95% CI | pos% |
|---|--:|---|--:|
| dp_abstain vs global | −0.2563 | [−0.3229, −0.1987] | 0.000 |
| dp_abstain vs d_lookup | −0.2393 | [−0.2947, −0.1858] | 0.000 |
| dp_abstain vs d_gbdt（连续 D）| −0.2266 | [−0.2906, −0.1615] | 0.000 |
| dp_abstain vs true_d_gbdt | −0.1385 | [−0.2044, −0.0695] | 0.000 |

all_data 同向全过（vs d_lookup −0.113 CI[−0.171,−0.053] 等）。候选策略以自身比较独立成立，
判决对象与冻结对象一致。

## 3. LODO（留一结构，level-1 外推压力测试；regret / Δ vs incumbent）

| held-out | global | d_lookup | dp_gbdt | dp_abstain |
|---|---|---|---|---|
| S_season | 0.184 / −0.113 | 0.200 / −0.130 | 0.055 / **+0.016** | 0.066 / +0.005 |
| S_trend | 1.949 / +0.000 | 1.802 / +0.147 | 1.307 / **+0.642** | 1.333 / +0.615 |
| S_both | 0.192 / +0.181 | **0.100 / +0.273** | 0.219 / +0.154 | 0.226 / +0.147 |

- **season/trend 全never-seen 仍安全且获益**（从未见过 season 也不伤它：Δ+0.016）。
- **S_both 兑现预注册的 aliasing 机制边界**（e32_policy docstring/A-39）：与 trend/season 共享
  特征签名 → D+P 外推到 S_both 不如 D-only（0.219 vs 0.100）——**无 harm（Δ 仍 +0.15）但次优**。
  正式声明须并列此点；abstain 对此类"自信外推"确认为盲（触发率不升）。

## 4. all_data 稳健性口径（n=560，含 S_ar）

定性结论全部保持：dp vs global CI[−0.305,−0.183]、vs d_lookup CI[−0.184,−0.065]、vs d_gbdt
CI[−0.250,−0.124]、vs true-D CI[−0.184,−0.043]；retention 105%；dp_abstain worst LCB −0.137
（vs d_lookup −0.786）。差异：abstain 在含 S_ar 时 mean regret 略升（+0.015 CI 跨 0）——S_ar
的高不确定性触发多余回退；LODO S_ar：dp Δ−0.007（中性，S_ar 本就无可路由增益）。

**稳定性发现（评审第十六轮，A-40① 的依据）**：S_ar 混入训练不只影响 S_ar 自身——
**S_both 子群的决策也随训练结构组成变化**（primary worst LCB −0.083 → all_data −0.137，
最差同为 snrHigh×S_both）。策略行为对训练语料结构组成敏感 ⇒ 最终声明范围（common-support
vs all-data vs 显式 OOD gate）是打开 holdout 前必须冻结的最后一个科学选择——**A-40① 冻结为
common-support 主声明**。

## 5. 解释边界与遗留

1. **level-1 声明**：只允许写"**结构特征可实现安全路由**"（平衡合成语料 + 留一结构）。
   "Pattern 跨域决策价值"须 level-2 真实 dataset/domain holdout（A-37⑤），未做。
2. **安全声明的准确边界（评审十六轮修正）**：判据 (iii) 只检查 S_season；dp_abstain 的
   **overall** worst-group LCB = −0.083（primary）/−0.137（all_data），均低于 −δ_safe=−0.05，
   最差为 S_both 子群（mean +0.003/−0.026 的零附近噪声，非 d_lookup season 式 mean −0.85 的
   结构性损伤；n=40 LCB 分辨率所限）。**正确措辞 = "消除了 F0 的季节性结构伤害并显著改善总体
   worst-group，尚未证明所有结构子群全局非劣"，不得写"策略已全面安全"。**
3. 判据在 **dev 语料**（seeds 0–19 + A31e namespace）成立；**confirmatory seeds 20–39 +
   independent reporter（门③）未开**——按 A-32d/评审十五轮路线，冻结最终策略后一次性打开。
4. 实现勘误：正式 seed=20260704 曾使 GBDT random_state 溢出 sklearn uint32 → 首次启动在
   第一个 fold 即崩（**未产生任何结果后修复**，取模压域、小 seed 行为不变，freeze 完整性无损）。

## 5b. 附录：固定 cell×origin 的严格 median 剂量反应（评审十六轮共识点 4，零成本 post-hoc）

F0 的"harm 随选中窗单调"存在跨 cell 混杂（cell×SNR×选中窗同变）。用本实验 records 的全动作
outer-test 标签，**固定 cell×origin** 比较 median@5/9/15/25（`dose_response_fixed_cell.json`）：

- **S_season：4/4 cell harm 严格单调**（如 snrHigh|full 0.091→0.186→0.382→1.111）——F0 主张
  在无混杂口径下**成立**；
- **S_trend：4/4 cell gain 严格单调**（snrLow|full 7.085→5.895）；
- **S_both：snrHigh 两 cell 非单调（w9 处内部最优）、snrLow 两 cell gain 单调**——中间结构存在
  **内部最优剂量**，正是其成为路由难点（LODO aliasing 案例）的机制根源；
- **S_ar：全程平坦（≈1.10–1.19）**——剂量不敏感，"无可路由增益"的直接证据（与 LODO Δ−0.007 互恰）。

结构×剂量交互在最严格口径下确立：同 cell 内 season/trend 方向相反、S_both 有内部极值。

## 6. 下一步

**confirmatory freeze 已注册为 A-40**（适用范围=common-support 主声明 / 候选=dp_abstain 含
直接 CI / estimand=locked transfer 主+replication 次 / 数据构造预锁（A38C 同构补齐）/
grouped full-refit 统计 / reporter 冻结 {dlinear_scratch, chronos}）。**打开 seeds 20–39 =
A-40 全部实现并再获拍板后的一次性动作**；level-2 真实 dataset/domain holdout 其后；
S_both aliasing 的 OOD gate 属新协议，不得对本轮结果事后调参。
