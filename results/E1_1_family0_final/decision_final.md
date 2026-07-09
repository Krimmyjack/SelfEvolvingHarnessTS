# E-3.3 Family 0 正式收尾（强化方案 b）— 判决

日期：2026-07-04　脚本：`run_family0_final.py`（每 cell 一个 B=1000 job + `--combine`）
base=operator_pool_v2.1（7 动作）；expanded=+8 剂量（median 9/15/25、savgol 21/31、MA 9/15/25）。
正式 CI=full-refit group bootstrap（A-33c），**每 cell B=1000 全跑满**（4 job 并行，A-34/A-36）。
产物：`cell_<slug>.json`（×4）+ `report.json`。**confirmatory seeds 20–39 未开**。

修正 exploratory（`../E1_1_family0/`）的 5 处越界：#1 B=1000 非 smoke；#2 worst-group 非劣（非仅
"无 harm"）；#3 聚合 CI + D-vs-global 真算；#4 median 支配只点级；#5 每 cell 独立文件可重放。

---

## 0. 一句话判决

> **F0 未通过 D-3.3v3.1 四门冻结判据**——决定性地卡在**门④ worst-group 安全**（median 剂量在
> **全部 4 cell 损伤季节序列**），并险失门②（聚合 CI 下界 −0.010）。**但这不是"F0 无用"，而是
> 全项目最强的"Pattern 有决策价值"证据**：固定 degradation cell 内最优剂量**随结构翻号**。
> ⇒ **F0 不冻结为 degradation-only 路由策略；保留候选算子 median@5/9/15/25；路由问题移交 E-3.2。**

---

## 1. headline：结构×剂量强交互 → degradation-only 路由结构不安全

cell×origin（=SNR×missing × 结构族）分解，**确定性**（raw nested，非 bootstrap；LCB 为 test-uid
重采样的子群安全屏）。14 个子群按 LCB 排序：

| cell（选中窗） | origin | n | Δ_raw | LCB |
|---|---|--:|--:|--:|
| snrLow\|full (w25) | **S_season** | 18 | **−0.666** | **−0.713** |
| snrLow\|miss (w15-25) | **S_season** | 23 | **−0.515** | **−0.623** |
| snrHigh\|miss (w15) | **S_season** | 17 | **−0.270** | **−0.282** |
| snrHigh\|miss (w15) | S_both | 29 | −0.064 | −0.153 |
| snrHigh\|full (w9) | **S_season** | 22 | **−0.091** | **−0.105** |
| snrLow\|miss | S_ar | 40 | +0.014 | −0.020 |
| snrLow\|full | S_ar | 40 | +0.032 | −0.012 |
| snrHigh\|full | S_both | 31 | +0.046 | −0.003 |
| snrHigh\|full | S_trend | 35 | +0.119 | −0.000 |
| snrLow\|miss | S_both | 11 | +0.442 | +0.246 |
| snrHigh\|miss | S_trend | 32 | +0.497 | +0.300 |
| snrLow\|full | S_both | 9 | +0.646 | +0.495 |
| snrLow\|miss | S_trend | 8 | +1.993 | +1.376 |
| snrLow\|full | S_trend | 5 | +3.937 | +2.993 |

- **S_season 在全部 4 cell 被 F0 median 剂量损伤**，harm 幅度**严格随选中窗单调**（w9 −0.091 /
  w15 −0.270 / w15-25 −0.515 / w25 −0.666），LCB **全部 < −δ_safe=0.05**（连最轻 w9 也 −0.105）。
  机制：median 窗 ≈ 语料主周期 24 → **抹平季节性**。
- **S_trend 反向获益**（+0.12→+3.94，随剂量放大）；S_both 多数获益；S_ar 中性（±0.03）。
- **门④在每个 cell 失败**（`safety_worst_group.pass_noninferiority=False`）→ cell（=degradation-only）
  剂量路由**结构不安全**：它按 cell 平均选一个重 median，帮了 trend/AR、**牺牲 season**。

**科学含义（升级，非降级）**：这正是 Pattern 携带 degradation 之外决策价值的**直接构造性证据**——
固定退化条件下，最优剂量由**结构**（而非 SNR/missing）决定并翻号。E-3.2 从"验 pattern 有无增量"
升为"验 **structure-conditioned 剂量 + abstain**（季节序列回退轻剂量/identity）能否同时吃到 S_trend
增益、避开 S_season 损伤、从而**同时**优于 global 与 D-only-Lookup"。

## 2. 供给增益：机制/方向坐实、幅度贴边（诚实 CI）

**cell 等权 aggregate Δ_supply（B=1000，各 cell 独立 boot_deltas 卷积，A-34）**：

| 量 | 值 | 门② CI_lo>0 |
|---|---|:--:|
| Δ_supply(cell 等权) | **+0.1116** | |
| 95% CI | **[−0.0100, +0.2716]** | ❌（−0.010，险失）|
| pos%（正向 replicate） | **0.954** | |
| 门① 点 > ε=0.03 | ✅ | |

聚合把 4 cell 一致正向的信息 pool 起来后，CI 远比任何单 cell 紧：**点 +0.112≫ε、95.4% replicate 为正、
但 95% 下界刚好触 0（−0.010）** → 门①过、门②**险失**（差 0.010）。诚实结论：**效应正、方向稳、幅度
接近但未过 95% 显著线**（小样本 4 cell×72–88 uid）。

**per-cell（raw 确定性 nested 与 bootstrap 均值分列；AI 强调勿混）**：

| cell | n | raw base→exp (Δ) | boot Δ | 95% CI | pos% | median/ma 选择share |
|---|--:|---|--:|---|--:|---|
| snrHigh\|full | 88 | 1.015→0.974 (+0.041) | +0.043 | [−0.053,+0.154] | 0.79 | 0.90 / 0.00 |
| snrHigh\|miss | 78 | 0.913→0.792 (+0.121) | +0.118 | [−0.057,+0.309] | 0.88 | 0.97 / 0.00 |
| snrLow\|full | 72 | 1.741→1.535 (+0.206) | +0.163 | [−0.098,+0.550] | 0.74 | 0.63 / 0.24 |
| snrLow\|miss | 82 | 1.664→1.547 (+0.116) | +0.122 | [−0.081,+0.434] | 0.81 | 0.93 / 0.02 |

- **per-cell CI 全跨 0**（小 n；与 exploratory 一致）。**raw 点 > boot 均值**（如 snrLow|full raw
  +0.206 vs boot +0.163）——raw 单次 nested 含选择乐观、boot 均值把选择不稳定摊入，二者分列。
- **median 点级支配、非策略级**（越界#4 修正）：median 选择 share 0.63–0.97 主导，但 **`f0_ma_w25`
  在最噪的 snrLow|full 仍占 24%** → 只能说 median 点估计更优，不能说策略层完全支配 ma。
  `f0_savgol_*`（尤 w31>周期）跨 cell 最差、从不入选（A-33d 端点过冲预测兑现）。

## 3. D-only Lookup vs global-single（held-out，AI 要求真算）

| 策略 | 动作 | cell 等权 held-out loss |
|---|---|--:|
| incumbent 全局单动作（base 池） | v_median@w5 | 1.3328 |
| **best 全局单一固定剂量**（扩池） | **f0_median_w15** | **1.2639** |
| **D-only Lookup**（cell 条件化选择） | 每 cell 最优剂量 | **1.2121** |

- 最优**单一固定**剂量已是 median **w15**（非 incumbent w5）→ 池全局欠剂量（1.333→1.264，−0.069）。
- **D-only（degradation 条件化）再胜 global-single +0.0518 > ε** → 剂量维**确有可路由异质性**、
  degradation 坐标能捕获一部分。
- **但这条 D-only 路由正是 §1 里结构不安全的那条**（它就是 cell 级选择）。所以"D-only 胜 global"与
  "D-only 伤 season"**并存** → 正解是 structure-conditioning（E-3.2），应**同时**胜过 global 与 D-only。

## 4. 判决（D-3.3v3.1 四门 / A-30d 五判据 / A-32d 分支）

**D-3.3v3.1 四门**：①点>ε ✅（聚合 +0.112、3/4 cell>ε）；②95% CI_lo>0 **❌**（聚合 −0.010 险失、
per-cell 全失）；③independent reporter 同向 —— 未算（confirmatory）；④worst-group LCB>−δ_safe
**❌ 决定性**（S_season 全 cell、worst −0.713）。→ **四门未齐、卡在④与②。**

**A-30d 五判据**：①强剂量成为 held-out selected ✅；②Δ_supply 聚合点>ε/CI 险失 ⚠；③deg 路由>全局
单动作（+0.052）✅；④reporter 同向 —— 待办；⑤worst-group 无显著回退 **❌**（此前误判✅，final 翻案）。

**A-30b**："缺 dosage 维"仍成立（median 剂量普遍被 held-out 选中、复现 witness），**F1–F5 降补充**——
但须并列**该维非 degradation 可安全路由、必须结构条件化**。

**A-32d 分支裁决**：F0 **不是可冻结的最终路由策略**（门④失败）→ **不打开 seeds 20–39**；
holdout 继续封存。收尾形式：
> 动作池缺平滑**强度**维（补入后 held-out 普遍偏好更重 median、cell 等权增益点 +0.112>ε、pos 95%）；
> **但该维在固定退化 cell 内随结构翻号**——median 帮 trend/AR、伤 season——故 degradation-only 路由
> **结构不安全（门④全 cell 失败）**。**冻结的是候选算子（median@5/9/15/25），不是路由策略；
> 结构条件化剂量 + abstain 的价值由 E-3.2 判定。**

## 5. 下一步（须你拍板；均未执行）

1. **进 E-3.2**（现在有了 F0 点名的具体假设）：6 臂 held-out policy regret——Global / D-only Lookup /
   D-only GBDT / **P-only GBDT** / **D+P GBDT** / D+P+interaction+**abstain**；主指标=regret（非动作准确率），
   报 arm-in-domain + LODO；判据 D-3.2d。**F0 给出可证伪靶**：D+P（结构感知）应同时 > global 与 D-only，
   靠对 S_season abstain 到轻剂量/identity 回收 S_trend 增益而不伤 season。
2. **A-31e 补样 + 连续 SNR residualization**（E-3.2 硬前置）：补低 SNR 下各结构、平衡 origin，
   防 Pattern 偷代理 SNR；顺带收窄 per-cell CI。
3. reporter 同向（门③）留 confirmatory 口径（report_target），与 seeds 20–39 一并、且**仅当**某个池/
   策略过前三门后才一次性打开。

## 6. 解释边界（不变）
F0 只涉 forecasting、median/savgol/MA 剂量、cell=SNR×missing 的选择单位。§1 的结构×剂量交互是
**cell 内 origin 分解**观察到的，其**可实现**决策价值（held-out policy regret）仍须 E-3.2 正式判定——
本 §只证"degradation 坐标不足以安全路由剂量"，E-3.2 才证"结构坐标能否安全且有增益地路由"。
