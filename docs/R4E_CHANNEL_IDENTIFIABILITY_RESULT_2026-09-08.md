# R4E · 按通道分开的可识别性：被总效应掩盖的局部规律存在吗（结果，2026-09-08）

证据类别：**MECHANISM / NEGATIVE-capable**（development 机制读数，不是能力、泛化或显著性证据）。
读数口径与判词由 `docs/R4D_PERCHANNEL_STAGE_A_AND_ACTION_DECOMPOSITION_REQUEST_2026-09-07.md` §4.5 冻结，
阈值沿用 R4A（`INFORMATIVE 0.70` / `WEAK 0.60`）与 R4C（分位网格 1/3、1/2、2/3，`material 0.005`），未调整。

- 脚本：`evaluation/main_protocol_p4/audit_r4e_channel_identifiability.py`
  （`python -m evaluation.main_protocol_p4.audit_r4e_channel_identifiability` 可复跑，**0 fit**，约 60 秒）
- 工件：`artifacts/main_protocol/r4e_channel_identifiability.json` / `.md`
- 输入：R4D-A 的三格库 `_scratch/r4d_a_three_cell_store.json`（该库的 12 次物理拟合早已花掉；本包只索引）

## 0. 一句话答案

> **存在，但只在 ctx 通道、只在 W2、且只在单面上成立，并且它不足以改变部署决定。**
> W2 的 `ctx` 通道上，`local_robust_z_peak` 对严重伤害的四折 LODO AUC = **0.826536**（delayed 面，四折
> 0.748168–0.968750 全部 ≥ 0.70，方向一致，`PATTERN_INFORMATIVE`），而同一批实例的 `total` 通道只有
> **0.656777**（`PATTERN_WEAK`，R4A 的原数）。support 面同样有一个满支撑的 INFORMATIVE 项
> （`local_robust_z_peak → helped`，0.720834）。**总效应确实掩盖了一条局部规律。**
> 但读数 2（最终裁决口径）在两个 program、三种面分组共 6 格上一致给出 `FIXED_CHOICE_NOT_BEATEN`：
> 27 个可见量里没有任何一个能让"看 Pattern 选处理"在 ≥3/4 折上同时胜过 always-P 和 always-identity。
> 所以 §5 分流表第一行的两个条件里，**前半条（某通道某可见量四折 ≥ 0.70）成立，后半条（选择胜过两个固定选择）不成立**。

## 1. 口径

对每个 (program, position, face, uid) 实例，R4D-A 给出三个通道（损失口径）：
`route = L_pr − L_rr`（共享模型被换）、`ctx = L_pp − L_pr`（这条序列自己的服务 context 被准备）、
`total = route + ctx = −g`。每通道各自定义 `helped_c = c < 0`、`severe_c = c > 0.30`。

`ctx` 通道只在服务窗被程序修改（`|ctx| > 1e-12`）的实例上读；其余实例的 `ctx` 按构造恰为 0，
两类都不属于，留着只会在两个类里都塞进硬零。行数：

| program | 服务窗被修改（进入 ctx 读数） | 未修改（ctx 恒 0，剔除） |
| --- | ---: | ---: |
| ANCESTOR | 335 / 840 | 505 |
| W2 | 723 / 840 | 117 |

与 R4D-A §5.1 的 622 = 505 + 117 逐条一致。

特征集 27 个 = R4A 的 12 个词表数值量 + 9 个机制量（`gap_period_aligned` 按 R4A 的
`NOT_COMPUTABLE` 直接跳过，它本来就不在 R4A 的 `VISIBLE_FEATURES` 里）+ R4C 的 6 个服务侧作用条件量。
统计函数（`Reader` / `auc` / `_average_ranks` / `_frozen_edges` / `stump_fold` / `feature_reading` / `verdict`）
与 `serving_observables` 全部 import 复用，未重写。R4C 的算子交叉核对 840/840 通过。

## 2. 边界自检

| 项 | 值 |
| --- | --- |
| Consumer fit | **0**（不 import 任何触发 `_serve` 的路径） |
| Harness 内部 LLM 调用 | **0** |
| held-out 读 | **0**（前沿 4056） |
| horizon 读 | **0**（本包不要 ORACLE 量，只读 `[origin−192, origin)`） |
| 读到的最大时间下标 | **3863**（= max origin 3864 − 1） |
| 原始窗读取次数 | 1680 |
| 行数 | 1680 =（2 program × 42 面 × 20 uid）；80 uid、21 位置、4 块 |
| 因 status 剔除 | 0（库内 1680 条全为 `OK`，无 `degenerate_uids`） |
| 既有文件编辑 / git 提交 / 新增 SHA / 子 Agent | 0 / 0 / 0 / 0 |
| 预测库写入 | 无（只读三格库） |
| 数据身份 / NaN | `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，NaN = 503712 |

全部读数在工件里保留 6 位小数。

## 3. 读数 1：per-channel LODO 可识别性

方向在训练折定，AUC 为四块 leave-one-block-out 的定向值。"清全折"= 该 (特征, 目标) 每折 ≥ 0.70 且方向一致。

| program | 通道 | 面 | n | helped | severe | 最佳 (特征\|目标) | 四折均值 | 四折最小 | 四折 | 清全折数（其中四折全可评） | 判词 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |
| ANCESTOR | route | support | 420 | 274 | 34 | `srv_donor_dispersion\|severe` | 0.617482 | 0.568627 | 0.592308 / 0.714286 / 0.594707 / 0.568627 | 0 (0) | **WEAK** |
| ANCESTOR | route | delayed | 420 | 278 | 38 | `srv_donor_dispersion\|helped` | 0.579399 | 0.453878 | 0.453878 / 0.725000 / 0.558124 / 0.580592 | 0 (0) | **WEAK** |
| ANCESTOR | route | 两面 | 840 | 552 | 72 | `local_robust_z_peak\|severe` | 0.534267 | 0.467302 | 0.569974 / 0.467302 / 0.549444 / 0.550347 | 0 (0) | **WEAK** |
| ANCESTOR | ctx | support | 184 | 126 | **2** | `missing_fraction\|severe` | 0.943050 | 0.898148 | 0.898148 / — / 0.987952 / — | 11 (**0**) | **INFORMATIVE**（见 §3.1） |
| ANCESTOR | ctx | delayed | 151 | 114 | **1** | `estimated_region_start_fraction\|helped` | 0.578913 | 0.527778 | 0.527778 / 0.658730 / 0.560964 / 0.568182 | 0 (0) | **UNINFORMATIVE** |
| ANCESTOR | ctx | 两面 | 335 | 240 | **3** | `missing_fraction\|severe` | 0.909802 | 0.904040 | 0.904040 / — / 0.915563 / — | 10 (**0**) | **INFORMATIVE**（见 §3.1） |
| ANCESTOR | total | support | 420 | 291 | 28 | `local_robust_z_peak\|severe` | 0.633725 | 0.525021 | 0.525021 / 0.596491 / 0.544966 / 0.868421 | 0 (0) | **WEAK** |
| ANCESTOR | total | delayed | 420 | 287 | 31 | `local_robust_z_peak\|severe` | 0.676036 | 0.599129 | 0.717803 / 0.599129 / 0.702528 / 0.684685 | 0 (0) | **WEAK** |
| ANCESTOR | total | 两面 | 840 | 578 | 59 | `local_robust_z_peak\|severe` | 0.650472 | 0.587191 | 0.614516 / 0.587191 / 0.616179 / 0.784000 | 0 (0) | **WEAK** |
| W2 | route | support | 420 | 277 | 50 | `spike_peak_over_tail_sd\|severe` | 0.616312 | 0.489583 | 0.670895 / 0.489583 / 0.616598 / 0.688172 | 0 (0) | **WEAK** |
| W2 | route | delayed | 420 | 253 | 70 | `spike_peak_over_tail_sd\|severe` | 0.608352 | 0.544000 | 0.718111 / 0.544000 / 0.600983 / 0.570312 | 0 (0) | **WEAK** |
| W2 | route | 两面 | 840 | 530 | 120 | `spike_peak_over_tail_sd\|severe` | 0.616580 | 0.538033 | 0.691684 / 0.538033 / 0.604482 / 0.632120 | 0 (0) | **WEAK** |
| W2 | ctx | support | 345 | 159 | 37 | `srv_period_filled_points\|severe` | 0.731432 | 0.646429 | 0.894547 / 0.705882 / 0.678869 / 0.646429 | 1 (**1**) | **INFORMATIVE** |
| W2 | ctx | delayed | 378 | 180 | 39 | `local_robust_z_peak\|severe` | **0.826536** | 0.748168 | 0.802768 / 0.968750 / 0.748168 / 0.786458 | 1 (**1**) | **INFORMATIVE** |
| W2 | ctx | 两面 | 723 | 339 | 76 | `local_robust_z_peak\|severe` | 0.778597 | 0.676103 | 0.745849 / 0.943182 / 0.676103 / 0.749254 | 0 (0) | **WEAK** |
| W2 | total | support | 420 | 269 | 60 | `spike_peak_over_tail_sd\|severe` | 0.648404 | 0.543599 | 0.703210 / 0.543599 / 0.662823 / 0.683983 | 0 (0) | **WEAK** |
| W2 | total | delayed | 420 | 270 | 75 | `local_robust_z_peak\|severe` | 0.656777 | 0.541126 | 0.724267 / 0.677560 / 0.684158 / 0.541126 | 0 (0) | **WEAK** |
| W2 | total | 两面 | 840 | 539 | 135 | `spike_peak_over_tail_sd\|severe` | 0.655731 | 0.596000 | 0.734056 / 0.596000 / 0.644599 / 0.648268 | 0 (0) | **WEAK** |

块顺序固定为 `[0:40] / [120:160] / [40:80] / [80:120]`。

### 3.1 ANCESTOR 的 ctx INFORMATIVE 是判词规则的产物，不是证据

必须先说清楚，否则这张表会被读成"ANCESTOR 也有强信号"。ANCESTOR 的 ctx 通道上严重伤害实例总共只有
**3 个**（support 2、delayed 1）。冻结规则取的是**有 AUC 的折**的最小值，而没有正例的折算不出 AUC、
被跳过而不是判负；于是 `[120:160]` 与 `[80:120]` 两折（0 个正例）被略过，0.909802 只由两折、总共 3 个正例
撑起。工件为每个 INFORMATIVE 项写了 `folds_with_an_auc` / `positives_per_fold` /
`supported_by_every_fold`：ANCESTOR ctx 的 10–11 个"清全折"项**全部** `supported_by_every_fold = false`。
本包不改冻结规则，但这一格按 §5 分流不应被当作"存在局部规律"的证据。

W2 的两项相反：四折都有正例（delayed 面每折 2–17 个正例，共 39 个；support 面 16–63 个，共 159 个），
`supported_by_every_fold = true`。

### 3.2 W2 的 ctx 信号是什么

- **delayed 面**：`local_robust_z_peak → severe_ctx`，pooled 原始 AUC 0.206641（< 0.5），四折训练侧方向
  一致（0.170929–0.222027）。方向读作：**context 的稳健 z 峰值越低，服务窗被准备后越可能出现严重伤害**。
  定向后四折 0.748168 / 0.968750 / 0.802768 / 0.786458，均值 0.826536。
- **support 面**：`local_robust_z_peak → helped_ctx`，pooled 0.716119，四折 0.709091–0.732845，均值
  0.720834；方向相反地读作 **z 峰值越高，服务侧准备越可能降低损失**。两面合起来是一个连贯的机制陈述：
  有明显尖峰的 context，清洗它有收益；没有尖峰的 context，清洗它容易把正常波动当成尖峰改坏。
- **两面合并即掉回 WEAK**（最小折 0.676103 < 0.70）。合并把两个方向相反的目标混在一起，这不是稳健性
  提升而是稀释；表里两个 INFORMATIVE 都是单面结论，报告时不得写成"W2 的 ctx 通道整体可识别"。
- `srv_period_filled_points`（R4C 的服务侧作用条件量）在 support 面 ctx 上给出 0.731432、pooled 0.754212，
  与机制一致：周期填补动作越多，服务侧改动越大，严重伤害的机会越多。这是 R4C 那 6 个量第一次在
  某个通道上进入最佳位置。

## 4. 读数 2：看 Pattern 选处理 vs 始终选同一个处理（最终裁决口径）

规则：训练折上按该特征的 1/3、1/2、2/3 分位与两个方向选阈值，使"选中者部署 P、未选者 identity"的
**全人群效用**最大；测试折读全人群效用。效用 = 该折全部服务实例上 `(-total if 选中 else 0)` 的均值——
identity 就是 `L_rr` 那一格，按构造恰为 0。对照 always-P、always-identity（= 0）、oracle 逐序列上界
（`mean(max(0, −total))`，**不可部署**）。

| program | 面 | 最佳规则（按较小的那个均值差） | 对 always-P 四折均值 | 对 always-identity 四折均值 | 同时胜两者的折数 | 受损序列数差（对 P / 对 identity） | 最坏单序列差（对 P / 对 identity） | always-P 效用 | oracle（不可部署） | 判词 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ANCESTOR | support | `spike_recurrence >= 0.000000` | +0.000000 | +0.267773 | 0/4 | 0.00 / +32.25 | 0.000000 / −0.815736 | 0.267773 | 0.333617 | **FIXED_CHOICE_NOT_BEATEN** |
| ANCESTOR | delayed | `gap_tail_fraction`（三折 `>= 0`，一折 `<= 0.395833`） | +0.000548 | +0.231404 | 1/4 | −0.50 / +32.75 | 0.000000 / −1.114118 | 0.230856 | 0.313096 | **FIXED_CHOICE_NOT_BEATEN** |
| ANCESTOR | 两面 | `spike_recurrence >= 0.000000` | +0.000000 | +0.249314 | 0/4 | 0.00 / +65.50 | 0.000000 / −1.231115 | 0.249314 | 0.323357 | **FIXED_CHOICE_NOT_BEATEN** |
| W2 | support | `spike_recurrence >= 0.000000` | +0.000000 | +0.267159 | 0/4 | 0.00 / +37.75 | 0.000000 / −1.363855 | 0.267159 | 0.397966 | **FIXED_CHOICE_NOT_BEATEN** |
| W2 | delayed | `spike_tail_count >= 0.000000` | +0.000000 | +0.239236 | 0/4 | 0.00 / +37.50 | 0.000000 / −1.125967 | 0.239236 | 0.361946 | **FIXED_CHOICE_NOT_BEATEN** |
| W2 | 两面 | `spike_recurrence >= 0.000000` | +0.000000 | +0.253198 | 0/4 | 0.00 / +75.25 | 0.000000 / −1.476989 | 0.253198 | 0.379956 | **FIXED_CHOICE_NOT_BEATEN** |

**27 个可见量里，一个都没有在 ≥3/4 折上同时胜过两个固定选择**（`folds_beating_both_fixed >= 3` 的特征集合为空）。
读这张表要抓住三件事：

1. **训练折上的最优规则退化成"全都选"。** always-P 的效用是正的（0.23–0.27），所以任何"排除一部分序列"
   的阈值在训练折上都会丢掉收益；最大化全人群效用的结果就是阈值退到分位下界、`treated_fraction = 1.0`，
   规则与 always-P 逐折相同（差 +0.000000 不是四舍五入，是同一个决策）。唯一的例外是 ANCESTOR delayed
   在 `[120:160]` 折上选了 `gap_tail_fraction <= 0.395833`（覆盖 96.7%），赢 always-P 0.002193，四折均值
   +0.000548，远低于 0.005，也只有 1/4 折。
2. **胜过 always-identity 是平凡的，不是本包的结论。** 两个程序整体都在帮忙，所以 +0.23 ~ +0.27 的差
   只说明"做点什么好过什么都不做"，代价是每折多 32–75 条受损序列、最坏单序列多损 0.82–1.48。
3. **头寸空间是有的，可见量拿不到。** oracle 逐序列上界 0.313–0.398，比 always-P 高 0.07–0.13。
   这 0.07–0.13 就是"完美的逐序列选择"能拿而 27 个可见量拿不到的部分。

## 5. 读数 3：通道对比（一句话）

**两个 program 都是 ctx 通道更可识别，而且 route 与 ctx 的最佳特征不同。**
ANCESTOR：ctx 0.909802（`missing_fraction|severe`，但见 §3.1 的支撑警告）> total 0.650472 > route 0.534267；
W2：ctx 0.778597（`local_robust_z_peak|severe`）> total 0.655731（`spike_peak_over_tail_sd|severe`）>
route 0.616580（同 `spike_peak_over_tail_sd|severe`）。W2 的 total 与 route 共用最佳特征、ctx 换了另一个，
这与 R4D-A 的 ρ(route, total) = 0.85–0.94、ρ(ctx, total) = 0.21–0.29 同向：total 的排序基本就是 route 的排序，
ctx 的规律因此被压掉。

## 6. 读数 4：R4A 复核

用本库的 `total` 复算 R4A 对 total 的最佳特征与四折均值，**四格全部一致，差为 0.0**（判据：同一
`(特征|目标)` 且 LODO 均值差 ≤ 5e-6，只在 R4A 的 21 个可见量上取最佳）：

| 格 | R4E 最佳 | R4E 四折均值 | R4A 最佳 | R4A 四折均值 | 行数 R4E / R4A | 一致 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| ANCESTOR support | `local_robust_z_peak\|severe_harm` | 0.633725 | 同 | 0.633725 | 420 / 420 | 是 |
| ANCESTOR delayed | `local_robust_z_peak\|severe_harm` | 0.676036 | 同 | 0.676036 | 420 / 420 | 是 |
| W2 support | `spike_peak_over_tail_sd\|severe_harm` | 0.648404 | 同 | 0.648404 | 420 / 420 | 是 |
| W2 delayed | `local_robust_z_peak\|severe_harm` | 0.656777 | 同 | 0.656777 | 420 / 420 | 是 |

派发口径点名的 ANCESTOR delayed `local_robust_z_peak → severe` 0.676036 精确复现。这条复核同时验证了
三格库的 `total` 与 m_r0k 库的 `−g` 是同一批数字、特征卡的重算与 R4A 逐位一致。

## 7. 明确回答 astra 的问题："被总效应掩盖的局部规律存在吗？"

**存在一条，范围比问题小。** 在 W2 的 `ctx` 通道上，`local_robust_z_peak` 对严重伤害的四折 LODO
从 total 的 0.656777 升到 0.826536（delayed 面），四折全部 ≥ 0.748、每折都有正例、方向一致；support 面
另有一条同一特征、相反方向对 `helped` 的满支撑 INFORMATIVE（0.720834）。astra 的判断在这一点上成立：
"效应从哪里产生"与"哪些信息能预测谁受益"确实是两个问题，同一个 total 里 route 的排序压过了 ctx 的排序
（R4D-A：ρ(ctx, total) 只有 0.21–0.29），把这条规律盖住了。

**但它有四条限定，缺一条都会把结论说大：**
1. 只在 `ctx` 通道上。`route` 通道——也就是 R4D-A 说的占 70–80% 的那个通道——三格九面全部 WEAK，
   最好 0.616580。承载多数效应的通道仍然不可识别。
2. 只在 W2 上。ANCESTOR 的 ctx INFORMATIVE 由 3 个严重伤害实例、两个能算 AUC 的折撑起（§3.1），
   不作证据。
3. 只在单面上。两面合并即掉回 WEAK（0.778597，最小折 0.676103）。
4. **它没有转化成更好的部署决定。** 读数 2 在全部 6 格给 `FIXED_CHOICE_NOT_BEATEN`，
   没有任何可见量能让"看 Pattern 选处理"胜过"给所有人同一个处理"。可识别一个通道上的严重伤害，
   与在**部署效用**（用 total）上赢过 always-P，是两回事——因为部署要为 route 通道的后果一起买单，
   而 route 不可识别。

按 §5 修订分流表：第一行要求"4.1 `ROUTE_DOMINANT`（已得）∧ 某通道某可见量四折 ≥ 0.70 **且**
看 Pattern 选处理胜过两个固定选择 ≥ 3/4 折"，**合取不成立**。落到第三行的口径：这组观察量在这个设置下
未表现出足够信息——但**不判定噪声、不关闭序列级研究**，因为 W2 ctx 上的 0.826536 是真的。

## 8. 正典 §10 五问

**Harness 行为改变了什么。** 没有。0 fit、0 LLM、0 held-out、0 horizon、不编辑任何既有文件、不提交 git、
不新增 SHA、不写任何 Skill / Episode / 预测库、不改风险线 / 协议 / 词表 / 阈值。只新增 1 个脚本、
2 个工件、本报告。

**数据上观察到了什么。** (i) 把效应劈成两个通道后，`ctx` 通道在两个 program 上都比 `total` 更可识别，
`route` 通道都比 `total` 更不可识别；(ii) W2 的 ctx 上出现本项目迄今第一个满支撑的 `PATTERN_INFORMATIVE`
（delayed 面 0.826536，support 面 0.720834，同一特征 `local_robust_z_peak`、方向互补）；(iii) R4C 的
服务侧作用条件量 `srv_period_filled_points` 第一次进入最佳位置（W2 ctx support 0.731432）；
(iv) 读数 2 六格全部 `FIXED_CHOICE_NOT_BEATEN`，训练折上的最优阈值退化成"全都选"，
而 oracle 与 always-P 之间还有 0.07–0.13 的空间；(v) 本库的 total 与 R4A 的四格读数逐位一致。

**当前最大方法不确定性。** ctx 通道的可识别性**是不是可部署的**，本包答不了。ctx 是一个反事实分量
（`L_pp − L_pr`），部署时没人能只领走 ctx 而不领走 route——除非 Consumer 结构本身改成 per-channel，
那正是尚未跑的变量 B。其次，1680 行来自 21 位置 × 80 uid 的重复观测、只有 4 个块，`[80:120]` 与
`[120:160]` 两折分别只有 33–77 与 49–70 行，单折 AUC 的不确定性没有量化（本包不做显著性）。第三，
"两面合并掉回 WEAK"到底是稀释还是不稳定，四个块分不开。

**是否仍与目标一致。** 一致，并且这次读数把分流表的路线钉死了：§5 第一行的合取不成立，所以
**不**把 (ctx, `local_robust_z_peak`) 直接送进 Observation 候选、**不**开 LLM 输入消融包。同时第三行的
"不判定噪声、不关闭序列级研究"被数据支持而不只是措辞——ctx 上确实有信号。下一个判据仍然是变量 B：
只有 per-channel Consumer 才能把"ctx 的可识别性"变成"可部署的选择"。

**下一项最小纵向切片。** 变量 B（per-channel Stage A 重打分）。它有两个本包无法替代的作用：
(a) route 通道按构造被拆掉，于是"ctx 可识别"能否兑现成部署效用可以直接用读数 2 的同一口径测；
(b) 用 B 的三格库重跑本脚本即可得到 per-channel 侧的同格对照（脚本只需换 `STORE` 常量）。
在 B 之前，若要花更小的预算，可做的**只**有 0 fit 的诊断：把 W2 ctx 的 delayed/support 方向差异
按 (position, uid) 配对追一遍，看它是同一批序列的两面还是两批序列——这决定 §3.2 的机制陈述能不能保留。

## 9. 这份结果不是什么

- **不是能力、泛化或显著性证据。** 全部 development，held-out 一个都没读（最大下标 3863 < 4056），
  horizon 一个都没读，行与行之间因重复观测不独立。
- **不是"序列状态可以用来选处理"的结论。** 恰恰相反：最终裁决口径六格全部 `FIXED_CHOICE_NOT_BEATEN`。
  §0 与 §7 的 0.826536 是"能不能看出谁会严重受损"，不是"看出来以后能不能赚到"。
- **不是"ANCESTOR 的 ctx 通道可识别"。** 那一格的 INFORMATIVE 由 3 个正例撑起，且两折无正例被冻结
  规则跳过（§3.1）。工件里的 `supported_by_every_fold` 就是为这件事留的。
- **不是对 R4A 判词的推翻。** R4A 对 **total** 的全部 WEAK 在本包精确复现（§6）；本包只是指出 total
  不是唯一可以问的对象。
- **不是 per-channel Consumer 的读数。** 三格库出自 pooled Consumer（R4D-A 变量 A）。`ctx` 是在 pooled
  模型下的反事实分量，不是 per-channel 系统会产生的量。
- **不是 R4C 的复核。** R4C 读的是 `d = g(W2) − g(ANCESTOR)` 的边际效应；本包对两个程序各自的三个通道
  分别读，未做 W2 − ANCESTOR 的差。只借用了它的 6 个服务侧作用条件量与分位/material 常量。
- **不涉及 W1 / W3 与两个参考程序**：三格库里只有 ANCESTOR 与 W2。
- **不改变任何已冻结的协议、roster、split、阈值、预算或词表**；未新增 SHA、目录或平台层；未提交 git。
