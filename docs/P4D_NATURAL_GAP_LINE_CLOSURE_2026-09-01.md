# 天然缺口线 · 结果与收口（2026-09-01）

**总判词**：Forecast 取得**首个天然缺口上的、双面有界的组合程序正证据**；泛化未解决；
Targeting 整体判负；Classification 现有菜单无稳定 headroom。

**放行**：无。`P4-Evolution`（H3）继续 HELD，Natural Final 继续封存，UCR TEST 保持零读取。

**工件**（全部 0 LLM、0 held-out 读、0 UCR TEST 字节）：

| 工件 | 内容 | Consumer fits |
| --- | --- | --- |
| `p4d_natural_gap_roster.json` | 数据源审计与天然缺口盘点 | 0 |
| `p4d_natural_gap_preflight.json` | 五门 Preflight | 3 |
| `p4d_gap_repairability_audit.json` + `p4d_gap_per_series_gain.npz` | 396 程序全扫描 + 完整逐序列增益张量 | 4764 |
| `p4d_param_correction_rerun.json` + `p4d_param_corrected_gain.npz` | 参数遗漏修正重跑 | 1236 |
| `p4f_matched_composition.json` + `p4f_matched_gain.npz` | 缺口/组合/顺序三项归因 | 432 |
| `p4g_cross_fitted_targeting.json` | 交叉拟合 Targeting 诊断 | 0 |
| `p4e_classification_headroom.json` | Adiac/ArrowHead TRAIN held-in 菜单扫描 | — |

---

## 0. 为什么会有这一轮：数据源勘误

P1–P4c 全部工件把数据集标注为 `KDD Cup 2018 with missing values`，**该标注是错的**。
`data/kdd2018/series_cache.npz` 建自 `..._without_missing_values.tsf`，缓存内 NaN 计数
为 0。机械核验：两版本 UID 270/270、长度 270/270 对应，**2,438,652 个观测位置逐值
比对最大偏差 `0.000e+00`**，即 without 版 = with 版经上游填补。天然缺口为
**503,712 / 2,942,364 点 = 17.119%，270/270 条序列全部含缺失**。

因此 P1–P4c 的数字保留、不覆盖，但其结论范围收窄为**无缺口的 outlier / level /
denoise 场景**。详见 `AGENTS.md` §8.1。

含缺失版本记为独立数据身份 `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING`，
与既有结果**平行、不并表**：结构可读性在含缺口数据上重算为 239/270，roster 成员与
旧线不同（Support-B 由 T119 起变为 T120 起）。

**关键语义**：`_apply_program(window, None)` 即 `_linear_integrity(window)`，且每个程序的
输出都再过一次 `_linear_integrity`，所以 identity **本身就是线性插补**，`impute_linear`
必为 ZERO_BEHAVIOR。这条线能回答的问题是「fft / ema / period-median / ar 补全能否
胜过线性插补」，不是「补全是否有用」。

### 几何：两个 origin 不可评价

八个 P4b/P4c origin 中 **1416 与 1656 记为 `NOT_EVALUABLE`，不记为失败**：个别 eval
序列在该 horizon 内零观测真值（1416: T101、T102；1656: T11），缺失感知 sMASE 对其
无定义，评分将变成拿预测去比对线性插补。丢 origin 而非丢序列，是为了保持
`harmed_fraction` 分母跨 origin 一致。实际扫描 origin = `[1176, 1896, 2136, 2376,
2616, 2856]`。

---

## 1. 主结果：首个天然缺口上的、双面有界的组合程序正证据

396 个程序枚举、171 个可读、**19 个 (program, origin) 双面稳定对**。按精确的
`support_a‖support_b` 逐序列增益向量去重后为 **7 个不同效果**，落在 **2 / 6 个
origin**。对照：P4c 在无缺口数据上是 **0 / 193**。

origin 2856 上的三个稳定程序**第一步全部是 `period_median_complete`**：

| 程序 | Support-A 增益 / 受害 | Support-B 增益 / 受害 |
| --- | --- | --- |
| `period_median_complete>outlier_iqr` | **+0.6496** / 3 of 20 | +0.2878 / 3 of 20 |
| `period_median_complete>outlier_mad` | +0.6108 / 3 of 20 | +0.3167 / 4 of 20 |
| `period_median_complete>winsorize` | +0.5620 / 3 of 20 | +0.3168 / 3 of 20 |

有界预算为聚合 ≥ +0.005 ∧ 受害比例 ≤ 0.20（=4 条）∧ 单序列最大伤害 ≤ 0.30，两面均在
预算内。

**这个结果的准确名称是：首个天然缺口上的、双面有界的组合程序正证据。**

它**不是** Agent 自主形成的 Skill，**不是** held-out 或 fresh 正结果，**不是**统计显著性
结论——通过的是预注册的有界门槛，不是显著性检验；P4b 那套 origin 级精确 Wilcoxon
（n=8）在本轮未运行。

---

## 2. 归因：缺口、组合、顺序（`p4f_matched_composition.json`）

同 roster、同 6 个 origin、同程序，17 个臂 × 两个数据版本，432 fits。

**缺口依赖 —— 全称成立。** 已填补版本上 **17 个臂无一稳定**，含全部单算子臂、全部组合
臂、两种顺序。设计自带的合理性检查通过：`completion_is_inert_on_filled_cache = True`。

origin 2136 的 `winsorize` 也是缺口依赖的，此前"该结果受 roster/origin 混淆"的表述
**予以撤回**：

| origin 2136 · `winsorize` | Support-A | Support-B |
| --- | --- | --- |
| 含缺口 | **+0.2588** | +0.2365 |
| 已填补 | **−0.0508** | +0.2593 |

**组合必要。** 2856 上 `period_median_complete` 单独、`period_complete` 单独、
`outlier_iqr` / `outlier_mad` 单独**均不稳定**，只有配对稳定。配对超出较优单项：

| 配对 | Support-A | Support-B |
| --- | --- | --- |
| `pmc>outlier_iqr` | **+0.1329** | −0.0160 |
| `pmc>outlier_mad` | +0.0940 | +0.0129 |
| `pmc>winsorize` | +0.0453 | +0.0130 |

组合的收益主要体现在**跨过风险门**，而非在聚合均值上翻倍。

**顺序有向。** 反序臂 `outlier_X>period_median_complete` 在 2856 全部不稳定。
前向减反向：

| 配对 | Support-A | Support-B |
| --- | --- | --- |
| `outlier_iqr` | +0.0000（反序增益恰为 0） | **+0.2174** |
| `outlier_mad` | **+0.2082** | **+0.2173** |
| `winsorize` | **+0.2251** | **+0.2765** |

量级与单算子的全部增益相当。机制上讲得通：identity 已是线性插补，一个 61 步的天然
缺口会被拉成直线段；先按周期中位数补出合理形状，异常检测才面对真实离群点，反序则
是在对插补人工制品做检测。

---

## 3. 泛化仍未解决

- 仅 **2 / 6** origin 存在稳定效果（2136、2856）；
- **没有任何程序跨多个 origin 稳定**；
- 从张量重算的准入阶梯显示，绑定约束首先在**同一 origin 内的跨序列组**：

| origin | A 面准入 | B 面准入 | 双面 | 性质 |
| --- | --- | --- | --- | --- |
| 1176 | 10 | 20 | **0** | A/B 准入集完全不相交 |
| 1896 | 16 | **0** | **0** | B 面准入数为零 |
| 2136 | 17 | 30 | 16 | 有交集 |
| 2376 | **0** | 2 | **0** | A 面准入数为零 |
| 2616 | 15 | 3 | **0** | A/B 准入集完全不相交 |
| 2856 | 14 | 29 | 3 | 有交集 |

**因此 `P4-Evolution` 与 Natural Final 均不放行。**

---

## 4. Targeting 整体判负，保留一个案例（`p4g_cross_fitted_targeting.json`）

396 程序去重后得 **68 个处处可读的不同效果**；每折在训练侧自选 6 程序菜单（恒含
`raw`，增益 0，允许弃权），深度 3 决策树，15 维部署可见特征。0 Consumer fit。

| | 跨面（12 折，主）| 跨 origin（6 折，次）|
| --- | --- | --- |
| raw | 0.000000 | 0.000000 |
| best-fixed | **+0.2629** | +0.2853 |
| 交叉拟合 Targeter | **+0.1908** | +0.2780 |
| per-series oracle | +0.6106 | +0.6188 |
| Targeter 胜过 best-fixed | 5 / 12 | 3 / 6 |
| 通过有界预算 | best-fixed 1/12、Targeter 2/12 | 1/6、1/6 |

**正式判词：`FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE`。**

跨面被列为主方案，因为绑定约束在单个 origin 内部（见 §3）；跨 origin 只有 2 个 origin
有稳定候选，留一法几乎没有统计量。

**保留的案例** —— origin 2856 → Support-B：

| | 程序 | 增益 | 受害比例 | 最大单序列伤害 | 通过 |
| --- | --- | --- | --- | --- | --- |
| best-fixed | `period_complete>outlier_iqr`（单选） | +0.1908 | 0.35 | 0.107 | ✗ |
| Targeter | 逐序列路由到 4 个程序 | **+0.2331** | **0.20** | 0.076 | **✓** |

> 这是**一个合法的交叉拟合实例，将固定选择的风险失败转为预算通过**。

**不得**写成「Targeting 已有效」或「项目完整机制已得到证明」。整体上 Targeter 输 7 折，
平均低于 best-fixed 0.072，即对 +0.3477 的 oracle 空间**捕获率为负**。

### 绑定门的准确表述

逐折看，best-fixed 有 5/12 折 `harmed_fraction ≤ 0.20`，却只有 1 折通过全部三条；
1176-b、2616-b、2616-a（增益 **+0.7039**）、2136-a 均因 `max_single_series_harm > 0.30`
被挡。

> **直接绑定门是最大单序列伤害；当前 15 维特征、深度 3 树和六程序菜单无法稳定提前
> 识别该风险。尚不能唯一归因于特征、模型容量或样本量。**

---

## 5. Classification：现有菜单无稳定 headroom（`p4e_classification_headroom.json`）

只读 Adiac / ArrowHead 的 `*_TRAIN.ts`；两个 archive 各 3 个 TEST member，**读取字节
数 0**。判词 **`ONE_FACE_POSITIVE_ONLY`**，**TEST 继续密封**。

| | Adiac | ArrowHead |
| --- | --- | --- |
| 类数 | 37 | 3 |
| fit / A / B | 184 / 90 / 116 | 18 / 9 / 9 |
| identity Macro-F1（A / B） | 0.2969 / 0.3623 | 0.8857 / 0.7833 |
| 菜单 18 → 验证器拒 | 5 | 5 |
| 评估 13 → 双面稳定 | **0** | **0** |

全局唯一正数是 Adiac 上 `hampel_filter` 在 B 面的 **+0.0073**。13 个评估过的算子里
**10 个改动点数为 0**；被拒的 5 个全部是 `COHORT_MODIFICATION_FRACTION_EXCEEDED`
（`denoise_savgol`、`denoise_wavelet`、`smooth_ma`、`fft_decompose`、`smooth_ema`）。

**仪器风险**：ArrowHead 的 A/B 各 9 个样本 3 个类，`0.20 × 3 = 0.6 类` 意味着一个类都
不许变差；Adiac 的 A 面 90 样本摊到 37 类约 2.4 个/类。该负结果中有多少来自算子不足、
多少来自面太小，本轮分不开。

---

## 6. first fault 区分

两条线的根因同源——曝光数据里没有算子库所修的那类缺陷（KDD 缺口被上游填补、
UCR TRAIN 完整且已 z-normalize）——但**绑定约束不同**：

- **Forecast**：有效 Program 已存在，卡在 **Observation / 选择与跨 origin 泛化**。
- **Classification**：现有修复菜单大多无行为，全局变换又被局部修改门拒绝，
  卡在 **Program Space / 验证契约**。

---

## 7. 仪器修正（`p4d_param_correction_rerun.json`）

`run_forecast_p1._params` 对 `period_complete`、`impute_ar`、`repair_level_shift`
返回 `{}`，而分类线对这三个传 `period=24`（`classification_component.py:548-549`）。
后果不是外观问题：

- `period_complete`（`s1_impute.py:105-109`）在 `period < 2` 时返回 `interp_nan(y)`，
  故 `period=0` 时**它就是线性插补、与 identity 同一**，其 ZERO_BEHAVIOR 是仪器读数
  而非空结果；
- `impute_ar` 的阶数取 `max(8, period)`，算子自身 docstring 记录
  `linear 2.00 / AR(8) 1.33 / AR(24) 0.34`，故 `period=0` 时预测线跑的是 AR(8)，
  **看不见季节周期**；
- `repair_level_shift` 接受 `period` 但未声明为公开属性，修正对它**完全无效**（重跑
  后读数零变化），分类线传它是无害但无意义的。

修正**只在 variant 线内声明**，冻结的 P1 Common DSL 未改动
（`frozen_p1_common_dsl_edited: false`）。102 个受影响程序重跑：**28 个读数改变、
2 个程序从半可读变全可读、0 个新增稳定**。判词
`CORRECTION_CHANGES_READINGS_WITHOUT_NEW_STABILITY`。修正后 `period_complete` 拿到
全表最高的单臂 A 面增益 **+0.572**（origin 1896），但 B 面 −0.129，仍是跨面不转移。

**方法学副产品**：18 个算子中有 4 个在含缺口数据上零行为，故 396 个程序里有大量别名。
任何按程序计数的结论**必须先按效果向量去重**，否则会把 1 个发现报成 13 个（本轮
19 → 7 的塌缩即由此而来）。

---

## 8. 本轮不主张什么

- 不主张 Agent / Skill / held-out / fresh 层面的任何正结果；
- 不主张 Targeting 有效；
- 不主张泛化问题有任何进展；
- 不主张 P1–P4c 的数字被修正或替代——它们在 without 版本上测得正确；
- 不主张 Classification 的负结果已排除"面太小"这一解释；
- 不主张阈值（0.20 / 0.30）需要调整——本轮未调，也未据此论证。

---

## 9. 下一轮的冻结要求

- **不得继续在这 6 个 origin、这 12 个折上拟合特征或树模型。** 本轮已回答预定问题。
- 新的自然结构特征与**未评价 origin** 必须在打开前冻结，然后做一次真正的外部验证。
- 全局可逆表示变换（scale / stationarize）需要**独立风险契约**，不得简单放宽现有 10%
  修改门：`COHORT_MODIFICATION_FRACTION_EXCEEDED` 度量被修改点的**比例**，而可逆全局
  变换按定义修改 100% 的点，其取值恒为 1.0，与温和程度和可逆性无关。放宽阈值只会
  同时放行真正危险的局部大改；正确做法是换一类不变量（重构误差上界、形状保持、
  变换参数仅由 origin 之前的数据估计）。

---

## 10. 追加更正（2026-09-01，不改动上文任何数字）

来源：`artifacts/main_protocol/p4h_training_intervention_geometry.json`
（0 LLM、0 Consumer fit、0 held-out 读）。

### 10.1 `outlier_iqr` 的 A 面读数

此前在会话中把 `outlier_iqr` 的 Support-A 聚合增益报为「六个 origin 全部恰为
+0.000」并疑为第三个仪器问题，**该表述错误，予以撤回**。实际状态是
`WINDOW_VERIFIER_REJECTED (1 windows)`，六个 origin 全部如此。误报源于报告表达式
用 `(value or 0)` 把缺失的 `aggregate_gain` 强制成了 0。该算子远非空操作：它在
support_b 语料上改动 200 个训练窗口中的 108 个、共 1042 个点。

上文 §1–§9 的任何数字不受影响；`p4f` 工件中该臂本来就以 `failed` 字段正确记录。

### 10.2 窗口验证器是「全有或全无」

Forecast 的 `MAX_MODIFIED_FRACTION = 0.35`（Classification 为 0.10）。
`ScopeExecutor._verify` 对**每个将实际执行的训练窗口**独立 `verify_candidate`，
**任一窗口被拒即拒掉整个程序**。因此 200 个窗口中 1 个超限，就使该程序在该面、
在**全部** origin 上被永久拒绝。

由此，§1 中「396 枚举 → 171 可读」的收缩**有一部分来自合法性门而非性能失败**，
报告与后续设计不得把二者混为一谈。这与 §5 中 Classification 的
`COHORT_MODIFICATION_FRACTION_EXCEEDED` 是同一机制的两个面。

### 10.3 「跨 origin」应更名为「跨评价窗口」

训练窗口为 `anchor − 192 : anchor + 48`，anchors 冻结为 `[312 … 852]`，保留条件
`anchor + 48 ≤ origin`。**任何 ≥ 900 的 origin 都让十个 anchor 全部通过**，且 P4
路径从未覆盖 anchor 列表。语料指纹在两个面上跨六个 origin **完全相同**
（200 窗口 = 20 序列 × 10 anchor）。

又因 `_evaluate` 的 `x_train`/`y_train` 只来自训练窗口，可推出：

> **对给定的 (program, face)，六个 origin 共用同一个已拟合的 Ridge；变化的只有
> `x_eval` 与真值切片。**

因此六个 origin **不是该干预的六次独立复现**，而是同一个模型的六个评价点。
§3 与 §4 中的「跨 origin」应准确读作**跨评价窗口**；它证明的是同一训练干预所得
模型在多个预测窗口上的时间稳定性，**不是**该程序在六套独立训练条件上都能重新
训练成功。

§1 的 origin-2856 正结果**不受影响**：其表述本就是「用
`period_median_complete>outlier_*` 准备的语料训出的模型，在两个不相交序列组上
都胜过 identity 训出的模型」。

### 10.4 对下一阶段几何的约束

单纯新增 origin **不再被视为真正的外部训练泛化**——它只给出新的评价窗口，训练
语料与干预完全不变。阶段二主几何改为**更换训练序列**（新 development cohort），
以真正生成新的训练语料与新模型；**更换 anchor block** 作为可选的时间稳健性测试，
不得替代前者。

标注要求：仍只是 **development cohort holdout，不是 Final/held-out**；统计单位是
**训练 cohort / face**，origin 只是其内部的重复评价点。

---

## 11. 阶段二收口与主体更正（2026-09-01 追加，不改动上文任何数字）

工件：`p4k_phase2_development.json`、`p4l_phase2_frozen_decisions.json`、
`p4m_phase2_confirmation.json`、`p4n_serving_side_gap.json`。

### 11.1 阶段二结果，按其真实主体记录

阶段二在 cohort 2（`DEVELOPMENT_CONFIRMATION`，非 fresh、非 held-out）上一次性
读数，7 个 arm、100 次 Consumer fit：

| arm | 平均增益 | 受害 | 最大单序列伤害 | 覆盖率 | 逐 cell 通过 |
| --- | --- | --- | --- | --- | --- |
| raw | 0.000000 | 0.00 | 0.0000 | — | 0/10 |
| best_fixed_O0 | **+0.277549** | 0.34 | 1.8190 | — | 1/10 |
| best_fixed_O1 | +0.277549 | 0.34 | 1.8190 | — | 1/10 |
| targeter_X0_O0 | +0.240677 | 0.36 | 1.6372 | 0.97 | 1/10 |
| targeter_X1_O0 | +0.256716 | 0.33 | 1.8190 | 1.00 | 1/10 |
| targeter_X0_O1 | +0.206641 | 0.35 | 1.8190 | 1.00 | 0/10 |
| targeter_X1_O1 | +0.218460 | 0.35 | 0.8606 | 1.00 | 2/10 |
| per-series oracle | +0.624676 | — | — | — | 10/10（上界，非结果）|

**这些数字的主体是"固定程序 / 开环树 Router"，不是 Harness。** 它们只应作为
两件事的记录：

1. **训练语料策展的 Program 前提**：冻结程序在**全新训练序列**上仍给出 +0.2775，
   保留 cohort 1 的 74.9%（+0.3706 → +0.2775）。有效动作可迁移。
2. **开环 Router 基线**：四个树 Targeter 全部低于 best-fixed；X1 一致优于 X0，
   O1 一致劣于 O0；`targeter_X1_O1` 的最大单序列伤害 0.8606 不到其余（1.82）
   的一半，是唯一通过 2/10 个 cell 的 arm。

**不得**据此宣称完整 Harness 的性能。判词缩窄为
`FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE` **仅对当前开环树 Router 成立**；
`AUC 0.587` 只约束**部署前静态预测**，不约束能读取 Support-A 真实反馈的 Slow。

同时按阶段三门控关闭 O1 这批表示算子：三个可逆视图在固定选择上贡献恰好为零
（`best_fixed_O1` 与 `best_fixed_O0` 选中同一程序、数值逐位相同），在 Targeter
菜单里为负贡献。开发端读数为 identity +0.3706 / seasonal +0.3424 /
detrend +0.2042 / difference **−0.5446**。

### 11.2 更上游的机制缺口：服务侧没有作用点

`p4n_serving_side_gap.json` 用两次机械演示（0 LLM）确认：

**A. 两个任务都不处理被服务的数据。**

| 任务 | 程序作用于 | 被评分的数据 |
| --- | --- | --- |
| Forecast | `_evaluate` 的 `train_rows` 循环内 | `_linear_integrity(raw[origin-192:origin])`，**无程序**；真值为 raw 切片 |
| Classification | `MacroF1ConsumerAdapter._prepared_fit(cell.fit_values)` | `cell.surface(face)` 直接进 `model.predict`，**未处理**；标签不变换 |

**B. Scope 无法豁免被伤害的序列。** `train_series_scope` 存在且 functional/v1 线
在用，但它过滤的是**训练行**，而 `roster("support_a")` 训练于 Support-B、评价于
Support-A，两集不相交。实测：Scope 限制到 **1/20** 条训练序列，仍移动
**20/20** 条评价序列（最大变化 0.5049 / 0.5667）。

**因此主协议至今测的是"训练语料策展 Harness"，不是"Pattern-conditioned、
Target-local Scoped Harness"。** 缺口在 ScopeSpec 的**上游**：服务序列从未被
处理过，所以"不要处理这条序列"这句话在当前仪器里无法表达。

**不主张**：任何 P4 数字是错的——它们对仪器实际实现的语义是正确的；也**不主张**
Scope 是 `A5 == K0` 的唯一成因（Source 卡不可达、严格门、0 Active Skill 同样
参与）。Scope 是目前发现的**最上游**机制缺口。

### 11.3 下一阶段（裁定：选乙）

补齐真正的 serving-side scoped pipeline，而非在 eval 分支简单多调一次程序：

```text
Raw pipeline     : raw train    -> raw model     -> raw serve context
Program pipeline : prepared train -> program model -> prepared serve context
Scope            : 选中序列走 Program pipeline；未选序列走 Raw pipeline
```

未选序列因此与 Static **逐位相等**，raw fallback 才是真的。代价是多次 Consumer
fit，必须计费。

Forecast 三个表面：`train_context + train_target`（沿用现语义）、`serve_context`
（只用 origin 之前的数据做因果准备）、`evaluation_truth`（始终 raw）。
Classification：fit features 与 serve features 同样处理，labels 不处理。

顺序为：serving-side evaluator → ScopeSpec（保存部署可见特征谓词而非 UID）→
0-LLM 生命周期预检 → Static / A3 / A5 主实验。不再优先建风险感知树 Targeter，
也不再寻找适合全部序列的全局程序。
