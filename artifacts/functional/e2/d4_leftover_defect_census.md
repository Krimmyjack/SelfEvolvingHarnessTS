# D4 leftover 自然缺陷普查

日期: 2026-08-28。地位: 纯计算; 0 LLM / 0 harness / 0 fit / 不改仓内代码 / 不 git commit。
普查脚本(唯一仓内写入例外允许的脚本): `_scratch/d4_leftover_defect_census.py`。
机器件: `artifacts/functional/e2/d4_leftover_defect_census.json`。

**一行结论**: traffic leftover 6 个可开课 cell **全部 family-aligned = poor**(0 NaN; 周窗水平位移每 cell 0–3 条; 原始 MAD-8 虽几乎全员命中但被占有率右尾混淆)。electricity leftover 21 列 = **spare-only / moderate**, 不够一个 cell。该池只能承担**易档**(identity / 无修复), **不能**承担中/难自然缺陷发现, 也**不能**提供 G1 修订触发基率 ≥2/课。

权威: `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`:36-38 (D4)、:44-57 (三档难度)、:61-64 (G1)。切片开封: `artifacts/functional/e2/d4_fresh_pool_inventory.md`:82-94、:111。

## 1. 加载器与切片(可复核)

### 1.1 traffic leftover = `_load_pool` 可用列序的 [480:862]

列序与过滤复用 `evaluation/functional/run_e2_s2a_forecast_oracle.py`:69-85 `_load_pool`:

- 路径: `run_batch_composition_headroom.py`:169-180 `_traffic_csv_path` → `shared_tsq_datasets/traffic/traffic.csv`
- 读入: `g3_sourcing.load_csv_columns` (`g3_sourcing.py`:70-111), `max_columns=900`, `max_rows=20000` (`forecast_oracle.py`:73-75)
- 可用过滤: `series.size >= ORIGIN_HELDOUT+48` = 1848 (`forecast_oracle.py`:79; 原点常量 :51-52)
- 本盘实测: 862 数值列全部可用; 列名 `0`…`860` + `OT`(不是字面 `861`; 第 862 个可用名是 `OT`)

已开切片(不得算 leftover):

| 切片 | 可用下标 | 列名 | 证据 |
| --- | --- | --- | --- |
| S2a recut 7 cell | 0–419 | `0`–`419` | `_recut` `forecast_oracle.py`:88-114; inventory `:91` |
| clean identity | 420–479 | `420`–`479` | `s2a_course_frozen.json`:53-167; inventory `:92` |
| **leftover** | **480–861** | **`480`–`860` + `OT` (382)** | inventory `:93`; 本普查复证 `usable[480]=="480"`, `usable[-1]=="OT"` |

leftover 再按列序切 6×60 + spare 22:

| cell | 列名 | n |
| --- | --- | ---: |
| `traffic_leftover_00` | `480`–`539` | 60 |
| `traffic_leftover_01` | `540`–`599` | 60 |
| `traffic_leftover_02` | `600`–`659` | 60 |
| `traffic_leftover_03` | `660`–`719` | 60 |
| `traffic_leftover_04` | `720`–`779` | 60 |
| `traffic_leftover_05` | `780`–`839` | 60 |
| `traffic_leftover_spare` | `840`–`860` + `OT` | 22 |

容量门: `CELL_WIDTH=60`, `N_TRAIN=40`, `N_FACE=20` (`forecast_oracle.py`:45-48; inventory `:9-21`)。6 cell 过门; spare 22 不够一个 cell。

### 1.2 electricity leftover = sweep 预声明的 21 列

切片先核 `artifacts/functional/e2/s2a_g0_electricity_sweep.json`:20-43 `leftover_unused`: `300`–`319` + `OT`(21 列)。与 inventory `:88` 一致。

加载器对照: `run_e2_s2a_electricity_sweep.py`:58-60 `_load_pool` 用同一 `load_csv_columns`, **役中** `max_columns=400`, `max_rows=2500`。本普查对 leftover 用同一列解析、`max_rows=30000`, 实测 leftover 长度 **26304**(全列)。2500 前缀只作敏感性: 全长 family-aligned bearing 6/21, 前缀 3/21, 都不达 rich 杠, 结论不变。

## 2. 缺陷分类法对齐

`evaluation/minipipe/feedback/fault_routes.json` 是 Harness first-fault 路由(`CRITIC_BLIND` / `SKILL_CONTENT_GAP` / …), **不是**时序缺陷 kind。缺陷族来自注入分类法:

| 族 | 定义出处 | 本普查度量 |
| --- | --- | --- |
| `missing` / `gap` | minipipe `contracts.py`:20-22; 连续 NaN 游程 `contracts.py`:59-72; 役中 forecast 注入 `injection.py`:91-128(80 个孤立 NaN); minipipe 块缺失 `injections.py`:45-51(12 或 30 点连续 NaN) | NaN 计数与连续 NaN 游程。**零不当缺失** |
| `impulsive_outlier` | `injection.py`:38-89(40 个孤立点, 幅度 `8.0 * nanstd(pristine[120:900])`); minipipe `injections.py`:52-57(6× 或 10× std 孤立点) | 有限点 `\|x−median\|/MAD` 阈值 5 与 8; 最大游程; 孤立 := 游程长度 ≤3 |
| `level_shift` | minipipe `injections.py`:58-63(40 点×1.5 std 或 64 点×3.0 std 块偏移) | 非重叠窗 168(小时周), 相邻窗中位数跳变 > `3 ×` 全局 MAD |
| 零结构 | 任务书: traffic 是占有率 | 精确 `0.0` 的零率 / 零游程数 / 最长零游程; `treated_as_missing=false` |

役中公共稳健 z **不是**本表主度量: `public_features.py`:275-279 用 `|x-med|/max(1.4826·MAD, 1e-8)`, 旗标 `OUTLIER_Z_THRESHOLD=4.0`(`contracts/observables.py`:16)。本表按任务书用**未缩放** `|x-med|/MAD` @5/8, 比公共 z≥4 更严, 更接近注入幅度 8。

`period_change` 在分类法里(`contracts.py`:20-22)但任务未要求, 未计。

## 3. 富度判据(写明, 两层)

**标题 `richness` = family-aligned**, 不把占有率右尾算作 `impulsive_outlier`。

- **missing**: poor = 0 条有 NaN; moderate = (0, 30%); rich ≥30%。
- **level_shift**: 同上, 计「至少 1 次周窗跳变 >3×全局 MAD」的序列比例。
- **impulse_raw_mad**: 同上, 计「至少 1 条孤立 MAD-z≥8 游程(长≤3)」的序列比例。这是机械检测器读数。
- **impulse_family_aligned**: 若同时满足占有率尺度(cell 中位 mean<0.3 且 MAD<0.15)、≥50% 序列有孤立 MAD-8、且孤立 MAD-8 游程中位数 ≥5, 标 `uncertain_occupancy_confounded`(日周期/占有率尾, 不是 `injection.py` 的 40 点 8×nanstd 尖峰)。
- **标题 poor**: 0 NaN **且** 周跳变序列 <10% **且** 无未混淆的 impulse。
- **标题 rich**: ≥30% 序列有 NaN、或未混淆 impulse、或周跳变。
- 否则 **moderate**。零永不计入缺失/缺口。

占有率混淆的计数依据(leftover 全池 382 列): 中位 mean≈0.049、中位 MAD≈0.022; 孤立 MAD-8 游程中位 33、p90 286(注入只标 40 个点); 70%+ 序列有 MAD-5 长游程(>3)。这是右偏占有率, 不是稀疏传感器尖峰。

## 4. traffic leftover: 缺失率与零结构

**382/382 序列 missing_rate = 0**(NaN 游程数 = 0, 最长 NaN 游程 = 0)。注册表同族 `missing_rate=0` 在 leftover 上被实证, 不是纸面标签。

零是结构性占有率, 不是缺失: 全池零率 min/med/p90/max = 0.085% / **0.445%** / 1.61% / 14.6%; 最长零游程 med/p90/max = **17 / 34 / 589** 小时; 28/382 序列零率 >1%。cell 04 有一条最长零游程 589(约 24.5 天连续零)。不得把这些零当 `gap` 注入对象。

## 5. 六 cell + spare 汇总

长度一律 17544。MAD / 均值量级是占有率(mean 中位 0.04–0.06, MAD 中位 0.015–0.024)。

### 5.1 富度与族计数

| cell | 列 | 标题富度 | raw-MAD impulse | missing | level_shift | family-aligned bearing | NaN 条 | 孤立 MAD8 条 | 周跳变条 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| leftover_00 | 480–539 | **poor** | rich | poor | moderate | 3/60 | 0 | 50 | 3 |
| leftover_01 | 540–599 | **poor** | rich | poor | moderate | 1/60 | 0 | 58 | 1 |
| leftover_02 | 600–659 | **poor** | rich | poor | moderate | 2/60 | 0 | 58 | 2 |
| leftover_03 | 660–719 | **poor** | rich | poor | poor | 0/60 | 0 | 56 | 0 |
| leftover_04 | 720–779 | **poor** | rich | poor | moderate | 1/60 | 0 | 55 | 1 |
| leftover_05 | 780–839 | **poor** | rich | poor | poor | 0/60 | 0 | 58 | 0 |
| spare | 840–OT | **poor**(不够 cell) | rich | poor | poor | 0/22 | 0 | 19 | 0 |

六 cell 同质: 无缺失、几乎无水平位移、MAD-8 检测器全员富但被占有率混淆。**没有跨 cell 难度梯度**。

### 5.2 分布(median / p90 / max)

缺失率六 cell 全为 0/0/0, 表中省略。

| cell | 零率 | 最长零游程 | MAD-5 率 | MAD-8 率 | MAD-8 孤立游程数 | 周跳变次数 | MAD | 一阶差分 MAD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 0.476% / 1.38% / 6.74% | 21 / 34 / 96 | 1.80% / 7.17% / 12.4% | 0.222% / 3.18% / 7.59% | 30 / 298 / 526 | 0 / 0 / 3 | 0.0199 / 0.0327 / 0.0626 | 0.00515 / 0.0105 / 0.0145 |
| 01 | 0.456% / 1.29% / 5.30% | 17 / 31 / 76 | 2.06% / 7.36% / 10.7% | 0.294% / 4.11% / 8.68% | 36.5 / 340 / 549 | 0 / 0 / 1 | 0.0234 / 0.0318 / 0.0407 | 0.00605 / 0.00885 / 0.0119 |
| 02 | 0.493% / 1.68% / 2.06% | 19 / 56 / 89 | 1.61% / 9.11% / 22.7% | 0.322% / 3.84% / 18.8% | 43.5 / 357 / 453 | 0 / 0 / 6 | 0.0243 / 0.0315 / 0.0478 | 0.00620 / 0.00905 / 0.0139 |
| 03 | 0.553% / 2.68% / 11.6% | 17.5 / 33 / 56 | 0.901% / 5.99% / 13.2% | 0.114% / 1.88% / 11.5% | 16.5 / 187 / 389 | 0 / 0 / 0 | 0.0222 / 0.0380 / 0.0709 | 0.00590 / 0.00931 / 0.0132 |
| 04 | 0.462% / 2.65% / 14.6% | 19 / 34 / **589** | 1.48% / 6.12% / 11.7% | 0.268% / 2.91% / 7.89% | 38 / 240 / 401 | 0 / 0 / 2 | 0.0210 / 0.0293 / 0.0435 | 0.00540 / 0.00840 / 0.0117 |
| 05 | 0.296% / 1.31% / 11.0% | 14 / 32 / 76 | 1.48% / 7.07% / 12.4% | 0.331% / 2.61% / 6.95% | 43 / 207 / 460 | 0 / 0 / 0 | 0.0231 / 0.0335 / 0.0485 | 0.00620 / 0.00901 / 0.0155 |
| spare | 0.251% / 0.815% / 2.73% | 13.5 / 29 / 76 | 0.351% / 5.43% / 5.97% | 0.054% / 3.60% / 4.91% | 6.5 / 164 / 337 | 0 / 0 / 0 | 0.0153 / 0.0248 / 0.0280 | 0.00370 / 0.00764 / 0.00840 |

cell 02 的 MAD-5 最长游程 max=274, 是连续高峰段, 不是孤立尖峰。cell 04 的 589 小时零游程是结构空窗, 不是 NaN 缺口。

## 6. electricity leftover(spare-only)

21 列, 长度 26304, **不够一个 60 列 cell**。标题富度 **moderate**(family-aligned bearing 6/21=28.6%<30% rich 杠)。不是占有率尺度(中位 mean 2088, 中位 MAD 594)。

| 指标 | median / p90 / max |
| --- | --- |
| missing_rate | 0 / 0 / 0 |
| 零率 | 0.023% / 0.061% / 0.099% |
| MAD-8 率 | 0 / 0.848% / 11.5% |
| 孤立 MAD-8 游程数 | 0 / 54 / 835 |
| 周跳变次数 | 0 / 2 / 4 |
| bearing 序列 | `303`,`307`,`310`,`317`,`318`,`OT`(6/21) |

役中 sweep 只读 2500 行时, bearing 降为 3/21、周跳变 0; 仍是 spare-only moderate。不能单独开课。

## 7. G1 判读

门: `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`:61-64 — 冻池前预估每课修订触发(冲突 / 负反馈 / 挤占); 拟阈值 ≥2/课(协议未另冻)。课程结构对照: `s2a_course_frozen.json`:17-51(每单元一 cell, 产例 + 受益; 另有 clean identity 与 gap 守卫)。三档难度: 同文件 :44-57。

按普查所见, 若用这 6 个 leftover cell 开课(每单元一 cell, 产例+受益):

- **缺口/缺失素材**: 每课期望 **0**。382/382 无 NaN, 与 `inject_gap_corpus`(`injection.py`:91-128)或 minipipe 块缺失完全不对应。
- **水平位移素材**: 每课期望 **≪1**。全池仅 7/382 序列有至少 1 次周窗跳变; 六 cell 计数为 3,1,2,0,1,0。达不到「每单元都能出冲突/负反馈」。
- **尖峰素材**: 机械 MAD-8 很密, 但是占有率高峰。若 Agent 对真实高峰跑 winsorize/hampel/outlier_mad, **符号未知**(可能系统性地负, 也可能像 S2a 注入尖峰那样全员可学)。这不是冲突场: 六 cell 同质, 没有「有的序列该修、有的不该修」的已测分歧。
- **修订触发期望**: 来自 leftover **缺陷几何**的冲突/负反馈 ≈ **0/课**, 低于拟 G1 杠 ≥2/课。`meets_proposed_g1_ge_2_per_lesson = false`。
- **可承担难度档**: **仅易档**(identity / 无修复), 近于已开的 `traffic_clean_identity_00`(`s2a_course_frozen.json`:24-30,53-167), 外加一层均匀的占有率 MAD 尾。不能承担中/难自然缺陷发现, 也不能承担冲突场。S2a 已记录注入 `impulsive_outlier` 下冲突场不可得(`s2a_course_frozen.json`:4-5 `R2_forecast=untested`; 设计书 :61-64 把该课升为 G1 门)。leftover 自然层不能补上这块。

未建模、未跑 Consumer。G1 是缺陷几何代理, 不是 Episode 标签。把占有率高峰当尖峰去修会不会出负反馈, 本普查标 **不确定**。

## 8. 引证索引

| 对象 | 位置 |
| --- | --- |
| D4 四标准 | `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`:36-38 |
| 三档难度 | 同文件 :44-57 |
| G1 | 同文件 :61-64 |
| 容量门常量 | `evaluation/functional/run_e2_s2a_forecast_oracle.py`:45-48,65-66,79 |
| `_load_pool` | 同文件 :69-85 |
| `_recut` | 同文件 :88-114 |
| traffic 路径 | `evaluation/functional/run_batch_composition_headroom.py`:169-180 |
| `load_csv_columns` | `evaluation/functional/task_episode_harness/agentic/g3_sourcing.py`:70-111 |
| leftover 开封表 | `artifacts/functional/e2/d4_fresh_pool_inventory.md`:82-94 |
| clean identity 列 | `artifacts/functional/e2/s2a_course_frozen.json`:53-167 |
| 课程结构 | 同文件 :17-51 |
| electricity leftover 名单 | `artifacts/functional/e2/s2a_g0_electricity_sweep.json`:20-43 |
| electricity `_load_pool` | `evaluation/functional/run_e2_s2a_electricity_sweep.py`:58-60 |
| 缺陷族集合 | `evaluation/minipipe/contracts.py`:20-22 |
| NaN 游程几何 | 同文件 :59-72 |
| minipipe 注入 | `evaluation/minipipe/corpus/injections.py`:45-74 |
| forecast 尖峰/缺口注入 | `evaluation/functional/task_episode_harness/injection.py`:38-89,91-128 |
| 公共稳健 z | `runtime/public_features.py`:275-279; `contracts/observables.py`:16 |
| fault routes(非缺陷 kind) | `evaluation/minipipe/feedback/fault_routes.json` 全文 |

## 9. 限制

- 零 Consumer / 零 Episode; 修订触发是几何代理。
- 原始 MAD-z 在占有率上会把高峰标成「离群」; 标题富度已扣除, 机械层仍保留在 `richness_raw_mad_impulse`。
- 周窗中位数跳变不是 minipipe 40–96 点往返 excursion(`public_features.py` `_level_candidate`, :150 起)。
- electricity leftover 主表用全长 26304; 役中 sweep 只读过 2500 行。
- `period_change` 未测。
- leftover 最后一列名是 `OT`, 不是 `861`; 切片仍是可用列序的 382 列。
