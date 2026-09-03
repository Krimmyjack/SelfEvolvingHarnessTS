# D4 预测 fresh 密封池盘点

日期: 2026-08-28。地位: 只读盘点, 0 LLM / 0 fit / 0 下载 / 不改代码 / 不跑实验。
权威: `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`:36-38 (D4 四标准)。
机器件: `artifacts/functional/e2/d4_fresh_pool_inventory.json`。

**一行结论**: 仓内+旁路包共 **32** 项时序资产、**25** 个 forecast 候选池; **无** 四标准全满足者。最接近 = `tsl_traffic_leftover_480_861` (382 列未入 cell, 容量过门), 缺 ④ 族已开封与 ③ 自然缺陷未实证。ETT / exchange / weather / M4 / ILI **已在** 旁路包 `shared_tsq_datasets`, 不是「需外部下载」。真正缺的标准集 top3 = Solar-Energy / PEMS-BAY / GEFCom2014。

## 1. 容量门常量

冻结定义 (forecast cell 几何) = `evaluation/functional/run_e2_s2a_forecast_oracle.py`:45-48:

| 常量 | 值 | 行 |
| --- | ---: | --- |
| `N_TRAIN` | 40 | :45 |
| `N_FACE` (Support / Delayed 每半) | 20 | :46 |
| `N_HELDOUT` | 20 | :47 |
| `CELL_WIDTH` | 60 | :48 |
| `ORIGIN_HELDIN` / `ORIGIN_HELDOUT` | 1104 / 1800 | :51-52 |
| 同构最小长度 | 1848 (`ORIGIN_HELDOUT+48`, :79) | :79 |
| 材料线 | `max(0.005, 1/n_half)` → n_half=20 时 = 0.05 | :65-66 |

闸门应用 = `evaluation/functional/run_e2_s2a_natural_pool.py`:31-32 (`N_TRAIN_GATE=40`, `N_FACE_GATE=20`), 判定 :110-115。模块头 :4 写明 *Capacity gate is the current forecast cell rule (TRAIN>=40, half>=20)*。

政策出处 (分类线终态, 预测线类比为「序列作行」) = `docs/CLS_LINE_FINAL_REPORT_2026-08-28.md`:43。

**不是** D4 容量门: G3 `MIN_SERIES_LENGTH=5760` (`g3_sourcing.py`:34-38)。

窗补充 (非门, 仅供长序列判断): `n_windows = max(0, (L-192-48)//24+1)`。门本身是序列作行, 不是窗作行。

## 2. 资产总表

`n_assets=32`。旁路包根 = `C:/Users/辉/Desktop/Agent/shared_tsq_datasets` (仓库外, 加载器已写死该路径)。注册表 = `artifacts/frozen/benchmark_v02/series_registry.jsonl` (**1919** 行, 本盘点逐行解析)。`clean_base` 物化 **868** / 1919。

### 2.1 预测 / 通用

| 资产 | 路径 | 序列数 | 典型长度 | 加载器 |
| --- | --- | ---: | ---: | --- |
| TSL electricity | `shared_tsq_datasets/electricity/electricity.csv` | 321 | 26304 | `run_e2_s2a_electricity_sweep.py`:46-60; `g3_sourcing.py`:70-111 |
| TSL traffic | `shared_tsq_datasets/traffic/traffic.csv` | 862 | 17544 | `run_batch_composition_headroom.py`:169-190; `run_e2_s2a_forecast_oracle.py`:69-85 |
| TSL weather Jena | `shared_tsq_datasets/weather/weather.csv` | 21 | 52696 (小时化 `::6` → 8783) | `g1.py`:3372-3384, 3387-3493 |
| ETT-small ×4 | `shared_tsq_datasets/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv` | 7/文件 | 17420 / 17420 / 69680 / 69680 | **无** 仓内 forecast cell 加载器; 仅 `s0_census.py`:256-263 纸面排除 |
| exchange_rate | `shared_tsq_datasets/exchange_rate/exchange_rate.csv` | 8 | 7588 | `g3_sourcing.py`:248-256 (provenance, 非 live cell) |
| ILI / illness | `shared_tsq_datasets/illness/national_illness.csv` | 7 | 966 | `g3_sourcing.py`:239-247 |
| M4 | `shared_tsq_datasets/m4/*-train.csv` | H414 / D4227 / W359 / M48000 / Q24000 / Y23000 | H≈700; W first20=457-2597; Y≈19-31 | 无 S2a 加载器; 历史 Hourly/Daily 工件见下 |
| registry `uci_electricity_load_diagrams` | `data/benchmark_v0_2/clean_base` + registry | 370 (盘上 171) | 1024 | `registry.py`:660-668; `run_e2_s2a_iv_expand.py`:52-97 |
| registry `monash:traffic_hourly` | 同上 | 862 (盘上 391; virgin 806 / probe 56) | 1024 | 同上 |
| registry `metr_la` | 同上 | 207 (盘上 89) | 1024 | `run_e2_s2a_iv_expand.py`:47-97 |
| registry `monash:nn5_daily` | 同上 | 91 (盘上 48) | 714-791 | 同上 |
| registry `monash:covid_deaths` | 同上 | 246 (盘上 104) | 212 | `registry.py`:660-668 |
| registry `gefcom2012_load` | 同上 | 20 (盘上 10) | 1024 | `sources.py`:208-219 |
| registry `noaa_global_hourly` | 同上 | 40 (盘上 19) | 1024 | `run_e2_noaa_health_check.py`:58-61 |
| registry legacy_monash 七套 | 同上 | 83 (盘上 36) | 187-1024 | 全部 `confirmed_exposed` |
| NOAA fresh v1 | `data/benchmark_noaa_fresh_v1/series/` | 20 | 8760 (`series/72422093820/record.json`:6) | `run_e2_noaa_fresh_materialize.py`:120-121; `run_e2_fresh_confirmation.py`:479 |
| NOAA raw v0 | `data/benchmark_v0/raw/noaa_global_hourly/` | 选择宇宙 64 (`sources.py`:67) | 不规则小时 | `run_e2_noaa_fresh_materialize.py`:121 |
| KDD 2018 T233 | `data/kdd2018/series_cache.npz` | roster 20 (cache 总数未开 npz) | 未本盘点 | `g1.py`:1881-1908; `e1.py`:289 |
| M3 quarterly zip | `data/m3_quarterly_dataset.zip` | 未解压 (96124 B) | 未知 | 无加载器 |
| Tourism quarterly zip | `data/tourism_quarterly_dataset.zip` | 未解压 (93833 B) | 未知 | 无加载器 |

注册表 `certified_virgin` **不是**「实验从未开封」: 那是 v0.2 冻结标签; 实验开封以 artifacts/docs 为准。

### 2.2 分类

| 资产 | 路径 | 规模 | 加载器 |
| --- | --- | --- | --- |
| UCR 本地 40 zip | `data/ucr_task_context/*.zip` | 40 数据集 (本盘点 `iterdir` 点名) | `run_e2_t6_cls_op_shared_harness.py`:117; `run_e2_action_credit_candidate_ordering.py`:33,265-267 |
| CLS-CONF D1/D2/D3 | `data/ucr_conf_downloaded/` | D1 已转码后算力停; D2/D3 仍封 (`ROSTER.md`:17-21,33-46) | `:2605`; `run_e2_capstone_epilepsy2.py`:103 |
| UEA .ts 旁路包 | `shared_tsq_datasets/{PEMS-SF,...}` | 分类样本; PEMS-SF = 144 步 (`s0_census.py`:278-281) | 无 forecast 加载器 |

### 2.3 异常检测

| 资产 | 路径 | 规模 | 加载器 |
| --- | --- | --- | --- |
| Yahoo S5 A1 | `data/benchmark_yahoo_s5_v1/` | 下载 67 / roster 65 / EXPOSED 24 / SEALED 41 | `run_e2_t6_natural_a5_a3.py`:297,5482-5486 |
| NAB v1.1 | `data/benchmark_nab_v1_1/raw/` | 8+6+7+10+6=37 | 同文件 :123, :540-544 |
| SMD | `shared_tsq_datasets/SMD/` | 28×38; train 708405 行 (`PROJECT_STATE`:74) | `run_e2_smd_entity_structure.py`:33-34 |
| PSM / SWaT | `shared_tsq_datasets/{PSM,SWaT}/` | 25 / 51 通道 | `g3_sourcing.py`:257-274 |
| MSL / SMAP | 同旁路包 | packed npy | 无仓内 AD 考场加载器 |

空目录 (有名无文件, 不计入可用资产): `electricity_15min`, `m5`, `weatherbench_daily`, `wiki_daily_100k` 等 (`s0_census_v1.md`:39; 本盘点复证 `n_files=0`)。

## 3. electricity / traffic 切片开封

| 切片 | 列/序列 | 开封状态 | 证据 |
| --- | --- | --- | --- |
| electricity recipe/G3 12+8 | 过守卫后的前 20 列 | 已在 development 使用 | `runner.py`:143-174; `PROJECT_STATE`:70 |
| electricity S2a 5 cell | 0-299 (300) | 已在 development 使用 | `s2a_g0_electricity_sweep.json`:9-15,51-56; 课程用 `_01/_03/_04` (`s2a_course_frozen.json`:17-45) |
| electricity leftover | 300-319 + `OT` (**21**) | 族已开; **本切片未评 Outcome** | `s2a_g0_electricity_sweep.json`:20-43 |
| traffic recipe 12+8 | 列 0-19 | 已在 development 使用 | `run_batch_composition_headroom.py`:163-164 |
| traffic S2a recut | 列 0-419 (7×60) | 已在 development 使用 | `run_e2_s2a_forecast_oracle.py`:88-114 |
| traffic clean identity | 列 420-479 | 已在 development 使用 | `s2a_course_frozen.json`:53-167 |
| traffic leftover | 列 480-861 (**382**) | 族已开; **本切片未入 cell** | 课程只用到 479; `_load_pool` 仍读入全部长列 (`forecast_oracle.py`:73-85) |
| registry electricity 370×1024 | 盘上 171 | 族已开 (与 TSL 同 UCI 族) | `g3_sourcing.py`:217-223 |
| registry traffic 862×1024 | 盘上 391; probe 56 | 同 PeMS 族 | `g3_sourcing.py`:225-237; registry `probe_consumed=56` |

`g3_sourcing.py`:211-215 写明: TSL electricity 与已打开的 UCI 族重叠, **不能** 扛跨域 fresh 主张。traffic 更是同一路网 (`:231-237`)。

## 4. 候选池资格四标准

门: ① 序列量 `n>=40`; ② 反馈容量 Support/Delayed 各 ≥20 **且** 能切出协议窗 (同构 L≥1848, 或已冻结的短 origin); ③ 自然缺陷异质性; ④ 从未开封。

**全满足: 无。**

| 候选池 | ① | ② | ③ | ④ | 开封 |
| --- | --- | --- | --- | --- | --- |
| `tsl_electricity_used_300` | 满足 (300) | 满足 (5×20/20; L=26304; 窗 1086) | 不满足 (注入 impulsive_outlier) | 不满足 | development |
| `tsl_electricity_leftover_21` | 不满足 (21) | 不满足 (半=10) | 不满足 (同清洁族) | 部分 | 切片未评 / 族已开 |
| `tsl_traffic_recipe_0_19` | 不满足 (20) | 不满足 (12+8) | 部分 | 不满足 | development |
| `tsl_traffic_s2a_recut_0_419` | 满足 | 满足 (7 cell; 窗 722) | 不满足 (注入) | 不满足 | development |
| `tsl_traffic_clean_420_479` | 满足 | 满足 (20/20) | 不满足 (明示无缺陷 identity) | 不满足 | development |
| **`tsl_traffic_leftover_480_861`** | **满足** (382; 6 cell) | **满足** | **部分** (同网占用; missing_rate=0; 未单测 leftover) | **部分** | 切片未评 / 族+文件已开 / INSTANCE_SEEN |
| `registry_uci_electricity_370x1024` | 满足 | 部分 (L=1024 扛不住 1104/1800) | 部分 (4 regime; 无缺失) | 不满足 | 同族已开 |
| `registry_traffic_hourly_862x1024` | 满足 | 部分 (L=1024) | 部分 | 不满足 | 同族; probe 56 |
| `metr_la` | 满足 | 部分 (盘上 89→1 cell; origin 792/888) | 部分 (注入 cell + 3 regime) | 不满足 | **已密封使用** (`s2a_iv_decomposition.json`:14-25; `V1_CROSS_DOMAIN_CLOSED_LOOP_PLAN.md`:1071+) |
| `metr_la_unused_remainder` | 不满足 (盘上剩 29) | 不满足 | 部分 | 部分 | 切片 vs offset 重叠不确定 |
| `monash_nn5_daily` | 满足 | 部分 (盘上 48<60; L≤791; 窗半 11) | 部分 (91/91 有自然缺失, max 0.0367) | 不满足 | development (`s0_census.py`:190-199) |
| `monash_covid_deaths` | 满足 | 不满足 (L=212; 窗 0) | 部分 | 不满足 | `g3_sourcing.py`:284 |
| `gefcom2012_load` | 不满足 (20) | 不满足 (半=10) | 部分 (20/20 有缺失) | 不满足 | `s0_census.py`:203-210 |
| `noaa_registry_40x1024` | 部分 (40 贴线无 heldout; 盘上 19) | 部分 (L=1024) | **满足** (40/40 自然缺失, max 0.52; `dataset_manifest.json`:104-110) | 不满足 | NOAA 族已开 |
| `noaa_fresh_v1` | 不满足 (20) | 不满足 (12+4; 半=10) | **满足** (`record.json` 原生缺测) | 不满足 | 2024+2025 EXPOSED (`AGENTS.md`:232) |
| `kdd_t233` | 不满足 (20) | 不满足 | 部分 | 不满足 | `g1.py`:1881-1887 |
| `weather_jena` | 不满足 (21) | 不满足 (11+8) | 部分 (Weather 课是损失跨度, 非缺陷普查) | 不满足 | `g1.py`:3460-3493; `s0_census.py`:153-162 |
| `ett_small` | 不满足 (7/文件) | 不满足 (序列作行) | 部分 | 部分 | **从未开 exam**; 能源族已暴露 (`s0_census.py`:256-263) |
| `exchange_rate` | 不满足 (8) | 不满足 | 部分 | 部分 | 无 exam; finance 族已暴露 |
| `illness_ili` | 不满足 (7) | 不满足 (L=966) | 部分 | 部分 | 无 exam; epidemiology 族已暴露 |
| `m4_hourly` | 满足 (414) | 部分 (L≈700; 历史 origin 604/652) | 部分 | 不满足 | `m4_hourly_stop_rule_fresh_cohort_report.json`:256-268 |
| `m4_daily` | 满足 (4227) | 部分 (长度不齐; L≥1848 比例未全量) | 部分 | 不满足 | `cross_series_workflow_m4_daily_target_plan.json`:75 |
| `m4_weekly` | 满足 (359) | 部分 (first20 部分 ≥1848; 无加载器) | 部分 | 部分 | e2 无独立 Weekly exam 工件 |
| `m4_monthly` | 满足 | 不满足 (first20 max 918; 窗半 14) | 部分 | 部分 | e2 检索 Monthly exam 零命中 |
| `m4_yearly` | 满足 | 不满足 (L≈20; 窗 0) | 部分 | 部分 | e2 检索 Yearly exam 零命中 |

### 最接近者

1. **`tsl_traffic_leftover_480_861`**: ①② 满足。缺 ④ (PeMS 族 + 整文件 INSTANCE_SEEN) 与 ③ (未对 leftover 做自然缺陷普查; 注册表该族 missing_rate=0)。
2. **`m4_weekly`**: ① 满足, ④ 仅「无 Weekly exam」。缺 ② 长度分数未冻结、③ 竞赛清洗、④ M4 Hourly/Daily 已开封。
3. **`ett_small`**: ④ 最干净 (从未开 exam)。缺 ①② (7 通道)。能源族重叠会挡 fresh-family 主张。

## 5. 缺失标准集 (仓内+旁路包都没有)

ETT / exchange_rate / weather / M4 / ILI **已在旁路包**, 见 §2.1, **不要**标「需外部下载」。

| 名称 | 规模量级 | 获取 | 单变量切片 | 状态 |
| --- | --- | --- | --- | --- |
| Solar-Energy (LSTF) | ~137 场 × ~5e4 小时 | TSL `dataset/solar` / thuml 数据发布 | 适合 | **需外部下载** |
| PEMS-BAY | ~325 传感器 × 5min × ~5e4 | DCRNN Drive (≠ metr_la, ≠ PEMS-SF .ts) | 适合 | **需外部下载** (下载后仍是 traffic 族) |
| GEFCom2014 load | 小时负荷, 数十区 × 2-3 年 | Kaggle (`sources.py`:221-234) | 适合 | **需外部下载** |
| Wind / WindPower | 1-7 通道 × 1e4-5e4 | NREL / TSL | 适合 | **需外部下载** |
| ENTSO-E actual total load | 国家/区小时 | `transparency.entsoe.eu` (`sources.py`:196-206) | 适合 | **需外部下载** |
| 官方 UCI 370×15min zip | 370 表 × 四年 15min | UCI dataset-321 (`sources.py`:174-185) | 适合 | **需外部下载** (与已开 TSL 321 同族) |

## 6. AD 快查 (次要)

Yahoo S5 A1 是现役 Target: `data/benchmark_yahoo_s5_v1`, work/held_in/held_out 各 65 个 `real_*.csv`。契约 `t6_42f_yahoo_a1_freeze.json`:11-16 = 下载 67、roster 65、丢 `real_54/62` (n=741)。前 24 条 EXPOSED, 余 **41 SEALED** (`AGENTS.md`:213-214)。加载器 `run_e2_t6_natural_a5_a3.py`:5482-5486。典型长 ≈1420。

NAB 37 文件全在 Source/AdExchange 角色上打开, 不能再称 fresh (`PROJECT_STATE`:72-73)。SMD 多变量, test 仍封, 单变量协议不匹配 (`PROJECT_STATE`:74)。

**够不够「安全弃权验收」**: 41 条 sealed 规模够做一次冻结后的四臂正确弃权验收 (D1 最廉腿), 前提是 development 管线先冻结、且不再打开这 24 条冒充 fresh。不够的是第二块未曝光自然 AD 域。#43 12/12 负只关当前菜单正效应, 不关弃权验收。

## 7. 检索备忘 (可复核)

- 注册表 14 个 `dataset_id` 的 n/长度/exposure/missing: 本盘点逐行解析 `series_registry.jsonl` (1919 行)。
- TSL CSV 行列: 本盘点 `csv.reader` 计行 (electricity 322×26304; traffic 863×17544; ETT/weather/exchange/illness 同上)。
- `shared_tsq` 空目录: 本盘点 `iterdir` `n_files=0`。
- ETT 作 forecast exam: `artifacts/` + `docs/` 除 s0 外无 ETTh1/ETTm1 考场工件。
- M4 Weekly/Monthly/Yearly 独立 exam: `artifacts/functional/e2` 文件名零命中; Hourly/Daily 有工件。
- Solar / PEMS-BAY / Wind: 旁路包顶层与 `data/` 无对应目录。

不确定 (禁止猜): KDD `series_cache.npz` 总序列数; M4 Daily/Weekly 全量 min/max 长度与 L≥1848 比例; leftover traffic/electricity 的逐列自然缺陷率; metr_la 历史 offset roster 与盘上 29 leftover 是否重叠。
