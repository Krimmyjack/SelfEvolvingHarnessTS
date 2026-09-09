# 数据资产审计（HEC 下一阶段 / 计划附录 E.3）

地位：只读台账、说明、目录元数据与适配器源码。0 新实验 LLM、0 Consumer fit、0 下载、0 新 SHA、不改生产代码/合同/主计划。  
平行于 Opus M-R0；本文件不执行 M-R0。  
机器件：`artifacts/main_protocol/m_data_asset_audit.json`。

纪律摘要：不打开 Solar zip、Yahoo-41 原始表、NAB 剩余官方序列标签、曝光未知的 UCR TEST。Solar 的 0 缺失只排除缺口依赖 Skill，不宣布全部 Skill 无增益。NOAA 2025 不得称 fresh。NAB 按族记录。UCR 按实验线 × 版本 × split 记录。

---

## 1. 可合法使用的数据角色表

| 资产 | 精确身份 | 路径 / 版本 | 合法角色 | 非法角色 |
| --- | --- | --- | --- | --- |
| KDD 2018 已填补 | `EXPOSED_DEVELOPMENT__KDD2018_WITHOUT_MISSING` | `data/kdd2018/raw/kdd_cup_2018_dataset_without_missing_values.tsf`；缓存 `data/kdd2018/series_cache.npz`（270 条，缓存 NaN=0） | development / 无缺口 outlier 线 replay | fresh Target；与含缺失并表；用它关闭 imputation |
| KDD 2018 含缺失 | `EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING` | `data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip`（Zenodo 4656719 v4，md5 已核）；270 条，可读 239；台账缺失 503,712 / 2,942,364 | **闭环课程与机制测量的主 development / Source（Phase S K0）/ 已跑过的 Target held-in** | fresh / Natural Final。`p4t` 上未评的 `(series, origin)` 对不能把整份身份改回 virgin |
| NOAA 2024 开发阵 | `benchmark_noaa_fresh_v1` `series/` 20 站 × 8760 | `data/benchmark_noaa_fresh_v1/` | **仅 development**（含 P4-Evolution NOAA 修订线） | fresh Target；HEC-1 课程域（适配器未接） |
| NOAA 2025 确认区 | 同一 cohort，索引 `[8760, 17520)` | `confirmation_2025/` + `raw_2025/` | **仅 development / replay**。历史 `FRESH_A5_DELIVERS` 是当时 fresh 区域上的反馈消耗式适应，现已打开 | 再次称 fresh；当作 Fast-only held-out |
| NOAA `beyond_17520` | 同站延长段 | 无当前书 | **保持密封** | 任何读取 |
| Solar 10 min | Monash `solar_10_minutes` / LSTF Solar-Energy 同源 | `data/solar_10_minutes/raw/solar_10_minutes_dataset.zip`（Zenodo 4656144 v3，md5 已核）；137 条 × 52560，**授权缺失计数 = 0** | **F2 密封终验候选**；隔离令有效。0 缺失**不**取消 K0 | development；现在打开；以 0 缺失宣布全部 Skill 无增益 |
| Yahoo S5 A1 24 条 | `yahoo_s5_a1` 字典序 roster 前 24 | 见 §4 | AD **development**（M0-a 检查面） | fresh；继续加算子拟合这 24 条 |
| Yahoo S5 A1 41 条 | 同 freeze 余下 41 | 见 §4 | **密封一次性终验**（须等 development 管线可冻结） | 现在读取 Outcome |
| NAB `realAdExchange` | NAB v1.1，本地 6/6 | `data/benchmark_nab_v1_1/raw/realAdExchange/` | development（原 Target，Outcome 已开） | fresh Target |
| NAB `realAWSCloudwatch` 本地 8 | Source cohort 1 | `.../realAWSCloudwatch/` 8 个 `ec2_cpu_*` | development / 已用尽的 Source | 第 5 Source cohort；virgin Target |
| NAB `realKnownCause` 本地 6 | Source cohort 2 | `.../realKnownCause/` | 同上 | 同上 |
| NAB `realTraffic` 本地 7/7 | Source cohort 3 | `.../realTraffic/` | 同上（更早还有检索级 INSTANCE_SEEN） | 把 AdExchange 结论当本族未核状态；virgin Target |
| NAB `realTweets` 本地 10/10 | Source cohort 4 | `.../realTweets/` | 同上 | 同上；Source family 已封顶 |
| UCR（本轮） | 多线多版本，见 §5 | 见 §5 | 分类 development 仅限**该线已开的 TRAIN**（及 capstone 已开的 Epilepsy2 TEST） | 用「P4d TEST 未读」宣称整个 UCR 未曝光；本轮 HEC 不使用 UCR |

现役 Skill（只按必要条件 + 已授权元数据）：

- Phase S K0：`outlier_mad({})` @ `serving_series_predicate[local_robust_z_peak>=3]`。**不要求缺失。**
- `period_median_complete → outlier_*` 与非恒等 impute：**要求缺失**（P4d / AGENTS §8.1）。

---

## 2. 已有适配器与接入缺口

| 资产 | 适配器 | 仅有代码？ | 既有运行证据 | 接入缺口 |
| --- | --- | --- | --- | --- |
| KDD 含缺失 | `preflight_natural_gap_variant.load_variant` → `run_main_baselines._cell` → `scoped_serving_evaluator` → `run_hec1` | 否 | HEC-1 三顺序 live；`p4o_scoped_serving_preflight.json` | **无。这是唯一接上 HEC-1 serving 的数据。** |
| KDD 已填补 | P1 `ForecastCell` + `series_cache.npz` | 否 | P1–P4c | 与含缺失身份平行，不可混用 |
| NOAA | 旧 `DATASET_CONFIGS['noaa']`（锚 240–660）；`run_forecast_p4_evolution_revision._load_noaa_cell`（12+4，2024 阵） | 否 | NOAA P4-Evolution 修订跑；`fresh_confirmation_v1` | **没有**把 NOAA 映成 HEC-1 课程几何（20/20 面、锚 `[312…852]`、origin 1176–3816、scoped serving）的工厂。这是 M0-b 的阻塞缺口 |
| Solar | P1 旗标 `traffic_or_solar_loader_available=False`。旧 plan-only 几何（12/4/4，period 144，stop 928） | 仅计划工件 | 无 Consumer Outcome | 无 HEC-1 serving；隔离令禁止为接适配器而开 zip |
| Yahoo-24 | `_load_yahoo_l1_roster` + `WindowedIForestAdapter`；P3 走同一 roster | 否 | #42g–#43、P3 AD | P1 `yahoo_loader_available=False`（P1 用合成夹具）。41 条故意不在 L1 roster |
| Yahoo-41 | 磁盘上有 `work/` 与双 vault，加载器切片停在 24 | 无授权 41 条评价路径 | 无 Outcome 跑数 | 保持密封 |
| NAB 五真实族（本地 37） | T6 `run_e2_t6_natural_a5_a3.py` | 否 | frozen plan v2、42d/42e census 与 replay | P1 `nab_loader_available=False`。不能当新 Target |
| UCR TRAIN-only | `classification_component.py`；P0b Adiac/ArrowHead | 否 | P0b / P1 / P3 / P4e | 本轮不进 HEC。Capstone 另有 TRAIN+TEST 加载器，且 **Epilepsy2 TEST 已在该线打开** |

---

## 3. 密封 / 曝光尚不清楚的资产

**已能定性（不是 UNKNOWN）：**

- Solar Outcome：密封；结构 AGGREGATE_SEEN。
- Yahoo-41 Outcome：密封。
- NOAA `beyond_17520`：密封、无新书。
- UCR CatsDogs D2：密封。
- UCR Adiac / ArrowHead TEST（P0/P4e 线）：密封。
- NAB 本地 37 条：按族全部作为 Source 或 AdExchange Target 打开，不能称 fresh。

**UNKNOWN 或需另裁（本审计故意未开材料）：**

1. Solar 上 `local_robust_z_peak>=3` 是否对任何服务序列成立 —— 只能看数值，隔离令禁止。
2. NOAA 磁盘上是否存在 index≥17520 的小时 —— 未扫 csv 正文。
3. Yahoo-41 在 freeze 长度门之后是否有进程读过标签列 —— 台账说没有；未打开那些 csv。
4. NAB 官方未落盘的 10 条（AWS 9 + `rogue_agent_key_updown`）：P0 记 `leftover_label_outcome_exposed_by_legacy_global_load=10`。本审计**未**打开 `labels/combined_windows.json` 复核键名。即使下载，也不能当作未曝光 Target，除非用户另裁 freshness。
5. `data/ucr_task_context/` 其余 zip 的 TEST 成员：除已列线外，未做逐库 TEST 台账。P4d 的 TEST 0 读**不能**外推到这些 zip。

---

## 4. Yahoo 67 / 65 / 24 / 41

加载规则（`run_e2_t6_natural_a5_a3.py`）：`sorted(freeze['roster'], key=file)[:24]`。

| 层 | 计数 | 内容 |
| --- | ---: | --- |
| 下载 | 67 | `raw/A1Benchmark/real_1.csv` … `real_67.csv` |
| 长度门丢弃 | 2 | `real_54.csv`、`real_62.csv`（n=741，`MIN_LENGTH=1000`） |
| 结构 roster | 65 | `work/`、`vaults/held_in/`、`vaults/held_out/` 各 65 个文件 |
| 序列内时间墙 | 65 条各自 | held-in `[0, 0.7n)`，held-out `[0.7n, n)` |
| 曝光 24 | 24 | **文件名字典序**前 24，不是 `real_1`–`real_24` |
| 密封 41 | 41 | roster 其余 |

曝光 24：`real_1, 10–19, 2, 20–30, 3`。  
因此 **`real_4`–`real_9` 在密封 41 里**，尽管编号更小。

这 24 条的时间 held-out 已被当作 `development_exposed_eval` 打开，不是真 held-out。真密封终验是**另外 41 条整条序列**（含它们自己的时间 held-in/out）。

---

## 5. UCR：按线、版本、split

| 线 | 档案 | TRAIN | TEST |
| --- | --- | --- | --- |
| P1/P3 分类 | Epilepsy2=`D3_reserve/EpilepticSeizures.zip`；GunPoint、PowerCons ∈ `ucr_task_context/` | 已曝光 | **本线未读**（无 held-out 加载器） |
| P0b / P4e | `data/main_experiment_p0/ucr_fresh/{Adiac,ArrowHead}.zip` | 已曝光（adapter / headroom） | **本线密封**（P4e：`ucr_test_member_bytes_read=0`） |
| CLS-CONF D1 | `ucr_conf_downloaded/D1/BinaryHeartbeat.zip` | 转码后载入，算力终止 | 转码产生 TEST txt，值已载入 |
| CLS-CONF D2 | `D2_sealed/CatsDogs.zip` | 未载入值 | 密封 |
| CLS capstone | 同一份 D3 `EpilepticSeizures.zip` | **已开** | **已开**（`unseal_record.first_read_scope`：TRAIN+TEST 解析为 float） |

P4d「UCR TEST 零读取」只约束那条天然缺口分类扫描，不使 capstone 的 Epilepsy2 TEST 或 D1 BinaryHeartbeat 变成未见。

---

## 6. Solar 与现役 Skill（不看原始数值）

授权元数据：137 条、等长 52560、缺失 0（`docs/D4_DOWNLOAD_FREEZE_2026-08-29.md` §4）。zip 未打开。

| Skill / 程序族 | 必要条件 | 在 Solar 上 |
| --- | --- | --- |
| K0 `outlier_mad` @ `z_peak>=3` | 不要求 NaN；要求服务序列满足谓词 | 0 缺失**不排除**。谓词是否成立 = **UNKNOWN**。增益 = **UNKNOWN** |
| `period_median_complete → outlier_*` | 要求原生缺口 | **排除** |
| 非恒等 impute | 无缺口时退化为 `_linear_integrity`（AGENTS §8.1） | **排除作为非恒等处理** |

不得写成「Solar 上所有 Skill 均无增益可能」。

---

## 7. 推荐核查的 Target 候选（公开官方说明，未下载）

本地**没有**仍合法的、未曝光的、带天然缺口的预测 Target：KDD 含缺失与 NOAA 2025 已开；Solar 密封但 0 缺失；NN5 / GEFCom2012 已是 development。

**主候选：Monash Wind Farms Minutely（含缺失版）。**

- 来源：Monash TSF 表（https://forecastingdata.org/ ，W Missing / W/O Missing）；Godahewa et al. 2021 表：Energy，339 条，最短 6345、最长 527040，Missing=Yes，分钟级；HF `Monash-University/monash_tsf` 的 `wind_farms_minutely`。
- 模式证据：官方同时发布含缺失 / 已填补两版，缺口是归档原生属性，不是本项目注入，也不是试出来的效用。
- 接入缺口：`data/` 下无 wind 目录；无 HEC-1 serving；下载未授权。公开最短长度撑得住 origin 3816+48；339 条够多个 40 序列 cell。逐序列缺口布局在下载前 UNKNOWN。
- 域邻近：与 Solar 同属发电，过程不同（风电 SCADA vs 光伏 10 min）。若 Solar 仍作 F2，披露即可，不是自动否决。

**次候选：London Smart Meters（含缺失版）。** Zenodo 4656072，`london_smart_meters_dataset_with_missing_values.zip`，5560 条、半小时、219.7 MB。最短 288，**撑不住**现行 HEC-1 origin 网格，要用必须先冻结「只按 header 长度门」或另一套 origin，再下载。

选择依据是任务需要（天然缺口的预测 Target）和公开元数据，不是本地试跑。

---

## 8. 需要用户另行授权的事项

1. 下载 Wind Farms（含缺失）或 London Smart Meters（含缺失）。  
2. 编写 NOAA → HEC-1 scoped serving 适配器（M0-b 才能跑；本审计不得改代码）。  
3. 在**不看 zip 内容**的前提下冻结 Solar F2 协议（origin/cell/用途）。  
4. 在 development 管线可冻结之后，一次性打开 Yahoo-41。  
5. 新书才能读 NOAA `beyond_17520`。  
6. CatsDogs D2 或其它仍密封的 UCR TEST（本轮 HEC 并不需要）。  
7. NAB 官方剩余 10 条：不建议当 fresh；若要用，先裁 freshness，而不是先下载。  
8. 计划中的 M-W / M5 / M4 / 闭环 LLM 与 fits：不在本审计权限内。

---

## 边界（本审计实际遵守）

未下载、未提交 Git、未新增 SHA、未改生产代码或 Opus 文件、未跑实验、未做缺失率重算、未做适配器预检、未打开密封或曝光未知的原始数值 / 标签 / Outcome。
