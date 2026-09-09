# 迁移准备 + 机械核对（2026-09-06）

地位：独立过夜包。0 LLM、0 Consumer fit、不改生产代码/合同/主计划/台账、不提交 Git、不新增 SHA、不写 NOAA 生产适配器、不打开 Wind 内容。不裁定方法。不等待 Opus。  
机器件：`artifacts/main_protocol/m_migration_readiness_20260906.json`。

---

## 第一页

1. **Wind 原包已就绪，内容仍未打开。** 官方 Zenodo 4654909 Version 2（with Missing Values）文件 `wind_farms_minutely_dataset_with_missing_values.zip` 已落到 `data/wind_farms_minutely/raw/`，本地 71,383,130 字节与官方 `file.size` 一致。只核了压缩包容器魔数 `PK`，未解压、未 Preview、未列成员、未跑加载器、未做数值统计。下载完成 ≠ 适配或可用性通过；未改曝光状态。
2. **NOAA 真正缺的不是 ForecastCell 类，而是当前 HEC-1 的 20/20 互斥规模。** `UnitContext` 从 KDD `p4s` 可读名单切 40 个 UID，经 `baselines._cell` 做成 A/B 对调的 20+20。fresh v1 只有 20 个不重复站；健康阵是 12+4=16。接线不能造出另外 20 条独立序列。禁止用重复站、同站不同窗口、或 2024/2025 混年凑数。这是**不能靠接线解决的几何冲突**。
3. **尚无结果可核。** `m_r0k_scope_workflow_development` 在 `artifacts/main_protocol/` 与 `docs/` 中不存在；同系列停在 `m_r0j_longitudinal_pairing`。未守候、未改 Opus 文件。
4. **明天能直接开始的一项准备：** 0-fit、fail-closed 的 NOAA 入口烟测——非 KDD 不得误走 `p4s`/`baselines._cell`；断言唯一站数 < 40、train∩eval=∅、不混年、2024 长度 8760 ≥ 3816+144+48。不写生产适配器，不在没有方法裁定前收缩 HEC-1 合同。

---

## Wind 原包（压缩包级）

| 项 | 值 |
| --- | --- |
| 记录 | https://zenodo.org/records/4654909 · DOI 10.5281/zenodo.4654909 |
| 标题 | Wind Farms Dataset (with Missing Values) |
| 版本 | 2（concept 上 `is_last=true`） |
| 文件名 | `wind_farms_minutely_dataset_with_missing_values.zip` |
| 官方体积 | 71,383,130 字节（页上 71.4 MB） |
| 许可 | `cc-by-4.0`（Zenodo API `metadata.license.id`） |
| 公开说明 | 339 座澳大利亚风电场分钟级功率，来源 AEMO / NEMWeb |
| 本地路径 | `data/wind_farms_minutely/raw/wind_farms_minutely_dataset_with_missing_values.zip` |
| 本地字节 | 71,383,130（与官方一致） |
| 传输 | 2026-09-06 01:18:59+08 开始，HTTP 200，约 25 s |
| 官方页 md5 | `b33eb66a9b63f0ae93d17a88b32cd28e`（只抄公开字段；**未**在本地计算哈希） |
| 内容 | 未打开。目录内无 `.tsf` / `.csv` 解出物 |
| 曝光 | `DOWNLOADED_CONTAINER_UNOPENED`；未擅自改成 EXPOSED |
| 未做 | 未换成已填补版或其它数据集 |

下载前本地无该目录。传输完成只证明容器在磁盘上；不解压就不能做长度门、roster 或 HEC-1 适配。

---

## NOAA 字段映射（可执行清单，不是适配器）

旧入口有两条，身份不同，不能混用：

- `DATASET_CONFIGS['noaa']` + `_fixed_roster`：消费过的 `noaa_global_hourly` / `benchmark_v0_2`，12 训 + 8 评、锚 `[240…660]`、origin 720/768。
- `_load_noaa_cell`：`benchmark_noaa_fresh_v1` 的 2024 阵，健康阵 12+4=16。`NoaaCell.roster(face)` **不按面交换**训评。

HEC-1 当前几何（合同 + `p4s` + `UnitContext`）：KDD 含缺失、40 UID 一块、A/B 各 20（末块 20+19）、训评互斥、锚 `[312,372,…,852]`、held-in origin 上限 3816、Support / delayed+48 / evaluation+144、三表面由 `scoped_serving_evaluator` 执行。`baselines._cell` **写死** `preflight.load_variant()`（KDD）。

授权元数据（只读 `manifest.json` 与各站 `record.json`，**未**打开 `values.npy` / 2025 csv / `beyond_17520`）：

- 2024：20 站，长度均为 8760，可覆盖 3816+144+48=4008。
- 2025 confirmation：16 站，`[8760,17520)`；缺 4 站：`72029953966, 72101299999, 72351399999, 72743094850`。
- `record.json` 有限计数显示 3 站近空（有限点 45 / 228 / 346），它们也不在 16 站健康阵里。

| 字段 | 现有来源 | 缺口 | 可直接实现 / 需选择 |
| --- | --- | --- | --- |
| `values` | 2024 `series/<id>/values.npy`（本包未打开） | 无工厂送进 HEC-1 `UnitContext` | 几何裁定后可接线 |
| `support_a/b` 20+20 互斥 | 20 个不重复站；健康 16；旧 NoaaCell 12+4 | **40 个 UID 不存在** | **几何冲突，需选择** |
| roster 面交换 | `ForecastCell.roster` 已交换；`NoaaCell` 不交换 | 直接复用 NoaaCell 会让两面共用 12+4 | 若采用双面互斥切分则可接线，否则需选择 |
| `observation_block` | 两套 cell 都是 `values[uid][:origin]` | 无 | 可直接实现 |
| 锚 `[312…852]` | HEC-1/KDD 配置；旧 NOAA 是 `[240…660]` | 用哪套锚是协议选择 | **需选择** |
| origin 1176–3816 与 +48/+144 | 2024 长度够；P4-Evolution 用过 8472–8712 | 不得用 2025 当额外序列或额外 origin 来凑几何 | 长度足够；origin 集合 **需选择** |
| period / 频率 | 双方 hourly，period 24 | 无 | 可直接实现 |
| 时区 | 朴素小时栅格；DATE 带 Z 则剥掉 tz | 无 IANA 名 | 命名时区 **UNKNOWN**；栅格可复用 |
| 缺失 | 物化写 NaN；评价对 truth 只计 finite；context 走 `_linear_integrity` | 近空站能否过 `seasonal_scale min_pairs=32` 未测值 | 评价器可复用；近空站是否入 roster **需选择** |
| 三表面 | `scoped_serving_evaluator` 已按 ForecastCell 形状实现 | 不要第二套评价器 | 可直接实现 |
| 特征卡 / 谓词 | `_feature_cards` / `_resolver` 吃 values+uids | 词汇若在 KDD 上拟合，迁到 NOAA 是方法问题 | 数组可接线；词汇迁移 **需选择** |
| 2024/2025 边界 | 两段都已是 development；`beyond_17520` 密封 | 混年凑独立序列禁止 | 默认不混；拼接会长序列也是新身份，**需选择** |

**几何冲突（接线解决不了）：** 当前 HEC-1 一块课要 40 个不重复序列才能同时满足互斥与 20/20 规模。20 个站 ID 不是 40 个。16 个健康站更不是。20 站在不注水的前提下最多做出旧规模的 12+8、现成的 12+4、或（若有人裁定接受半数 served 规模）一个 10+10 交换面。做不出当前 20/20 块，更做不出六块课程。是否缩小 NOAA cell、改用 Wind、或放弃 NOAA 作 HEC-1 规模的 M0-b，**不是本包裁定**。

最小拟改文件（以后若授权写适配器，仍不要新框架）：

1. `evaluation/main_protocol_p4/run_main_baselines.py` `_cell`
2. `evaluation/main_protocol_p4/run_hec1.py` `readable_uids` / `block_uids` / `UnitContext`
3. `evaluation/main_protocol_p4/run_forecast_p4_evolution_revision.py` `NoaaCell.roster` / `_load_noaa_cell`
4. `hec1_contract.py` 仅在方法裁定改课程域时才动；本包列出但不改

0-fit smoke 要检查：fits=0；UID 并集计数 = 训+评；训评交空；无重复站；无窗口冒充序列；无同站跨年两行；`record.json` 长度 ≥ 4008；身份是 fresh-v1 不是 KDD / 不是已消费 v0_2；若已建成 `ForecastCell` 则 `roster("support_a")` 必须 B 训 A 评；请求 20/20 而唯一站 < 40 则 fail closed。

---

## Opus 机械核对

目标包 `m_r0k_scope_workflow_development` **尚未出现**。规则清单（21=前5+后16、人口未被结果筛选、正负差可复算、Scope 子集父子差、fits 不计缓存/未测/消失异常为免费、开发上界不当晋升）无对象可核。发现问题时本应只通知 Opus、不改其文件；本次无问题可通知。

---

## 边界

未改生产代码、合同、阈值、主计划、台账、历史工件、Opus 文件。未提交 Git。未新增 SHA。未打开 Solar / Yahoo-41 / UCR 密封 TEST / NOAA `beyond_17520` / NOAA `values.npy` / Wind zip 内容。未试效用。
