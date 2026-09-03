# G1 KDD Cup 2018（含缺失原版）触发富度普查

日期: 2026-08-29。地位: 纯计算; 0 LLM / 0 harness / 0 fit / 不改仓内代码 / 不 git commit。
普查脚本: `_scratch/g1_kdd_trigger_census.py`。
机器件: `artifacts/functional/e2/g1_kdd_trigger_census.json`。

**一行结论**: KDD 含缺失原版 **G1 过** 拟 ≥2/课门（每课事件代理 15.00）。4 个满员 cell 事件代理 [3, 3, 3, 3]（均值 3.00）;缺口冲突 cell 4/4,尖峰/位移冲突 cell 4/4,挤占 cell 4/4。

权威: `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md` G1（:76-79）与三档难度（:59-72）;
cell 几何 `evaluation/functional/run_e2_s2a_forecast_oracle.py`:45-52;
度量口径对齐 `artifacts/functional/e2/d4_leftover_defect_census.md`。
池角色: development 触发富池（G1 / Stage 3 / 硬档）; **不承担 fresh 主张**。

## 1. 解析与切片（可复核）

### 1.1 TSF

- 路径: `data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip` 内成员 `kdd_cup_2018_dataset_with_missing_values.tsf`
- 行格式: `series_name:city:station:air_quality_measurement:start_timestamp:v1,v2,...`
- `@attribute`: `series_name`, `city`, `station`, `air_quality_measurement`, `start_timestamp`
- 缺失记号 `?` → NaN; 空 token 亦记 NaN（本盘空 token = 0）
- 序列按**文件序**编号 `file_index=0..269`（名 `T1`…`T270`）
- 实测: 270 条（期望 270）; `?` token 503,712; 非有限点 503,712（期望 503,712）

### 1.2 cell 切分

按文件序 4×60 + spare 30。容量门 `CELL_WIDTH=60` = `N_TRAIN=40` + `N_HELDOUT=20`。

| cell | file_index | n | 角色 |
| --- | --- | --- | --- |
| kdd_missing_00 | 0–59 | 60 | forecast_cell_60 |
| kdd_missing_01 | 60–119 | 60 | forecast_cell_60 |
| kdd_missing_02 | 120–179 | 60 | forecast_cell_60 |
| kdd_missing_03 | 180–239 | 60 | forecast_cell_60 |
| kdd_missing_spare | 240–269 | 30 | spare_30 |

课程参照: 每单元一 cell、课程 5 单元。本池只有 4 个满员 cell; 第 5 单元须复用其一或动 spare（spare=30 < 60,不够一课）。每课事件代理 = 5 × 4 个满员 cell 的均值。

## 2. 计算定义（写明）

| 量 | 定义 |
| --- | --- |
| 缺口游程 | 连续非有限点; 与 `contracts.py`:59-72 同构。报告数量 / 最长 / 中位（无游程则中位=0） |
| MAD 离群率 | 有限点 `\|x−median\|/MAD` 阈 5 与 8; 缺失位不计入离群。孤立游程 := 长度 ≤3 |
| 水平位移 | 非重叠窗 168（小时周）,相邻窗中位数跳变 > 3× 全局 MAD。与 traffic 普查同一 hop,不是 hop-1 |
| 缺口可行动 | `missing_rate ≥ 2%` **或** `max_run ≥ 24` |
| 缺口偏低 | `missing_rate < 1%` **且** `max_run < 12` |
| 不对称组合 | 同一 cell 内有序对 (i,j), i≠j, i 可行动且 `value_i − value_j ≥ Δ`。Δ_缺口=5%; Δ_尖峰=5 次孤立 MAD-8; Δ_位移=1 次周跳变 |
| 可打包组合数 | `min(#不重复受益方, #不重复受害方)` — 一课里互不重叠的「一益一害」对数上界 |
| 事件代理 / cell | 缺口冲突几何命中 +1; 尖峰**或**位移冲突几何命中 +1; ≥2 族各有 ≥5 条可行动序列则 SUPPLY_DISPLACEMENT +1。上限 3 |
| 每课期望 | `5 × mean(4 个满员 cell 的事件代理)`。拟门 ≥2/课 |

这是缺陷几何代理,不是 Episode 标签。未跑 Consumer,不声称实际 CONFLICT / 负反馈次数。

不对称组合的方法含义: 合并增益可以来自高缺口 / 高尖峰序列,同时至少一条低缺口 / 低尖峰序列在同一全局处理后受害或被无必要处理 — 聚合指标可藏害。这是 S2a 注入同质场不可得冲突的反面结构前提。

## 3. 全池概要

- 序列 270/270; 有缺失 270/270; 缺失点 503,712
- 长度 min/med/max = 9504 / 10898 / 10920
- 缺失率 <1% / <2% 条数 = 6 / 14; 绝对偏低（rate<1% 且 max_run<12）= 0
- 最长缺口 ≥24h / ≥168h 条数 = 268 / 243
- 缺失率 min/med/p90/max = 0.55% / 11.72% / 34.04% / 97.67%
- 缺口游程数 med/p90/max = 271.5 / 620.5 / 1119
- 最长缺口 med/p90/max = 246 / 1981 / 10666
- 中位缺口游程 med/p90/max = 1 / 2 / 10666
- 孤立 MAD-8 游程数 med/p90/max = 12 / 54.2 / 292
- 周跳变次数 med/p90/max = 2 / 6 / 15
- 城市: Beijing=210, London=60
- 测项: PM2.5=59, PM10=54, NO2=52, CO=35, O3=35, SO2=35

按测项（异质性的自然轴,不是注入同质）:

- `CO`: n=35, missing med/p90/max=16.68%/28.57%/45.34%, isoMAD8 med=24
- `NO2`: n=52, missing med/p90/max=9.43%/24.94%/74.30%, isoMAD8 med=1
- `O3`: n=35, missing med/p90/max=10.87%/22.63%/38.91%, isoMAD8 med=5
- `PM10`: n=54, missing med/p90/max=26.90%/44.45%/97.67%, isoMAD8 med=12
- `PM2.5`: n=59, missing med/p90/max=10.62%/28.94%/77.28%, isoMAD8 med=16
- `SO2`: n=35, missing med/p90/max=10.29%/22.06%/38.82%, isoMAD8 med=75

空气质量极值常是预报相关事件,不是 occupancy 右尾。机械 MAD-8 仍报告; 标题冲突素材只认**跨序列不对称**,不把「全员都有污染高峰」算成冲突场。

## 4. Cell 汇总

| cell | n | 缺失率 med/p90/max | 缺失 IQR / range | 缺口可打包组合 | 尖峰可打包组合 | 位移可打包组合 | 事件代理 | 混合度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kdd_missing_00 | 60 | 11.79% / 26.86% / 50.54% | 9.31% / 42.48% | 28 (pairs 984; act 60/lo 0) | 49 (pairs 1406; act 49) | 42 (pairs 1483; act 42) | 3 | 6 meas / 1 city / 10 sta |
| kdd_missing_01 | 60 | 13.38% / 30.35% / 44.46% | 15.91% / 35.34% | 29 (pairs 1089; act 60/lo 0) | 48 (pairs 1382; act 48) | 45 (pairs 1503; act 45) | 3 | 6 meas / 1 city / 10 sta |
| kdd_missing_02 | 60 | 11.26% / 28.07% / 42.55% | 7.42% / 33.88% | 24 (pairs 960; act 60/lo 0) | 44 (pairs 1368; act 44) | 45 (pairs 1522; act 45) | 3 | 6 meas / 1 city / 10 sta |
| kdd_missing_03 | 60 | 11.80% / 39.94% / 57.68% | 10.66% / 57.13% | 46 (pairs 1222; act 60/lo 0) | 45 (pairs 1454; act 45) | 45 (pairs 1497; act 45) | 3 | 6 meas / 2 city / 16 sta |
| kdd_missing_spare | 30 | 6.34% / 74.47% / 97.67% | 49.77% / 96.93% | 16 (pairs 302; act 28/lo 0) | 21 (pairs 350; act 21) | 20 (pairs 371; act 20) | 3 | 3 meas / 1 city / 14 sta |

### 4.1 异质性读数

冲突场的结构前提是 cell 内自然异质。文件序把同一站点的 6 测项排在一起,因此满员 cell 默认测项混合; 北京 35 站 × 6 测项 = 210 条在前,伦敦在后,故 cell 00–02 为北京单城,cell 03 起跨城。

| cell | miss min | med | p90 | max | IQR | CV | 测项 H (bit) | 测项 max share | 城市 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kdd_missing_00 | 8.07% | 11.79% | 26.86% | 50.54% | 9.31% | 0.549 | 2.585 | 16.67% | Beijing:60 |
| kdd_missing_01 | 9.12% | 13.38% | 30.35% | 44.46% | 15.91% | 0.534 | 2.585 | 16.67% | Beijing:60 |
| kdd_missing_02 | 8.67% | 11.26% | 28.07% | 42.55% | 7.42% | 0.520 | 2.585 | 16.67% | Beijing:60 |
| kdd_missing_03 | 0.55% | 11.80% | 39.94% | 57.68% | 10.66% | 0.878 | 2.391 | 26.67% | Beijing:30, London:30 |
| kdd_missing_spare | 0.74% | 6.34% | 74.47% | 97.67% | 49.77% | 1.242 | 1.552 | 43.33% | London:30 |

判读: 若缺失率 IQR / range 接近 0 且测项单一,则接近 S2a 注入同质场（冲突不可得）。本盘满员 cell 缺失 CV 0.52–0.88、range 34–57pp,测项熵在北京三 cell 达 log2(6)=2.585 bit（六测项均分）,是冲突场的结构前提。

**有无 vs 强度**: 全池绝对偏低序列 = 0（几乎没有「干净对照」）。北京三 cell 最低缺失率仍约 8–9%,冲突几何来自**强度与游程结构**（中位游程 1h 的孤立小时洞 vs 最长缺口中位 246h、个别逾万点近死序列）,不是「有的有洞、有的没有」。cell 03 / spare 才出现接近 0.5–0.7% 的伦敦低缺失条,以及 spare 上 97.67% 近空序列。这与 S2a「人人注入 80 个孤立 NaN」仍相反。

## 5. 每课触发素材量级

参照: 每单元一 cell,课程 5 单元。下表是 4 个满员 cell 的几何素材; spare 不计入每课均值。

| cell | 缺口可打包组合 | 尖峰+位移可打包组合 | 缺口事件 | 尖峰/位移事件 | 挤占事件 | 事件代理合计 |
| --- | --- | --- | --- | --- | --- | --- |
| kdd_missing_00 | 28 | 91 | 1 | 1 | 1 | 3 |
| kdd_missing_01 | 29 | 93 | 1 | 1 | 1 | 3 |
| kdd_missing_02 | 24 | 89 | 1 | 1 | 1 | 3 |
| kdd_missing_03 | 46 | 90 | 1 | 1 | 1 | 3 |

- 4 cell 事件代理: 3, 3, 3, 3
- 满员 cell 均值: 3.00
- **每课期望（5 × 均值）: 15.00**（三类事件可同单元叠加的上限）
- 保守敏感度: 只计缺口 = 5.00/课; 缺口+尖峰/位移 = 10.00/课
- 缺口可打包组合 4 cell: [28, 29, 24, 46]
- 尖峰可打包组合 4 cell: [49, 48, 44, 45]
- 位移可打包组合 4 cell: [42, 45, 45, 45]

可打包组合是「一益一害」对数的互不重叠上界,量级上远大于事件代理。事件代理才是与门槛 ≥2/课对齐的单位（一单元一次程序应用最多打出有限次 CONFLICT / 负反馈 / 挤占,不会把所有 pair 都变成独立 Episode）。15/课把缺口、尖峰/位移、挤占各计 1,是事件**类型**上限; 过门不依赖这层叠加 — 只计缺口仍 5.00/课。

## 6. G1 判读

门: 冻池前预估每课修订触发（冲突 / 负反馈 / 挤占）; 拟阈值 ≥2/课（协议未另冻）。

**结论: 过。** `meets_proposed_g1_ge_2_per_lesson = true`。

判据:

- 事件代理每课 = 5 × mean([3, 3, 3, 3]) = 15.00,≥ 拟门 2。保守敏感度（只计缺口冲突、不计尖峰/位移/挤占）= 5.00,仍过门。
- 缺口不对称: 4 个满员 cell 中 4/4 具备冲突几何;可打包组合 [28, 29, 24, 46]（Δ缺失率 ≥ 5% 且受益方缺口可行动）。
- 尖峰/位移不对称: 4 cell 中 4/4 具备冲突几何;尖峰可打包 [49, 48, 44, 45],位移可打包 [42, 45, 45, 45]。
- 供给挤占: 4 cell 中 4/4 同时有 ≥2 族、每族 ≥5 条可行动序列。
- 与 S2a 注入同质场对照: 本池 270/270 有缺失,但标题触发认的是 cell 内跨序列差值,不是「人人有洞」本身。人人有洞且差值不足 仍会判零冲突几何。
- 未跑 Consumer; 负反馈符号不确定。事件代理只计冲突几何与挤占,不计「全局修污染高峰可能系统变差」的单侧负反馈。

与 traffic leftover 普查对照（`d4_leftover_defect_census.md` §7）: traffic 6 cell 无 NaN、周跳变每 cell 0–3 条、MAD-8 被占有率混淆,修订触发期望 **≈0/课**, `meets_proposed_g1_ge_2_per_lesson = false`。KDD 含缺失原版每课事件代理 **15.00**,量级从「零触发」转到「每课可数的冲突/挤占事件代理,相对 traffic 的 0/课至少高一个数量级」。

可承担难度档（几何代理,非正式预注册）: 可承担中/难自然缺陷发现与冲突场（development）; 不得包装为 fresh。易档不是本池主角色。

## 7. 长度不齐与协议 origins

协议: `ORIGIN_HELDIN=1104`, `ORIGIN_HELDOUT=1800`, 同构最短 `ORIGIN_HELDOUT+48=1848`。

| 项 | 值 |
| --- | --- |
| 长度 min / max | 9504 / 10920 |
| 全部 ≥ 1848 | true |
| n < 1848 | 0 |
| n < ORIGIN_HELDIN+48=1152 | 0 |
| 北京长度集合 | 10898 |
| 伦敦长度集合 | 9504, 10920 |
| 兼容性 | 全部序列 ≥1848,与 origins 1104/1800 按序列独立切窗无碍。 |

长度不齐是城际覆盖窗不同,不是缺测到 1848 以下。按序列独立切 origin 时与 1104/1800 兼容; 若未来把不等长序列当对齐面板,须另冻对齐规则（本普查不假设对齐）。

## 8. 限制

- 零 Consumer / 零 Episode; 修订触发是缺陷几何代理,不是 CONFLICT/NEG/DISPLACEMENT 标签。
- 现役 S2a forecast 菜单是 identity/outlier_iqr/outlier_mad/hampel/winsorize,不含插补。缺口素材要变成真实 Episode,课程菜单须有缺口修复动作; 本普查不假设菜单已扩。
- 空气质量 MAD-8 高峰可能是真实污染事件。本普查不把「全员有高峰」算冲突; 只认跨序列不对称。对高峰做 winsorize 的系统负反馈标不确定,未计入 ≥2/课。
- 水平位移用 hop=168 相邻窗,与 traffic 普查可比; 不是 hop-1 滚动计数,也不是 minipipe 40–96 点往返 excursion。
- 第 5 单元没有第 5 个满员 cell; 每课=5×均值是参照课程长度的外推,复用 cell 会降低独立触发次数（未另打折）。
- 文件序切 cell 会把同站多测项捆在同一 cell,这是自然测项混合,不是随机重排。重排会改组合计数,本普查不重排。
- 空 token 按 NaN 计; 本盘空 token=0,与官方 503,712 对齐。
- 北京三 cell 几乎没有绝对偏低序列; 缺口冲突是 8–50% 的强度差与游程结构差,不是有无差。过门不依赖把 15/课当成独立 Episode 数 — 只计缺口仍 5/课。
- 不 git commit; 不改仓内代码。

## 9. 引证索引

| 对象 | 位置 |
| --- | --- |
| G1 | `docs/MAIN_EXPERIMENT_DESIGN_SKELETON_2026-08-28.md`:76-79 |
| 三档难度 | 同文件 :59-72 |
| 触发富/fresh 角色分离 | 同文件 v1.1 :11-12 |
| 容量门 / origins | `evaluation/functional/run_e2_s2a_forecast_oracle.py`:45-52 |
| traffic 普查口径 | `artifacts/functional/e2/d4_leftover_defect_census.md`; `_scratch/d4_leftover_defect_census.py` |
| NaN 游程 | `evaluation/minipipe/contracts.py`:59-72 |
| 下载冻结 | `docs/D4_DOWNLOAD_FREEZE_2026-08-29.md` §1 / §4 |
