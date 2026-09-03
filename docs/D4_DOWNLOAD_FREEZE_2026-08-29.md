# D4 下载冻结(预下载,2026-08-29 01:3x)

地位:依 sol 四裁之(4)(2026-08-29 01:2x 入典)在**下载执行前**冻结来源、版本、
用途与选择规则。本文档一经提交不再修改;下载后仅追加校验结果小节。

## 1. KDD Cup 2018(含缺失值原版)

- **来源与版本**:Monash Time Series Forecasting Repository,Zenodo record
  [4656719](https://zenodo.org/records/4656719)(Version 4, 2020-06-14),文件
  `kdd_cup_2018_dataset_with_missing_values.zip`(2.5 MB,官方
  md5 `bd6af1e03c576b6c95c5e4b58aba831f`)。内容:270 条小时序列(北京 35 站 +
  伦敦 24 站 × 多污染物),2017-01-01 至 2018-03-31,长度 9504-10920,缺失以
  原生空洞保留(HF 镜像口径 ≈503,712 缺失点)。
- **落盘**:`data/kdd2018/raw/kdd_cup_2018_dataset_with_missing_values.zip`
  (与既有去缺失变体同目录并存,文件名区分)。
- **用途(锁定)**:development 触发富池——G1 修订触发基率实测、Stage 3 pilot
  课程、Phase 2 硬档课程。**永不包装为 fresh**;不进入 Phase 3 密封池。
  族级曝光(dev 曾用 T117/T153 winsorize 线及 270 条去缺失缓存)如实披露,
  与其 development 角色无碍。
- **允许的分析**:无限制(dev 池)。

## 2. Solar 10 Minutes(= LSTF Solar-Energy 同源)

- **来源与版本**:Monash Time Series Forecasting Repository,Zenodo record
  [4656144](https://zenodo.org/records/4656144)(Version 3, 2020-06-11),文件
  `solar_10_minutes_dataset.zip`(4.6 MB,官方 md5
  `84c0de18383c911091a3cd274661b029`)。内容:137 条 10 分钟频序列,阿拉巴马
  2006 光伏电站出力,原始出处 NREL solar-power-data,与 Lai et al. (2017)
  LSTF Solar-Energy 同源。
- **落盘**:`data/solar_10_minutes/raw/solar_10_minutes_dataset.zip`。
- **用途(锁定)**:F2 密封 fresh 新族终验,仅承担 C5c(经验价值新域泛化)
  最强主张;不承担其他任何实验。
- **序列/时间窗选择规则(预冻结)**:全部 137 条序列入池,不做基于内容的挑选;
  时间窗取全年原始区间;cell 构成、origin、难度构成等一切协议参数在 Phase 3
  密封协议中冻结,且**不得以本数据内容为条件**(只能以 dev 池经验为依据)。
- **允许的分析(隔离令)**:下载后仅做完整性核验——zip md5、tsf 可解析、
  序列计数(=137)、token 计数(=52560)、缺失计数;**此外一切分析禁止**
  (不做缺陷普查、不做分布刻画、不算任何 consumer Outcome),直至 Phase 3
  密封协议开考。
- **资格披露**:旧跨序列线存在 plan-only 冻结工件
  `artifacts/functional/e2/cross_series_workflow_solar_target_plan.json`
  (`AGGREGATE_SEEN`:观察过 137×52560 结构;`numeric_values_parsed=false`;
  `program_or_consumer_outcome_computed=false`)。按两级终验裁定:
  **Outcome 未见成立,F2 资格有效**;结构曝光随论文披露。另:
  `monash_weather_daily` 太阳辐射变量曾在 w1 开发使用——气象测量族,与光伏
  发电产出族不同数据生成过程,不构成本池族级曝光。

## 3. 校验与追加纪律

下载后:`Get-FileHash -Algorithm MD5` 比对官方 md5;解包做上述完整性核验;
结果以「§4 校验结果」小节追加(唯一允许的追加),连同典中条目一并提交。

## 4. 校验结果(2026-08-29 01:4x 追加)

| 项 | KDD | Solar |
| --- | --- | --- |
| md5 | `bd6af1e03c576b6c95c5e4b58aba831f` **匹配** | `84c0de18383c911091a3cd274661b029` **匹配** |
| 序列数 | 270 ✓ | 137 ✓ |
| 长度 | 9504-10920 ✓ | 52560(全体等长)✓ |
| 缺失 | **503,712 点;270/270 条有缺失** | 0 ✓ |

KDD 下载途中 zenodo 出现 TLS 握手抽风(schannel 6 连败),改用 Python/OpenSSL
栈完成 Solar 下载;两文件 md5 均与官方一致,传输层波折不影响完整性。
Solar 自本节起进入隔离(除本节四项外零分析)。
