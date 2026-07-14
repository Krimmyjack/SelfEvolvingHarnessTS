# Benchmark 数据设计与使用协议

## 1. 目的

本 Benchmark 用于评估：面对不同 Domain、Dataset、实体和数据缺陷时，Harness 能否选择或生成合适的数据准备程序，使同一套下游预测任务获得稳定增益，同时避免有害修改，并在没有可靠收益时安全 abstain。

Benchmark、Method 和 Experiment 必须分离：本目录定义数据、身份、划分和使用纪律；Harness 如何更新属于 Method；具体更新顺序和方法对比属于预注册 Experiment。

## 2. 三层数据单位

- **Domain**：研究领域，例如 Traffic、Energy、Epidemiology、Cash Demand、Weather。
- **Dataset**：独立采集或发布的数据集合，例如 METR-LA、Monash Traffic、UCI Electricity。
- **Series/Entity**：一个可独立划分的实体，例如传感器、客户电表、负荷区域或国家。

“同一 Domain 多 Dataset”用于研究跨 Dataset 迁移；“同一 Dataset 多 Series”用于研究未见实体泛化。两者不能混为同一种证据。

## 3. 当前数据 roster

| 用途/Domain | Dataset | Series | 当前定位 |
| --- | --- | ---: | --- |
| Init，多领域历史语料 | Legacy core | 80 | 只用于构造初始 Harness |
| Init，历史暴露扩展 | probe-consumed Monash Traffic | 56 | 只用于构造初始 Harness |
| Traffic | METR-LA | 207 | Headline；20 个空间 block 为外层划分原子 |
| Traffic | Monash Traffic Hourly | 862 | Headline；缺少坐标，跨 split 邻近传感器泄漏风险需披露 |
| Energy load | UCI Electricity | 370 | Headline |
| Energy load | GEFCom2012 Load | 20 | Supplementary；不能单独承载强结论 |
| Epidemiology | Monash COVID Deaths | 246 | Headline；需同时披露原尺度 MAE 和 sMASE 分母诊断 |
| Cash demand | Monash NN5 Daily | 91 | Headline |
| Unseen Weather | NOAA Global Hourly | 40 注册、38 合格 | U；完全未见 Domain |

Registry 共 1919 条，1867 条通过准入。冻结的外层角色计数为 Support-A 578、Support-B 344、Dev-Query 299、Final-Query 608、U 38。

## 4. Init Harness 与评估数据必须分开

初始 Harness `H0` 只由冻结的 Init Corpus 构造：

```text
80 legacy_core + 56 probe_consumed_extension = 136 series
```

Init Corpus 可以反复使用，因为它已经暴露；它可用于构造初始 operator memory、seed policy、代码模板和先验规则，但不能支持“未见数据泛化”结论。Weather U、Final-Query 和 Fresh Support-A 不得参与 `H0` 构造。

当前 v0.1 的实现把这 136 条暴露数据也编码为 `role=support_a`，所以 578 并不等于 578 条 fresh 数据：

```text
Support-A manifest total 578
- Init Harness exposed 136
= Fresh Support-A 442
```

v0.2 应在接口和报告中把 `init_harness=136` 与 `support_a_fresh=442` 分开，并从 Fresh Support-A 的 discovery/validation 统计中排除 Init UID。当前 `432/146` 是对全部 578 条 Support-A 的划分，不能解释为纯 fresh 计数。

## 5. 外层角色的含义

| 角色 | 用途 | 访问纪律 |
| --- | --- | --- |
| Init Corpus | 构造 `H0` | 可重复；不进入评估分母 |
| Support-A Discovery | 搜索、生成、拟合和更新 Harness | 可重复 |
| Support-A Validation | 判断 Discovery 找到的更新是否可晋级 | 可重复；不能参与被评候选的搜索 |
| Dev-Query | 新实体上的开发诊断和全链路 dry-run | 可重复，因此不是无偏测试 |
| Support-B | 方法冻结后的一次性 confirmation | 全局一次性；消费后不得调参重试 |
| Final-Query | 冻结 roster 的正式评价 | 单次 evaluation campaign |
| U | 完全未见 Weather Domain 的迁移、安全性和 abstention | 最后访问；严格计账 |

推荐生命周期：

```text
Init Corpus -> H0
-> Support-A Discovery/Validation 上顺序学习
-> Dev-Query 重复诊断
-> 冻结方法、代码 SHA、预算和 seeds
-> Support-B 一次确认
-> 单次 Final campaign
-> Weather U
```

## 6. 三种 held-out 不能混淆

1. **Held-out future**：同一条 series 的未来时间段。
2. **Held-out entity**：同一 Dataset 中未参与 Harness 开发的 sensor/meter/zone。
3. **Held-out Dataset/Domain**：整个 Dataset 或 Domain 未参与开发。

当前 Benchmark 已实现前两种；Weather U 提供 held-out Domain。Traffic 和 Energy 虽各有两个 Dataset，但 v0.1 仍把两个 Dataset 的不同实体分别放入 Support/Dev/Final，因此不是严格的“一个 Dataset 训练、另一个 Dataset 整体测试”。若研究 same-domain cross-dataset transfer，实验 manifest 还需冻结 `dataset_role=source/target`。

## 7. 每条 series 内部的时间划分

外层角色决定“哪个实体属于哪个池”；内层 chronological split 决定“同一实体的哪些时间点用于训练或评价”：

```text
|----------- inner-train -----------|-- inner-validation --|-- held-out future --|
```

- `inner-train`：候选 PrepPolicy 处理后训练 fresh downstream model。
- `inner-validation`：early stopping、Support-A feedback 和允许的模型选择。
- `held-out future`：只用于计算最终 utility，方法不可提前读取真值。

METR-LA 的外层单位是空间 block，而不是单个传感器；同一 block 的传感器必须进入同一角色。

## 8. 下游模型评价口径

主 judge 必须按 Dataset/Config 独立训练，不能把 COVID、Traffic、Electricity、NN5 等窗口混入同一个模型：

```text
dataset/config x program x scenario x dose x corruption replicate x model seed
-> 该 dataset/config 的全部 eligible series 共同训练 fresh model
-> 在各 series 的 held-out future 上评价
```

所有方法共享相同 series eligibility、窗口、benchmark-owned normalization、canonical ingestion、model seeds 和 CRN corruption。Adam-DLinear 使用 3 个 model seeds；它们是同一 UID 的重复测量，必须先平均，不能作为独立 bootstrap 样本。series-equal loss 使用确定性权重 `1/W_s`，不能用 weighted sampler 近似。

当前 v0.1 evaluator 实际按整个 role 做 cross-dataset pooled training；其 Dev 饱和结论和 timeout 只能作为诊断。修正为 per-dataset/config judge 后应以新版本重新生成 Dev 报告和 timeout，不得覆盖已冻结的 v0.1。

## 9. Harness 是顺序学习，下游 judge 不是跨 Domain 联合训练

Harness 可以按照预注册 Domain 顺序逐步更新，例如：

```text
H0 -> Energy episode -> H1 -> Traffic episode -> H2
-> Cash/Epidemiology episode -> H3 -> freeze
```

具体 Domain 顺序必须在实验前冻结，因为顺序会影响 continual-learning 结果。每个 Dataset 的 downstream judge 仍独立 fresh 训练；跨 Domain joint model 只能作为额外 global baseline，不能替代主 judge。

## 10. Natural 与 Controlled-Corruption 双轨

- **Natural Track**：原始真实数据，`dose=0`；检验自然收益、伤害和 abstention。
- **Controlled Track**：预注册 missing、block missing、spike、Gaussian noise、level shift 和 timestamp disorder；检验 Harness 是否能识别并修复已知缺陷。

Corruption 只作用于模型可见的训练历史或 context，held-out future target 保持干净。类型、dose、replicate 和 seed 必须在方法结果前冻结；不能因为方法没有收益就继续加噪声。所有方法读取同一份 content-hash 键控的 corruption realization。

## 11. GEFCom2014 决策

GEFCom2014 不是 v0.2 阻塞项。论文级 v1 可把 **Load Track** 作为 Energy supplementary dataset；Wind、Solar 和 Price 必须分别归入 Renewable Generation 或 Electricity Market Price Track，不能与 electricity load 混为一个 Dataset。若目标是增强 Energy 的独立实体和跨 Dataset 证据，OPSD/ENTSO-E 的优先级高于 GEFCom2014。

## 12. 目录结构与权威文件

```text
data/
  BENCHMARK_DATA.md                 # 本文：项目叙述和数据使用协议
  _artifacts/                       # 历史 Init 数据资产
  benchmark_v0/
    acquisition_manifest.json       # 原始资产路径与 SHA256
    incoming/                       # 人工下载的原始压缩包
    raw/                            # 不可变原始源
    probe_cache/                    # 无 utility 的准入探针缓存
  benchmark_v0_1/
    clean_base/                     # v0.1 物化序列；按稳定 slot 存储
results/Benchmark_v0_1/
  series_registry.jsonl
  split_manifest.json
  dataset_manifest.json
  support_a_subsplit.json
  corruption_grid.json
  benchmark_manifest_v0.yaml
```

原始和 clean-base 数据被 `.gitignore` 排除，不能依赖 Git worktree 传播；合并分支后仍必须显式迁移这些目录，并使用 acquisition manifest、registry 和冻结 SHA 做完整性验证。Final-Query 数据可以物化，但任何 utility 访问必须经过 ledger 和单次 campaign。
