# TS Data-Readiness Forecast Benchmark v3 — 规范性补充

> 状态：已批准。本文与 `Benchmark_v0_Forecast_Design.md` v2 合并解释为 Benchmark v3。
> 本文覆盖 v2 中冲突的 wording；未涉及条款保持 v2 原文。实现接口的完整展开见
> `SelfEvolvingHarnessTS/docs/superpowers/specs/2026-07-13-benchmark-data-metrics-pipeline-design.md`。

## 1. 数据补充

- v0/v1 候选全部下载或预留人工导入：Monash、METR-LA、UCI ELD、ENTSO-E、
  GEFCom 2012/2014；新增 NOAA NCEI Global Hourly/ISD temperature 作为 traffic 升为
  seen domain 后的整封 weather U。
- 当前 legacy 实物为 83 条，而非 80 条：四个主 config 各 20，另有
  `us_births/saugeenday/sunspot` 各 1。83 条全部 `confirmed_exposed`，只入 Support-A/Dev。
- raw/clean_base/derived/incoming 四层置于
  `SelfEvolvingHarnessTS/data/benchmark_v0/`；来源 revision 与 raw SHA256 必须登记。

## 2. Corruption RNG

禁止列表索引 seed。唯一 RNG 键为 canonical SHA256 编码：

```text
(benchmark_version, clean_content_sha, scenario, dose, replicate_idx)
```

headline `replicate_idx=(0,1)`。重排或取子集不得改变既有 uid 的结果。派生规则与每个
realized seed/digest 入 manifest；所有方法读取同一物化 corruption，以 CRN 配对。

## 3. Method API、normalization 与 ingestion

- `adapt` 的 `feedback_api` 仅开放 Support-A inner-val + closed-form 通道并逐次记账。
- v0 normalization 为 benchmark-owned：从 method 调用前的退化 inner-train 有限观测冻结
  mean/std，所有方法/模型/context 共用。`znorm`、`minmax_norm` 及所有
  `changes_target_space=True` 算子不在 action surface。放开 adaptive normalization 必须 bump
  benchmark version。
- canonical ingestion 只线性插补 NaN，首尾最近值钳制；±inf、全 NaN、变长非法。
  记录 `ingestion_fill_rate`，>0.01 标描述性依赖旗。
- `train_effect` 的固定 raw-degraded context 先 canonical ingestion 并缓存，所有方法 bit 级共用。
- Raw 的正式标签为 **No-op + canonical ingestion**；缺失处理收益是相对 canonical linear fill
  的增量。

## 4. Downstream trainer identity

闭式判官保持 `series_weight="equal"`。Benchmark Adam-DLinear 与 P6 的 pooled-window Adam
有意不同，采用加权 loss、禁止 weighted sampler：

```text
L = Σ_s Σ_(w∈s) (1/W_s)ℓ_sw / Σ_s Σ_(w∈s) (1/W_s)
L_batch = N/(bS) · Σ_(j∈batch) (1/W_s(j))ℓ_j
```

窗口均匀无放回 shuffle，末尾短 batch 同式。model seeds 固定 `(0,1,2)`；所有方法共享
初始化和 batch-order CRN。batch size、epochs、Adam 参数、设备确定性模式与公式入 manifest。

## 5. 指标与完整折叠顺序

主指标保持 sMASE。完整折叠为：

```text
同 uid × scenario × corruption replicate：先平均 3 model seeds
→ 同 uid × scenario：平均 2 corruption replicates
→ 同 uid：冻结 dose/scenario 等权平均
→ cell 内 series 等权
→ regime 层 dataset 宏平均
```

每个 dose/scenario 单独披露。bootstrap API 只接受每 uid 恰一行 paired gain，拒绝未折叠的
seed/replicate/dose 行。harm `δ=0.05`；bootstrap `B=2000`，master seed=`20260713`，
comparison/cell 子 seed 由 canonical hash 派生。所有方法共享 model seeds 与 CRN。

## 6. Baseline 与 oracle

- Raw、best-fixed、H_ref 走公开 Method API 与同一 Pipeline。
- `oracle_transfer` 是 runner 特权诊断：Support-A 按 cell 选程序，冻结 mapping 后在 Query 评。
- `oracle_insample` 是 runner 特权诊断：Query 上选且评，只是 winner's-curse 膨胀上包络。
- oracle 不是 Method，不声称遵守 method visibility，也不进 headline 排名。

## 7. Support 与单次 Final evaluation campaign

- Support-A 可反复开发；进入后续阶段前须完整契约 dry-run，并记录 artifact SHA。
- Support-B 是 code/config 冻结后的一次 confirmation。失败后不得改 SHA 再进入同一 campaign。
- Final-Query 不按“每方法首次访问”开箱，而按一次 campaign：

```text
freeze benchmark version
→ freeze roster + method/code SHA + runner SHA + budgets + seeds + order
→ verify Support-A dry-run + Support-B confirmation SHA
→ WAL durable commit 单次 unseal（先于读取任何 Final 值）
→ 每方法访问前 durable method_access
→ 全 roster 执行并记录 method_result
→ campaign_close，永久关闭
```

Baseline 全链路验证只能用 Dev-Query。roster 冻结后新增方法只能进入新 Final split/新 benchmark
version，或留在可重复 Dev-Query。

## 8. Ledger 与失败分类

Ledger 改造 P6 的锁、append-only WAL、fsync、event hash chain、replay 校验。事件至少含：

```text
campaign_freeze / unseal / method_access / method_result / campaign_close
```

每次访问登记 `campaign_id/method_id/code_sha/run_id` 与 benchmark/split/materialization SHA。

Final 上的方法异常——抛错、变长、±inf、全 NaN、禁用 transform、冻结 timeout——是该方法
terminal `invalid/failed_timeout`；Query 已消费，不得改码重跑。仅断电、进程终止、磁盘 I/O、
硬件/evaluator crash 等方法无关事故可 resume，且必须完全相同的 campaign/run_id、method 与
runner code SHA、输入/materialization SHA、checkpoint bytes。修 evaluator 代码会使 campaign
作废，而不是授权重试。

## 9. Freeze 补充清单

manifest 还必须冻结：corruption scenarios/doses/replicates/派生 seeds；normalization 与
ingestion；model seeds、batch order、series-equal loss 公式与 optimizer 参数；harm `δ`；
bootstrap seed/B；以及同硬件同路径 Dev 实测 `2×p95` 后写成数值的 prepare/trainer timeout。

## 10. 执行顺序

```text
probe
→ benchmark freeze
→ Dev 全链路 dry-run
→ 数据物化与 Final 封存
→ 方法/实验 prereg + Support-B confirmation
→ 单次 Final evaluation campaign
```
