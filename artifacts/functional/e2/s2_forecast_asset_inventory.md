# S2 forecast 资产盘点(设计前置,只读)

本书 = Stage 2 设计前置的预测(forecast)任务线只读资产盘点。0 LLM / 0 fit / 0 下载;未跑全仓 pytest;未碰 `methods/` `contracts/` `runtime/` `operators/` 与密封件。权威:台账 v1 冻结条(`STAGE_REPORT` :1216-1220)、sol 裁 B(:1222-1226)、常备纪律(:1254-1256)、`docs/CLS_LINE_FINAL_REPORT_2026-08-28.md`:43-46。机器件:`s2_forecast_asset_inventory.json`。

## 七项一行结论

| # | 项 | 一行结论 |
| --- | --- | --- |
| 1 | runner / harness 入口 | 现役键 = `forecast\|ridge\|sMASE`(TEH 回落)与 `forecast\|pooled_ridge_a1\|sMASE`(T5 铸造);S1 四臂是分类特化,不能靠翻 `task_kind` 跑 forecast。 |
| 2 | 数据与反馈容量 | 现役 cell(12+8 / 12+4)对「对半后每面 ≥20 行」**不达标**;仅重切 Monash/UCI/METR(n≥40)才过。 |
| 3 | A5>A3 +31.7% | Frep 回放 84 vs 123(`t6_45_frep_a5a3_replay.md`:95);v1 冻结前 Guidance 卡年代;不能当阶梯/修订正账重引。 |
| 4 | Episode / store / 卡 | Frep 工件里有 Source `fresh_batch_guidance_*` 与 Target-local `fast_winner_forecast_*`;仓内无已提交的可检索 forecast Episode 库。 |
| 5 | 缺陷注入族 | forecast = `impulsive_outlier` / `gap` / T0 cycle spike-burst / minipipe / benchmark corruption;**无**分类 `impulse_v2`。 |
| 6 | Consumer | 生命周期就绪只有 ridge×sMASE(pooled + per_channel);DLinear/kNN 是探针,未接入 G1/S1/T5。 |
| 7 | AD 守卫 | `#31` shared 卡是预测域且无 `task_kind` 轴;AD v3 已归档;分类卡惰性守卫缺 forecast 宿主。 |

## 1. runner / harness 入口

**判定**:recipe / T5 / Frep / G1-e1/t1 是 forecast 现役仪器;现代 S1 课程四臂是**分类特化**,不是通用 `task_kind` 分发器。这是 S2 成本估算的承重项。

**键谱系**

- 铸造规则:`task_type|downstream_model_class|metric.name`(`experience_memory.py`:76)。
- 回落:`forecast|ridge|sMASE`(`experience_memory.py`:77;`t1.py`:88)。
- T5 现役铸造:`forecast|pooled_ridge_a1|sMASE`(`run_e2_t5_lifecycle_dual_consumer.py`:262-267;`t5_lifecycle_v1.md`:25,35)。
- 历史方言:`forecast|ridge_smase`(`experience_memory.py`:539/566/593)——现役检索永不铸造,旧 load 才会碰到。
- 工厂第三方言:`forecast_task_spec_v1` 默认 `dlinear_shared` + `nRMSE`(`contracts/task.py`:329-346),不是现役 ridge/sMASE 生命周期。

**现役 forecast runner**(evaluation/functional)

- recipe / Consumer 条件化:`run_batch_composition_headroom.py`(v6 `_evaluate`,ridge + sMASE;traffic 名册 12+8 于 :163-164)。
- Fresh / Frep:`run_e2_fresh_confirmation.py`,`run_e2_t6_45_frep_a5a3_replay.py`。
- T5 生命周期:`run_e2_t5_lifecycle_dual_consumer.py` `f_task_spec()` :262-267。
- T1/T1b:`run_e2_t1_flip_control.py`,`run_e2_t1b_training_side_flip.py`。
- 兼容:`run_e2_t6_forecasting_compat_0b.py`。
- TEH:`task_episode_harness/{t1,e1,runner}.py`;G1 `task_episode_harness/agentic/runner.py`。
- 历史 v1:`run_v1_fastpath.py` 等。

**现代 agentic / S1 对 `task_kind=forecast`**

- S1 四臂:`TASK_KIND = classification`,`CONSUMER_ID = ridge-raw-plus-difference-v1`,`METRIC = accuracy`(`run_e2_s1_curriculum_four_arms.py`:4,275-277)。`_scope_v1_admits` 于 :1683 拒绝任何非 classification 的 `scope.task_kind`。S1a oracle 同样三常量写死(`run_e2_s1a_curriculum_oracle_audit.py`:88-90)。修订特征提取写死 `task_kind="classification"`(`skill_revision.py`:78)。导入的是分类 shared harness / UCR / `impulse_v2`。
- G1 agentic 是 **forecast 特化** 而非分发器:目标文案写死 "downstream forecast Consumer",度量 sMASE,窗口 240 点(`agentic/runner.py`:351-372)。`PUBLIC_CONTEXT_TASK_KIND = "forecast"`(`public_context.py`:31)。默认 Source 卡 `source_investigation_v1` 适用 `task_kind==forecast`(`source_skill.py`:49-60)。
- **S2 含义**:不能在 S1 四臂/修订/五轴课程上把 `task_kind` 拨成 forecast。必须付「forecast 适配器或新 runner」的成本。G1 今天能跑 forecast,但不承载 S1 课程。

## 2. 数据资产与反馈容量

**新门**(分类线终态,S2 选靶硬门):Support/delayed **对半后每面 ≥20 行**,材料线 ≤0.05(`CLS_LINE_FINAL_REPORT_2026-08-28.md`:43)。

**单位说明**:分类「行」= TRAIN 实例。本盘点把 forecast 严格类比成 **序列作行**(要 n_train≥40)。现役仪器实际读的是 origin 窗(`n_eval × n_delayed_origins`),另列。预测材料/伤害线仍冻在 0.005 / −0.005,不是 `1/n`。

注册表:`artifacts/frozen/benchmark_v02/series_registry.jsonl`,1919 行(只读解析)。

| 资产 | n × 长度 | 暴露 | 序列作行(重切) | 现役 cell |
| --- | ---: | --- | --- | --- |
| `monash:traffic_hourly` | 862 × 1024 | virgin 806 / probe 56 | PASS | 12+8 FAIL(`run_batch…`:163-164) |
| `uci_electricity_load_diagrams` | 370 × 1024 | virgin | PASS | 12+8 FAIL;族已 outcome 暴露(`g3_sourcing` / TSL electricity.csv) |
| `metr_la` | 207 × 1024 | virgin | PASS | 未成现役 cell |
| `monash:nn5_daily` | 91 × 714–791 | virgin | PASS(半≈45) | 若 12+8 则 FAIL |
| `monash:covid_deaths` | 246 × 212 | virgin | n PASS | 长度不抵 240 / 3072 协议 |
| `noaa_global_hourly`(v0.2) | 40 × 1024 | virgin | 贴线(20/20) | 12+8/12+4 FAIL |
| `gefcom2012_load` | 20 × 1024 | virgin | FAIL(半=10) | — |
| `legacy_monash:fred_md` | 20 × 728 | confirmed_exposed | FAIL | — |
| `legacy_monash:nn5_daily` | 20 × 791 | exposed | FAIL | — |
| `legacy_monash:tourism_monthly` | 20 × 187–330 | exposed | FAIL | — |
| NOAA fresh v1 | 20 × **8760** | 2024+2025 EXPOSED | FAIL(半=10) | Frep 12+4 / 16 序列 FAIL |
| KDD T233 | 12+8 | 已暴露 | FAIL | `g1.py`:1881-1887 |

**现役协议 cell 的 origin 窗读数**(非 CLS「行」类比):electricity/T233 = 8×3=24(贴线 PASS);traffic = 8×1=8 FAIL;NOAA Frep = 4×3=12 FAIL(`t6_45_frep_a5a3_replay.md`:23-29)。

NOAA 2025 确认目录在盘,本盘点未开值;AGENTS 记 2024/2025 均已 EXPOSED,`beyond_17520` 仍封。

## 3. A5>A3 +31.7% 正账

**读数工件**:`artifacts/functional/e2/t6_45_frep_a5a3_replay.md`:3,17,95。链判定 `CHAIN_REPRODUCED`;原仪器联合条款未过,原文写 `FRESH_A5_FAILS`。主格 pooled **84 vs 123(−31.7%)**,相对原 `FRESH_A5_DELIVERS` 的 69 vs 123(−43.9%)。

**年代**:v1 冻结前。机制 = recipe 编译的 Guidance 卡(`fresh_batch_guidance_pooled_v1` / `per_channel_v1`) + Target-local `fast_winner_forecast_*`,**不是** 阶梯 v2 / 五轴 Scope / 修订环。冻结权威:`STAGE_REPORT`:1216-1218。

**Frep-b**:`t6_45_frep_b_symmetric_deploy.md` 修 F1/F2 后 held-out pooled A5 +0.059 vs A3 −0.217,差 +0.276。开发块、已暴露 NOAA 2024,不得引为 `FRESH_A5_DELIVERS` 复证。

**v1 冻结下重挣差距**

1. 旧卡无五轴 Scope / 供给档 / 修订字段。
2. S2 只读 Skill 层,不能改 Skill/Memory 结构去「升级」旧卡。
3. NOAA 2024/2025 已 EXPOSED,不是新鲜密封考。
4. 12+4 过不了新 ≥20 行门。
5. S1 四臂不能宿主 forecast(第 1 项)。
6. per_channel 是转移边界 / TIE(`t6_45_frep_a5a3_replay.md`:7)。
7. n=1 配对开发抽,只保方向。
8. 键方言(`ridge|sMASE` / `pooled_ridge_a1|sMASE` / `ridge_smase`)未统一则 Memory 复用会空窗。

## 4. 预测侧 Episode / store / 卡

**有(工件 / 跑次 store,非仓内现役可检索库)**

| 形态 | id | 出处 |
| --- | --- | --- |
| Source Guidance | `fresh_batch_guidance_pooled_v1`, `fresh_batch_guidance_per_channel_v1` | `t6_45_frep_b_symmetric_deploy.json`:55-66 |
| Target-local | `fast_winner_forecast_pooled_ridge_a1_smase_e1v2_outlier_iqr`, `fast_winner_forecast_per_channel_ridge_a1_smase_e1v2_repair_level_shift` | 同上 :55,101 |
| 默认 Source 整合器 | `source_investigation_v1`(适用 `task_kind==forecast`) | `source_skill.py`:49-60 |
| T5 Episode 键 | `forecast\|pooled_ridge_a1\|sMASE` | `t5_lifecycle_v1.md`:25,35 |
| TEH 历史键 | `forecast\|ridge\|sMASE` | `t1.py`:88;`w1_task_episode_harness_report.json` |
| 不可检索方言 | `forecast\|ridge_smase` | `experience_memory.py`:539+ |
| 共享候选(非 live store) | `shared_outlier_repair_with_per_series_guard_v1` | `shared_capability_candidate_v2.json` |

**无**:已提交、可供 S2 直接检索的 forecast Episode 银行。Frep/T5 store 是跑次工件。

## 5. 缺陷注入族

forecast **没有** 分类 `impulse_v2` / `burst_cls2`(`run_e2_t6_cls_op_shared_harness.py`:396-397 属分类线)。

| family | 实现 | 模板常量 | 长度门 |
| --- | --- | --- | --- |
| `impulsive_outlier` | `task_episode_harness/injection.py`:38-89 | amp 8.0, count 40, seed 7; scale `nanstd[120:900]` | 标签时间戳池 ≥ count;锚 312–852 step 60, H=48 |
| `gap` | 同文件 :91-128 | count 80, seed 11, 写 NaN | 同池 ≥ count |
| T0/T1 cycle spike/burst | `run_e2_t0_ad_instrument.py`:78-89 | `EVENT_DIVISOR=112`, spacing 50, boundary 25, `SIGMA_PREFIX=168`; 1 点 spike ×6σ/10σ + 3 点 burst | NOAA 区协议;T1 只注 12 条 train |
| minipipe | `evaluation/minipipe/corpus/injections.py`:8-10,45-74 | missing / impulsive_outlier / level_shift / period_change | 固定 L=240, ctx=192, fut=48 |
| benchmark v0.2 | `evaluation/benchmark_v02/corruption.py` | block/scattered / spike / level_shift / gaussian / local_permutation | dose × 序列长 |

## 6. Consumer

forecast Consumer **住在 batch runner 内联**,不是 `consumers/` 模块(`evaluation/functional/consumers/__init__.py`:3-8)。

| 身份 | 状态 |
| --- | --- |
| `pooled_ridge_a1` + sMASE | 生命周期现役(T5 `f_task_spec` :264-267;Frep 主格) |
| `per_channel_ridge_a1` + sMASE | 同族结构变体;Frep 第二格 |
| `forecast\|ridge\|sMASE` / `ridge_smase` / `dlinear_shared\|nRMSE` | 方言,后两者非现役生命周期 |
| `ridge_alpha1/100`, `dlinear_closed_form`, `knn_analog_k3` | 探针 only(`consumer_axis_checkpoint.json`) |
| benchmark `DLinear` / `LSTMForecaster` | v0.2 模型,未接 G1/S1/T5 |

**多 Consumer 对照现成度**:仅 pooled vs per_channel(同 ridge 族)现成。ridge vs DLinear/kNN **未** 接入现代生命周期。

## 7. AD 线守卫资产

**`#31` 卡** `shared_outlier_repair_with_per_series_guard_v1`(`shared_capability_candidate_v2.json`:37-44;编译于 `run_e2_shared_capability_candidate.py`:188):

- 形态:`SHARED_CANDIDATE` / `GUIDANCE` / `target_support_required` / 无免费 TRY。
- v2 程序:`outlier_iqr`, `outlier_mad`。
- 守卫:`min_per_series_gain` delayed `lt -0.005` → VETO + RESCOPE。
- 适用 = `missing_fraction` / `local_robust_z_peak`——**无 `task_kind` 轴**。
- 证据来自 forecast traffic+NOAA。特征匹配时会在 forecast cell 上点火,因此**不是**跨任务惰性负控。

**AD Source**:`source_investigation_ad_v1/v2/v3`(`ad_source_skill.py`:24-28),适用 `task_kind==anomaly_detection`。v3 TRY 空、RISK 三清洗算子;因行为效果不可归因已归档(`AGENTS.md`:211)。临时 store 未提交。检索若滤 `task_kind`,可作第二负控,但不是现成 S2 宿主。

**意图中的 S2 惰性守卫** = 分类卡(`task_kind==classification`)在 forecast cell 上零检索零供给。Scope AST 已有该叶(`source_skill.py`:59;S1 :1683)。**现成度:谓词就绪,宿主未就绪**——S1 四臂不能挂 forecast cell。

## 义务

零代码改动(本盘点两工件 + 台账执行方条目除外);0 LLM / 0 fit / 0 下载;未跑全仓 pytest;未 spawn;未碰密封件与他线文件。
