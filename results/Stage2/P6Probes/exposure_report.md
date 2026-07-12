# P6 曝光账本报告（exposure_report.md）

- 生成脚本：`SelfEvolvingHarnessTS/diagnostics/p6_exposure_ledger.py`（只读审计，可重跑）
- 语料指纹：monash_clean.meta.jsonl sha256[:16]=`bca8cef57a1af3c3`；monash_clean.npz=`238d5867d86336d2`（mtime 2026-06-20 17:29:42）；monash_real.npz=`62077ad797ad000a`
- meta schema：`config`/`item_id` + 退化统计；**无 series_uid 字段**——uid 按生产代码 `data/load_real.py` L115 约定派生：`series_uid = f"{config}:{item_id}"`。

## 三值分类定义

- **confirmed_exposed**：持久化产物存在该 uid 级 literal 证据（字段级 JSON 解析优先，文本匹配仅完整 `config:item_id`，裸短 item_id 永不单独匹配），或某次已文档化真实运行的选择逻辑**确定性重建**覆盖该 uid（evidence 注明 `reconstructed:*`）。
- **uncertain_legacy_exposure**：所在 domain 被某次无 uid 级日志且**不可重建**的真实运行消费过。
- **certified_virgin**：**不适用于 legacy 83 条**。该类只能授予 P6 冻结之后新下载并 content-hash 登记的序列——legacy 语料在冻结前已存在于本机并被多次运行装载，无法对其出具『从未接触』的构造性证明，故本账本永不输出该类。

## 每 domain × class 计数

| config | confirmed_exposed | uncertain_legacy_exposure | certified_virgin |
|---|---|---|---|
| nn5_daily | 20 | 0 | 0 (不适用) |
| fred_md | 20 | 0 | 0 (不适用) |
| tourism_monthly | 20 | 0 | 0 (不适用) |
| covid_deaths | 20 | 0 | 0 (不适用) |
| us_births | 1 | 0 | 0 (不适用) |
| saugeenday | 1 | 0 | 0 (不适用) |
| sunspot | 1 | 0 | 0 (不适用) |
| **合计(83)** | **83** | **0** | **0** |

置信度分布：high=80，medium=3（medium=仅单一重建链支撑，见下）。

## 消费过真实数据的运行台账（args 来源逐条注明）

| run | 日期 | 语料 | 选择/重建 | LLM | uid级日志 |
|---|---|---|---|---|---|
| real_longrun_R0/R1/R2/R1c/R2c+fc_maintable+calibrate_eps | 2026-06-20/21 | monash_real(12) | ALL(默认 npz 全量) | R1/R2/R1c/R2c=flash；R0/R0''/fc_maintable/R4=无 LLM | ✗ |
| real_longrun_R0'(encoder=real, monash_clean) | 2026-06-20 17:30 | monash_clean(83) | ALL 83 → split_encoder_eval(frac=0.5, seed=0)：pre 半入编码器预训练、ev 半入诊断语料 | 无 | ✗ |
| run_stream_s1[s1_flash,s1_pro] | 2026-06-23 | monash_clean(83) | min_signals=5, max_per_domain=8, n_per_signal=3（重建=每大域前 8 条） | flash/pro | ✗ |
| run_stream_s1[s1_flash_chronos,s1_pro_chronos] | 2026-06-23 | monash_clean(83) | 同上 + --substrate chronos（选择与 substrate 无关） | flash/pro | ✗ |
| run_stream_s1_v2[s1_{flash,pro}{,_chronos}_v2] | 2026-06-24/25 | monash_clean(83) | min_signals=5, max_per_domain=0(未给→全量!), n_per_signal=4(默认)（重建=每大域全部 20 条） | flash/pro | ✗ |
| anchor_maintable(run_main_table) | 2026-07-02 | monash_real(12) | ALL 12（build_real_corpus 全量） | 无（chronos 判官） | ✗ |
| anchor_s1(run_stream_s1) | 2026-07-02 | monash_real(12) | min_signals=4, max_per_domain=0 → 3 域全 12 条 | flash | ✓ |
| P5Quadrant(run_p5_quadrant) | 2026-07-09/10 | monash_real(12) | ALL 12 × 4 preset | 无（true judge） | ✓ |
| P5A3Final(run_p5a3_final) | 2026-07-10 | monash_real(12) | episodes: signals[idx%12] 轮询（seeds 80-99 × 3 → 60 episodes 覆盖全 12） | 是（seeds 80-99 一次性消耗） | ✗ |
| AdaCTS 清洗代实验(eval_out_*) | AdaCTS 时期（monash_corrupt） | monash_real 同源 12 条（corrupted 版） | ALL 12 × corruption 网格 | deepseek+heuristic | ✓ |

provenance 细节：
- **real_longrun_R0/R1/R2/R1c/R2c+fc_maintable+calibrate_eps**：BUILD §8 R0-R4 行；日志 _real_step{1b,2}{,_chronos}.log、_real_step1b_chronos_table.log、_fc_maintable.log（mtime 2026-06-20/21）；transcript 2026-06-20 verbatim 命令（均无 --npz/--configs → 默认 12 信号全量）
- **real_longrun_R0'(encoder=real, monash_clean)**：transcript 2026-06-20 verbatim：run_real_longrun --mode diag --encoder real --npz .../monash_clean.npz --encoder-cache .../frozen_lstm_real_h64.pt；BUILD §8 R0' 行；工件 evaluators/_artifacts/frozen_lstm_real_h64.pt mtime 2026-06-20 17:30:35（npz 建成后 53s）
- **run_stream_s1[s1_flash,s1_pro]**：transcript 2026-06-23 两条 verbatim 命令（--k 2 --epochs 2 --max-per-domain 8 --n-per-signal 3）；BUILD §4.5「同配置」+ §8 复现行；runs/s1_{flash,pro}/ summary mtime 2026-06-23 11:28/11:59
- **run_stream_s1[s1_flash_chronos,s1_pro_chronos]**：transcript 2026-06-23T04:35Z verbatim 链式命令（含 --max-per-domain 8 --n-per-signal 3 --substrate chronos）；runs/ mtime 12:43/13:10 (+0800)
- **run_stream_s1_v2[s1_{flash,pro}{,_chronos}_v2]**：rerun_v2.sh 全文 verbatim 存于 transcript 6e3ee906（2026-06-24 Write 工具记录）：--npz monash_clean --llm {flash,pro} --k 2 --epochs 2 --substrate {frozen,chronos} --out-dir runs/s1_*_v2，无 --max-per-domain/--n-per-signal；4 个 out-dir summary mtime 2026-06-24 23:19 → 06-25 01:16
- **anchor_maintable(run_main_table)**：results/anchor_maintable/config.json（持久化参数：npz=AdaCTS/data/monash_real.npz, task=forecast, seeds=2, judge=chronos + 全库代码指纹）
- **anchor_s1(run_stream_s1)**：results/anchor_s1/config.json（持久化参数 + 代码指纹含 run_stream_s1.py=ff07d77b0bba3525）；flash_run/summary.json；candidates_flash.jsonl 的 split_fingerprint.held_in/held_out.series_uids 携 literal uid（S0.5+F2）
- **P5Quadrant(run_p5_quadrant)**：results/Stage2/P5Quadrant/records.jsonl —— 每行 literal series_uid 字段（uid 级日志✓）
- **P5A3Final(run_p5a3_final)**：manifest.json（'12 signals x FORECAST_PRESETS'）+ records.jsonl（series_family 级）+ run_p5a3_final._episodes 确定性轮询重建
- **AdaCTS 清洗代实验(eval_out_*)**：AdaCTS/data/eval_out_{deepseek,heuristic}.metrics.jsonl —— 每行 literal config+item_id 字段（uid 级✓）；monash_corrupt.meta.jsonl

## run_stream_s1 选择重建结论

1. **确定性**：`real_domains()`（run_stream_s1.py L63-89）从 npz 顺序装载信号（`load_signals` 按存储序迭代、无 RNG），按 `config` 分组（保插入序）→ `len>=min_signals` → `sort(key=(-len, name))` → `group[:max_per_domain]`。全程无随机性；`--order-seed` 只重排 domain 顺序不改成员；选择发生在 proposer 构造之前且不读 `--llm`/`--substrate`。
2. **与 LLM 臂无关 → flash 与 pro 选择相同集合（可证完备）**：同一确定性函数、同 npz、同参数；且 8 个 run 的 summary.json 域序全部等于重建 canonical 序（gate G8 ✓）。
3. **2026-06-23 四 run（flash/pro × frozen/chronos 判官）**：`--max-per-domain 8` → 每大域前 8 条（npz 序=T1..T8）× 4 域 = **32 条**。
4. **2026-06-24/25 四 v2 run**：rerun_v2.sh（transcript 全文恢复）**未传 --max-per-domain** → 默认 0=全量 → 每大域全部 20 条 = **80 条**（意外发现：v2 消费面远大于 BUILD.md 文档化的 32 条）。
5. **anchor_s1（2026-07-02）**：monash_real.npz + `--min-signals 4` → 3 域全 12 条（config.json 持久化参数，最强 args 证据）。
6. **代码漂移防护**：当前 run_stream_s1.py sha256[:16] 与 anchor_s1 config 指纹（2026-07-02）比对 → 一致（current=ff07d77b0bba3525）。6-23/6-24 run 早于指纹日期，另以 summary 域序 + BUILD §4.5 文档化语义（『每 domain 取前 N 信号』）交叉钉住。

## 验证 gates

- [PASS] G1 load_signals(monash_clean)==meta 83 条且保序（min_len/finite 无淘汰） — loaded=83
- [PASS] G2 monash_real 12 条 ⊂ monash_clean 且 z-score 后序列 bit 级相同 — 12 uids=['fred_md:T1', 'fred_md:T2', 'fred_md:T3', 'fred_md:T4', 'nn5_daily:T1', 'nn5_daily:T2', 'nn5_daily:T3', 'nn5_daily:T4', 'tourism_monthly:T1', 'tourism_monthly:T2', 'tourism_monthly:T3', 'tourism_monthly:T4']
- [PASS] G3 cap8 选择域=4 大域、各 8 条、皆为各域前 8（npz 序） — {"covid_deaths": ["covid_deaths:T1", "covid_deaths:T2", "covid_deaths:T3", "covid_deaths:T4", "covid_deaths:T5", "covid_deaths:T6", "covid_deaths:T7", "covid_deaths:T8"], "fred_md": ["fred_md:T1", "fred_md:T2", "fred_md:T3", "fred_md:T4", "fred_md:T5", "fred_md:T6", "fred_md:T7", "fred_md:T8"], "nn5_daily": ["nn5_daily:T1", "nn5_daily:T2", "nn5_daily:T3", "nn5_daily:T4", "nn5_daily:T5", "nn5_daily:T6", "nn5_daily:T7", "nn5_daily:T8"], "tourism_monthly": ["tourism_monthly:T1", "tourism_monthly:T2", "tourism_monthly:T3", "tourism_monthly:T4", "tourism_monthly:T5", "tourism_monthly:T6", "tourism_monthly:T7", "tourism_monthly:T8"]}
- [PASS] G4 v2 选择=4 大域全 20 条（80 条），且 cap8 ⊂ v2 — |v2|=80
- [PASS] G5 anchor_s1 选择=monash_real 3 域全 12 条 — domains=['fred_md', 'nn5_daily', 'tourism_monthly']
- [PASS] G6 P5A3 轮询重建覆盖全 12 条 — |p5a3|=12
- [PASS] G7 R0' split(seed=0) pre∪ev=83 且不相交 — |pre|=43 |ev|=40
- [PASS] G8 8 个 S1 run 的 summary.json 域序=重建 canonical 序（covid,fred,nn5,tourism） — s1_flash:OK; s1_pro:OK; s1_flash_chronos:OK; s1_pro_chronos:OK; s1_flash_v2:OK; s1_pro_v2:OK; s1_flash_chronos_v2:OK; s1_pro_chronos_v2:OK
- [PASS] G8b anchor_s1/flash_run summary 域序=fred,nn5,tourism（config.json note 同） — ['fred_md', 'nn5_daily', 'tourism_monthly']
- [PASS] G9 run_stream_s1.py 当前 sha256[:16] == anchor_s1(2026-07-02) 指纹（选择代码未漂移） — current=ff07d77b0bba3525 anchor=ff07d77b0bba3525
- [PASS] G10 monash_clean.npz mtime 早于全部 S1 run（2026-06-23 起） — npz mtime=2026-06-20 17:29:42

## 12 个 P5 uid 与重建集的重叠

- P5Quadrant literal series_uid（12 条）：`['fred_md:T1', 'fred_md:T2', 'fred_md:T3', 'fred_md:T4', 'nn5_daily:T1', 'nn5_daily:T2', 'nn5_daily:T3', 'nn5_daily:T4', 'tourism_monthly:T1', 'tourism_monthly:T2', 'tourism_monthly:T3', 'tourism_monthly:T4']`
- 关系：P5 12 条 = monash_real 全量 = {nn5_daily,fred_md,tourism_monthly}×{T1..T4}，**⊂ 2026-06-23 cap8 集（各域前 8）⊂ v2 全量集**；covid_deaths 不在 monash_real（P5 无 covid）。
- bit 级同源验证（G2）：monash_real 12 条与 monash_clean 对应条目 z-score 后逐点相同 → PASS → P5/AdaCTS 时期消费的就是同一批底层真实序列。

## P5IdentityGate 验证（任务点 3）

- 扫描 3 个文件：真实 uid 命中 = 0 （无 → P5-A 确为合成 anomaly/forecast slice，记录 uid 形如 c40_p2_0）
- domain 名出现：无

## 意外发现（超出任务列举的证据源）

1. **v2 四连跑消费全 80 条大域序列**（runs/s1_*_v2，2026-06-24/25）：BUILD.md 只文档化了 cap8 的 flash/pro 复现行；v2 的『无 cap 全量』只能从会话转录恢复的 rerun_v2.sh 得知。
2. **chronos 判官双跑**（runs/s1_{flash,pro}_chronos，2026-06-23）：BUILD.md 未记；transcript verbatim 命令含 `--max-per-domain 8` → 与 cap8 同集。
3. **R0' 编码器事件（2026-06-20）**：`run_real_longrun --mode diag --encoder real --npz monash_clean` 把 **全部 83 条**（含 us_births/saugeenday/sunspot 单条域）按 seed=0 分层对半：pre 半（43 条）进 **frozen LSTM 编码器预训练**（工件 evaluators/_artifacts/frozen_lstm_real_h64.pt 至今存在），ev 半（40 条）进诊断语料。三个单条域 series 都落在 pre 半（重放验证✓）→ 它们的唯一曝光通道即编码器预训练。**P6 注意**：该编码器权重本身内嵌 legacy 真实数据——若 P6 在 virgin 序列上使用它做判官底座，构成一条间接耦合通道（不违规，但预注册应声明）。
4. **AdaCTS 清洗代 literal 证据**：AdaCTS/data/eval_out_{deepseek,heuristic}.metrics.jsonl 逐行含 config+item_id 字段（monash_corrupt = monash_real 同 12 条的受蚀版）→ 12 条重叠序列早在 AdaCTS 时期即被 LLM 清洗实验逐 uid 消费并留档。
5. **monash_real ⊂ monash_clean（bit 级）**：两语料 base_std 至 15+ 位小数一致且数组逐点相等（G2），即 anchor/P5/AdaCTS 全部 12-信号运行消费的正是 83 条中的 12 条。
6. **results/Stage2/P6Probes/ 已有 u_admission 探针**（electricity_hourly/traffic_hourly）——P6 新数据入场流程已在走 content-hash 登记路线，与本账本结论一致。
7. **anchor_s1 候选级日志其实携带 uid**：candidates_flash.jsonl 每条候选的 `split_fingerprint.held_in/held_out.series_uids` 内嵌 literal series_uid 列表（Stage 0 F2 series_uid 分组的副产品）→ anchor_s1 的 12 条消费有 uid 级留档，非任务预设的『无 uid 日志』。

## 对 P6 的含义

- **legacy 83 条全部只能进 C0/D**（对照/开发池）：83 条 confirmed_exposed（uncertain=0——所有已识别真实运行的选择均可确定性重建）。没有任何一条可申领 certified_virgin。
- **V/U（virgin/未见评测池）必须来自 P6 冻结后的新下载**，入库时做 content-hash 登记 + uid 级消费日志（本次审计暴露的教训：deploy_stream/forward_transfer 仅 cell 级、run_stream_s1 无选择落盘——P6 harness 应强制 per-series manifest）。
- **单条域（us_births/saugeenday/sunspot）置信度=medium**：唯一曝光链是 R0' 编码器事件的重建（无 literal 工件）。保守起见仍记 confirmed_exposed（重建确定性 + 命令 verbatim + 工件 mtime 链），不因证据薄而降级为『可当 virgin 用』。
- **frozen_lstm_real_h64.pt 为 legacy-data-derived 工件**：P6 若复用需在预注册中声明（见意外发现 3）。

## 方法与局限

- 字段级扫描：591 个 json/jsonl（成功解析 591）；文本兜底 655 个文件；literal uid 命中 12 条：`['fred_md:T1', 'fred_md:T2', 'fred_md:T3', 'fred_md:T4', 'nn5_daily:T1', 'nn5_daily:T2', 'nn5_daily:T3', 'nn5_daily:T4', 'tourism_monthly:T1', 'tourism_monthly:T2', 'tourism_monthly:T3', 'tourism_monthly:T4']`。
- 语料 meta 清单的处理：AdaCTS/data/monash_{real,corrupt}.meta.jsonl 属『被整体消费语料的 uid 清单'（manifest），按任务定义计入 literal 证据；审计对象 monash_clean.meta.jsonl 本身不在扫描根内（它定义账本行而非曝光证据）。对应 12 条序列的消费性 literal 证据独立存在于 eval_out_*.metrics.jsonl / P5Quadrant/records.jsonl / candidates_flash.jsonl。
- 重建假设：npz 未被改写（G10 mtime 早于全部 run；内容 hash 已钉于报告头部）；选择代码语义自 2026-06-23 起未变（G9 指纹钉 2026-07-02→今；6-23/24 由 BUILD §4.5 文档语义 + G8 输出域序间接钉住）；会话转录中的命令字符串即实际执行命令（Bash tool_use 记录）。
- 保守规则：所有 gate 联动——任一重建 gate FAIL 时对应 `reconstructed:*` 证据自动不授予，相关序列自然回落 uncertain_legacy_exposure；本次运行 gates 全 PASS。
