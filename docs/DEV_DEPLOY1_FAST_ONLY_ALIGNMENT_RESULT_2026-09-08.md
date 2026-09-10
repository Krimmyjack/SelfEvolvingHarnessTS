# DEV-DEPLOY-1 结果：部署口径对齐（第一包，Part A + B + C 全部完成）

日期：2026-09-08 至 2026-09-09。执行者：Opus（本会话）。任务书：
`docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_TASK_2026-09-08.md`。

**状态：三项核心交付全部完成并落盘。** Part A（影子审计）、Part C（0-LLM
确定性基线）先于 Part B 完成；Part B（冻结 Fast-only 真实运行，120 次决策）
此前因缺少 `terra` 中转的 base_url/模型身份而阻塞——用户随后给出
`https://api.nowaterapi.xyz/v1`，核实为已知的 `gpt-5.6-sol` 端点（一次真实
最小billed调用确认 `returned_models == ["gpt-5.6-sol"]`），随即用新 key 完成
全部 120 次真实决策，0 故障、0 账户/权限边界、0 触及包级上限。

代码：`evaluation/main_protocol_p4/run_dev_deploy1.py`（单一逻辑 Runner，
`--part {shadow,baseline,fastonly,aggregate,all}`）。产物：

- `artifacts/main_protocol/dev_deploy1_fast_only_alignment.json`：Part A 全部
  120 行、Part C 校准/冻结读数、菜单 oracle、成本。
- `artifacts/main_protocol/dev_deploy1_part_b__{g1,g2}_u{9,10,11}.json`：Part B
  六个（组、单元）真实运行的原始输出（一单元一进程，逐决策落盘）。
- `artifacts/main_protocol/dev_deploy1_part_b_aggregate.json`：Part B 的
  逐单元/等权聚合读数（0 新增 LLM/fits，只重读已有输出）。
- `_scratch/dev_deploy1/{parts_a_c_output.json, g1/, g2/, part_b_budget.json}`。

## 0. 三问先答

**Support 曾经改写了多少及其净价值？** 116 条可判定决策（120 条中 4 条因
`chosen_candidate_id` 为空——select 输出异常，不可重建，明确记 UNKNOWN）中，
Support/准入门改写了 **63 条（54.3%）**，保留（选择即交付）**53 条（45.7%）**——
与 AGENTS §5.4(6) 记录的历史读数 **53/116 = 45.7%** 精确一致（独立复算得到的
交叉验证，见 §3）。改写的净值在两个面上都清楚为正：

- Origin 面：116 条的 pop-level 均值，"Fast 原选择若直接交付"仅 **0.0193**；
  实际经 Support 交付的均值为 **0.2175**——差一个数量级。
- Delayed(+48) 面：原选择均值 **0.0530** vs 实际交付均值 **0.1357**。
- 单独看 63 条被改写的：+48 净值 **+9.593**（35 条帮助 / 23 条伤害，5 条打平或
  数据缺口）；其中"选了 identity 却仍被部署别的程序"的 12 条，净值精确复现为
  **+3.5842（7 正 / 5 负）**，与 AGENTS §5.4(6) 完全一致。

  **解释边界**：这是"固定实际历史的一次决策影子"（任务书 §2A 原话）——只把
  这一格的 Support 覆盖去掉重算，course 其余部分、其余 119 格、写回的知识都
  原样保留。它不是"如果全程不设 Support 会怎样"的端到端反事实。**Part B 的
  真实冷跑证实了同一方向的结论**（见 §0 下一条）：两条独立证据线一致指向
  Support/准入门净值为正、且量级不小。

**冻结 Fast-only 相对 raw 和确定性基线怎样？** 已由 120 次真实决策直接回答，
不再是假说。Origin 面等权三单元均值：

| | raw | D-safe | **F（Fast-only，g1）** | **F（Fast-only，g2）** | D-fixed | 历史 Support 交付 | 菜单 oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| origin | 0 | 0 | **0.0577** | **0.0971** | 0.1182 | 0.2103 | 0.4725 |
| delayed(+48) | 0 | 0 | 0.1361 | 0.1420 | 0.1854 | 0.1311 | — |

**冻结 Fast-only 在 origin 面输给了 D-fixed（一个不做任何自适应的单一固定
winsorize 程序）**，两组（0.058、0.097）都低于 D-fixed 的 0.118；同时明显
跑赢 raw/D-safe（都是 0），也明显低于同一 Skill、同一模型但**经 Support
准入门交付**的历史水平（0.210）。风险画像佐证同一结论：u9 上 F 的
harmed_fraction 高达 **0.65**（20 条里 13 条材料性受损）、单序列最大伤害
**0.84**，两组都未过既有四线中的三线；这正是准入门原本要拦的那类候选。

**归因边界（重要，见 §2.4 末尾）**：F 与历史交付的 0.21 差距**不能整段
记到"拿掉准入门"头上**——Part B 相对历史同时去掉了准入门**和**同批次
u9→u10→u11 的 Episode 历史累积，本报告未拆分二者各自贡献。真正单独隔离
"只去掉准入门、其余不变"这一个变量的证据是本条上一段（§0 第一条）：同一
知识、同一历史，只把这一格的准入决定去掉重算，得到 0.0193 vs 0.2175
这一纯净读数。Part B 的 F 数字应理解为"无门 + 无同批历史"的联合效应，是
该联合效应的一个真实观测下界，不是准入门单独价值的估计。

**当前差距更像候选供给、选择、信息不足，还是菜单缺乏价值？** 就 u9/u10/u11 而言，
菜单本身有明确 headroom：机械 oracle（逐序列事后从合法菜单+identity 里选最优，
见 §4.1）origin 面均值为 **0.407 / 0.281 / 0.730**（三单元），显著高于实际交付的
**0.190 / 0.125 / 0.321**，也显著高于 D-fixed 的 **-0.163 / 0.119 / 0.399**。
菜单不缺价值；差距更像"选择/信息不足"——同一菜单在存在时能给出远高于当前实际
交付的收益，说明瓶颈不在候选供给的机制覆盖上。这与 Part A 的发现一致：Support
门在纠正"选错"上贡献了实际交付里的大部分收益。

## 1. Part A：DEV-SEQ-3 g2 全部 120 条历史决策的影子审计

范围：u9(1656)/u10(2136)/u11(2376) × {slow-candidate, knowledge-frozen} ×
20 条序列 = 120 条决策，逐条对照：

1. `chosen_candidate_id`（Fast 在当前 Support 之前真正选中的候选），
2. `winner_candidate_id`/`deployed`（Support 与准入门之后实际交付的程序）。

### 1.1 方法：不重问模型，重建执行记录而非匹配名字

原始 `dev_seq3_guidance_delivery__g2.json` 只为"交付"程序保留完整 typed steps
（`result.winner_program`）；未交付的候选只留下一个渲染标签（如
`outlier_mad` 或 `hampel_filter(window=7)`），typed 参数在写盘前已经丢失。
`chosen_program_from_row` 按以下规则重建，绝不只按名字匹配：

- `chosen_candidate_id == "identity"` → 该候选就是 identity，origin/delayed
  读数按定义为 0.0（raw 管线本身跑的就是这个）。
- `chosen_candidate_id` 为空 → 记 `UNKNOWN`（select 输出异常，不猜是 identity
  还是别的）。本批出现 4 例，全部见 §3.2。
- 命中一条 `verifier_rejected` 探测记录 → 确定性合法性失败，raw 实际执行，
  记 0（任务书原话："确定性合法性失败按既定且实际执行的 raw 回退记 0"）。
- 其余情况：从 `candidate_programs` 的渲染标签反解出 typed steps（这套
  DSL 的参数全是数值/布尔原语，`%s` 渲染对 `ast.literal_eval` 无损可逆），
  **立即用重建出的 steps 在该序列的 origin 面重新评分，与该候选当时已经
  记录的 probe 增益比对**；容差内一致才采信，否则整条降级为 `UNKNOWN` 并
  记录不一致的具体数值。本批 96 条尝试重建，**0 条自检失败**——96 条全部
  精确复现了历史记录的 origin 面增益。只有自检通过之后，才在缺失的
  delayed(+48) 面上花一次新读数（新 fit，不产生新 LLM 调用）。

只有在"选择 ≠ 交付"时才需要为 delayed 面花新读数；两者相同时直接复用交付
侧已经记录的 delayed 读数，0 新增计算。

### 1.2 结果分布

| 状态 | 条数 |
|---|---|
| `IDENTITY_SELECTED`（Fast 明确选 identity） | 20 |
| `RECONSTRUCTED`（从标签重建并自检通过） | 96 |
| `UNKNOWN`（select 输出异常，不可重建） | 4 |

选择与交付一致（53 条，45.7%）vs 改写（63 条，54.3%）——与 AGENTS §5.4(6)
"53/116 = 45.7%" 精确一致，是本次独立重算对历史读数的一次交叉验证，而非
重复引用同一次计算。

改写细分：20 条 `IDENTITY_SELECTED` 里 **12 条仍被部署了别的程序**
（identity 不能否决 pool 里其余候选继续被探测和准入），另外 51 条来自
`RECONSTRUCTED`（Fast 选的候选没通过准入，管线继续探测直到 identity 兜底或
另一候选通过）。

### 1.3 净值（+48 面）

- 63 条改写整体：+9.593（35 帮助 / 23 伤害 / 5 打平或数据缺口）。
- 其中"选 identity 却被改写"12 条：+3.5842（7 帮助 / 5 伤害）——与
  AGENTS §5.4(6) 完全一致。

本审计**不改写任何活动历史、Skill 或提示**，产出只进本报告与其 JSON 工件；
0 新增 LLM，46 次新增 fit（全部用于按需的候选重建自检 + 缺失 delayed 面）。

## 2. Part B：冻结 Fast-only 真实运行

### 2.1 阻塞解除与开跑前核验

用户给出 `terra` 的 base_url `https://api.nowaterapi.xyz/v1`——核对后发现
这正是既有记忆中已记录的 `nowaterapi`/`gpt-5.6-sol` 端点，只是换了新 key。
开跑前做了两次 0 假设的核验，均通过才动用正式预算：

1. **知识快照**：不从某次历史运行的 store 里读一个可能已被清理的路径，
   而是用与 `run_dev_seq3.build()` 完全相同的确定性函数
   （`base.load(ORDERING)` → `live._state_at_k1` → `R._resolve_k0_snapshot`
   → `compile_snapshot`）重新编译"A"（`knowledge-frozen`，未接 B 卡的父
   快照）。重算得到的 SHA
   `98dea3b037b790528cd745f8cbdf657da0285c25f91df53e9c572a0dfc87563b`
   与 DEV-SEQ-3 g2 `knowledge-frozen` 臂历史记录的 `knowledge_version`
   **逐字节一致**——0 LLM、0 fits 确认。
2. **模型身份**：一次最小 billed 调用（`run_dev_seq2.transport_preflight`），
   确认 `returned_models == ["gpt-5.6-sol"]`，与请求别名一致，非 Flash 家族。
3. **单条真实决策烟测**（u9/T14，`--limit 1`）：Fast 选中的候选与历史记录
   完全相同（`cand_skill_..._outlier_mad`），我方 0-LLM Scorer 算出的 origin
   增益 **-0.127428** 与历史 probe 记录逐位相同——证实 Part B 的评分管线与
   历史评分管线是同一套确定性计算，而不是另一套近似实现。

### 2.2 执行

按任务书固定：`experience_episodes=()`；每条决策独立会话，只调用
`TTHAMethod.prepare`（`runtime_prior_slot=False, pool_mode="actionable"`，
与 `run_online_round` 内部用的默认值一致），不调用 `run_online_round` 的
Support/准入搜索；`trace.chosen_candidate_id` 命中的候选只过确定性窗口
验证器，通过即交付，不再探测/比较其余候选；`chosen` 为空或
`"identity"` 一律交付 identity（与历史记录里这两种情况 100% 落地为
identity 的既成事实一致）；不可解析的候选 ID 出于安全考虑回退到 identity
但标为 `UNKNOWN_CANDIDATE_FALLBACK_TO_IDENTITY`，不冒充成 Fast 的主动选择。
`deployment_gain_at_origin`/`fixed_program_gain_at_plus48` 由同一 0-LLM
`Scorer`在决策定案后评分，从未回传给 Fast。

一单元一进程（6 个进程：g1/g2 × u9/u10/u11，按 g1→g2、u9→u10→u11 顺序
串行发车），并发 2（同单元内两个 Fast 会话并行，跨线程共享的
`ReplayPredictionCache` 加锁保护），每条决策完成即写入
`_scratch/dev_deploy1/{group}/u{position}.json` 检查点。

**结果：120/120 全部完成，0 故障、0 `ACCOUNT_OR_PERMISSION_FAULT`、
0 触及包级或分组 LLM 上限。** 实际用量：573 次物理 LLM（≈4.8 次/决策，
低于 24 的 per-sequence 上限）、108 次新增 fit、总墙钟约 79 分钟（含 6 次
进程冷启动）——远低于任务书 900/组、2000/包、6 小时的上限。交付方式分布：
`DEPLOYED` 98 条（Fast 选中的候选通过合法性检查、原样交付，不论好坏）、
`IDENTITY_SELECTED` 13 条、`EMPTY_SELECTION_FALLBACK_TO_IDENTITY` 9 条；
**0 条 `LEGALITY_FALLBACK_RAW`、0 条 `UNKNOWN_CANDIDATE_FALLBACK_TO_IDENTITY`**
——这次真实运行里 Fast 从未提出过非法或无法解析的候选。

### 2.3 主表：F vs raw / D-fixed / D-safe / 历史 Support 交付

| | u9 | u10 | u11 | 三单元等权均值 |
|---|---:|---:|---:|---:|
| raw | 0 | 0 | 0 | 0 |
| D-fixed (winsorize) | -0.163480 | 0.119261 | 0.398927 | 0.118236 |
| D-safe (identity) | 0 | 0 | 0 | 0 |
| **F g1（origin）** | -0.062319 | 0.085685 | 0.149617 | **0.057661** |
| **F g2（origin）** | -0.058828 | 0.088700 | 0.261534 | **0.097135** |
| 历史 Support 交付（origin，两臂均值） | 0.184916 | 0.124582 | 0.321361 | 0.210286 |
| F g1（delayed+48） | 0.076492 | 0.069388 | 0.262333 | 0.136071 |
| F g2（delayed+48） | 0.078919 | 0.080676 | 0.266330 | 0.141975 |
| D-fixed（delayed+48） | 0.011793 | 0.084721 | 0.459763 | 0.185426 |
| 历史 Support 交付（delayed+48） | 0.101235 | 0.022001 | 0.270166 | 0.131134 |

风险画像（origin 面，逐单元，两组一致）：

| 单元 | harmed_fraction | max_single_series_harm | 既有四线通过 |
|---|---:|---:|---|
| u9 | **0.65**（13/20） | **0.84** | 否（aggregate、harmed_fraction、single_series_harm 均未过） |
| u10 | 0.15 | 0.18 | 是 |
| u11 | 0.15–0.20 | 0.08–0.32 | g2 过；g1 未过（single_series_harm） |

### 2.4 解读

**F（冻结 Fast-only，无准入门）在 origin 面上系统性地低于 D-fixed**——
两组（0.058、0.097）都不及一个完全不自适应的单一固定 winsorize 程序
（0.118）。它显著跑赢 raw/D-safe（都是 0），说明 Fast 的候选生成/inspect
机制确实提取到了信息，不是随机噪声；但它明显低于同一 Skill、同一模型、
只是多了一道 Support 准入门的历史交付水平（0.210）。**§0 影子审计给出的
假说在此被证实**：拿掉准入门，Fast 自己一次性提出的候选不足以撑起它历史
上呈现的表现，历史表现的很大一部分来自准入门筛掉了坏候选、而不是 Fast
选得准。u9 的风险画像是最直接的证据：13/20 条序列材料性受损、单序列最大
伤害达 0.84，这正是准入门设计出来要拦截的场景，而这里没有门可拦。

在 delayed(+48) 面，F（0.136、0.142）与历史 Support 交付（0.131）接近，
都低于 D-fixed（0.185）——origin 与 delayed 面的相对排名并不一致，说明
"当格看起来好"和"48 步之后还站得住"是两个不同的问题，任务书要求把两者分开
报告是对的：不能只看一个面就下"F 整体更差/更好"的结论。

g1 与 g2 方向一致（F 均低于 D-fixed，均高于 raw），量级有差（0.058 vs
0.097）——两组共用同一开发数据、同一冻结知识，差异反映的是同一批决策点上
两次独立 LLM 采样的抖动，不是跨域复现，报告不合并成一个数字。

**口径澄清（Fable 2026-09-08 23:08 可行性检查已预先指出，此处正式确认）**：
F 相对历史 Support 交付同时去掉了**两样**东西——Support/准入门，以及
DEV-SEQ-3 原有的跨单元 Episode 历史（u9→u10→u11 内 `arm.history` 的累积）。
本报告只把 F 与历史交付的**总差距**（0.058/0.097 vs 0.210）归因于"拿掉
准入门"是不完整的：这个数字同时反映了"没有门"和"没有同批次内的历史反馈"
两个变化，二者的独立贡献本报告未拆分。真正**单独隔离准入门**这一个变量
的读数是 §0/§1 的影子审计（"原选择" 0.0193 vs "实际交付" 0.2175，同一
知识、同一历史，只去掉这一格的准入决定）——那才是"准入门净值为正"这一
结论的直接证据。Part B 的 F 数字应读作"无准入门 **且** 无同批历史反馈"
的联合效应下界，不应单独引用为"准入门的价值是 0.21-0.06=0.15"这类归因；
拆分两个变量的贡献需要第三种设计（保留同批历史、只去掉准入门），本包
未做，留给后续包。

## 3. Part C：真 0-LLM 基线（Static/raw、D-fixed、D-safe）

### 3.1 候选菜单

从当前冻结算子注册表（`operators/registry.py`，单一真源）机械生成：
`allowed_tasks` 含 `forecast`、非 shape-changing、非 deprecated alias 的
单步算子，按名字排序，共 **23 个**，另加 identity。只有
`period_median_complete` 要求非空参数（`period`），从 `ctx.config["period"]`
确定性绑定；其余全部使用注册表默认（空参数）。

### 3.2 校准（u7=1176 / u8=1416，全 20 条 origin 面）

23 个候选中 **11 个在两个校准单元上均被确定性窗口验证器拒绝**
（`denoise_median/savgol/stl/wavelet`、`fft_decompose`、`minmax_norm`、
`repair_level_shift`、`smooth_ema/ma`、`stl_decompose`、`znorm`——多数是
全局平滑/表示变换类，修改比例超过 0.35 的窗口验证上限），明确记
`UNAVAILABLE`，不补参、不改门槛。其余 12 个读数完整：

| 候选 | 两单元等权均值 | u7 未过线 | u8 未过线 |
|---|---:|---|---|
| **winsorize** | **0.192544** | single_series_harm | harmed_fraction, single_series_harm |
| outlier_iqr | 0.159789 | （无） | harmed_fraction, single_series_harm |
| outlier_mad | 0.137141 | single_series_harm | harmed_fraction, single_series_harm |
| impute_ssm | 0.046085 | harmed_fraction | harmed_fraction, single_series_harm |
| impute_ema | 0.050538 | harmed_fraction, single_series_harm | harmed_fraction, single_series_harm |
| impute_linear | 0.0 | aggregate | aggregate |
| period_complete | 0.0 | aggregate | aggregate |
| resample_uniform | 0.0 | aggregate | aggregate |
| impute_fft | -0.061522 | aggregate, harmed_fraction, single_series_harm | harmed_fraction, single_series_harm |
| hampel_filter | -0.047326 | harmed_fraction, single_series_harm | aggregate, harmed_fraction, single_series_harm |
| period_median_complete | -0.187989 | 三线全未过 | 三线全未过 |
| impute_ar | -24.207268 | 三线全未过 | 三线全未过 |

**D-fixed = winsorize**（两单元等权均值最高，0.192544），不预先保证安全。

**D-safe：无候选在两个校准单元上都通过既有四线 → 冻结 identity，
明确记 `NO_QUALIFIED_FIXED_PROGRAM`。** 这不是挑选后的结论：12 个读数完整
的候选**没有一个**同时满足 `min_treated`/`aggregate>=material`/
`harmed_fraction`/`max_single_series_harm` 四线在两个校准单元上都过——多数
卡在"聚合为正、但个别序列伤害超线"，与项目历史上反复出现的"聚合过线、逐序列
有害"模式一致。

### 3.3 冻结后在 u9/u10/u11 的读数（不再选择，仅评分）

| | u9 origin | u10 origin | u11 origin | 三单元等权均值 origin |
|---|---:|---:|---:|---:|
| raw | 0.0 | 0.0 | 0.0 | 0.0 |
| D-fixed (winsorize) | -0.163480 | 0.119261 | 0.398927 | 0.118236 |
| D-safe (identity) | 0.0 | 0.0 | 0.0 | 0.0 |

D-fixed 在 u9 上实际为负（harmed_fraction=0.75，四线中的 gate 在 u9/u11 也不
过），说明"效用基线"与"安全基线"确实是两件事，任何一个都不能替代另一个
（任务书原话："不能只打败退化为 raw 的 D-safe,就宣称打败了强确定性基线;
也不能忽略 D-fixed 的风险来宣称其全面更好"）。+48 面、四线通过情况的完整
逐单元数据见 JSON 工件 `baseline_frozen_scoring`。

## 4. 辅助诊断（同包完成，不作为准入门）

### 4.1 菜单内事后改进空间（菜单 oracle）

同一 23 候选菜单（12 个可读 + identity），逐序列事后选最优，origin 面：

| 单元 | 实际交付（历史两臂均值） | D-fixed | 菜单 oracle |
|---|---:|---:|---:|
| u9 | 0.185 | -0.163 | **0.407** |
| u10 | 0.125 | 0.119 | **0.281** |
| u11 | 0.321 | 0.399 | **0.730** |

oracle 只用于诊断，未进入 B/C 的任何输入；它是"完美逐序列 targeter"的上界，
不是可部署策略。u11 上 D-fixed 恰好逼近甚至超过 u9/u10 相对 oracle 的比例，
说明菜单 headroom 在不同单元上并不均匀——不能从单一单元外推"菜单已经够用"
或"菜单还差很多"。

### 4.2 可辨别尺度

Part C 的两次校准单元共用同一 g2 数据，暂不具备独立方差估计的材料。Part A
的 4/120（3.3%）UNKNOWN 集中在 `knowledge-frozen` 臂的 4 个不同序列/单元
组合，样本太小不足以判断是否与该臂特有的知识状态相关，留作观察，不据此下
因果结论。Part B 的 g1/g2 是同一批 20-series 决策点上的两次独立 LLM 采样
（配对 n=120，非独立簇），§2.4 已如实说明只报方向一致性，不建立方差可信
区间。

## 5. 成本

| | 用量 | 上限 |
|---|---:|---:|
| 新增 LLM 调用（Part B，120 决策） | 573 | 2000（每组 ≤900） |
| 新增 Consumer fits（Part A+C+B 合计） | 148 + 108 = 256 | 2000 |
| 墙钟（Part A+C） | 约 20 分钟 | 6 小时（共享） |
| 墙钟（Part B，6 进程合计） | 约 79 分钟 | 6 小时（共享） |

Part A 单独 46 fits；Part C 校准 48 fits + 冻结评分/菜单 oracle 诊断
54 fits；Part B 评分（origin+delayed，120 决策）108 fits。Part A/C 全部为
确定性 CPU 计算，无外部服务调用，脚本本身幂等（同一命令重跑得到逐位相同
的结果），按 AGENTS §7 的反过度工程原则未为其单独建设 checkpoint/resume
机制。Part B 调用了真实 LLM，因此按任务书要求做了每决策落盘的检查点
（`_scratch/dev_deploy1/{g1,g2}/u{9,10,11}.json`）与跨进程续预算的
`part_b_budget.json`；六个进程全部一次性顺利跑完，未触发任何恢复路径，
但检查点已就位、经过烟测验证（`--limit 1` 的单条决策即被检查点正确记录
并在概念上可被后续调用跳过）。

## 6. 交付清单

1. 本报告：`docs/DEV_DEPLOY1_FAST_ONLY_ALIGNMENT_RESULT_2026-09-08.md`。
2. `artifacts/main_protocol/dev_deploy1_fast_only_alignment.json`（plan、
   Part A 全部 120 行、Part C 校准/冻结读数、菜单 oracle、成本）。
3. `artifacts/main_protocol/dev_deploy1_part_b__{g1,g2}_u{9,10,11}.json` +
   `dev_deploy1_part_b_aggregate.json`（Part B 原始输出与聚合读数）。
4. `evaluation/main_protocol_p4/run_dev_deploy1.py`（Part A/B/C 实现，
   `--part shadow|baseline|fastonly|aggregate|all`）；`_scratch/dev_deploy1/`
   留存 Part A/C 副本与 Part B 检查点。
5. AGENTS.md §5.5 状态锁已追加一条记录本次结果（含 Part B）。

未获额外授权，本次改动未提交 Git。密钥（用户提供的 terra/nowater key）
只以进程环境变量形式短暂注入，未写入任何任务书、工件或本报告。
