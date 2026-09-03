# 代码审查报告: batch_recipe 栈 (batch_composition_headroom / M0a census)

- 审查角色: 独立代码审查者 (未参与任何被审代码编写, 只读审查, 0 LLM, 未 spawn 其他 Agent)
- 审查范围: 3 个被审文件 + artifacts/functional/e2/ 下的对照工件
- 审查基准: 取快照时刻(review 启动后工作区已稳定)的文件内容; 快照副本存于 `/tmp/bch_review_snapshot_202627/`
- 日期: 2026-08-20

---

## 0. 被审文件状态(行数 / 最后修改时间)

> ⚠️ **审查稳定性警告(重要)**. 本审查进行期间, 该工作区**正被其他进程并发修改**:
> `run_batch_composition_headroom.py` 在 20:15 → 20:19 → 20:23 三次变化
> (117441 B → 117816 B → 129117 B; 2798 行 → 2807 行 → 3094 行; sha256
> 3f4dac57… → … → 4536ef97…), 且它的 mtime(20:23)**晚于其全部工件**
> (最新工件 m0a_mask_geometry_census_traffic_v1 为 19:50)。M0b 工作区文件
> (contracts/observables.py、runtime/public_features.py、agentic/fast_path.py、
> agentic/runner.py、tests/minipipe/test_public_feature_calibration.py)在
> 20:21:42 被批量 touch(sha 未变, 内容与 git diff 一致)。审查结论基于最后稳定的
> 快照; 相关风险见 §G。

| 文件 | 行数 | Bytes | 最后修改时间 | sha256(前 16) |
| --- | ---: | ---: | --- | --- |
| evaluation/functional/run_batch_composition_headroom.py | 3094 | 129117 | 2026-08-20 20:23:03 | 4536ef974edd5a26 |
| evaluation/functional/run_e2_m0a_mask_geometry_census.py | 1315 | 53790 | 2026-08-20 19:47:53 | c17a57dff34e1aff |
| evaluation/functional/run_e2_m0a_mask_geometry_census_traffic.py | 1064 | 44475 | 2026-08-20 19:45:03 | cc7f43281d630ffc |

对照工件 (artifacts/functional/e2/): batch_recipe_{electricity,T233,traffic}_v1(.json/.md)、
consumer_conditioned_recipe_v1(.json/.md)、m0a_mask_geometry_census_v1(.json/.md, 冻结)、
m0a_mask_geometry_census_traffic_v1(.json/.md)、batch_composition_headroom_v1(.json/.md)、
masked_single_program_v1(.json/.md)、per_series_geometry_response_v1(.json/.md)。

审查期间只允许写入: `artifacts/functional/e2/_review_backup/`(工件副本, 18 个文件)与本报
告本身。未修改任何被审文件。

---

## A 级发现(影响已报告数字的正确性, 或数据边界/泄漏)

**A 级: 无。**

以下逐项给出 A 类检查的代码行证据(每条均核对了实际计算的最大索引, 见 §边界清单):

1. **traffic 配方窗口 ≤ 1848, 决不越 sealed_from_index=3072** — 通过。
   `run_batch_composition_headroom.py:224-233`（`_task_windows`）：traffic 的 Support 取
   `_TRAFFIC_DEVELOPMENT_ORIGINS[:2]`=(1104,1368)、delayed 取 `[2:]`=(1800)
   （`L165`），`farthest = max(support+delayed)+HORIZON = 1800+48 = 1848`，`> 3072` 即 raise
   （`L228-232`）。`_evaluate_assignment` 的实际最大读取索引 = 1800+48−1 = **1847**
   （`L316` eval 上下文 `raw[origin-192:origin]`、`L322` truth `raw[origin:origin+48]`、
   `L327` `raw[:origin]`；train 锚点 312..852 → ≤900）。已实测 traffic.csv 行数 17544，1847 远在界内。
2. **traffic 普查 ≤ 1104, 结构性 max_rows 上限** — 通过。
   `run_e2_m0a_mask_geometry_census.py:278-310`（`_traffic_values`）：`cutoff`（=1104，
   `_traffic_work_items` `L325`）若 `>= _TRAFFIC_SEALED_FROM_INDEX(3072)`（`L289`）或
   `> _TRAFFIC_RECIPE_FARTHEST_READ(1848)`（`L294`）即 raise；`load_csv_columns(..., max_rows=cutoff)`
   （`L299-301`）硬性只读 0..1103。工件 `provenance.csv_rows_read=1104`、`csv_rows_loaded=1104`，
   我独立核对一致。
3. **electricity/T233 窗口(经 load_cohort)只落在 development origins 允许范围** — 在项目语义下通过。
   `_task_windows`（`L234-238`）直接使用冻结 Task roster 的 `support_origins/delayed_origins` =
   **(3072,3120,3168)/(3216,3264,3312)**（e1v2_task_01, e1.py `_task_spec` `L234`、`base=3072`
   = `_UNEXPOSED_FIRST_SUPPORT_ORIGIN` `e1.py:106`）。实测序列长度: T233=10920, electricity=20000
   (CSV load_csv_columns 默认 max_rows 截断), 最大使用索引 3312+48−1=**3359** 均在界内。
   本项目语义下, 这些即为这两个开发 cohort 的 e1v2 冻结 Roster 开发原点; 脚本自始至终未打开任何
   sealed Outcome 工件(NOAA/g3_final_query_outcome/KDD W3 T211-T230"delayed truth"),
   工件自身的 exposure 字段亦如实声明。
   **语义对照(需读报告者自行取舍)**: 与 traffic 的"显式 3072 密封边界 + 窗口压在其下"相比,
   electricity/T233 是把 e1v2 Roster 原点(起点即 3072, 即公共前缀截断点)直接当作开发原点。若审查方
   将 3072 视为全局密封线, 则这两列的任务_01 窗口(3072–3359)处于"未暴露块"内——但按 e1.py 注释这
   恰是 e1v2 开发 Task 所在块, 且没有任何代码打开 Outcome 工件。**本报告判为不构成泄漏(A), 但作为
   B 级"语义严格性"提示记录(见 B1)。**
4. **无任何路径打开 sealed Outcome 或未授权 roster** — 通过。
   全部数据读取点枚举见 §边界清单; 连入的 cohort 仅 `agentic.runner.load_cohort` 暴露的
   electricity/T233/e31/weather(`runner.py:125-201`)与 traffic 冻结 screening roster
   (`headroom.py:163-164, 200-203`)。冻结普查工件 `m0a_mask_geometry_census_v1.json` 只被
   读(`L807` `_m0a_rows`; traffic census `L348`), 且被 traffic census 以 `_sha256`(`traffic_census.py:410-417`)
   记录"运行后 byte-identical"(实测 284cad38… 与工件记录一致)。census 主模块只在以冻结 cohort 集合
   原样复现时才写自身冻结工件(`census.py:1136-1148` 拒绝任何其他选择; `L1274-1280`)。

---

## B 级发现(真实缺陷 / 语义风险, 不影响已产出工件数字)

### B1 — 冻结 v1 工件与当前源码存在 schema 漂移; 源码比自己的工件"新"
- 证据: 当前 `make_batch_recipe` 写入 5 个字段 `adoption_rule_version`、`consumer_variant`、
  `delayed_stability_bar`、`identity_absolute_loss`、`consumer_variant_scope`(源码 `L1711-1741`),
  但冻结 `batch_recipe_{electricity,T233,traffic}_v1.json` **均不含这些键**(读取得到 `None`),
  而 `adoption_trace` 含 `delayed_bar/delayed_margin`——说明这批工件由**旧一版脚本**生成。
  结合 §0: 脚本 mtime(20:23)晚于全部工件(最晚 19:50), 且在审查期间继续被改。
- 影响: 已产出的所有核心数字均可由当前代码逐位复现(见 §F), 故**不改变任何已报告数字**;
  但"工件可由当前工作区逐字节复现"不成立, 工件的 provenance 自述字段(v2、per_channel 等)并非
  来自这些 v1 工件生成时刻的代码。即**代码→工件可追溯性**弱于声明。
- 建议(不实施): 冻结工件应记录生成用的 source sha。

### B2 — CC R 的 T233@per_channel 采用了"delayed 净负"掩码方案(v1 规则缺陷的留痕)
- 证据: `consumer_conditioned_recipe_v1.json` T233 per_channel `adopted_plan` = MASKED_PLAN
  `smooth_ma`, excluded [T233,T236,T246,T247,T254,T256], support **+0.0327707**, delayed
  **−0.0761477**(负). 采用路径 = "masked plan cleared the delayed stability check"。
  这是 `ADOPTION_RULE_V2`(`headroom.py:1512-1526`)正文所记载的 v1 失败例:
  v1 的 bar = best_full_batch delayed = **−0.146275**(`make_batch_recipe` v1 分支 `L1564-1568`),
  因为 −0.076 ≥ −0.146 故 PASS; 若按 v2(bar = max(bar_delayed,0)=0)则该候选 FAIL、退回。
- 影响: 该格子的**决策**是 v1 规则的已知缺陷产物(delayed 门没有把 identity 视为在位者)。
  数字本身(−0.076)如实且算术正确; 受影响的是该格子的"adopted"含义, 非数字。属 B。
- 相关: v2 规则存在于同一模块, 但**没有任何已交付路径调用它**——`_run_recipe`(`L1999`)、
  `_run_consumer_conditioned`(`L2774, 2785`)均硬编码 `adoption_rule_version="v1"`;
  `recipe_v2_all_cells` 模式(见 B3)是唯一 v2 调用点且无工件。

### B3 — v2 all-cells 路径在交付物中从未被执行(dead 且含硬编码预期)
- 证据: `_run_recipe_v2_all_cells`(`L2239-2327`)存在, 写 `batch_recipe_v2_all_cells_v1.json/.md`
  (`RECIPE_V2_ALL_CELLS_*` `L152-158`), CLI mode `recipe_v2_all_cells`(`L2988, L3004-3005`);
  但 artifacts/functional/e2/ 下**不存在**该协议工件 → 该路径从未运行过, 其输出、刻画的
  `EXPECTED_V2_CORRECTION` 判定(`L2270-2286`, 硬编码"仅 T233@per_channel"为唯一预期变化)全部未经验证。
- 影响: 未覆盖/未使用的代码路径; 若未来运行, 任何其他 cell 的 v2 变化都会被标
  `UNEXPECTED_CELL_CHANGE`(未必是 bug, 但该判定从未被校准过)。不影响现存工件数字。

### B4 — traffic 配方路径的读盘规模与普查不同(结构上限只在普查有)
- 证据: recipe 路径 `load_cohort`(traffic 分支 `headroom.py:187-217`)调
  `g3_sourcing.load_csv_columns(_traffic_csv_path())` → `max_rows=20000` 默认值
  (`g3_sourcing.py:74, 89`), 实测 traffic.csv 17544 行 → **整表 0..17543 载入内存**,
  其中包含 ≥1848 乃至 ≥3072(sealed 线)的行——虽无任何计算触碰 ≥1847 的索引(由窗口算术保证),
  但与普查"max_rows=1104 结构性不读"(`census.py:299-301`)不对称。
- 影响: 计算数字不受影响(无索引 ≥1848 被使用), 无 Outcome 被打开; 但"绝不读取密封区"在 recipe
  路径是靠窗口算术而非结构性上限保证。建议(不实施)在 recipe 的 traffic 加载处也传 `max_rows=1848`。

---

## C 级发现(风格/健壮性建议 — **只记录, 不实施**)

1. 路径可移植性不对称: `headroom.py:169-180` 与 `census.py:262-275` 对 traffic 都做了
   `C:/…` 与 `/mnt/c/…` 双候选, 但 `agentic.runner.load_cohort` 的 electricity 分支
   (`agentic/runner.py:156-158`)只写 `C:/Users/辉/desktop/agent/…`, 缺 `/mnt` 回退——
   在本机复现 electricity/T233 任一 cell 会直接 `FileNotFoundError`(已实测); traffic 可复现。
   建议统一为共享的双候选 helper。
2. `_traffic_*` 常量/双候选路径逻辑在三个被审文件与 traffic census 之间手工镜像, 靠
   `wiring_checks`(`traffic_census.py:121-132`)交叉核对; 建议收敛为单一来源。
3. `_evaluate_assignment` 内联 `import statistics`(`L371`); `run_cohort`/`run_masked_cohort`/
   `make_batch_recipe` 各自重复 load_cohort + 全 Program 编译(re-run 成本); 量级小, 非缺陷。
4. `_run_recipe_v2_all_cells` 的"预期变化"硬编码(`L2270-2272`)应与规则文本同源定义。
5. B2 与 v1/v2 规则并存但未统一: 建议要么产出一个 v2 工件, 要么把 v1 规则缺陷在 v1 工件内
   显式标注, 避免读者把 T233@per_channel 的负 delayed "adopted"误读为正向结论。

---

## 边界读取点清单(全部数据读取点 + 每个 cohort 实际读到的最大索引)

| # | 位置(文件:行) | 读取内容 | cohort | 最大索引 | 约束 |
| --- | --- | --- | --- | ---: | --- |
| 1 | headroom.py:190 `g3_sourcing.load_csv_columns(_traffic_csv_path())` | traffic.csv 全表 (default max_rows=20000; 实测 17544 行) | traffic(recipe) | 17543 载入 / **1847 使用** | 计算 ≤1847; 密封 3072 未被使用(见 B4) |
| 2 | headroom.py:186 `_load_exposed_cohort` → agentic/runner.py:125-201 | electricity.csv(321 通道, max_rows 截 20000)/T233(KDD npz 10920) | electricity/T233 | **3359** (delayed 3312+48−1) | e1v2_task_01 Roster 原点 3072–3360; 见 A3/B1 语义说明 |
| 3 | headroom.py:290 raw[anchor-192:anchor+48] (train) | 训练窗口 | 全部 | ≤900 | anchor 312..852 (kdd.py `_config`) |
| 4 | headroom.py:316/322/327 eval 上下文/truth/metric | origin-192:origin, origin:origin+48, :origin | 全部 | =#2/#1 对应 | 预测特征仅用 origin 之前历史 |
| 5 | headroom.py:807 M0A_CENSUS.read_text | 冻结普查工件 | — | — | 只读; sha 已核对 284cad38… |
| 6 | census.py:299-301 load_csv_columns(max_rows=1104) | traffic.csv 0..1103 | traffic(census) | **1103** | 结构性; 双 guard L289-298 |
| 7 | census.py:373-382 load_cohort + series[:support_origins[0]] | electricity/T233 public 前缀 | census | 3071 | 只取 train_uids |
| 8 | traffic_census.py:116 RECIPE_ARTIFACT / :176 SCREENING_ARTIFACT / :348 FROZEN_CENSUS | 工件 | — | — | 只读; 6 项 wiring_checks 全过 |
| 9 | traffic_census.py:361-363 load_cohort + series[:cutoff] | electricity/T233 train 前缀 | repro 检查 | 3071 | 24 行 exact 复现, all_exact=True |
| 10 | traffic_census.py:414-417 `_sha256` | 冻结普查字节 | — | — | 只读, 校验不与写 |

- 未授权 roster 检查: 连入唯一开发 cohort 集 {electricity, T233, e31, weather}(runner.py:133-185)
  与 traffic 冻结 screening 12/8 分(recipe `L163-164`；census `L102-103`); 未发现对
  NOAA / g3_final_query_outcome / KDD W3 T211-T230("delayed truth")任一 Outcome 工件的读取。

---

## 算术抽查(纯 JSON 核算, 未重跑; 每工件 ≥3 处, 附算式)

标识 `✓`: 逐位相等; `≈`: 末位 ULP 级差(相对 ~1e-16)。

1. **batch_recipe_electricity_v1**: 
   - `delayed_margin = delayed_aggregate_gain − delayed_bar`: 0.01634269521 − 0.00002194032 = **0.01632075490** = trace.delayed_margin ✓
   - 采用侧 = 掩码保留侧: comparison.support.adopted (0.0346431221) == masked_single_program best_plan support ✓; comparison.delayed.adopted (0.0163426952) == masked delayed ✓
   - BEST_FULL_BATCH 落法一致性: T233 同项 adopted == best_full_batch(support/delayed 均 0.0721555840/0.1166270757) ✓
2. **batch_recipe_traffic_v1**:
   - margin: 1.0470750570 − 1.0197743357 = **0.0273007213** = trace.delayed_margin ✓
   - outlier_mad 候选 stability_check=NOT_REACHED(首个 PASS 即停, 后续不求值) ✓; 其支持增益 0.66390447 = 掩码搜索 final_support(重跑日志一致) ✓
   - comparison.support.adopted (0.6652773536) − best_full_batch (0.6074409394) = 0.0578364143 = 掩码 step1 delta(重跑日志 +0.057836) ✓
3. **batch_recipe_T233_v1**:
   - winsorize trace: 0.0621620164 − 0.1166270757 = **−0.0544650594** = delayed_margin ✓ (FAIL)
   - outlier_iqr trace: 0.0474650838 − 0.1166270757 = **−0.0691619919** = delayed_margin ✓ (FAIL)
   - 两候选皆 FAIL → 回退 BEST_FULL_BATCH: adopted == best_full_batch (一致) ✓
4. **batch_composition_headroom_v1**(electricity & T233):
   - `aggregate_gain == mean(per_origin_gain)`: 两 cohort × 全部 7 program = **14/14 ✓**
   - `harmed_eval_series == {uid: per_eval_series_gain < −0.005}`, count 与 total_harm = −Σ harmed: 每 program 全对(如 electricity winsorize 4 系列 −0.0935831; T233 winsorize 2 系列 −0.6786476) ✓
   - `interaction = validated_composition_gain − sum_of_chosen_per_series_gains`:
     electricity −0.019068558 − 0.211376639 = **−0.230445197** ✓; T233 −0.253026847 ✓
   - `headline.support.composition == composition_interaction.validated_composition_gain` ✓
   - `composition_detail.{support,delayed}.aggregate == mean(per_origin)` ✓
5. **masked_single_program_v1**:
   - `full_batch_support/delayed.aggregate == mean(per_origin)` ✓; 每 accepted_mask 的 support/delayed 亦 ✓
   - 接受的 step 的 support 增益 == 对应 accepted_mask.support.aggregate(逐 step 核对) ✓
   - `headline.support.masked == best_masked_plan.support.aggregate`, delayed 同 ✓; bar == full_batch_support[best_full] ✓
6. **m0a_mask_geometry_census_traffic_v1**:
   - `summary_train.decision_point_count(12) == len(rows)`; `mask_class_counts` 求和==12 (MIXED 9 + OUTLIER_ONLY 2 + LEVEL_ONLY 1) ✓
   - `pss.divergent_count(10)` == 逐行 pss_divergent 求和 ✓; fraction 10/12=0.8333 ✓
   - `provenance.csv_rows_read=1104` == `cohort_meta.window_provenance.csv_rows_loaded`; `census_farthest_index_read=1103` === cutoff−1 ✓
   - sanity all_pass(12); frozen_reproduction all_exact(24), 0 mismatches ✓; wiring_check_all_pass ✓
7. **m0a_mask_geometry_census_v1(冻结)**:
   - 545 行 == 228(T233)+108(electricity)+209(weather); full_report_pooled 336 == 228+108 ✓
   - electricity/T233 各 12 行 e1v2_task_01(cutoff 3072), 四 sanit 全真 ✓
8. **consumer_conditioned_recipe_v1**:
   - pooled 节(adopted_plan/comparison/adoption_path/programs_searched/harm_account)与三个冻结 v1 配方**逐位相等**: electricity/T233/traffic 各 5 字段 = **15/15 ✓**(含任务要求补的 T233)
   - pooled interaction 数字 == headroom 工件 composition_interaction.interaction(electricity/T233/traffic 3/3 ✓)
   - per_channel 复核(v1 bar): T233 bar=−0.146275 <0 → 候选 −0.0761477 ≥ bar PASS 而 identity(0) 不作为在位者 → 确认 B2

---

## 字节复验(确定性, Task F)

- **备份**: 已把 9 组相关工件(18 文件)复制到 artifacts/functional/e2/_review_backup/(只复制; 已确认备份与 live 逐字节一致, 审查全程未再变化)。
- **重跑最便宜 cell**: `python evaluation/functional/run_batch_composition_headroom.py --mode recipe --cohorts traffic`(exit 0)。
- **结果**:
  - 磁盘工件逐字节比较 — 备份 vs 重跑后 live: **IDENTICAL**(全部 18 文件)。
    原因: `_run_recipe` 对已存在的 v1 工件**跳过写入**(`L2018-2024`, "RECIPE traffic skip write");
    故重跑不会覆盖冻结工件(这本身是写保护, 符合设计)。
  - 内存重算比较(同一代码重算 in-memory recipe 与工件核心字段): 
    * adopted_plan(种类/program/excluded 系列)、comparison 的 best_full_batch_program、adoption_path、
      PASS/FAIL/NOT_REACHED 全部一致;
    * 掩码搜索逐步日志与工件逐位一致(0.665277 / 0.636855 reject / 0.663904 / 0.608167 reject…);
    * 但 **FULL JSON 字节不同(DIFFERENT)**, 原因有二: (i) 当前代码新增 5 个 schema 字段
      (B1 的 schema 漂移); (ii) 关键数值有 ~1e-16 的末位 ULP 差
      (如 support.adopted recompute=0.6652773536463223 vs stored=…462215, 相对差 <8e-16)。
      该差异量与决策/6 位小数的报告精度无关, 特征为**跨环境 BLAS/浮点**差异
      (np.linalg.solve, run_e2_cross_series_curation.py:2802-2836), 非逻辑改动。
  - **同类可选(普查薄壳)**: 未另行重跑(成本不对称, traffic census 用 ProcessPoolExecutor 会
    放大开销); 但已对其 `_frozen_reproduction_check` 的 all_exact=True 与 frozen census sha
    记录做了交叉核实。
- 结论: 决策与所有报告位(6 位小数)确定性成立; byte 级可复现性**在当前工作区对磁盘工件成立、
  对"源码重算"仅到末位 ULP**。逐字节 IDENTICAL(磁盘) / DIFFERENT(内存, 有解释)。

---

## 变更面清点(Task G)

- **git status / diff --stat(仅跟踪修改)**:
  - 修改 6 个跟踪文件: contracts/observables.py(+7)、runtime/public_features.py(+28)、
    task_episode_harness/agentic/fast_path.py(+22)、task_episode_harness/agentic/runner.py(+3)、
    methods/ttha/harness/h0/snapshot.lock.json(±)、tests/minipipe/test_public_feature_calibration.py(+141)。
  - 其中 observables.py / public_features.py / fast_path.py / agentic/runner.py = **M0b 的
    四个工作区文件**(M0b 向公共观察契约新增 4 字段: level_region_fraction、
    level_region_end_fraction、outlier_region_end_fraction、level_only_post_shift_support_sufficient,
    见 observables.py diff 与 run_t233_supply_obs_ab.py:100-106); snapshot.lock.json + tests 为
    其连锁产物。**这批(3 个被审脚本 + 其工件)均未写这四个文件**——被审脚本的全部写操作都指向
    artifacts/functional/e2/(`L2026-2030/2320-2326/2910-2914/2956-2961/3072-3077`, census
    `L1273-1280`, traffic census `L995-1000`), 已在 grep 中排除任何对 runtime/、contracts/ 的写入。
  - `m0a_mask_geometry_census_v1.json`(冻结普查): mtime 01:39 **早于**本批全部工件(15:12+),
    sha 284cad38… 与 traffic census 运行后记录一致, 审查前后一致 → **未被触碰** ✓。
  - runtime/ 下无任何文件被这批脚本写入; runtime/public_features.py 的未提交修改是 M0b 的,
    traffic census 的 `_frozen_reproduction_check` 正是为把"M0b 修改的加性"变成可测事实而设
    (traffic_census.py:400-405, all_exact=True)。
- **本批实际改动面**: 3 个新脚本 + artifacts/functional/e2/ 下上述 9 组工件(该目录整体为
  untracked 新目录, 还含更早的 w1/…/autonomous_… 等非本批工件, git 无法区分)→ 以 mtime 为据,
  本批时间窗 15:12–19:50, 与声明面一致。19:50 同时更新的 m0b_field_wiring_report_v1.md、
  t233_supply_obs_ab_v1.md、t233_newobs_geometry_recut_v1.md、weather_per_task_readability_v1.md、
  in_domain_adaptation_trajectory_v1.md 与本批同刻生成(属同一工作会话的相邻产物, 不在本批
  声明面内)。
- **风险提示**: 脚本本体在本批工件生成后又修改(见 §0), 即"当前代码→已冻结工件"不可逐字节
  复现(B1); 且审查期间工作区被外部并发修改。若后续继续演进, 建议冻结工件时固化 source sha。

---

## 总裁定

**NUMBERS_TRUSTWORTHY** —— 在 6 位小数报告精度与全部决策层面, 三个被审脚本与其
batch_recipe / consumer_conditioned / headroom / masked / m0a 普查工件逐位自洽、边界不越限、
identity 管线对称、per_channel 与 pooled 特征/ridge 完全对等、搜索/门控分离成立、算术全对、
确定性复验通过(磁盘工件 byte-identical, 内存重算仅末位 ULP 差); 需按范围提示的信任边界为:
(i) `consumer_conditioned_recipe_v1.json` 的 **T233@per_channel** 采用格子的"adopted"语义
(v1 延迟门把 identity 排除在位, delayed −0.076 的采用是已记录的 v1 缺陷, 数字本身正确),
(ii) electricity/T233 的窗口位于 e1v2_Roster 的 3072+ 块(在"开发 cohort 冻结原点"语义下成立),
(iii) v1 工件相对当前源码存在 schema 漂移且跨环境无法逐字节复现(数值不受影响),
(iv) traffic 配方路径读盘 17544 行但仅用 ≤1847(结构上限仅普查具备)。
