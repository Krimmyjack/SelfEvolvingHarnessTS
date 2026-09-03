# D4 接线规格:HEC-1 四件(在 Source-v3 机器上加装,冻结中立)

日期:2026-09-03。地位:主线规格;执行方(P4 代码线 / Opus)实现。**原则**:全部落在
`evaluation/main_protocol_p4/`(新模块 + 新 runner `run_hec1.py`);**不改** `methods/ttha/*` 冻结面(改动会
轮转 `runtime_bundle_sha` lock);若确需在 `online_loop.run_online_round` 加钩子,必须默认关闭、字节等价复现
现行为、附聚焦测试并披露 lock 重写。Source-v3 runner(`run_source_line_v3.py`)与其工件**不得受影响**
(回归:`--dry-run` 输出与 `p4w3_source_line_v3.dry_run.json` 字段一致)。0 LLM 烟测是交付门。

复用清单(已存在,直接用):`v1._machinery()`、`loop.run_online_round(... candidate_scopes,
scope_resolver, scope_revision_preflight, program_supply_verifier, resupplied_programs,
risk_refusal_selector=distance.selector, risk_refusal_slow_agent=clause_slow)`、
`restricted_draft.RestrictedDraft / DraftLedger`(MAX_REVISIONS=2)、`scope_clause_agent.ScopeClauseSlowAgent`
(Runtime 写 manifest、Slow 只写 `predicate[-1]`)、`scope_repair_distance`(最少排除序列数)、
`scope_narrowing_preflight`、`scoped_serving_evaluator.scoped_evaluate`(同次产出 raw/program 预测)、
`admission_policy` / `bounded`、`_promote`(delayed → 重遇 → 激活)、`llm_cache`。

---

## W1 外环整合 `outer_loop.py`

**目的**:每 k 单元对 online 臂做一次"普查 → Slow 提案 → replay 筛选 → restricted Draft"。

**接口**:
```
consolidate(*, bank, ledger, k_index, slow, tool, replay, budget, rng=None) -> OuterStepRecord
```
- `bank`:本臂本顺序**已处理**单元的 Episode 集合(逐序列增益、22 维特征、serving_scope、准入结论、归因字段、
  程序步骤、行为指纹);只读。**不含**评价面(+144)记录、未来单元、他臂、held-out。
- 普查(确定性):按 `task_consumer_key × 行为指纹(逐序列增益向量去重)` 分组 → 候选三类:
  (a) ≥1 次 POSITIVE 且无卡的程序 → ADD 候选(沿用阶梯 v2 计价);(b) ledger 中 `REVISABLE` Draft 有新
  证据 → 修订候选;(c) Active Skill 重复 CONFLICT/NEGATIVE → 收窄/撤销候选。`FLAGGED` Draft 不进修订,进
  "观察/程序漂移信号"表。
- Slow:每步 ≤ `budget.outer_llm_per_step`(拟 2)次,经 W2 工具链得到完整 predicate。
- replay 筛选:对每个候选(程序 + Scope),在 bank 覆盖的已处理 cell 上用 `scoped_evaluate` 重解析 Scope 并
  重算逐序列增益(记 `replay_fits`,0 LLM);任一 cell 违反 bounded_risk 任一线或聚合 ≤ material → 淘汰
  (记 `REPLAY_SCREEN_REJECTED` + 违反项);通过 → `ledger.open_restricted(...)`。**不激活、不部署。**
- 输出 `OuterStepRecord`:候选数、淘汰数与原因、Draft 数、Slow 调用数、`replay_fits`、耗时。
- **验收**:(1) bank 为空或无候选 → 0 LLM、0 fit、记录空步;(2) 注入合成 bank(含一条已处理 cell 上违反 msh
  的候选)→ 必被淘汰;(3) 通过者出现在 `ledger.resupplied_programs()`;(4) 从不写 Active。

## W2 Scope 阈值工具 `scope_threshold_tool.py` + Slow 输出收窄

**目的**:Slow 只给语义(feature, direction, rationale),阈值由工具在 bank 上按冻结分箱校准;并列 ScopeFit shadow。

**接口**:
```
calibrate(*, feature, direction, rows, bins, policy) -> Clause | NoFeasibleThreshold
best_stump(*, rows, bins, policy, vocabulary) -> (feature, direction, threshold, objective)
```
- `rows`:bank 内该程序的逐序列记录(特征向量、增益);`bins`:冻结分箱边界(与 Scope 归纳同源,
  `observable_numeric_bin` / `s1._binned_contract_leaves` 同款,**不新造**);`policy`:bounded_risk_v1。
- `calibrate`:候选阈值 = 该特征分箱边界;对每个候选,解析集 = rows 中满足现有子句 ∧ 新子句者;取满足
  (hf ≤0.20 ∧ msh ≤0.30 ∧ 聚合 ≥ material ∧ |解析集| ≥ MIN_TREATED)的**最宽**阈值(并列取更粗分箱);
  无 → `NoFeasibleThreshold`。
- Slow 输出 schema 改为 `{feature, direction, rationale}`;`ScopeClauseSlowAgent` 在 `predicate[-1]` 位组装
  `{feature, op: direction, threshold: <tool>}`。Slow 若仍返回数值阈值 → **忽略并记录** `LLM_THRESHOLD_IGNORED`,
  以工具值为准。`NoFeasibleThreshold` → 反馈 Slow 换 feature/direction(≤2 次)→ 仍无 → 本步弃权记录。
- `best_stump`:在 `vocabulary × {<=, >=} × bins` 上按同一可行性 + 目标(过预算的最大聚合增益)取最优;每次
  Slow 提案时并列记录 `shadow = best_stump(...)`;后续单元对 shadow Scope 的评分由 W4 以 `shadow_fits` 完成。
- **反 Router 边界**(写进 docstring):工具不选程序、不决定是否行动、不选特征;只校准 Slow 指名方向的阈值。
- **验收**:(1) 合成 rows 上 `calibrate` 取到预期最宽可行边界;(2) 无可行时返回 `NoFeasibleThreshold`;
  (3) Slow 数值阈值被忽略且有记录;(4) `best_stump` 与 `calibrate` 在同一 rows 上的可行性判断一致;
  (5) 特征不在词汇 → 拒绝。

## W3 归因字段 + 三态机(扩展 `restricted_draft.py`)

**目的**:每次验证面失败按"失败线 → 归因"进入 `WAITING / REVISABLE / FLAGGED`,并约束允许动作。

**数据**:`RestrictedDraft` 增 `state`、`verification_attempts`(≤3)、`history[]`(每次验证:window、
treated_prev、treated_now、`attribution = {new_entrant:[...], continuing:[...], left:[...]}`、per-series gain、
failed_lines、state_after)。
**分类函数**:
```
classify_failure(*, failed_lines, per_series_gain, treated_prev, treated_now, material) -> state
```
- 仅 `coverage_floor` → `WAITING`;
- 任一线的受害/负贡献由 CONTINUING 主导(CONTINUING 中 gain < −material 的负贡献和 > NEW_ENTRANT 的)→ `FLAGGED`;
- 否则受害全为 NEW_ENTRANT → `REVISABLE`。优先级 FLAGGED > REVISABLE > WAITING。
**转移**:`REVISABLE` → Slow 可加子句(W2;`revisions ≤2`)→ 再验;`WAITING` → 每单元检查谓词在当前窗口
解析数 ≥ MIN_TREATED 则自动排一次验证(不耗修订);课程末仍 WAITING → 关 `PATTERN_NOT_REENCOUNTERED`;
`FLAGGED` → 只允许原样再验一次;再败 → 关 `EFFECT_NONSTATIONARY` 并向外环普查登记"观察/程序漂移信号";
任一状态 `verification_attempts` 达 3 → 归档。**任何状态都不删 Draft**(证据保留)。
**验收**:用 Source-v3 工件三案例回放——1896→`FLAGGED`、2376→`WAITING`、2616→`REVISABLE`;修订/验证计数上限
生效;WAITING 自动验证不耗修订;`FLAGGED` 拒绝加子句请求。

## W4 Runner `run_hec1.py` + 计分与记录

**目的**:Phase S / Phase T 的臂、顺序、单元、面、预算、故障分类、检查点、记录。

- **合同**:导入 `hec1_contract.py`,`assert_frozen()` 不过 → `BLOCKED_ON_CONTRACT`(照 v3)。
- **臂**:Static(raw 全部)、A5-frozen(K0;**每单元后重置 store/snapshot 到 K0**)、A5-online(K0;写回 + 外环)、
  A3-online(h0;写回 + 外环);K0 空 → 按合同缩臂。
- **顺序**:三份冻结单元序列;`--ordering forward|reverse|interleaved`;每顺序独立 store 根目录与 run_label。
- **单元**:Support-A(o)→ delayed(o+48)→ 评价面(o+144,只计分,结果写 `scoring_ledger`,**不进 bank**);
  部署 = 该臂当时 Active Skill 的机械召回 + Scope 解析 → 程序模型路由;识别不到 → identity。
- **内环 Slow 关闭**:`allow_slow=False`;所有 Slow 走 W1 外环(每 k 单元;online 臂)。
- **单一权威门**(D1 规格矛盾 #2:Source-v3 round 2856 `delayed_gate.passes=False` 与 online_loop
  `delayed_event` approved 并存):P4 `_gate`(含 coverage_floor)为**唯一**激活权威;online_loop 的
  delayed 批准结果只记录、不得单独调用 `activate_approved` 或写 Active;每单元记 `gate_disagreement =
  {p4_gate, online_loop_event, resolved_by: "p4_gate"}`;**验收**:构造一个 online_loop 批准而 P4 门拒绝的
  合成单元 → Active 集合不变且 disagreement 被记录。
- **预算账本**:`llm_fast`、`llm_outer`、`replay_fits`、`shadow_fits`、`course_fits`、`baseline_fits` 分列;
  单元臂 LLM 帽、Forward 总帽(≤500)硬守卫(第 N+1 次在后端前阻断、不计费,照 split-3)。
- **故障分类**:`UnitFault` → identity 弃权 + 记录 + 继续;`RunFault` → 中止,`RUN_BLOCKED_NO_VERDICT`。
- **Fast 原始决定记录**:每次 Fast 调用存脱敏原文摘要与分类 `{PROPOSED, ABSTAINED_WITH_REASON, EMPTY_OUTPUT,
  MALFORMED}`;零候选轮必带分类。
- **cache**:`llm_cache` 以规范化 prompt 字节为键;命中率入账;解码参数记录。
- **检查点**:每单元臂后落盘(rows、ledger、bank 摘要、账本、wall);`--resume` 按 (ordering, position, arm) 去重。
- **H1–H3 落账**:每次 delayed/重遇/评价面对 Active 或 Draft 的解析集,记 treated_prev/now、NEW/CONT/LEFT、
  翻号、离开原因(特征退出谓词)。
- **事后 0 LLM 脚本**(单独文件,不在 runner 内):`audit_hec1_best_safe_global.py`(每单元评价面全局评估
  冻结菜单、过预算取最优、无则 identity;advantage 表)、`audit_hec1_readout.py`(曲线、符号检验、三分账、
  生命周期表、H1–H3 一致性、Slow vs ScopeFit)。均走 oracle 墙,读 `scoring_ledger`,不读 bank。
- **`--smoke`(0 LLM,交付门)**:(1) 合同冻结断言;(2) 三顺序单元序列构造成功、曝光交集为空;(3) 两个单元上
  identity 部署与评价面计分跑通(Static 与 A3-frozen 空 K0 形态);(4) W3 三案例回放;(5) W1 空 bank 与合成
  bank 各跑一步;(6) W2 合成 rows 校准;(7) `run_source_line_v3.py --dry-run` 字段回归一致。

---

## 交付与回报

- 提交只用 allowlist(新模块、runner、两个 audit 脚本、聚焦测试、smoke 记录),不用 `git add -A`;
  不触碰另一执行线的未提交文件以外的历史工件。
- 回报:(1) 文件清单与行数;(2) smoke 七项输出;(3) 聚焦测试通过数;(4) 对 `methods/ttha/*` 是否有任何改动
  (应为"无";若有,列明并给 lock 处置);(5) 偏离本规格之处及理由;(6) 规格矛盾。
