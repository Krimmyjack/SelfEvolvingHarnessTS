# V1 Sequential Validation Plan（总任务书执行）

执行日期：2026-08-10。三角色：根 Agent（统筹）+ 目标判断者（Gate/Claim 定稿）+ 审查者（子代理，代码/证据审查）+ 推进者（实现运行）。纪律：只验证一个主要因果假设/阶段；不新增 Schema/SHA/Receipt/Ledger/Gate/测试平台；不清理历史 SHA；不同时改多个 Surface；LLM 不批准自己的修改；所有 Patch 经确定性 Runtime/Support/delayed 验证；Target outcome 不进 Agent/LLM 输入；Runner 不按 outcome 手工指定赢家；不消费 virgin 搜正控。

执行范围：P0 → P1（如需要）→ P2 → 最多一个 P3 分支。P4/P5 只排期不执行。

---

## P0：现有 Slow 正控证据去重（2026-08-10 已核实）

**结论：REMOVE_A 型 → 进入 P1。**

证据（以报告 JSON + frozen program 为准，不信文字总结）：

- `artifacts/functional/e2/w1_real_slow_agent_positive_control_report.json`
- verdict = `REAL_SLOW_AGENT_PATCH_PASS`；llm_calls = 1
- manifest.frozen_program = `[{"op": "outlier_iqr", "params": {}}]` —— **1 步**
- 反事实表：A_only(impute_ssm)=−0.15432，B_only(outlier_iqr)=+0.04386，A_then_B=−0.10249
- 结构含义：LLM 从反事实表推出"删 A 留 B"，Skill 是 1-step B-only —— 即 A→B 变成 B（**删除有害步骤**），**不是** A→B 变成 A→C（单步替换）

P0 决策：`ALREADY_COVERED_NO_RERUN` **不成立** → P1 需要真实 B→C 替换正控。

---

## P1 冻结设计：REAL_SLOW_AGENT_REPLACE_STEP_POSITIVE_CONTROL

### 唯一假设

> 在一个已确认存在 delayed-stable replacement headroom 的 Program defect 上，真实 Slow Agent 能自主选择一个单步替换（A→B → A→C），形成可执行 Target-local Skill，并由正常 Fast 入口实际采用。

### 案例结构（数值扫描确认，非文字总结）

- incumbent A→B：support gain < −MATERIAL
- A-only：< +MATERIAL（近零或不能形成正向 Skill）
- B-only：< +MATERIAL（删除 B 最多回到 identity，B 本身不显著正——否则该换 A 而非 B）
- 存在 C1 ∈ 合法 DSL（≠ A,B）：gain(A→C1) ≥ +MATERIAL 且 delayed(A→C1) ≥ −MATERIAL（delayed-stable headroom）
- C2：冻结的对照候选（合法 DSL；语义上同族但无 headroom）

### 信息墙（LLM 输入只含）

- incumbent frozen Workflow（A→B 步骤 + 参数）
- 当前公开 TS Context（window_context + bound_period）
- incumbent Support/delayed failure 数值（gain_AB、delayed_gain_AB）
- 可编辑 step index（=1，第二个 step）
- ≤2 replacement operator contracts（op + 参数 defaults + category/stage/tags）
- Task/Consumer objective（forecast，training_windows_only，cohort Ridge sMASE）
- H0 verifier 与 feedback budget（1 正式调用 + ≤1 已有 schema retry）

**LLM 不得看到**：A-only/B-only 反事实、replacement gain（A→C 数值）、winner 标签、future、A→C delayed。

### Patch 落地

不新增 REPLACE_STEP Schema。沿用 `skill_library.entries/{skill_id}` ADD。新 Skill 的 frozen Program 必须体现两步替换（steps[0]==A，steps[1]∈{C1,C2}）。确定性 Harness 只补齐：surface 模板实例化、surface required dependency SHAs、schema 格式。**不得补齐**：替换位置、replacement operator、Program steps、Scope/Risk 语义。

### 执行链

FailurePatternCard → TTHASlowAgent.propose_edit（真实 core/backend）→ structural preflight（2-step 结构检查）→ EditController.apply_to_fork（SKILL_LIBRARY_GAP）→ compile → Support replay @origin → delayed @origin+HORIZON → Target-local Skill → 正常 TTHAMethod.prepare（SealedProbeBackend explore）→ chosen=cand_skill_* → 沿 PreparationResult.program 执行 → remove-skill 对照。

### PASS 条件（10 条全成立）

1. LLM 选择明确 step replacement（2-step frozen program）
2. 不是删除步骤（len==2）
3. replacement steps 来自 LLM（解析自 manifest body，非脚本注入）
4. Controller 真实落地（apply_to_fork 无异常）
5. compiler/verifier 通过
6. Support replay @origin ≥ +MATERIAL
7. delayed @origin+HORIZON ≥ −MATERIAL
8. 正常 Fast 入口实际选择 Skill（chosen == cand_skill_*）
9. 实际执行 Program 与冻结 replacement steps 一致
10. remove Skill 后行动变化（chosen_ctrl != cand_skill_*）

### Verdict 档位（预注册）

`REAL_SLOW_AGENT_REPLACE_STEP_PASS` / `INFEASIBLE_NO_TRUE_REPLACEMENT_CONTROL`（扫描空，停止不消费 virgin）/ `NO_EXECUTABLE_MANIFEST` / `WRONG_REPLACEMENT_SELECTED`（结构合格但选 C2）/ `SUPPORT_REPLAY_REJECTED` / `DELAYED_REJECTED` / `FAST_ADOPTION_FAILED` / `REMOVAL_CONTROL_NO_FLIP`

Binding/Controller 代码错误：允许一次最小修复一次重跑。replacement 选择错误或 delayed 翻负：诚实记录停止，不调 Prompt/Pattern/候选。只有 PASS 进 P2。

### 扫描结果与案例冻结（2026-08-10）

扫描（GEFCom origin 904，双向枚举 182 组合）产出 18 个满足全部结构条件的案例（报告 `artifacts/functional/e2/scan_v1_replace_step_cases.json`，所有数值来自真实 ScopeExecutor evaluate）。P1 只需一个案例，冻结 **case 0**（与任务书"固定 denoise 前缀"模式对应）：

```
domain = gefcom2012_load（已暴露；P0 正控域）
origin = 904；delayed = 952
incumbent：denoise_median → outlier_iqr
  gain_AB = −0.13021（NEGATIVE）；delayed_AB = −0.06126
  A-only = 0.0（denoise 前缀近零 no-op）
  B-only = −0.13021（B 单步本身有害；删除 B 回 identity）
C1 = winsorize（同族 outlier）：support +0.40001 / delayed +0.25721（双正）
C2 = outlier_mad（同族对照）：denoise_median→outlier_mad = −0.06461（无 headroom）
```

说明：结构上 B 单步负 + A 近零 —— 本案例语义是"第二个 step 有害、替换为 C"。与 P0（REMOVE_A 删 A 留 B）的差异：P1 要求**保持 A、替换 B 为两步 program**，且 LLM 看不到单算子反事实，只能从 failure 数值 + 候选契约语义选 C。

<!-- P1_CASE_FROZEN -->

---

## P2 冻结设计：TRUE_NORMAL_ENTRY_AGENTIC_ONLINE_PILOT

### 唯一假设

> 在真实正常入口中，Agent 自主提出的 Workflow 经 Support、Episode 和 delayed 写回后，能够改变后续正常入口的 Harness 行为，或明确暴露第一个真实阻塞面。

### 数据

- 新 certified-virgin cohort：traffic offset=240（offset=40/120 已消费；registry 固定顺序 = series_uid 排序后跳过已用段）
- 3 轮在线：(648,696), (744,792), (840,888)；每轮 ≤2 次 prepare（→ ≤2 Support receipts）
- 运行前冻结：cohort、origins、模型（gpt-5.6-luna temp=0）、预算、Harness。outcome 不进入 Agent/LLM 输入。

### 运行方式（真实组件，无 runner 代替）

- 真实 `TTHAMethod.prepare` → 真实 `TTHAFastAgent` inspect/propose（确定性）/select（真实 LLMSelectBackend）
- Agent 自主 Typed Workflow（runner 不固定两步组合、不调 `_combos()`、不手工选算子）
- 每 Support 后立即写 Episode（tll.write_target_episode）→ delayed 到达后原位更新（update_delayed_status）
- 同一生命周期：探索状态（_explored/_deprioritized）跨轮继承、Memory 累积经 experience_episodes 进下一轮 prepare、同一 LLM client 计数
- 运行中禁止 Slow Path 修改；轨迹结束后才定位 first fault；abstain/reject 如实记录

### 因果检查（keep vs remove last update）

本轮正式 prepare 之前，用 memory[:-1] + 相同探索状态/Context/预算/模型做只观察的对照 prepare；比较 pool/chosen/abstention 是否回退。真实 LLM 不稳定无法归因 → `INCONCLUSIVE_LLM_VARIANCE`，不靠投票制造结论。

### 每轮记录项

可见 Context；Memory/Skill References（episode_ids + relations）；Agent proposal（pool）；chosen candidate；Program steps；verifier 结果；Support gain；Episode relation/status；delayed gain；Skill 状态；下一轮 pool/order/action；LLM 调用次数；Support receipts 数量。

### Verdict（预注册）

`NORMAL_ONLINE_ADAPTATION_PASS`（行为变化 + remove 对照回退 + 无 harm）/ `MEMORY_WRITTEN_NOT_USED` / `AGENT_PROGRAM_SUPPLY_STALLED`（全 abstain）/ `NO_ACTIONABLE_FEEDBACK` / `NEGATIVE_TRANSFER`（行为变化归因 Memory 但有 harm）/ `INCONCLUSIVE_LLM_VARIANCE` / `NO_ELIGIBLE_VIRGIN_COHORT`

P2 是 first-fault discovery pilot，不宣称 A5/A3 或自然 Slow Update 已成立。

---

## P3 分支判定（P2 后，最多一个）

P3-A 自然 replacement（fault 指向 Program + 合法 replacement headroom）/ P3-B Program-effect Observation / P3-C Scope / P3-D Risk / P3-E Memory / P3-F Control。只由 P2 第一个真实 fault 触发；无明确触发 → 报告并停止。B 单步负且无 headroom → 结论为单算子 Risk（Fast Path 降级），不强行调用 Slow Agent。

---

## 审查记录（角色 B，2026-08-10）

P1 runner 与扫描脚本预审（信息墙/真实组件/承重/verdict 映射）：

- **BLOCKER（已修）**：checks 混入 case dict / recheck float / llm_calls int → `all(v is True)` 恒 False → `REAL_SLOW_AGENT_REPLACE_STEP_PASS` 永不可达。修复：PASS 判定改为承重布尔键白名单（10 条件映射键）。
- **MAJOR 2（已修）**：`REMOVAL_CONTROL_NO_FLIP` 死码（并入 FAST_ADOPTION_FAILED）。修复：独立档位——adopted 但 removal 未翻转时发。
- **MAJOR 3（已修）**：CASE 冻结断言不完整（delayed=None → gateway 暴露全序列含 future 给 Slow Agent）。修复：12 字段全断言。
- **MAJOR 4（已修）**：扫描只枚举单方向组合 → 可能假 INFEASIBLE。修复：双向枚举 (a,b)+(b,a)。
- MINOR（已处理）：死窗口 origin（GEFCom 976/1000、traffic 936 越界）→ 有效 origin {904,928,952}/{648,744,840}；delayed_AB 扫描补测（信息墙 failure 数值）；runner 复测补 A/B-only 与 C2 无-headroom 属性；预算硬停记 checks。
- 审查确认 PASS：信息墙（无 A/B-only 反事实、无 replacement gain、无 winner、无 future）、真实组件链、runner 不替 LLM 决定 Workflow、10 条件全覆盖。

---

## P1 运行结果（2026-08-10）

**verdict = `WRONG_REPLACEMENT_SELECTED`**（预注册档位；报告 `artifacts/functional/e2/w1_real_slow_agent_replace_step_report.json`）

- 案例承重复测全过：AB=−0.13021、A_only=0.0、B_only=−0.13021、AC1=+0.40001/delayed+0.25721、AC2=−0.06461（无 headroom）、`case_headroom_confirmed=true`、`c2_no_headroom_confirmed=true`
- LLM 产出结构合格 2 步替换：`denoise_median → outlier_mad`（保持 A、替换 B——**替换契约理解正确**）
- 但选择 **C2（outlier_mad）** 而非 C1（winsorize）：在信息墙限制下（无 A/B-only 反事实、无 replacement gain，只有 incumbent failure 数值 + 候选契约语义），LLM **无法从语义层面区分同族候选**（winsorize 与 outlier_mad 同为 outlier 类、schema 相似）
- llm_calls=2（1 正式 + 1 已有 schema retry，预算内）

对比证据链：P0（给反事实数值表）→ LLM 正确推出"删 A 留 B"（REMOVE_A PASS）；P1（不给反事实，仅 failure + 语义契约）→ 选错替换。**结论：当前 Slow Agent 的归因/修改决策依赖数值反事实证据（grounded evidence）；无 grounded evidence 时不能从语义层面做出正确的单步替换选择。** 这直接约束自然 Slow Path 的可行性——自然失败输入若不携带反事实分解，LLM 的替换决策不可靠（指向 P3-B Program-effect grounding 的触发条件，但本轮 Gate 未过不执行）。

按任务书纪律：replacement 选择错误 → 诚实记录并停止；不调 Prompt/Pattern/候选；**P1 不 PASS → P2 不启动**。

---

## P1.5 冻结设计：BOUNDED_SLOW_REPLACEMENT_RUNTIME_SELECTION（用户裁决 2026-08-10）

优先级调整：**Program-effect grounding 降为后续效率优化**（仅当两候选验证成本不可接受或需减少 Support receipts 时再研究）。新实验把选择权交给确定性 Runtime，LLM 只做有界候选 supplier。

### 唯一假设

> Slow Agent 能在两个有界 replacement 提案内覆盖至少一个有效候选，Runtime 能用两个 Support receipts 选择有效修改，并经 delayed、Skill 和正常入口完成更新。

### 案例（复用 P1 development case，不泄露 headroom）

- GEFCom 904：incumbent `denoise_median→outlier_iqr` = −0.13021（NEGATIVE）；A-only=0.0；B-only=−0.13021
- 候选池（冻结，LLM 可见契约）：C1=winsorize（实际 +0.400/+0.257 双正，**不告知**）、C2=outlier_mad（实际 −0.0646 无 headroom，**不告知**）

### 流程

1. **Slow Agent 产生两个不同候选**（复用单 Manifest 接口，不新增 Schema）
   - 调用 1：提出第一个 B→C replacement（同 P1 信息墙：incumbent + Context + failure 数值 + 可编辑 step index + 两候选契约 + objective + budget）
   - 调用 2：告知"候选 X 已被提出，请提出另一个不同 replacement"——只提供已提案算子名，**不提供其 gain/验证信息**
   - 两候选必须 LLM 产生、合法 2-step Typed Program（steps[0]==A、steps[1]∈{C1,C2}）、只替换同一 step
   - 重复同一候选 → `CANDIDATE_DIVERSITY_FAILED`
2. **Runtime 验证两个候选**：每候选 compiler/verifier + Support receipt（计真实 Target budget）。冻结选择规则：①拒绝非法/无效 ②gain ≥ MATERIAL 才接纳 ③双正 → Support gain 高者 ④相同按 canonical ID ⑤全非正 → abstain。**Runtime 选择，不由 LLM 自评。**
3. **写反馈**：两合法候选都写 Experience Episode；未被选择者仅 EPISODE_ONLY 不形成 Skill；赢家 LOCAL_DRAFT → 冻结赢家 Program → delayed 只开赢家 → delayed ≥ M → LOCAL_ACTIVE；delayed < M（翻负/冲突）→ RESTRICTED/CONFLICT 不得接纳 → `DELAYED_REJECTED`
4. **正常入口验证**：Skill 写入（ADD skill_library.entries/{skill_id}，frozen program=赢家 2-step）→ 正常 TTHAMethod.prepare → pool 含 Skill → chosen=cand_skill_* → 沿 PreparationResult.program 执行 → remove-skill 后行动回退

### Verdict（预注册）

`BOUNDED_SLOW_REPLACEMENT_PASS` / `CANDIDATE_DIVERSITY_FAILED` / `NO_VALID_REPLACEMENT_SUPPLIED` / `SUPPORT_SELECTION_FAILED` / `DELAYED_REJECTED` / `FAST_ADOPTION_FAILED` / `REMOVAL_CONTROL_NO_FLIP`

### 通过后能说 / 不能说

能说：Slow Agent 能作为有界 Program candidate supplier，Runtime 在两个 Target Support 预算内完成有效 Harness Update 的选择、验证和采用。
不能说：LLM 能准确预测哪个候选最好；Program-effect Observation 已解决；自然 Slow Path 已成立；跨域自主进化已成立。

### 后续顺序（本实验 PASS 后）

TRUE_NORMAL_ENTRY_AGENTIC_ONLINE_PILOT → 等自然失败 → 自然 Slow Agent 最多两个 replacement → Runtime Support 选择 → delayed 接纳 → fresh 正常入口验证 → matched-budget A5/A3。

---

## P1.5 运行结果（2026-08-10）

**verdict = `BOUNDED_SLOW_REPLACEMENT_PASS`**（报告 `artifacts/functional/e2/w1_bounded_slow_replacement_runtime_selection_report.json`；llm_calls=3 ≤ 预算 4）

- LLM 有界提案（abstain 重试语义：abstain 不消耗候选预算，4 次调用预算内）：调用 1 abstain → 重试提 outlier_mad（无效 C2）→ 调用 2（prior=outlier_mad，只告知算子名）提 winsorize（有效 C1）→ diversity 满足
- Runtime 实测两候选（2 Support receipts 计入真实 budget）：outlier_mad −0.0646 → NEGATIVE Episode（EPISODE_ONLY 不形成 Skill）；winsorize +0.40001 → 赢家（唯一 ≥M）
- delayed 只开赢家：winsorize @952 = +0.25721 ≥ M → LOCAL_ACTIVE（winner episode relation=POSITIVE）
- Skill 写入（apply_to_fork，LLM 原始 manifest + 确定性契约修复）→ 正常入口 @952 chosen=cand_skill_gefcom2012_denoise_median_winsorize、执行 +0.25721、executed_program_matches_frozen=true → remove-skill 对照回退（cand_denoise_median）
- 承重 checks 8/8 true（case_headroom/diversity/llm_calls_le_4/controller_applied/delayed_positive/actual_adoption/executed_matches_frozen/removal_flip）

**可说的 Claim**：Slow Agent 能作为有界 Program candidate supplier，Runtime 在两个 Target Support 预算内完成有效 Harness Update 的选择、验证和采用（LLM 不必准确预测哪个候选最好）。P1（LLM 单候选选错→全链断）vs P1.5（LLM 提两候选含有效→Runtime 实测选择→全链通）构成架构修正的对照证据。
**不可说**：LLM 能准确预测哪个候选最好；Program-effect Observation 已解决；自然 Slow Path 已成立；跨域自主进化已成立。

后续顺序（用户裁决）：PASS → TRUE_NORMAL_ENTRY_AGENTIC_ONLINE_PILOT（P2）→ 等自然失败 → 自然 Slow Agent 最多两个 replacement → Runtime Support 选择 → delayed 接纳 → fresh 正常入口验证 → matched-budget A5/A3。Program-effect Observation 降为后续效率优化（仅当两候选验证成本不可接受或需减少 Support receipts 时再研究）。

---

## P2 运行结果（2026-08-10，两次运行 + 归因审查）

**verdict = `INCONCLUSIVE_LLM_VARIANCE`**（报告 `artifacts/functional/e2/w1_true_normal_entry_agentic_online_pilot_report.json`，含 `verdict_review`）

两次运行（traffic offset=240，3 轮在线，同一 Context/Memory/prompt）：

| 运行 | 结果 |
|---|---|
| 第 1 次 | 3 轮 × 2 次 prepare 全 abstain（llm_calls=6）→ 机械 AGENT_PROGRAM_SUPPLY_STALLED。诊断（diag_p2_select.py）：abstain 是 LLM 基于 Context 的合理决策（"Complete recent coverage and no missing runs indicate no preprocessing is needed"——coverage=1.0、无缺失 → 无需预处理） |
| 第 2 次 | 轮 1：denoise_median（+0.0 Episode ABSTAIN）→ hampel_filter（−0.0364 harm，NEGATIVE）→ 轮 2/3：无候选 abstain（llm_calls=2——LLM 未被调用）→ 机械 NEGATIVE_TRANSFER |

**归因审查（verdict_review）推翻机械 NEGATIVE_TRANSFER**：
1. harm（hampel −0.0364）发生在轮 1 探测（memory 影响前）——预算内探测风险，非 transfer
2. 轮 2/3 abstain 的 ctrl pool=["identity"]（无候选）——机制空转；flip 是假信号（runner 判定已修正：ctrl flip 仅在 pool 非空时算数）
3. 两次运行 LLM 行为不一致（全 abstain vs 有行动）→ 无法归因 → INCONCLUSIVE_LLM_VARIANCE

**暴露的机制 first fault**：确定性 propose 探索迭代不跳过 no-op 过滤候选——`_next_explore_op` 取 OPS_ALL 第一个未探索算子（轮 2/3 = impute_ar 等缺失族），fast_agent 层 `_noop_ops_for_context` 过滤剔除后**不继续推进** → 无候选 → 空转（outlier_iqr/winsorize 等非 no-op 算子轮不到；LLM 决策面被截断、预算浪费）。修法涉及 Control 设计（backend 需感知 no-op 集或 fast_agent 过滤后推进）——留给用户裁决，本轮不修。

**P3 判定**：该 fault 是机制/Control 类（非行为类），不匹配任何 P3 分支触发条件（P3-A/E/F 均不符）。按任务书：报告 first fault，不换 cohort、不搜索自然案例、不扩 Pattern，停止提交。后续顺序（用户裁决）：等自然失败 → 自然 Slow 两候选 → Runtime 选择 → delayed → fresh 入口验证 → matched-budget A5/A3。

---

## FILTER_AWARE_EXPLORATION_ADVANCE_CONTROL（用户批准修复，2026-08-10）

P3-F Control 修正：P2 暴露的 first fault（propose 探索迭代不跳过 no-op 过滤候选 → 空池 + 预算浪费）属于 P3-F Control，用户批准最小修复。

### 最小修复（三处，不改 Prompt/Memory/Risk/Pattern/Support 预算）

1. `wiring.DeterministicStrategyBackend._next_explore_op(eligible=None)`：eligible 参数化（默认 self._operators 向后兼容）；扫描跳过 no-op/ineligible 直到第一个合法且未探索候选，全部耗尽才 None
2. `wiring._eligible_ops(messages)` 静态方法：从 propose 请求消息的 `allowed_operator_contracts` 提取算子名（fast_agent 已按当前 Context 剔除 no-op——`_noop_ops_for_context`）；返回 None=契约未渲染（回退原行为）、空 tuple=真耗尽（abstain）。不读 gain
3. `SealedProbeBackend.complete` + `wiring.complete` 的 propose 分支：`_next_explore_op(self._eligible_ops(request.messages))`

语义：no-op 只在当前 Context 下跳过（不永久 explored）；每次最多扫 inventory 一遍；pending/select-选中才 explored 原语义不变。

### 第一层：零 outcome 机械验收 = FILTER_AWARE_EXPLORATION_CONTROL_PASS

（`run_v1_filter_aware_exploration_acceptance.py`，traffic offset=240 @648；无 executor.evaluate 调用）

- A：无缺失 Context 首个合法候选到达（denoise_median）✓
- B：探索推进（_explored=[denoise,hampel]）→ impute_* 缺失族被跳过、下一 eligible 候选到达（**winsorize**——outlier_iqr 被 verifier 实测排除，eligible 由 verifier 决定，不断言具体算子）✓
- C：全部可行动算子耗尽 → pool=['identity']（正确 abstain）✓
- D：缺失 Context（observed_extra 覆写）+ 同 explored 状态 → impute_linear 可供应（no-op 判定随 Context 变化）✓
- E：全程不读 outcome ✓

### 第二层：development replay = 通过（development regression，不称 fresh PASS）

P2 cohort（traffic offset=240）重放 3 轮（修复后完整重跑；报告 `w1_true_normal_entry_agentic_online_pilot_report.json`）：

| 验收点 | 修复前（P2 第 2 次运行） | 修复后 replay |
|---|---|---|
| R2/R3 空转 | pool=`['identity']`，llm_calls=2（LLM 未被调用） | 每轮 pool=`['identity','cand_denoise_median']`，llm_calls=6（LLM 每轮被调用）✓ |
| 每轮候选推进 | 轮 2/3 无候选 | 轮轮有候选（abstain 不消耗 → 重复提案 denoise 是正确语义）✓ |
| LLM abstain 性质 | 供应空池（机制故障） | selector 决策（rationale 引用 context："recent.coverage" 数据干净）✓ |

机械 verdict=AGENT_PROGRAM_SUPPLY_STALLED（全 abstain），但性质已变：**supply 正常、LLM 基于 Context 稳定 abstain = Agent 安全选择**（用户裁决解释规则）。修复目标达成。

---

## P2-V2 结果（2026-08-10，offset=360 新 virgin cohort，修复后）

**verdict = AGENT_PROGRAM_SUPPLY_STALLED（机械档位）/ interpretation = `agent_safe_abstain`**（报告 `w1_true_normal_entry_agentic_online_pilot_report_v2.json`）

- `supply_ok=True`：3 轮 × 2 次 prepare 每轮 pool=`['identity','cand_denoise_median']`（有候选）、llm_calls=6（LLM 每轮被调用）
- LLM 6 次全 abstain，rationale 一致引用 context（recent.coverage / maximum_missing_run_length 等：数据干净 → 无需预处理）——**稳定 abstain**
- first_fault=None（无行动 → 无自然失败）

按用户裁决解释规则：pool 正常 + LLM 稳定 abstain = **Agent 安全选择，不是 supply stall**。修复（filter-aware exploration）后供应机制正常；这些 traffic 窗口数据质量高 → LLM 合理保守。无自然失败 → bounded Slow replacement 链等待事件驱动触发（等数据质量问题的自然 Context 出现，或用户在更宽窗口/其他域裁定运行）。

---

## CONTEXT_ACTIONABLE_NATURAL_OPERATION_PILOT（2026-08-10，uci offset=40）

用户裁决：不无限等待自然失败、不按 outcome 换 cohort——按**部署可见 Context**（预注册条件：level shift |center_delta|>1 / outlier seasonal_residual>1.5 / missing cov<1）选择下一个 virgin cohort，运行真实 3 轮正常入口。registry 顺序扫描：traffic 400-760 全干净（cov=1.0、sr<0.37）→ uci offset=0 已消费 → **uci offset=40 第一个满足**（三决策点 level shift：center_delta @648=+10.8 / @744=+6.0 / @840=−24.9）。选择全程不读 gain。

**结果：明显可行动 Context 下 LLM 仍 6/6 abstain → `agent_safe_abstain` 机械档位，但按用户分支判定 = Agent selection first fault**（报告 `w1_true_normal_entry_agentic_online_pilot_report_context_actionable_uci.json`）。

诊断（单决策点 dump）：
- select prompt **明确包含** `change.median_robust_center_delta: 10.822133`（level shift 强信号）——LLM 看到了
- 但 rationale evidence_refs 只引用 coverage/missing_run/acf/seasonal_residual/coverage_delta/missing_run_count_delta——**完全忽略 center_delta**
- rationale："complete coverage, no missing runs, strong calendar-period autocorrelation... use identity"
- **first fault = Agent selection 感知盲区**：LLM selector 决策以"缺失/覆盖/自相关"为中心（缺失处理聚焦的历史遗产），level shift / outlier 维度不消费
- 复合因素：单候选供给（每轮只提案探索序第一个 denoise_median；abstain 不消耗 → 探索不推进 → repair_level_shift 轮不到）——即使 LLM 想处理 level shift 池中无对应算子

不触发 Slow 链（无行动失败——无行动）。修复方向（用户裁决后执行）：selector 决策信号扩展（level shift 进入决策）或候选供给多候选化（Control 层）。

---

## LEVEL_SHIFT_PROGRAM_HEADROOM_PREMISE（2026-08-10，零 LLM）

用户裁决（纠正过早定案）：不能判 selector 感知盲区——候选池只有 denoise_median（无 repair_level_shift）、center_delta 大不证明修复有效。正确顺序：先零 LLM headroom 检查。

**verdict = `VERIFIER_REJECTED`**（报告 `w1_level_shift_headroom_premise_report.json`）

- UCI offset=40 三决策点（648/744/840）：identity=0.0、denoise_median=0.0、**repair_level_shift=None（verifier 拒绝）**
- 拒绝原因：`MODIFICATION_FRACTION_EXCEEDED`（repair_level_shift 默认参数修改 2/60 窗口超 H0 max_modified_fraction=0.35）
- 完整逻辑链：repair_level_shift 不可行动（verifier 层事实）→ `_actionable_operators` 排除出候选池 → **候选池只有 denoise_median 是 H0 verifier 的正确约束，非 Supply bug** → LLM 面对 [denoise] 池 + level shift Context → denoise 不匹配 → **abstain 合理**

按用户分支：无 headroom（不可验证）→ LLM abstain 正确行为 → **停止 level-shift selection 方向；不改 Prompt、不扩候选**。候选可用性因果测试（LEVEL_SHIFT_CANDIDATE_AVAILABILITY_TEST）不执行（前提 headroom 不存在）。

**准确结论（用户裁决口径）**：Agent 面对明显 center shift 时只获得不匹配的 denoise 候选并 abstain；repair_level_shift 在当前 H0 verifier 下不可执行——Supply 与 Selector 的分离在本案例中无法继续（无可用 headroom 算子）。

---

## PUBLIC_LEVEL_EXCURSION_BINDING_PREMISE = BOUND_CANDIDATE_DELAYED_STABLE_HEADROOM（2026-08-10）

用户裁决：默认 repair 被 verifier 拒不是"没有 headroom"——算子支持局部区域（region_start/end/estimated_offset），公开特征提取器产生对应字段。验证局部几何绑定（参数全来自公开 Context，不手工指定）。

**结果**（UCI offset=40 三决策点；报告 `w1_public_level_excursion_binding_premise_report.json`）：

| 决策点 | level 区域 | union 区域（绑定参数） | 宽度 | verifier | Support/delayed | 档位 |
|---|---|---|---|---|---|---|
| @648 | [0.606,0.668] | [0.607,0.838] | 0.23 | 过 | −0.024 / +0.001 | VERIFIER_PASS_NO_HEADROOM |
| **@744** | [0.528,0.582] | [0.528,0.730] | 0.20 | 过 | **+0.083 / +0.072** | **STABLE_HEADROOM** |
| @840 | [0.468,0.515] | [0.468,0.988] | 0.52 | — | — | REGION_TOO_WIDE（union 延伸 >0.35） |

- **level-shift family 有 headroom，默认全局模式被 verifier 拦下（45.7% > 0.35）；局部几何绑定工作（@744 双正）**
- 边界效应：绑定参数用公开 mapping 的 union 区域（用户指令）——@648 union 宽于 level 区域 → 修复含非 level 信号 → support 负；@840 union 延伸至 0.988 → 超限。如实记录（不后验挑参数）
- **进入下一实验（用户裁决预定义）：LEVEL_SHIFT_CANDIDATE_AVAILABILITY_TEST——CandidatePool [denoise_median, bound repair_level_shift]，同 Context/Prompt/LLM/预算，不告知 gain → 区分 Supply vs Selector**（Treatment 选 repair → Supply first fault → 只修每轮 ≤2 context-relevant 候选供应；Treatment abstain → Selector 未消费 level-shift 证据 → 优先 Runtime 验证两候选；两次 Treatment 不一致 → INCONCLUSIVE_LLM_VARIANCE）

---

## LEVEL_SHIFT_CANDIDATE_AVAILABILITY_TEST = SELECTOR_SUPPLY_FIRST_FAULT（2026-08-10）

前提：BOUND_CANDIDATE_DELAYED_STABLE_HEADROOM（@744）。同 Context/Prompt/LLM/预算，只改变候选池（不告知 gain）。bound repair 参数全来自公开 Context（post_validator 硬约束：=特征值）。

**结果**（@744；报告 `w1_level_shift_candidate_availability_test_report.json`）：

| 臂 | pool | chosen | rationale |
|---|---|---|---|
| Control [denoise] | identity+denoise | identity | coverage/缺失（denoise 不匹配 level shift） |
| Treatment 1 [denoise, bound repair] | identity+denoise+repair | **cand_repair_level_shift** | **引用 change.median_robust_center_delta / scale_delta** |
| Treatment 2（重复） | 同 | **cand_repair_level_shift**（一致） | 同 |

**结论（用户裁决解释规则命中）**：
- **Selector 能理解 level shift**——repair 在池中时 LLM 两次一致选择，rationale 明确消费 center_delta/scale_delta 证据
- **真正 first fault = Program Supply**——repair_level_shift 默认参数被 `_actionable_operators` verifier 实测排除（45.7% > 0.35）→ 候选池永远只有 denoise_median → LLM 无从选择（abstain 是对"不匹配池"的合理响应）
- 之前"selector 感知盲区"判断被完全推翻（用户裁决预判：Supply 与 Selector 未分离）

**修复方向（用户预定义）**：只修复每轮最多两个 context-relevant candidates 的供应——(a) `_actionable_operators` 对带 public_parameter_bindings 的算子用绑定参数（公开特征值）实测 verifier（bound repair 能过 0.35）→ repair 进候选池；(b) 每轮最多两个候选（max_propose_candidates 装置已验证）。待用户裁决后实施。

---

## 统一 Program Supply 修复 + CONTEXT_BOUND_PROGRAM_SUPPLY_DEVELOPMENT_PASS（2026-08-10）

用户裁决：实施 (a)+(b) 统一修复（不是两个独立方法——共同修复同一 first fault：Program Supply）。

### 修复实施（文件记录）

- `SelfEvolvingHarnessTS/methods/ttha/fast_agent.py`：
  - `_actionable_operators`：带 public_parameter_bindings 的算子用公开 Context 特征值构造绑定参数候选实测 verifier（绑定不完整 → 不可行动，不 fallback 全局模式；不放宽 0.35；不读 outcome）
  - propose_contracts 排序：绑定参数完整的算子前置（参数完整性是 context-relevant 相关性来源——一般规则不硬编码算子名）
- `evaluation/functional/run_v1_sealed_a5_a3.py`：
  - `SealedProbeBackend`：默认 max_propose_candidates=2（每轮最多两个候选）；`_cand(request, op)` 对带 bindings 的算子从 request 解析公开特征构造绑定参数（post_validator 硬约束=特征值）；`_public_features_from(request)` 解析 public_input.features
  - `LLMSelectBackend`：默认 max_propose_candidates=2
- 期间修复：prepare 内条件性局部 import 遮蔽模块级 OPERATOR_METADATA（空 memory 时不执行 → UnboundLocalError）→ 独立别名 `_OP_META`

### Development integration 验收 = CONTEXT_BOUND_PROGRAM_SUPPLY_DEVELOPMENT_PASS

（报告 `w1_context_bound_program_supply_integration_report.json`；UCI offset=40 @744→@792，12/12）

pool=[identity, cand_repair_level_shift, cand_denoise_median]（repair 前置）→ LLM 选 repair → 绑定 Program（参数=特征值）→ verifier 过 → Support +0.0827 / delayed +0.0723 → Episode POSITIVE + Skill 写回 → @792 正常入口 chosen=cand_skill_uci-bound-repair-v1（**Skill 实际采用**）→ 无 future 读取。

不称 fresh/一般化能力。下一步（用户裁决顺序）：CONTEXT_BOUND_NORMAL_ENTRY_FRESH_PILOT（新 virgin、公开 Context 选 cohort、3 轮在线）。

---

## SOURCE_EVIDENCE_EXECUTION_RIGHT_AUDIT（2026-08-10，零 LLM）

用户裁决：保留 A5_NEGATIVE_TRANSFER 为核心里程碑第一个可信负结果，但先审查 Source Evidence 实际权限（不改 budget、不调 Prompt）。

**根因（决定性）**：Runner bug——`_run_round` 写 Episode 时 `op=str(chosen)`（候选 ID 'cand_repair_level_shift'）→ `workflow_signature` 带 `cand_` 前缀 → `resolve_order` 按算子名（'repair_level_shift'）匹配失败 → **Source 与 Target Episode 从未进入 signed 判定/渲染**（渲染空、无 Reference）→ **A5 的 LLM 输入与 A3 完全相同**（Source Experience 零权限——未接入而非权限过强）。验证：算子名写法（无前缀）→ workflow_signature='repair_level_shift' → 成对双负 → **RISK_PRIOR** → 渲染 Reference 3 "carried only negative evidence... weak reference: no context matching. Avoid them unless current evidence contradicts."

**Audit verdict 修正**：A5/A3 差异（pair1 A5 abstain vs A3 Skill）纯属 **LLM 方差**（Memory 从未生效）→ 按用户定义应为 **INCONCLUSIVE_LLM_VARIANCE**（不是 NEGATIVE_TRANSFER——无 Memory 效应可归因）。用户预判的 WEAK_SOURCE_RISK_OVERAUTHORIZED **不成立**（权限从未授予——是更前端的绑定 bug）。

**修复**：`_run_round` 的 `write_target_episode(op=steps[0][0])`（算子名）。重跑 A5/A3（development regression——已暴露 pair）→ Source Experience 将真正渲染（pair1 → ref3 RISK、pair2 → ref1 POSITIVE）→ 观察真实 Source 效应。**修复后任何新 Harness 版本需新 virgin pair 做最终确认（A5/A3 V2）**。

---

## MATCHED_BUDGET A5/A3 修复后重跑 = INCONCLUSIVE_LLM_VARIANCE（development regression，2026-08-10）

修复（Episode op=算子名）后第三次运行（报告含 verdict_review）：

- **pair1**：A5 R1 行动（repair −0.0198，delayed **+0.117** 正）vs A3 全 abstain（0 行动）——A5 更好
- **pair2**：A5/A3 几乎相同（均 repair −0.0027 d=−0.060；A5 多一次 denoise 0.0）
- 判定修正：harm 增加但 delayed utility 也增加（探索成本被收益补偿）→ 非负迁移（A3 零行动时 harm 比较失真）——机械修正后 A5_CONFIRMATION_PASS（pair1 util_better）
- **方差审查**：三次运行 pair1 结果互相矛盾（修复前 A5 差/修复后 A5 好）——LLM 方差主导 → **INCONCLUSIVE_LLM_VARIANCE**（用户原则：不投票、如实报告）

development regression 意义：Source Experience 已真实渲染（pair1 ref3 RISK、pair2 ref1 POSITIVE）——**weak risk 渲染未形成 veto**（A5 仍行动）——用户预判的 WEAK_SOURCE_RISK_OVERAUTHORIZED 未出现（audit 根因是更前端的绑定 bug，已修）。A5/A3 的 Memory 效应在 LLM 方差下无法在单次运行确认——需新 virgin pair（A5/A3 V2）做最终确认。

---

## FRESH A5/A3 V2 = NO_SIGNAL（2026-08-10，3 virgin pair）

用户裁决：作废 A5_NEGATIVE_TRANSFER（绑定 bug 已修），3 个新 virgin pair 每 pair 一次（不投票）。报告 `w1_matched_budget_context_bound_a5_a3_v2_report.json`（含 verdict_review）。

**机械检查 15/15 全过**（绑定修复最终验证）：Source Episode 可被 resolver 匹配 ✓、A5 渲染 Reference ✓、A3 不渲染 ✓（空 Memory 正确）、两臂同构造 ✓、future sealed ✓。（判定 bug 修复：A3_renders_reference=False 是预期正确——mech_ok 白名单排除。）

**结果**：3 个 pair 的 A5/A3 **行为完全相同**（identical）——aggregated 逐项一致（proposal 12/12、harm 1/1、util 0.5896/0.5896、skill 2/2 1/1）。

- pair1（src POSITIVE +0.027）：A5/A3 均 R1 repair −0.0198（d=+0.117）、R2 abstain
- pair2（src POSITIVE +0.025）：A5/A3 均 R1 repair −0.0027、R2 repair +0.113（d=+0.086）
- pair3（src NEGATIVE −0.444）：A5/A3 均 R1 repair +0.458（d=+0.235）→ R2 用 skill（+0.007/d+0.106）×2——**Source NEGATIVE（ref3 "Avoid"）未阻止 LLM 选 repair**（weak risk 无 veto 权——用户预判 WEAK_SOURCE_RISK_OVERAUTHORIZED 彻底不成立）

**结论**：Source Experience 已真实接入渲染（机械检查确认）但**未改变 Target 行为**（候选顺序/选择相同——LLM 在强 Context 信号下选择 repair 不依赖 Reference）。按用户判定：**NO_SIGNAL**（两臂行为完全相同）。

**用户 Gate**：NO_SIGNAL → **换有自然 Program headroom 的数据切片**（需 Source 与 Target 信号可能冲突/Reference 可能改变选择的场景——当前 pair 的 Source/Target 同算子导致 Reference 与 LLM 自然选择一致）。

---

## MEMORY_DECISION_CONTENTION_A5A3 前提扫描 = NO_PREMISE（2026-08-10）

用户裁决：V2 的 NO_SIGNAL 不说明 Memory 无效——缺少"Memory 能改变选择"的空间（Target Context 强指向 repair、Reference 与 Target 信号一致）。contention 实验需要：两个合法且与 Context 相关的候选 + Target Context 不决定选谁 + Source 提供区分信息。

**扫描结论**：
- 竞争条件（repair+winsorize 双可行动 + level 信号）：**已消费 pair 的 Target（uci 80/120/200）本具备**（repair/winsorize/outlier_iqr/outlier_mad/hampel 全可行动）——设计可行
- **剩余 virgin 无竞争 pair**：uci 330（无 repair）、traffic 780+/metr_la 168+（长度边界）、noaa（repair 无 level 信号——假可行动恒等）、covid/nn5（无信号）
- **NO_PREMISE**（用户分支）：当前 registry virgin 数据无法在不看 outcome 下找到两个合法相关候选——**数据/Program Supply 问题，不是 Memory 问题**

**附带发现（诊断价值）**：V2 的候选池（runner 固定 "denoise_median","repair_level_shift"）——denoise 在 uci 无效果被 verifier 剔除 → 池实际只有 repair+identity → **无竞争空间**——V2 NO_SIGNAL 的部分解释（候选池设计限制了 Memory 区分空间，非 Memory 无效证据）。

**下一步（用户分支 NO_PREMISE）**：需要新数据源（level+outlier 竞争共存的 series——下载/生成），或接受"当前数据下 Memory 竞争测试不可行"。

---

## P0 竞争式 plan-only 测试 = LLM_VARIANCE（2026-08-10，用户 Gate P0）

**用户裁决（两 Gate）**：P0 = 已消费 UCI 80/120/200 上的竞争式 plan-only 测试（相同 Context + 相同候选池 [bound repair_level_shift, winsorize, identity] + 相同候选顺序；四档 Memory：M_positive/M_remove/M_negative/M_conflict；只观察 chosen/probe 顺序/abstention；不写 Skill、不算 gain；不读取或重新评价 outcome）。P1（Monash Weather fresh A5/A3）只有 P0 出现明确 MEMORY_ACTION_SIGNAL 才执行。

**装置**（审查者 7/7 PASS，2026-08-10）：
- `run_v1_plan_only_contention_test.py`：PlanOnlyBackend（propose 固定提案 [repair_bound, winsorize]，忽略 ref2/ref3 deprioritize——池成员/顺序是控制变量）+ LLMSelectBackend select（Reference 渲染 + 真实 LLM gpt-5.6-luna temp=0）
- Source Episode 从已暴露报告重建（pair1 repair 双负 −0.360/−0.105 → RISK_PRIOR；pair2 双正 +0.034/+0.014 → POSITIVE_PRIOR；M_conflict = 两条同 op）——零新 outcome
- `fast_agent.prepare` 全流程静态（verify_candidate 无 gain 语义）；零写回；24 次 LLM 调用 = 3 Context × 4 档 × 2 重复 × 1 select

**结果（24 决策，报告 w1_plan_only_contention_report.json）**：

| Context | M_positive | M_remove | M_negative | M_conflict |
|---|---|---|---|---|
| offset80 | repair（2/2）| identity/abstain（2/2）| **identity↔winsorize（1/1 翻转）** | **repair↔identity（1/1 翻转）** |
| offset120 | repair（2/2）| identity/abstain（2/2）| winsorize（2/2）| repair（2/2，判定 POSITIVE_PRIOR）|
| offset200 | identity（2/2）| identity（2/2）| winsorize（2/2）| identity（2/2，池无 repair）|

**关键发现**：
1. **offset120 三档稳定行动差异**（2/2 稳定）：M_positive→repair（rationale 引用 "Reference 1…directs probing it first"）、M_remove→abstain（"No signed experience"）、M_negative→winsorize（"winsorize…should receive the Support probe before the weak-risk repair_level_shift"）——**Reference 渲染明确改变真实 LLM 行动**（Memory-to-action 前提有初步证据，非"无行动效力"）
2. **offset80 两档翻转**（同 prompt temp=0 两次不同：M_negative abstain↔winsorize；M_conflict repair↔identity）——真实 LLM selection variance
3. **offset200 无竞争**：@792 下 repair 不可行动（level offset 仅 +3.3 vs 80/120 的 −11.5/−119.4，绑定参数实测 verifier 拒绝）→ 池 [identity, winsorize] 单候选——数据/Program Supply 限制再次印证
4. **M_conflict 未渲染 ref2**：radius 模式下按 Context 半径判定（pair1 负向证据落查询半径外 → offset80/200=UNKNOWN、offset120=POSITIVE_PRIOR）——CONFLICT 档需要更近 Context 或 weak_reference（成对判定）才呈现 ref2
5. 池一致性：四档 Memory 下池完全相同（pool_consistent_across_memory=True）✓

**判定 = LLM_VARIANCE**（用户分支 3）：存在不稳定档位 → 不能把新数据结果归因给 Memory → **P1（Monash fresh A5/A3）不启动**。但稳定档（offset120）的三档差异说明 Reference 有行动效力——"Memory 改变行动"前提**未被否定，被 variance 污染**。

**下一步选项（待用户裁决）**：
- 量化/记录 LLM selection variance（更多重复建立 baseline 分布）后，若稳定差异在多 Context 复现 → MEMORY_ACTION_SIGNAL → 进入 P1
- 或先修 select 不稳定源（接口/模型/prompt 层）
- 或（数据侧）获取有 level+outlier 竞争的自然数据（P1 的 Monash Weather），届时用多次重复稳定后的装置

---

## BOUNDED_TWO_CANDIDATE_RUNTIME_CONTROL = DEVELOPMENT_PASS（2026-08-10，用户裁决选 B）

**用户裁决**：不再选 A（无限量化方差）或 C（直接跑 Monash）。科学解释拆分：
- **Memory action sensitivity：已观察到**（P0 offset120 三档稳定差异 + rationale 引用 Reference——开发级 MEMORY_ACTION_SIGNAL）；
- **单候选 LLM selection reliability：不足**（offset80 同 prompt 翻转；provider 无 seed 能力——runtime/agent_backend.py `_CAPABILITY_FLAGS.provider_seed=False`，temp=0 无效力）→ 继续增加重复只能估计方差，不能让 Harness 更可靠。

**唯一改变面 = Control**：LLM 不再单独决定唯一候选并删除其余；LLM 参与候选生成/排序，Runtime 在固定预算内验证最多两个候选，用真实 Support 选择赢家。不改 Memory/Prompt/Observation/Program/Risk。规则：LLM chosen 非 identity 优先探测；LLM abstain 不删除池中合法候选；剩余预算探测第二候选；signed positive 提高顺序；weak negative/conflict 只降级不 veto；Runtime 用真实 Support 选赢家；每轮预算上限 2；每个实际 probe 写 Episode；只让 Support 赢家进 Skill；delayed 决定保留/降级。后续顺序：双候选 Runtime Control 开发验收 → Monash 只读 feasibility → fresh matched-budget A5/A3 → 自然失败时复用同一双候选机制接 Slow Path。

**开发验收（run_v1_bounded_two_candidate_runtime_control.py，零新增 LLM 调用）**：用 P0 已记录输出（offset80）三条 replay——A=M_positive rep0（chosen=repair）、B=M_negative rep1（chosen=winsorize）、C=M_remove rep0（chosen=identity）——池相同 [identity, repair, winsorize]。**结果：三条不同 LLM 输出收敛同一 Runtime 赢家 winsorize**（@792 实测 repair −0.0198 负、winsorize +0.0376 正向、delayed +0.104 正向）：
- A：M_positive 引导 LLM 选 repair（实测负向）→ Runtime 拒绝 → 探测 winsorize → 赢家 winsorize
- B：LLM 选 winsorize → 早停（1 probe）
- C：LLM abstain → 不 veto → 探测 repair 负 → winsorize 正向 → 赢家 winsorize
- delayed 降级路径演示：合成 delayed 翻负（−0.1）→ update_delayed_status → RESTRICTED/CONFLICT、Skill 不写
- 附带：offset120 @792 双负（repair −0.0027/winsorize −0.041）——三条一致 abstain（演示"Memory 引导选负向候选时 Runtime 拒绝接受"）
- 审查者 8/8 PASS（replay 硬断言、Runtime 规则、赢家一致性、预算 2、降级状态机、零 LLM/零落盘/零 virgin、诚实措辞）；非阻断注记：check 4 all-None 偏松、signed 排序无独立对照、Episode 为记录级非记忆级

**结论（development，不承重 A5/A3）**：Runtime 实测接管最终选择后，LLM 单点随机翻转不再决定结果——"Source Memory 是否让正向候选更早进入两次 Support 预算"成为可测问题。

---

## MONASH_WEATHER_FEASIBILITY = PASS（2026-08-10，用户 Gate P1 feasibility）

**数据源**：Monash Forecasting Repository（forecastingdata.org）Weather：3,010 条澳大利亚气象站日序列（rain/mintemp/maxtemp/solar），min 1,332（官方），Zenodo record 4654822（weather_dataset.zip 38.8MB，md5 57155594af0883ccd5e63a5948976796，CC-BY-4.0，.tsf 格式）。下载 + md5 校验 + .tsf 解析 + npz 缓存（data/monash_weather_v1/）。

**feasibility 扫描（run_v1_monash_feasibility.py，只读公开 Context + 静态 verifier，不读任何 gain）**：3,010 条全扫（批级 12 批 + partial JSONL + checkpoint 恢复），决策点 origin=792（与 uci 装置同窗口），周期自动检测（extract_public_features 的 _dominant_acf_period——日数据天然适配）。

**aggregate 结果**：
- length_ok 3010/3010（min 1,332 ≥ 984 完整窗口）
- level 信号合理（1.0 < level_excursion_score < 1e6）：1,761（58.5%）
- repair_actionable：1,183（39.3%）；winsorize_actionable：3,010（100%——天气数据普遍有极端值）；outlier_iqr_actionable：2,924（97.1%）
- **竞争条件（level + repair + winsorize/outlier_iqr 双可行动）：917（30.5%）**

**冻结 roster**：按 series_name 排序取前 120 条竞争序列（10 个 20 支 cohort 容量；pair 分配在 fresh 脚本定）——artifacts/functional/e2/w1_monash_feasibility_roster.jsonl。

**Verdict = PASS**（competition ≥ 120；否则 STOP_LOW_PREVALENCE 再查 Solar）。对比 uci 的 NO_PREMISE（剩余 virgin 无竞争 pair）——Monash 竞争 prevalence 30.5%，数据侧前提成立。审查者 6/6 PASS（零 gain、预注册竞争条件、字典序冻结、静态 verifier、数据自洽）+ median 口径修复（finalize 固化为脚本）。

---

## FRESH MATCHED-BUDGET A5/A3（Monash 分层平衡）= NO_SIGNAL（2026-08-10，承重）

**用户前置修正（三个，全部零 outcome）**：
1. **多决策点 Scope 检查**：917 条竞争序列逐条 `ScopeExecutor.verify()` @600/@792/@888 × repair/winsorize——**231/917 通过（25%）**（失败全为 repair 窗口 verifier 拒绝；winsorize 三决策点全过）
2. **日频配置冻结**：daily_regular、period=7（周，公开规则，非 hourly/24）、anchors=(312..852) 同 uci 集合、CONTEXT_LENGTH=192/HORIZON=48；**分层平衡版 B 冻结**（用户裁决：mintemp 前 60 + maxtemp 前 60 → 6 cohort × 10+10，角色 train 6/2 per type、support 2/2、query 2/2；断言 1-3 全过 FROZEN_OK）
3. **Runner 消费 trace.candidate_program_steps**（禁止按算子名重建 Workflow——探测与 Skill 写回均用真实 trace 参数）

**运行（每 pair 一次，15 次 LLM select）**：pair1 C0→C1、pair2 C2→C3、pair3 C4→C5；Source @600（Runtime 控制，全保留不挑正例）→ Target A5/A3 双臂 R1 @792/R2 @888。

**结果（三 pair A5/A3 逐轮 identical）**：
- pair1：Source winsorize +0.019；R1 探测序同 [repair, winsorize]（A5 chosen=repair vs A3 chosen=identity——**Memory 影响 LLM 选择但被 Runtime 吸收**）→ 赢家 winsorize（+0.068/d+0.015 POSITIVE）→ Skill；R2 skill（+0.129/d+0.114 POSITIVE）
- pair2：Source winsorize +0.034；R1 赢家 winsorize（+0.170/d+0.023）→ Skill；R2 skill（+0.070/**d −0.014 CONFLICT**——delayed 翻负降级演示）
- pair3：Source winsorize +0.032；R1 双负（repair −0.0048/winsorize −0.0057）→ **Runtime abstain**（拒绝负向）；R2 repair（+0.0086/d+0.069 POSITIVE）
- 指标逐字段相同：first_pos、harm（1/0.067、1/0.040、1/0.006）、delayed_utility、winner、skill——**全部 identical**

**判定 = NO_SIGNAL**（审查者装置合规 PASS；三 pair 两臂实测指标完全相同，任何 A5 正效应档位都不成立）。**精确解释**（审查者建议 + 写入报告）：Source Memory 未在 Runtime 控制的探测结果上产生可测差异（探测序与 gain 确定且未分叉）；**非**"Memory 对 LLM 选择无影响"——pair1 R1 chosen 实际不同（A3=identity vs A5=repair），但 BOUNDED_TWO_CANDIDATE_RUNTIME_CONTROL 将 LLM 选择与实测结果解耦，差异未传导到最终指标。

**结果口径（限定）**：组成平衡的 Monash Weather 温度变量 cohort；不扩展为全部 Weather/跨变量类型/Shared Capability。

**下一步选项（待用户裁决）**：NO_SIGNAL 在此装置下说明 Memory 渲染的 LLM 选择差异不改变最终形成路径——需要"Memory 能改变探测结果"的空间（如候选池 >2 且候选间无确定性优劣、或 Source 与 Target 信号方向冲突的场景），或接受当前装置下 Memory 的价值在于（尚未被验证的）"更早进入预算"语义。

---

## 三候选预算二最终验证 = INCONCLUSIVE_PROTOCOL_FAILURE（2026-08-11，Memory family 收口）

**用户裁决（A*）**：候选池 [bound repair, winsorize, outlier_iqr]、Target 预算 2（Runtime 只验证排序前二，不因失败打开第三）、Source 固定序评估三候选全写 Episode、新 Monash virgin cohort（分层平衡、零 outcome 冻结）。**唯一授权 Harness 行为修改**：fast_propose_v1.json maxItems 2→3 + candidate_policy.json agent_program_slots 2→3、total_k 3→4（identity_slots=1）——行为修改、三个一致性字段。零 outcome smoke 门（三候选过 Schema/池 identity+3 Program/无截断/绑定一致——6/6 过）。报告真实性修复：prepare FAILED → INCONCLUSIVE_PROTOCOL_FAILURE（不允许其他 verdict）+ 记录 receipt.error。

**实施链**：
- 三候选 precheck 全扫（2,890 条 eligible × 3 决策点 × 3 候选 verify）→ 序列级通过 583 → scale-floor 预筛剔除 55 → cohort 级贪心验证（全窗口 verify + 失败序列替换）→ 最终 baseline 检查 6 cohort × 3 origin 全过 → **FROZEN_OK**（120 条：60 mintemp + 60 maxtemp，6 cohort 10+10，角色 6/2/2；consumed = 两候选 120 + 旧三候选 120（consumed_3cand_old.json 重建）——与全部已消费 0 交集）
- 装置缺陷修复：①schema 容量（A* 授权）；②prepare FAILED 仍探测（早退）；③signed positive 提高顺序未实现（ThreeCandBackend 存 _prioritized ref1 → rank 0）；④**大小写 bug（审查者承重发现）**：`"FAILED"` vs `PreparationStatus.FAILED.value="failed"`——失败守卫与 INCONCLUSIVE 判定全部失效
- provider 偶发故障：LLM 返回 choices=None → LLMSelectBackend `resp.choices[0]` TypeError → prepare FAILED（pair2 A3 R2、pair3 A5 R2）——SDK 层对异常响应的偶发 TypeError

**结果**：三 pair 跑完（每 pair 一次）。数据层面：A5/A3 的 chosen、探测集合、探测序、赢家、指标（first_pos/harm/delayed/winner/skill）逐轮逐字段相同（候选 > 预算时排序后前二被验证，第三候选从未进入预算——pair1 Source outlier_iqr +0.139 最高但 Target R1 截断在外）。**但 2 轮 prepare 失败（pair2 A3 R2、pair3 A5 R2）被大小写 bug 静默吞掉（失败轮仍探测写 Episode）→ 修复后重判 = INCONCLUSIVE_PROTOCOL_FAILURE**（按用户裁决，不允许 NO_SIGNAL）。

**审查者**：装置合规（a/b/c/f/g PASS）；承重缺陷（d/e：大小写 bug 使守卫与聚合失效；i：merged interpretation 硬编码 chosen 差异断言与数据矛盾——已修复为数据驱动陈述）。

**下一步选项（待用户裁决）**：
- 重跑三 pair（每 pair 第二次——provider 偶发故障重试先例：pair2 API 故障重试被接受）→ 若干净 NO_SIGNAL/NEGATIVE → **关闭 Source Memory Transfer family**（不再调 renderer/radius/Prompt），承认 bounded Runtime search 比现有 Source Memory 更有用，转向正常入口的 Program-only Slow Path 自动触发与更新闭环
- 或按 INCONCLUSIVE 收口（三候选装置已全部就绪，重跑只需 LLM 调用）

---

## P0 FINAL = FRESH_3CAND_A5A3_INFEASIBLE_DATA_EXHAUSTION（2026-08-11，用户裁决选 A）

**准确落账**（用户裁决）：不是"Memory 无效"——**Source Memory budget benefit = NOT ESTABLISHED**。当前证据只支持：两候选 fresh NO_SIGNAL；三候选内容无差异但因两轮协议失败只能 INCONCLUSIVE_PROTOCOL_FAILURE；第四批 fresh 温度 cohort 数据不足（排除已消费 360 条后 mintemp 仅 59/60、maxtemp 仅 20/60——三候选多决策点通过率低 + 三批消费耗尽排序池）→ **FRESH_3CAND_A5A3_INFEASIBLE_DATA_EXHAUSTION**。停止 Monash 温度 Memory 线；不用单 pair/rain-solar 混合/新数据搜索救结果。Memory 线"暂停、未建立"，等 P3 自然 Slow Path 建立后再获取新 Dataset（一次数据投入同时服务三目标：cross-domain A5/A3 + 自然失败 Slow Path + 数据处理效果）。

---

## P3.1-A 降级 = RUNNER_ORCHESTRATED_MECHANISM_OBSERVED（2026-08-11，用户修正）

**用户只读核对承重修正**：P3.1-A 的"自动触发"实为 Runner 编排（_feedback_round/_auto_slow_update 都在 evaluation/functional/run_v1_method_level_auto_slow_wiring.py，Runner 自己判断 NEGATIVE/CONFLICT、自己调 propose_edit/Controller）——"Runner 不手工调用"是所有权口径错误。**真实结论：RUNNER_ORCHESTRATED_MECHANISM_OBSERVED（部分观察到）**——组件路径（feedback → Patch → replay → Skill → 下一轮采用 → removal）可编排，但**方法层自动触发未实现**。且当前权威落盘报告均为 ACTION_UNAVAILABLE（A 的 PASS 报告曾被 --live 覆盖，无独立可审计工件）。报告真实性 bug 已修复：live 模式不再写 zero_live_llm=True、A/B note/文件/verdict 分离（w1_method_level_auto_slow_wiring_report.json vs _live_report.json）。

## P3.1-A2 = METHOD_OWNED_SLOW_UPDATE_WIRING_PASS（2026-08-11，方法层所有权）

**用户裁决（只改一个行为）**：触发判断和 Slow Update 所有权移入方法层——TTHAMethod.handle_feedback（methods/ttha/method.py）：Runner 只提交 Episode/feedback（append_experience_episode + handle_feedback）；Runner 不读 relation、不调 propose_edit/Controller/Slow 链；method 内部完成 material NEGATIVE/CONFLICT 判定、Slow Agent 调用、_resolve_apply_manifest（surface 模板 + dependency SHA）、apply_to_fork、replay/delayed（evaluator 回调）、self._snapshot 更新；下一轮 prepare() 自动读更新后 snapshot；零 live LLM（SlowReplayBackend）；removal 恢复。**验收 1-8 全过**（r1 outlier_iqr −0.130 → handle_feedback 触发 → applied → snapshot 更新 → r2 同实例池含 cand_skill_ → removal h0 无 skill 恢复）。**Verdict = METHOD_OWNED_SLOW_UPDATE_WIRING_PASS**（审查者确认待补）。仍只证明方法层接线，不证明自然自进化。

## P3.1-B = ACTION_UNAVAILABLE；first fault = MANIFEST_TO_TYPED_PROGRAM_BINDING_GAP（2026-08-11，用户修正）

真实 Slow Agent（1 次调用）输出 manifest 通过 schema（ADD/surface/skill-entry/1）但 **frozen steps 无法从自由文本 body 解析**（"Frozen program steps:" marker 不是 skill-entry/1 schema 要求——Fast Agent 靠文本搜索依赖非契约格式）。**准确 first fault（用户修正）**：Slow Agent 的语义 EditManifest 与 Runtime 所需的 Typed Program 之间缺少机器可验证的 **Credit-to-Update Binding**——不是"LLM 格式不稳定"，不靠重试/Prompt 解决。下一步 **P3.1-B2**（Typed Patch Binding）：FailurePatternCard 提供 ≤2 Runtime-owned Typed Patch IDs → Slow Agent 选 Patch ID 或 ABSTAIN → Runtime 按 Patch ID 取冻结 steps → Controller 写 Skill；未知 Patch ID → ACTION_UNAVAILABLE；LLM 不再手写 frozen steps。

---

## P3.1-A2+B2 = METHOD_OWNED_SLOW_UPDATE_WIRING_PASS（2026-08-11，方法层 + Typed Patch Binding）

**用户裁决**：A2（触发与 Slow Update 所有权移入方法层）+ B2（Typed Patch Binding——FailurePatternCard 提供 ≤2 Runtime-owned Typed Patch IDs → Slow Agent 选 Patch ID 或 ABSTAIN → Runtime 按 ID 取冻结 steps → Controller 写 Skill；未知 ID → ACTION_UNAVAILABLE；LLM 不再手写 frozen steps；最小结构化字段变更：EditManifest.patch_id 可选 + slow_edit_v1 可选 property——无 skill-entry/2、无迁移）。

**A2（方法层所有权）**：TTHAMethod.handle_feedback（method.py）——Runner 只提交 Episode（append_experience_episode + handle_feedback），不读 relation、不调 propose_edit/Controller；method 内完成 material NEGATIVE/CONFLICT 判定、propose_edit、_resolve_apply_manifest（surface 模板 + dependency SHA）、apply_to_fork、replay/delayed（evaluator 回调）、self._snapshot 更新；下一轮 prepare 自动读新 snapshot。验收 1-8 全过。

**B2（Typed Patch Binding）**：Slow Agent 输出 patch_id（真实 LLM 从 card 白名单选择——不手写 steps）→ _steps_for_patch_id 白名单查表（未知 → no_frozen_program → ACTION_UNAVAILABLE）→ **Runtime 机器生成 Skill body**（Frozen marker + steps JSON——apply 之前）+ **固定 skill_kind=capability**（Fast Agent 只供应 CAPABILITY——真实 LLM 曾输出 bootstrap 致 r2 无 skill；Runtime-owned 绑定覆盖消费契约）。

**结果（当前代码重跑，独立报告）**：
- replay：METHOD_OWNED_SLOW_UPDATE_WIRING_PASS（零 live LLM）
- **live：METHOD_OWNED_SLOW_UPDATE_WIRING_PASS**（真实 Slow Agent 1 次调用选 patch_id="patch-replace-b-with-winsorize" → 白名单 steps → Runtime body → r2 同实例池含 cand_skill_ 且选中 → removal 恢复；checks 7=False 真实标注）
- 审查者：代码层六项 PASS（顺序/绑定/未知 ID/所有权/Schema 最小/无 future）；报告真实性修复（checks 7 live=False、live 专属 note/路径、ACTION_UNAVAILABLE 档、checks 3 改名）

**能力表（修正后）**：Runner 编排 replay 链 = RUNNER_ORCHESTRATED_MECHANISM_OBSERVED（P3.1-A 降级——PASS 工件缺失）；**方法层自动触发 ✓**；真实 Slow Agent 合法 Manifest ✓；**Manifest→可执行 Program ✓（B2）**；自然 Harness 自进化**未建立**（下一步 P3.2 自然 pilot——待用户批准）。

---

## 批准权修复 + 负控 = REPLAY_REJECTION_NEGATIVE_CONTROL_PASS（2026-08-11，用户 P0 缺口）

**用户裁决（P0 缺口）**：handle_feedback 无论 replay 成功与否都更新 snapshot——"LLM 不批准自己、由 replay/delayed 批准"未落实。修复：**approved 门控**——`support.verification.passed 且 support_gain ≥ M` 且 delayed 不显著负向（< −M）才更新 self._snapshot；否则 stage=replay_rejected（snapshot 保持原版本）。

**负控（零 LLM，已知无效候选 outlier_mad——gefcom 上 support −0.065 < M）三项断言全过**：① Patch 被 replay 否决（stage=replay_rejected）② snapshot 保持原版本（未更新）③ 下一轮候选池无该 Skill → **REPLAY_REJECTION_NEGATIVE_CONTROL_PASS**。

**live 意外价值**：真实 Slow Agent 两次运行分别选了 winsorize（PASS——全链采用）与 **outlier_mad（无效候选）→ 批准权正确否决（PATCH_REPLAY_FAILED——合法档位）**——"LLM 不批准自己"在真实选择下直接验证。live 的候选选择变异性（provider 无 seed）如实记录（单次不重试挑答案）。

**三模式（当前代码，独立报告）**：replay = METHOD_OWNED_SLOW_UPDATE_WIRING_PASS；live = PATCH_REPLAY_FAILED（真实否决）；负控 = REPLAY_REJECTION_NEGATIVE_CONTROL_PASS。报告真实性：live 不写 zero_live_llm、三模式独立路径/note、budget_exceeded → INCONCLUSIVE_PROVIDER_FAILURE（max_calls=2：1 正式 + 1 契约内 schema 重试）。

**P3.2 前置全部满足**：自动触发（方法层）✓、Patch 可执行（B2）✓、正常入口消费 ✓、无 Runner 旁路 ✓、**失败 Patch 确定性拒绝 ✓（批准权）**——自然 fresh pilot 待用户批准。

---

## P3.2 首跑降级 = UNAVAILABLE_PROTOCOL_AND_FRESHNESS_FAILURE（2026-08-11，独立复核 5 Blocker + 5 Major）

**用户 + 独立子 Agent 复核**：P3.2 首跑不能作为"自然 Harness 自进化"证据。5 个 Blocker：①冻结 roster 用 train/support/query 但 v6._evaluate 只认 train/eval → 六次 probe gain 全 null（Runner 写成 0.0）→ NO_NATURAL_FAILURE 实为 SUPPORT_OUTCOME_UNAVAILABLE/PROTOCOL_FAILURE；②freeze 调完整 evaluate（读 downstream future）→ solar cohort 已打开 outcome 不能称 fresh；③delayed 批准只拒绝 dg < −M——gain=None/NaN/verifier 失败仍可能批准（正确条件：delayed.verification.passed AND gain 非空有限 AND gain ≥ −M AND episode_id 匹配）；④PASS 不要求下一轮实际采用（未验证 chosen/program/执行/removal）；⑤"最多一次 Slow 调用"未落实（每 probe 触发、预算 8——破坏 first-fault attribution）。Major：Slow Agent 工具绑定 series[:600]（R2/R3 过期 Context）；allowed_tools 未绑定；applicability 由 LLM 决定（非严格 Target-local）；pending 不检查 episode_id；Fast Episode 无完整 delayed 写回；"进入 pool"≠"自主选择并执行"；snapshot 仅进程内。

**仍成立的能力**：method-owned trigger、Typed Patch binding、候选编译、replay rejection 在 development 环境成立；**自然 fresh 的反馈、批准与下一轮真实采用尚未建立**。

**最小修复顺序（7 项）**：①support/query→eval 角色映射 + gain=None 立即协议失败不写 Episode；②delayed verifier 通过 + gain 有限 + episode_id 检查；③全 Pilot 只允许第一个 material fault + 一次正式 Slow 调用；④每轮同步 Slow Agent 公开工具 Context；⑤Runtime 绑定 allowed_tools + claim 限定 context/cohort-local；⑥PASS 必须验证下一轮 chosen/program/执行/removal；⑦freeze 删除完整 evaluate 重新冻结真正未打开 outcome 的 cohort。修复后才消费新 virgin cohort 重跑。

---

## P3.2 复核修复版首跑 = ACTION_UNAVAILABLE（2026-08-11，审查者确认修复落实）

**7 项修复全部落地**（审查者逐项 PASS）：①_evaluate_monash 角色映射 + gain=None → protocol_failure 不写 Episode；②delayed 批准 = verifier 通过 AND gain 有限 AND ≥ −M AND episode_id 匹配（method.py handle_feedback_delayed）；③只允许第一个 material fault 触发（R3 失败 −0.157 未触发）+ max_calls=2；④Slow Agent 工具 Context 每轮同步；⑤allowed_tools 从冻结 steps 绑定（method.py 两处）；⑥PASS 需下一轮 chosen/池含 skill + removal（h0 重跑无 skill）双验证；⑦freeze 只静态 verify（零 gain——不读 future）。

**首跑（solar cohort，真实 LLM）**：R1 repair +0.117（中性偏正）、**R2 自然失败 repair −0.201 → 自动触发 → 真实 Slow Agent 1 次调用**、R3 失败 −0.157（协议：不再触发）。真实 LLM 输出 manifest（edit_id="add_avoid_negative_level_shift"）但 **patch_id=None**（未按 Typed Patch 协议选 ID——LLM 协议遵循不稳定的又一实例）→ 白名单查表失败 → **ACTION_UNAVAILABLE**（合法档位——装置正确拒绝，未误报 PASS/NO_NATURAL_FAILURE/PROTOCOL_FAILURE）。**不重跑**（用户纪律：不重试挑选 LLM 答案）。

审查者非阻塞观察：Blocker 4 验证路径本跑未达（未到 pending）；episode_mismatch 分支 pending 滞留；freeze 死参数；delayed 为即时仿真（origin+HORIZON）非真实时间到达——端到端 delayed 批准路径待 future 运行验证。

**状态**：修复版装置协议健康（自然失败触发 ✓、单次调用 ✓、缺失 patch_id 正确拒绝 ✓）；自然自进化**仍未建立**（首次真实触发因 LLM 未输出 patch_id 而 ACTION_UNAVAILABLE）。

---

## P3.2 Monash 数据面收口 = MONASH_SOLAR_PROGRAM_REPLACEMENT_PREMISE_UNAVAILABLE（2026-08-11，用户裁决选 C）

**准确结论（用户修正）**：不是"LLM 协议失败"——**自然失败发生在 repair_level_shift，但 Program-replacement 生成器在 structural family 中找不到任何合法替代 → Typed Patch 动作空间为空 → ACTION_UNAVAILABLE 正确**。方案 C 的 preflight 已实现（smoke 三 case ✓）但这次未真正使用（typed_patch_options=[]——无白名单自然无法要求 LLM 选 ID）。

**不选 A/B 的理由**：A（rain）预期无 material failure（方法价值低）+ repair=0/winsorize=+0.14~+0.84 的数值**已被查看**（不能称 virgin/outcome-sealed——至少需区分已暴露样本 vs 剩余 143 条全体）；B（人为调整探测顺序）破坏自然入口与 first-fault attribution——即使 PASS 也只是 development positive control。

**能力边界**：自然失败检测 ✓；方法层自动触发 ✓；无合法动作时安全拒绝 ✓；**自然 Program Update 未建立**（原因：Program Supply 空，非 LLM 推理错误）。单算子 repair_level_shift 负向继续由现有 signed Episode/Risk 降级处理，不强行启动 Program replacement。

**下一阶段（用户裁决）**：一次新数据投入（outcome-blind 准备）顺序服务三目标：①自然 Context→Program headroom；②自然 Slow Update；③Source Experience 跨域 A5/A3。数据前提只检查公开 Context + verifier（长度容 Source/R1/delayed/R2/delayed；≥1 有自然信号的 defect family；family ≥2 合法行为不同替代 Program；Source/Target 互斥；不跑 Consumer、不读 gain、不因 outcome 换 series）。**优先 outlier family**（winsorize/outlier_iqr/outlier_mad/hampel_filter——比 structural 的单一 repair 更容易形成真实可归因替代空间）。顺序实验：P0 premise-only → P1 自然 Fast 轨迹 → P2 失败且 ≥2 替代 → 自然 Slow Update → P3 Support/delayed 后下一轮采用 → P4 同资产 matched-budget A5/A3。任何阶段前提不满足即关闭对应 family，不换 origin 找答案。**停止消费 Monash solar/rain**。

---

## KDD_CUP_2018：P0 PREMISE_OK → P1 NATURAL_FAILURE_DETECTED → P2 PATCH_SUPPORT_REJECTED（2026-08-11）

**P0 premise-only（零 outcome）**：KDD Cup 2018（270 条 hourly 空气质量序列——PM2.5/PM10/NO2/CO/O3/SO2——Zenodo 4656756）——抽样 60/60：长度 ✓（10,898 ≥ 984）、outlier 信号 ✓、**outlier family 4 个合法替代 ✓**（winsorize/outlier_iqr/outlier_mad/hampel 全 actionable——对比 Monash solar 的 structural 空）→ **PREMISE_OK**。

**P1 自然 Fast 轨迹（零 LLM）**：冻结 cohort（T1 起 20 支——长度/公开 Context/outlier family 静态合法）→ R1 @600 **winsorize −0.029（material 失败）** → **NATURAL_FAILURE_DETECTED**（KDD 2018 同时满足"自然失败 + ≥2 合法替代"——P2 前提完整）。

**P2 自然 Slow Update**：R1 winsorize 失败 → 方法层自动触发 → **真实 Slow Agent 1 次调用 → 输出合法白名单 patch_id="patch-winsorize-to-outlier_mad"（契约 preflight 强制生效——方案 C 首次真正启用）** → 白名单 steps [outlier_mad] → Support replay **−0.0014 < M → 否决（PATCH_SUPPORT_REJECTED——snapshot 不变）**。批准权在自然场景完整闭环（LLM 合法选择 + replay 拒绝无 headroom 替代）。

**验证的能力**：自然失败检测 ✓、方法层自动触发 ✓、契约绑定（LLM 白名单选择）✓、**批准权自然场景闭环 ✓（LLM 不批准自己）**。**自然 Program Update 仍未发生**（该 Context 的替代无 Support headroom——PATCH_SUPPORT_REJECTED 合法档）。该轨迹触发机会已用（只第一个 material fault）→ 新轨迹需新 cohort（KDD 剩余 250 条——新自然样本）。

---

## P4.0 = MEMORY_ACTION_SIGNAL + P4.1 = NO_SIGNAL（2026-08-11，跨域 Memory Gate + fresh A5/A3）

**P4.0 跨域 Memory 行动性 Gate（零新数据 plan-only）**：Monash outlier-family Source Episode（单条 winsorize 双正——weak_reference 成对判定——跨域 radius 距离在日频/小时频间不可比（n_hist≥3 → radius 全 UNKNOWN——诊断修正为单条 → weak 模式））注入 KDD 已暴露 Context。三干预 M_source/M_remove/M_signswap ×2：**5 条件全过**——①resolver 渲染 ✓（Reference 1: ['winsorize']——跨域 Memory 首次真实渲染）②top-2 集合稳定 ✓ ③集合不同 ✓（M_source {winsorize} vs M_remove {winsorize, outlier_mad}——Memory 渲染经确定性 propose 分支（ref1 短路）改变进入预算的候选**集合**——不只排序）④方向可解释 ✓（signswap NEGATIVE → winsorize 降级出 top-2）⑤不投票 ✓。审查者 8/8 PASS（零新 outcome/排除 repair/池合法/渲染真实/集合比较/方向/不投票/报告真实性）。**MEMORY_ACTION_SIGNAL**——跨域 Memory 行动性首次正面证据。

**P4.1 fresh matched-budget A5/A3（新 KDD virgin cohort K1——长度/公开 Context/静态 verifier 冻结）**：A5（Monash winsorize POSITIVE）vs A3（空）——3 候选池/预算 2/3 轮在线/两阶段 delayed。**原始 verdict = NO_SIGNAL**（两臂轨迹几乎相同：R1 winsorize +0.007、R2 +0.073、R3 −0.143 失败 → 触发 → pending → approved——两臂都形成 Skill；metrics 全同——first_pos (1,1)、harm 1/0.143、delayed 0.807——A5 ref1 短路错过 outlier_mad +0.120 但 metrics 无差异）。

**用户复核（2026-08-11）→ verdict 降级为 CROSS_DOMAIN_VALUE_INCONCLUSIVE_EVALUATION_SEMANTICS**。五处承重问题（审查者/用户逐行核验，均已确认）：
1. 探测按候选池顺序 `pool_ops[:2]`（run_v1_kdd2018_cross_domain_a5_a3.py:149），未严格执行为 LLM chosen-first 的 Runtime 语义（:127）；
2. `delayed_utility` 只累计每轮第一个 probe（:230-232）——A3 R3 第二次探测 outlier_mad +0.1113（report probe 2）被丢弃；
3. Slow Patch 的 Support replay 也读取了 Target feedback（:195）但未计入 matched-budget 指标——A5/A3 实际反馈消耗未完整计算；
4. `skill_formed` 只检查 delayed approved（:235-237）——脚本头声明的"下一轮正常入口采用和 removal"从未执行；
5. Fast Path 用确定性 SealedProbeBackend（:135-137）——真实 LLM 只出现在 P4.0 plan-only Gate 与 Slow Patch。

**保留结论（仅两条）**：跨域 Memory 改变行动 = 成立（P4.0）；跨域 Memory 减少试错或提高效用 = 未建立——且**不能**据此声称 Memory 无价值（估量语义未验）。单 Target 且装置语义受损——不进入任何 cross-domain 能力结论。

**P4.2 自然采用轮 R4（2026-08-11，用户复核裁决第一阶段）= ADOPTION_RETRIEVAL_GATE_BLOCKED（承重）**：
装载 P4.1 delayed 批准的 materialized snapshot（96f83039...——skill_library 含 winsorize_negative_outlier_mad：Frozen program=outlier_mad，observable_applicability={"all":[task_kind==forecast, **clipping_probe_direction==negative**]}）→ R4 正常入口（origin 984——P4.1 消费终点后的首个 virgin 起点；delayed 1032）ADOPT vs REMOVE（原 h0）双臂（零新 LLM——sealed 确定性同 P4.1 装置）。
- C1 skill_in_snapshot ✓（装载验证）；C2 skill_retrieved ✗；C3 chosen_is_skill ✗；C4 executed ✓（chosen=winsorize 执行一次 @984：support −0.0841——**又一次自然失败**、delayed +0.1993）；C5 removal_differs ✗（ADOPT==REMOVE：chosen/程序/gain 全同）。
- **诊断**：clipping_probe_direction 只由 fixed_probe_panel 计算（public_tools.py:83-90），而全 evaluation/functional 无任何 runner 传 panel（grep 验证）→ 该特征在评估装置恒为 "unknown" → retrieval 的 evaluate_applicability 永不匹配 → skill 不进 harness view → _skill_frozen_candidates（fast_agent.py:786）无 cand_skill_* → **下一轮采用不可能**。适用性条件的来源：Slow LLM 提案（card 的 observable_signature 只有 task_kind；slow agent 的 view 特征从不含 probe 方向——条件系从失败叙事推断，schema 白名单合法但评估装置永不填充该特征）。
- **结论**：P4.1 的 snapshot_updated 是必要非充分——**批准 ≠ 采用**。自然更新回路在检索门处未闭合：R4 的 winsorize 再失败（−0.0841）正是该 skill 设计要处理的场景，但 harness 两臂都仍选 winsorize（批准产物对下一轮零影响）。修复方向（待用户裁决）：①评估装置填充 fixed_probe_panel（部署保真）；②批准门校验 applicability 在部署特征空间可满足；③适用性以 card observable_signature 为限（不采纳 LLM 额外条件）。

**保留结论（P4.0/P4.1/P4.2 合并）**：跨域 Memory 行动性成立；跨域预算价值 = 不可判定（估量语义受损）；自然 Slow Patch 批准闭环成立（Support/delayed 批准）但**下一轮采用未成立**（检索门阻断）。

**P4.3 Applicability Binding 修复（A+B）+ dev replay = APPLICABILITY_BINDING_DEV_REPLAY_PASS（2026-08-11，用户裁决）**：
用户裁定 first fault = **Applicability-to-Observation Binding Gap**（Slow Agent 编造 Schema 合法但运行时不可满足的 Scope 条件）；修复 A+B 最小化，明确不做 C（不填 panel）。
- **A（Runtime 约束 Applicability 来源）**：method.py `_applicability_from_card`——observable_applicability 由 Runtime 从 card 的 observable_signature（公开 Observation）机器生成（B2 同款所有权）；空 signature → {"const": True}；manifest 级字段同步（controller 校验两者一致——edit_controller.py:488-493）。
- **B（批准前机械可达性检查）**：method.py `_applicability_reachable`——①特征 ∈ card observable_signature ②∈ 当前 Fast 入口特征空间 ③当前公开 Context 可检索（evaluate_applicability）；不满足 → stage=applicability_unreachable + ACTION_UNAVAILABLE（apply 之前返回——不 fork/不写 active snapshot）；fast_features 缺失 → fail-safe 拒绝（applicability_uncheckable）。
- **Section 1 门机制测试**（零 LLM/零数据）：A 单测 ×2 ✓；B 单测 ×3 ✓（card 绑定 @984 可检索；编造条件按 provenance/context 两路拒绝）；handle_feedback_support 集成 ×2 ✓（clip 签名 card → applicability_unreachable + ACTION_UNAVAILABLE 不写 snapshot；task_kind card → pending——A 绑定后 B 通过走到 pending）。
- **Section 2 dev replay**（已暴露 @984，sealed 确定性同 P4.1 装置，零 LLM）：装载已批准 snapshot（96f83039...）→ 按 A 修正 skill applicability（task_kind==forecast）→ R1 正常入口 **Skill 被检索** ✓（view + cand_skill_* 入池）；R2 **chosen = cand_skill_winsorize_negative_outlier_mad、program = [outlier_mad]** ✓；R3 实际执行一次 ✓（support −0.0608、delayed +0.2567——数据事实如实记录，本实验不声称增益方向）；R4 removal 对照 ✓（h0：无 skill 候选、chosen=cand_winsorize——候选消失、行为恢复）。
- **结论（dev-level——只证修复机制有效，不追溯旧实验为 fresh PASS）**：修复后"批准产物在下一轮正常入口被检索→选中→执行→removal 恢复"全链闭合。**自然 Target-local 自进化仍差最后一步：冻结修复后在新的自然失败轨迹上做最终确认（NATURAL_TARGET_LOCAL_PROGRAM_EVOLUTION_PASS）**——需新 virgin 数据（用户裁决：dev replay 先不用）。
- **审查者裁定**：A+B 忠实实现用户裁决（A 只取 card observable_signature 且 manifest 级同步闭环；B 三子检查在 apply 前、fail-safe、两入口一致；未做 C）；replay 重跑逐位可复现、仅读已暴露 @984、零新 LLM/零新数据；生产采纳路径无绕过（旧试点 runner 直调 controller 属裁决边界外）。附注两项已处置：①B 内 evaluate_applicability 包 try/except——非法 AST 干净拒绝（invalid_applicability）不逃逸；②const True 端到端集成用例补上（empty_sig_card → pending——controller 一致性校验实际通过）。**接线跟进项**：最终确认 runner 的 handle_feedback_support 必须传 fast_features（不传则 fail-safe 阻塞批准——裁决预期行为）。

**P4.4 Program-effect Context 诊断 = OBSERVABLE_SCOPE_SEPARATION_FOUND（候选识别级，附条件）（2026-08-11，用户裁决）**：
用户裁定 P4.3 之后 first fault = **PROGRAM_CONDITIONED_SCOPE_OBSERVATION_INSUFFICIENT**（task_kind-only Scope 过宽——@984 outlier_mad Support −0.0608/delayed +0.2567 冲突）——**先诊断、不盲目换 virgin 数据**。
- **9 点矩阵**（全部已暴露报告提取——零新数据/零新 LLM）：outlier_mad 三类在**修改局部性**上单调分离——双正@888（T117）= 1 簇/span 0.23%；冲突@984 = 3 簇/2.3%；近零@600（T1）= 5 簇/13.5%；winsorize 对照（pp×2/nn/np×2）仅中位数方向一致（pp 更局部）但**逐点不可分**（T1@600 np 是最局部化 winsorize 点）；hampel 双负已算未参与分离。
- **审查者附条件裁定**：FOUND 属"候选识别级"而非"分离确立级"——om_pp 仅 1 点（1-vs-2 分离、7 特征无多重比较校正）、n_clusters/span_fraction 同一 localization 信号族（非独立证据）、系列内反例（T117 winsorize 600→888→984 效用翻转而几何几乎不动——共变主要见于 om 侧且与算子固有行为混叠）、几何作用域 [0,origin) 整前缀 vs 评估逐 anchor 窗口（代理量非复现）。**8 条 caveat 已入报告**。
- **硬条件**：未冻结任何阈值——本次判定只提名候选特征（localization 信号族）；**阈值冻结严格门控于 P4.5 dev replay（扩展系列/origin 网格、跨算子逐点验证：正向 Context 检索、冲突/风险 Context 降级或要求 Support、removal 正确）之后**；P4.5 之前"分离"不得作为已确立 Scope 规则写入 Skill。
- **P4.5 修正（2026-08-11，审查者决定性发现）→ P4.4 的 FOUND 正式降级为"整段作用域伪影"**：①整段应用的 changed 点 100% 位于索引 <120（T117@888 [47,48]、@984 [47,48,50,68,69]、T1@600 27/27 全 <120）而最早评估窗口从索引 120 开始（anchor 312 → [120,360]）——**P4.4 的 span/clusters 度量的是评估从不应用程序的区域**；②共享锚点对照：pp@888 与 np@984 在共享 9 锚点上对齐观测逐位相同（aff 0.016165/cl 2.6038/span 0.214151）而效用相反——表观分离全部来自窗口集合组成（@984 多出 anchor-852 组）。P4.4 从未冻结规则（caveat 5 已警告作用域不匹配）——此为对候选信号的正式否定，非收回。
- **用户顺序**（P4.4 之后）：→ 最小 Observation + Scope 修复 → development 正/负 Context replay → 薄在线入口与 checkpoint → 一次 fresh natural confirmation。

**P4.5 Scope 学习机制切片 = SCOPE_PATCH_SELECTION_ABSTAINED（2026-08-11，用户裁决）**：
用户裁决 P4.5 **不是**"为 outlier_mad 写死 0.01"（case patch）而是 Scope 学习机制实验；四点修正：①Observation 与真实执行 Scope 对齐（training_windows 逐窗口应用）②阈值 Target-local ③证据不足保持 Draft（不安装/不退回 const:true+LOCAL_ACTIVE）④由 Harness 生成 Scope Patch（Runtime 机械生成 ≤2 候选、midpoint 阈值；选择器选择/abstain；Runtime 编译）。
- **机制链全部工作**：对齐观测（9 特征——每窗口 affected fraction/簇/跨度/跨窗口覆盖率/系列分布/cohort 聚合，窗口 [anchor−192, anchor+48] 与评估逐位同构）→ Runtime 从对照区间机械生成 2 候选（aff_mean < 0.0169715 midpoint、span_mean > 0.2097055 midpoint）→ 选择器 margin_ratio 排序（0.4535 / 0.1012）→ **最佳 < 0.5 → abstain（不冻结不安装）**。
- **决定性证据**（共享锚点对照，审查者发现后编码入 runner）：pp@888 与 np@984 在共享锚点（≤840）上对齐观测**逐位相同（max_diff=0.0）而效用相反**——对齐后不存在任何类间定位信号；表观差异（aff 0.0162→0.0178）全部来自 @984 多出的 anchor-852 窗口组（数据 vintage，非效用信号）。**P4.4 的"分离"= 整段作用域伪影 + 窗口集合组成伪影**。
- **审查者裁定**：ABSTAINED 成立且理由更强（共享窗口上 pp/np 逐位相同）；margin_ratio<0.5 门槛合理（1-vs-2 点、9 特征无多重比较校正、跨 cohort 不可比）；修正 3 忠实执行（无安装/无 snapshot 写入）；coverage 实现缺陷已修复（原恒 1.0）；abstain 分支报告已补逐点观测与 n_windows；replay 链（编译/正负 replay）本运行未执行（abstain 短路——机制验证到选择器为止，如实披露）。
- **结论（dev-level）**：Scope 学习机制的原型链（Observations → Runtime Patch 生成 → 选择/abstain）成立；但当前已暴露数据上**无可用 Scope 信号**——outlier_mad Skill 保持 task_kind-only（过宽）且不自动优先（LOCAL_DRAFT_REQUIRES_SUPPORT 语义——每次 Target Support 确认）；不消费 fresh 数据。用户后续顺序不变：薄在线入口与 checkpoint → 之后若有新证据再议 Scope。

**OPERATIONAL_TARGET_LOCAL_SELF_EVOLUTION_LOOP（P0-P5，2026-08-12，用户裁决）= OPERATIONAL_TARGET_LOCAL_SELF_EVOLUTION_DEV_PASS**：
唯一方法改动：**requires_target_support 执行权限**（复用 risk_guards，不新增 Schema/状态系统）。
- **P0**（run_v1_draft_authority_gap_check.py）：宽 Scope capability Skill（task_kind-only、无 guard）经正常 prepare——G1 检索 ✓ G2 进池 ✓ G3 在 Agent 前 ✓ G4 chosen=skill（不经 Support 即成最终 Program）→ **DRAFT_EXECUTION_AUTHORITY_GAP 确认**。
- **P1**（method.py + fast_agent.py）：method——宽 Scope（const:true 或 leaves⊆{task_kind}）新 Skill 写入 `risk_guards.requires_target_support=true`（manifest 级同步；不回改已批准 skill）；fast_agent——合并顺序 **ACTIVE → Agent → DRAFT → signed-risk degraded**（DRAFT 不挤 Agent、不因 Reference 1 自动优先）。验收（run_v1_execution_authority_check.py）：DRAFT 不自动优先（chosen=cand_winsorize）✓ ACTIVE 保留优先 ✓ H0 不变 ✓ = EXECUTION_AUTHORITY_P1_PASS。
- **P2**（methods/ttha/online_loop.py）：唯一薄在线入口——run_online_round（prepare 一次/chosen-first/预算 2 含 Slow replay/verifier 拒绝单独记/立即写 Episode/第一正向 winner/SLOW_UPDATE_DEFERRED_NO_TARGET_BUDGET/两阶段 pending）+ open_delayed（episode_id 匹配的 handle_feedback_delayed）+ activate_approved（仅批准后 set_active）+ current_status；16 项统一指标（delayed_utility 计实际 winner）。
- **P3**（run_v1_operational_self_evolution_loop.py，四场景 + 8 条件总验收）：A traffic 正向复用（A5 first-pos=1 vs A3=2、delayed 正向、A5 R2 ACTIVE skill 第一 probe=skill）；B GEFCom 正控（outlier_iqr −0.1302 → 方法层自动触发 → replay winsorize +0.4000（与 bounded 报告逐位一致）→ delayed +0.2572 → approved → **guard 写入** → R2 检索不优先 → 经 Support 后 winner → removal 轨迹恢复）；C 坏 Patch（outlier_mad −0.0646 → support_rejected → snapshot 不变 → 下一轮无 skill）；D Scope 不可识别（P4.5 独立重算 abstain——不读旧报告 verdict）→ **8/8 → OPERATIONAL_TARGET_LOCAL_SELF_EVOLUTION_DEV_PASS**。
- **P4**（run_v1_a5a3_runtime_regression.py）：统一入口 KDD T117 三轮 + R4——轨迹与 P4.1 一致（R1/R2 winsorize 正向、R3 −0.1426 → slow → outlier_mad +0.1199/+0.1113 approved → skill）；R4 检索 ✓ 不自动优先 ✓ 探测执行 ✓。**语义差异如实记录**：统一入口下 material failure 立即触发 Slow（P2 语义 7）——A3 R3 的第二探测由 replay 替代（P4.1 的 probe 4 vs 3 差异在统一入口下消失——两臂同轨 NO_SIGNAL 语义）。verdict = CURRENT_RUNTIME_A5A3_DEVELOPMENT_REGRESSION（不声称 cross-domain）。
- **P5**（run_v1_persistence_check.py）：active.json + materialized 树重载（重启语义）→ snapshot 重载 ✓ Draft Guard 持久化 ✓ current_status 字段 ✓ 下一轮行为一致 ✓ = PERSISTENCE_CHECK_PASS。
- **审查者裁定（P3，8 项清单独立核实）**：Runner 未旁路方法层 ✓ Draft 真实不优先 ✓ 预算含 Slow replay（probe1+replay1=2）✓ delayed 匹配 episode_id ✓ winner 来自真实 steps_map ✓ snapshot 仅确定性批准后更新 ✓ 零 future read（各 origin 已暴露、GEFCom 数值与 bounded 报告逐位一致）✓ 无报告布尔代替真实轨迹（D 独立重算）✓。caveats：场景 A 数值不复现 sealed 报告（registry/系列差异——结构成立已披露）、A5 Source Episode 数值混合来源（均已暴露）、场景 D"不自动优先"为组合验证（D abstain + B guard + P1 预检）、场景 C no_skill_next_round 检查较弱（与 snapshot_unchanged 组合成证据）、Slow 触发调用门在 online_loop（materiality 复判在方法层——语义 7 分工）、B R2 winner 来源靠探测轨迹+removal 对照。

**E0 Operational Semantics Hardening = E0_HARDENING_PASS（2026-08-12，用户综合结论裁决）**：
用户综合结论：DEV_PASS 保留（外部 AI 指出问题已解决 7 项——chosen-first/预算含 replay/采用检查/A+B Binding/Draft Guard/统一入口/持久化）；仍成立 4 项（fresh 自然未建立、A5 价值未建立、Source Memory 可能压制探索——ref1 短路变"候选裁决器"需双槽、Slow Agent 仍是 bounded Patch selector）；新发现 3 项通用缺口（Draft 权限是排序非硬授权、统一指标 3 错误、P5 只持久化 snapshot）。
- **E0 七项修正**（零新数据/零 live LLM）：①chosen proposal 与 authorized deployment 分离（RoundResult 加 chosen_proposal——DRAFT 被选择只申请 Support）；②first-positive 用合法 Support receipt index（原 len(actual_probed_programs) 会算入 verifier 拒绝）；③Slow replay 进 actual_probed_programs（kind=slow_replay）+ harm；④delayed gain None 不转 0（保持未评估——不掩盖协议失败）；⑤memory_resolution_status 公开（DecisionTrace 新字段——no_memory/rendered/rendered_empty/contrast_pack/injection_failed——A5 不静默退化 A3）；⑥Slow 调用透传合法 Operator contracts + TaskContext（method.py handle_feedback/handle_feedback_support 新参数）；⑦current_status 分类修正（restricted 只含 SAFETY、bootstrap 单独列）。
- **验收（run_v1_e0_hardening_check.py，9/9）**：DRAFT 被选择 ≠ 授权（operators=() 池只余 DRAFT skill——chosen=DRAFT 但探测负向 → authorized_deployment=None）✓ DRAFT 探测正向 → 授权 ✓ first-pos=receipt 序 ✓ replay 进轨迹（+0.4000）✓ contracts 透传（outlier_iqr/winsorize）✓ delayed None 保持未评估 ✓ memory status（A5 rendered/A3 no_memory）✓ 状态分类 ✓。P3/P1 回归重跑仍 PASS（E0 不破坏既有验收）。
- **E0 后准确口径**：Draft 已是"可观测的权限门"（chosen≠authorized）；"必须 Support 才能部署"在统一入口内不可绕过（DRAFT 只有经探测正向才成 authorized deployment）。

**E1 fresh 自然闭环 = NO_NATURAL_FAILURE（2026-08-12，用户装置确认）**：
装置（用户确认）：预冻结新 KDD cohort E1（剩余 virgin 排除 K0/K1——20 支冻结，w1_kdd2018_frozen_cohort_e1.jsonl，零 gain）；origin 600/792/888；Fast sealed；真实 Slow ≤1 次；接受合法负档。
- **结果**：同 cohort 内依次尝试 5 支系列（T135-T139）——15 次探测**全正向**（0.0302/0.2991/0.0661 三档重复）——无 material failure → **NO_NATURAL_FAILURE**（LLM 0 调用——无物可学）。
- **装置级发现**：①评价是 cohort 级（v6._evaluate cohort mean）——换 series0 不改变 gain（5 支轨迹完全相同）；②sealed force_pool 按算子序探索（不读 features）——探测序固定 winsorize→outlier_mad→hampel；③**E1 cohort 在 600-936 窗口上 outlier family 无失败信号**（对比 K1 T117 @888 −0.143——数据不同）。
- **含义**：自然失败不是普遍存在——该 cohort 该窗口无物可学（合法负档）。fresh 闭环需更长轨迹（origin 984/1080——virgin）或不同候选/装置（用户裁决）。E1 不因失败换 cohort——NO_NATURAL_FAILURE 如实收口。

**E2 Source Memory 双槽 = MEMORY_TWO_SLOT_CONTROL_PASS（2026-08-12，用户裁决）**：
用户裁决：接受 E1 负档转 E2（已暴露数据/零 LLM）；装置纪律——不再逐 series 重复 cohort evaluator（一次 cohort 一个轨迹；series 粒度属后续独立机制）。
- **双槽实现**（仅一个 Control）：DeterministicStrategyBackend/SealedProbeBackend 加 `reserve_exploration_slot`——propose 的 ref1 分支追加一个当前 Context 探索候选（跳过 ref1 算子本身/explored/deprioritized——deprioritized 耗尽后回退）。无 ref1 路径/A3/H0 不变。
- **三臂验证（5/5）**：C1 Source 不能删除探索槽（two 池 [identity, winsorize, denoise_median] vs hard 池 [identity, winsorize]——ref1 短路被双槽解除）✓；C2 正例最多优先一个 trial（winsorize 第一且只 1 次）✓；C3 负例只能降级不能封杀（负例臂 R1 池无 winsorize——降级；R2 跨轮 backend 状态 UNKNOWN 耗尽 → deprioritized 回退——winsorize 进池 +0.5881）✓；C4 Target 反馈覆盖 Source 排序（verdict 层：R1 (src,) winsorize=POSITIVE_PRIOR（Source 优先生效）→ R2 (src, Target负向) =UNKNOWN（中和——Target 覆盖）✓；C5 三臂预算 ≤2 全同 ✓。
- **诊断过程发现**（如实记录）：①sealed 的 ref1 短路在 SealedProbeBackend.complete（父类修改需两处）；②负例回退验证需跨轮复用 backend（explore 状态累积）；③"Target 覆盖"的正确语义是 verdict 层（POSITIVE_PRIOR→UNKNOWN/RISK）而非 instruction Reference 解析（DecisionTrace 不存 instruction；弱引用渲染对 delayed_pattern 键有要求）。
- **下一步 E3**（E2 通过后）：fresh A5-two-slot vs A3——真实 Fast LLM、同一 Candidate DSL、同 Target Support 上限、Slow replay 计入预算、delayed 次数一致、Skill 必须采用/removal 才算形成；承重指标 feedback_to_reliable_local_skill/harm_before_recovery/final_delayed_utility/normal_entry_adoption_delta/abstention；**≥2 Target Dataset 同向才声称 cross-domain benefit**。
- **审查者裁定（E2）**：PASS 成立（5/5、代码与运行双重验证、确定性复现）。5 条 caveat 已处理：①**承重**——traffic 臂实际探测系列（0414c7e9=T635）因 registry 漂移未被先验暴露——"零新数据"仅 origin/算子级成立（fresh 求值——消费记录以 E2 报告为准；三臂同装置相对比较不受影响）；②C4 的 UNKNOWN 真实机制 = n_hist 2→4 的 weak→radius 模式切换 + @984 半径外排除（非成对中和——归因修正入报告）；③两 backend 的 reserve 消费语义不一致（wiring propose 即消耗 vs sealed select 才消耗——E3 若用 wiring 变体需注意）；④traffic 臂 role=final_query——无测试区泄漏；⑤C4 判别力依赖 verdict 层重放（R2 探测本身无判别力——如实记录）。

**纵向集成第一步 = INTEGRATION_FAILED_C5（2026-08-12，用户裁决"一次开发集成"）**：
目标：真实 LLM Fast → Runtime-owned 双槽 → Support → 正向 winner → Fast winner → Draft Skill → delayed 批准 → snapshot → 下一正常入口复用 → removal（KDD T117 已暴露；零新数据）。
- **生产接线**（三处）：fast_agent.prepare 加 `runtime_prior_slot`——Slot P 从 resolve_order 结构化结果（POSITIVE_PRIOR 算子——不解析 Reference 文本）机器生成 ≤1 先验候选（cand_prior_*）；探索槽检查——LLM 提案无任何非 P 算子候选 → AgentProtocolError(EXPLORATION_SLOT_EMPTY)（协议失败档）。method.py 加 `handle_fast_winner`——Fast winner → machine manifest（精确用 trace.candidate_program_steps；宽 Scope → requires_target_support=true；B 检查；apply → pending 两阶段；confirmed_cause=SKILL_LIBRARY_GAP——fault-routes 已注册）。online_loop：runtime_prior_slot/allow_fast_skill 透传；**Slow replay 正向即 winner 并停止探测**；**open_delayed 只对 winner 开 delayed**（未部署候选无反事实 delayed 污染 Memory）。
- **运行结果（真实 LLM，13-15 次调用）**：C1 双槽填充 ✓（A5 池 [identity, cand_prior_winsorize, repair_level_shift_local]——P+E 都进池——**真实 LLM 负责排序：chosen=repair_level_shift_local（探索候选被选——P 未被选——LLM 自由 ✓）**）；C2 A3 无 prior ✓；C3 Fast winner → Skill 批准 ✓（fast_winner_repair_level_shift——delayed +0.0971 批准 → snapshot）；C4 guard ✓；C6 removal ✓；C7 预算 ✓；**C5 失败**——skill 检索 ✓（view 层 retrieved）但 **verify 拒绝（绑定参数跨 Context 失效）**——repair_level_shift 的 R1 绑定参数在 R2 @888 上下文被 verifier 拒绝 → skill 候选不进池 → 未探测。
- **归因（如实，经诊断修正）**：C5 失败**不是**绑定参数失效（诊断：R1 参数在 R2 @888 的 verify 全过——preserve/max_modified 全组合 0/108 拒绝；_parse_frozen_steps 解析正常）——**CONFIRMED（plan-only inspect 重放）**：R2 真实 LLM inspect 声明区域 (37,136) vs skill 冻结修改区域 [41,152) 不匹配 + preserve_outside（h0 verification 配置本就 true）→ OUTSIDE_SCOPE_MODIFICATION 拒绝。sealed 装置 inspect 恒 [0,1] 不发生——确定性/真实装置行为差异。

**CONTEXT_BOUND_SKILL_REBINDING = CONTEXT_BOUND_SKILL_REBINDING_DEV_PASS（2026-08-12，用户裁决——第四种机制，不选①②③）**：
用户裁决：Skill 保存**可重用 Workflow 模板**——每轮按当前公开 Context 通过 registry 的 public_parameter_bindings 重新绑定动态参数（region/offset），生成当前轮 Program 实例，按该候选自己的当前 Scope 验证（preserve_outside 保留——不退化全窗口），Target Support 授权执行。①②（全窗口/固化旧区域）绕过安全边界或退化一次性记录；③（禁区域型 Skill）收窄目标——均不选。
- **实现（四处）**：①fast_agent._skill_frozen_candidates(view, features)——模板解析后逐 op 用当前 features 按 registry bindings 重绑（缺失 → 安全拒绝）；②prior 同一绑定逻辑（缺失 → prior 不生成）；③_regions_for——Skill/prior 候选 verify 用**绑定后候选自身区域**（region_start/end_fraction），Agent 新提案用 LLM inspect 区域；④handle_fast_winner 加 support_gain——Fast winner 已获 Support 直接复用（不重开相同 Context×Program 的 Support——不计预算重复仪器评估）。
- **验证（重放已暴露 T117 @888，真实 LLM 一次运行）5/5**：C1 rebind 参数 ≠ R1（[0.0417,0.1532] vs [0.0467,0.1717]）；C2 匹配当前 features；C3 **skill 程序进池——由 level_shift_local_repair 承载（CandidatePool 按 program sha 去重——rebind 后与 LLM 提案程序相同——正确语义）**；C4 探测获 Support +0.1274（已暴露窗口）；C5 removal 翻转。
- **意义**：纵向集成 **7/7 接通**（C5 从失败转通过）——"双槽、Fast winner、Skill 形成与批准、跨轮重新实例化"全链就绪——**立即进入 fresh A5/A3（E3.1）**。
- **审查者裁定（2026-08-12）**：DEV_PASS 成立——纵向集成 7/7 接通（集成 runner 6/7 + rebinding check 补全 C5——两条运行合读）。7 条 caveat 记录：①skill 候选自身 verify 路径未实证（sha 去重在 verify 前剔除同程序候选——_regions_for 的 skill 分支代码审查级正确、本运行未执行）；②C3 归因力有限（池程序 == LLM 提案程序时 rebinding 与 LLM 独立提案不可区分——本运行两者并存）；③support_gain 计量路径无实证记录（纯代码审查——rebinding check 未触发）；④C2 占位（由 C3 _matches 承载）；⑤C5 removal 结构性弱（h0 无 skill + sealed 不产 cand_skill_*——程序集差异的间接证据）；⑥取整边界泛化风险（_regions_for int() vs 算子 floor/ceil 对一般序列终点可能差 1 索引→假拒绝 fail-closed；T117 精确回环未发生）；⑦docstring 区域表述易误读（[41,152) 是旧参数在 888 下的漂移区域）。

**核心判断（2026-08-12 用户）：第一缺口 = 多轨迹共同归因**：
项目已具备"单次失败→有界修改→Support/delayed 审批→Skill"自进化内核；**尚未实现"多条轨迹聚合→共同错误归因→共同 Harness 修改"**（后者才是最重要能力缺口）。当前 Slow Path 接收一条失败 Episode 立即触发（method.py:517 单 Card→≤2 Patch）——不是"多条失败轨迹→聚合→共同 first fault→共同修改→组外验证"。其他缺口：Episode 反馈粒度不足（在线只留 cohort 平均 gain——per-view 未保留——Slow 看不到哪些 series 恶化/错误集中在哪）；自然替代 headroom 不稳定（安全闭环成立、共同有效修复未成立）；跨域 Source Memory 价值未建立（E31 Source 是伪 prior——signswap 中和为 CONFLICT）；Context 表达偏粗。**Memory-to-action 已有证据、Memory-to-benefit 尚未建立**。定位："将现有 Harness 的经验复用、Skill 更新和验证机制，重新组织为面向时序数据适配的、Context-conditioned、带分级执行权限的自进化 Harness"。
**下一步（用户裁决——最小纵向切片，不建平台）**：①保留现有 per-view Action–Response ②对已有 Episode 轻量确定性分组 ③找至少一个重复 first-fault 组 ④验证组存在共同 replacement headroom ⑤Slow Agent 基于整组 Contrast Capsule 提出 Patch ⑥组外同域 Support/delayed 验证 ⑦下一正常入口复用。只需普通 Episode 列表 + 轻量分组（无向量库/Pattern Graph/新 Ledger）。**不继续为 E31 找 Source 数据**（A5/A3 跨域线暂缓——Source 内容不可用）。

**GROUP_FAULT 最小纵向切片（2026-08-12 实施）= 机制全链验证**：
- **A per-view 保留**（online_loop 生产）：Episode 存 per_view_gain（per-series）+ support_origin——学习证据（不渲染进 instruction——signed_radius 只取 recent./change. 键）。
- **B/C group_fault.py（新生产模块）**：iter_failure_episodes / group_first_faults（按 Workflow+sign 轻量分组——≥2 组）/ build_contrast_capsule（per-view 聚合：worsen/improve fraction、cohort gain 分布、delayed signs）/ find_common_headroom（替代在组内各 origin replay——全部 ≥M 才共同）。
- **D method.handle_group_feedback（组级 Slow 入口）**：整组 Capsule → Slow propose → B 检查 → apply → **组内 replay（各组内 Episode 的 Support 窗口全部 ≥M）** → **组外 holdout（未参与归因的同域窗口不劣 ≥−M）** → pending → delayed 批准。
- **dev 验证（run_v1_group_fault_dev.py——已暴露 T117，零新 outcome）**：winsorize NEGATIVE 组（@888 −0.143 + @984 −0.0841）→ capsule（worsen 0.625）→ headroom（outlier_mad 无共同〔@984 −0.061〕；**hampel 共同正向〔+0.034/+0.015〕**）→ 组级 Slow（hampel patch——组内 replay 全正 → holdout @600 +0.0013 不劣 → pending）→ **delayed @1032 −0.1166 → delayed_rejected（批准门正确拒绝——Support 共同 headroom ≠ delayed 稳定——hampel 在此 Context 的 delayed 翻负——不写 snapshot）**。对照组：提 outlier_mad（无共同 headroom）→ 组内 replay 拒绝（机制正确）。
- **切片结论**：多轨迹共同归因机制链完整（分组→capsule→headroom→组级 Patch→组内/组外验证→delayed 门控）——本组数据的"共同有效修复"在 delayed 层未成立（数据事实——与用户判断"自然替代 headroom 不稳定"一致）——安全闭环。下一轮复用待未来组（有共同 headroom + delayed 稳定的组）出现。
- **自动触发接线（2026-08-12 完成）**：online_loop 加 allow_group_slow/group_min/group_card_builder/group_holdout_origin——每轮探测后失败 Episode 积累 → group_first_faults（≥group_min）→ handle_group_feedback 自动触发（组内/组外验证 → pending）→ open_delayed 批准组 pending。**dev 验证（run_v1_group_auto_trigger_dev.py——sealed T117 四轮）= GROUP_AUTO_TRIGGER_DEV_PASS**：@600/@792 正 → @888 −0.143（1 条未达阈值）→ @984 −0.0841（**2 条 winsorize NEGATIVE → 自动触发**）→ 组级 hampel（组内 replay 全正 + holdout 不劣 → pending）→ delayed @1032 −0.1166 → **delayed_rejected**（安全闭环）。**多轨迹共同归因已完整接入主链（触发层自动化）**——runner 不再编排分组/触发。
- **审查者裁定（2026-08-12）**：切片机制链成立（重跑逐字节复现 + capsule 数学独立重算一致 + 与 P4.4/P4.5 报告数值逐位交叉核对）。caveat：①**自动触发未接入 online_loop**（失败积累→分组→组级触发的编排只在 dev runner——触发层在切片外——下一步接线）；②"多轨迹"实为同系列双 origin（T117@888+@984——分组机制与系列无关但跨系列未实证）；③步骤 7 复用与 delayed 批准路径未实证（本组被拒天然无复用）；④**@1032 是新评估窗口**（delayed 时间边界——两阶段门控本意）——"零新 outcome"仅支持侧严格成立（表述修正）；⑤对照组（outlier_mad 拒绝）为逻辑推导非独立执行；⑥小项（headroom docstring"≥组失败 gain"未实现——代码严格 ≥M；_sign_of 对 delayed-only 失败归 NEGATIVE；holdout 双重调用；ev edit_id 命名潜在不一致；预注册 verdict 列表补 GROUP_FAULT_DELAYED_* 标签）。

装置（用户裁决）：virgin KDD cohort E31（T153-T172——排除 K0/K1/E1 已用——固定顺序冻结零 gain）；Monash 正/负/冲突 Source 组合；真实 Fast LLM；A5（双槽）vs A3（空）——唯一差异 Source；三轮适配 + 一轮采用；每轮 Support ≤2（Slow replay 计入——ReplaySlowAgent）；四轮累计 ≤8；第一正向即停；delayed 只开 winner；Fast winner → Draft Skill（rebinding 跨轮可用）；removal sealed plan-only。
- **结果（判定修正后 NO_SIGNAL）**：两臂可比轮次（600/792/888）指标全同——fb=1（首轮即形成 fast_winner_repair_level_shift skill）、harm=0、delayed=1.2348（同轮同 winner）、adopt=True（R4 检索/探测/removal 翻转）。A5 receipts=2 vs A3=3——A5 R888 LLM abstain（方差——非 Source 效应——LLM 响应未持久化、归因不可证但不在预注册指标内）。**两臂 R600 完全同轨**（level_shift_repair +2.5062——同一程序同一 gain——A5 memory rendered 下 LLM **独立提出了与 A3 相同的程序**〔Source episode 是 winsorize——winner 是 repair_level_shift——审查修正措辞〕——post_validator 绑定参数与 features 严格相等 + CandidatePool sha 去重 → 两臂必然同程序——**无区分空间**——与 P4.1 V2 的教训同型：Reference 与 LLM 自然选择一致）。
- **判定修正（2026-08-12）**：原 NEGATIVE_TRANSFER 基于**跨轮不可比 delayed**（A5 R600 winner delayed vs A3 R888 winner delayed——不同 Context 不同程序）——修正为同轮比较 → NO_SIGNAL（用户档："Memory 已真实生效但最终轨迹和指标相同"）。报告已更新（数据未变——判定重算）。
- **含义**：fresh 装置下 Source Experience 未改变 Target 行为（区分空间不足）——NO_SIGNAL 不说明 Memory 无效（同 V2 先例）；**单 Target 的 fresh 证据 = 无方向性**——cross-domain benefit 仍未建立；≥2 Target 同向复现的前提（区分空间）未满足。E31 cohort 消费记录以报告为准。
- **审查者裁定（2026-08-12）**：**NO_SIGNAL 成立**（判定修正后函数在报告数据上重算一致；NEGATIVE_TRANSFER 不成立——唯一触发源是跨轮 delayed 比较已修正；CONTENT_INCONCLUSIVE 不成立——两臂均未 abstain、有可比较轨迹）。装置忠实（E31 与 K0/K1/E1 两两不相交、全库无其他 KDD 消费记录、冻结零 gain、Source 组合早于运行 2 天冻结）；判定修正无 LLM 重跑证据链（报告从已生成数据重算——文件时间戳证实）。5 条 caveat 记录：①LLM 调用数不可核对（仅 max_calls=40 硬停未触发可验证）；②R888 abstain 不对称归因不可证（LLM 响应未持久化——不在预注册指标内）；③docs 措辞已修正（Source=winsorize、winner=repair_level_shift——"LLM 独立提出与 A3 相同的程序"）；④R792 Slow 事件 stage 未入报告（manifest base_sha=h0 与活动快照 h1 不匹配→apply 前置失败→预算未消耗——预算语义不受影响）；⑤"Memory 已真实生效"只证明指令注入（渲染路径验证）——LLM 是否读过不可直接观测。

---

## 决策记录

- P0（2026-08-10）：REMOVE_A 型，进入 P1。
- P1（2026-08-10）：WRONG_REPLACEMENT_SELECTED——LLM 在无反事实信息墙下选错替换候选（C2 而非 C1）。停止；不调 Prompt/Pattern/候选。
- P1.5（2026-08-10，用户裁决）：不做 Program-effect grounding，改 BOUNDED_SLOW_REPLACEMENT_RUNTIME_SELECTION（LLM 有界候选 supplier + Runtime 实测选择）→ **PASS**。按用户后续顺序进入 P2。
