# S1a-r2 legal evolution treatment audit

protocol: `s1a_r2_legal_treatment_audit_v1`  parent r1: `837b537`  evidence grade: **development**

0 LLM / 0 fit.  Sealed oracles reused, not rescored.  r1 artifacts not overwritten.

## 1. Three legality rules

### a_no_target_local_carry

Target-local Skill 禁止跨单元携带进下一单元 Fast 视图。

- canon: AGENTS.md:174-175 Target-local Skill 在当前 Domain held-in Support 上形成，由同域 delayed 更新；冻结后仅同域 held-out 使用
- canon: AGENTS.md:76-81 合法通路 = Source Episode → census → Slow consolidation → audited Source-derived Skill → Fast
- canon: AGENTS.md:184-191 Fast 禁止读取未匹配当前 Domain 的 Source Target-local Card
- gap: 现行 Target-local 卡由 run_e2_t6_cls_op_shared_harness.py:610-613 _card_builder 写入 observable_signature = {task_kind: classification}；method.py:89-105 _applicability_from_card 因此只编译 task_kind 叶；retrieval.py:278-282 evaluate_applicability 对所有 classification 单元为真。Target-local 卡不是经验卡（retrieval.py:158-164），T1 惰性闸口（retrieval.py:274）拦不住它。照跑会测到宽 Scope bug，不是合法演化。
- audit: 本审计按正典拦截跨单元 Target-local 携带，不按现行 task_kind-only 匹配放行。

### b_heldin_positive_authorizes

单元计入可授权 Source 证据，须 held-in Support 与 delayed 均被现役生命周期判为 POSITIVE（材料级正向）。禁止自造阈值。

- canon: AGENTS.md:174-175 / 139-146 Support 与 delayed 仅 held-in
- canon: AGENTS.md:172-173 Episode 不自动获执行权
- live: experience_memory.py:398-451 classify_relation: agg >= +0.005 且逐 view >= -0.005 → POSITIVE；cls_scope_adapter.py:31-36 分类 view = 逐类 recall
- live: method.py:742-757 handle_fast_winner Support != POSITIVE → support_rejected，不形成 Draft
- live: online_loop.py:201-204 Support = POSITIVE 才 LOCAL_DRAFT
- live: method.py:1466-1492 handle_feedback_delayed 改为 classify_relation == POSITIVE 才 approved；NEUTRAL / CONFLICT / NEGATIVE 丢弃 pending（旧门 dg >= -0.005 已废）
- live: signed_radius.py:40 MATERIAL_THRESHOLD = 0.005
- proxy: 密封 oracle 只评了拼接 held-in 池一次（_wine_heldin_pool）。本审计把该读数送入同一个 classify_relation，作为 Support 与 delayed 两道门的代理；不发明四分切片，也不改用 heldin_material_line。

### c_unguided_authorizes_try

仅未受旧 Skill 引导的正例可授权新 Shared TRY。未引导 = 该单元 Fast 视图不存在指向同 Program 族的 TRY / capability 卡。

- canon: AGENTS.md:176-177 Shared Capability 需多 Domain 重复正向
- live: source_skill.py:217-257 authorization_audit: 仅 UNGUIDED POSITIVE 可授权新 TRY；conditioned 只可确认/反驳/撤回
- live: source_skill.py:249-256 LOO: 去掉任一 Task 后 UNGUIDED POSITIVE 仍须 >= min_distinct_tasks（cls harness :168 MIN_DISTINCT_TASKS=2）→ 2 个正例 loo_minimum=1，TRY 不授权（does_not_survive_leave_one_out）
- live: retrieval.py:195-238 / 274 T1: 无授权 TRY 且无重复 scoped RISK 的经验卡 Fast 不可见

## 2. Learnability (oracle-set operators only)

Approval proxy = `classify_relation` on the sealed combined held-in reading.  Threshold = 0.005 (experience_memory.py:408 / signed_radius.py:40).  The oracle `heldin_material_line` is **not** used.

| unit | oracle program | held-in | relation | held-out Δacc | label |
|---|---|---|---|---|---|
| GunPointAgeSpan__impulse_v2 | hampel_filter | 0.37499999999999994 | POSITIVE | 0.2626582278481012 | **LEARNABLE** |
| GunPoint__impulse_v2 | hampel_filter | 0.4666666666666667 | POSITIVE | 0.4066666666666667 | **LEARNABLE** |
| ECG200__impulse_v2 | repair_burst_segment | 0.0 | NEUTRAL | 0.040000000000000036 | **HELDOUT_ONLY** |
| Wine__impulse_v2 | identity | 0.0 | ABSTAIN | 0.0 | **N/A** |
| ToeSegmentation1__impulse_v2 | repair_burst_segment | 0.08333333333333326 | POSITIVE | 0.030701754385964952 | **LEARNABLE** |
| Lightning2__impulse_v2 | repair_burst_segment | 0.16666666666666663 | POSITIVE | 0.09836065573770492 | **LEARNABLE** |
| Herring__impulse_v2 | hampel_filter | 0.0 | NEUTRAL | 0.046875 | **HELDOUT_ONLY** |
| Ham__impulse_v2 | identity | 0.0 | ABSTAIN | 0.0 | **N/A** |
| GunPoint__burst_cls2 | outlier_iqr | 0.0 | NEUTRAL | 0.013333333333333308 | **HELDOUT_ONLY** |

### Cluster learnable counts

- hampel (GPA / GunPoint / Herring): LEARNABLE **2**/3 (['GunPointAgeSpan__impulse_v2', 'GunPoint__impulse_v2']); HELDOUT_ONLY ['Herring__impulse_v2']; independent families **1** (['GunPointFamily']).
- repair_burst_segment (ECG200 / Toe / Lightning2): LEARNABLE **2**/3 (['ToeSegmentation1__impulse_v2', 'Lightning2__impulse_v2']); HELDOUT_ONLY ['ECG200__impulse_v2']; independent families **2**.
- GunPoint↔GPA: GunPointAgeSpan and GunPoint share family_key=GunPointFamily and have identical pattern_view (byte-equal).  Scope v1 uses features not dataset names, so both may formally count as Source evidence; independence is weakened and is reported separately as n_independent_learnable_families.

## 3. Legal timelines on the r1-frozen 6-unit course

### Forward

order: `['GunPointAgeSpan__impulse_v2', 'Wine__impulse_v2', 'GunPoint__impulse_v2', 'Ham__impulse_v2', 'Herring__impulse_v2', 'GunPoint__burst_cls2']`

first legal Fast-visible difference: **null**

approvable transfer channels: **[]**

| i | unit | learnability | episode | A5 Fast | legality |
|---|---|---|---|---|---|
| 1 | GunPointAgeSpan__impulse_v2 | LEARNABLE | positive_unguided | K0 inert Slow card only; no Target-local carry | L-TL-FORM AGENTS.md:174-175 + method.py:742-757 + online_loop.py:201-204 + method.py:1466-1492; L-TL-NOCARRY AGENTS.md:174-175,184-191; 不按 task_kind-only 宽 Scope 放行; L-UNGUIDED 本单元 Fast 无同族 TRY 卡; L-SCOPE 未满 ≥2 独立可学正例或交为空; L-LOO source_skill.py:249-256; L-T1 retrieval.py:274 inert → Fast 不可见 |
| 2 | Wine__impulse_v2 | N/A | identity_or_empty | K0 inert Slow card only; no Target-local carry | L-EP AGENTS.md:172-173 Episode 可记，不获执行权 |
| 3 | GunPoint__impulse_v2 | LEARNABLE | positive_unguided | K0 inert Slow card only; no Target-local carry | L-TL-FORM AGENTS.md:174-175 + method.py:742-757 + online_loop.py:201-204 + method.py:1466-1492; L-TL-NOCARRY AGENTS.md:174-175,184-191; 不按 task_kind-only 宽 Scope 放行; L-UNGUIDED 本单元 Fast 无同族 TRY 卡; L-SLOW AGENTS.md:76-81; L-SCOPE Scope v1 五轴 (STAGE_REPORT 2026-08-25 20:1x); L-LOO source_skill.py:249-256; L-T1 retrieval.py:274 inert → Fast 不可见 |
| 4 | Ham__impulse_v2 | N/A | identity_or_empty | K0 inert Slow card only; no Target-local carry | L-EP AGENTS.md:172-173 Episode 可记，不获执行权 |
| 5 | Herring__impulse_v2 | HELDOUT_ONLY | heldout_only_not_authorizing | K0 inert Slow card only; no Target-local carry | L-APPROVE method.py:1466-1492 relation != POSITIVE → delayed_rejected; experience_memory.py:449-451 NEUTRAL is |agg| < 0.005 |
| 6 | GunPoint__burst_cls2 | HELDOUT_ONLY | heldout_only_not_authorizing | K0 inert Slow card only; no Target-local carry | L-APPROVE method.py:1466-1492 relation != POSITIVE → delayed_rejected; experience_memory.py:449-451 NEUTRAL is |agg| < 0.005 |

#### unit 1 GunPointAgeSpan__impulse_v2

- held-in classify_relation=POSITIVE (headroom=0.37499999999999994).  Support Draft + delayed approve would both pass (method.py:742-757 / 1466-1492).
- Target-local Skill may form in-domain; it is NOT carried into the next unit Fast view.
- Slow cannot form Source-derived Skill yet (unguided learnable=1, independent_families=1, intersection=empty).
- authorization_audit TRY authorized=False loo_min=0 withheld=does_not_survive_leave_one_out (source_skill.py:249-256; MIN_DISTINCT_TASKS=2).
- TRY not authorized → T1 inert experience card withheld from Fast (retrieval.py:274).  A5 Fast still equals K0 on this surface.
- Slow: candidate=False independent=1 TRY=False loo=0 withheld=does_not_survive_leave_one_out

#### unit 2 Wine__impulse_v2

- oracle set is identity/empty.  Episode if any is ABSTAIN (classify_relation is_identity; experience_memory.py:428-430).

#### unit 3 GunPoint__impulse_v2

- held-in classify_relation=POSITIVE (headroom=0.4666666666666667).  Support Draft + delayed approve would both pass (method.py:742-757 / 1466-1492).
- Target-local Skill may form in-domain; it is NOT carried into the next unit Fast view.
- Slow may write a Source-derived candidate (Scope v1: n=2 formal learnable, intersection non-empty).
- independence weakened: formal 2 / independent 1 (GunPoint family; identical pattern_view).
- authorization_audit TRY authorized=False loo_min=1 withheld=does_not_survive_leave_one_out (source_skill.py:249-256; MIN_DISTINCT_TASKS=2).
- TRY not authorized → T1 inert experience card withheld from Fast (retrieval.py:274).  A5 Fast still equals K0 on this surface.
- Slow: candidate=True independent=1 TRY=False loo=1 withheld=does_not_survive_leave_one_out

#### unit 4 Ham__impulse_v2

- oracle set is identity/empty.  Episode if any is ABSTAIN (classify_relation is_identity; experience_memory.py:428-430).

#### unit 5 Herring__impulse_v2

- oracle-set program is HELDOUT_ONLY (held-in relation=NEUTRAL, headroom=0.0, held-out utility=0.046875).  Target feedback would not approve; not Source evidence.

#### unit 6 GunPoint__burst_cls2

- oracle-set program is HELDOUT_ONLY (held-in relation=NEUTRAL, headroom=0.0, held-out utility=0.013333333333333308).  Target feedback would not approve; not Source evidence.

### Reverse

order: `['GunPoint__burst_cls2', 'Herring__impulse_v2', 'Ham__impulse_v2', 'GunPoint__impulse_v2', 'Wine__impulse_v2', 'GunPointAgeSpan__impulse_v2']`

first legal Fast-visible difference: **null**

approvable transfer channels: **[]**

| i | unit | learnability | episode | A5 Fast | legality |
|---|---|---|---|---|---|
| 1 | GunPoint__burst_cls2 | HELDOUT_ONLY | heldout_only_not_authorizing | K0 inert Slow card only; no Target-local carry | L-APPROVE method.py:1466-1492 relation != POSITIVE → delayed_rejected; experience_memory.py:449-451 NEUTRAL is |agg| < 0.005 |
| 2 | Herring__impulse_v2 | HELDOUT_ONLY | heldout_only_not_authorizing | K0 inert Slow card only; no Target-local carry | L-APPROVE method.py:1466-1492 relation != POSITIVE → delayed_rejected; experience_memory.py:449-451 NEUTRAL is |agg| < 0.005 |
| 3 | Ham__impulse_v2 | N/A | identity_or_empty | K0 inert Slow card only; no Target-local carry | L-EP AGENTS.md:172-173 Episode 可记，不获执行权 |
| 4 | GunPoint__impulse_v2 | LEARNABLE | positive_unguided | K0 inert Slow card only; no Target-local carry | L-TL-FORM AGENTS.md:174-175 + method.py:742-757 + online_loop.py:201-204 + method.py:1466-1492; L-TL-NOCARRY AGENTS.md:174-175,184-191; 不按 task_kind-only 宽 Scope 放行; L-UNGUIDED 本单元 Fast 无同族 TRY 卡; L-SCOPE 未满 ≥2 独立可学正例或交为空; L-LOO source_skill.py:249-256; L-T1 retrieval.py:274 inert → Fast 不可见 |
| 5 | Wine__impulse_v2 | N/A | identity_or_empty | K0 inert Slow card only; no Target-local carry | L-EP AGENTS.md:172-173 Episode 可记，不获执行权 |
| 6 | GunPointAgeSpan__impulse_v2 | LEARNABLE | positive_unguided | K0 inert Slow card only; no Target-local carry | L-TL-FORM AGENTS.md:174-175 + method.py:742-757 + online_loop.py:201-204 + method.py:1466-1492; L-TL-NOCARRY AGENTS.md:174-175,184-191; 不按 task_kind-only 宽 Scope 放行; L-UNGUIDED 本单元 Fast 无同族 TRY 卡; L-SLOW AGENTS.md:76-81; L-SCOPE Scope v1 五轴 (STAGE_REPORT 2026-08-25 20:1x); L-LOO source_skill.py:249-256; L-T1 retrieval.py:274 inert → Fast 不可见 |

#### unit 1 GunPoint__burst_cls2

- oracle-set program is HELDOUT_ONLY (held-in relation=NEUTRAL, headroom=0.0, held-out utility=0.013333333333333308).  Target feedback would not approve; not Source evidence.

#### unit 2 Herring__impulse_v2

- oracle-set program is HELDOUT_ONLY (held-in relation=NEUTRAL, headroom=0.0, held-out utility=0.046875).  Target feedback would not approve; not Source evidence.

#### unit 3 Ham__impulse_v2

- oracle set is identity/empty.  Episode if any is ABSTAIN (classify_relation is_identity; experience_memory.py:428-430).

#### unit 4 GunPoint__impulse_v2

- held-in classify_relation=POSITIVE (headroom=0.4666666666666667).  Support Draft + delayed approve would both pass (method.py:742-757 / 1466-1492).
- Target-local Skill may form in-domain; it is NOT carried into the next unit Fast view.
- Slow cannot form Source-derived Skill yet (unguided learnable=1, independent_families=1, intersection=empty).
- authorization_audit TRY authorized=False loo_min=0 withheld=does_not_survive_leave_one_out (source_skill.py:249-256; MIN_DISTINCT_TASKS=2).
- TRY not authorized → T1 inert experience card withheld from Fast (retrieval.py:274).  A5 Fast still equals K0 on this surface.
- Slow: candidate=False independent=1 TRY=False loo=0 withheld=does_not_survive_leave_one_out

#### unit 5 Wine__impulse_v2

- oracle set is identity/empty.  Episode if any is ABSTAIN (classify_relation is_identity; experience_memory.py:428-430).

#### unit 6 GunPointAgeSpan__impulse_v2

- held-in classify_relation=POSITIVE (headroom=0.37499999999999994).  Support Draft + delayed approve would both pass (method.py:742-757 / 1466-1492).
- Target-local Skill may form in-domain; it is NOT carried into the next unit Fast view.
- Slow may write a Source-derived candidate (Scope v1: n=2 formal learnable, intersection non-empty).
- independence weakened: formal 2 / independent 1 (GunPoint family; identical pattern_view).
- authorization_audit TRY authorized=False loo_min=1 withheld=does_not_survive_leave_one_out (source_skill.py:249-256; MIN_DISTINCT_TASKS=2).
- TRY not authorized → T1 inert experience card withheld from Fast (retrieval.py:274).  A5 Fast still equals K0 on this surface.
- Slow: candidate=True independent=1 TRY=False loo=1 withheld=does_not_survive_leave_one_out

## 4. Verdict

**HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH**

the frozen 6-unit course still has LEARNABLE oracle-set operators (hampel on GPA and GunPoint; repair_burst on Toe and Lightning2), but neither frozen order has a legal, unguided, subsequently-approvable transfer channel.  Target-local carry is forbidden.  Shared TRY is not authorized (LOO needs 3 unguided positives; each cluster has only 2 LEARNABLE members; hampel independence is 1 family).  T1 therefore withholds any Slow candidate from Fast.  Mechanical search over the same 9-unit pool found 0 pending-arbitration 2+1 rearrangement(s).

### Reorganization search (9-unit pool, no expansion)

over the frozen 9-unit pool, no expansion, no rescoring; a 6-unit course is a hit iff some Program has >=2 LEARNABLE oracle-set sources whose Scope-v1 pattern intersection is non-empty, and a later unit is a LEARNABLE matching field for that same Program.  GunPoint family independence is reported, not used as a silent veto on the mechanical hit test (a hit still carries the independence note).

candidates pending arbitration: **0**

- program `hampel_filter`: learnable=['GunPointAgeSpan__impulse_v2', 'GunPoint__impulse_v2'] independent_families=['GunPointFamily'] matching_fields=['GunPointAgeSpan__impulse_v2', 'GunPoint__impulse_v2'] blocked=['fewer_than_2_independent_families', 'no_external_or_third_learnable_matching_field']
- program `repair_burst_segment`: learnable=['ToeSegmentation1__impulse_v2', 'Lightning2__impulse_v2'] independent_families=['Lightning2', 'ToeSegmentation1'] matching_fields=['ToeSegmentation1__impulse_v2', 'Lightning2__impulse_v2'] blocked=['no_external_or_third_learnable_matching_field']

## 5. S1b domain-binding spec

The exam asks whether legal evolution changes Fast behaviour.  Current Target-local cards match every classification unit (task_kind-only applicability).  Running S1b against that matcher measures the wide-Scope bug (copying a frozen winner across domains), not Harness evolution.  A runner-layer filter implements AGENTS.md:174-191 as already written.  It does not change held-in budgets, menus, Consumer, splits, or oracle keys.

- **cell / unit construction**: stamp every newly minted Target-local Skill with domain_namespace = current unit dataset (already on the Episode; copy it onto the Skill entry or a runner-owned side table).  Do not put dataset name into observable_applicability.
- **cross-unit snapshot carry into the next Fast view**: drop any Target-local capability (frozen program steps; not an experience card — retrieval.py:158-164) whose domain_namespace != current unit.  This is the AGENTS.md:184-191 wall.
- **Source-derived experience cards**: admit them to Fast only when Scope v1 matches: task_kind × consumer_id × metric × pattern_view intersection × Program geometry.  Dataset name is not an axis.  If methods-layer Scope compile step ③ is not yet live, the runner evaluates this 5-axis predicate as an exam-wall before retrieve.

四步修复序第③步（STAGE_REPORT 2026-08-25 17:3x / 17:35）才是 methods 层 Scope 编译：Target-local 限本域；跨域绑 Task×Consumer×Metric+部署可见 Pattern+Program 几何。本书不改 methods/。理由：单假设纪律；③ 是行为机制变更，需要自己的切片、锁与测试；S1b 只需要考试墙与正典对齐，避免把宽 Scope bug 当成处理组。③ 落地后删除 runner 墙，不得长期叠两道门。

## Cost

- Fast LLM: 0
- Slow LLM: 0
- Consumer fits: 0 (this pass 0)
- wall clock: 0.02 s / 5400 s
- downloads: 0
- oracle rescoring: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **no_fast_llm**: True
- **no_slow_llm**: True
- **no_a3_a5_adaptation_arm**: True
- **no_oracle_rescore**: True
- **no_injection_scan**: True
- **no_pool_expansion**: True
- **r1_artifacts_not_overwritten**: True
- **sealed_oracles_not_rewritten**: True
- **downloads**: 0
- **this_book_ran_no_adaptation_arm_and_did_not_recompute_oracle_numbers**: True
- **wall_clock_held**: True
- **full_repo_pytest_not_run**: True

## Outside the book

- authorization_audit LOO with min_distinct_tasks=2 requires 3 unguided positives (loo_minimum after dropping one is n-1).  r1 treated 2 cluster positives as enough for TRY.  Live code: source_skill.py:249-256.
- ECG200 repair_burst_segment is HELDOUT_ONLY: held-in headroom=0 / held-out +0.04 (s1_oracle/ECG200__impulse_v2.json programs row).  Same failure mode as Herring hampel.  The arbitration preview that the burst-repair cluster might host a legal course is not supported by the sealed held-in numbers.
- GunPoint burst outlier_iqr is HELDOUT_ONLY (held-in=0 / held-out +0.0133).
- ToeSegmentation1 hampel_filter held-in is POSITIVE (+0.0833, one row at n=12) but is not in the oracle set (held-out utility 0).  Official table does not count it.  Even as an extra hampel source it disagrees with GPA/GP on period_change_score, so it does not match the GPA∩GP Scope; including it as a third source still yields no LEARNABLE matching field.
- GPA and GunPoint pattern_view are byte-equal.  A Scope built from only those two is the full pattern; no other pool unit matches it.
- classification online_loop still does not write task_episode_id; source_skill.build_skill_payload still does not write evidence_distinct_task_count; Fast-guard stays off.  Unchanged from r1 outside-book.
