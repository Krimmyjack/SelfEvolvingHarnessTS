# S1-diag -- behavior funnel + candidate-cap pairing

protocol: `s1_cold_policy_map_v2_arbitration`  git: `258c28d66fb1029b1ab8eaf552c4136368f463c5`  Part B LLM: 12 / 12

Original Part B (full-menu ranking) is **void** and was not run.

voided_not_run: the full-menu ranking probe was never executed in this session; there is no exploratory ranking material to archive

## Part A -- behavior funnel (0 LLM)

S1c and the listed historical runners persist DecisionTrace.candidate_ids as `pool` after compile + noop filter + CandidatePool.build(total_k) + verifier selectable filter.  They do not persist the propose-stage payload (all raw Agent candidates before truncation/verify).  `proposal_count` is len(non-identity pool after that filter).  Funnel therefore starts at compiled_verified_pool, not at the raw LLM list.  Operator names on unexecuted pool members are recovered from candidate_id aliases; executed members have workflow_signature.

Earliest available layer: **compiled_verified_pool**.  Raw proposals persisted: **False**.

### Wording (required)

在当前课程、当前 Prompt、当前候选预算、这一次随机运行中，15 个含菜单正解的臂-单元机会命中 1 次（冻结部署口径）。同一 15 个机会里，菜单正解被执行 4 次、在编译池/执行层出现 4 次。这是这一次冷提案策略的召回读数，不是数据就绪的固有难度，也不是稳定概率。

### S1c 15 opportunities (do not mix other strata)

| unit | arm | oracle | proposed/exec layer | executed | deployed | deepest breakpoint |
|---|---|---|---|---|---|---|
| MiddlePhalanxOutlineCorrect__impulse_v2 | A3-reset | repair_level_shift | True | True | False | executed_not_passed |
| MiddlePhalanxOutlineCorrect__impulse_v2 | K0-fixed | repair_level_shift | True | True | False | executed_not_passed |
| MiddlePhalanxOutlineCorrect__impulse_v2 | A5-online | repair_level_shift | True | True | False | executed_not_passed |
| DistalPhalanxOutlineCorrect__burst_cls2 | A3-reset | outlier_iqr | False | False | False | not_proposed |
| DistalPhalanxOutlineCorrect__burst_cls2 | K0-fixed | outlier_iqr | False | False | False | not_proposed |
| DistalPhalanxOutlineCorrect__burst_cls2 | A5-online | outlier_iqr | False | False | False | not_proposed |
| PowerCons__impulse_v2 | A3-reset | hampel_filter | True | True | True | deployed |
| PowerCons__impulse_v2 | K0-fixed | hampel_filter | False | False | False | not_proposed |
| PowerCons__impulse_v2 | A5-online | hampel_filter | False | False | False | not_proposed |
| GunPointOldVersusYoung__impulse_v2 | A3-reset | hampel_filter | False | False | False | not_proposed |
| GunPointOldVersusYoung__impulse_v2 | K0-fixed | hampel_filter | False | False | False | not_proposed |
| GunPointOldVersusYoung__impulse_v2 | A5-online | hampel_filter | False | False | False | not_proposed |
| ECG200__impulse_v2 | A3-reset | repair_burst_segment | False | False | False | not_proposed |
| ECG200__impulse_v2 | K0-fixed | repair_burst_segment | False | False | False | not_proposed |
| ECG200__impulse_v2 | A5-online | repair_burst_segment | False | False | False | not_proposed |

Breakpoint counts: `{"executed_not_passed": 3, "not_proposed": 11, "deployed": 1}`

Attribution: 未提出=提案召回 / 提出未选=选择或弃权 / 选了被拒=合法性 / 执行后未过=反馈或效果。

### Slow boundaries (A5)

each boundary spent 1 of 6 allowed Slow LLM calls and wrote source_investigation_cls_v1 (already in K0, dropped at the next-unit wall).  authorized_try and risk_authorized stayed empty because the census had no unguided positive and no two-Task same-operator harm.  The empty carry is missing compilable evidence, not a Slow-budget starve.

### Historical strata (not mixed into 1/15)

- `t6_cls_op_r2_three_arms`  rounds=8  protocol=`t6_cls_op_r2_three_arms_v1`  git=`cb03eb688210f521c931895454b44d30048c1928`
- `t6_cls_op_r2_a5_replay`  rounds=2  protocol=`t6_cls_op_r2_a5_replay_v1`  git=`03f2c1bc8eb42216935b4aabd7f02895279927ca`
- `t6_cls_conf_dev_ecg200`  rounds=2  protocol=`t6_cls_conf_dev_v1`  git=`168cc99a39873ae8c1f7c51e9a6d0aa76186474c`

## Part B -- candidate-cap pairing (propose only)

one variable K, realized jointly by fast_propose_v1 maxItems, harness-view candidate_policy.agent_program_slots/total_k, and TaskContext.deployment_constraints.maximum_candidates.  Original K=3 is the S1c live cap (schema maxItems=3 and TaskContext maximum_candidates=3).  Expanded K=5 is the same observation, instruction text, menu contracts, and frozen inspect, with only those three K fields raised.  methods/h0 files were not written.

inspect was frozen from public features and shared across the pair.  S1c's stochastic inspect payload was not persisted, so a live inspect replay would have added a second random variable and would have exhausted the 12-call cap (inspect+tools+two proposes).  This isolates K; it is not a claim that the inspect equals S1c's inspect.

Isolation: Part B outputs are diagnostic isolation material.  They must not enter any future arm Fast view, Skill store, Memory, or prompt.  They are not a ranking of the menu and they do not authorize a program.

| unit | oracle (posthoc) | K=3 ops / hit | K=5 ops / hit | reading |
|---|---|---|---|---|
| PowerCons__impulse_v2 | hampel_filter | repair_level_shift / False | repair_level_shift / False | proposal_semantics_insufficient |
| GunPointOldVersusYoung__impulse_v2 | hampel_filter | (empty) / False | (empty) / False | proposal_semantics_insufficient |
| DistalPhalanxOutlineCorrect__burst_cls2 | outlier_iqr | error / False | repair_level_shift / False | proposal_semantics_insufficient |
| MiddlePhalanxOutlineCorrect__impulse_v2 | repair_level_shift | (empty) / False | (empty) / False | incomplete_pair |

Frozen reading: **proposal_semantics_insufficient**

K=5 在本次已完成的 3 个单元抽签中均未出现菜单正解（PowerCons / GPOVY / Distal；ECG200 未跑，MiddlePhalanx 因帽满未提案）。扩大帽也没有把候选数从 0–1 抬到接近 5——槽位未被用满，本次看不到截断解释。按冻结判读只写「提案语义不足」，不得再拆成 observation 不足或策略偏置。n=1 对/单元，不是稳定概率。

S1c A3-reset original-K column (already paid, not inspect-paired):

- MiddlePhalanxOutlineCorrect__impulse_v2 oracle=repair_level_shift proposed/exec=True executed=True deployed=False breakpoint=executed_not_passed
- DistalPhalanxOutlineCorrect__burst_cls2 oracle=outlier_iqr proposed/exec=False executed=False deployed=False breakpoint=not_proposed
- PowerCons__impulse_v2 oracle=hampel_filter proposed/exec=True executed=True deployed=True breakpoint=deployed
- GunPointOldVersusYoung__impulse_v2 oracle=hampel_filter proposed/exec=False executed=False deployed=False breakpoint=not_proposed
- ECG200__impulse_v2 oracle=repair_burst_segment proposed/exec=False executed=False deployed=False breakpoint=not_proposed

## Graded-hypothesis track

在这一次 S1c 课程里，15 个含菜单正解的臂-单元机会有 11 个断在「未提出」、3 个断在「执行后未过」（MiddlePhalanx 三臂都执行了 repair_level_shift 但 delayed 未批准）、1 个走通部署。因此「一张 Scope 匹配的 hypothesis 卡提高正确族进入有限帽的概率」是针对召回断点的合理假设，待因果实验验证，不是本诊断或 S1c 已经证明的事实。仍须走完：合法独立 Episode → 机器 Scope 匹配 → 专用 prior 槽 → 提案分布实测改变 → Target 自批 Support/delayed → 成本或 regret 改善且 harm 不升。本书封顶，无 diag-r2/r3。

## Obligations

- **methods_runtime_contracts_operators_unmodified**: True
- **existing_runners_unmodified**: True
- **course_and_budgets_unmodified**: True
- **downloads**: 0
- **consumer_fits**: 0
- **llm_part_a**: 0
- **llm_part_b**: 12
- **llm_cap**: 12
- **ranking_probe_not_run**: True
- **probe_outputs_isolated**: True
- **sealed_oracles_not_rewritten**: True
- **full_repo_pytest_not_run**: True
- **no_diag_r2_r3**: True
- **wording_1_of_15_is_this_run_only**: True
- **graded_hypothesis_is_hypothesis_not_fact**: True
