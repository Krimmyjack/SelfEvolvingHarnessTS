# CLS-CONF -- frozen confirmation on an unused UCR Target

protocol: `t6_cls_conf_unused_target_v1`  target: **None**  evidence grade: **DEVELOPMENT**

## Verdict

**INSTRUMENT_UNREADABLE**



- non-identity Target-local Skill formed: None
- A3 minus Static held-out accuracy: None (material line None)
- worst per-class recall delta: None (zero class harm: None)
- deployment purity: None



## Part A -- how the Target was chosen

candidate pool = every zip in data/ucr_task_context whose name no non-inventory file in the repository mentions; keep the binary ones whose official TRAIN row count is in [40, 400]; take the lexicographically first. A file naming >= 30 of the pool is a data inventory, not an experiment roster, and its mentions are not usage.  Fixed before the rule was applied and independent of any outcome.

- pool: 40 zips; eligible: []; selected: **None** (no eligible dataset)

| dataset | usage hits | claiming runner(s) | train rows | classes | excluded because |
|---|---|---|---|---|---|
| BeetleFly | 17 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_shared_capability_s0_census.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 20 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| BirdChicken | 19 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 20 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| Coffee | 23 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py | 28 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| Computers | 6 | run_e2_integrated_context_harness_evolution.py | 250 | 2 | name_already_appears_in_the_repository |
| DistalPhalanxOutlineCorrect | 4 | run_e2_action_credit_candidate_ordering.py | 600 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| DodgerLoopWeekend | 1 | (none) | None | None | not_loadable_as_binary_ucr, name_already_appears_in_the_repository |
| Earthquakes | 4 | run_e2_action_credit_candidate_ordering.py | 322 | 2 | name_already_appears_in_the_repository |
| ECG200 | 119 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls1_qualification_gate.py, run_e2_t6_cls1_r2_qualification_gate.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py, run_e2_temporary_excursion_skill_headroom.py | 100 | 2 | name_already_appears_in_the_repository |
| ECGFiveDays | 10 | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 23 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| FordA | 28 | run_e2_curvature_corrected_action_credit.py, run_e2_shared_capability_s0_census.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py | 3601 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| FordB | 14 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 3636 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| FreezerRegularTrain | 6 | run_e2_source_prior_evidence_fusion.py | 150 | 2 | name_already_appears_in_the_repository |
| FreezerSmallTrain | 4 | run_e2_task_risk_confirmation_adaptation_curve.py | 28 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| GunPoint | 144 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls1_qualification_gate.py, run_e2_t6_cls1_r2_qualification_gate.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_t6_cls3_paired_consumer_gate.py, run_e2_t6_cls4_burst_repair_gate.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py, run_e2_temporary_excursion_skill_headroom.py | 50 | 2 | name_already_appears_in_the_repository |
| GunPointAgeSpan | 155 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 135 | 2 | name_already_appears_in_the_repository |
| GunPointMaleVersusFemale | 6 | run_e2_source_prior_evidence_fusion.py | 135 | 2 | name_already_appears_in_the_repository |
| GunPointOldVersusYoung | 6 | run_e2_source_prior_evidence_fusion.py | 136 | 2 | name_already_appears_in_the_repository |
| Ham | 379 | run_e2_source_outlier_local_behavior_audit.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_task_conditioned_impulse_repair_control.py, run_e2_task_risk_confirmation_adaptation_curve.py | 109 | 2 | name_already_appears_in_the_repository |
| HandOutlines | 5 | run_e2_integrated_context_harness_evolution.py | 1000 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| Herring | 4 | run_e2_task_risk_confirmation_adaptation_curve.py | 64 | 2 | name_already_appears_in_the_repository |
| HouseTwenty | 18 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 40 | 2 | name_already_appears_in_the_repository |
| KeplerLightCurves | 1 | (none) | None | None | not_loadable_as_binary_ucr, name_already_appears_in_the_repository |
| Lightning2 | 87 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 60 | 2 | name_already_appears_in_the_repository |
| MiddlePhalanxOutlineCorrect | 102 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 600 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| MoteStrain | 11 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 20 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| PhalangesOutlinesCorrect | 164 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 1800 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| PowerCons | 5 | run_e2_integrated_context_harness_evolution.py | 180 | 2 | name_already_appears_in_the_repository |
| ProximalPhalanxOutlineCorrect | 103 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 600 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| SemgHandGenderCh2 | 5 | run_e2_integrated_context_harness_evolution.py | 300 | 2 | name_already_appears_in_the_repository |
| ShapeletSim | 4 | run_e2_action_credit_candidate_ordering.py | 20 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| SonyAIBORobotSurface1 | 11 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 20 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| SonyAIBORobotSurface2 | 20 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 27 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| Strawberry | 4 | run_e2_task_risk_confirmation_adaptation_curve.py | 613 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| ToeSegmentation1 | 24 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 40 | 2 | name_already_appears_in_the_repository |
| ToeSegmentation2 | 6 | run_e2_source_prior_evidence_fusion.py | 36 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| TwoLeadECG | 16 | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 23 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| Wafer | 10 | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 1000 | 2 | train_rows_outside_40_400, name_already_appears_in_the_repository |
| Wine | 4 | run_e2_action_credit_candidate_ordering.py | 57 | 2 | name_already_appears_in_the_repository |
| WormsTwoClass | 5 | run_e2_integrated_context_harness_evolution.py | 181 | 2 | name_already_appears_in_the_repository |
| Yoga | 5 | run_e2_integrated_context_harness_evolution.py | 300 | 2 | name_already_appears_in_the_repository |

### Pool exhausted

the local UCR inventory is exhausted with respect to this book's own selection rule: every one of the 40 datasets in data/ucr_task_context is already a roster member of some prior classification experiment in this repository, and the only two whose sole mention is incidental cannot be loaded as binary UCR at all.

Datasets claimed by prior runners:

- `evaluation/functional/run_e2_action_credit_candidate_ordering.py`: DistalPhalanxOutlineCorrect, Earthquakes, ShapeletSim, Wine
- `evaluation/functional/run_e2_curvature_corrected_action_credit.py`: BeetleFly, Coffee, ECG200, ECGFiveDays, FordA, GunPoint, TwoLeadECG, Wafer
- `evaluation/functional/run_e2_integrated_context_harness_evolution.py`: Computers, HandOutlines, PowerCons, SemgHandGenderCh2, WormsTwoClass, Yoga
- `evaluation/functional/run_e2_pattern_mass_multiplicity_headroom.py`: BeetleFly, Coffee, ECG200, GunPoint
- `evaluation/functional/run_e2_program_binding_harness_update.py`: FordB, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, ProximalPhalanxOutlineCorrect, SonyAIBORobotSurface1
- `evaluation/functional/run_e2_promoted_binding_capability_transfer.py`: BirdChicken, GunPointAgeSpan, HouseTwenty, PhalangesOutlinesCorrect, SonyAIBORobotSurface2, ToeSegmentation1
- `evaluation/functional/run_e2_shared_capability_s0_census.py`: BeetleFly, FordA
- `evaluation/functional/run_e2_source_outlier_local_behavior_audit.py`: Ham
- `evaluation/functional/run_e2_source_prior_evidence_fusion.py`: FreezerRegularTrain, GunPointMaleVersusFemale, GunPointOldVersusYoung, ToeSegmentation2
- `evaluation/functional/run_e2_t6_cls1_qualification_gate.py`: ECG200, GunPoint
- `evaluation/functional/run_e2_t6_cls1_r2_qualification_gate.py`: ECG200, GunPoint
- `evaluation/functional/run_e2_t6_cls2_value_corruption_gate.py`: ECG200, GunPoint, Ham
- `evaluation/functional/run_e2_t6_cls3_paired_consumer_gate.py`: GunPoint
- `evaluation/functional/run_e2_t6_cls4_burst_repair_gate.py`: GunPoint
- `evaluation/functional/run_e2_t6_cls_op_shared_harness.py`: BeetleFly, BirdChicken, Coffee, ECG200, ECGFiveDays, FordA, FordB, GunPoint, GunPointAgeSpan, HouseTwenty, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, PhalangesOutlinesCorrect, ProximalPhalanxOutlineCorrect, SonyAIBORobotSurface1, SonyAIBORobotSurface2, ToeSegmentation1, TwoLeadECG, Wafer
- `evaluation/functional/run_e2_task_conditioned_impulse_repair_control.py`: Ham
- `evaluation/functional/run_e2_task_context_label_evidence_witness.py`: Coffee, ECG200, FordA, GunPoint
- `evaluation/functional/run_e2_task_risk_action_credit_transfer.py`: BeetleFly, Coffee, ECG200, ECGFiveDays, FordA, GunPoint, TwoLeadECG, Wafer
- `evaluation/functional/run_e2_task_risk_confirmation_adaptation_curve.py`: FreezerSmallTrain, Ham, Herring, Strawberry
- `evaluation/functional/run_e2_temporary_excursion_skill_headroom.py`: ECG200, GunPoint

Loader rejects: DodgerLoopWeekend, KeplerLightCurves

- **DodgerLoopWeekend**: binary labels but the table is not finite (the series carry missing values), and 20 TRAIN rows is below the row floor anyway
- **KeplerLightCurves**: the archive contains no <name>_TRAIN.txt member; it is packaged differently from the rest of the pool

no rule was relaxed to manufacture a Target.  Widening 'unused' after seeing that the strict pool is empty would be choosing the confirmation set with the answer in view.

Options for the mainline:

- narrow 'unused' to 'never used under the C38 impulse family', which would admit the six datasets whose only prior use was run_e2_integrated_context_harness_evolution and the three from run_e2_source_prior_evidence_fusion -- weaker than a virgin Target but still outside the impulse line
- confirm on a frozen split of an already-used dataset that the impulse line never scored, and say plainly that it is a split-level rather than dataset-level confirmation
- authorise one download, which the current discipline forbids

## Part B -- two arms

| arm | Skill formed | first-Skill LLM | first-Skill executions | held-in delayed | held-out acc | vs identity | worst class recall d | abstained rounds | Support/delayed agree:disagree |
|---|---|---|---|---|---|---|---|---|---|

## Budget

- LLM: 0 of 40
- Consumer fits: 0 of 200
- wall clock: 17.3 s

## Obligations

- **selection_rule_not_relaxed**: True
- **llm_spent_before_stopping**: 0
- **downloads**: 0
- **methods_package_unmodified**: True
- **artifact_not_committed**: True
