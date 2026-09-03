# CLS-CONF-r3a -- semantic runner audit and unused-target selection

protocol: `t6_cls_conf_unused_target_v3a`  evidence grade: **DEVELOPMENT**

> ## CANDIDATE POOL EMPTY
>
> Semantic audit found no dataset that is unused under the
> impulse condition pair **and** binary / TRAIN in [40, 400] / loadable.
> Exclusion chains are in the 40-row table. Arms were not run.

## Verdict

**CANDIDATE_POOL_EMPTY**

after semantic runner audit, no dataset is unused under the impulse pair and also binary / TRAIN[40,400] / loadable; arms not run

- selected: **None**
- eligible: []
- arms run: 0
- LLM: 0; Consumer fit: 0; downloads: 0

## Rule

condition_pair_used iff a claiming runner is EXECUTES_CONDITION_PAIR and the dataset is in that runner's actual roster (constants/calls, not file-name mentions). Eligible = NOT condition_pair_used AND binary AND official TRAIN rows in [40, 400] AND loadable. Selected = lexicographic first eligible. No new rule.

r1/r2 artifacts were not overwritten. Census cache reused (`_scratch/_conf_name_census.json`).

## Part A -- 20-runner semantic audit

counts: EXECUTES_CONDITION_PAIR=10, INCIDENTAL_TOKEN=0, NO_TOKEN=10

| runner | class | roster | evidence |
|---|---|---|---|
| `run_e2_action_credit_candidate_ordering.py` | **EXECUTES_CONDITION_PAIR** | DistalPhalanxOutlineCorrect, Earthquakes, ShapeletSim, Wine | `evaluation/functional/run_e2_action_credit_candidate_ordering.py:34` TARGET_DATASETS = ("ShapeletSim", "Wine", "Earthquakes", "DistalPhalanxOutlineCorrect",) |
| `run_e2_curvature_corrected_action_credit.py` | **EXECUTES_CONDITION_PAIR** 存疑归严 | BeetleFly, Coffee, ECG200, ECGFiveDays, FordA, GunPoint, TwoLeadECG, Wafer | `evaluation/functional/run_e2_curvature_corrected_action_credit.py:34` DATASETS = ("Coffee", "ECG200", "FordA", "GunPoint", "Wafer", "ECGFiveDays", "TwoLeadECG", "BeetleFly",) |
| `run_e2_integrated_context_harness_evolution.py` | **EXECUTES_CONDITION_PAIR** | Computers, HandOutlines, PowerCons, SemgHandGenderCh2, WormsTwoClass, Yoga | `evaluation/functional/run_e2_integrated_context_harness_evolution.py:38` TARGET_DATASETS = ("Computers", "PowerCons", "Yoga", "SemgHandGenderCh2", "WormsTwoClass", "HandOutlines",) |
| `run_e2_pattern_mass_multiplicity_headroom.py` | **NO_TOKEN** | BeetleFly, Coffee, ECG200, GunPoint | `evaluation/functional/run_e2_pattern_mass_multiplicity_headroom.py:28` DATASETS = ("Coffee", "ECG200", "GunPoint", "BeetleFly") |
| `run_e2_program_binding_harness_update.py` | **EXECUTES_CONDITION_PAIR** | FordB, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, ProximalPhalanxOutlineCorrect, SonyAIBORobotSurface1 | `evaluation/functional/run_e2_program_binding_harness_update.py:32` TARGET_DATASETS = ("FordB", "Lightning2", "MoteStrain", "SonyAIBORobotSurface1", "ProximalPhalanxOutlineCorrect", "MiddlePhalanxOutlineCorrect",) |
| `run_e2_promoted_binding_capability_transfer.py` | **EXECUTES_CONDITION_PAIR** | BirdChicken, GunPointAgeSpan, HouseTwenty, PhalangesOutlinesCorrect, SonyAIBORobotSurface2, ToeSegmentation1 | `evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34` TARGET_DATASETS = ("BirdChicken", "HouseTwenty", "ToeSegmentation1", "PhalangesOutlinesCorrect", "SonyAIBORobotSurface2", "GunPointAgeSpan",) |
| `run_e2_shared_capability_s0_census.py` | **NO_TOKEN** | (none) | `evaluation/functional/run_e2_shared_capability_s0_census.py:285` "UCR archive (BeetleFly, FordA, ... ), and the .ts classification "  # comment mention only; forecasting census, no UCR impulse run |
| `run_e2_source_outlier_local_behavior_audit.py` | **NO_TOKEN** | (none) | `evaluation/functional/run_e2_source_cohort_policy_premise.py:39` SOURCE_DATASETS = ("monash:traffic_hourly", "metr_la")  # imported; not UCR Ham |
| `run_e2_source_prior_evidence_fusion.py` | **EXECUTES_CONDITION_PAIR** | FreezerRegularTrain, GunPointMaleVersusFemale, GunPointOldVersusYoung, ToeSegmentation2 | `evaluation/functional/run_e2_source_prior_evidence_fusion.py:38` TARGET_DATASETS = ("FreezerRegularTrain", "ToeSegmentation2", "GunPointMaleVersusFemale", "GunPointOldVersusYoung",) |
| `run_e2_t6_cls1_qualification_gate.py` | **NO_TOKEN** | GunPoint | `evaluation/functional/run_e2_t6_cls1_qualification_gate.py:59` DATASET = "GunPoint" |
| `run_e2_t6_cls1_r2_qualification_gate.py` | **NO_TOKEN** | ECG200, GunPoint | `evaluation/functional/run_e2_t6_cls1_r2_qualification_gate.py:54` LADDER = ("GunPoint", "ECG200") |
| `run_e2_t6_cls2_value_corruption_gate.py` | **NO_TOKEN** | GunPoint | `evaluation/functional/run_e2_t6_cls2_value_corruption_gate.py:58` DATASET = "GunPoint" |
| `run_e2_t6_cls3_paired_consumer_gate.py` | **NO_TOKEN** | GunPoint | `evaluation/functional/run_e2_t6_cls3_paired_consumer_gate.py:48` DATASET = "GunPoint" |
| `run_e2_t6_cls4_burst_repair_gate.py` | **NO_TOKEN** | GunPoint | `evaluation/functional/run_e2_t6_cls4_burst_repair_gate.py:654` "dataset": "GunPoint" |
| `run_e2_t6_cls_op_shared_harness.py` | **EXECUTES_CONDITION_PAIR** | GunPointAgeSpan, Lightning2, MiddlePhalanxOutlineCorrect, PhalangesOutlinesCorrect, ProximalPhalanxOutlineCorrect | `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:132` SOURCE_DATASETS = ("ProximalPhalanxOutlineCorrect", "MiddlePhalanxOutlineCorrect", "Lightning2") |
| `run_e2_task_conditioned_impulse_repair_control.py` | **NO_TOKEN** | (none) | `evaluation/functional/run_e2_task_conditioned_impulse_repair_control.py:27` DATASETS = ("monash:traffic_hourly", "legacy_monash:fred_md")  # not UCR Ham |
| `run_e2_task_context_label_evidence_witness.py` | **EXECUTES_CONDITION_PAIR** | Coffee, ECG200, FordA, GunPoint | `evaluation/functional/run_e2_task_context_label_evidence_witness.py:33` DATASETS = ("Coffee", "ECG200", "FordA", "GunPoint") |
| `run_e2_task_risk_action_credit_transfer.py` | **EXECUTES_CONDITION_PAIR** | BeetleFly, Coffee, ECG200, ECGFiveDays, FordA, GunPoint, TwoLeadECG, Wafer | `evaluation/functional/run_e2_task_risk_action_credit_transfer.py:31` SOURCE_DATASETS = ("Coffee", "ECG200", "FordA", "GunPoint") |
| `run_e2_task_risk_confirmation_adaptation_curve.py` | **EXECUTES_CONDITION_PAIR** | FreezerSmallTrain, Ham, Herring, Strawberry | `evaluation/functional/run_e2_task_risk_confirmation_adaptation_curve.py:40` TARGET_DATASETS = ("Herring", "Ham", "FreezerSmallTrain", "Strawberry") |
| `run_e2_temporary_excursion_skill_headroom.py` | **NO_TOKEN** | ECG200, GunPoint | `evaluation/functional/run_e2_temporary_excursion_skill_headroom.py:27` DATASETS = ("ECG200", "GunPoint") |

### Conservative-uncertain (存疑归严)

- `evaluation/functional/run_e2_curvature_corrected_action_credit.py`: 存疑归严: file tokens only read W48/W49 report fields; run() also re-applies W48 _inject on TRAIN fit-only for Woodbury calibration (no TEST, no named CONDITIONS, no stable_task_event). Counted as executing the fit-only half of the pair.

### source_prior vs r2 brief

r2/task brief treated file-local tokens as incidental field reads. evaluate() reuses W56/W55 planner which injects both conditions, then W56 _prepare_train_execution injects fit_only_artifact and opens TEST. Semantic standard => EXECUTES.

## Part B -- 40-dataset eligibility

| dataset | used | TRAIN | classes | loadable | claiming (executing) | excluded | evidence |
|---|---|---|---|---|---|---|---|
| BeetleFly | yes | 20 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_shared_capability_s0_census.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_risk_action_credit_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| BirdChicken | yes | 20 | 2 | yes | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_promoted_binding_capability_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34 |
| Coffee | yes | 28 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| Computers | yes | 250 | 2 | yes | run_e2_integrated_context_harness_evolution.py / **run_e2_integrated_context_harness_evolution.py** | condition_pair_used | evaluation/functional/run_e2_integrated_context_harness_evolution.py:38 |
| DistalPhalanxOutlineCorrect | yes | 600 | 2 | yes | run_e2_action_credit_candidate_ordering.py / **run_e2_action_credit_candidate_ordering.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_action_credit_candidate_ordering.py:34 |
| DodgerLoopWeekend | no | — | — | no | (none) / **(none)** | not_loadable_as_binary_ucr | ValueError: invalid UCR table: DodgerLoopWeekend/TRAIN |
| ECG200 | yes | 100 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls1_qualification_gate.py, run_e2_t6_cls1_r2_qualification_gate.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py, run_e2_temporary_excursion_skill_headroom.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py** | condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| ECGFiveDays | yes | 23 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_risk_action_credit_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| Earthquakes | yes | 322 | 2 | yes | run_e2_action_credit_candidate_ordering.py / **run_e2_action_credit_candidate_ordering.py** | condition_pair_used | evaluation/functional/run_e2_action_credit_candidate_ordering.py:34 |
| FordA | yes | 3601 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_shared_capability_s0_census.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| FordB | yes | 3636 | 2 | yes | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_program_binding_harness_update.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_program_binding_harness_update.py:32 |
| FreezerRegularTrain | yes | 150 | 2 | yes | run_e2_source_prior_evidence_fusion.py / **run_e2_source_prior_evidence_fusion.py** | condition_pair_used | evaluation/functional/run_e2_source_prior_evidence_fusion.py:38 |
| FreezerSmallTrain | yes | 28 | 2 | yes | run_e2_task_risk_confirmation_adaptation_curve.py / **run_e2_task_risk_confirmation_adaptation_curve.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_task_risk_confirmation_adaptation_curve.py:40 |
| GunPoint | yes | 50 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls1_qualification_gate.py, run_e2_t6_cls1_r2_qualification_gate.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_t6_cls3_paired_consumer_gate.py, run_e2_t6_cls4_burst_repair_gate.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py, run_e2_temporary_excursion_skill_headroom.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py** | condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| GunPointAgeSpan | yes | 135 | 2 | yes | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py** | condition_pair_used | evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34 |
| GunPointMaleVersusFemale | yes | 135 | 2 | yes | run_e2_source_prior_evidence_fusion.py / **run_e2_source_prior_evidence_fusion.py** | condition_pair_used | evaluation/functional/run_e2_source_prior_evidence_fusion.py:38 |
| GunPointOldVersusYoung | yes | 136 | 2 | yes | run_e2_source_prior_evidence_fusion.py / **run_e2_source_prior_evidence_fusion.py** | condition_pair_used | evaluation/functional/run_e2_source_prior_evidence_fusion.py:38 |
| Ham | yes | 109 | 2 | yes | run_e2_source_outlier_local_behavior_audit.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_task_conditioned_impulse_repair_control.py, run_e2_task_risk_confirmation_adaptation_curve.py / **run_e2_task_risk_confirmation_adaptation_curve.py** | condition_pair_used | evaluation/functional/run_e2_task_risk_confirmation_adaptation_curve.py:40 |
| HandOutlines | yes | 1000 | 2 | yes | run_e2_integrated_context_harness_evolution.py / **run_e2_integrated_context_harness_evolution.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_integrated_context_harness_evolution.py:38 |
| Herring | yes | 64 | 2 | yes | run_e2_task_risk_confirmation_adaptation_curve.py / **run_e2_task_risk_confirmation_adaptation_curve.py** | condition_pair_used | evaluation/functional/run_e2_task_risk_confirmation_adaptation_curve.py:40 |
| HouseTwenty | yes | 40 | 2 | yes | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_promoted_binding_capability_transfer.py** | condition_pair_used | evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34 |
| KeplerLightCurves | no | — | — | no | (none) / **(none)** | not_loadable_as_binary_ucr | KeyError: "There is no item named 'KeplerLightCurves_TRAIN.txt' in the archive" |
| Lightning2 | yes | 60 | 2 | yes | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py** | condition_pair_used | evaluation/functional/run_e2_program_binding_harness_update.py:32 |
| MiddlePhalanxOutlineCorrect | yes | 600 | 2 | yes | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_program_binding_harness_update.py:32 |
| MoteStrain | yes | 20 | 2 | yes | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_program_binding_harness_update.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_program_binding_harness_update.py:32 |
| PhalangesOutlinesCorrect | yes | 1800 | 2 | yes | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34 |
| PowerCons | yes | 180 | 2 | yes | run_e2_integrated_context_harness_evolution.py / **run_e2_integrated_context_harness_evolution.py** | condition_pair_used | evaluation/functional/run_e2_integrated_context_harness_evolution.py:38 |
| ProximalPhalanxOutlineCorrect | yes | 600 | 2 | yes | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_program_binding_harness_update.py:32 |
| SemgHandGenderCh2 | yes | 300 | 2 | yes | run_e2_integrated_context_harness_evolution.py / **run_e2_integrated_context_harness_evolution.py** | condition_pair_used | evaluation/functional/run_e2_integrated_context_harness_evolution.py:38 |
| ShapeletSim | yes | 20 | 2 | yes | run_e2_action_credit_candidate_ordering.py / **run_e2_action_credit_candidate_ordering.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_action_credit_candidate_ordering.py:34 |
| SonyAIBORobotSurface1 | yes | 20 | 2 | yes | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_program_binding_harness_update.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_program_binding_harness_update.py:32 |
| SonyAIBORobotSurface2 | yes | 27 | 2 | yes | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_promoted_binding_capability_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34 |
| Strawberry | yes | 613 | 2 | yes | run_e2_task_risk_confirmation_adaptation_curve.py / **run_e2_task_risk_confirmation_adaptation_curve.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_task_risk_confirmation_adaptation_curve.py:40 |
| ToeSegmentation1 | yes | 40 | 2 | yes | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py / **run_e2_promoted_binding_capability_transfer.py** | condition_pair_used | evaluation/functional/run_e2_promoted_binding_capability_transfer.py:34 |
| ToeSegmentation2 | yes | 36 | 2 | yes | run_e2_source_prior_evidence_fusion.py / **run_e2_source_prior_evidence_fusion.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_source_prior_evidence_fusion.py:38 |
| TwoLeadECG | yes | 23 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_risk_action_credit_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| Wafer | yes | 1000 | 2 | yes | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py / **run_e2_curvature_corrected_action_credit.py, run_e2_task_risk_action_credit_transfer.py** | train_rows_outside_40_400, condition_pair_used | evaluation/functional/run_e2_curvature_corrected_action_credit.py:34 |
| Wine | yes | 57 | 2 | yes | run_e2_action_credit_candidate_ordering.py / **run_e2_action_credit_candidate_ordering.py** | condition_pair_used | evaluation/functional/run_e2_action_credit_candidate_ordering.py:34 |
| WormsTwoClass | yes | 181 | 2 | yes | run_e2_integrated_context_harness_evolution.py / **run_e2_integrated_context_harness_evolution.py** | condition_pair_used | evaluation/functional/run_e2_integrated_context_harness_evolution.py:38 |
| Yoga | yes | 300 | 2 | yes | run_e2_integrated_context_harness_evolution.py / **run_e2_integrated_context_harness_evolution.py** | condition_pair_used | evaluation/functional/run_e2_integrated_context_harness_evolution.py:38 |

### Eligible TRAIN rows

Empty. Every row-eligible binary loadable dataset sits in an EXECUTES actual roster.

## Part C -- stop

This book stops here. No two-arm run, no LLM, no injection of confirmation cells.

