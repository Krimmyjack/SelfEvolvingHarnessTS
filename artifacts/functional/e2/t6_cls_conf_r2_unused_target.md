# CLS-CONF -- frozen confirmation on an unused UCR Target

protocol: `t6_cls_conf_unused_target_v2`  target: **None**  evidence grade: **DEVELOPMENT**

## Verdict

**PREDICTION_GATE_FAILED**

machine recount of eligible/selected disagrees with the pre-registered prediction; stopped before any LLM; the exclusion rule was not relaxed or tightened

- non-identity Target-local Skill formed: None
- A3 minus Static held-out accuracy: None (material line None)
- worst per-class recall delta: None (zero class harm: None)
- deployment purity: None



## Part A -- how the Target was chosen

r2 unused = never used under the impulse defect-repair condition pair: if any claiming runner file contains fit_only_artifact or stable_task_event, the dataset is out (over-exclude). Keep the binary ones whose official TRAIN row count is in [40, 400]; take the lexicographically first. Name-appearance is no longer an exclusion.  Fixed before the rule was applied and independent of any outcome.

- pool: 40 zips; eligible: []; selected: **None** (no eligible dataset)

| dataset | usage hits | claiming runner(s) | train rows | classes | excluded because |
|---|---|---|---|---|---|
| BeetleFly | 17 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_shared_capability_s0_census.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 20 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| BirdChicken | 19 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 20 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| Coffee | 23 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py | 28 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| Computers | 6 | run_e2_integrated_context_harness_evolution.py | 250 | 2 | claiming_runner_used_impulse_condition_pair |
| DistalPhalanxOutlineCorrect | 4 | run_e2_action_credit_candidate_ordering.py | 600 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| DodgerLoopWeekend | 1 | (none) | None | None | not_loadable_as_binary_ucr |
| Earthquakes | 4 | run_e2_action_credit_candidate_ordering.py | 322 | 2 | claiming_runner_used_impulse_condition_pair |
| ECG200 | 119 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls1_qualification_gate.py, run_e2_t6_cls1_r2_qualification_gate.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py, run_e2_temporary_excursion_skill_headroom.py | 100 | 2 | claiming_runner_used_impulse_condition_pair |
| ECGFiveDays | 10 | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 23 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| FordA | 28 | run_e2_curvature_corrected_action_credit.py, run_e2_shared_capability_s0_census.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py | 3601 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| FordB | 14 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 3636 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| FreezerRegularTrain | 6 | run_e2_source_prior_evidence_fusion.py | 150 | 2 | claiming_runner_used_impulse_condition_pair |
| FreezerSmallTrain | 4 | run_e2_task_risk_confirmation_adaptation_curve.py | 28 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| GunPoint | 144 | run_e2_curvature_corrected_action_credit.py, run_e2_pattern_mass_multiplicity_headroom.py, run_e2_t6_cls1_qualification_gate.py, run_e2_t6_cls1_r2_qualification_gate.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_t6_cls3_paired_consumer_gate.py, run_e2_t6_cls4_burst_repair_gate.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_context_label_evidence_witness.py, run_e2_task_risk_action_credit_transfer.py, run_e2_temporary_excursion_skill_headroom.py | 50 | 2 | claiming_runner_used_impulse_condition_pair |
| GunPointAgeSpan | 155 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 135 | 2 | claiming_runner_used_impulse_condition_pair |
| GunPointMaleVersusFemale | 6 | run_e2_source_prior_evidence_fusion.py | 135 | 2 | claiming_runner_used_impulse_condition_pair |
| GunPointOldVersusYoung | 6 | run_e2_source_prior_evidence_fusion.py | 136 | 2 | claiming_runner_used_impulse_condition_pair |
| Ham | 379 | run_e2_source_outlier_local_behavior_audit.py, run_e2_t6_cls2_value_corruption_gate.py, run_e2_task_conditioned_impulse_repair_control.py, run_e2_task_risk_confirmation_adaptation_curve.py | 109 | 2 | claiming_runner_used_impulse_condition_pair |
| HandOutlines | 5 | run_e2_integrated_context_harness_evolution.py | 1000 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| Herring | 4 | run_e2_task_risk_confirmation_adaptation_curve.py | 64 | 2 | claiming_runner_used_impulse_condition_pair |
| HouseTwenty | 18 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 40 | 2 | claiming_runner_used_impulse_condition_pair |
| KeplerLightCurves | 1 | (none) | None | None | not_loadable_as_binary_ucr |
| Lightning2 | 87 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 60 | 2 | claiming_runner_used_impulse_condition_pair |
| MiddlePhalanxOutlineCorrect | 102 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 600 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| MoteStrain | 11 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 20 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| PhalangesOutlinesCorrect | 164 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 1800 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| PowerCons | 5 | run_e2_integrated_context_harness_evolution.py | 180 | 2 | claiming_runner_used_impulse_condition_pair |
| ProximalPhalanxOutlineCorrect | 103 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 600 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| SemgHandGenderCh2 | 5 | run_e2_integrated_context_harness_evolution.py | 300 | 2 | claiming_runner_used_impulse_condition_pair |
| ShapeletSim | 4 | run_e2_action_credit_candidate_ordering.py | 20 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| SonyAIBORobotSurface1 | 11 | run_e2_program_binding_harness_update.py, run_e2_t6_cls_op_shared_harness.py | 20 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| SonyAIBORobotSurface2 | 20 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 27 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| Strawberry | 4 | run_e2_task_risk_confirmation_adaptation_curve.py | 613 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| ToeSegmentation1 | 24 | run_e2_promoted_binding_capability_transfer.py, run_e2_t6_cls_op_shared_harness.py | 40 | 2 | claiming_runner_used_impulse_condition_pair |
| ToeSegmentation2 | 6 | run_e2_source_prior_evidence_fusion.py | 36 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| TwoLeadECG | 16 | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 23 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| Wafer | 10 | run_e2_curvature_corrected_action_credit.py, run_e2_t6_cls_op_shared_harness.py, run_e2_task_risk_action_credit_transfer.py | 1000 | 2 | train_rows_outside_40_400, claiming_runner_used_impulse_condition_pair |
| Wine | 4 | run_e2_action_credit_candidate_ordering.py | 57 | 2 | claiming_runner_used_impulse_condition_pair |
| WormsTwoClass | 5 | run_e2_integrated_context_harness_evolution.py | 181 | 2 | claiming_runner_used_impulse_condition_pair |
| Yoga | 5 | run_e2_integrated_context_harness_evolution.py | 300 | 2 | claiming_runner_used_impulse_condition_pair |

### Prediction gate

- predicted eligible: ['Computers', 'FreezerRegularTrain', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'PowerCons', 'SemgHandGenderCh2', 'WormsTwoClass', 'Yoga']
- actual eligible: []
- predicted selected: **Computers**
- actual selected: **None**
- passed: **False**

### Impulse-condition token hits on claiming runners

- **BeetleFly**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **BeetleFly**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **BeetleFly**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **BirdChicken**: `run_e2_promoted_binding_capability_transfer.py` tokens=fit_only_artifact,stable_task_event
- **BirdChicken**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **Coffee**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **Coffee**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **Coffee**: `run_e2_task_context_label_evidence_witness.py` tokens=fit_only_artifact,stable_task_event
- **Coffee**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **Computers**: `run_e2_integrated_context_harness_evolution.py` tokens=fit_only_artifact,stable_task_event
- **DistalPhalanxOutlineCorrect**: `run_e2_action_credit_candidate_ordering.py` tokens=fit_only_artifact,stable_task_event
- **Earthquakes**: `run_e2_action_credit_candidate_ordering.py` tokens=fit_only_artifact,stable_task_event
- **ECG200**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **ECG200**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **ECG200**: `run_e2_task_context_label_evidence_witness.py` tokens=fit_only_artifact,stable_task_event
- **ECG200**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **ECGFiveDays**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **ECGFiveDays**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **ECGFiveDays**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **FordA**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **FordA**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **FordA**: `run_e2_task_context_label_evidence_witness.py` tokens=fit_only_artifact,stable_task_event
- **FordA**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **FordB**: `run_e2_program_binding_harness_update.py` tokens=fit_only_artifact,stable_task_event
- **FordB**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **FreezerRegularTrain**: `run_e2_source_prior_evidence_fusion.py` tokens=stable_task_event
- **FreezerSmallTrain**: `run_e2_task_risk_confirmation_adaptation_curve.py` tokens=fit_only_artifact,stable_task_event
- **GunPoint**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **GunPoint**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **GunPoint**: `run_e2_task_context_label_evidence_witness.py` tokens=fit_only_artifact,stable_task_event
- **GunPoint**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **GunPointAgeSpan**: `run_e2_promoted_binding_capability_transfer.py` tokens=fit_only_artifact,stable_task_event
- **GunPointAgeSpan**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **GunPointMaleVersusFemale**: `run_e2_source_prior_evidence_fusion.py` tokens=stable_task_event
- **GunPointOldVersusYoung**: `run_e2_source_prior_evidence_fusion.py` tokens=stable_task_event
- **Ham**: `run_e2_task_risk_confirmation_adaptation_curve.py` tokens=fit_only_artifact,stable_task_event
- **HandOutlines**: `run_e2_integrated_context_harness_evolution.py` tokens=fit_only_artifact,stable_task_event
- **Herring**: `run_e2_task_risk_confirmation_adaptation_curve.py` tokens=fit_only_artifact,stable_task_event
- **HouseTwenty**: `run_e2_promoted_binding_capability_transfer.py` tokens=fit_only_artifact,stable_task_event
- **HouseTwenty**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **Lightning2**: `run_e2_program_binding_harness_update.py` tokens=fit_only_artifact,stable_task_event
- **Lightning2**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **MiddlePhalanxOutlineCorrect**: `run_e2_program_binding_harness_update.py` tokens=fit_only_artifact,stable_task_event
- **MiddlePhalanxOutlineCorrect**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **MoteStrain**: `run_e2_program_binding_harness_update.py` tokens=fit_only_artifact,stable_task_event
- **MoteStrain**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **PhalangesOutlinesCorrect**: `run_e2_promoted_binding_capability_transfer.py` tokens=fit_only_artifact,stable_task_event
- **PhalangesOutlinesCorrect**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **PowerCons**: `run_e2_integrated_context_harness_evolution.py` tokens=fit_only_artifact,stable_task_event
- **ProximalPhalanxOutlineCorrect**: `run_e2_program_binding_harness_update.py` tokens=fit_only_artifact,stable_task_event
- **ProximalPhalanxOutlineCorrect**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **SemgHandGenderCh2**: `run_e2_integrated_context_harness_evolution.py` tokens=fit_only_artifact,stable_task_event
- **ShapeletSim**: `run_e2_action_credit_candidate_ordering.py` tokens=fit_only_artifact,stable_task_event
- **SonyAIBORobotSurface1**: `run_e2_program_binding_harness_update.py` tokens=fit_only_artifact,stable_task_event
- **SonyAIBORobotSurface1**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **SonyAIBORobotSurface2**: `run_e2_promoted_binding_capability_transfer.py` tokens=fit_only_artifact,stable_task_event
- **SonyAIBORobotSurface2**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **Strawberry**: `run_e2_task_risk_confirmation_adaptation_curve.py` tokens=fit_only_artifact,stable_task_event
- **ToeSegmentation1**: `run_e2_promoted_binding_capability_transfer.py` tokens=fit_only_artifact,stable_task_event
- **ToeSegmentation1**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **ToeSegmentation2**: `run_e2_source_prior_evidence_fusion.py` tokens=stable_task_event
- **TwoLeadECG**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **TwoLeadECG**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **TwoLeadECG**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **Wafer**: `run_e2_curvature_corrected_action_credit.py` tokens=fit_only_artifact
- **Wafer**: `run_e2_t6_cls_op_shared_harness.py` tokens=fit_only_artifact,stable_task_event
- **Wafer**: `run_e2_task_risk_action_credit_transfer.py` tokens=fit_only_artifact,stable_task_event
- **Wine**: `run_e2_action_credit_candidate_ordering.py` tokens=fit_only_artifact,stable_task_event
- **WormsTwoClass**: `run_e2_integrated_context_harness_evolution.py` tokens=fit_only_artifact,stable_task_event
- **Yoga**: `run_e2_integrated_context_harness_evolution.py` tokens=fit_only_artifact,stable_task_event


## Part B -- two arms

| arm | Skill formed | first-Skill LLM | first-Skill executions | held-in delayed | held-out acc | vs identity | worst class recall d | abstained rounds | Support/delayed agree:disagree |
|---|---|---|---|---|---|---|---|---|---|

## Budget

- LLM: 0 of 40
- Consumer fits: 0 of 200
- wall clock: 2.1 s

## Obligations

- **selection_rule_not_relaxed**: True
- **selection_rule_not_tightened**: True
- **llm_spent**: 0
- **downloads**: 0
- **methods_package_unmodified**: True
- **r1_artifacts_not_overwritten**: True
