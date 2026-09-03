# S1a-r1 curriculum qualification + dual-layer oracle + reachability

protocol: `s1a_curriculum_oracle_audit_v1`  curriculum: **development positive-control curriculum**  evidence grade: **development**

## Isolation

本文件不得进入任何臂的 prompt/store/检索视野

Oracle files live under `artifacts/functional/e2/s1_oracle/` and are exam keys.  They must not enter any arm prompt, store, or retrieval view.

## Pool (pre-declared, not edited after scoring)

8 impulse-v2 substrates × fit_only_artifact + 1 GunPoint burst (CLS-2 `inject_burst_noise`, seed 202608254).  Consumer = ridge.  No injection-parameter scan.  No pool expansion.

## Part A -- dual-layer oracle

| unit | legal set | oracle set | identity residual | menu-best residual | upper bound | identity held-out |
|---|---|---|---|---|---|---|
| GunPointAgeSpan__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,outlier_iqr,hampel_filter,resample_uniform | hampel_filter | +0.2722 | +0.0095 | 0.8544 | 0.5823 |
| GunPoint__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,outlier_iqr,hampel_filter,resample_uniform | hampel_filter | +0.5200 | +0.1133 | 0.8533 | 0.3333 |
| ECG200__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,outlier_iqr,outlier_mad,repair_burst_segment,resample_uniform | repair_burst_segment | +0.2000 | +0.1600 | 0.8000 | 0.6000 |
| Wine__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,hampel_filter,resample_uniform | identity | +0.1481 | +0.1481 | 0.6296 | 0.4815 |
| ToeSegmentation1__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,outlier_iqr,outlier_mad,hampel_filter,repair_burst_segment,resample_uniform | repair_burst_segment | +0.1140 | +0.0833 | 0.5702 | 0.4561 |
| Lightning2__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,winsorize,repair_burst_segment,resample_uniform | repair_burst_segment | +0.2623 | +0.1639 | 0.5738 | 0.3115 |
| Herring__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,outlier_iqr,outlier_mad,hampel_filter,repair_burst_segment,resample_uniform | hampel_filter | +0.1250 | +0.0781 | 0.5781 | 0.4531 |
| Ham__impulse_v2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,hampel_filter,repair_burst_segment,resample_uniform | identity | +0.1143 | +0.1143 | 0.6381 | 0.5238 |
| GunPoint__burst_cls2 | impute_linear,impute_fft,impute_ema,period_complete,period_median_complete,impute_ssm,impute_ar,denoise_median,outlier_iqr,repair_level_shift,resample_uniform | outlier_iqr | +0.1133 | +0.1000 | 0.8533 | 0.7400 |

## Part B -- qualification

**FULL_CURRICULUM_QUALIFIED**

- positives (pool): ['GunPointAgeSpan__impulse_v2', 'GunPoint__impulse_v2', 'ECG200__impulse_v2', 'ToeSegmentation1__impulse_v2', 'Lightning2__impulse_v2', 'Herring__impulse_v2', 'GunPoint__burst_cls2']
- compatible cluster: ['GunPointAgeSpan__impulse_v2', 'GunPoint__impulse_v2', 'Herring__impulse_v2']
- program geometry: ['hampel_filter']
- clusters: [{"program": "hampel_filter", "unit_ids": ["GunPointAgeSpan__impulse_v2", "GunPoint__impulse_v2", "Herring__impulse_v2"], "pattern_intersection": {"missing_fraction": "zero", "longest_missing_run_fraction": "zero", "local_robust_z_peak": "high", "estimated_region_end_fraction": "high", "level_region_fraction": "very_low", "level_region_end_fraction": "very_low", "outlier_region_end_fraction": "very_low", "estimated_level_offset": "low", "period_reliability": "high", "period_evidence_status": "OK"}, "compatible": true}, {"program": "repair_burst_segment", "unit_ids": ["ECG200__impulse_v2", "ToeSegmentation1__impulse_v2", "Lightning2__impulse_v2"], "pattern_intersection": {"missing_fraction": "zero", "longest_missing_run_fraction": "zero", "local_robust_z_peak": "high", "estimated_region_end_fraction": "high", "level_region_fraction": "very_low", "level_region_end_fraction": "very_low", "outlier_region_end_fraction": "very_low", "level_excursion_score": "high", "estimated_level_offset": "low", "period_reliability": "high", "period_evidence_status": "OK"}, "compatible": true}, {"program": "outlier_iqr", "unit_ids": ["GunPoint__burst_cls2"], "pattern_intersection": {"missing_fraction": "zero", "longest_missing_run_fraction": "zero", "local_robust_z_peak": "high", "estimated_region_start_fraction": "low", "estimated_region_end_fraction": "high", "level_region_fraction": "very_low", "level_region_end_fraction": "very_low", "outlier_region_end_fraction": "very_low", "level_excursion_score": "high", "estimated_level_offset": "low", "period_change_score": "medium", "period_reliability": "high", "period_evidence_status": "OK"}, "compatible": false}]
- pattern intersection (no dataset name): {"missing_fraction": "zero", "longest_missing_run_fraction": "zero", "local_robust_z_peak": "high", "estimated_region_end_fraction": "high", "level_region_fraction": "very_low", "level_region_end_fraction": "very_low", "outlier_region_end_fraction": "very_low", "estimated_level_offset": "low", "period_reliability": "high", "period_evidence_status": "OK"}
- limitation: none

### Frozen course

- units: ['GunPointAgeSpan__impulse_v2', 'Wine__impulse_v2', 'GunPoint__impulse_v2', 'Ham__impulse_v2', 'Herring__impulse_v2', 'GunPoint__burst_cls2']
- forward: ['GunPointAgeSpan__impulse_v2', 'Wine__impulse_v2', 'GunPoint__impulse_v2', 'Ham__impulse_v2', 'Herring__impulse_v2', 'GunPoint__burst_cls2']
- reverse: ['GunPoint__burst_cls2', 'Herring__impulse_v2', 'Ham__impulse_v2', 'GunPoint__impulse_v2', 'Wine__impulse_v2', 'GunPointAgeSpan__impulse_v2']
- rule: positives then identities in pre-declared pool order; interleave; append burst; reverse is the exact reverse.  No outcome-driven reordering after this rule.

## Part C -- reachability

### C1 K0 inventory

- existing classification Episodes: 20
- Wine precheck is **not** an Episode.
- existing Slow card visibility: **Slow-only**

| episode_id | relation | program | compiled | three-tier |
|---|---|---|---|---|
| ProximalPhalanxOutlineCorrect/stable_task_event_target_outlier_mad_source_ProximalPhalanxOutlineCorrect_r1_p1 | NEUTRAL | outlier_mad | episode_only | Slow-only |
| MiddlePhalanxOutlineCorrect/fit_only_artifact_target_outlier_mad_source_MiddlePhalanxOutlineCorrect_r1_p1 | CONFLICT | outlier_mad | episode_only_conflict | Slow-only |
| MiddlePhalanxOutlineCorrect/stable_task_event_target_outlier_mad_source_MiddlePhalanxOutlineCorrect_r1_p1 | NEUTRAL | outlier_mad | episode_only | Slow-only |
| Lightning2/stable_task_event_target_winsorize_source_Lightning2_r1_p1 | NEUTRAL | winsorize | episode_only | Slow-only |
| PhalangesOutlinesCorrect/fit_only_artifact_target_outlier_mad_a3_PhalangesOutlinesCorrect_r1_p1 | CONFLICT | outlier_mad | episode_only_conflict | Slow-only |
| ProximalPhalanxOutlineCorrect/fit_only_artifact_target_outlier_mad_source_ProximalPhalanxOutlineCorrect_r1_p1 | CONFLICT | outlier_mad | episode_only_conflict | Slow-only |
| ProximalPhalanxOutlineCorrect/fit_only_artifact_target_repair_level_shift_source_ProximalPhalanxOutlineCorrect_r1_p2 | NEGATIVE | repair_level_shift | episode_only_negative | Slow-only |
| ProximalPhalanxOutlineCorrect/stable_task_event_target_repair_level_shift_source_ProximalPhalanxOutlineCorrect_r1_p1 | NEUTRAL | repair_level_shift | episode_only | Slow-only |
| ProximalPhalanxOutlineCorrect/stable_task_event_target_hampel_filter_source_ProximalPhalanxOutlineCorrect_r1_p2 | NEUTRAL | hampel_filter | episode_only | Slow-only |
| MiddlePhalanxOutlineCorrect/fit_only_artifact_target_outlier_iqr_source_MiddlePhalanxOutlineCorrect_r1_p2 | NEUTRAL | outlier_iqr | episode_only | Slow-only |
| MiddlePhalanxOutlineCorrect/stable_task_event_target_repair_level_shift_source_MiddlePhalanxOutlineCorrect_r1_p1 | NEUTRAL | repair_level_shift | episode_only | Slow-only |
| Lightning2/fit_only_artifact_target_winsorize_source_Lightning2_r1_p1 | NEUTRAL | winsorize | episode_only | Slow-only |
| PhalangesOutlinesCorrect/fit_only_artifact_target_repair_level_shift_a3_PhalangesOutlinesCorrect_r1_p2 | CONFLICT | repair_level_shift | episode_only_conflict | Slow-only |
| PhalangesOutlinesCorrect/fit_only_artifact_target_repair_level_shift_a3_PhalangesOutlinesCorrect_r2_p1 | NEGATIVE | repair_level_shift | episode_only_negative | Slow-only |
| PhalangesOutlinesCorrect/fit_only_artifact_target_repair_level_shift_a5_PhalangesOutlinesCorrect_r1_p1 | CONFLICT | repair_level_shift | episode_only_conflict | Slow-only |
| PhalangesOutlinesCorrect/fit_only_artifact_target_repair_level_shift_a5_PhalangesOutlinesCorrect_r2_p1 | NEGATIVE | repair_level_shift | episode_only_negative | Slow-only |
| GunPointAgeSpan/fit_only_artifact_target_hampel_filter_a3_GunPointAgeSpan_r1_p1 | POSITIVE | hampel_filter | target_local_capability | Fast-visible (not an experience card; frozen steps) |
| GunPointAgeSpan/fit_only_artifact_target_hampel_filter_a3_GunPointAgeSpan_r2_p1 | POSITIVE | hampel_filter | target_local_capability | Fast-visible (not an experience card; frozen steps) |
| GunPointAgeSpan/fit_only_artifact_target_outlier_iqr_a5_GunPointAgeSpan_r2_p1 | NEUTRAL | outlier_iqr | episode_only | Slow-only |
| ECG200/fit_only_artifact_target_outlier_mad_a3_ECG200_r2_p1 | NEGATIVE | outlier_mad | episode_only_negative | Slow-only |

### C2 compiler count semantics

distinct context_summary.task_episode_id strings (risk_skill.py:72-74, 98, 109, 177).  In the forecasting G1 path those ids are e1v2_task_NN.  Not dataset, not cell, not run, unless the writer put that string into task_episode_id.

classification Episode write path (methods/ttha/online_loop.py:180-193) never sets context_summary.task_episode_id.  _task_of therefore returns '' for every classification Episode.  Two curriculum units that both harm outlier_mad collapse to one counted Task.  That is not independent curriculum-unit counting.  Slow source census is a different counter: run_e2_t6_cls_op_shared_harness.py:896-908 sets task_episode_id = dataset/condition, so THAT audit counts cells.  source_skill.build_skill_payload (source_skill.py:472-478) still does not copy that count onto evidence_distinct_task_count, so Fast-guard never sees it.

### C3 expected Fast-view divergence

- Fast-TRY reachable in course: True
- Fast-guard reachable in course: False
- first visible difference: {"unit_index": 2, "at": "start of the next unit", "kind": "target_local_capability_carry", "detail": "A5 Fast view gains a non-experience-card capability with frozen steps; K0-fixed still has only the inert Slow card.  retrieval.py:166-168 / 274-275."}

| i | unit | positive | A5 Fast | K0-fixed Fast |
|---|---|---|---|---|
| 1 | GunPointAgeSpan__impulse_v2 | True | adds Target-local Skill after this unit | reset to K0 (inert Slow card) at unit start; may form a Target-local Skill inside the unit only |
| 2 | Wine__impulse_v2 | False | no Fast-visible knowledge write from this unit | reset to K0 (inert Slow card) at unit start; may form a Target-local Skill inside the unit only |
| 3 | GunPoint__impulse_v2 | True | may add Fast-TRY experience card after Slow integration | reset to K0 (inert Slow card) at unit start; may form a Target-local Skill inside the unit only |
| 4 | Ham__impulse_v2 | False | no Fast-visible knowledge write from this unit | reset to K0 (inert Slow card) at unit start; may form a Target-local Skill inside the unit only |
| 5 | Herring__impulse_v2 | True | may add Fast-TRY experience card after Slow integration | reset to K0 (inert Slow card) at unit start; may form a Target-local Skill inside the unit only |
| 6 | GunPoint__burst_cls2 | True | no Fast-visible knowledge write from this unit | reset to K0 (inert Slow card) at unit start; may form a Target-local Skill inside the unit only |

### C4 Slow rehearsal

code deduction confirms the existing card shape (TRY=NO_AUTHORIZED_ACTIVE_RECOMMENDATION, no evidence_distinct_task_count) and the deterministic authorization audit.  Card wording is not required for the visibility predicate.  Slow rehearsal not spent.

## Total verdict

**FULL_CURRICULUM_QUALIFIED**

## Cost

- Fast LLM: 0 / 0
- Slow rehearsal LLM: 0 / 8
- Consumer fits: 69 / 500
- wall clock: 103.32 s / 5400 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **no_fast_llm**: True
- **slow_rehearse_llm**: 0
- **slow_rehearse_cap**: 8
- **no_a3_a5_adaptation_arm**: True
- **no_injection_scan**: True
- **no_pool_expansion**: True
- **oracle_isolated**: True
- **downloads**: 0
- **ucr_conf_downloaded_not_opened**: True
- **fit_budget_held**: True
- **wall_clock_held**: True
- **full_repo_pytest_not_run**: True

## Outside the book

- classification online_loop does not write task_episode_id; risk_skill counts would collapse across curriculum units.
- source_skill.build_skill_payload does not write evidence_distinct_task_count, so Fast-guard is off for experience cards even after two harm cells.
- classification shared harness does not call run_risk_skill_lifecycle.
- C40 Target-local hampel is Fast-visible as a capability but must not be placed in K0 or K0-fixed is contaminated.
- ECG200/ToeSegmentation1/Lightning2 form a second compatible cluster on repair_burst_segment (not the frozen action family).  ECG200 hampel remains illegal under the 0.10 cohort cap; the three-substrate hampel fate table is unchanged.
- Wine hampel is legal (0.0297) but held-out class harm Δrecall_0=-0.444 excludes it from the oracle set.
- GunPoint burst stretch oracle is outlier_iqr at +0.0133 (just above the material line); repair_burst_segment is illegal or not in the oracle set on that unit.
