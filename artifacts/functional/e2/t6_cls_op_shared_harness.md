# CLS-OP -- second Task on the shared Harness

protocol: `t6_cls_op_shared_harness_v1`  evidence grade: **DEVELOPMENT**

## Verdict

**SECOND_TASK_LIFECYCLE_CLOSED**  --  qualifier **SUPPLY_STARVED_BY_WINDOW_VERIFIER**

9 of 14 held-in rounds bought zero legal Support receipts, so every arm froze on identity and the three-arm table carries no signal.  The lifecycle closed; the contest did not happen.  Re-running any arm at a looser maximum_modified_fraction to harvest a better number would be tuning the protocol for a result, so it is left as the mainline's call.

- real Episodes formed: 4 ({'NEUTRAL': 3, 'CONFLICT': 1})
- three-arm cells: ['GunPointAgeSpan/A3', 'GunPointAgeSpan/A4', 'GunPointAgeSpan/A5', 'PhalangesOutlinesCorrect/A3', 'PhalangesOutlinesCorrect/A4', 'PhalangesOutlinesCorrect/A5']
- deployment purity: True

DEVELOPMENT.  Every UCR split here was already opened by W48/W55/W56 and the local event is a controlled injection, so this is a lifecycle-closure reading, not a fresh classification Capability claim.  Whether A5 beats A3 or A4 is reported as measured and is not part of the pass condition.

## Budget

- LLM: 70 of 80 (fast 69, slow 1)
- Consumer fits: 16 of 600
- wall clock: 2425.9 s

## Source Experience

Episodes: 4  by relation: {'NEUTRAL': 3, 'CONFLICT': 1}

Source-derived Skill written: True

- **WHEN**: Apply this guidance when the current Workspace has task_kind == classification.
- **OBSERVE**: Inspect task_kind and whether outlier_mad or winsorize is being considered in the current Workspace, together with the relevant Context condition. The available evidence is immaterial-only, so do not infer a preferred operator from it.
- **TRY**: NO_AUTHORIZED_ACTIVE_RECOMMENDATION
- **RISK**: The census warns against treating immaterial outcomes as evidence that outlier_mad or winsorize should be preferred, avoided, or promoted. Do not turn the absence of an authorized recommendation into a new operator rule.
- **VERIFY**: Before believing any future result, its own Target Support must show a material, task-specific outcome for the observed classification Context and must identify whether the result supports, refutes, or leaves the hypothesis immaterial.
- **FALLBACK**: When OBSERVE does not support a guarded hypothesis, make no operator recommendation and retain the existing behavior until task-specific Target Support provides evidence.

## Three-arm table

| dataset | arm | deploy source | applied | held-out acc | gain vs identity | A5-A3 | A5-A4 | harmed classes |
|---|---|---|---|---|---|---|---|---|
| GunPointAgeSpan | A3 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.5823 | +0.0000 |  |  | [] |
| GunPointAgeSpan | A4 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.5823 | +0.0000 |  |  | [] |
| GunPointAgeSpan | A5 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.5823 | +0.0000 | 0.0 | 0.0 | [] |
| PhalangesOutlinesCorrect | A3 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.4697 | +0.0000 |  |  | [] |
| PhalangesOutlinesCorrect | A4 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.4697 | +0.0000 |  |  | [] |
| PhalangesOutlinesCorrect | A5 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.4697 | +0.0000 | 0.0 | 0.0 | [] |

## First fault

**candidate exists but the window verifier rejects it**

ScopeExecutor.verify is cohort-all-or-nothing: it runs verify_candidate on every training window and one rejection rejects the whole candidate.  A forecasting or AD cohort has a dozen windows; a classification cohort has one window per fit row -- 42 to 1260 here -- so the chance that at least one row exceeds maximum_modified_fraction approaches one and the round starves before it can buy a Support receipt.

- starved rounds (zero legal Support receipts): 9 of 14
- deployment constraint in force: maximum_modified_fraction = 0.1

| dataset | condition | fit rows | program | mean frac | max frac | rows over 0.10 | passes 0.10 | passes 0.20 | passes 0.35 |
|---|---|---|---|---|---|---|---|---|---|
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | winsorize | 0.0875 | 0.1000 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | outlier_iqr | 0.0500 | 0.0500 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | outlier_mad | 0.0500 | 0.0500 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | hampel_filter | 0.0701 | 0.1625 | 59 | False | True | True |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | repair_level_shift | 0.0014 | 0.5750 | 1 | False | False | False |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | repair_burst_segment | 0.0000 | 0.0000 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | denoise_savgol | 0.9749 | 1.0000 | 420 | False | False | False |
| ProximalPhalanxOutlineCorrect | fit_only_artifact | 420 | smooth_ma | 0.8875 | 1.0000 | 420 | False | False | False |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | winsorize | 0.0875 | 0.1000 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | outlier_iqr | 0.0500 | 0.0500 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | outlier_mad | 0.0500 | 0.0500 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | hampel_filter | 0.0701 | 0.1625 | 59 | False | True | True |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | repair_level_shift | 0.0014 | 0.5750 | 1 | False | False | False |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | repair_burst_segment | 0.0000 | 0.0000 | 0 | True | True | True |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | denoise_savgol | 0.9749 | 1.0000 | 420 | False | False | False |
| ProximalPhalanxOutlineCorrect | stable_task_event | 420 | smooth_ma | 0.8875 | 1.0000 | 420 | False | False | False |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | winsorize | 0.0874 | 0.1000 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | outlier_iqr | 0.0500 | 0.0500 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | outlier_mad | 0.0500 | 0.0500 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | hampel_filter | 0.0692 | 0.1875 | 61 | False | True | True |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | repair_level_shift | 0.0027 | 0.5750 | 2 | False | False | False |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | repair_burst_segment | 0.0000 | 0.0000 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | denoise_savgol | 0.9749 | 1.0000 | 420 | False | False | False |
| MiddlePhalanxOutlineCorrect | fit_only_artifact | 420 | smooth_ma | 0.8874 | 1.0000 | 420 | False | False | False |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | winsorize | 0.0874 | 0.1000 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | outlier_iqr | 0.0500 | 0.0500 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | outlier_mad | 0.0500 | 0.0500 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | hampel_filter | 0.0692 | 0.1875 | 61 | False | True | True |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | repair_level_shift | 0.0027 | 0.5750 | 2 | False | False | False |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | repair_burst_segment | 0.0000 | 0.0000 | 0 | True | True | True |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | denoise_savgol | 0.9749 | 1.0000 | 420 | False | False | False |
| MiddlePhalanxOutlineCorrect | stable_task_event | 420 | smooth_ma | 0.8874 | 1.0000 | 420 | False | False | False |
| Lightning2 | fit_only_artifact | 42 | winsorize | 0.0534 | 0.0534 | 0 | True | True | True |
| Lightning2 | fit_only_artifact | 42 | outlier_iqr | 0.1297 | 0.3140 | 26 | False | False | True |
| Lightning2 | fit_only_artifact | 42 | outlier_mad | 0.1275 | 0.2826 | 26 | False | False | True |
| Lightning2 | fit_only_artifact | 42 | hampel_filter | 0.1116 | 0.1601 | 30 | False | True | True |
| Lightning2 | fit_only_artifact | 42 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| Lightning2 | fit_only_artifact | 42 | repair_level_shift | 0.2681 | 0.8367 | 28 | False | False | False |
| Lightning2 | fit_only_artifact | 42 | repair_burst_segment | 0.1009 | 0.2653 | 21 | False | False | True |
| Lightning2 | fit_only_artifact | 42 | denoise_savgol | 0.8642 | 0.9074 | 42 | False | False | False |
| Lightning2 | fit_only_artifact | 42 | smooth_ma | 0.8445 | 0.8948 | 42 | False | False | False |
| Lightning2 | stable_task_event | 42 | winsorize | 0.0534 | 0.0534 | 0 | True | True | True |
| Lightning2 | stable_task_event | 42 | outlier_iqr | 0.1297 | 0.3140 | 26 | False | False | True |
| Lightning2 | stable_task_event | 42 | outlier_mad | 0.1275 | 0.2826 | 26 | False | False | True |
| Lightning2 | stable_task_event | 42 | hampel_filter | 0.1116 | 0.1601 | 30 | False | True | True |
| Lightning2 | stable_task_event | 42 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| Lightning2 | stable_task_event | 42 | repair_level_shift | 0.2681 | 0.8367 | 28 | False | False | False |
| Lightning2 | stable_task_event | 42 | repair_burst_segment | 0.1009 | 0.2653 | 21 | False | False | True |
| Lightning2 | stable_task_event | 42 | denoise_savgol | 0.8642 | 0.9074 | 42 | False | False | False |
| Lightning2 | stable_task_event | 42 | smooth_ma | 0.8445 | 0.8948 | 42 | False | False | False |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | winsorize | 0.0875 | 0.1000 | 0 | True | True | True |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | outlier_iqr | 0.0501 | 0.1500 | 1 | False | True | True |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | outlier_mad | 0.0500 | 0.0750 | 0 | True | True | True |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | hampel_filter | 0.0655 | 0.1875 | 157 | False | True | True |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | repair_level_shift | 0.0068 | 0.5750 | 15 | False | False | False |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | repair_burst_segment | 0.0000 | 0.0000 | 0 | True | True | True |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | denoise_savgol | 0.9749 | 1.0000 | 1260 | False | False | False |
| PhalangesOutlinesCorrect | fit_only_artifact | 1260 | smooth_ma | 0.8872 | 1.0000 | 1260 | False | False | False |
| GunPointAgeSpan | fit_only_artifact | 95 | winsorize | 0.1062 | 0.1067 | 89 | False | True | True |
| GunPointAgeSpan | fit_only_artifact | 95 | outlier_iqr | 0.0469 | 0.2267 | 12 | False | False | True |
| GunPointAgeSpan | fit_only_artifact | 95 | outlier_mad | 0.1951 | 0.4333 | 54 | False | False | False |
| GunPointAgeSpan | fit_only_artifact | 95 | hampel_filter | 0.0678 | 0.1933 | 18 | False | True | True |
| GunPointAgeSpan | fit_only_artifact | 95 | denoise_median | 0.0000 | 0.0000 | 0 | True | True | True |
| GunPointAgeSpan | fit_only_artifact | 95 | repair_level_shift | 0.6927 | 0.8333 | 85 | False | False | False |
| GunPointAgeSpan | fit_only_artifact | 95 | repair_burst_segment | 0.1710 | 0.4333 | 54 | False | False | False |
| GunPointAgeSpan | fit_only_artifact | 95 | denoise_savgol | 0.9961 | 1.0000 | 95 | False | False | False |
| GunPointAgeSpan | fit_only_artifact | 95 | smooth_ma | 0.9952 | 1.0000 | 95 | False | False | False |

## Obligations

- **methods_package_unmodified**: True
- **new_files**: ['evaluation/functional/consumers/cls_scope_adapter.py (the one permitted thin evaluate_fn adapter)', 'evaluation/functional/run_e2_t6_cls_op_shared_harness.py (the runner; a book that writes an artifact needs one, and the adapter deliberately holds no protocol -- reported for ruling)']
- **forbidden_data_untouched**: no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened by this runner; the only data root is data/ucr_task_context
- **legacy_capability_card_not_injected**: the W56 promoted Capability card was read for archaeology only and was never written into experience_memory or a snapshot
- **fast_never_saw_raw_source_episodes**: Source Episodes stayed in their own per-cell Method instances; the Target arms construct with empty Memory and receive Source evidence only as the audited Skill on the snapshot
- **deviations**: ["The family's fit/support split is reused byte-for-byte, but the legacy support pool is quartered into per-round Support and delayed surfaces, because the shared lifecycle needs a delayed surface the family never produced.", 'The deployment-visible observation is a fixed ~3200-point window of the fit cohort rather than the whole block; the executor still acts on every fit row.  Without it, one actionability probe cost 215 s.', 'Slow Path is off inside the held-in rounds; the only Slow call is the Source consolidation that authors the six-section card.', "The per-view axis is the class axis, so CONFLICT means 'accuracy rose while a class recall fell'.  This is stricter than the family's accuracy-only gate and is the main reason POSITIVE is rare here."]
- **backend**: live Fast Agent

## Execution incident (session accounting)

The environment spawned the --run command twice (05:14 and 05:33 UTC, overlapping 05:33-05:47).  This artifact is the second run plus the 0-LLM first-fault annotation.  The duplicate reached the same all-identity table and the same held-out numbers (llm 71, fits 18).  True session API spend: 71 + 70 + 4 = 145 against the book cap of 80; each process stayed within its own <=80 ledger, and the overrun comes only from the double spawn.  No purity assertion tripped in either run.
