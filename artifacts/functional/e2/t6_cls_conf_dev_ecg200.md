# CLS-DEV-ECG200 -- development-grade conf lifecycle on local ECG200

protocol: `t6_cls_conf_dev_v1`  target: **ECG200**  evidence grade: **development**

## Verdict

**DEV_CHAIN_NO_POSITIVE**

DEV_CHAIN_POSITIVE iff a non-identity Target-local Skill formed and the frozen Fast-only deployment beats Static identity by at least max(0.005, 1/n) on held-out accuracy with no per-class recall falling more than 0.005.

- non-identity Target-local Skill formed: False
- A3 minus Static held-out accuracy: 0.0 (material line 0.01)
- worst per-class recall delta: 0.0 (zero class harm: True)
- deployment purity: True
- forbidden label unused: CLS_CHAIN_CONFIRMED

development only.  ECG200 was previously used by the W48 / W49 / curvature lines under the same impulse condition pair (audit: artifacts/functional/e2/t6_cls_conf_r3_selection.json). This run is therefore not an independent confirmation. Every judgement stays at evidence_grade=development. The label CLS_CHAIN_CONFIRMED must not be used.  The impulse is a controlled injection; a positive here is a second development-grade Target-local Skill, not a fresh confirmation.

## Honesty constraint

ECG200 was previously used by the W48 / W49 / curvature lines under the same impulse condition pair (audit: artifacts/functional/e2/t6_cls_conf_r3_selection.json). This run is therefore not an independent confirmation. Every judgement stays at evidence_grade=development. The label CLS_CHAIN_CONFIRMED must not be used.

This artifact is **development** evidence.  It is not an independent confirmation and must not be cited as CLS_CHAIN_CONFIRMED.

## Substrate

- archive: `data/ucr_task_context/ECG200.zip` (298857 bytes)
- TRAIN rows × length: 100 × 96; classes: 2
- selection: task-book CLS-DEV-ECG200; --dataset default ECG200; archive data/ucr_task_context/ECG200.zip
- condition: `fit_only_artifact`

## Held-in trajectory

| arm | round | retrieved_skill_ids | chosen | probes | winner | Support receipts | delayed | abstain | relation(s) |
|---|---|---|---|---|---|---|---|---|---|
| A3 | r1 | build_contrastive_candidates,inspect_and_localize,select_or_identity_and_verify | identity | 1 | None | 0 | None | True | - |
| A3 | r2 | build_contrastive_candidates,inspect_and_localize,select_or_identity_and_verify | remove_broad_extreme_deviations | 2 | None | 1 | None | True | NEGATIVE |

## Two-arm readouts

| arm | Skill formed | first-Skill LLM | first-Skill executions | held-in delayed | held-out acc | vs identity | recall by class | recall delta | worst class recall d | Support/delayed agree:disagree | deploy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | False | - | - | n/a | 0.6000 | +0.0000 | {'0': 0.6388888888888888, '1': 0.578125} | {'0': 0.0, '1': 0.0} | +0.0000 | 0:0 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY |
| STATIC | False | - | - | n/a | 0.6000 | +0.0000 | {'0': 0.6388888888888888, '1': 0.578125} | {'0': 0.0, '1': 0.0} | +0.0000 | 0:0 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY |

## Lifecycle shape vs GunPointAgeSpan A3 positive

GunPointAgeSpan A3 (development positive, `artifacts/functional/e2/t6_cls_op_r2_three_arms.json`): r1 retrieved bootstrap-only `build_contrastive_candidates,inspect_and_localize,select_or_identity_and_verify`; proposal `hampel_filter`; Support +0.50 → delayed +0.40; Skill in r1; held-out 0.8513 vs identity 0.5823 (+0.2690); per-class recall delta {'0': 0.28125, '1': 0.2564102564102564}; Support-delayed 2:0; first-Skill LLM 6 / executions 1.

This run A3: retrieved bootstrap-only on both rounds (same three h0 cards; no Source card); r1 proposed `repair_level_excursion` (verifier rejected, 0 Support); r2 proposed `outlier_mad` / `remove_broad_extreme_deviations` (Support −0.1429, NEGATIVE, delayed not opened) plus `repair_early_level_shift` (verifier rejected). No Skill formed. Held-out 0.6000 = Static identity 0.6000; per-class recall delta {0: 0.0, 1: 0.0}; Support-delayed 0:0.

Same-shape: A3-only, bootstrap-three retrieval, cohort verifier, `maximum_candidates=3`, held-in r1/r2 → freeze → Fast-only held-out, deploy purity holds.

Different-shape: GunPoint formed a hampel Target-local Skill in r1 with Support/delayed both positive; ECG200 never proposed hampel, and the post-freeze menu diagnostic shows `hampel_filter` rejected here by `COHORT_MODIFICATION_FRACTION_EXCEEDED` (cohort fraction 0.128 > 0.10, 47/70 windows over the per-window cap). The one legal receipt (`outlier_mad`) was harmful on Support. Scope-compiler development should treat this as "same impulse family, different Program geometry / verifier fate", not as a second hampel positive.

Same-shape / different-shape notes are for Scope-compiler development only.  Shared-capability induction is not authorized from this development pair.

## Cost

- LLM: 10 / 40
- Consumer fits: 12 / 200
- wall clock: 229.2 s (cap 5400 s)
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **new_gates_opened**: none; cohort scope and maximum_candidates=3 are exactly r2's
- **target_never_used_before**: FALSE.  ECG200 is a local already-used substrate (W48/W49/curvature under the same impulse pair); this is development reuse, not a virgin Target
- **downloads**: 0
- **forbidden_data_untouched**: no Yahoo, NOAA, NAB or SMD path is opened; the only data root opened for values is data/ucr_task_context
- **artifact_not_committed**: True
- **difference_read_ran_after_the_freeze**: True
- **sealed_d2_d3_untouched**: True
- **ucr_conf_downloaded_not_opened**: True
- **not_an_independent_confirmation**: True
- **cls_chain_confirmed_label_not_used**: True
