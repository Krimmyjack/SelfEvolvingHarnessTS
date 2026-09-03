# CLS-OP-r2 A5 mechanism replay (inertness invariant)

protocol: `t6_cls_op_r2_a5_replay_v1`  evidence grade: **MECHANISM**

## Verdict

**VISIBILITY_INVARIANT_HOLDS**

MECHANISM.  Tests the inertness invariant on the exposed C40 ledger after the Fast-visibility fix.  Held-out accuracy is recorded, not judged.  No capability claim.

- card absent every Fast view: True
- C40 level-shift-only pattern: False
- legal Support receipts: 1
- proposal families A3-same-type: False
- source card installed / Slow-visible: True / True

## Backend lock

- artifact field `obligations.backend` = live Fast Agent
- r2 git_head: cb03eb688210f521c931895454b44d30048c1928
- locked model: gpt-5.6-sol
- locked base_url: https://api.agicto.cn/v1
- probe: True

## Source card install

- skill_id: source_investigation_cls_v1
- store skill ids: ['build_contrastive_candidates', 'inspect_and_localize', 'select_or_identity_and_verify', 'source_investigation_cls_v1']
- preflight Fast view: ['build_contrastive_candidates', 'inspect_and_localize', 'select_or_identity_and_verify']
- preflight Slow view: ['build_contrastive_candidates', 'inspect_and_localize', 'select_or_identity_and_verify', 'source_investigation_cls_v1']

## Per-round comparison (replay vs C40 A5)

| dataset | round | side | retrieved contains card | non-identity pool | family | probe kind | Support | delayed |
|---|---|---|---|---|---|---|---|---|
| GunPointAgeSpan | r1 | c40_a5 | True | ['localized_level_shift_repair'] | ['level-shift'] | ['verifier_rejected'] | 0 | None |
| GunPointAgeSpan | r1 | replay_a5 | False | ['repair_local_level_shift'] | ['level-shift'] | ['verifier_rejected'] | 0 | None |
| GunPointAgeSpan | r2 | c40_a5 | True | ['repair_local_level_excursion'] | ['level-shift'] | ['verifier_rejected'] | 0 | None |
| GunPointAgeSpan | r2 | replay_a5 | False | ['intrinsic_iqr_outlier_repair', 'outlier_iqr'] | ['other', 'other'] | ['probe'] | 1 | None |

## Held-out (information only)

- accuracy: 0.5822784810126582
- vs identity: 0.0
- applied: identity
- deploy source: FROZEN_LEDGER_NO_INCUMBENT_IDENTITY

## retrieval_binding_miss mislabel

- the r2_annotate classifier would say: **retrieval_binding_miss**
- nature: MISLABEL: the card is withheld by the Fast-visibility predicate, not a retrieval-binding failure
- classifier modified: False

## Budget

- LLM: 9 of 90 (fast 9, slow 0)
- Consumer fits: 3 of 600
- wall clock: 545.9 s

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **other_line_files_untouched**: True
- **forbidden_data_untouched**: no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened; the only data root is data/ucr_task_context
- **source_card_reused_from_r2_ledger**: True
- **slow_not_re_run**: True
- **deficit_classifier_unmodified**: True
- **backend**: locked r2 Fast path gpt-5.6-sol @ https://api.agicto.cn/v1
- **full_repo_pytest_not_run**: True
- **downloads**: 0
