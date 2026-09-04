# v1.2.1-Core P0b readiness audit

**Audit: `P0B_COMPLETE`. Execution: `P0B_PASS__P1_BASELINE_SMOKE_RELEASED`. P1 release: `True`.**

All P0b safety/accounting contracts pass under v1.2.1; P1 full Core baseline smoke is next. Final outcomes remain sealed and RQ3 is not exercised.

No Natural Final outcome was opened by this runner.

## Gate ledger

| gate | status |
|---|---|
| supersession | `PASS` |
| exposure_fresh_pool | `PASS` |
| adapter | `PASS` |
| program_space | `PASS_DESCRIPTIVE_INVENTORY` |
| treatment_reachability_event | `PASS_RQ1_RQ2__RQ3_NOT_EXERCISED` |
| baseline_smoke | `PASS_MINIMAL_CONTRACT_SMOKE` |
| cost | `PASS_COST_ACCOUNTING_FREEZE` |

## Frozen Final roster

- Forecast: Traffic leftover columns 480..861; Solar-Energy all 137 series.
- Classification: Adiac and ArrowHead; TEST bytes remain unread.
- AD: Yahoo S5 sealed 41; Fresh NAB is `FINAL_POOL_UNAVAILABLE`.

## Classification TRAIN-only adapter

| dataset | TRAIN shape | classes | Support-A Macro-F1 | Support-B Macro-F1 | TEST bytes read |
|---|---:|---:|---:|---:|---|
| Adiac | 390x176 | 37 | 0.296853 | 0.362295 | False |
| ArrowHead | 36x251 | 3 | 0.885714 | 0.783333 | False |

## Program-space inventory

Coverage is descriptive; it is not a release gate.

| task | B_main | current P_effect | actual coverage |
|---|---:|---:|---:|
| forecast | 4 | 18 | 22.22% |
| classification | 4 | 19 | 21.05% |
| anomaly_detection | 4 | 11 | 36.36% |

No DSL, two-step, targeting, or AD-budget expansion is authorized.

## RQ3 claim ceiling

- forecast: `RQ3_NOT_EXERCISED` — no applied revision followed by a similar re-encounter
- classification: `RQ3_NOT_EXERCISED_METRIC_MISMATCH` — historical full chain used Accuracy, not v1.2.1 Macro-F1
- anomaly_detection: `RQ3_NOT_EXERCISED` — development events do not form one revision-to-re-encounter chain

## Minimal baseline and cost accounting

Ten baseline contracts passed on all three task fixtures; this makes no performance claim.

The Core roster has 13 methods. Planned caps: 3630 full Support logical evaluations, 2184 Query evaluations, and 3360 LLM calls. No affordability threshold is imposed.

## Release decision

P0b is complete and P1 full Core Baseline Smoke is released. Do not start P2 or open a Natural Final outcome until P1 passes.

Machine-readable detail: `artifacts/main_protocol/p0_readiness_20260830.json`.
