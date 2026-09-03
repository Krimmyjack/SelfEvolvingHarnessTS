# #42h feedback unit

verdict: **FEEDBACK_UNIT_UNRESOLVED**

flags: none (no unit was selected, so `FEEDBACK_DIRECTION_WEAK` does not attach)

Part 0 sha: `80fd455d19d04508eedc465fca33d9fda2da06a0`

0 LLM / 120 AD fits (cap 150) / 0 retrain. 41 sealed unread. `methods/` not edited.

**Not a science claim. Not Capability.** Decision gate only. Transductive development diagnosis: fit on `[0,.70n)` and score inside held-in. This result **must not** open the 41 sealed series. A candidate still needs an EXPOSED-24 multi-round Harness replay (Support and delayed as separate receipts; delayed must not mint the proposal it then classifies) before it can freeze as an official unit.

eval zone: **development_exposed_eval**. True held-out = remaining 41 sealed.

## U0 reproduce-or-stop

Frozen sol anchor reproduced **exactly**:

- macro = `-0.00697816676077546`
- harmed = 2
- worst = `-0.1`
- choices = `{identity: 21, hampel_filter: 2, outlier_mad: 1}`
- picks: real_19 hampel, real_23 hampel, real_29 mad

### Pre-registration ambiguity (self-declared)

Book text said U0 = four-window **macro**. That readout **failed** the anchor (`identity 19 / winsorize 1 / hampel 2 / mad 1 / iqr 1`, macro −0.00762, worst −0.20). The numbers that reproduce the anchor are the **#42g-b pooled-union** feedback Δ on `[.30n,.70n)`. U0 is bound to that estimand.

Consequence: **U1 as written (same zone, drop window-macro) collapses onto U0.** Same choices, same C2. Declared, not hidden.

## A1 full sparsity (EXPOSED, legal)

| series | `[0,.30n)` | r1 S | r1 D | r2 S | r2 D | development_exposed_eval |
|---|---:|---:|---:|---:|---:|---:|
| real_1.csv | 0 | 0 | 0 | 0 | 0 | 2 |
| real_10.csv | 0 | 0 | 0 | 0 | 0 | 1 |
| real_11.csv | 0 | 0 | 0 | 0 | 0 | 1 |
| real_12.csv | 1 | 0 | 0 | 0 | 0 | 1 |
| real_13.csv | 1 | 0 | 0 | 0 | 1 | 1 |
| real_14.csv | 1 | 0 | 0 | 0 | 0 | 0 |
| real_15.csv | 1 | 0 | 0 | 0 | 1 | 1 |
| real_16.csv | 0 | 0 | 0 | 0 | 0 | 1 |
| real_17.csv | 0 | 0 | 0 | 1 | 1 | 2 |
| real_18.csv | 1 | 0 | 0 | 0 | 0 | 0 |
| real_19.csv | 0 | 0 | 0 | 1 | 1 | 2 |
| real_2.csv | 0 | 0 | 0 | 0 | 0 | 2 |
| real_20.csv | 0 | 1 | 0 | 0 | 0 | 2 |
| real_21.csv | 0 | 0 | 0 | 0 | 0 | 2 |
| real_22.csv | 0 | 0 | 0 | 0 | 0 | 1 |
| real_23.csv | 5 | 0 | 3 | 2 | 0 | 2 |
| real_24.csv | 1 | 0 | 0 | 0 | 0 | 2 |
| real_25.csv | 0 | 0 | 0 | 0 | 0 | 1 |
| real_26.csv | 0 | 0 | 0 | 0 | 0 | 5 |
| real_27.csv | 0 | 0 | 0 | 0 | 0 | 2 |
| real_28.csv | 1 | 0 | 0 | 0 | 1 | 2 |
| real_29.csv | 1 | 0 | 0 | 1 | 0 | 3 |
| real_3.csv | 0 | 1 | 0 | 0 | 0 | 1 |
| real_30.csv | 0 | 1 | 0 | 0 | 0 | 1 |

9 series have base-train events. 10 have any `[.30n,.70n)` event. 14 have any scorable held-in event. Zero-event series cannot authorize a positive adopt; FPR is veto/harm-guard only.

## Part B — four units (C2 is the selection criterion)

Safety bar (must all hold): policy macro Δ > +0.005, harmed ≤ 2/24, worst ≥ −0.02.

| unit | C2 macro | harmed | worst | regret vs oracle +0.0375 | C1 combined | C3 bilateral | safe |
|---|---:|---:|---:|---:|---:|---:|---|
| U0 pooled-union `[.30,.70)` | **−0.006978** | 2 | −0.100 | −0.0445 | 0.425 (10) | 10 | no |
| U1 (collapses to U0) | −0.006978 | 2 | −0.100 | −0.0445 | 0.425 | 10 | no |
| U2 scorable held-in `[19,.70n)` | **+0.005059** | 2 | **−0.071** | −0.0325 | 0.393 (14) | 14 | no (worst) |
| U3 cohort micro-pool | 0 (stays identity) | 0 | 0 | −0.0375 | 1.0 / 4 pts, descriptive | 14 | no (macro) |

U0/U1 adopt: real_19 hampel, real_23 hampel, real_29 mad.

U2 adopt: real_19 hampel, real_24 hampel, real_28 hampel, real_29 mad. Harmed = real_19, real_24. Closest unit: just clears the material macro line, then fails worst.

U3 cohort feedback Δ all NEGATIVE → stays identity. C1 4/4 agree is “all programs hurt both zones,” not a transferable positive direction. Descriptive only.

## Part C selection

No unit is safety-qualified → **FEEDBACK_UNIT_UNRESOLVED**.

No official feedback unit is frozen. #42g-c / 41 sealed stay closed. A later book may redesign units, but this book must not add a fifth candidate or change the bar.

## Deliverables (not committed)

- `artifacts/functional/e2/t6_42h_feedback_unit.json`
- `artifacts/functional/e2/t6_42h_feedback_unit.md`
- runner `--feedback-unit-v1` (dirty vs 80fd455)
