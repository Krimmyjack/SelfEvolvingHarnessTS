# #42g-b2 derived readouts

decision gate (not a science claim): **FEEDBACK_UNIT_REDESIGN**

Part 0 sha: `eef656b89ea285c829936c5437abc514287f6b6b`

Source: `t6_42g_b_menu_headroom.json` only. 0 LLM / 0 fit / no data-file reads. 41 sealed unread. `methods/` not edited. No fallback refit.

eval zone: **development_exposed_eval**. True held-out = remaining 41 sealed series.

Claim cap for A1: `outcome_oracle_upper_bound` is the max local menu space **if each series were correctly identified**. It does not prove a legally observable Scope at deploy time and is not Capability evidence.

## A1 outcome_oracle_upper_bound

Δ_oracle = **+0.0375** (mean of per-series best development_exposed_eval Δ among {identity, iqr, mad, hampel, winsorize}).

Positive series (best Δ > +0.005): **14 / 24**. real_21 is +0.005003 (just over the material line).

Chosen counts use menu order on exact ties (identity first when several sit at 0):

| program | n chosen | contribution sum | contribution mean |
|---|---:|---:|---:|
| identity | 10 | 0 | 0 |
| hampel_filter | 6 | +0.383 | +0.0160 |
| outlier_mad | 3 | +0.318 | +0.0132 |
| outlier_iqr | 3 | +0.069 | +0.0029 |
| winsorize | 2 | +0.131 | +0.0055 |

Largest single gifts: real_17 mad +0.229; real_28 hampel +0.200; real_10 winsorize +0.083.

## A2 direction (10 event-bearing series only)

Relation: POSITIVE Δ > +0.005; NEGATIVE Δ < −0.005; else NEUTRAL. Not raw sign.

3×3 = feedback row × development_exposed_eval column.

| program | agree | rate | notable off-diagonal |
|---|---:|---:|---|
| identity | 10/10 | 1.00 | all NEUTRAL×NEUTRAL (Δ≡0) |
| outlier_iqr | 5/10 | 0.50 | no POSITIVE feedback at all |
| outlier_mad | 7/10 | 0.70 | one POSITIVE feedback is NEUTRAL on eval |
| hampel_filter | 1/10 | **0.10** | 4 NEGATIVE feedback → POSITIVE eval; 2 POSITIVE feedback → NEGATIVE eval |
| winsorize | 4/10 | 0.40 | no POSITIVE feedback; 3 NEGATIVE → POSITIVE eval |

Four non-identity **combined agree rate = 0.425 ≤ 0.50**.

argmax = set of programs within 0.005 of that zone’s max. Intersection ⇒ compatible: **3 / 10** (real_13, real_3, real_30). The other seven disagree on who is best.

## A3 winner overlap (local-bar sets from #42g-b)

| | iqr 6 | mad 5 | hampel 7 | winsorize 4 |
|---|---|---|---|---|
| iqr | — | real_24 | 29, 30 | 10, 29 |
| mad | | — | 17, 2 | 17 |
| hampel | | | — | 17, 29 |

iqr ∩ mad ∩ hampel = **∅**. Union (incl. winsorize) = **14** series.

Local winners barely share series. That is why a single global program cannot harvest the oracle.

## Decision gate (priority: FEEDBACK_UNIT_REDESIGN first)

- combined non-identity agree **0.425 ≤ 0.50** → **FEEDBACK_UNIT_REDESIGN**
- this **outranks** SCOPE_LINE_MATERIAL even though Δ_oracle **+0.0375 ≥ +0.02**
- do not route Scope-line vs Supply-line until feedback has a transferable direction

Not entered in any science-claim table.

## Missing pre-registrations

None required by the book. Identity’s 1.0 agreement is tautological (Δ≡0) and is **not** folded into the 0.425 combined rate.

## Deliverables (not committed)

- `artifacts/functional/e2/t6_42g_b2_derived_readouts.json`
- `artifacts/functional/e2/t6_42g_b2_derived_readouts.md`
