# M-1 -- feedback-margin gating, GPMvF half protocol

protocol: `m1_margin_gate_v1`  evidence grade: **development-mechanism (pilot)**  git: `19c6b227abf1a831bdfdb0808ddb42f74224b48e`

**MARGIN_GATING_CONFIRMED**

A5-scoped supply candidate dual-gate converted 2/4; deployed held-out positive; harm 0.  Confirmation-surface margin gating stands; margin layering belongs in Gate 4.

> a conversion is experience supplying a candidate through the mechanical channel, adjudicated by Target feedback on the half confirmation surface.  It is not evidence that the agent learned to propose the family.  Half-protocol readings are margin-mechanism evidence only and must not be ranked as capability against the G3 quarter baseline.  A guided positive counts zero toward Source cross-domain authorization. Pilot; GunPointFamily same-family note.

## 1. Unique variable and implementation

- **Fixed**: W-1 dual-source hampel card, Scope, operator, Consumer, `maximum_candidates=3`, per-run LLM/fit caps, `GunPointMaleVersusFemale__impulse_v2` substrate and inject seed.
- **Unique variable**: held-in slice allocation only.
- **Implementation**: one held-in round; Support = concat(r1_support, r2_support) n=21; delayed = concat(r1_delayed, r2_delayed) n=19.  Dual gate preserved.  Eval-layer repack of `s1._build_cell` surfaces; methods / cell builder untouched.
- **Rejected composition**: ps0b stored `half_slices` (same-round support+delayed) -- that collapses the dual gate.
- **Label discipline**: half-protocol readings are margin-mechanism evidence only.  Not a capability comparison against the G3 quarter baseline.  Pilot.  GunPointFamily same-family note.  Guided positives count zero toward Source cross-domain authorization.

## 2. Arithmetic precondition (0 fit)

| surface | composition | n | identity | program | reading | 1/n | margin | meets 2x |
|---|---|---|---|---|---|---|---|---|
| quarter `r1_support` | sealed | 11 | 7/11 | 9/11 | +0.1818 | 0.0909 | — | material=True |
| quarter `r1_delayed` | sealed | 10 | 7/10 | 7/10 | +0.0000 | 0.1000 | — | material=False |
| quarter `r2_support` | sealed | 10 | 6/10 | 8/10 | +0.2000 | 0.1000 | — | material=True |
| quarter `r2_delayed` | sealed | 9 | 5/9 | 7/9 | +0.2222 | 0.1111 | — | material=True |
| **half Support** | `r1_support+r2_support` | 21 | 13/21 | 17/21 | +0.1905 | 0.0476 | **4.00×** | True |
| **half delayed** | `r1_delayed+r2_delayed` | 19 | 12/19 | 14/19 | +0.1053 | 0.0526 | **2.00×** | True |

Quarter G3 margin: **1.35×** (`reproducibility_margin_ge_2x` = false).  Role-concat half min margin: **2.00×**.  Precondition: **PASS**.

## 3. Eight-run protocol (fresh state, half, checkpoint+resume)

| run | arm | inject | Support+ | delayed | supply dual-gate | applied | held-out | worst class | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| m1_a5_1 | A5-scoped | True | True | True | True | `hampel_filter` | +0.1867 | +0.1400 | 4 | 7 |
| m1_a5_2 | A5-scoped | True | True | True | True | `hampel_filter` | +0.1867 | +0.1400 | 4 | 7 |
| m1_a5_3 | A5-scoped | False | False | True | False | `hampel_filter` | +0.1867 | +0.1400 | 4 | 6 |
| m1_a5_4 | A5-scoped | False | False | True | False | `hampel_filter` | +0.1867 | +0.1400 | 3 | 6 |
| m1_a3_1 | A3 | False | False | True | False | `hampel_filter` | +0.1867 | +0.1333 | 4 | 6 |
| m1_a3_2 | A3 | False | False | True | False | `hampel_filter` | +0.1867 | +0.1400 | 4 | 6 |
| m1_a3_3 | A3 | False | False | True | False | `hampel_filter` | +0.1867 | +0.1400 | 4 | 6 |
| m1_a3_4 | A3 | False | False | False | False | `identity` | +0.0000 | +0.0000 | 2 | 1 |

## 4. Supply-candidate six-stage funnel (A5-scoped)

| stage | A5-scoped half | G3 quarter control (not re-run) |
|---|---|---|
| card in Fast view | 4/4 | 4/4 |
| entered pool | 2/4 | 3/4 |
| Support material+ | 2/4 | 0/4 |
| delayed approved (funnel) | 4/4 | 0/4 |
| supply dual-gate converted | 2/4 | 0/4 (G3 material+ 0/4) |
| any-path family deploy | 4/4 | 1/4 (includes agent-authored a5_4) |

G3 quarter is a **mechanism contrast**, not a capability ranking.

## 5. A3 contrast (cold proposal on the readable half surface)

- A3 runs: 4; family deploys: 3; mean held-out: +0.1400; harm: 0
- A3 agent families: `hampel`
- G3 quarter A3: one cold hampel deploy (a3_2, +0.1867); this book asks whether the half surface also lets a cold proposal convert.

## 6. Cost

- LLM: 29 / 100
- Consumer fits: 45 / 100
- wall: 1068.7 s / 7200 s
- downloads: 0

## 7. Obligations

- **unique_variable_is_slice_allocation_only**: True
- **methods_contracts_runtime_operators_unmodified**: True
- **w1_g3_wiring_unchanged**: True
- **same_card_scope_operator_consumer_caps**: True
- **maximum_candidates_3**: True
- **grants_execution_false**: True
- **g3_quarter_baseline_not_rerun**: True
- **half_readings_not_capability_ranked_vs_quarter**: True
- **oracle_sealed_grader_only**: True
- **oracle_not_loaded_into_harness**: True
- **guided_positive_counts_zero_toward_source_auth**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True
- **pilot_gunpoint_family_note**: True
- **semantic_discipline**: a conversion is experience supplying a candidate through the mechanical channel, adjudicated by Target feedback on the half confirmation surface.  It is not evidence that the agent learned to propose the family.  Half-protocol readings are margin-mechanism evidence only and must not be ranked as capability against the G3 quarter baseline.  A guided positive counts zero toward Source cross-domain authorization. Pilot; GunPointFamily same-family note.

## 9. Outside the book

- A5 inject=False on m1_a5_3,m1_a5_4 with the card still in Fast view (same prepare/identity-only miss seen in PS-2 / G3).  Those runs still deployed hampel via an agent-authored program; they are not supply conversions.
- A5 funnel delayed_approved is 4/4 because _inject_funnel credits any delayed-approved hampel winner, including the agent path on inject=False runs.  The causal cell is supply_dual_gate_converted 2/4.
- A3 cold-proposed hampel and deployed it on 3/4 at the same held-out +0.1867 (a3_4 identity, LLM 2 / fit 1).  The half surface is readable for a cold proposal, not only the supply channel.  Mechanism-consistent with margin gating of the confirmation surface; not a capability ranking vs G3 A3 1/4.
- Every converting deploy on this unit printed the same held-out +0.1867 (deterministic given hampel_filter).  Delayed half margin sits exactly on the 2.00x bar.
