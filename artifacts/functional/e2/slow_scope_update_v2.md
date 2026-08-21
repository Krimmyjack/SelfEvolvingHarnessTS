# Slow Scope/Risk self-update v2 -- the EDIT_SURFACE_DEFECT repair

**Overall: `SLOW_CLOSES_SCOPE_GAP_BY_VETO`** -- every moved pooled cell vetoed to identity: 99999904140 no longer crosses -0.005 anywhere and per_channel did not move; the forgone +0.029688 aggregate per arm is booked in the detail

#18 showed the Scope/Risk edit surface itself was defective: the guard evaluation context saw only the aggregate (projection), the `scope_risk_guards` key had no tracked reader (placebo), and the offered surfaces were never the minimal one-entry edit.  This slice sutures the surface (O1 passthrough, registry entry, compiler gate, pre-registered empty list), proves the migrated state reproduces the recorded episodes digit-for-digit before any LLM call, and retries the Slow draw on the repaired surface. The replay is development-level and claims nothing about held-out performance.

## The suture (tracked files, both sha256 sides)

| file | role | before | after |
| --- | --- | --- | --- |
| `evaluation/functional/run_e2_warm_vs_cold_recipe_search.py` | O1: the instrument interface stops projecting the measured per-series vector | `bab5feb8e6ad...` | `ffd532752100...` |
| `methods/ttha/harness/h0/verification.json` | pre-registers scope_risk_guards as [] in the h0 authoring | `5b3dc62e72d2...` | `0dfa07bdd4a9...` |
| `methods/ttha/harness/harness_surfaces.json` | registers verification.rules.scope_risk_guards (one PATCH surface, pointer /scope_risk_guards) | `e3289f776f91...` | `e222daff8902...` |
| `methods/ttha/harness/h0/snapshot.lock.json` | regenerated snapshot lock (the suture moves the content and the compiler identity the lock pins) | `1e54a67ea021...` | `bbddc14b14cb...` |
| `methods/ttha/harness/compiler.py` | the adoption-gate evaluator is promoted into tracked machinery; newly registered in v2 | `c04a08fe0705...` | `fd5794518547...` |

## Migration (0 LLM)

| store | surface | key diff | receipt |
| --- | --- | --- | --- |
| `a5_pooled` | `verification.rules` | +['scope_risk_guards'] / -[] / ~[] | `4a8f5e0b20bb...` |
| `a3_pooled` | `verification.rules` | +['scope_risk_guards'] / -[] / ~[] | `7be2bd5a390b...` |
| `a5_per_channel` | `verification.rules` | +['scope_risk_guards'] / -[] / ~[] | `986987b0c693...` |
| `a3_per_channel` | `verification.rules` | +['scope_risk_guards'] / -[] / ~[] | `dcc2561246b9...` |

## Non-regression gate (0 LLM, before any Slow call)

State: post-migration: every active snapshot carries the empty 'scope_risk_guards' list; no guard is enforced in this replay.

| cell | mode | reproduces digit-for-digit | retrains |
| --- | --- | --- | ---: |
| `a5_pooled` | DIRECT_RECALL | True | 18 |
| `a3_pooled` | FULL_PRICE_SEARCH | True | 75 |
| `a5_per_channel` | DIRECT_RECALL | True | 9 |
| `a3_per_channel` | DIRECT_RECALL | True | 9 |

## B1 -- where the fold puts the fault (re-run, deterministic)

| episode | with the per-series risk reading | aggregate only |
| --- | --- | --- |
| `a5_pooled` | `RISK_GAP` at OUTCOME_RISK | `NO_ACTIONABLE_FAULT` |
| `a3_pooled` | `SELECTION_MISS` at CANDIDATE_SELECTION | `SELECTION_MISS` |

Primary cell `a5_pooled`, cause `RISK_GAP`.

## B2 -- the Slow retry on the sutured surface

### Round KIMI (configured `gpt-5.6-luna`)

| sample | outcome | reason | served by | llm calls |
| ---: | --- | --- | --- | ---: |
| 1 | `NO_PROPOSAL` | insufficient_public_evidence | ['gpt-5.6-luna'] | 1 |
| 2 | `NO_PROPOSAL` | insufficient_public_evidence | ['gpt-5.6-luna'] | 1 |

### Round OPUS (configured `claude-opus-5`)

| sample | outcome | reason | served by | llm calls |
| ---: | --- | --- | --- | ---: |
| 1 | `PROPOSAL` | -- | ['claude-opus-5'] | 1 |

- Surface: `verification.rules.scope_risk_guards`.
- Guard: `per_series_harm_line_veto` -- min_per_series_gain `lt` -0.005000 -> VETO_AND_FALL_BACK on the delayed window, applies to every_adoption.
- Slow's rationale: Public confirmation records show the adopted pooled plan (outlier_mad) cleared the aggregate delayed bar at +0.029688 while one evaluation series (99999904140) lost 0.125557 on the same window, far past the pre-registered harm line of 0.005. The aggregate projection hid that per-series loss. Reading the measured per-series vector and refusing any adoption whose worst evaluation series falls below the harm line closes that gap deterministically, in both DIRECT_RECALL and FULL_PRICE_SEARCH cells where the same harm was observed. Identity stays unfilterable, so abstention remains available.
- Verdict: `PATCH_APPLIED`; surfaces changed ['verification.rules.scope_risk_guards'].
- draw 3; the earlier draws abstained.  This proposal is a second draw from a stochastic process and is labelled as such wherever it is reported.
- **Backend-dependent:** the proposal came from the OPUS round after the KIMI round abstained twice: the grammar repair was necessary but not sufficient for the weaker backend -- a real robustness finding about the Harness's serving model

## B3 -- replay, before and after

The unguarded half is the gate's, paid once; only plans the guard moves cost fresh retrains.

| cell | plan before | plan after | delayed before | delayed after | 99999904140 before | 99999904140 after | +retrains |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `a5_pooled` | `outlier_mad` | `identity` | +0.029688 | +0.000000 | -0.125557 | +0.000000 | 0 |
| `a3_pooled` | `outlier_mad` | `identity` | +0.029688 | +0.000000 | -0.125557 | +0.000000 | 0 |
| `a5_per_channel` | `identity` | `identity` | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 |
| `a3_per_channel` | `identity` | `identity` | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0 |

### Per evaluation series, delayed gain

| cell | `99999903062` | `99999904140` | `99999923908` | `99999963862` |
| --- | ---: | ---: | ---: | ---: |
| `a5_pooled` before | +0.071979 | -0.125557 | +0.157494 | +0.014835 |
| `a5_pooled` after | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `a3_pooled` before | +0.071979 | -0.125557 | +0.157494 | +0.014835 |
| `a3_pooled` after | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `a5_per_channel` before | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `a5_per_channel` after | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `a3_per_channel` before | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `a3_per_channel` after | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## The watched series

- Hurt before in: ['a3_pooled', 'a5_pooled'].
- Still hurt after in: none.
- Pooled cells that moved: ['a3_pooled', 'a5_pooled'].
- per_channel cells that moved: none.
- New harmed series anywhere: none.
- Forgone aggregate booked (veto to identity): {'a3_pooled': '+0.029688 -> +0.000000', 'a5_pooled': '+0.029688 -> +0.000000'}.
- **Backend-dependent:** the proposal came from the OPUS round after the KIMI round abstained twice: the grammar repair was necessary but not sufficient for the weaker backend -- a real robustness finding about the Harness's serving model

## Backend identity of every Slow draw

#18 is backfilled at config level (the response-level identity was not recorded then); #19 records returned_models per draw. The two precisions are never merged into one count.

| slice | draw | outcome | reason | identity | precision |
| --- | --- | --- | --- | --- | --- |
| #18 | live attempt 1 | `NO_PROPOSAL` | lost: the #18 runner did not record it | `gpt-5.6-luna` @ agicto | config-level backfill |
| #18 | live run 2 attempt 1 | `NO_PROPOSAL` | no_authorized_minimal_edit | `gpt-5.6-luna` @ agicto | config-level backfill |
| #18 | live run 2 attempt 2 | `NO_PROPOSAL` | insufficient_public_evidence | `gpt-5.6-luna` @ agicto | config-level backfill |
| #19 | KIMI sample 1 | `NO_PROPOSAL` | insufficient_public_evidence | ['gpt-5.6-luna'] | response-level |
| #19 | KIMI sample 2 | `NO_PROPOSAL` | insufficient_public_evidence | ['gpt-5.6-luna'] | response-level |
| #19 | OPUS sample 1 | `PROPOSAL` | -- | ['claude-opus-5'] | response-level |

## Cost and integrity

- LLM calls: 3 / 10.
- Consumer retrains: 111 / 200 (the gate pays the unguarded half once; B3 pays only the guard-moved delta: 0).
- Experience rows written (provenance `slow_scope_update_v2`): 0.
- Frozen surface: 33 files, drift [].
- Registry arithmetic: v1 registered 32 unique files (the task book's '32+1'; one listed entry duplicates an FC entry and dedupes).  v2 re-registers 4 of them in post-fix form with both sha256 sides, adds compiler.py, and checks the other 28 for zero drift: 33 total..
- Wall seconds: 57.9.
