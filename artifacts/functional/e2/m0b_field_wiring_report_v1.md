# M0b field wiring report v1

**Single Observation-surface change.** This round wires the four M0a-judged-informative
split-geometry fields into the public observation contract and nothing else.
The original `post_shift_support_sufficient` is untouched (kept for comparison,
same formula, same constants, same union reading). No historical
`feature_context_sha` is migrated, no old replay is rewritten, no new hash
system is introduced; the sha changes only for new runs, as designed.
`schema_version` strings are unchanged. KDD W3 (T211-T230), NOAA,
`g3_final_query_outcome` and all other sealed data were not read.
`m0a_*` artifacts and `run_e2_m0a_mask_geometry_census.py` were read as
reference only, never modified.

## 1. The four wired fields

Added to the mapping in `runtime/public_features.py` and to
`OBSERVABLE_FEATURES` in `contracts/observables.py`, with definitions equal to
the M0a census:

| field | type | definition (identical to M0a) |
| --- | --- | --- |
| `level_region_fraction` | number | `mean(level_mask)` |
| `level_region_end_fraction` | number | `(last True index + 1) / n` of `level_mask`, `0.0` if empty |
| `outlier_region_end_fraction` | number | same read on the `_expand()`-ed outlier region already used by the union (`runtime/public_features.py:279`), not a re-invented expansion |
| `level_only_post_shift_support_sufficient` | boolean | the frozen pss formula (`max(0, (1-end)*240) >= 24`, constants `_DOWNSTREAM_WINDOW_POINTS` / `_POST_SHIFT_SUPPORT_MIN_POINTS`) evaluated on `level_region_end_fraction` instead of the union end fraction |

Code points: values computed at `runtime/public_features.py:289-291`
(immediately after the union at `:288`, from the same in-scope `level_mask`
and `outlier_region` variables); emitted at `:316-324`; helper
`_end_fraction` at `:67-71`; the frozen pss formula extracted verbatim into
`_post_shift_support_sufficient` at `:74-78` and now called for **both** the
existing union field and the new level-only field (`:321-324`), so the two
readings cannot drift apart. Behavior of the existing field is unchanged
(verified bit-identically below). `outlier_region_fraction` was **not** added
(M0a judged it collapse-prone under train-scope means and excluded it from the
minimal set). The `missing > 0 => level_mask all-zero` branch (`:282-287`) is
untouched; the new fields inherit it (tested).

Vocabulary note: in `OBSERVABLE_FEATURES` the new boolean is declared *before*
`post_shift_support_sufficient` because two existing consumers
(`evaluation/functional/task_episode_harness/g1.py:2877` `_clause_condition`,
`:2645` invented-feature audit) resolve feature names by substring scan in
declaration order; declaring the longer name first prevents it from ever being
mis-resolved to the shorter one. No behavior of those consumers changes for
any text that exists today.

## 2. Zero-LLM deterministic verification

Cross-check table: `artifacts/functional/e2/m0b_field_wiring_report_v1.json`.

* Sample: 24 decision points from `m0a_mask_geometry_census_v1.json` (545
  rows), stratified up to 3 rows per (cohort x mask_class) cell present in the
  census, topped up from the global pool, seed 20260820. Coverage: all three
  cohorts (T233 6, electricity 12, weather 6) and all four classes (MIXED 9,
  LEVEL_ONLY 9, OUTLIER_ONLY 3, AMBIGUOUS 3).
* Prefixes rebuilt exactly as M0a built them
  (`load_cohort` + `values[uid][:support_origins[0]]`), extractor re-run.
* **Value-for-value cross-check: 24 / 24 rows fully match** on all four fields
  (`level_region_fraction`, `level_region_end_fraction`,
  `outlier_region_end_fraction` exact float equality;
  `level_only_post_shift_support_sufficient` == M0a `level_only_pss`).
  0 mismatches.
* **Regression: 24 / 24 rows bit-identical on every pre-existing mapping key**,
  compared against the pre-change extractor loaded from
  `git show HEAD:runtime/public_features.py` (commit `a2fb69a`) run on the same
  prefixes. The added-key set is exactly the four new fields on every row.
  `post_shift_support_sufficient` equals M0a's recorded
  `mapping_post_shift_support_sufficient` on all 24 rows (union semantics
  untouched).
* All new values finite, mapping stays inside the closed vocabulary
  (`runtime/public_features.py:332-333` assertion exercised on every row and in
  the tests below).

### Test suite status

Full suite (`python -m pytest tests -q -p no:randomly`), same machine, three runs:

| state | result |
| --- | --- |
| baseline (change stashed) | 16 failed, 648 passed, 3 skipped |
| after wiring, before h0 lock regen | 51 failed + 9 errors (all traced to one cause, below) |
| final (wiring + lock regen + new tests) | **16 failed, 652 passed, 3 skipped — failure set identical to baseline, 0 new, 0 fixed** |

The 16 failures are pre-existing and unrelated (they fail without the change;
e.g. `AgentProtocolError: stage_result names the wrong stage` replay-backend
drift in `test_ttha_agent`, the stale `m0-h2` release lock in `test_m0_release`
/ `test_f1_forecast_pilot`, MKL-environment-sensitive `test_valuator`). They
were not touched: not this task's problem to fix.

### All test-side changes (complete list)

1. `methods/ttha/harness/h0/snapshot.lock.json` — regenerated with the
   compiler's own `--write-lock` (the mechanism its error message names). The
   lock pins `dependency_shas` including the source shas of
   `contracts/observables.py` and `runtime/public_features.py`; the vocabulary
   change therefore invalidated it and cascaded into 44 failures/errors
   (`test_ttha_h0`, `test_edit_controller`, `test_slow_edit_contract`,
   `test_ttha_agent`, `test_minipipe_two_cycles`, `test_ttha_method`,
   `test_reference_wind_tunnel`) via `compile_snapshot(verify_lock=True)`.
   Diff of the regenerated lock: exactly
   `dependency_shas.observable_contract`,
   `dependency_shas.runtime:public_features`, and the derived
   `runtime_bundle_sha`; **`harness_content_sha` (semantic content) unchanged**,
   49/52 keys unchanged. This is a mechanical contract-snapshot update, no
   behavior assertion weakened. The *release* lock
   `artifacts/releases/m0-h2/harness/snapshot.lock.json` was already stale at
   baseline and was deliberately left alone.
2. `tests/minipipe/test_public_feature_calibration.py` — extended (the
   existing public-features test host, so no new test file was needed):
   * `test_m0b_mapping_emits_exactly_the_frozen_key_set` — exact 17-key mapping
     enumeration (13 legacy + 4 new) + closed-vocabulary subset, on all corpus
     families; makes silent field creep impossible.
   * `test_m0b_split_geometry_fields_match_their_m0a_definitions` — recomputes
     all four definitions from the extraction's own `level_mask` /
     `outlier_indices` + `_expand` and asserts equality, plus finiteness and
     [0,1] bounds.
   * `test_m0b_union_pss_semantics_are_untouched` — asserts
     `post_shift_support_sufficient` still equals the frozen formula on the
     union end fraction, and reproduces the M0a OUTLIER-source pss divergence
     deterministically (tail spike: union reading False, level-only True).
   * `test_m0b_missing_branch_forces_level_fields_to_zero` — the frozen
     missing-zeroing branch propagates into the new fields.
   No existing assertion was modified or deleted anywhere; the pre-existing
   `test_fast_agent_and_probe_panel_share_one_base_feature_extractor` compares
   full dicts and now covers the new fields automatically.

## 3. Micro behavior check (2-3 exposed Tasks, small LLM budget)

**Entry point** (existing, minimal, single-arm):
`evaluation/functional/task_episode_harness/agentic/runner.py::_run_arm` — the
A3 cold arm of `run_g1_pipeline`, i.e. base `h0` snapshot compiled from
`methods/ttha/harness/h0`, empty Skill start (`source_prior=None`, no local
Skills), frozen public context from `g1._w3_context_for`, workspace tool budget
6. A scratch driver outside the repo performed only the same per-task setup
`run_g1_pipeline` does (per-cohort state under `.m0b_micro_state/`, one arm,
one task), with the LLM budget clamped to **15 calls per task** via
`BudgetedAgentBackend(maximum_calls=15)`. Model/endpoint: the repo's frozen
`NF_MODEL`/`NF_BASE_URL`.

Tasks and selection basis:

* `T233 / e1v2_task_01` — frozen M0a hampel-NEGATIVE task (level-share high):
  specified by the M0b instruction.
* `T233 / e1v2_task_13` — frozen M0a outlier-POSITIVE block: specified by the
  M0b instruction.
* `electricity / e1v2_task_07` — the instruction asked for an electricity
  OUTLIER_ONLY or AMBIGUOUS task; **no electricity task has an OUTLIER_ONLY or
  AMBIGUOUS representative row** in the M0a JSON (all nine representatives are
  MIXED; OUTLIER_ONLY/AMBIGUOUS occur only on non-representative scope series).
  Reading the intent (outlier-dominated electricity decision point),
  `e1v2_task_07` was selected because its representative row is the sharpest
  electricity outlier case M0a recorded: pss-divergent with
  `pss_divergence_source=OUTLIER` (`outlier_region_end_fraction` 0.9821 vs
  `level_region_end_fraction` 0.2667, `union_pss` False vs `level_only_pss`
  True), and its scope contains 2 OUTLIER_ONLY rows. This deviation from the
  letter of the spec is flagged as such.

LLM cost: 4 + 9 + 5 = 18 calls; no task exceeded 15 (hard cap armed).

### (a) Do the new fields reach INSPECT/PROPOSE-visible tool receipts?

Yes, in all three tasks. Every `summarize_series` receipt served during the
INSPECT stage contained all four fields (task_01: 1 series, task_13: 3 series,
task_07: 1 series). Receipt values are **bit-identical to the M0a census rows**
for the exact (task, series) pairs the Agent inspected — e.g. task_01 x T234:
`level_region_fraction=0.0185546875`, `level_region_end_fraction=0.1826171875`,
`outlier_region_end_fraction=0.9856770833333334`,
`level_only_post_shift_support_sufficient=true`; verified likewise for
task_13 x T240 and task_07 x series "0".

**Mapping -> receipt transduction path** (cohort-scope gateway, the one the
runner actually uses):

1. `runtime/public_features.py:309-331` — base mapping construction (new
   fields `:316-324`).
2. `methods/ttha/public_tools.py:82-91` — agent-facing
   `extract_public_features` copies `dict(base.mapping)` wholesale (adds only
   the four probe-direction fields).
3. `evaluation/functional/task_episode_harness/agentic/gateway.py:203-216` —
   `CohortScopePublicToolGateway.call` computes the features per requested
   series and puts them under `public_result["features"]` for
   `summarize_series`; receipt minted at `:232-237` via
   `PublicToolReceipt.create` (`methods/ttha/public_tools.py:103-128`).
4. `evaluation/functional/task_episode_harness/agentic/fast_path.py:295-305` —
   receipts recorded into the stage trace; `:192-203` grounds INSPECT
   hypothesis citations in exactly the served feature keys (so a new-field
   citation is only accepted because the gateway really served it).
5. `runner.py:765-770` builds the gateway; `:835-845` drives
   inspect/propose/select over it.
   (The single-series `LocalPublicToolGateway` at
   `methods/ttha/public_tools.py:223-227` inherits the fields through the same
   base mapping.)

### (b) Do grounded hypotheses / first-choice Workflow family diverge by mechanism class?

**Mixed: hypotheses diverge by mechanism, the first-choice Workflow family does
not.** Behavior observation only; no Utility claim is made or implied.

* Hypothesis level — clear mechanism-keyed differentiation. In task_07 the
  Agent formed two hypotheses that split the union exactly along the new
  geometry: an outlier hypothesis whose region ends at the served
  `outlier_region_end_fraction` (0.7677) citing that field, and a separate
  narrow level hypothesis ([0.116, 0.1246]) citing the level fields. In
  task_01 the lead hypothesis's region ends at the served
  `level_region_end_fraction` (0.1826), not at the union end (0.9857). Before
  M0b those region boundaries were not observable at all (only the union was).
* First-choice family — homogeneous: all three tasks led with the level
  family (`repair_level_shift_local` / `repair_level_shift_explore` /
  `repair_level_shift_focus`). The outlier family appeared only as the second
  candidate on task_01 (`hampel_filter_spike`, addressing the same lead
  hypothesis). On the outlier-flagged tasks (13, 07) no outlier-family
  candidate was proposed.
* Stops: task_01 and task_13 ended in `REQUEST_OBSERVATION` (no Support probe
  consumed); task_07 probed its level candidate once (support gain -0.0432)
  and cleanly abstained (`AGENT_ABSTAIN`).

### (c) Verbatim fragments citing the new fields

task_01 (T233, lead hypothesis):

```json
{"hypothesis_id": "h1", "pattern_type": "extreme_deviation",
 "region_fractions": [0.0, 0.1826171875],
 "evidence_features": ["local_robust_z_peak", "level_excursion_score",
                        "estimated_level_offset",
                        "level_only_post_shift_support_sufficient"],
 "confidence": "high"}
```

task_13 (T233, lead hypothesis):

```json
{"hypothesis_id": "level_excursion_early", "pattern_type": "level_excursion",
 "region_fractions": [0.015625, 0.021752450980392156],
 "evidence_features": ["level_excursion_score", "estimated_level_offset",
                        "level_only_post_shift_support_sufficient",
                        "post_shift_support_sufficient"],
 "confidence": "medium"}
```

task_07 (electricity, both hypotheses):

```json
{"hypothesis_id": "extreme_dev_early_wide", "pattern_type": "extreme_deviation",
 "region_fractions": [0.007083333333333333, 0.7677083333333333],
 "evidence_features": ["local_robust_z_peak", "outlier_region_end_fraction"],
 "confidence": "medium"}
{"hypothesis_id": "level_shift_narrow", "pattern_type": "level_excursion",
 "region_fractions": [0.11625, 0.12458333333333334],
 "evidence_features": ["level_excursion_score", "estimated_level_offset",
                        "level_only_post_shift_support_sufficient"],
 "confidence": "high"}
```

All citations passed the INSPECT grounding validator, i.e. each cited field was
really served in a tool receipt in that same stage.

## Noted ambiguities and side effects (none blocking, all disclosed)

1. **Numeric bin edges not added.** `contracts/observables.py`
   `_NUMERIC_BIN_EDGES` has explicit fraction edges for the legacy region
   fields but the M0b instruction authorizes only mapping + vocabulary
   entries, so the three new numeric fields fall back to the default edges
   (0, 1, 3, 6) in `observable_numeric_bin`: any non-zero fraction bins as
   `very_low`. Nothing crashes and no current consumer bins these fields, but
   RLS-style applicability on binned values of the new fields would be
   near-degenerate until a future, separately-authorized edge decision.
2. **Electricity task class.** As above: no OUTLIER_ONLY/AMBIGUOUS
   representative exists in electricity; `e1v2_task_07` chosen on the
   documented outlier-divergence basis.
3. **Micro side effects.** The micro created scratch state under
   `.m0b_micro_state/` (context cache + snapshot stores), the unavoidable
   working artifacts of the existing runner entry; raw per-task traces were
   written outside the repo (`%TEMP%\m0b_micro_out\`). No repo experiment
   state directory was reused or overwritten.
4. **Pre-existing suite failures.** The 16 baseline failures listed above were
   left exactly as found.

## Verdict

All four fields wired, defined identically to M0a (24/24 value-for-value), the
legacy observation surface bit-identical (24/24), union pss untouched, suite at
baseline-green plus four new contract tests, and the fields demonstrably reach
the Agent's receipts and its grounded hypotheses on all three micro tasks.
