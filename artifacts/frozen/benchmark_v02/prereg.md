# benchmark-v0.2 — preregistration

Written and frozen **while the v0.2 Dev-Query run was still executing**, so that every
decision rule below was fixed before any v0.2 number existed. That timing is the only thing
that makes a null result here worth anything.

Supersedes nothing in the frozen design spec
(`2026-07-13-benchmark-data-metrics-pipeline-design.md` + amendment-1); it *fills a gap in
it* and freezes three faces the spec left open or that v0.1 got wrong.

---

## 0. The finding that forced this version

The frozen spec **never specified the downstream training pool's scope.**

§7 (Trainers) fixes what a trainer does — `L=48`, `H=48`, seeds `(0,1,2)`, ridge
`lambda=1e-3`, series-equal weighting, fixed epochs. It says nothing about *how many models
are fitted and over which series*. A search of all three frozen documents for
`shared model / per-config / training pool / one model / per dataset / joint` returns
nothing on the subject.

So the two prior arenas were each filling that gap silently, and differently:

| version | training unit | problem |
| --- | --- | --- |
| v0 | one model per `dataset × regime` cell | **illegal.** §5 forbids exposing `regime_tag` ("private regime tags … are never exposed"), and slicing the training pool by it hands the model exactly that label. |
| v0.1 | one model per role (all datasets pooled) | **legal but coupling.** A program applied to COVID moved the shared weights that scored traffic, so "the effect of this program on this dataset" was never isolated. |
| v0.2 | one model per `dataset` | this document. |

This is recorded plainly because the earlier working note described the v0.2 change as
"returning the training unit to the level the frozen contract stipulates." **That was
wrong** — the contract stipulates no level. v0.2 is a new protocol decision, argued below,
not the repair of a deviation.

---

## 1. Frozen face: the downstream training unit

**One closed-form model is trained per `(program, scenario, dose, replicate, dataset)`,**
on that dataset's inner-train only, with series-equal weighting.

Rationale, in order of weight:

1. **`dataset_id` is public; `regime_tag` is not.** Slicing by dataset leaks nothing a
   method could not already know. Slicing by regime hands over a benchmark-private label.
   This is what makes the v0.2 unit legal where v0's was not.

2. **It makes a dataset-conditioned oracle a world that was actually trained.** Under the
   role-pooled unit, an oracle that picks `forward_fill` for COVID and `seasonal_fill` for
   traffic describes a *mixed* corpus — and its loss was read off two models, each fitted to
   a corpus prepared with one program throughout. No model was ever fitted to the corpus the
   policy produces. See §3.

3. **Attribution.** With the pooled unit, a program's measured effect on dataset X includes
   its effect on every other dataset's contribution to the shared weights.

Accepted cost, stated in advance: GEFCom2012 contributes 3 Dev-Query series, so its
per-dataset judge is high-variance. GEFCom is already `claim_tier: supplementary` and no
GEFCom-only conclusion may be reported. This was true before v0.2 and does not change.

Consequence, stated in advance: **v0.1 Dev numbers are not comparable to v0.2 Dev numbers.**
Different arena (the version string re-draws the split and the corruption seeds) and
different ruler (the training unit). No v0.1 figure may be quoted beside a v0.2 figure as if
they measured the same thing.

---

## 2. Frozen face: the program pool

v0.1's public pool held four programs — `raw`, `forward_fill`, `seasonal_fill`, `h_ref` —
and **all four fill missing values.** On the `spike`, `gaussian`, `level_shift`, and
`local_permutation` lanes every program scored identically to four decimal places, because
not one of them can touch an outlier, a noise floor, a structural break, or a shuffled
ordering.

A pool that cannot act differently cannot make *choosing* mean anything. C1 — the claim that
the best preparation depends on the condition — was therefore untestable against v0.1, and
the "saturation" v0.1 reported on those lanes was a property of the pool, not of the data.

The v0.2 pool is organized by **mechanism coverage, not expected performance**
(`benchmark/programs.py`, pinned in `program_pool.json` with the SHA256 of every file that
implements it):

| defect mechanism | programs |
| --- | --- |
| none | `raw` (No-op + canonical ingestion) |
| point/block missing | `forward_fill`, `seasonal_fill` |
| outlier / spike | `winsorize`, `denoise_median` |
| additive noise | `denoise_stl`, `denoise_savgol`, `denoise_wavelet` |
| reference ladder | `h_ref` (the incumbent under test; not selectable by any oracle) |

**Declared capability gaps** — two mechanisms the operator library cannot answer, declared
here rather than papered over with a candidate that cannot work:

- **`timestamp_irregularity`** (scenario `local_permutation`). The corruption is realized in
  the value domain, because the legacy Monash series carry no timestamps at all and a real
  timestamp array cannot be perturbed uniformly across the roster without breaking CRN. A
  "re-align the timestamps" operator therefore has nothing to grip — the index is already
  uniform and correct.
- **`structural_break`** (scenario `level_shift`). `operators/registry.py` contains no
  changepoint-detection or segment-normalization operator. Nothing in the pool can find a
  break, so nothing can repair one.

A gap is a **finding about the operator library** and is reported as one. Any program added
later to close a gap is a pool change, and a pool change is a version bump.

**Exclusions are also frozen** (recorded in `program_pool.json`): `outlier_iqr`,
`outlier_mad`, `smooth_ma`, `smooth_ema`, `impute_linear`, `impute_ema`, `impute_fft`,
`period_complete` are excluded as mechanism-redundant; `znorm` and `minmax_norm` are excluded
by the frozen spec (`changes_target_space=True`). Deciding *after* seeing results which
operators "count" is the shortest path from a null to a discovery, so the exclusion list is
part of the freeze.

**Fallback is a hard failure, not a footnote.** `denoise_stl` and `denoise_wavelet` both
carry documented fallbacks to `denoise_savgol`; `denoise_savgol` and `denoise_median`
silently degrade to numpy equivalents when scipy is absent. An operator that quietly becomes
a different operator forges the very conditioning signal this benchmark measures. So:
`assert_pool_dependencies` refuses to build the pool unless scipy, statsmodels, and pywt all
import, the period is passed explicitly to `denoise_stl` so it never reaches its
`_guess_period` branch, and `run_pool_with_provenance` re-executes the pool under the
operator ledger. Verified before freezing: **162 (series × corruption-cell) executions,
zero masquerades.**

---

## 3. Frozen face: oracle semantics

The oracle is what defines "headroom", and headroom is what the L2 (TTHA) investment decision
turns on. It has to describe a reachable world.

**Retrained oracle (Gate-bearing).** A policy picks one pool program per
`(cell, scenario, dose)`. The corpus those picks produce is then **assembled and a model is
trained on it**, per dataset, through the identical path a Method takes. Two variants:

- `oracle_transfer_retrained` — policy selected on **Support-A**, executed on Dev-Query.
  **This is the ceiling the Gate reads.** Selection happens on disjoint series, so it is a
  ceiling a method could aim at rather than a winner's-curse artifact.
- `oracle_insample_retrained` — policy selected on Dev-Query itself. **Inflation envelope
  only**: reports how much of any apparent ceiling is winner's curse.

**Untrained-counterfactual oracle (descriptive only, NOT a Gate input).** The v0/v0.1
oracle: a program picked per cell, but each cell's loss read off a model trained on a corpus
prepared with *one* program throughout. No model was ever fitted to the mixed corpus those
picks describe. It is retained in the report under
`headroom_untrained_counterfactual_descriptive` purely so the v0.1 report stays legible, and
it may not be quoted as headroom.

**Neither oracle may select `h_ref`.** H_ref is the incumbent under test. A ceiling allowed
to pick it would make "headroom" partly a statement about H_ref's own quality instead of
about the pool's reachable space.

**What this does and does not measure.** Conditioning on `(scenario, dose)` alone would be
coherent under any training unit — a corruption realization is global to a run, so a single
program applied corpus-wide is a trained world. But that only tests the **degradation** axis.
C1 as this project states it is `H* = f(pattern, task)`, and *pattern varies across series
within a single run*. Testing it therefore requires assigning different programs to different
series, which produces a mixed corpus, which is precisely why the oracle must be retrained.
The retrained oracle conditions on both axes (`regime` inside `cell_id`, and
`scenario`/`dose`).

---

## 4. Frozen face: Support-A sub-split

**A-validation holds `certified_virgin` series only.** Any overlap group containing a
`confirmed_exposed`, `uncertain_legacy_exposure`, or `probe_consumed` series is forced into
A-discovery.

Those series — the Init-136 (80 `legacy_core` + 56 `probe_consumed`) and any other exposed
row — already fed the incumbent H_ref. Validating an update to that harness on data that
helped form it is a closed loop: the gate would be asking the harness to be judged by its own
training experience, and would pass any update that merely memorised its own history.

Realized in v0.2: 555 Support-A series → **420 discovery / 135 validation**, with 136 groups
forced to discovery by exposure. The membership is driven by `exposure_class`, not a
hard-coded uid list, so it stays correct if the roster changes. If the forcing ever empties
the validation half, the build fails loudly rather than degrading.

---

## 5. Frozen face: reporting protocol and detectability

### Two axes, never folded together

- **Headline aggregation**: `dataset × regime` (the frozen ladder: cell series-equal →
  regime dataset-macro → equal mean over regimes). Unchanged.
- **Mechanism diagnostics**: `dataset × scenario × dose`, its own table
  (`mechanism_diagnostics`). A question like "can anything in this pool touch a level shift?"
  is invisible in the headline fold, which averages every scenario together.

`programs_indistinguishable` (all pool programs agree to 1e-4) flags a cell where the pool
has **no action**. Such a cell may never be read as saturation.

### The resampling unit is the overlap group, not the series

METR-LA's 207 sensors are **20 spatial blocks**; sensors inside a block sit on the same
stretch of freeway. An IID bootstrap over series treats them as 207 independent draws and
reports an interval roughly `sqrt(10)` too narrow. `cluster_bootstrap_ci90` draws whole
overlap groups with replacement. For datasets whose overlap group is the series itself this
reduces exactly to the IID bootstrap, so it is applied everywhere.

**Disclosed optimism**: `monash:traffic_hourly`'s 862 sensors are one Bay Area road network,
but the pinned Monash release ships no sensor coordinates, so its groups are singletons and
its intervals remain optimistic. METR-LA is the spatially clean traffic read. The two are
cross-checks on each other, not independent confirmations.

### A cell that cannot resolve the effect says so

`power_panel` reports, per cell: effect size, 90% cluster-bootstrap CI, standard error,
`mde_80` (smallest true effect detectable at α=0.05 two-sided, power 0.80 — `2.8016 × SE`),
the material-headroom UID fraction, ingestion fill rate, and seasonal-scale validity.

A cell is marked **`diagnostic_unavailable`** when any of:
- `pool_cannot_act` — every program scores identically;
- `n_clusters < 2` — one independent unit admits no interval;
- `mde_80 > ε` — the instrument could not have seen a material effect even if present;
- `seasonal_scale_warning` — the sMASE denominator is tiny (COVID), so magnitudes are
  inflated.

**A `diagnostic_unavailable` cell contributes nothing to a saturation claim.** Absence of a
detected effect is not evidence of absence when `mde_80 > ε`.

`ε = SATURATION_GAP = 0.02` (frozen, conventional — a declared convention, not a calibrated
quantity).

---

## 6. The Gate — preregistered decision rule

Evaluated on **Dev-Query only**. Final-Query stays sealed; it is not read, not counted, not
peeked at.

Natural and Controlled lanes are judged **separately**:

| Natural | Controlled | verdict |
| --- | --- | --- |
| headroom | — | **real-world data preparation has value.** Strongest available conclusion. |
| saturated | headroom | **mechanism capability holds, real-world payoff is limited.** Write it that way; do not upgrade it. |
| saturated | saturated | **no measurable space even under a mechanism-covering pool.** An honest negative; the capability track still ships. |

**TTHA-0 (six-arm pilot) launches if and only if** headroom on the Gate-bearing ceiling
(`oracle_transfer_retrained` — transferable, not in-sample) is:

1. **material** — `gain_over_raw > ε` on the headline fold, and
2. **transferable** — measured against `oracle_transfer_retrained`, never
   `oracle_insample_retrained`, and
3. **readable** — in cells that are not `diagnostic_unavailable`, with
   `|effect| > mde_80`.

Otherwise TTHA-0 is shelved and the work is written up as the capability track.

**Headroom is a necessary condition, not a sufficient one.** It says an action difference
exists that *could* be chosen between. It says nothing about whether a harness can learn to
choose it. If headroom is present and TTHA-0 later fails, the failure is located in the
harness's ability to select, not in the benchmark.

---

## 7. Immutable — not touched by v0.2

Metric (sMASE, `m=7/24`, `≥32` seasonal pairs), harm threshold `δ=0.05`, `ε=0.02`,
`L=48`/`H=48`/stride 4, ridge `lambda=1e-3`, series-equal weighting, benchmark-owned
normalization (fitted on pre-method degraded inner-train, shared by every program), canonical
ingestion (linear interpolation with endpoint clamping), the `changes_target_space` ban on
the action surface, the corruption seed rule (content-hash keyed, CRN), the 9-cell corruption
grid, Support-B as one-shot post-freeze confirmation, and the sealed Final-Query.

The version string re-draws the outer split, the corruption realizations, and the Support-A
partition — by rule, not by hand.

---

## 8. Artifacts

`results/Benchmark_v0_2/` — `series_registry.jsonl`, `split_manifest.json`,
`support_a_subsplit.json`, `corruption_grid.json`, **`program_pool.json`**,
`dataset_manifest.json`, `metr_la_spatial_blocks.json`, `benchmark_manifest_v0.yaml`
(binds every one of the above by SHA256; `final_query_state: sealed`).

`data/benchmark_v0_2/clean_base/` — version-scoped derived layer. `data/benchmark_v0/{raw,
incoming}` remains the immutable shared source layer.

v0.2 roster: registry 1919 / eligible 1867; roles — support_a 555, support_b 331,
dev_query 373, final_query 570, u 38.
