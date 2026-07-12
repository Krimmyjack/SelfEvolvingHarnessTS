# TS Data-Readiness Benchmark Data, Metrics, and Pipeline Design

> Status: approved design. This is the implementation-facing companion to
> `idea/Benchmark_v0_Forecast_Design.md` v3. The benchmark document governs the
> scientific protocol; this document governs code interfaces.

## 1. Scope

Build a reproducible forecast data-readiness benchmark that materializes all
v0-core and v1-paper candidates, keeps raw/clean/derived layers distinct,
freezes splits before final evaluation, sends every method through one common
pipeline, and protects Final-Query with one frozen evaluation campaign. Existing
operators and model classes are reusable; frozen P6 code is not modified.

Headline frequencies are daily and hourly with `L=48`, `H=48`, and
`MIN_LEN=207`. Monthly data are Dev-only under this protocol.

## 2. Data and storage

```text
SelfEvolvingHarnessTS/data/benchmark_v0/
  raw/          immutable sources, source metadata, SHA256
  clean_base/   entity series, timestamps, natural-missing masks, resampling
  derived/      deterministic corruption artifacts/manifests
  incoming/     user-supplied restricted archives before validation
```

Sources are: pinned Monash `nn5_daily`, `covid_deaths`, `traffic_hourly`, and
`electricity_hourly`; METR-LA from the DCRNN authors' release; UCI
ElectricityLoadDiagrams20112014; user-supplied ENTSO-E Actual Total Load;
user-supplied GEFCom 2012/2014 load tracks; and NOAA NCEI Global Hourly/ISD
temperature as the new sealed weather domain.

METR-LA 5-minute readings and UCI 15-minute kW readings use a frozen hourly
mean. Raw files never change. Missing license, unresolved overlap, insufficient
length, undefined seasonal scale, or excessive natural missingness keeps a
source in candidate/Dev status.

The legacy artifact contains 83 exposed series: 20 each from `nn5_daily`,
`fred_md`, `tourism_monthly`, and `covid_deaths`, plus one each from
`us_births`, `saugeenday`, and `sunspot`. All are Support-A/Dev only.

## 3. Probe, registry, and materialization

`probe` may inspect source metadata and clean inner-train structure but never
calculates model loss, utility, or method comparisons. It produces source
revisions/hashes; series/site identity, frequency, length, natural missingness,
license, overlap and exposure; regime features/tags; and admission reasons.

`materialize` is deterministic and fail-loud. It never substitutes the next
item when a hash-selected item fails. Natural missing masks are saved before
clean-base filling.

## 4. Split and corruption

Outer split is atomic by `series_uid`; overlap groups cannot cross
Support/Query. Inner split is final 48 for test, preceding 48 for val, and all
earlier values for train.

The unique corruption seed is derived from canonical SHA256 encoding of:

```text
(benchmark_version, clean_content_sha, scenario, dose, replicate_idx)
```

This is invariant to source order and subset selection. Headline
`replicate_idx=(0,1)`. The manifest stores the rule and every realized
seed/digest. Every method consumes identical materialized corruptions (CRN).

## 5. Method API and ownership

```python
prepared = method.prepare(train_series, task_spec, observed_pattern_spec)
method.adapt(support_data, feedback_api, budget)  # optional
```

`feedback_api` exposes only Support-A inner-val through the closed-form channel
and records every call. Query/U, clean future, private regime tags, and an
unrestricted downstream trainer are never exposed.

Prepared output must keep length, dimensionality, original physical units, and
declared deterministic replay. `znorm`, `minmax_norm`, and every operator with
`changes_target_space=True` are excluded from the v0 method action surface and
baseline wrappers.

Normalization is benchmark-owned. Mean/std are frozen from finite observations
in pre-method degraded inner-train and shared by every method, baseline,
training window, evaluation context, and inverse transform. Adaptive
normalization requires a benchmark version bump.

## 6. Canonical ingestion

The unique ingestion rule linearly interpolates NaNs with nearest-value endpoint
clamping. Infinity and all-NaN outputs are invalid, and length cannot change.
It records `ingestion_fill_rate` per method/uid/scenario/cell; values above 0.01
set a descriptive dependency flag.

For `train_effect`, raw degraded evaluation context is ingested once, cached by
uid/scenario/replicate, and reused bit-for-bit by every method. Context/joint
effects ingest prepared context separately. Raw is reported as **No-op +
canonical ingestion**; missing-data gains are incremental to canonical linear
fill.

## 7. Trainers

All trainers share eligibility, ingestion, normalization, `L=48`, `H=48`,
inner-train-only windows, stride, and model seeds `(0,1,2)`. Epochs are fixed;
Query labels never control early stopping.

Closed-form DLinear keeps lambda `1e-3` and series-equal sufficient statistics.
Benchmark Adam-DLinear intentionally differs from pooled-window P6 Adam. It
uses weighted loss, never weighted sampling:

```text
L = sum_s sum_(w in s) (1/W_s) loss_(s,w)
    / sum_s sum_(w in s) (1/W_s)
```

With uniform no-replacement shuffle, a batch of size `b` from `N` windows and
`S` series optimizes:

```text
L_batch = N/(b*S) * sum_(j in batch) (1/W_s(j)) loss_j
```

The final short batch uses the same formula. Manifest identity freezes
initialization/shuffle seed, batch size, epochs, all Adam parameters,
deterministic device mode, and this formula. LSTM-scratch is independent but
uses the same series-equal weighting.

## 8. Metric and aggregation

```text
scale_i = mean(|y_t-y_(t-m)|) on clean inner-train observed pairs
sMASE_i = mean(|y_test-y_pred|) / scale_i
```

Use `m=7` daily and `m=24` hourly. Require at least 32 observed seasonal pairs.
A scale no larger than `1e-8*max(1,mean(abs(y_train)))` makes the series
Dev-only before split freeze.

Required folding order:

```text
within uid x scenario x corruption replicate: average 3 model seeds
-> within uid x scenario: average 2 corruption replicates
-> within uid: equal-average frozen dose/scenario values
-> cell: series-equal mean
-> regime: dataset macro mean
```

Per-dose/scenario tables remain separate. Bootstrap accepts exactly one paired
gain row per uid and rejects uncollapsed seed/replicate/dose rows. All methods
share model seeds and CRN.

`absolute_gain = loss_reference-loss_method` is the sole inferential gain.
Relative gain is descriptive. Harm is `loss_method-loss_reference > 0.05`.
Bootstrap uses `B=2000`, master seed `20260713`, and canonical hash-derived
comparison/cell sub-seeds. METR-LA inference is conditional on that road network
unless frozen spatial blocks support a broader claim.

## 9. Baselines and oracles

Public-API baselines are Raw (No-op + canonical ingestion), best-fixed (one
Support-A-selected universal program), and H_ref (current deterministic ladder
plus random-supply wrapper filtered through the benchmark contract).

Oracles are privileged runner diagnostics, not Methods:

- `oracle_transfer`: learn a per-cell mapping on Support-A, freeze it, evaluate
  it on Query;
- `oracle_insample`: select and evaluate on Query, labeled as a winner's-curse
  inflated envelope.

Neither oracle enters headline ranking.

## 10. Support and Final campaign

Support-A is repeatable development and requires a full contract dry-run SHA.
Support-B is one-shot confirmation after code/config freeze. A failed
confirmation cannot be repaired and resubmitted to the same campaign.

```text
freeze benchmark version
-> freeze roster, method/code SHAs, budgets, seeds, order
-> verify Support-A dry-run and Support-B confirmation SHAs
-> WAL-commit one campaign unseal before reading Final-Query
-> run every roster entry
-> close the campaign permanently
```

Baseline validation uses Dev-Query, never Final-Query. A method added after
roster freeze needs a new Final split and benchmark version/milestone.

## 11. Ledger and failure classes

Ledger adapts P6 locking, append-only WAL, fsync, event hash chain, and replay.
Events are `campaign_freeze`, `unseal`, `method_access`, `method_result`, and
`campaign_close`. Every access records campaign/method/code/run IDs and all
manifest SHAs. Unseal and method access are durable before their corresponding
reads.

Method exceptions, contract violations, changed length, infinity, all-NaN,
forbidden transforms, and frozen timeout overrun are terminal `invalid` or
`failed_timeout` results. Query is consumed; changed code cannot retry.

Only method-independent interruption such as power loss, process termination,
disk I/O, or hardware/evaluator crash may resume. Resume requires identical
campaign/run IDs, method and runner code SHAs, input/materialization SHA, and
checkpoint bytes. An evaluator patch invalidates the campaign.

## 12. Freeze surface

Freeze source and content hashes; registry/overlap/regimes/splits/salts;
corruption scenarios/doses/replicates/derived seeds; normalization/ingestion;
eligibility/windows/stride/model identity/seeds/batch order/weighted loss and
optimizer parameters; harm 0.05; bootstrap B=2000 and seed 20260713; numeric
prepare/trainer timeouts measured as `2*empirical p95` on same-path,
same-hardware Dev runs; and campaign roster/budgets/code SHAs/order.

## 13. Modules

```text
benchmark/
  sources.py registry.py materialize.py split.py corruption.py
  method_api.py ingestion.py trainers.py metrics.py aggregate.py
  baselines.py ledger.py probe.py runner.py report.py
run_benchmark.py
tests/test_benchmark_*.py
```

Files may merge only if there remains one corruption key, one ingestion rule,
one gain function, and one aggregation function. Frozen P6 stays untouched.

## 14. Required verification

Test-first implementation must prove corruption reorder/subset invariance; raw
immutability, resampling, overlap, and legacy exclusion; method visibility and
normalization ban; ingestion/fill accounting/bit-identical train context;
weighted loss and deterministic unweighted shuffle; rejection of duplicate uid
or uncollapsed repeats; distinct oracle paths; ledger-before-read and tamper
detection; terminal method failure and exact-hash idempotent infrastructure
resume; and Final gating on dry-run, confirmation, and roster membership.

```text
probe
-> benchmark freeze
-> Dev full-pipeline dry-run
-> data materialization and Final sealing
-> method/experiment preregistration and Support-B confirmation
-> one Final evaluation campaign
```
