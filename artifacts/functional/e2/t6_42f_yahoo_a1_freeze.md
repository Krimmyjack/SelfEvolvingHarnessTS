# #42f Yahoo S5 A1 freeze

verdict: **TARGET_FROZEN_SEALED**

Authorized download after `NO_LOCAL_SEALED_CANDIDATE`. A1Benchmark only. A2/A3/A4 not fetched. UCR-AD not mixed. NAB / NOAA 2025 / SMD not read.

0 LLM / 0 AD fit / 0 retrain.

## Source

- Official `yahoo/ydata-labeled-time-series-anomalies` GitHub: 404
- HuggingFace `YahooResearch/ydata-labeled-time-series-anomalies-v1_0`: gated
- Mirror used: `muditdham/time-series-auto-encoder` commit `24958b84f9b472b73c745cb20eb578d858021f1e`
- Path: `dataset/A1Benchmark/real_1.csv` … `real_67.csv` (67 files)
- Raw stays untracked: `data/benchmark_yahoo_s5_v1/raw/A1Benchmark/`

## Layout (public schema)

- Columns: `timestamp,value,is_anomaly` on every file
- Label form: **inline point label**
- No official train/test files → 0.7n outer wall is compatible (not LAYOUT_UNEXPECTED_STOP)
- Single natural cohort `yahoo_s5_a1` (no invented groups)

## Length gate (T6 `MIN_LENGTH=1000`)

- Downloaded 67
- Roster **65**
- Dropped (structure only): `real_54.csv` n=741, `real_62.csv` n=741
- Roster not chosen by labels

## Split (outer wall only)

- held-in = `[0, int(0.7n))` — feedback environment for #42g Part A
- held-out = `[int(0.7n), n)` — zero-feedback; open only in #42g Part D
- #42g r1…rR / Support / delayed / Slow must stay inside held-in

## Event map (frozen, fixture-proven)

Point labels → contiguous `is_anomaly==1` runs → event row-sets via existing `merge_events`.

Synthetic fixture (no real labels): `[0,0,1,0,1,1,1,0,0]` → `[[2],[4,5,6]]`; empty+silent F1=1; empty+noisy F1=0. **Pass.**

## Isolation

| store | path | contents |
|---|---|---|
| work (Agent-facing) | `data/benchmark_yahoo_s5_v1/work/` | `timestamp,value` only, 65 files |
| held-in vault | `data/benchmark_yahoo_s5_v1/vaults/held_in/` | row_index + timestamp + label bytes, 65 files |
| held-out vault | `data/benchmark_yahoo_s5_v1/vaults/held_out/` | same, held-out rows only, 65 files |

No label-value aggregates in this report. Roster SHAs are in `t6_42f_yahoo_a1_freeze.json` (identity lock for this roster only; not a new Manifest system).

## Exposure

- download-before: AGGREGATE_SEEN
- after value load: INSTANCE_SEEN
- held-in labels: sealed until #42g Part A
- held-out labels: sealed until #42g Part D

## #42g readiness

Exam contract issued. Next book must pre-freeze max rounds, feedback budget, per-round windows, and a Fast-only held-out entry. This freeze does not write those workflows.
