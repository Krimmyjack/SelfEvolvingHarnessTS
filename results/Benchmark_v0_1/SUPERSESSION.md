# benchmark-v0.1 supersedes benchmark-v0

`benchmark-v0` was never consumed: its Final-Query was sealed at freeze and no unseal
event was ever written (`results/Benchmark_v0/virgin_ledger.jsonl` contains exactly one
`benchmark_freeze` event). Nothing is spent by replacing it, and `results/Benchmark_v0/`
is retained unchanged so the sealed state stays auditable.

## Why a new version rather than an edit

The benchmark version string is an input to both the outer-split group hash and the
corruption seed. v0.1 changes things that those hashes must depend on, so reusing the v0
string would have produced two different arenas both claiming to be `benchmark-v0` --
precisely the "declared != executed" drift this project has been burned by before.

What changed, and why each forces a version bump:

| change | why it cannot be an in-place edit |
| --- | --- |
| METR-LA's atomic split unit is now the **spatial block**, not the sensor | the outer split's group hash is computed over `overlap_group`; redefining it redraws every METR-LA role |
| corruption grid widened from 3 cells to 9 | the corruption seed is keyed on `(version, content_sha, scenario, dose, replicate)`; new cells are new draws |
| NOAA weather pool grown from 12 to 64 stations (6 -> 38 admitted U series) | the roster changed |
| METR-LA (207) and GEFCom2012 (20) entered the roster | the roster changed |

## Known integrity gap in v0 (pre-existing, discovered 2026-07-14)

`results/Benchmark_v0/benchmark_manifest_v0.yaml` pins
`acquisition_manifest_sha256 = 52d3311c...`, but the shared source-layer file
`data/benchmark_v0/acquisition_manifest.json` no longer hashes to that value. The drift
predates this supersession: it happened when METR-LA and GEFCom2012 were acquired into the
shared source store, which rewrote the acquisition manifest in place.

Consequence: **v0's acquisition binding is not verifiable and v0 must not be re-frozen or
re-run.** Its split/registry bindings still verify, and its Final-Query was never opened,
so nothing was measured against a broken binding. v0.1 avoids the recurrence by pinning
its own acquisition digest (`cdfe8b41...`) at its own freeze.

The root cause -- one mutable acquisition manifest shared across versions -- is why v0.1
also separates the **immutable source layer** (`data/benchmark_v0/raw/`, `incoming/`,
shared and append-only) from the **derived layer** (`data/benchmark_v0_1/clean_base/`,
version-scoped). `clean_base/record.json` carries `overlap_group`, which is a protocol
decision rather than a fact about the bytes, so a version that redefines it needs its own
derived layer. v0's `clean_base` is untouched.

## Carried forward unchanged

- The three Controlled-v0 missingness cells (`block` 0.12 / 0.24, `scattered` 0.12) are
  the same questions, so v0 and v0.1 remain comparable cell-by-cell. Their *seeds* differ,
  because the version is part of the corruption key.
- Init Harness membership (136 series = 80 legacy_core + 56 probe_consumed_extension) is
  exposure-driven, not hash-driven, so it is identical in v0.1. See
  `init_harness_manifest.json`.
- Headline geometry (L=48, H=48, stride 4, min_length 207), harm threshold 0.05, bootstrap
  B=2000 / seed 20260713, benchmark-owned normalization.
