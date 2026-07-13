# benchmark-v0.1 supersedes benchmark-v0

`benchmark-v0` was never consumed: its Final-Query was sealed at freeze and no unseal
event was ever written (`results/Benchmark_v0/virgin_ledger.jsonl` contains exactly one
`benchmark_freeze` event). Nothing is spent by replacing it, and `results/Benchmark_v0/`
is retained unchanged so the sealed state stays auditable.

## Why a new version rather than an edit

The benchmark version string is an input to both the outer-split group hash and the
corruption seed. v0.1 changes things those hashes must depend on, so reusing the v0
string would have produced two different arenas both claiming to be `benchmark-v0` --
precisely the "declared != executed" drift this project has been burned by before.

What changed, and why each forces a version bump:

| change | why it cannot be an in-place edit |
| --- | --- |
| METR-LA's atomic split unit is now the **spatial block**, not the sensor | the outer split's group hash is computed over `overlap_group`; redefining it redraws every METR-LA role |
| corruption grid widened from 3 cells to 9 | the corruption seed is keyed on `(version, content_sha, scenario, dose, replicate)`; new cells are new draws |
| NOAA weather pool grown from 12 to 64 stations (6 -> 38 admitted U series) | the roster changed |
| METR-LA (207) and GEFCom2012 (20) entered the roster | the roster changed |

## Integrity defect 1 (pre-existing): the acquisition binding drifted

`results/Benchmark_v0/benchmark_manifest_v0.yaml` pins
`acquisition_manifest_sha256 = 52d3311c...`, but the shared source-layer file
`data/benchmark_v0/acquisition_manifest.json` no longer hashes to that value. The drift
predates this supersession: it happened when METR-LA and GEFCom2012 were acquired into
the shared source store, which rewrote the acquisition manifest in place.

The root cause -- one mutable acquisition manifest shared across versions -- is why v0.1
separates the **immutable source layer** (`data/benchmark_v0/raw/`, `incoming/`, shared
and append-only) from the **derived layer** (`data/benchmark_v0_1/clean_base/`,
version-scoped). `clean_base/record.json` carries `overlap_group`, which is a protocol
decision rather than a fact about the bytes, so a version that redefines it needs its own
derived layer. v0's `clean_base` is untouched.

## Integrity defect 2 (pre-existing): line endings broke the digests

The frozen artifacts are identified by SHA256 of their bytes, but they were written with
`Path.write_text`, which uses the platform's newline convention -- on Windows, CRLF.
`registry.py` happened to pin `newline="\n"`; nothing else did. So the artifacts on disk
were a **mix of LF and CRLF**, and each recorded digest was computed over whichever its
writer happened to emit.

That made the freeze irreproducible in two directions at once:

- A re-freeze of byte-identical content on Linux would emit LF and therefore *different
  digests*; the manifest would not verify.
- Git's `text=auto` normalizes CRLF to LF in the object store and restores CRLF on
  Windows checkout, so a clean checkout produced files that no longer hashed to the
  digests recorded in the manifest shipped beside them -- the integrity check meant to
  catch tampering fires on an untampered checkout. Confirmed in practice: after merging
  to `main`, `results/Benchmark_v0/series_registry.jsonl` no longer matched its pinned
  `registry_sha256`.

v0.1 fixes both ends. Every artifact whose digest is pinned now goes through
`benchmark/materialize.py: write_text_lf` (byte-exact LF, platform-independent), and
`.gitattributes` marks the artifact and pinned-document paths `-text` so git never
rewrites them. All eight v0.1 bindings -- registry, split, dataset manifest, Support-A
subsplit, corruption grid, spatial blocks, Dev report, acquisition manifest -- verify
byte-exactly with zero CRLF.

**Consequence for v0: it must not be re-frozen or re-run.** Its acquisition binding is
unverifiable and its digests record mixed-newline bytes that cannot be retro-fixed. Its
split and registry content is still readable, and its Final-Query was never opened, so
nothing was ever measured against a broken binding.

## Carried forward unchanged

- The three Controlled-v0 missingness cells (`block` 0.12 / 0.24, `scattered` 0.12) are
  the same questions, so v0 and v0.1 stay comparable cell-by-cell. Their *seeds* differ,
  because the version is part of the corruption key.
- Init Harness membership (136 series = 80 legacy_core + 56 probe_consumed_extension) is
  exposure-driven, not hash-driven, so it is identical in v0.1. See
  `init_harness_manifest.json`.
- Headline geometry (L=48, H=48, stride 4, min_length 207), harm threshold 0.05,
  bootstrap B=2000 / seed 20260713, benchmark-owned normalization.
