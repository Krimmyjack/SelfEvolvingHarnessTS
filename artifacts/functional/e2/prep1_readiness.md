# PREP-1 readiness

construction only.  label: **partial**.  no live CPA.  no D3 values/labels.  no STAGE_REPORT.

## 1. Aggregator (ITT) — run1 partial checkpoint

source: `artifacts/functional/e2/s1v2_forward_run1.checkpoint.json`  
rows **18 / 28**.  complete positions 1–4.  run2 missing.  freeze `delta_material` = **0.102632** (0.050000 + 0.052632).

**provisional run1 verdict (not a finished-course freeze): `TREATMENT_EMPTY`**  
first-fault: `fewer_than_2_distinct_unguided_positive_tasks`  
merged: same, because run2 is absent.

ITT on the one finished Scope-matched beneficiary (GPOvY): inject 0/1, miss counted as A5 failure; conditional conversion rate = null (no successful inject).

### Per-unit sample (position 1, all four arms)

| # | role | unit | arm | deployed | held-out | regret | worst | harm | probes | LLM | fits | supply |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | producer_A | PowerCons | Static | identity | +0.0000 | +0.1167 | +0.0000 | 0 | 0 | 0 | 1 | 0 |
| 1 | producer_A | PowerCons | A3-reset | identity | +0.0000 | +0.1167 | +0.0000 | 0 | 2 | 4 | 5 | 0 |
| 1 | producer_A | PowerCons | K0-fixed | identity | +0.0000 | +0.1167 | +0.0000 | 0 | 1 | 5 | 1 | 0 |
| 1 | producer_A | PowerCons | A5-online | outlier_iqr | +0.0444 | +0.0722 | −0.0667 | 1 | 1 | 6 | 6 | 0 |

Full 18-row table is in `prep1_readiness.json` `aggregator.run1.per_unit`.

### Knowledge timeline (partial)

- Producer Episodes: A5 recorded earned/unearned rows from positions 1 and 3 (see json).
- Boundary cards after positions 1–4: **all `card_compiled=false`**, withheld `fewer_than_2_distinct_unguided_positive_tasks`.  Expected leaf count if compiled = 19.
- Beneficiary inject (GPOvY, position 4): entered_pool false, dual-gate false, identity deploy.
- First A5 vs K0 deploy divergence: **position 1** (A5 `outlier_iqr` vs K0 `identity`) — local adaptation, not the expected card-boundary fork at position 4.  No compiled card has appeared.

### Frozen gates (read from freeze; evaluated on partial rows)

- regret Δ = 0.102632; gap vs A3 0.237 (clears on partial sum) / vs K0 0.044 (does not).
- cost: 1 probe saved vs A3 over 2 convertible units (0.5 < 1) — cost gate false.
- These numbers are **not** a finished-course claim.

## 2. Capstone smoke (`--smoke-synthetic`)

| check | result |
|---|---|
| ok | **true** |
| backend | scripted, 0 CPA |
| TRAIN / TEST in memory | 80×178 / 11420×178; not written |
| D3 zip opened | **false** |
| Static / A3 / A5 | all identity deploy, gain 0.0 |
| llm / fits | 0/1 · 3/4 · 3/4 |
| checkpoint after Static then A3/A5 | wrote; resume skip list empty (first pass) |
| smoke score | `CAPSTONE_NEUTRAL` (synthetic noise; not an exam) |
| constants | seed 20260827, subset 476, caps 15/25 from freeze |
| h0 sha | `c3427b4e…` matches CAP-1 freeze |
| A5 pool | parameterized loader; smoke used empty path |

## 3. Unseal chain

Default CLI (no flags): prints lock and exits 2.

| probe | result |
|---|---|
| no record | locked |
| incomplete record (one forward only) | locked |
| complete record (forward ×2 SIGNAL + reverse SIGNAL + paths) | accepted |
| `load_official_d3` without record | refused |
| declared D3 zip open | refused (path string checked; file not opened) |

`--run` without an unseal file is the same lock.  PREP-1 does not parse D3 even if a valid record is later written.

## 4. Obligations

- in-flight files not written: `run_e2_s1v2_forward_course.py`, `run_e2_s1_curriculum_four_arms.py`, `methods/`
- `docs/STAGE_REPORT*` not written
- CPA / live relay: 0
- D3 numeric series and labels: 0
- full-repo pytest: not run
