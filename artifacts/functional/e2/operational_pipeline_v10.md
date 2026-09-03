# One live trajectory, both guard actions legal

**Overall: `LIVE_RESCOPE_CONTAINS_WITHOUT_COLLATERAL`** -- the chain closed, the firing window's harm is gone, the aggregate is kept by projection, no non-crossing series was routed and no clean episode moved

Action the Agent chose: `RESCOPE_MASK_HARMED_SERIES`.

Retry of operational_pipeline_v9, which stopped at COMPILER_REJECTS on the behaviour-predicate contract mismatch.

## Trajectory

| step | mode | plan before | plan after | support | delayed | harmed before | harmed after | gate |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `task_A` | FULL_PRICE_SEARCH | outlier_mad | outlier_mad | +0.072486 | +0.306380 | none | none | inactive |
| `task_B` | DIRECT_RECALL | identity | identity | +0.000000 | +0.000000 | none | none | inactive |
| `task_C` | DIRECT_RECALL | outlier_mad | outlier_mad | +0.191203 | +0.029688 | 99999904140 | 99999904140 | inactive |
| `task_D` | DIRECT_RECALL | outlier_mad | outlier_mad -route 2 | +0.205520 | +0.095879 | 99999904140, 99999963862 | none | moved |

## The Slow decision

Guard `delayed_per_series_harm_rescope`: min_per_series_gain `lt` -0.005000 -> **RESCOPE_MASK_HARMED_SERIES** on the delayed window, applies to every_adoption.

> The observed aggregate gain concealed one evaluation-series gain of -0.1255567, well below the declared -0.005 harm line. Use the measured per-series delayed vector to serve identity only to series crossing that line while preserving the adopted plan elsewhere, then re-check the same guard.

- draw 1: PROPOSAL

Counters: valid 1 / protocol-failed 0 / LLM 1.

## What the action bought, against a veto twin

| step | as run | delayed | under veto | delayed | delta |
| --- | --- | ---: | --- | ---: | ---: |
| `task_A` | outlier_mad | +0.306380 | outlier_mad | +0.306380 | +0.000000 |
| `task_B` | identity | +0.000000 | identity | +0.000000 | +0.000000 |
| `task_C` | outlier_mad | +0.029688 | identity | +0.000000 | +0.029688 |
| `task_D` | outlier_mad -route 2 | +0.095879 | identity | +0.000000 | +0.095879 |

## Collateral check

`clean` -- each routed series is required to satisfy the guard's own test on the row's measured pre-gate vector, and no episode that harmed nothing may have had its decision moved

| step | crossing (measured) | routed | clean adoption | moved |
| --- | --- | --- | --- | --- |
| `task_A` | none | none | True | False |
| `task_B` | none | none | True | False |
| `task_C` | 99999904140 | none | False | False |
| `task_D` | 99999904140, 99999963862 | 99999904140, 99999963862 | False | True |

## Claim discipline

- Evidence level: **DEVELOPMENT**, **ON_GPT_5_6_SOL**.
- the guard reads the delayed window and the routing it produces is scored on that same window, exactly as a veto is.  This trajectory does not show that a rescope chosen on one window survives on another; no held-out window is opened.
- aggregate_gain is the mean of the per-series vector, so removing negative terms raises it.  A rescoped aggregate above the unguarded one is that arithmetic, not a free improvement: the series taken out are served identity and get nothing.
- every window here had its outcome opened by #17; this produces no fresh confirmation and no A5-over-A3 result

## Cost and integrity

- LLM calls: 3 / 8.
- Consumer retrains: 129 / 200.
- Frozen surface: 39 files, drift [].
- Wall seconds: 155.4.
