# One live trajectory, both guard actions legal

**Overall: `COMPILER_REJECTS`** -- EditShapeError: slow_edit.edit_manifest does not satisfy oneOf (0 matches): slow_edit.edit_manifest.predicted_agent_behavior_change[2] does not satisfy oneOf (0 matches): slow_edit.edit_manifest.predicted_agent_behavior_change[2] does not match pattern

Action the Agent chose: `RESCOPE_MASK_HARMED_SERIES`.

## Trajectory

| step | mode | plan before | plan after | support | delayed | harmed before | harmed after | gate |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `task_A` | FULL_PRICE_SEARCH | outlier_mad | outlier_mad | +0.072486 | +0.306380 | none | none | inactive |
| `task_B` | DIRECT_RECALL | identity | identity | +0.000000 | +0.000000 | none | none | inactive |
| `task_C` | DIRECT_RECALL | outlier_mad | outlier_mad | +0.191203 | +0.029688 | 99999904140 | 99999904140 | inactive |

## The Slow decision

Guard `delay-min-series-harm-rescope`: min_per_series_gain `lt` -0.005000 -> **RESCOPE_MASK_HARMED_SERIES** on the delayed window, applies to every_adoption.

> Rescope an adopted program when any evaluation series loses beyond the public harm line, while preserving the program for series without that loss.

- draw 1: PROPOSAL

Counters: valid 1 / protocol-failed 0 / LLM 1.

## What the action bought, against a veto twin

| step | as run | delayed | under veto | delayed | delta |
| --- | --- | ---: | --- | ---: | ---: |
| `task_A` | outlier_mad | +0.306380 | outlier_mad | +0.306380 | +0.000000 |
| `task_B` | identity | +0.000000 | identity | +0.000000 | +0.000000 |
| `task_C` | outlier_mad | +0.029688 | identity | +0.000000 | +0.029688 |

## Collateral check

`clean` -- each routed series is required to satisfy the guard's own test on the row's measured pre-gate vector, and no episode that harmed nothing may have had its decision moved

| step | crossing (measured) | routed | clean adoption | moved |
| --- | --- | --- | --- | --- |
| `task_A` | none | none | True | False |
| `task_B` | none | none | True | False |
| `task_C` | 99999904140 | none | False | False |

## Claim discipline

- Evidence level: **DEVELOPMENT**, **ON_GPT_5_6_SOL**.
- the guard reads the delayed window and the routing it produces is scored on that same window, exactly as a veto is.  This trajectory does not show that a rescope chosen on one window survives on another; no held-out window is opened.
- aggregate_gain is the mean of the per-series vector, so removing negative terms raises it.  A rescoped aggregate above the unguarded one is that arithmetic, not a free improvement: the series taken out are served identity and get nothing.
- every window here had its outcome opened by #17; this produces no fresh confirmation and no A5-over-A3 result

## Cost and integrity

- LLM calls: 3 / 8.
- Consumer retrains: 54 / 200.
- Frozen surface: 39 files, drift [].
- Wall seconds: 107.0.
