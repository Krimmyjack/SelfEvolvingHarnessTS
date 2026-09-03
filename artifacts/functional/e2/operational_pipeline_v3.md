# K trajectories of the one-run operational pipeline

**Overall: `PIPELINE_NEVER_EXERCISES_SLOW`** -- all 3 trajectories stopped before the Slow edit; the chain's last four links go untested in this sample

K = 3, fixed before the first draw. Every trajectory is on the record whatever it did: none was discarded, re-thrown or seeded. Each ran on a store of its own, noaa_fresh x pooled, arm A5, Slow pinned to `claude-opus-5`.

## Per trajectory

| trajectory | verdict | reached Slow | LLM | retrains |
| --- | --- | --- | ---: | ---: |
| `T1` | `PIPELINE_RUNS_NO_FAULT_SAMPLE` | no | 2 | 123 |
| `T2` | `PIPELINE_RUNS_NO_FAULT_SAMPLE` | no | 2 | 123 |
| `T3` | `PIPELINE_RUNS_NO_FAULT_SAMPLE` | no | 2 | 57 |

### Reasons

- `T1`: the trajectory ran end to end but produced no harm past the line, so the fold had nothing actionable to name; links 6 to 9 -- attribution to a Scope/Risk face, the Slow edit, the compiler and the post-update behaviour -- are untested in this sample
- `T2`: the trajectory ran end to end but produced no harm past the line, so the fold had nothing actionable to name; links 6 to 9 -- attribution to a Scope/Risk face, the Slow edit, the compiler and the post-update behaviour -- are untested in this sample
- `T3`: the trajectory ran end to end but produced no harm past the line, so the fold had nothing actionable to name; links 6 to 9 -- attribution to a Scope/Risk face, the Slow edit, the compiler and the post-update behaviour -- are untested in this sample

## Trajectory tables

### `T1` -- `PIPELINE_RUNS_NO_FAULT_SAMPLE`

| step | window | mode | card | local Skill | shortlist | cited | overlap | plan before | plan after | support | delayed | harmed | retrains | LLM |
| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| `task_A` | 1104 | FULL_PRICE_SEARCH | hit | -- | repair_level_shift, outlier_iqr | R1-2, R3-1 | 1 | `outlier_iqr` | `outlier_iqr` | +0.016726 | +0.066941 | 99999923908 | 87 | 2 |
| `task_B` | 1800 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 9 | 0 |
| `task_C` | 9864 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 12 | 0 |
| `task_D` | 10560 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 9 | 0 |

### `T2` -- `PIPELINE_RUNS_NO_FAULT_SAMPLE`

| step | window | mode | card | local Skill | shortlist | cited | overlap | plan before | plan after | support | delayed | harmed | retrains | LLM |
| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| `task_A` | 1104 | FULL_PRICE_SEARCH | hit | -- | repair_level_shift, outlier_iqr | R1-2, R3-1 | 1 | `outlier_iqr` | `outlier_iqr` | +0.016726 | +0.066941 | 99999923908 | 87 | 2 |
| `task_B` | 1800 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 9 | 0 |
| `task_C` | 9864 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 12 | 0 |
| `task_D` | 10560 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 9 | 0 |

### `T3` -- `PIPELINE_RUNS_NO_FAULT_SAMPLE`

| step | window | mode | card | local Skill | shortlist | cited | overlap | plan before | plan after | support | delayed | harmed | retrains | LLM |
| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| `task_A` | 1104 | FULL_PRICE_SEARCH | hit | -- | repair_level_shift, outlier_iqr | R1-2, R3-1 | 1 | `outlier_iqr` | `outlier_iqr` | +0.016726 | +0.066941 | 99999923908 | 21 | 2 |
| `task_B` | 1800 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 9 | 0 |
| `task_C` | 9864 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 12 | 0 |
| `task_D` | 10560 | DIRECT_RECALL | hit | hit | -- | -- | 0 | `identity` | `identity` | +0.000000 | +0.000000 | none | 9 | 0 |

## task_A shortlist stability, all draws on record

| run | shortlist | cited | overlap | adopted | support | delayed | ladder |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| fresh_confirmation_v1 (#17) a5_pooled | outlier_iqr, outlier_mad | R1-2, R1-1, R3-1 | 2 | `outlier_mad` | +0.072486 | +0.306380 | `GATE_PASS_ADOPT_NAMED` |
| operational_pipeline_v1 (#21) | repair_level_shift, outlier_mad | R1-1, R3-1 | 1 | `outlier_mad` | +0.072486 | +0.306380 | `GATE_PASS_ADOPT_NAMED` |
| operational_pipeline_v2 (#22) | repair_level_shift, hampel_filter | R1-1, R1-2, R1-3 | 0 | `identity` | +0.000000 | +0.000000 | `GATE_FAIL_FALLBACK_IDENTITY` |
| operational_pipeline_v3/T1 | repair_level_shift, outlier_iqr | R1-2, R3-1 | 1 | `outlier_iqr` | +0.016726 | +0.066941 | `GATE_PASS_ADOPT_NAMED` |
| operational_pipeline_v3/T2 | repair_level_shift, outlier_iqr | R1-2, R3-1 | 1 | `outlier_iqr` | +0.016726 | +0.066941 | `GATE_PASS_ADOPT_NAMED` |
| operational_pipeline_v3/T3 | repair_level_shift, outlier_iqr | R1-2, R3-1 | 1 | `outlier_iqr` | +0.016726 | +0.066941 | `GATE_PASS_ADOPT_NAMED` |

6 draws, 4 distinct shortlists, 5 adopted a program, 5 cited a clause naming something they shortlisted. Overlap is computed from the clause texts against the frozen menu, not from what the Agent reported. n is small; nothing here is causal.

## Cost and integrity

- LLM calls: 6 / 24.
- Consumer retrains: 303 / 450.
- Frozen surface: 38 files, drift [].
- Wall seconds: 76.1.
