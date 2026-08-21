# One continuous operational run

**Overall: `COMPILER_REJECTS`** -- ValueError: guard rationale must be 1..600 characters

One un-relayed run of the V1 Harness on noaa_fresh x pooled, arm A5, Slow pinned to `claude-opus-5`. Development level: every window was locked before the run from the #17/#19 registers, nothing beyond index 17520 was read, A5-vs-A3 was not re-estimated and no new method was introduced.

## P2 -- the non-regression gate

PASS. 4 of 4 #19 task_C episodes reproduce digit-for-digit through the promoted enforcement path; 111 retrains, 0 LLM.

## Trajectory

| step | window | mode | card | local Skill | plan before gate | plan after gate | support | delayed | harmed | gate | retrains | LLM |
| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| `task_A` | 1104 | FULL_PRICE_SEARCH | hit | -- | `outlier_mad` | `outlier_mad` | +0.072486 | +0.306380 | none | inactive | 21 | 2 |
| `task_B` | 1800 | DIRECT_RECALL | hit | hit | `identity` | `identity` | +0.000000 | +0.000000 | none | inactive | 9 | 0 |
| `task_C` | 9864 | DIRECT_RECALL | hit | hit | `outlier_mad` | `outlier_mad` | +0.191203 | +0.029688 | 99999904140 | inactive | 18 | 0 |

### Per evaluation series, delayed gain

| step | phase | `99999903062` | `99999904140` | `99999923908` | `99999963862` |
| --- | --- | ---: | ---: | ---: | ---: |
| `task_A` | before gate | +0.567041 | +0.192931 | +0.017726 | +0.447821 |
| `task_A` | after gate | +0.567041 | +0.192931 | +0.017726 | +0.447821 |
| `task_B` | before gate | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `task_B` | after gate | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| `task_C` | before gate | +0.071979 | -0.125557 | +0.157494 | +0.014835 |
| `task_C` | after gate | +0.071979 | -0.125557 | +0.157494 | +0.014835 |

## Lifecycle

- Draft written: True (`fast_winner_e1v2_outlier_mad`).
- Probe fresh_probe: gain +0.205806, se +0.188765, gain/se +1.090277 -- out of selection.
- Promotion: True -> `fast_winner_e1v2_outlier_mad`.

## Attribution -- on this run's own record

- With the per-series risk reading: `RISK_GAP` at OUTCOME_RISK.
- Through the aggregate alone: `NO_ACTIONABLE_FAULT`.

## Cost and integrity

- LLM calls: 4 / 20.
- Consumer retrains: 165 / 300.
- Frozen surface: 36 files, drift [].
- Wall seconds: 104.5.
