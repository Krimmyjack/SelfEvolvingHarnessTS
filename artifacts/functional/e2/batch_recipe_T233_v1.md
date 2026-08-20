# batch recipe -- `T233` v1

One adopted data-processing plan for one already-exposed batch, produced by the frozen three-step recipe: scan the program menu at full batch, run a greedy harm-ordered exclusion mask search on the two best programs, then apply the delayed stability gate.

**Engineering effect measurement, not authorization evidence.**  No Skill is written, no Episode is formed, no Fast or Slow path is entered, and no execution right is granted or implied.  0 LLM calls. Already-exposed development data only.

> **Caveat.** The delayed window participated in the adoption decision, so for this recipe it is no longer an out-of-selection readout.  Both columns reported here are now in-selection.  Any external claim about this recipe needs a fresh window.

## Adopted plan

- kind: `BEST_FULL_BATCH`
- program: `winsorize`
- reverted to identity: none
- treated series: 12 of 12
- how to apply: apply `winsorize` to every training series, then retrain the Consumer once

Adoption path: no masked plan cleared the delayed stability check; fell back to the best full-batch plan, whose delayed gain is positive.

## Comparison

| plan | support | delayed |
| --- | ---: | ---: |
| **adopted** (`winsorize`, full batch) | +0.072156 | +0.116627 |
| best full batch (`winsorize`) | +0.072156 | +0.116627 |
| identity | 0.000000 | 0.000000 |

## Harm account (evaluation series worse than identity)

| plan | harmed series | total harm |
| --- | ---: | ---: |
| adopted | 2 | 0.678648 |
| best_full_batch | 2 | 0.678648 |
| identity | 0 | 0.000000 |

## Adoption trace

Rule (frozen, no tunable threshold): Candidates are the masked plans, ordered by Support aggregate gain, highest first.  A masked plan is adopted only if it also clears the delayed stability check: its delayed aggregate gain must be at least the delayed aggregate gain of the best full-batch plan.  The first candidate that clears it is adopted.  If none clears it, fall back to the best full-batch plan.  If the best full-batch plan's delayed aggregate gain is not positive, fall back to identity.

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `winsorize` | T234, T235, T244 | +0.139734 | +0.062162 | +0.116627 | FAIL |
| `outlier_iqr` | T234, T240, T244, T247 | +0.132319 | +0.047465 | +0.116627 | FAIL |

## Menu scan (full batch, Support aggregate gain)

| program | support aggregate gain |
| --- | ---: |
| `winsorize` (searched) | +0.072156 |
| `outlier_iqr` (searched) | +0.058576 |
| `outlier_mad` | +0.023298 |
| `repair_level_shift` | -0.063758 |
| `hampel_filter` | -0.165671 |
| `denoise_median` | -0.528073 |
| `smooth_ma` | -0.663197 |

## Reverted-series geometry (descriptive only)

The adopted plan reverts nothing, so there is no reverted-series geometry to describe.

For reference, the masked candidate the stability gate turned down would have reverted `T234`, `T235`, `T244` ({'MIXED': 3}).

