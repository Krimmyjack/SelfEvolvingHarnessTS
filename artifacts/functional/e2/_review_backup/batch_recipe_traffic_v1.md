# batch recipe -- `traffic` v1

One adopted data-processing plan for one already-exposed batch, produced by the frozen three-step recipe: scan the program menu at full batch, run a greedy harm-ordered exclusion mask search on the two best programs, then apply the delayed stability gate.

**Engineering effect measurement, not authorization evidence.**  No Skill is written, no Episode is formed, no Fast or Slow path is entered, and no execution right is granted or implied.  0 LLM calls. Already-exposed development data only.

> **Caveat.** The delayed window participated in the adoption decision, so for this recipe it is no longer an out-of-selection readout.  Both columns reported here are now in-selection.  Any external claim about this recipe needs a fresh window.

## Adopted plan

- kind: `MASKED_PLAN`
- program: `outlier_iqr`
- reverted to identity: `6`
- treated series: 11 of 12
- how to apply: apply `outlier_iqr` to every training series except ['6'], then retrain the Consumer once

Adoption path: masked plan cleared the delayed stability check.

## Comparison

| plan | support | delayed |
| --- | ---: | ---: |
| **adopted** (`outlier_iqr`, 1 reverted) | +0.665277 | +1.047075 |
| best full batch (`outlier_iqr`) | +0.607441 | +1.019774 |
| identity | 0.000000 | 0.000000 |

## Harm account (evaluation series worse than identity)

| plan | harmed series | total harm |
| --- | ---: | ---: |
| adopted | 0 | 0.000000 |
| best_full_batch | 0 | 0.000000 |
| identity | 0 | 0.000000 |

## Adoption trace

Rule (frozen, no tunable threshold): Candidates are the masked plans, ordered by Support aggregate gain, highest first.  A masked plan is adopted only if it also clears the delayed stability check: its delayed aggregate gain must be at least the delayed aggregate gain of the best full-batch plan.  The first candidate that clears it is adopted.  If none clears it, fall back to the best full-batch plan.  If the best full-batch plan's delayed aggregate gain is not positive, fall back to identity.

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `outlier_iqr` | 6 | +0.665277 | +1.047075 | +1.019774 | PASS |
| `outlier_mad` | 5, 6, 7 | +0.663904 | +0.965374 | -- | NOT_REACHED |

## Menu scan (full batch, Support aggregate gain)

| program | support aggregate gain |
| --- | ---: |
| `outlier_iqr` (searched) | +0.607441 |
| `outlier_mad` (searched) | +0.550279 |
| `winsorize` | +0.451495 |
| `repair_level_shift` | +0.060883 |
| `hampel_filter` | -0.075704 |
| `smooth_ma` | -0.259221 |
| `denoise_median` | -0.635374 |

## Reverted-series geometry (descriptive only)

Fields from the frozen M0a census, read verbatim.  No threshold is fitted here and nothing in this section feeds the adoption rule.

Reverted 1 series ({}); retained 11 ({}).

| field | reverted mean | retained mean | reverted range | retained range | ranges overlap |
| --- | ---: | ---: | --- | --- | --- |

Fields whose observed ranges do not overlap between reverted and retained: none.

