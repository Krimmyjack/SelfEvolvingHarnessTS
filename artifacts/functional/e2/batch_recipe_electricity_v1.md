# batch recipe -- `electricity` v1

One adopted data-processing plan for one already-exposed batch, produced by the frozen three-step recipe: scan the program menu at full batch, run a greedy harm-ordered exclusion mask search on the two best programs, then apply the delayed stability gate.

**Engineering effect measurement, not authorization evidence.**  No Skill is written, no Episode is formed, no Fast or Slow path is entered, and no execution right is granted or implied.  0 LLM calls. Already-exposed development data only.

> **Caveat.** The delayed window participated in the adoption decision, so for this recipe it is no longer an out-of-selection readout.  Both columns reported here are now in-selection.  Any external claim about this recipe needs a fresh window.

## Adopted plan

- kind: `MASKED_PLAN`
- program: `outlier_iqr`
- reverted to identity: `1`, `2`, `6`
- treated series: 9 of 12
- how to apply: apply `outlier_iqr` to every training series except ['1', '2', '6'], then retrain the Consumer once

Adoption path: masked plan cleared the delayed stability check.

## Comparison

| plan | support | delayed |
| --- | ---: | ---: |
| **adopted** (`outlier_iqr`, 3 reverted) | +0.034643 | +0.016343 |
| best full batch (`winsorize`) | +0.011741 | +0.000022 |
| identity | 0.000000 | 0.000000 |

## Harm account (evaluation series worse than identity)

| plan | harmed series | total harm |
| --- | ---: | ---: |
| adopted | 1 | 0.010755 |
| best_full_batch | 4 | 0.093583 |
| identity | 0 | 0.000000 |

## Adoption trace

Rule (frozen, no tunable threshold): Candidates are the masked plans, ordered by Support aggregate gain, highest first.  A masked plan is adopted only if it also clears the delayed stability check: its delayed aggregate gain must be at least the delayed aggregate gain of the best full-batch plan.  The first candidate that clears it is adopted.  If none clears it, fall back to the best full-batch plan.  If the best full-batch plan's delayed aggregate gain is not positive, fall back to identity.

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `outlier_iqr` | 1, 2, 6 | +0.034643 | +0.016343 | +0.000022 | PASS |

## Menu scan (full batch, Support aggregate gain)

| program | support aggregate gain |
| --- | ---: |
| `winsorize` (searched) | +0.011741 |
| `outlier_iqr` (searched) | -0.002188 |
| `denoise_median` | -0.023076 |
| `outlier_mad` | -0.033614 |
| `hampel_filter` | -0.060530 |
| `repair_level_shift` | -0.129560 |
| `smooth_ma` | -0.476227 |

## Reverted-series geometry (descriptive only)

Fields from the frozen M0a census, read verbatim.  No threshold is fitted here and nothing in this section feeds the adoption rule.

Reverted 3 series ({'MIXED': 3}); retained 9 ({'LEVEL_ONLY': 1, 'MIXED': 6, 'AMBIGUOUS': 2}).

| field | reverted mean | retained mean | reverted range | retained range | ranges overlap |
| --- | ---: | ---: | --- | --- | --- |
| `outlier_region_fraction` | 0.130968 | 0.049371 | [0.009766, 0.317057] | [0.000000, 0.299479] | True |
| `level_region_fraction` | 0.025391 | 0.013527 | [0.022461, 0.030599] | [0.000000, 0.031250] | True |
| `outlier_region_end_fraction` | 0.636719 | 0.334346 | [0.455078, 0.939779] | [0.000000, 0.998047] | True |
| `level_region_end_fraction` | 0.519857 | 0.177590 | [0.333984, 0.808919] | [0.000000, 0.783529] | True |
| `union_region_fraction` | 0.145616 | 0.061487 | [0.032227, 0.323242] | [0.000000, 0.314779] | True |
| `union_region_end_fraction` | 0.636719 | 0.406612 | [0.455078, 0.939779] | [0.000000, 0.998047] | True |
| `outlier_point_fraction` | 0.080078 | 0.022063 | [0.002604, 0.213216] | [0.000000, 0.156576] | True |
| `local_robust_z_peak` | 36.928369 | 4.775100 | [4.721435, 99.318764] | [2.644004, 7.368812] | True |
| `level_excursion_score` | 54.903548 | 3.329351 | [3.372454, 154.795629] | [0.000000, 7.183327] | True |

Fields whose observed ranges do not overlap between reverted and retained: none.

