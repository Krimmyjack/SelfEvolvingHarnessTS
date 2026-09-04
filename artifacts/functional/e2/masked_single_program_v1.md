# masked-single-program v1

## Purpose

The companion composition experiment returned `COMPOSITION_NO_HEADROOM` on both cohorts: per-series responses are strongly heterogeneous, but a per-series argmax loses essentially the whole summed gain to a cross-series retraining interaction, because the singleton-scope gain is not the credit a series contributes once the whole batch is treated.  This run tests the low-dimensional variant that does not need an additive credit signal at all: **one program, plus a harm-driven exclusion mask**.  It starts from that program applied to the whole batch, reverts one treated series to identity at a time, and re-measures the aggregate with a real retrain after every single revert.  A revert is kept only if the measured aggregate improved; the first revert that does not improve is rolled back and ends the search.

**This is an engineering effect measurement, not authorization evidence.**  No Skill is written, no Episode is formed, no Fast or Slow path runs, and no execution right is granted or implied.  Data is already-exposed development data only; nothing sealed is reachable.

**Selection happens on the Support window only.**  This was declared before any number was read and is enforced in the code: every accept/reject decision reads the Support aggregate, and the delayed window is evaluated once for each accepted mask purely so it can be reported honestly.  The delayed column never steers the search, and it is the only column here that is out of selection.

The revert queue is ordered by each series' singleton per-series gain under the program, most harmful first.  That ordering is a heuristic that decides what to *try* next; it never decides what is *kept*.  No per-series gain is summed anywhere in this run.

## Cohort `electricity`

**Verdict: `MASKED_SINGLE_PROGRAM_IMPROVES`**

Task `e1v2_task_01`, Support origins [3072, 3120, 3168], delayed origins [3216, 3264, 3312]; 12 training series, 8 evaluation series.  Programs searched: `winsorize`, `outlier_iqr` (top two by full-batch Support aggregate gain).

### Headline

| plan | support | delayed |
| --- | ---: | ---: |
| identity | 0.000000 | 0.000000 |
| best full batch single program (`winsorize`) | +0.011741 | +0.000022 |
| **best masked plan** (`outlier_iqr`, 3 reverted) | +0.034643 | +0.016343 |

Masked minus best full batch: support +0.022903, delayed +0.016321.  Masked minus its own program's full batch: support +0.036831, delayed +0.018207.

Harm account on the evaluation side: identity 0 series; best full batch 4 series / 0.093583 total; best masked plan 1 series / 0.010755 total.  Harm smaller than full batch: `True`.

### Greedy exclusion trace

`winsorize` -- full batch +0.011741, 0 revert(s) accepted, final Support +0.011741, final delayed +0.000022.

| step | reverted series | support after | delta | decision |
| ---: | --- | ---: | ---: | --- |
| 1 | 10 | +0.010800 | -0.000940 | REJECTED_AND_STOPPED |

`outlier_iqr` -- full batch -0.002188, 3 revert(s) accepted, final Support +0.034643, final delayed +0.016343.

| step | reverted series | support after | delta | decision |
| ---: | --- | ---: | ---: | --- |
| 1 | 2 | +0.017891 | +0.020079 | ACCEPTED |
| 2 | 6 | +0.025635 | +0.007743 | ACCEPTED |
| 3 | 1 | +0.034643 | +0.009009 | ACCEPTED |
| 4 | 8 | +0.034643 | +0.000000 | REJECTED_AND_STOPPED |

| accepted mask | reverted so far | support | delayed | harmed eval series | total eval harm |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | (none, full batch) | -0.002188 | -0.001864 | 3 | 0.136380 |
| 1 | 2 | +0.017891 | +0.001344 | 1 | 0.083892 |
| 2 | 2, 6 | +0.025635 | +0.004548 | 1 | 0.048997 |
| 3 | 1, 2, 6 | +0.034643 | +0.016343 | 1 | 0.010755 |

### Reverted-series geometry (descriptive only)

Reverted: `1`, `2`, `6` ({'MIXED': 3}).  Retained: 9 series ({'LEVEL_ONLY': 1, 'MIXED': 6, 'AMBIGUOUS': 2}).  Fields from the frozen M0a census, read verbatim; no threshold is fitted and nothing here contributes to the verdict.

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

Fields whose observed ranges do not overlap between the two groups: none.

Readout equivalence check (masked executor reproduces the frozen `_evaluate` readout exactly on the empty mask): `True`.

## Cohort `T233`

**Verdict: `MASKED_IMPROVES_SUPPORT_ONLY`**

Task `e1v2_task_01`, Support origins [3072, 3120, 3168], delayed origins [3216, 3264, 3312]; 12 training series, 8 evaluation series.  Programs searched: `winsorize`, `outlier_iqr` (top two by full-batch Support aggregate gain).

### Headline

| plan | support | delayed |
| --- | ---: | ---: |
| identity | 0.000000 | 0.000000 |
| best full batch single program (`winsorize`) | +0.072156 | +0.116627 |
| **best masked plan** (`winsorize`, 3 reverted) | +0.139734 | +0.062162 |

Masked minus best full batch: support +0.067579, delayed -0.054465.  Masked minus its own program's full batch: support +0.067579, delayed -0.054465.

Harm account on the evaluation side: identity 0 series; best full batch 2 series / 0.678648 total; best masked plan 1 series / 0.173800 total.  Harm smaller than full batch: `True`.

### Greedy exclusion trace

`winsorize` -- full batch +0.072156, 3 revert(s) accepted, final Support +0.139734, final delayed +0.062162.

| step | reverted series | support after | delta | decision |
| ---: | --- | ---: | ---: | --- |
| 1 | T235 | +0.086841 | +0.014685 | ACCEPTED |
| 2 | T234 | +0.118326 | +0.031485 | ACCEPTED |
| 3 | T244 | +0.139734 | +0.021408 | ACCEPTED |
| 4 | T239 | +0.120446 | -0.019289 | REJECTED_AND_STOPPED |

| accepted mask | reverted so far | support | delayed | harmed eval series | total eval harm |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | (none, full batch) | +0.072156 | +0.116627 | 2 | 0.678648 |
| 1 | T235 | +0.086841 | +0.016079 | 2 | 0.389951 |
| 2 | T234, T235 | +0.118326 | +0.028870 | 2 | 0.353958 |
| 3 | T234, T235, T244 | +0.139734 | +0.062162 | 1 | 0.173800 |

`outlier_iqr` -- full batch +0.058576, 4 revert(s) accepted, final Support +0.132319, final delayed +0.047465.

| step | reverted series | support after | delta | decision |
| ---: | --- | ---: | ---: | --- |
| 1 | T234 | +0.104700 | +0.046123 | ACCEPTED |
| 2 | T247 | +0.117861 | +0.013161 | ACCEPTED |
| 3 | T240 | +0.127084 | +0.009223 | ACCEPTED |
| 4 | T244 | +0.132319 | +0.005235 | ACCEPTED |
| 5 | T239 | +0.127094 | -0.005225 | REJECTED_AND_STOPPED |

| accepted mask | reverted so far | support | delayed | harmed eval series | total eval harm |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | (none, full batch) | +0.058576 | +0.155763 | 3 | 0.176997 |
| 1 | T234 | +0.104700 | +0.061924 | 1 | 0.050584 |
| 2 | T234, T247 | +0.117861 | +0.092874 | 1 | 0.052240 |
| 3 | T234, T240, T247 | +0.127084 | +0.039718 | 1 | 0.044931 |
| 4 | T234, T240, T244, T247 | +0.132319 | +0.047465 | 3 | 0.056223 |

### Reverted-series geometry (descriptive only)

Reverted: `T234`, `T235`, `T244` ({'MIXED': 3}).  Retained: 9 series ({'MIXED': 9}).  Fields from the frozen M0a census, read verbatim; no threshold is fitted and nothing here contributes to the verdict.

| field | reverted mean | retained mean | reverted range | retained range | ranges overlap |
| --- | ---: | ---: | --- | --- | --- |
| `outlier_region_fraction` | 0.068685 | 0.032154 | [0.013346, 0.133138] | [0.001628, 0.077474] | True |
| `level_region_fraction` | 0.014974 | 0.015842 | [0.013021, 0.018555] | [0.013021, 0.029622] | True |
| `outlier_region_end_fraction` | 0.646701 | 0.594799 | [0.186198, 0.985677] | [0.136393, 0.948568] | True |
| `level_region_end_fraction` | 0.300673 | 0.343461 | [0.042318, 0.677083] | [0.041992, 0.678385] | True |
| `union_region_fraction` | 0.075955 | 0.044343 | [0.023438, 0.133138] | [0.014648, 0.078125] | True |
| `union_region_end_fraction` | 0.646701 | 0.634621 | [0.186198, 0.985677] | [0.136393, 0.948568] | True |
| `outlier_point_fraction` | 0.045247 | 0.018410 | [0.005534, 0.097331] | [0.000326, 0.055339] | True |
| `local_robust_z_peak` | 19.126874 | 8.512173 | [7.786648, 28.312926] | [4.037169, 19.307298] | True |
| `level_excursion_score` | 10.046626 | 8.478619 | [6.603289, 14.093720] | [4.645349, 15.625703] | True |

Fields whose observed ranges do not overlap between the two groups: none.

Readout equivalence check (masked executor reproduces the frozen `_evaluate` readout exactly on the empty mask): `True`.

## Reading

The low-dimensional variant does find Support headroom where the per-series argmax composition found none, and the difference is exactly the thing that broke the composition: nothing is added up here, so no step depends on the additive credit the pooled Consumer does not honour.  On both cohorts the best masked plan beats the best full-batch single program on Support, and the evaluation-side harm drops on both -- electricity from 4 harmed series / 0.0936 to 1 / 0.0108, T233 from 2 / 0.6786 to 1 / 0.1738.  Reverting a handful of series is also enough to change which program wins: `outlier_iqr` is *below identity* on electricity at full batch (-0.002188) and becomes the cohort's best plan once three series are reverted (+0.034643), so the full-batch ranking is not the post-mask ranking.

The two cohorts then split on the one column the search never saw.  On electricity the delayed aggregate rises monotonically with every accepted revert (-0.001864, +0.001344, +0.004548, +0.016343), so the Support-only search happened to track the honest window and the verdict is unqualified.  On T233 it does not: the first accepted revert raises Support from +0.072156 to +0.086841 while cutting the delayed gain from +0.116627 to +0.016079, the final mask recovers only part of that, and the second searched program behaves the same way (`outlier_iqr` delayed +0.155763 at full batch, +0.047465 masked).  On that cohort the full batch is close to the delayed ceiling and the Support-only greedy step is buying Support at the delayed window's expense, which is why the verdict is `MASKED_IMPROVES_SUPPORT_ONLY` and not an improvement claim.

So the mechanism is real and cheap, and Support-gain-driven exclusion is not by itself safe to read as a downstream improvement.  What separates the two cohorts is not visible in the Support column, which is the part a deployable version of this would have to solve.

## Verdict summary

| cohort | verdict | best masked plan | support | delayed |
| --- | --- | --- | ---: | ---: |
| electricity | `MASKED_SINGLE_PROGRAM_IMPROVES` | `outlier_iqr` minus 3 series | +0.034643 | +0.016343 |
| T233 | `MASKED_IMPROVES_SUPPORT_ONLY` | `winsorize` minus 3 series | +0.139734 | +0.062162 |

