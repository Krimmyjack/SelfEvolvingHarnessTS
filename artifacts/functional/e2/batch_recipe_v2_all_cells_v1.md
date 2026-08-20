# batch recipe v2 -- all cells

**Engineering effect measurement, not authorization evidence.**  adoption_rule_version v2.  Consumer variants live only inside this experiment runner.  0 LLM.  v1 recipe artifacts were not overwritten.

v2 delayed gate: a masked plan is adopted only if its delayed aggregate gain >= max(best full-batch delayed, 0).  Identity is an incumbent on the delayed window.  v1 failure case: T233@per_channel in consumer_conditioned_recipe_v1.json.

| cell | kind | program | reverted | support | delayed | vs v1 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `electricity` `pooled` | `MASKED_PLAN` | `outlier_iqr` | `1`, `2`, `6` | +0.034643 | +0.016343 | UNCHANGED |
| `electricity` `per_channel` | `MASKED_PLAN` | `denoise_median` | `10`, `3`, `4` | +0.129359 | +0.138523 | UNCHANGED |
| `T233` `pooled` | `BEST_FULL_BATCH` | `winsorize` | none | +0.072156 | +0.116627 | UNCHANGED |
| `T233` `per_channel` | `MASKED_PLAN` | `winsorize` | `T233`, `T234`, `T241`, `T247`, `T256` | +0.019022 | +0.030410 | EXPECTED_V2_CORRECTION |
| `traffic` `pooled` | `MASKED_PLAN` | `outlier_iqr` | `6` | +0.665277 | +1.047075 | UNCHANGED |
| `traffic` `per_channel` | `MASKED_PLAN` | `outlier_iqr` | `3`, `8` | +0.303184 | +0.355472 | UNCHANGED |

## `electricity` `pooled`

**vs v1: UNCHANGED**

Windows: Support [3072, 3120, 3168], delayed [3216, 3264, 3312].

Adopted: `MASKED_PLAN` `outlier_iqr` reverted `1`, `2`, `6`.

Gain: support +0.034643, delayed +0.016343.

Identity absolute loss (sMASE, recorded, not a rule): support 1.225149, delayed 1.070756.

Full-batch delayed champion `hampel_filter` DIFFERS from support champion `winsorize`.

| program | support | delayed |
| --- | ---: | ---: |
| `hampel_filter` | -0.060530 | +0.049344 |
| `winsorize` | +0.011741 | +0.000022 |
| `outlier_iqr` | -0.002188 | -0.001864 |
| `outlier_mad` | -0.033614 | -0.045287 |
| `repair_level_shift` | -0.129560 | -0.051877 |
| `denoise_median` | -0.023076 | -0.211834 |
| `smooth_ma` | -0.476227 | -0.782224 |

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `outlier_iqr` | 1, 2, 6 | +0.034643 | +0.016343 | +0.000022 | PASS |

## `electricity` `per_channel`

**vs v1: UNCHANGED**

Windows: Support [3072, 3120, 3168], delayed [3216, 3264, 3312].

Adopted: `MASKED_PLAN` `denoise_median` reverted `10`, `3`, `4`.

Gain: support +0.129359, delayed +0.138523.

Identity absolute loss (sMASE, recorded, not a rule): support 1.198469, delayed 1.080227.

Full-batch delayed champion `denoise_median` equals support champion `denoise_median`.

| program | support | delayed |
| --- | ---: | ---: |
| `denoise_median` | +0.114777 | +0.118209 |
| `outlier_iqr` | +0.036924 | +0.042210 |
| `winsorize` | +0.036621 | +0.040160 |
| `smooth_ma` | +0.039755 | +0.024076 |
| `repair_level_shift` | +0.002362 | +0.016993 |
| `outlier_mad` | +0.002677 | +0.012966 |
| `hampel_filter` | -0.128489 | -0.125955 |

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `denoise_median` | 10, 3, 4 | +0.129359 | +0.138523 | +0.118209 | PASS |
| `smooth_ma` | 0, 1, 10, 3 | +0.085006 | +0.072251 | -- | NOT_REACHED |

## `T233` `pooled`

**vs v1: UNCHANGED**

Windows: Support [3072, 3120, 3168], delayed [3216, 3264, 3312].

Adopted: `BEST_FULL_BATCH` `winsorize` reverted none.

Gain: support +0.072156, delayed +0.116627.

Identity absolute loss (sMASE, recorded, not a rule): support 1.591078, delayed 1.458117.

Full-batch delayed champion `outlier_mad` DIFFERS from support champion `winsorize`.

| program | support | delayed |
| --- | ---: | ---: |
| `outlier_mad` | +0.023298 | +0.163539 |
| `outlier_iqr` | +0.058576 | +0.155763 |
| `winsorize` | +0.072156 | +0.116627 |
| `hampel_filter` | -0.165671 | -0.058383 |
| `repair_level_shift` | -0.063758 | -0.331884 |
| `denoise_median` | -0.528073 | -0.553491 |
| `smooth_ma` | -0.663197 | -0.762802 |

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `winsorize` | T234, T235, T244 | +0.139734 | +0.062162 | +0.116627 | FAIL |
| `outlier_iqr` | T234, T240, T244, T247 | +0.132319 | +0.047465 | +0.116627 | FAIL |

## `T233` `per_channel`

**vs v1: EXPECTED_V2_CORRECTION**

Windows: Support [3072, 3120, 3168], delayed [3216, 3264, 3312].

Adopted: `MASKED_PLAN` `winsorize` reverted `T233`, `T234`, `T241`, `T247`, `T256`.

Gain: support +0.019022, delayed +0.030410.

Identity absolute loss (sMASE, recorded, not a rule): support 1.000198, delayed 0.738377.

Full-batch delayed champion `outlier_iqr` DIFFERS from support champion `smooth_ma`.

| program | support | delayed |
| --- | ---: | ---: |
| `outlier_iqr` | +0.004915 | +0.047948 |
| `outlier_mad` | -0.000166 | +0.043658 |
| `winsorize` | +0.008666 | +0.038466 |
| `repair_level_shift` | +0.004500 | -0.011542 |
| `hampel_filter` | -0.010929 | -0.048090 |
| `denoise_median` | +0.004342 | -0.103864 |
| `smooth_ma` | +0.016910 | -0.146275 |

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `smooth_ma` | T233, T236, T246, T247, T254, T256 | +0.032771 | -0.076148 | +0.000000 | FAIL |
| `winsorize` | T233, T234, T241, T247, T256 | +0.019022 | +0.030410 | +0.000000 | PASS |

## `traffic` `pooled`

**vs v1: UNCHANGED**

Windows: Support [1104, 1368], delayed [1800].

Adopted: `MASKED_PLAN` `outlier_iqr` reverted `6`.

Gain: support +0.665277, delayed +1.047075.

Identity absolute loss (sMASE, recorded, not a rule): support 2.017431, delayed 2.299831.

Full-batch delayed champion `outlier_iqr` equals support champion `outlier_iqr`.

| program | support | delayed |
| --- | ---: | ---: |
| `outlier_iqr` | +0.607441 | +1.019774 |
| `winsorize` | +0.451495 | +0.959141 |
| `outlier_mad` | +0.550279 | +0.958206 |
| `hampel_filter` | -0.075704 | +0.208603 |
| `repair_level_shift` | +0.060883 | +0.068161 |
| `smooth_ma` | -0.259221 | +0.030984 |
| `denoise_median` | -0.635374 | -0.037371 |

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `outlier_iqr` | 6 | +0.665277 | +1.047075 | +1.019774 | PASS |
| `outlier_mad` | 5, 6, 7 | +0.663904 | +0.965374 | -- | NOT_REACHED |

## `traffic` `per_channel`

**vs v1: UNCHANGED**

Windows: Support [1104, 1368], delayed [1800].

Adopted: `MASKED_PLAN` `outlier_iqr` reverted `3`, `8`.

Gain: support +0.303184, delayed +0.355472.

Identity absolute loss (sMASE, recorded, not a rule): support 1.612334, delayed 1.592408.

Full-batch delayed champion `winsorize` DIFFERS from support champion `outlier_iqr`.

| program | support | delayed |
| --- | ---: | ---: |
| `winsorize` | +0.265470 | +0.371586 |
| `outlier_iqr` | +0.299313 | +0.348216 |
| `outlier_mad` | +0.254072 | +0.319627 |
| `smooth_ma` | +0.205232 | +0.182934 |
| `denoise_median` | +0.158455 | +0.168635 |
| `hampel_filter` | +0.103591 | +0.130606 |
| `repair_level_shift` | +0.008036 | +0.025347 |

| candidate | reverted | support | delayed | bar | check |
| --- | --- | ---: | ---: | ---: | --- |
| `outlier_iqr` | 3, 8 | +0.303184 | +0.355472 | +0.348216 | PASS |

