# consumer-conditioned recipe v1

**Engineering effect measurement, not authorization evidence.**  The Consumer variant lives only inside this experiment runner; the frozen pooled Consumer used by v6 `_evaluate` is unchanged.  No Skill is written, no Episode is formed, no Fast or Slow path is entered, and no execution right is granted or implied.  0 LLM calls.  Already-exposed development data only.

Question: the pooled ridge Consumer ate the additive per-series composition.  Does a per-channel ridge -- same features, no cross-channel pooling -- unlock that headroom, and does the adopted batch recipe change with the Consumer?

> **Caveat.** The delayed window participates in recipe adoption, so both columns of a recipe are in-selection.  Pooled composition numbers for electricity and T233 are loaded from the already-accepted headroom artifact when present; traffic pooled composition is run here if that artifact has no traffic row.  per_channel numbers are produced in this session.

| cohort | judgment | pooled interaction | per_channel interaction | composition flip | recipe changed |
| --- | --- | ---: | ---: | --- | --- |
| `electricity` | `CONSUMER_STRUCTURE_CHANGES_RECIPE` | -0.230445 | -0.052500 | True | True |
| `T233` | `CONSUMER_STRUCTURE_CHANGES_RECIPE` | -0.253027 | -0.017264 | True | True |
| `traffic` | `CONSUMER_STRUCTURE_CHANGES_RECIPE` | -0.366011 | -0.067018 | False | True |

## Cohort `electricity`

**Judgment: `CONSUMER_STRUCTURE_CHANGES_RECIPE`**

Windows: Support [3072, 3120, 3168], delayed [3216, 3264, 3312].

### Interaction (validated composition - additive expectation)

| Consumer | additive expected | validated composition | interaction |
| --- | ---: | ---: | ---: |
| `pooled` | +0.211377 | -0.019069 | -0.230445 |
| `per_channel` | +0.180449 | +0.127949 | -0.052500 |

### Composition vs best single program (Support)

| Consumer | verdict | best single | composition | composition - best |
| --- | --- | ---: | ---: | ---: |
| `pooled` | `COMPOSITION_NO_HEADROOM` | +0.011741 (winsorize) | -0.019069 | -0.030809 |
| `per_channel` | `SELECTIVE_COMPOSITION_HEADROOM_PRESENT` | +0.114777 (denoise_median) | +0.127949 | +0.013172 |

Composition flip under per_channel: yes.

### Adopted recipe

| Consumer | kind | program | reverted | support | delayed |
| --- | --- | --- | --- | ---: | ---: |
| `pooled` | `MASKED_PLAN` | `outlier_iqr` | `1`, `2`, `6` | +0.034643 | +0.016343 |
| `per_channel` | `MASKED_PLAN` | `denoise_median` | `10`, `3`, `4` | +0.129359 | +0.138523 |

### per_channel menu scan (full batch, Support aggregate gain)

| program | support aggregate gain |
| --- | ---: |
| `denoise_median` (searched) | +0.114777 |
| `smooth_ma` (searched) | +0.039755 |
| `outlier_iqr` | +0.036924 |
| `winsorize` | +0.036621 |
| `outlier_mad` | +0.002677 |
| `repair_level_shift` | +0.002362 |
| `hampel_filter` | -0.128489 |

## Cohort `T233`

**Judgment: `CONSUMER_STRUCTURE_CHANGES_RECIPE`**

Windows: Support [3072, 3120, 3168], delayed [3216, 3264, 3312].

### Interaction (validated composition - additive expectation)

| Consumer | additive expected | validated composition | interaction |
| --- | ---: | ---: | ---: |
| `pooled` | +0.259402 | +0.006375 | -0.253027 |
| `per_channel` | +0.077244 | +0.059980 | -0.017264 |

### Composition vs best single program (Support)

| Consumer | verdict | best single | composition | composition - best |
| --- | --- | ---: | ---: | ---: |
| `pooled` | `COMPOSITION_NO_HEADROOM` | +0.072156 (winsorize) | +0.006375 | -0.065781 |
| `per_channel` | `SELECTIVE_COMPOSITION_HEADROOM_PRESENT` | +0.016910 (smooth_ma) | +0.059980 | +0.043070 |

Composition flip under per_channel: yes.

### Adopted recipe

| Consumer | kind | program | reverted | support | delayed |
| --- | --- | --- | --- | ---: | ---: |
| `pooled` | `BEST_FULL_BATCH` | `winsorize` | none | +0.072156 | +0.116627 |
| `per_channel` | `MASKED_PLAN` | `smooth_ma` | `T233`, `T236`, `T246`, `T247`, `T254`, `T256` | +0.032771 | -0.076148 |

### per_channel menu scan (full batch, Support aggregate gain)

| program | support aggregate gain |
| --- | ---: |
| `smooth_ma` (searched) | +0.016910 |
| `winsorize` (searched) | +0.008666 |
| `outlier_iqr` | +0.004915 |
| `repair_level_shift` | +0.004500 |
| `denoise_median` | +0.004342 |
| `outlier_mad` | -0.000166 |
| `hampel_filter` | -0.010929 |

## Cohort `traffic`

**Judgment: `CONSUMER_STRUCTURE_CHANGES_RECIPE`**

Windows: Support [1104, 1368], delayed [1800].

### Interaction (validated composition - additive expectation)

| Consumer | additive expected | validated composition | interaction |
| --- | ---: | ---: | ---: |
| `pooled` | +1.024166 | +0.658155 | -0.366011 |
| `per_channel` | +0.449845 | +0.382827 | -0.067018 |

### Composition vs best single program (Support)

| Consumer | verdict | best single | composition | composition - best |
| --- | --- | ---: | ---: | ---: |
| `pooled` | `SELECTIVE_COMPOSITION_HEADROOM_PRESENT` | +0.607441 (outlier_iqr) | +0.658155 | +0.050714 |
| `per_channel` | `SELECTIVE_COMPOSITION_HEADROOM_PRESENT` | +0.299313 (outlier_iqr) | +0.382827 | +0.083514 |

Composition flip under per_channel: no.

### Adopted recipe

| Consumer | kind | program | reverted | support | delayed |
| --- | --- | --- | --- | ---: | ---: |
| `pooled` | `MASKED_PLAN` | `outlier_iqr` | `6` | +0.665277 | +1.047075 |
| `per_channel` | `MASKED_PLAN` | `outlier_iqr` | `3`, `8` | +0.303184 | +0.355472 |

### per_channel menu scan (full batch, Support aggregate gain)

| program | support aggregate gain |
| --- | ---: |
| `outlier_iqr` (searched) | +0.299313 |
| `winsorize` (searched) | +0.265470 |
| `outlier_mad` | +0.254072 |
| `smooth_ma` | +0.205232 |
| `denoise_median` | +0.158455 |
| `hampel_filter` | +0.103591 |
| `repair_level_shift` | +0.008036 |

