# batch-composition-headroom v1

## Purpose

The project's primary readout is now a batch question: for one batch of data, is the downstream effect after Harness processing better than (a) doing nothing (identity) and (b) the best single program applied to the whole batch?  This run measures the *headroom* of selective per-series treatment.  If one program's per-series response points in different directions on different series, then choosing the best program per series -- identity included -- should beat any uniform full-batch treatment on the aggregate readout.

**This is an engineering effect measurement, not authorization evidence.**  No Skill is written, no Episode is formed, no Fast or Slow path runs, and no execution right is granted or implied.  The per-series argmax is fitted on the same Support outcomes it is then scored on, so the Support column is an upper bound on selective headroom rather than a deployable policy; the delayed column is the out-of-selection readout.  Data is already-exposed development data only.

Protocol reuse: cohorts from `agentic.runner.load_cohort`, Consumer and Judge from `_evaluate_origins` / v6 `_evaluate` (ridge, sMASE, macro gain over identity), windows from the frozen Task roster, compile path from `task_episode_harness.runner._compiled`.

## Cohort `electricity`

**Verdict: `COMPOSITION_NO_HEADROOM`**

Task `e1v2_task_01`, Support origins [3072, 3120, 3168], delayed origins [3216, 3264, 3312]; 12 training series, 8 evaluation series.

### Aggregate gain over identity (higher is better)

| plan | support | delayed |
| --- | ---: | ---: |
| identity (baseline) | 0.000000 | 0.000000 |
| full batch: denoise_median | -0.023076 | not evaluated |
| full batch: hampel_filter | -0.060530 | not evaluated |
| full batch: outlier_iqr | -0.002188 | not evaluated |
| full batch: outlier_mad | -0.033614 | not evaluated |
| full batch: repair_level_shift | -0.129560 | not evaluated |
| full batch: smooth_ma | -0.476227 | not evaluated |
| full batch: winsorize **(best single)** | +0.011741 | +0.000022 |
| **composition (per-series argmax)** | -0.019069 | -0.213512 |

Composition minus best single program: support -0.030809, delayed -0.213534. Ordering preserved on the delayed window: `False`.

### Per-series assignment

| training series | chosen program | its per-series gain | best-single-program gain on this series |
| --- | --- | ---: | ---: |
| 0 | denoise_median | +0.042699 | +0.016249 |
| 1 | winsorize | +0.010299 | +0.010299 |
| 2 | denoise_median | +0.047075 | +0.011098 |
| 3 | denoise_median | +0.030274 | +0.002556 |
| 4 | outlier_mad | +0.002572 | +0.000983 |
| 5 | hampel_filter | +0.007445 | -0.003089 |
| 6 | repair_level_shift | +0.010912 | -0.007845 |
| 7 | denoise_median | +0.006049 | -0.003077 |
| 8 | smooth_ma | +0.016622 | +0.002659 |
| 9 | denoise_median | +0.023013 | +0.021290 |
| 10 | outlier_iqr | +0.006635 | -0.008598 |
| 11 | winsorize | +0.007780 | +0.007780 |

A per-series gain is the aggregate batch gain when that one series is treated and everything else is left alone -- the same singleton-scope readout the repository already uses.  Identity is always available at exactly 0.

### Harm account

| plan | aggregate support gain | harmed eval series | total eval harm | harmed training series | total training harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | +0.000000 | 0 | 0.000000 | 0 | 0.000000 |
| denoise_median | -0.023076 | 5 | 0.630183 | 6 | 0.274700 |
| hampel_filter | -0.060530 | 6 | 0.662323 | 8 | 0.136501 |
| outlier_iqr | -0.002188 | 3 | 0.136380 | 1 | 0.016112 |
| outlier_mad | -0.033614 | 6 | 0.388801 | 2 | 0.013790 |
| repair_level_shift | -0.129560 | 5 | 1.170078 | 4 | 0.228909 |
| smooth_ma | -0.476227 | 7 | 3.815595 | 10 | 0.454797 |
| winsorize | +0.011741 | 4 | 0.093583 | 2 | 0.016443 |
| COMPOSITION | -0.019069 | 4 | 0.469559 | 0 | 0.000000 |

Two harm columns, because they answer different questions.  The eval-side count is how many downstream evaluation series ended up worse than identity under that plan; it is the one that can falsify the composition.  The training-side count is how many treated training series carry a negative singleton response; the composition drives it to zero **by construction** (identity is in the argmax), so it is a consistency check, not a finding.

### Composition interaction (reported, not assumed)

Sum of the chosen per-series gains: +0.211377.  Validated composition gain from a single retrain under the assignment: -0.019069.  Cross-series retraining interaction: -0.230445.

### Response divergence

Strongest divergence: `denoise_median` gains +0.047075 on `2` and loses -0.069529 on `10` -- a spread of 0.116604 for the same program on the same batch.  5 series positive, 6 negative.

| program | positive series | negative series | best | worst | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| denoise_median | 5 | 6 | 2 +0.047075 | 10 -0.069529 | 0.116604 |
| repair_level_shift | 4 | 4 | 0 +0.015609 | 11 -0.097151 | 0.112759 |
| smooth_ma | 2 | 10 | 8 +0.016622 | 4 -0.089898 | 0.106520 |
| hampel_filter | 3 | 8 | 0 +0.029388 | 10 -0.027176 | 0.056563 |
| outlier_iqr | 2 | 1 | 9 +0.019650 | 2 -0.016112 | 0.035761 |
| winsorize | 5 | 2 | 9 +0.021290 | 10 -0.008598 | 0.029888 |
| outlier_mad | 1 | 2 | 9 +0.011573 | 6 -0.007756 | 0.019328 |

Readout equivalence check (per-series-assignment retrain reproduces the frozen `_evaluate` readout exactly on every uniform assignment): `True`.

## Cohort `T233`

**Verdict: `COMPOSITION_NO_HEADROOM`**

Task `e1v2_task_01`, Support origins [3072, 3120, 3168], delayed origins [3216, 3264, 3312]; 12 training series, 8 evaluation series.

### Aggregate gain over identity (higher is better)

| plan | support | delayed |
| --- | ---: | ---: |
| identity (baseline) | 0.000000 | 0.000000 |
| full batch: denoise_median | -0.528073 | not evaluated |
| full batch: hampel_filter | -0.165671 | not evaluated |
| full batch: outlier_iqr | +0.058576 | not evaluated |
| full batch: outlier_mad | +0.023298 | not evaluated |
| full batch: repair_level_shift | -0.063758 | not evaluated |
| full batch: smooth_ma | -0.663197 | not evaluated |
| full batch: winsorize **(best single)** | +0.072156 | +0.116627 |
| **composition (per-series argmax)** | +0.006375 | -0.376956 |

Composition minus best single program: support -0.065781, delayed -0.493583. Ordering preserved on the delayed window: `False`.

### Per-series assignment

| training series | chosen program | its per-series gain | best-single-program gain on this series |
| --- | --- | ---: | ---: |
| T233 | repair_level_shift | +0.014088 | +0.004615 |
| T234 | identity | +0.000000 | -0.015622 |
| T235 | denoise_median | +0.023067 | -0.019930 |
| T236 | outlier_iqr | +0.012717 | +0.007496 |
| T239 | repair_level_shift | +0.018313 | -0.008589 |
| T240 | hampel_filter | +0.010516 | +0.000033 |
| T241 | denoise_median | +0.026612 | +0.013488 |
| T244 | hampel_filter | +0.001182 | -0.010730 |
| T246 | winsorize | +0.035165 | +0.035165 |
| T247 | hampel_filter | +0.006028 | -0.001343 |
| T254 | outlier_iqr | +0.077396 | +0.067924 |
| T256 | smooth_ma | +0.034317 | +0.028733 |

A per-series gain is the aggregate batch gain when that one series is treated and everything else is left alone -- the same singleton-scope readout the repository already uses.  Identity is always available at exactly 0.

### Harm account

| plan | aggregate support gain | harmed eval series | total eval harm | harmed training series | total training harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | +0.000000 | 0 | 0.000000 | 0 | 0.000000 |
| denoise_median | -0.528073 | 6 | 4.315661 | 10 | 1.178658 |
| hampel_filter | -0.165671 | 5 | 1.474818 | 6 | 0.323019 |
| outlier_iqr | +0.058576 | 3 | 0.176997 | 4 | 0.080476 |
| outlier_mad | +0.023298 | 3 | 0.304388 | 4 | 0.069435 |
| repair_level_shift | -0.063758 | 4 | 0.836482 | 5 | 0.244757 |
| smooth_ma | -0.663197 | 8 | 5.305575 | 11 | 1.587449 |
| winsorize | +0.072156 | 2 | 0.678648 | 4 | 0.054871 |
| COMPOSITION | +0.006375 | 3 | 1.009216 | 0 | 0.000000 |

Two harm columns, because they answer different questions.  The eval-side count is how many downstream evaluation series ended up worse than identity under that plan; it is the one that can falsify the composition.  The training-side count is how many treated training series carry a negative singleton response; the composition drives it to zero **by construction** (identity is in the argmax), so it is a consistency check, not a finding.

### Composition interaction (reported, not assumed)

Sum of the chosen per-series gains: +0.259402.  Validated composition gain from a single retrain under the assignment: +0.006375.  Cross-series retraining interaction: -0.253027.

### Response divergence

Strongest divergence: `denoise_median` gains +0.026612 on `T241` and loses -0.389852 on `T254` -- a spread of 0.416463 for the same program on the same batch.  2 series positive, 10 negative.

| program | positive series | negative series | best | worst | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| denoise_median | 2 | 10 | T241 +0.026612 | T254 -0.389852 | 0.416463 |
| smooth_ma | 1 | 11 | T256 +0.034317 | T254 -0.304896 | 0.339213 |
| hampel_filter | 5 | 6 | T256 +0.027457 | T254 -0.117752 | 0.145210 |
| repair_level_shift | 4 | 5 | T239 +0.018313 | T254 -0.086886 | 0.105198 |
| outlier_iqr | 6 | 4 | T254 +0.077396 | T234 -0.023530 | 0.100926 |
| winsorize | 5 | 4 | T254 +0.067924 | T235 -0.019930 | 0.087854 |
| outlier_mad | 4 | 4 | T254 +0.053711 | T247 -0.030835 | 0.084546 |

Readout equivalence check (per-series-assignment retrain reproduces the frozen `_evaluate` readout exactly on every uniform assignment): `True`.

## Reading

Per-series responses are strongly heterogeneous -- every treatment in the menu has both materially helped and materially hurt series inside the same batch -- so `RESPONSES_HOMOGENEOUS` is ruled out on both cohorts.  The heterogeneity is real; the naive way of cashing it in is not.  Choosing each series' own argmax and applying all of those choices at once loses roughly the entire summed per-series gain to a cross-series retraining interaction, and lands below the best single full-batch program on Support and below identity on the delayed window.

The mechanism is visible in the numbers rather than inferred: the Consumer is one ridge model pooled over the whole treated batch, so a singleton-scope gain measures the marginal effect of treating one series *while the other eleven stay untreated*, and that quantity is not the credit each series contributes once they are all treated together.  Selective composition on this instrument therefore needs a credit signal that is defined jointly, or a composition rule that is validated rather than assembled.  Nothing here says selective treatment cannot pay; it says the additive singleton proxy is not the way to find out.

## Verdict summary

| cohort | verdict | identity | best single | composition |
| --- | --- | ---: | ---: | ---: |
| electricity | `COMPOSITION_NO_HEADROOM` | 0.000000 | +0.011741 (winsorize) | -0.019069 |
| T233 | `COMPOSITION_NO_HEADROOM` | 0.000000 | +0.072156 (winsorize) | +0.006375 |

