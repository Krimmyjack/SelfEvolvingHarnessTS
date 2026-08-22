# T1 flip control: one Program, two Consumers, opposite signs?

**TASK_FLIP_CONFIRMED_POSITIVE_CONTROL** (evidence grade: **POSITIVE_CONTROL**, permanent).  0 LLM, 30/60 forecasting retrains, 72/300 AD evaluations.

## Part A -- the injection

- Zone **`[431, 900)`**, seed **20260823**, cycle counter from slot 0 (independent of T0).  main-line ruling on T0's blocking finding: the injection lives in the training block, at P's measured action region [120, 900) minus the T0 calibration block [143, 431); the remainder [120, 143) is 23 long, shorter than the two boundary exclusions, and is discarded.

| series | events | skips | spacing rejections | indices |
| --- | ---: | ---: | ---: | --- |
| `72203812897` | 4 | 0 | 1 | 476, 611, 667, 770 |
| `72259003927` | 4 | 0 | 2 | 599, 661, 784, 836 |
| `72329003935` | 4 | 0 | 2 | 499, 572, 726, 827 |
| `72422093820` | 4 | 0 | 5 | 533, 599, 657, 791 |
| `72435653866` | 4 | 0 | 1 | 492, 697, 805, 860 |
| `72438093819` | 4 | 0 | 3 | 457, 511, 589, 691 |
| `72511654737` | 4 | 0 | 0 | 558, 633, 720, 866 |
| `72529014768` | 4 | 0 | 1 | 554, 620, 677, 727 |
| `72605654791` | 4 | 0 | 3 | 565, 700, 798, 858 |
| `72654014936` | 4 | 0 | 5 | 481, 573, 776, 849 |
| `72793494248` | 4 | 0 | 5 | 540, 606, 673, 802 |
| `74486514719` | 4 | 0 | 6 | 487, 540, 626, 763 |

Total **48** events, 0 skips.  Composition: `{'burst/+1/6s/x3': 8, 'spike/+1/10s/x1': 8, 'spike/+1/6s/x1': 16, 'spike/-1/10s/x1': 8, 'spike/-1/6s/x1': 8}`.  Ledger frozen to `_scratch/phase_t/injected/t1` before any Consumer reading.

Copy integrity (A3): every copy differs from its original exactly at the ledger indices -- **True**; originals re-read after the run and unchanged -- **True**.

## Part B -- the arms

B3 same-byte assertion: **600** comparisons, all equal: **True**; P(B) reproducible on recall: **True**.

| program | forecasting support agg | forecasting delayed agg | delayed per-eval-series (min) | AD pooled F1 | AD aggregate gain | AD per-series (min) |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `identity` | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 0.0000 | 0.0000 |
| `outlier_iqr` | 0.1780 | 0.2723 | 0.0586 | 0.5714 | -0.1046 | -0.3000 |
| `outlier_mad` | 0.2198 | 0.3255 | 0.1674 | 0.6260 | -0.0455 | -0.1636 |
| `hampel_filter` | -0.0890 | 0.0648 | -0.0904 | 0.4138 | -0.2808 | -0.7273 |
| `winsorize` | 0.7122 | 0.4059 | 0.3320 | 0.4144 | -0.2681 | -0.7273 |

Identity is the baseline on the injected block (B2); its gains are zero by construction.

## B4 readings (not part of the verdict)

- Twin-block background alarm rate per 1000 points: min 4.2644 / median 7.4627 / max 14.9254.
- Guard read-through (`min_per_series_gain` on the AD vector, in-service grammar, no code change):

| program | min per-series AD gain | fires at -0.005 | crossing series |
| --- | ---: | --- | --- |
| `outlier_iqr` | -0.3000 | True | 72203812897, 72329003935, 72422093820, 72435653866, 72511654737, 72605654791 |
| `outlier_mad` | -0.1636 | True | 72203812897, 72329003935, 72422093820, 72435653866 |
| `hampel_filter` | -0.7273 | True | 72203812897, 72329003935, 72422093820, 72435653866, 72438093819, 72511654737, 72529014768, 72654014936, 72793494248, 74486514719 |
| `winsorize` | -0.7273 | True | 72203812897, 72259003927, 72329003935, 72422093820, 72435653866, 72438093819, 72511654737, 72529014768, 72605654791, 72654014936, 74486514719 |

## Part C -- the verdict

Identity-arm AD reading (C3): pooled P 0.5172 / R 0.9375 / F1 0.6667; degenerate: **False**.

| program | forecasting delayed agg | AD agg | flip |
| --- | ---: | ---: | --- |
| `outlier_iqr` | 0.2723 | -0.1046 | forecasting_up_ad_down |
| `outlier_mad` | 0.3255 | -0.0455 | forecasting_up_ad_down |
| `hampel_filter` | 0.0648 | -0.2808 | forecasting_up_ad_down |
| `winsorize` | 0.4059 | -0.2681 | forecasting_up_ad_down |

on the AD side the pre-registered material line of +/-0.005 means 'at least one event changed hands': with 48 ledger events a single event is about 0.017 of aggregate displacement, so no 0.005-level resolution is claimed.  The flip judgement is made at the aggregate layer only.

aggregate only; per-series comparison happens inside each task (forecasting: the 4 eval series; AD: the 12 train series), never paired across tasks (C4)

## Ambiguities (reported, not self-adjudicated)

- P(B) is applied once to the contiguous block [120, 900), not per anchor window as the in-service path does.  For winsorize/outlier_iqr/outlier_mad the statistics pool 780 points instead of 240; hampel_filter is rolling-local and differs only at window edges.  The alternative (per-window application with per-window AD detection) makes B3's element-wise assertion ill-shaped and costs ~648 AD evaluations against the budget of 300.  The flip estimand under the shared geometry is intact; magnitude comparability with the in-service per-window menu gains is not claimed.
- The detector warm-up consumes [382, 431): no ledger event can land there (the legal range starts at 456), so scoring is unaffected, but the block's first 49 points are structurally unscored.
- Forecasting retrains are counted per (arm, origin) by the line's convention (30); the six origins share one design matrix per arm, so the fits are numerically identical refits and the physically distinct fits are five.
- The twin-block background alarm level is a background level, never a false-positive rate (T0 C5(ii) caliber).
- task_B's triple window at 1800 is not scored: the book scopes task_A.  A flip read on task_B would be a new slice.

## Cost

- LLM calls 0.  Forecasting retrains 30 of 60.  AD evaluations 72 of 300.
- Frozen surface FROZEN_SURFACE_V9: 39 unique files, drift after run: none.
- Sealed: NOAA 2025, everything beyond 17520, SMD official test and labels.
- Wall seconds 1.8.
