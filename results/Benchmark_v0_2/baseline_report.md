# Benchmark benchmark-v0.2 Dev-Query baseline report

Final-Query was not read. Every value below is repeatable Dev-Query sMASE.

One closed-form model is trained per `(program, scenario, dose, replicate, dataset)`. The two `*_retrained` oracles pick a program per `(cell, scenario, dose)` and are then **trained on the corpus those picks produce** -- the same path a Method takes. The `*_untrained_counterfactual` rows are the v0/v0.1 oracle, kept for continuity and excluded from the Gate: they read each cell's loss off a model trained on a single-program corpus, so they describe a world no model was fitted to.

## The three-point C1 comparison

| role | baseline | overall (macro) | series-micro (descriptive) |
| --- | --- | --- | --- |
| floor (no-op) | `raw` | 11.481494 | 15.949263 |
| floor (best single program) | `best_fixed` | 10.978415 | 15.226536 |
| incumbent | `h_ref` | 11.558342 | 16.065421 |
| CEILING -- gate | `oracle_transfer_retrained` | 10.788125 | 15.040908 |
| envelope (winner's curse) | `oracle_insample_retrained` | 10.541011 | 14.835802 |
| descriptive only | `oracle_transfer_untrained_counterfactual` | 10.986093 | 15.196234 |
| descriptive only | `oracle_insample_untrained_counterfactual` | 10.831308 | 15.113816 |

## Every program in the frozen pool

| program | mechanism | overall (macro) |
| --- | --- | --- |
| `raw` | none | 11.481494 |
| `forward_fill` | missing | 11.591508 |
| `seasonal_fill` | missing | 11.696687 |
| `winsorize` | outlier_spike | 11.884032 |
| `denoise_median` | outlier_spike | 11.171052 |
| `denoise_stl` | additive_noise | 10.978415 |
| `denoise_savgol` | additive_noise | 11.328558 |
| `denoise_wavelet` | additive_noise | 11.552050 |
| `h_ref` | reference | 11.558342 |

## Where the pool has no action at all

The pool cannot act on 0 dataset/scenario/dose cell(s) -- every program scores identically there. That is a capability gap in the operator library, not evidence the data has nothing to gain.


Best-fixed program selected on Support-A: `denoise_stl`.

## What H_ref actually does

- Indistinguishable from Raw on 322/373 Dev series (86.3%).
- Better than Raw on 32; worse on 19.
- Net vs Raw (series-micro): +0.116157 (positive = worse than doing nothing).

## Headroom, measured from both floors

The oracle reverts to Raw in 0 cell(s): `[]`. In those cells the apparent gain over H_ref is H_ref's own damage refunded, not repair space.

| cell | oracle pick | gain over Raw | gain over H_ref | H_ref self-harm |
| --- | --- | --- | --- | --- |
| `gefcom2012_load|seasonal_high` | denoise_savgol, forward_fill, seasonal_fill, winsorize | +0.1164 | +0.1164 | +0.0000 |
| `gefcom2012_load|structured_mixed` | denoise_stl | +0.0594 | +0.0594 | +0.0000 |
| `metr_la|low_structure` | denoise_stl | +0.3138 | +0.3138 | +0.0000 |
| `metr_la|structured_mixed` | denoise_savgol | +0.3701 | +0.3701 | +0.0000 |
| `metr_la|trend_high` | denoise_stl | +0.3605 | +0.3605 | +0.0000 |
| `monash:covid_deaths|seasonal_high` | denoise_median, denoise_savgol, denoise_stl, forward_fill | +3.6172 | +3.9533 | +0.3360 |
| `monash:covid_deaths|trend_high` | denoise_median, denoise_savgol, denoise_stl, raw | +6.3545 | +7.3153 | +0.9607 |
| `monash:nn5_daily|seasonal_high` | denoise_stl, seasonal_fill, winsorize | +0.0156 | +0.0156 | +0.0000 |
| `monash:nn5_daily|structured_mixed` | denoise_stl, forward_fill, winsorize | +0.0082 | +0.0082 | +0.0000 |
| `monash:nn5_daily|trend_high` | denoise_stl, seasonal_fill | +0.0590 | +0.0590 | +0.0000 |
| `monash:traffic_hourly|low_structure` | denoise_savgol, denoise_stl | +0.0484 | +0.0484 | +0.0000 |
| `monash:traffic_hourly|seasonal_high` | denoise_median, winsorize | +0.0981 | +0.0981 | -0.0000 |
| `monash:traffic_hourly|structured_mixed` | denoise_stl, winsorize | +0.0578 | +0.0578 | +0.0000 |
| `uci_electricity_load_diagrams|low_structure` | forward_fill, raw, winsorize | +0.0405 | +0.0405 | +0.0000 |
| `uci_electricity_load_diagrams|seasonal_high` | denoise_stl, forward_fill, seasonal_fill | +0.1317 | +0.1317 | +0.0000 |
| `uci_electricity_load_diagrams|structured_mixed` | forward_fill, raw, winsorize | +0.0578 | +0.0578 | +0.0000 |
| `uci_electricity_load_diagrams|trend_high` | denoise_stl, forward_fill | +0.1570 | +0.1570 | +0.0000 |
