# Benchmark benchmark-v0.1 Dev-Query baseline report

Final-Query was not read. Every value below is repeatable Dev-Query sMASE.

## Baselines (frozen folding ladder)

| baseline | overall (macro) | series-micro (descriptive) |
| --- | --- | --- |
| raw | 14.918809 | 14.050149 |
| best_fixed | 14.797151 | 13.959095 |
| h_ref | 15.055400 | 14.117777 |
| oracle_transfer | 14.821310 | 14.040775 |
| oracle_insample | 14.792513 | 13.950029 |

Best-fixed program selected on Support-A: `forward_fill`.

## What H_ref actually does

- Indistinguishable from Raw on 0/299 Dev series (0.0%).
- Better than Raw on 190; worse on 109.
- Net vs Raw (series-micro): +0.067628 (positive = worse than doing nothing).

## Headroom, measured from both floors

The oracle reverts to Raw in 0 cell(s): `[]`. In those cells the apparent gain over H_ref is H_ref's own damage refunded, not repair space.

| cell | oracle pick | gain over Raw | gain over H_ref | H_ref self-harm |
| --- | --- | --- | --- | --- |
| `gefcom2012_load|seasonal_high` | seasonal_fill | +0.0079 | +0.0078 | -0.0001 |
| `gefcom2012_load|structured_mixed` | seasonal_fill | +0.0129 | +0.0129 | -0.0000 |
| `metr_la|low_structure` | forward_fill | +0.0098 | +0.0099 | +0.0000 |
| `metr_la|structured_mixed` | seasonal_fill | +0.0121 | +0.0120 | -0.0001 |
| `metr_la|trend_high` | forward_fill | +0.0222 | +0.0222 | +0.0000 |
| `monash:covid_deaths|seasonal_high` | forward_fill | +1.2740 | +3.5550 | +2.2810 |
| `monash:covid_deaths|trend_high` | forward_fill | +0.6206 | +0.8906 | +0.2700 |
| `monash:nn5_daily|low_structure` | seasonal_fill | +0.0031 | +0.0031 | -0.0000 |
| `monash:nn5_daily|seasonal_high` | seasonal_fill | +0.0097 | +0.0097 | +0.0000 |
| `monash:nn5_daily|structured_mixed` | seasonal_fill | +0.0044 | +0.0044 | -0.0000 |
| `monash:traffic_hourly|low_structure` | forward_fill | +0.0125 | +0.0124 | -0.0001 |
| `monash:traffic_hourly|seasonal_high` | forward_fill | +0.0050 | +0.0068 | +0.0019 |
| `monash:traffic_hourly|structured_mixed` | forward_fill | +0.0086 | +0.0081 | -0.0006 |
| `uci_electricity_load_diagrams|low_structure` | forward_fill | +0.0030 | +0.0030 | -0.0000 |
| `uci_electricity_load_diagrams|seasonal_high` | seasonal_fill | +0.0364 | +0.0364 | +0.0000 |
| `uci_electricity_load_diagrams|structured_mixed` | forward_fill | +0.0087 | +0.0087 | -0.0001 |
| `uci_electricity_load_diagrams|trend_high` | forward_fill | +0.0236 | +0.0235 | -0.0001 |
