# R4F relationship atlas -- EXPLORATORY machine reading

Evidence class: EXPLORATORY / HYPOTHESIS_GENERATING; nothing here is evidence; anything entering the method must be frozen and re-tested on unread data

Boundary: fits=0, llm=0, held_out_reads=0, max_time_index_read=3863 (frontier 4056).
Rows: {'pooled': 1680, 'per_channel': 1680}; features: 27.

## A. Top-30 relationships (stability score = median |Spearman| over blocks x share of blocks with the pooled sign)

| # | consumer | program | channel | feature | n | Spearman pooled | by block | same-sign blocks | median abs | score | AUC helped (raw) | AUC severe (raw) | n severe | headlined before |
| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | pooled | ANCESTOR | ctx | local_robust_z_peak | 335 | -0.41242 | 0:40=-0.407386, 120:160=-0.58, 40:80=-0.353113, 80:120=-0.513035 | 4/4 | 0.46021 | 0.46021 | 0.604298 | 0.680723 | 3 | yes |
| 2 | pooled | W2 | ctx | local_robust_z_peak | 723 | -0.347005 | 0:40=-0.331135, 120:160=-0.546741, 40:80=-0.298125, 80:120=-0.44098 | 4/4 | 0.386057 | 0.386057 | 0.657022 | 0.2693 | 76 | yes |
| 3 | pooled | W2 | ctx | outlier_region_end_fraction | 723 | -0.289047 | 0:40=-0.262292, 120:160=-0.42199, 40:80=-0.270245, 80:120=-0.405937 | 4/4 | 0.338091 | 0.338091 | 0.620037 | 0.332272 | 76 | NEW |
| 4 | pooled | W2 | ctx | spike_sign_balance | 723 | -0.29889 | 0:40=-0.348739, 120:160=-0.427499, 40:80=-0.221736, 80:120=-0.30917 | 4/4 | 0.328954 | 0.328954 | 0.631084 | 0.319714 | 76 | NEW |
| 5 | pooled | W2 | ctx | spike_recurrence | 723 | -0.307259 | 0:40=-0.295748, 120:160=-0.494055, 40:80=-0.264429, 80:120=-0.345816 | 4/4 | 0.320782 | 0.320782 | 0.652931 | 0.328907 | 76 | NEW |
| 6 | pooled | W2 | ctx | spike_head_count | 723 | -0.2955 | 0:40=-0.312579, 120:160=-0.318967, 40:80=-0.263479, 80:120=-0.328905 | 4/4 | 0.315773 | 0.315773 | 0.646932 | 0.346396 | 76 | NEW |
| 7 | pooled | W2 | ctx | spike_recency | 723 | 0.291938 | 0:40=0.269018, 120:160=0.427833, 40:80=0.258081, 80:120=0.36135 | 4/4 | 0.315184 | 0.315184 | 0.361422 | 0.685471 | 76 | NEW |
| 8 | pooled | W2 | ctx | spike_peak_over_tail_sd | 723 | -0.268907 | 0:40=-0.28026, 120:160=-0.2963, 40:80=-0.282384, 80:120=-0.083916 | 4/4 | 0.281322 | 0.281322 | 0.628645 | 0.380572 | 76 | yes |
| 9 | pooled | ANCESTOR | ctx | outlier_region_end_fraction | 335 | -0.232486 | 0:40=-0.066078, 120:160=-0.354785, 40:80=-0.346732, 80:120=-0.150264 | 4/4 | 0.248498 | 0.248498 | 0.550614 | 0.74498 | 3 | NEW |
| 10 | pooled | W2 | ctx | spike_tail_count | 723 | -0.207644 | 0:40=-0.07923, 120:160=-0.251903, 40:80=-0.235991, 80:120=-0.353342 | 4/4 | 0.243947 | 0.243947 | 0.575406 | 0.397899 | 76 | NEW |
| 11 | pooled | ANCESTOR | ctx | spike_peak_over_tail_sd | 335 | -0.220615 | 0:40=-0.208613, 120:160=-0.412143, 40:80=-0.153995, 80:120=-0.213904 | 4/4 | 0.211259 | 0.211259 | 0.539298 | 0.838353 | 3 | yes |
| 12 | per_channel | ANCESTOR | total | local_robust_z_peak | 840 | -0.244238 | 0:40=-0.203583, 120:160=-0.009998, 40:80=-0.332382, 80:120=-0.189498 | 4/4 | 0.196541 | 0.196541 | 0.673809 | 0.357538 | 12 | yes |
| 13 | per_channel | W2 | total | srv_tail48_period_filled | 840 | -0.202197 | 0:40=-0.193097, 120:160=-0.009641, 40:80=-0.180522, 80:120=-0.236621 | 4/4 | 0.18681 | 0.18681 | 0.5728 | 0.469128 | 14 | NEW |
| 14 | pooled | W2 | total | spike_peak_over_tail_sd | 840 | -0.168298 | 0:40=-0.175913, 120:160=-0.204167, 40:80=-0.189708, 80:120=-0.119128 | 4/4 | 0.182811 | 0.182811 | 0.574547 | 0.350365 | 135 | yes |
| 15 | per_channel | ANCESTOR | ctx | local_robust_z_peak | 335 | -0.219924 | 0:40=-0.373133, 120:160=-0.112449, 40:80=-0.184335, 80:120=-0.167112 | 4/4 | 0.175724 | 0.175724 | 0.482423 | 0.973054 | 1 | yes |
| 16 | per_channel | W2 | total | spike_peak_over_tail_sd | 840 | -0.155804 | 0:40=-0.179746, 120:160=-0.201424, 40:80=-0.167758, 80:120=-0.11308 | 4/4 | 0.173752 | 0.173752 | 0.572483 | 0.529747 | 14 | yes |
| 17 | per_channel | W2 | total | gap_tail_fraction | 840 | -0.201279 | 0:40=-0.198674, 120:160=-0.009818, 40:80=-0.143434, 80:120=-0.183056 | 4/4 | 0.163245 | 0.163245 | 0.568123 | 0.546091 | 14 | NEW |
| 18 | per_channel | ANCESTOR | total | spike_peak_over_tail_sd | 840 | -0.181021 | 0:40=-0.136521, 120:160=-0.180135, 40:80=-0.225561, 80:120=-0.13543 | 4/4 | 0.158328 | 0.158328 | 0.662271 | 0.62188 | 12 | yes |
| 19 | per_channel | ANCESTOR | total | outlier_region_end_fraction | 840 | -0.228151 | 0:40=-0.197956, 120:160=-0.102397, 40:80=-0.287952, 80:120=-0.11411 | 4/4 | 0.156033 | 0.156033 | 0.635512 | 0.39845 | 12 | NEW |
| 20 | per_channel | ANCESTOR | total | srv_fill_divergence | 660 | -0.162766 | 0:40=-0.108772, 120:160=-0.189199, 40:80=-0.203013, 80:120=-0.072255 | 4/4 | 0.148985 | 0.148985 | 0.616139 | 0.465752 | 12 | NEW |
| 21 | pooled | ANCESTOR | ctx | level_region_end_fraction | 335 | -0.024935 | 0:40=-0.090173, 120:160=0.305416, 40:80=-0.070878, 80:120=-0.35071 | 3/4 | 0.197794 | 0.148346 | 0.510526 | 0.444277 | 3 | NEW |
| 22 | pooled | W2 | total | spike_head_count | 840 | -0.15732 | 0:40=-0.169287, 120:160=-0.242931, 40:80=-0.222097, 80:120=0.066358 | 3/4 | 0.195692 | 0.146769 | 0.561813 | 0.373165 | 135 | NEW |
| 23 | per_channel | ANCESTOR | total | estimated_region_start_fraction | 840 | 0.120501 | 0:40=0.040637, 120:160=0.211255, 40:80=0.133631, 80:120=0.150004 | 4/4 | 0.141817 | 0.141817 | 0.443181 | 0.311443 | 12 | NEW |
| 24 | pooled | ANCESTOR | ctx | level_excursion_score | 335 | -0.029807 | 0:40=-0.089395, 120:160=0.288529, 40:80=-0.074303, 80:120=-0.349532 | 3/4 | 0.188962 | 0.141721 | 0.511732 | 0.444277 | 3 | NEW |
| 25 | pooled | ANCESTOR | ctx | level_region_fraction | 335 | -0.02583 | 0:40=-0.089945, 120:160=0.286409, 40:80=-0.073465, 80:120=-0.338715 | 3/4 | 0.188177 | 0.141133 | 0.512456 | 0.444277 | 3 | NEW |
| 26 | pooled | W2 | ctx | srv_period_filled_frac | 723 | 0.132054 | 0:40=0.1072, 120:160=0.154497, 40:80=0.118715, 80:120=0.236262 | 4/4 | 0.136606 | 0.136606 | 0.432895 | 0.613479 | 76 | NEW |
| 27 | per_channel | ANCESTOR | total | spike_head_count | 840 | -0.185858 | 0:40=-0.088101, 120:160=-0.09452, 40:80=-0.270371, 80:120=-0.177972 | 4/4 | 0.136246 | 0.136246 | 0.650194 | 0.386524 | 12 | NEW |
| 28 | pooled | ANCESTOR | ctx | gap_recurrence | 335 | 0.080415 | 0:40=0.088132, 120:160=-0.441569, 40:80=0.139637, 80:120=0.221495 | 3/4 | 0.180566 | 0.135424 | 0.444649 | 0.799699 | 3 | NEW |
| 29 | pooled | W2 | ctx | level_region_end_fraction | 723 | -0.135178 | 0:40=-0.145142, 120:160=-0.022047, 40:80=-0.115861, 80:120=-0.25067 | 4/4 | 0.130501 | 0.130501 | 0.535214 | 0.471406 | 76 | NEW |
| 30 | pooled | W2 | ctx | level_region_fraction | 723 | -0.134594 | 0:40=-0.145075, 120:160=-0.019634, 40:80=-0.115691, 80:120=-0.252322 | 4/4 | 0.130383 | 0.130383 | 0.53511 | 0.471406 | 76 | NEW |

## B. Tree probe (depth<=3, min_leaf 40; what got selected, never a router)

### pooled|ANCESTOR|total (n=840)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | spike_peak_over_tail_sd | 15.003306 | 840 | -0.235036 | 785 / -0.203972 | 55 / -0.678411 |
| 1 | srv_gap_points | 144.5 | 785 | -0.203972 | 737 / -0.2192 | 48 / 0.029843 |
| 2 | outlier_region_end_fraction | 0.541667 | 737 | -0.2192 | 543 / -0.18211 | 194 / -0.323013 |

LODO root features: {"[0:40]": "spike_peak_over_tail_sd", "[120:160]": "missing_fraction", "[40:80]": "spike_peak_over_tail_sd", "[80:120]": "spike_peak_over_tail_sd"}; feature use across folds: {"spike_peak_over_tail_sd": 3, "local_robust_z_peak": 2, "missing_fraction": 2, "gap_tail_fraction": 1, "estimated_region_start_fraction": 1, "srv_fill_divergence": 1, "estimated_region_end_fraction": 1, "outlier_region_end_fraction": 1}

### pooled|ANCESTOR|ctx (n=335)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | local_robust_z_peak | 10.035146 | 335 | -0.181361 | 287 / -0.084024 | 48 / -0.763353 |
| 1 | local_robust_z_peak | 5.723253 | 287 | -0.084024 | 195 / -0.041534 | 92 / -0.174085 |
| 2 | estimated_region_end_fraction | 0.997396 | 195 | -0.041534 | 153 / -0.025273 | 42 / -0.100773 |
| 5 | outlier_region_end_fraction | 0.755208 | 92 | -0.174085 | 48 / -0.127352 | 44 / -0.225066 |

LODO root features: {"[0:40]": "local_robust_z_peak", "[120:160]": "local_robust_z_peak", "[40:80]": "local_robust_z_peak", "[80:120]": "local_robust_z_peak"}; feature use across folds: {"local_robust_z_peak": 4, "outlier_region_end_fraction": 1, "estimated_region_start_fraction": 1, "estimated_region_end_fraction": 1, "srv_period_filled_frac": 1}

### pooled|W2_pmc_then_outlier_mad|total (n=840)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | spike_peak_over_tail_sd | 15.003306 | 840 | -0.22866 | 785 / -0.194555 | 55 / -0.715441 |
| 1 | spike_peak_over_tail_sd | 5.274864 | 785 | -0.194555 | 528 / -0.134722 | 257 / -0.31748 |
| 2 | longest_missing_run_fraction | 0.033854 | 528 | -0.134722 | 272 / -0.060974 | 256 / -0.213079 |
| 5 | srv_fill_divergence | 1.588137 | 257 | -0.31748 | 209 / -0.366591 | 48 / -0.103639 |

LODO root features: {"[0:40]": "spike_peak_over_tail_sd", "[120:160]": "spike_peak_over_tail_sd", "[40:80]": "spike_peak_over_tail_sd", "[80:120]": "spike_peak_over_tail_sd"}; feature use across folds: {"spike_peak_over_tail_sd": 4, "period_reliability": 2, "srv_fill_divergence": 2, "srv_gap_points": 2, "srv_donor_dispersion": 1, "local_robust_z_peak": 1, "outlier_region_end_fraction": 1, "longest_missing_run_fraction": 1}

### pooled|W2_pmc_then_outlier_mad|ctx (n=723)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | spike_peak_over_tail_sd | 11.873903 | 723 | -0.015439 | 648 / 0.036237 | 75 / -0.461913 |
| 1 | spike_recency | 0.820312 | 648 | 0.036237 | 311 / -0.028705 | 337 / 0.096168 |
| 2 | local_robust_z_peak | 5.381874 | 311 | -0.028705 | 192 / 0.005745 | 119 / -0.084289 |
| 5 | srv_fill_divergence | 1.097911 | 337 | 0.096168 | 274 / 0.068787 | 63 / 0.215256 |

LODO root features: {"[0:40]": "local_robust_z_peak", "[120:160]": "local_robust_z_peak", "[40:80]": "spike_peak_over_tail_sd", "[80:120]": "spike_peak_over_tail_sd"}; feature use across folds: {"local_robust_z_peak": 4, "spike_peak_over_tail_sd": 3, "spike_recency": 2, "srv_fill_divergence": 2, "srv_period_filled_points": 2, "srv_donor_dispersion": 1}

### per_channel|ANCESTOR|total (n=840)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | local_robust_z_peak | 4.440897 | 840 | -0.045074 | 523 / -0.016266 | 317 / -0.092604 |
| 1 | estimated_region_end_fraction | 0.221354 | 523 | -0.016266 | 40 / -0.118055 | 483 / -0.007836 |
| 3 | srv_period_filled_frac | 0.291637 | 483 | -0.007836 | 130 / 0.027843 | 353 / -0.020975 |
| 6 | srv_period_filled_points | 11.5 | 317 | -0.092604 | 217 / -0.055871 | 100 / -0.172315 |
| 7 | spike_recency | 0.049479 | 217 | -0.055871 | 41 / -0.00527 | 176 / -0.067658 |
| 10 | outlier_region_end_fraction | 0.763021 | 100 | -0.172315 | 58 / -0.100387 | 42 / -0.271645 |

LODO root features: {"[0:40]": "local_robust_z_peak", "[120:160]": "local_robust_z_peak", "[40:80]": "spike_peak_over_tail_sd", "[80:120]": "local_robust_z_peak"}; feature use across folds: {"local_robust_z_peak": 4, "srv_fill_divergence": 3, "estimated_region_end_fraction": 3, "srv_period_filled_points": 2, "outlier_region_end_fraction": 2, "srv_tail48_period_filled": 1, "longest_missing_run_fraction": 1, "srv_gap_points": 1, "srv_period_filled_frac": 1, "spike_peak_over_tail_sd": 1, "spike_recency": 1}

### per_channel|ANCESTOR|ctx (n=335)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | spike_head_count | 18.5 | 335 | -0.039378 | 283 / -0.022376 | 52 / -0.131909 |
| 1 | longest_missing_run_fraction | 0.106771 | 283 | -0.022376 | 243 / -0.01262 | 40 / -0.081639 |
| 2 | local_robust_z_peak | 4.862622 | 243 | -0.01262 | 128 / -0.003625 | 115 / -0.022632 |

LODO root features: {"[0:40]": "local_robust_z_peak", "[120:160]": "spike_head_count", "[40:80]": "local_robust_z_peak", "[80:120]": "spike_head_count"}; feature use across folds: {"local_robust_z_peak": 3, "spike_recency": 2, "spike_head_count": 2, "spike_peak_over_tail_sd": 2}

### per_channel|W2_pmc_then_outlier_mad|total (n=840)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | srv_tail48_period_filled | 2.5 | 840 | -0.109411 | 549 / -0.05722 | 291 / -0.207873 |
| 1 | estimated_region_start_fraction | 0.002604 | 549 | -0.05722 | 180 / -0.104178 | 369 / -0.034314 |
| 2 | spike_peak_over_tail_sd | 4.544547 | 180 | -0.104178 | 108 / -0.043804 | 72 / -0.194738 |
| 5 | srv_fill_divergence | 1.374319 | 369 | -0.034314 | 304 / -0.012219 | 65 / -0.137653 |
| 8 | outlier_region_end_fraction | 0.778646 | 291 | -0.207873 | 247 / -0.155372 | 44 / -0.502594 |
| 9 | spike_peak_over_tail_sd | 3.793438 | 247 | -0.155372 | 89 / -0.064536 | 158 / -0.20654 |

LODO root features: {"[0:40]": "srv_tail48_period_filled", "[120:160]": "srv_tail48_period_filled", "[40:80]": "srv_tail48_period_filled", "[80:120]": "srv_tail48_period_filled"}; feature use across folds: {"srv_fill_divergence": 4, "srv_tail48_period_filled": 4, "spike_peak_over_tail_sd": 4, "estimated_region_start_fraction": 3, "outlier_region_end_fraction": 3, "gap_longest_run_steps": 1, "spike_recency": 1}

### per_channel|W2_pmc_then_outlier_mad|ctx (n=723)

| node | feature | threshold | n | mean | left n/mean | right n/mean |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0 | spike_head_count | 18.5 | 723 | -0.028479 | 671 / -0.020272 | 52 / -0.134377 |
| 1 | spike_peak_over_tail_sd | 12.740381 | 671 | -0.020272 | 627 / -0.015167 | 44 / -0.093028 |
| 2 | longest_missing_run_fraction | 0.028646 | 627 | -0.015167 | 267 / 0.000323 | 360 / -0.026654 |

LODO root features: {"[0:40]": "srv_period_filled_points", "[120:160]": "spike_head_count", "[40:80]": "srv_fill_divergence", "[80:120]": "spike_head_count"}; feature use across folds: {"srv_fill_divergence": 3, "srv_period_filled_points": 2, "longest_missing_run_fraction": 2, "srv_period_filled_frac": 2, "spike_head_count": 2, "gap_recurrence": 1, "spike_peak_over_tail_sd": 1, "srv_tail48_period_filled": 1, "period_reliability": 1}

## C. Relationships between programs and Consumers

- **pooled|total_ANC_vs_total_W2**: {"n": 840, "spearman": 0.553627, "by_block": {"[0:40]": 0.629425, "[120:160]": 0.560574, "[40:80]": 0.431501, "[80:120]": 0.603563}}
- **pooled|ANCESTOR|route_vs_ctx**: {"n": 840, "spearman": -0.026899, "spearman_ctx_active_only": -0.028704}
- **pooled|W2_pmc_then_outlier_mad|route_vs_ctx**: {"n": 840, "spearman": -0.131306, "spearman_ctx_active_only": -0.142205}
- **per_channel|total_ANC_vs_total_W2**: {"n": 840, "spearman": 0.488587, "by_block": {"[0:40]": 0.449828, "[120:160]": 0.657195, "[40:80]": 0.485058, "[80:120]": 0.401362}}
- **per_channel|ANCESTOR|route_vs_ctx**: {"n": 840, "spearman": -0.027113, "spearman_ctx_active_only": -0.062344}
- **per_channel|W2_pmc_then_outlier_mad|route_vs_ctx**: {"n": 840, "spearman": -0.021251, "spearman_ctx_active_only": -0.017793}
- **pooled_vs_per_channel|ANCESTOR**: {"n": 840, "spearman_total": 0.040996, "spearman_route": 0.003446, "spearman_ctx": 0.192567, "spearman_static_L_rr": 0.477137}
- **pooled_vs_per_channel|W2_pmc_then_outlier_mad**: {"n": 840, "spearman_total": -0.023085, "spearman_route": 0.005895, "spearman_ctx": 0.097611, "spearman_static_L_rr": 0.477137}
### m_r0k per-series gain Spearman matrix

| | ANCESTOR | W1 | W2 | CAND | CAND |
| --- | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | 1.0 (n=840) | 0.453122 (n=840) | 0.553627 (n=840) | 0.647185 (n=540) | 0.717054 (n=540) |
| W1 | 0.453122 (n=840) | 1.0 (n=840) | 0.389012 (n=840) | 0.673036 (n=540) | 0.452287 (n=540) |
| W2 | 0.553627 (n=840) | 0.389012 (n=840) | 1.0 (n=840) | 0.449397 (n=540) | 0.505989 (n=540) |
| CAND | 0.647185 (n=540) | 0.673036 (n=540) | 0.449397 (n=540) | 1.0 (n=540) | 0.675218 (n=540) |
| CAND | 0.717054 (n=540) | 0.452287 (n=540) | 0.505989 (n=540) | 0.675218 (n=540) | 1.0 (n=540) |

- **pooled|helped_overlap_ANC_W2**: {"n": 840, "both_helped": 439, "neither": 162, "only_ANC": 139, "only_W2": 100}

## D. Gallery

Figures in `artifacts/main_protocol/r4f_gallery/`: pooled_ANCESTOR_total_worst6.png, pooled_ANCESTOR_total_best6.png, pooled_ANCESTOR_ctx_worst3_best3.png, pooled_W2_total_worst6.png, pooled_W2_total_best6.png, pooled_W2_ctx_worst3_best3.png, per_channel_ANCESTOR_total_worst6.png, per_channel_ANCESTOR_total_best6.png, per_channel_ANCESTOR_ctx_worst3_best3.png, per_channel_W2_total_worst6.png, per_channel_W2_total_best6.png, per_channel_W2_ctx_worst3_best3.png

| figure | panel | uid | origin | block | program | total | route | ctx | moved pts | NaN in ctx | z_peak | missing_fraction |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled_ANCESTOR_total_worst6.png | 0 | T185 | 1464 | [80:120] | ANCESTOR | 2.021521 | 2.021521 | 0.0 | 0 | 10 | 1.211797 | 0.052083 |
| pooled_ANCESTOR_total_worst6.png | 1 | T119 | 3576 | [0:40] | ANCESTOR | 1.180821 | 1.180821 | 0.0 | 0 | 167 | 8800000000.0 | 0.869792 |
| pooled_ANCESTOR_total_worst6.png | 2 | T188 | 1464 | [80:120] | ANCESTOR | 1.115755 | 1.12578 | -0.010025 | 1 | 60 | 4.42082 | 0.3125 |
| pooled_ANCESTOR_total_worst6.png | 3 | T11 | 3576 | [0:40] | ANCESTOR | 1.058035 | 1.058035 | 0.0 | 0 | 167 | 10200000000.0 | 0.869792 |
| pooled_ANCESTOR_total_worst6.png | 4 | T118 | 1176 | [0:40] | ANCESTOR | 0.947602 | 0.949517 | -0.001915 | 1 | 9 | 3.514452 | 0.046875 |
| pooled_ANCESTOR_total_worst6.png | 5 | T117 | 2184 | [0:40] | ANCESTOR | 0.878831 | 0.878831 | 0.0 | 0 | 11 | 2.697963 | 0.057292 |
| pooled_ANCESTOR_total_best6.png | 0 | T23 | 1224 | [120:160] | ANCESTOR | -7.198668 | -0.377629 | -6.821039 | 4 | 23 | 12.707406 | 0.119792 |
| pooled_ANCESTOR_total_best6.png | 1 | T23 | 1176 | [120:160] | ANCESTOR | -5.293311 | 1.55387 | -6.847181 | 3 | 11 | 24.206724 | 0.057292 |
| pooled_ANCESTOR_total_best6.png | 2 | T23 | 1464 | [120:160] | ANCESTOR | -4.421447 | -0.07319 | -4.348257 | 1 | 4 | 10.041209 | 0.020833 |
| pooled_ANCESTOR_total_best6.png | 3 | T23 | 1416 | [120:160] | ANCESTOR | -2.948581 | -0.199439 | -2.749142 | 1 | 9 | 10.675561 | 0.046875 |
| pooled_ANCESTOR_total_best6.png | 4 | T146 | 3816 | [40:80] | ANCESTOR | -2.923333 | -0.707543 | -2.21579 | 1 | 64 | 10.870281 | 0.333333 |
| pooled_ANCESTOR_total_best6.png | 5 | T114 | 1176 | [0:40] | ANCESTOR | -2.613365 | -1.426097 | -1.187268 | 2 | 14 | 10.039961 | 0.072917 |
| pooled_ANCESTOR_ctx_worst3_best3.png | 0 | T14 | 2856 | [40:80] | ANCESTOR | 0.770323 | 0.356792 | 0.413531 | 8 | 83 | 10.03033 | 0.432292 |
| pooled_ANCESTOR_ctx_worst3_best3.png | 1 | T104 | 2616 | [0:40] | ANCESTOR | -1.846311 | -2.226987 | 0.380676 | 4 | 63 | 4.442419 | 0.328125 |
| pooled_ANCESTOR_ctx_worst3_best3.png | 2 | T150 | 1704 | [40:80] | ANCESTOR | 0.313522 | -0.049372 | 0.362893 | 29 | 70 | 10.073492 | 0.364583 |
| pooled_ANCESTOR_ctx_worst3_best3.png | 3 | T23 | 1176 | [120:160] | ANCESTOR | -5.293311 | 1.55387 | -6.847181 | 3 | 11 | 24.206724 | 0.057292 |
| pooled_ANCESTOR_ctx_worst3_best3.png | 4 | T23 | 1224 | [120:160] | ANCESTOR | -7.198668 | -0.377629 | -6.821039 | 4 | 23 | 12.707406 | 0.119792 |
| pooled_ANCESTOR_ctx_worst3_best3.png | 5 | T23 | 1464 | [120:160] | ANCESTOR | -4.421447 | -0.07319 | -4.348257 | 1 | 4 | 10.041209 | 0.020833 |
| pooled_W2_total_worst6.png | 0 | T152 | 2904 | [40:80] | W2 | 1.848864 | 1.634776 | 0.214087 | 17 | 32 | 3.032817 | 0.166667 |
| pooled_W2_total_worst6.png | 1 | T2 | 1176 | [80:120] | W2 | 1.826279 | 0.471642 | 1.354637 | 42 | 75 | 3.592551 | 0.390625 |
| pooled_W2_total_worst6.png | 2 | T158 | 2904 | [40:80] | W2 | 1.794245 | 1.193747 | 0.600498 | 24 | 30 | 4.263448 | 0.15625 |
| pooled_W2_total_worst6.png | 3 | T149 | 3624 | [40:80] | W2 | 1.644398 | 1.669706 | -0.025308 | 1 | 142 | 12100000000.0 | 0.739583 |
| pooled_W2_total_worst6.png | 4 | T143 | 3624 | [40:80] | W2 | 1.616498 | 1.615858 | 0.00064 | 1 | 142 | 11100000000.0 | 0.739583 |
| pooled_W2_total_worst6.png | 5 | T155 | 3624 | [40:80] | W2 | 1.568729 | 1.774245 | -0.205517 | 2 | 143 | 11500000000.0 | 0.744792 |
| pooled_W2_total_best6.png | 0 | T23 | 1224 | [120:160] | W2 | -7.591419 | 1.063481 | -8.6549 | 25 | 23 | 12.707406 | 0.119792 |
| pooled_W2_total_best6.png | 1 | T23 | 1176 | [120:160] | W2 | -5.419003 | 1.516842 | -6.935845 | 20 | 11 | 24.206724 | 0.057292 |
| pooled_W2_total_best6.png | 2 | T23 | 1464 | [120:160] | W2 | -5.233779 | -3.134368 | -2.099411 | 4 | 4 | 10.041209 | 0.020833 |
| pooled_W2_total_best6.png | 3 | T114 | 1176 | [0:40] | W2 | -3.552209 | -1.451297 | -2.100912 | 15 | 14 | 10.039961 | 0.072917 |
| pooled_W2_total_best6.png | 4 | T119 | 1896 | [0:40] | W2 | -3.289524 | -3.298295 | 0.008772 | 3 | 40 | 4.121888 | 0.208333 |
| pooled_W2_total_best6.png | 5 | T146 | 3816 | [40:80] | W2 | -2.960438 | -2.669667 | -0.290771 | 32 | 64 | 10.870281 | 0.333333 |
| pooled_W2_ctx_worst3_best3.png | 0 | T2 | 1176 | [80:120] | W2 | 1.826279 | 0.471642 | 1.354637 | 42 | 75 | 3.592551 | 0.390625 |
| pooled_W2_ctx_worst3_best3.png | 1 | T152 | 2136 | [40:80] | W2 | 1.086344 | -0.082793 | 1.169137 | 39 | 69 | 5.106859 | 0.359375 |
| pooled_W2_ctx_worst3_best3.png | 2 | T155 | 1656 | [40:80] | W2 | 1.000946 | -0.118625 | 1.119571 | 34 | 39 | 1.427702 | 0.203125 |
| pooled_W2_ctx_worst3_best3.png | 3 | T23 | 1224 | [120:160] | W2 | -7.591419 | 1.063481 | -8.6549 | 25 | 23 | 12.707406 | 0.119792 |
| pooled_W2_ctx_worst3_best3.png | 4 | T23 | 1176 | [120:160] | W2 | -5.419003 | 1.516842 | -6.935845 | 20 | 11 | 24.206724 | 0.057292 |
| pooled_W2_ctx_worst3_best3.png | 5 | T23 | 1416 | [120:160] | W2 | -2.368385 | -0.162083 | -2.206302 | 4 | 9 | 10.675561 | 0.046875 |
| per_channel_ANCESTOR_total_worst6.png | 0 | T2 | 1224 | [80:120] | ANCESTOR | 0.769697 | 0.769697 | 0.0 | 0 | 83 | 1.604505 | 0.432292 |
| per_channel_ANCESTOR_total_worst6.png | 1 | T116 | 2184 | [0:40] | ANCESTOR | 0.484121 | 0.484121 | 0.0 | 0 | 28 | 2.852346 | 0.145833 |
| per_channel_ANCESTOR_total_worst6.png | 2 | T110 | 1224 | [0:40] | ANCESTOR | 0.398504 | 0.398504 | 0.0 | 0 | 91 | 1.451863 | 0.473958 |
| per_channel_ANCESTOR_total_worst6.png | 3 | T152 | 1224 | [40:80] | ANCESTOR | 0.397345 | 0.397345 | 0.0 | 0 | 77 | 1.856117 | 0.401042 |
| per_channel_ANCESTOR_total_worst6.png | 4 | T146 | 1224 | [40:80] | ANCESTOR | 0.372964 | 0.372964 | 0.0 | 0 | 93 | 1.958184 | 0.484375 |
| per_channel_ANCESTOR_total_worst6.png | 5 | T1 | 2376 | [0:40] | ANCESTOR | 0.365649 | 0.412719 | -0.04707 | 4 | 0 | 4.377577 | 0.0 |
| per_channel_ANCESTOR_total_best6.png | 0 | T16 | 1656 | [40:80] | ANCESTOR | -2.322126 | -0.965112 | -1.357014 | 46 | 67 | 27.429291 | 0.348958 |
| per_channel_ANCESTOR_total_best6.png | 1 | T104 | 2616 | [0:40] | ANCESTOR | -1.447391 | -1.353566 | -0.093826 | 4 | 63 | 4.442419 | 0.328125 |
| per_channel_ANCESTOR_total_best6.png | 2 | T234 | 1176 | [120:160] | ANCESTOR | -1.311783 | -1.311783 | 0.0 | 0 | 0 | 1.862465 | 0.0 |
| per_channel_ANCESTOR_total_best6.png | 3 | T158 | 2136 | [40:80] | ANCESTOR | -1.080925 | -0.784476 | -0.296449 | 36 | 63 | 7.380856 | 0.328125 |
| per_channel_ANCESTOR_total_best6.png | 4 | T110 | 1176 | [0:40] | ANCESTOR | -0.85994 | -0.676169 | -0.183771 | 1 | 85 | 8.390665 | 0.442708 |
| per_channel_ANCESTOR_total_best6.png | 5 | T235 | 1176 | [120:160] | ANCESTOR | -0.826671 | -0.826671 | 0.0 | 0 | 0 | 1.87579 | 0.0 |
| per_channel_ANCESTOR_ctx_worst3_best3.png | 0 | T23 | 1176 | [120:160] | ANCESTOR | 0.345583 | -0.00313 | 0.348712 | 3 | 11 | 24.206724 | 0.057292 |
| per_channel_ANCESTOR_ctx_worst3_best3.png | 1 | T15 | 1704 | [40:80] | ANCESTOR | 0.220458 | -0.063733 | 0.284191 | 44 | 99 | 12.000845 | 0.515625 |
| per_channel_ANCESTOR_ctx_worst3_best3.png | 2 | T114 | 2904 | [0:40] | ANCESTOR | 0.09046 | -0.098731 | 0.189191 | 31 | 4 | 26.30514 | 0.020833 |
| per_channel_ANCESTOR_ctx_worst3_best3.png | 3 | T182 | 1176 | [80:120] | ANCESTOR | -0.75031 | 0.644574 | -1.394884 | 5 | 116 | 18.432041 | 0.604167 |
| per_channel_ANCESTOR_ctx_worst3_best3.png | 4 | T16 | 1656 | [40:80] | ANCESTOR | -2.322126 | -0.965112 | -1.357014 | 46 | 67 | 27.429291 | 0.348958 |
| per_channel_ANCESTOR_ctx_worst3_best3.png | 5 | T144 | 1656 | [40:80] | ANCESTOR | -0.807311 | -0.144341 | -0.662971 | 42 | 38 | 26.97963 | 0.197917 |
| per_channel_W2_total_worst6.png | 0 | T188 | 1176 | [80:120] | W2 | 1.186306 | 1.186306 | 0.0 | 0 | 174 | 15900000000.0 | 0.90625 |
| per_channel_W2_total_worst6.png | 1 | T14 | 1176 | [40:80] | W2 | 0.845616 | 0.903632 | -0.058015 | 38 | 77 | 3.607854 | 0.401042 |
| per_channel_W2_total_worst6.png | 2 | T188 | 1416 | [80:120] | W2 | 0.624077 | 0.258771 | 0.365306 | 21 | 97 | 7.099903 | 0.505208 |
| per_channel_W2_total_worst6.png | 3 | T194 | 1224 | [80:120] | W2 | 0.60959 | 0.549333 | 0.060258 | 22 | 101 | 2.106612 | 0.526042 |
| per_channel_W2_total_worst6.png | 4 | T155 | 3624 | [40:80] | W2 | 0.480215 | 0.485521 | -0.005306 | 2 | 143 | 11500000000.0 | 0.744792 |
| per_channel_W2_total_worst6.png | 5 | T143 | 3624 | [40:80] | W2 | 0.472112 | 0.473878 | -0.001765 | 1 | 142 | 11100000000.0 | 0.739583 |
| per_channel_W2_total_best6.png | 0 | T104 | 2616 | [0:40] | W2 | -2.915516 | -2.1076 | -0.807916 | 42 | 63 | 4.442419 | 0.328125 |
| per_channel_W2_total_best6.png | 1 | T16 | 1656 | [40:80] | W2 | -2.489334 | -0.90843 | -1.580904 | 74 | 67 | 27.429291 | 0.348958 |
| per_channel_W2_total_best6.png | 2 | T2 | 1176 | [80:120] | W2 | -1.716316 | -1.724304 | 0.007988 | 42 | 75 | 3.592551 | 0.390625 |
| per_channel_W2_total_best6.png | 3 | T158 | 2184 | [40:80] | W2 | -1.516333 | -1.426162 | -0.090172 | 18 | 24 | 2.030725 | 0.125 |
| per_channel_W2_total_best6.png | 4 | T158 | 2136 | [40:80] | W2 | -1.365165 | -1.308106 | -0.057059 | 66 | 63 | 7.380856 | 0.328125 |
| per_channel_W2_total_best6.png | 5 | T234 | 1176 | [120:160] | W2 | -1.329957 | -1.329957 | 0.0 | 0 | 0 | 1.862465 | 0.0 |
| per_channel_W2_ctx_worst3_best3.png | 0 | T15 | 1704 | [40:80] | W2 | 0.281093 | -0.089269 | 0.370362 | 74 | 99 | 12.000845 | 0.515625 |
| per_channel_W2_ctx_worst3_best3.png | 1 | T188 | 1416 | [80:120] | W2 | 0.624077 | 0.258771 | 0.365306 | 21 | 97 | 7.099903 | 0.505208 |
| per_channel_W2_ctx_worst3_best3.png | 2 | T14 | 1224 | [40:80] | W2 | -0.790402 | -1.098297 | 0.307895 | 32 | 104 | 2.425077 | 0.541667 |
| per_channel_W2_ctx_worst3_best3.png | 3 | T16 | 1656 | [40:80] | W2 | -2.489334 | -0.90843 | -1.580904 | 74 | 67 | 27.429291 | 0.348958 |
| per_channel_W2_ctx_worst3_best3.png | 4 | T104 | 2616 | [0:40] | W2 | -2.915516 | -2.1076 | -0.807916 | 42 | 63 | 4.442419 | 0.328125 |
| per_channel_W2_ctx_worst3_best3.png | 5 | T144 | 1656 | [40:80] | W2 | -0.931434 | -0.289359 | -0.642075 | 60 | 38 | 26.97963 | 0.197917 |
