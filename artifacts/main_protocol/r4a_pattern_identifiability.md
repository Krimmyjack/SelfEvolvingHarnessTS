# R4A pattern identifiability -- machine reading

Evidence class: MECHANISM / NEGATIVE-capable; not a capability claim

Boundary: fits=0, llm=0, held_out_reads=0, max_time_index_read=3911 (frontier 4056).
Loader: `evaluation.main_protocol_p4.preflight_natural_gap_variant.load_variant`, NaN=503712.

## Per (program, face)

| cell | rows | uids | helped | severe_harm | mean g | best visible (feature / target) | LODO mean | LODO folds | verdict | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | 420 | 80 | 0.692857 | 0.066667 | 0.245427 | local_robust_z_peak\|severe_harm | 0.633725 | [0:40]=0.525021, [120:160]=0.596491, [40:80]=0.544966, [80:120]=0.868421 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| ANCESTOR\|delayed_face | 420 | 80 | 0.683333 | 0.073810 | 0.224645 | local_robust_z_peak\|severe_harm | 0.676036 | [0:40]=0.717803, [120:160]=0.599129, [40:80]=0.702528, [80:120]=0.684685 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| W1_hampel_filter\|support_face | 420 | 80 | 0.459524 | 0.292857 | -0.031355 | local_robust_z_peak\|severe_harm | 0.639015 | [0:40]=0.624490, [120:160]=0.778708, [40:80]=0.605546, [80:120]=0.547315 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| W1_hampel_filter\|delayed_face | 420 | 80 | 0.509524 | 0.235714 | 0.045469 | spike_peak_over_tail_sd\|helped | 0.637288 | [0:40]=0.641711, [120:160]=0.802469, [40:80]=0.559970, [80:120]=0.545000 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| W2_pmc_then_outlier_mad\|support_face | 420 | 80 | 0.640476 | 0.142857 | 0.272318 | spike_peak_over_tail_sd\|severe_harm | 0.648404 | [0:40]=0.703210, [120:160]=0.543599, [40:80]=0.662823, [80:120]=0.683983 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| W2_pmc_then_outlier_mad\|delayed_face | 420 | 80 | 0.642857 | 0.178571 | 0.185003 | local_robust_z_peak\|severe_harm | 0.656777 | [0:40]=0.724267, [120:160]=0.677560, [40:80]=0.684158, [80:120]=0.541126 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| W3_outlier_mad_then_pmc\|support_face | 420 | 80 | 0.692857 | 0.066667 | 0.245427 | local_robust_z_peak\|severe_harm | 0.633725 | [0:40]=0.525021, [120:160]=0.596491, [40:80]=0.544966, [80:120]=0.868421 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| W3_outlier_mad_then_pmc\|delayed_face | 420 | 80 | 0.683333 | 0.073810 | 0.224645 | local_robust_z_peak\|severe_harm | 0.676036 | [0:40]=0.717803, [120:160]=0.599129, [40:80]=0.702528, [80:120]=0.684685 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face (ref) | 240 | 60 | 0.616667 | 0.137500 | 0.211554 | spike_head_count\|severe_harm | 0.628758 | [120:160]=0.706897, [40:80]=0.591487, [80:120]=0.587891 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face (ref) | 280 | 60 | 0.600000 | 0.153571 | 0.184201 | spike_head_count\|severe_harm | 0.611330 | [120:160]=0.672367, [40:80]=0.696346, [80:120]=0.465278 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| CAND_outlier_iqr({})>winsorize({})\|support_face (ref) | 240 | 60 | 0.720833 | 0.070833 | 0.331931 | estimated_region_end_fraction\|helped | 0.630549 | [120:160]=0.595442, [40:80]=0.564064, [80:120]=0.732143 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face (ref) | 280 | 60 | 0.700000 | 0.067857 | 0.247086 | local_robust_z_peak\|severe_harm | 0.804630 | [120:160]=0.966102, [40:80]=0.699217, [80:120]=0.748571 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |

## Frozen-bin stump, utility over the whole served population

| cell | treat-all utility | best stump | mean vs treat-all | mean vs treat-none |
| --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | 0.245427 | spike_recurrence >= 0.000000 | +0.000000 | +0.267773 |
| ANCESTOR\|delayed_face | 0.224645 | spike_sign_balance >= 0.000000 | +0.000277 | +0.231133 |
| W1_hampel_filter\|support_face | -0.031355 | spike_peak_over_tail_sd >= 6.000000 | +0.081614 | +0.046130 |
| W1_hampel_filter\|delayed_face | 0.045469 | spike_head_count >= 1.000000 | +0.050479 | +0.121932 |
| W2_pmc_then_outlier_mad\|support_face | 0.272318 | spike_recurrence >= 0.000000 | +0.000000 | +0.267159 |
| W2_pmc_then_outlier_mad\|delayed_face | 0.185003 | spike_recurrence >= 0.000000 | +0.000000 | +0.239236 |
| W3_outlier_mad_then_pmc\|support_face | 0.245427 | spike_recurrence >= 0.000000 | +0.000000 | +0.267773 |
| W3_outlier_mad_then_pmc\|delayed_face | 0.224645 | spike_sign_balance >= 0.000000 | +0.000277 | +0.231133 |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face | 0.211554 | spike_recurrence >= 0.000000 | +0.000000 | +0.184196 |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face | 0.184201 | spike_recurrence >= 0.000000 | +0.000000 | +0.216454 |
| CAND_outlier_iqr({})>winsorize({})\|support_face | 0.331931 | spike_recurrence >= 0.000000 | +0.000000 | +0.329973 |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face | 0.247086 | spike_recurrence >= 0.000000 | +0.000000 | +0.268125 |

## ORACLE row (severe_harm, pooled AUC)

| cell | oracle feature | oracle AUC | best visible AUC | margin |
| --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | oracle_future_spike | 0.536990 | 0.582407 | -0.045417 |
| ANCESTOR\|support_face | oracle_future_gap | 0.539723 | 0.582407 | -0.042684 |
| ANCESTOR\|support_face | oracle_future_level_shift | 0.518313 | 0.582407 | -0.064094 |
| ANCESTOR\|delayed_face | oracle_future_spike | 0.513517 | 0.666929 | -0.153412 |
| ANCESTOR\|delayed_face | oracle_future_gap | 0.606145 | 0.666929 | -0.060784 |
| ANCESTOR\|delayed_face | oracle_future_level_shift | 0.540924 | 0.666929 | -0.126005 |
| W1_hampel_filter\|support_face | oracle_future_spike | 0.564630 | 0.638184 | -0.073554 |
| W1_hampel_filter\|support_face | oracle_future_gap | 0.530700 | 0.638184 | -0.107484 |
| W1_hampel_filter\|support_face | oracle_future_level_shift | 0.502614 | 0.638184 | -0.135570 |
| W1_hampel_filter\|delayed_face | oracle_future_spike | 0.568819 | 0.623305 | -0.054486 |
| W1_hampel_filter\|delayed_face | oracle_future_gap | 0.501054 | 0.623305 | -0.122251 |
| W1_hampel_filter\|delayed_face | oracle_future_level_shift | 0.548287 | 0.623305 | -0.075018 |
| W2_pmc_then_outlier_mad\|support_face | oracle_future_spike | 0.559722 | 0.650023 | -0.090301 |
| W2_pmc_then_outlier_mad\|support_face | oracle_future_gap | 0.532407 | 0.650023 | -0.117616 |
| W2_pmc_then_outlier_mad\|support_face | oracle_future_level_shift | 0.524907 | 0.650023 | -0.125116 |
| W2_pmc_then_outlier_mad\|delayed_face | oracle_future_spike | 0.549275 | 0.683459 | -0.134184 |
| W2_pmc_then_outlier_mad\|delayed_face | oracle_future_gap | 0.504309 | 0.683459 | -0.179150 |
| W2_pmc_then_outlier_mad\|delayed_face | oracle_future_level_shift | 0.525990 | 0.683459 | -0.157469 |
| W3_outlier_mad_then_pmc\|support_face | oracle_future_spike | 0.536990 | 0.582407 | -0.045417 |
| W3_outlier_mad_then_pmc\|support_face | oracle_future_gap | 0.539723 | 0.582407 | -0.042684 |
| W3_outlier_mad_then_pmc\|support_face | oracle_future_level_shift | 0.518313 | 0.582407 | -0.064094 |
| W3_outlier_mad_then_pmc\|delayed_face | oracle_future_spike | 0.513517 | 0.666929 | -0.153412 |
| W3_outlier_mad_then_pmc\|delayed_face | oracle_future_gap | 0.606145 | 0.666929 | -0.060784 |
| W3_outlier_mad_then_pmc\|delayed_face | oracle_future_level_shift | 0.540924 | 0.666929 | -0.126005 |

Structural verdict: **NO_FUTURE_ADVANTAGE_DETECTED**

## transport_flip

| program | units | flip rate (all) | flip rate (support helped) | best visible | LODO mean | verdict | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR | 420 | 0.219048 | 0.316151 | gap_recurrence | 0.548406 | PATTERN_WEAK | best LODO mean 0.548406 is below 0.60, but the folds disagree on sign, which the frozen rule reads as WEAK |
| W1_hampel_filter | 420 | 0.207143 | 0.450777 | level_excursion_score | 0.527003 | PATTERN_WEAK | best LODO mean 0.527003 is below 0.60, but the folds disagree on sign, which the frozen rule reads as WEAK |
| W2_pmc_then_outlier_mad | 420 | 0.214286 | 0.334572 | missing_fraction | 0.559865 | PATTERN_WEAK | best LODO mean 0.559865 is below 0.60, but the folds disagree on sign, which the frozen rule reads as WEAK |
| W3_outlier_mad_then_pmc | 420 | 0.219048 | 0.316151 | gap_recurrence | 0.548406 | PATTERN_WEAK | best LODO mean 0.548406 is below 0.60, but the folds disagree on sign, which the frozen rule reads as WEAK |
| CAND_outlier_iqr({})>hampel_filter({}) | 240 | 0.229167 | 0.371622 | spike_head_count | 0.607093 | PATTERN_WEAK | best LODO mean in [0.60, 0.70) |
| CAND_outlier_iqr({})>winsorize({}) | 240 | 0.225000 | 0.312139 | period_reliability | 0.546944 | PATTERN_UNINFORMATIVE | best LODO mean below 0.60 |

## Not computable as defined

- `gap_period_aligned`: the definition asks for the share of context gaps whose within-day phase (period=24) coincides with the horizon's phases, but the horizon is 48 = 2 x 24 steps and therefore covers every one of the 24 phases; the quantity is identically 1.0 wherever a gap exists and 0/0 otherwise, so it separates nothing.  Reported as defined rather than swapped for another definition.
