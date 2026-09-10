# R4D-B per-channel three-cell -- machine reading

Evidence class: MECHANISM / INSTRUMENT; development only; not a capability or generalisation claim

Boundary: physical_fits=240 / cap 260 (target 240), llm=0, held_out_reads=0, max_time_index_read=3911 (frontier 4056).
Rows: 1680 (1680 OK); not-fittable series: 0; not-evaluable rows: 0.

## 1. baseline quality (per-channel Static vs pooled Static)

| scope | n | pc Static mean | pooled Static mean | delta mean | delta median | share pc worse | p10 | p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| support_face | 420 | 0.992580068 | 1.462312899 | -0.469732832 | -0.255676629 | 0.280952381 | -1.603342482 | 0.364572482 |
| delayed_face | 420 | 1.246216203 | 1.527425461 | -0.281209258 | -0.184632772 | 0.380952381 | -1.281050049 | 0.63105145 |
| both_faces | 840 | 1.119398136 | 1.49486918 | -0.375471045 | -0.223985062 | 0.330952381 | -1.400571284 | 0.527999834 |

consumer_choice_in_scope: False

## 2. three-cell decomposition (pc vs pooled)

| consumer | program | scope | n | route share | severe n | severe route-worst share | rho(route,total) | rho(ctx,total) | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| per_channel | ANCESTOR | support_face | 420 | 0.727289 | 3 | 0.666666667 | 0.755415357 | 0.433600067 | ROUTE_DOMINANT |
| per_channel | ANCESTOR | delayed_face | 420 | 0.791368246 | 9 | 1.0 | 0.879168021 | 0.234632563 | ROUTE_DOMINANT |
| per_channel | ANCESTOR | both_faces | 840 | 0.753710679 | 12 | 0.916666667 | 0.816480533 | 0.352889693 | ROUTE_DOMINANT |
| per_channel | W2_pmc_then_outlier_mad | support_face | 420 | 0.73319472 | 8 | 0.875 | 0.880168216 | 0.310515432 | ROUTE_DOMINANT |
| per_channel | W2_pmc_then_outlier_mad | delayed_face | 420 | 0.732890161 | 6 | 0.833333333 | 0.831814246 | 0.413020756 | ROUTE_DOMINANT |
| per_channel | W2_pmc_then_outlier_mad | both_faces | 840 | 0.733052175 | 14 | 0.857142857 | 0.856912228 | 0.360796139 | ROUTE_DOMINANT |
| pooled_R4D_A | ANCESTOR | support_face | 420 | 0.768009429 | 28 | 0.964285714 | 0.901856263 | 0.21428704 | ROUTE_DOMINANT |
| pooled_R4D_A | ANCESTOR | delayed_face | 420 | 0.80376688 | 31 | 0.967741935 | 0.937469357 | 0.245078368 | ROUTE_DOMINANT |
| pooled_R4D_A | ANCESTOR | both_faces | 840 | 0.784907566 | 59 | 0.966101695 | 0.920054511 | 0.228908554 | ROUTE_DOMINANT |
| pooled_R4D_A | W2_pmc_then_outlier_mad | support_face | 420 | 0.714483157 | 60 | 0.65 | 0.867865464 | 0.29190122 | ROUTE_DOMINANT |
| pooled_R4D_A | W2_pmc_then_outlier_mad | delayed_face | 420 | 0.703445421 | 75 | 0.786666667 | 0.846592425 | 0.255882915 | ROUTE_DOMINANT |
| pooled_R4D_A | W2_pmc_then_outlier_mad | both_faces | 840 | 0.709106337 | 135 | 0.725925926 | 0.857872308 | 0.273619555 | ROUTE_DOMINANT |

## 3. persistence (total channel; R4A-b vocabulary)

| consumer | program | cross-face Spearman median | face | position share | uid share (null) | residual | verdict |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| per_channel | ANCESTOR | 0.073282443 | support_face | 0.105128165 | 0.24838322 (0.18345726) | 0.646488616 | MOSTLY_NOISE |
| per_channel | ANCESTOR | 0.073282443 | delayed_face | 0.112341201 | 0.230573763 (0.18345726) | 0.657085036 | MOSTLY_NOISE |
| per_channel | W2_pmc_then_outlier_mad | 0.144360902 | support_face | 0.137983029 | 0.317412734 (0.18345726) | 0.544604238 | MIXED |
| per_channel | W2_pmc_then_outlier_mad | 0.144360902 | delayed_face | 0.119426818 | 0.307846327 (0.18345726) | 0.572726855 | MIXED |
| pooled_R4D_A | ANCESTOR | 0.034586466 | support_face | 0.102972664 | 0.201908006 (0.18345726) | 0.69511933 | MOSTLY_NOISE |
| pooled_R4D_A | ANCESTOR | 0.034586466 | delayed_face | 0.105067789 | 0.215059178 (0.18345726) | 0.679873033 | MOSTLY_NOISE |
| pooled_R4D_A | W2_pmc_then_outlier_mad | 0.142857143 | support_face | 0.104049096 | 0.200145864 (0.18345726) | 0.695805041 | MOSTLY_NOISE |
| pooled_R4D_A | W2_pmc_then_outlier_mad | 0.142857143 | delayed_face | 0.131501076 | 0.212886436 (0.18345726) | 0.655612488 | MOSTLY_NOISE |

## 4. identifiability (best feature per program x channel x face group) and selection

| consumer | cell | n | best feature | lodo mean | lodo min | verdict | full-support INFORMATIVE |
| --- | --- | ---: | --- | ---: | ---: | --- | --- |
| per_channel | ANCESTOR|route|support_face | 420 | srv_donor_dispersion|severe_harm | 0.837962963 | 0.805555556 | PATTERN_INFORMATIVE | [] |
| per_channel | ANCESTOR|route|delayed_face | 420 | estimated_region_start_fraction|severe_harm | 0.752765178 | 0.688920455 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|route|both_faces | 840 | srv_period_filled_points|severe_harm | 0.728151264 | 0.662909091 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|ctx|support_face | 184 | outlier_region_end_fraction|helped | 0.590123524 | 0.472107438 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|ctx|delayed_face | 151 | estimated_region_start_fraction|helped | 0.683219518 | 0.590909091 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|ctx|both_faces | 335 | srv_donor_dispersion|helped | 0.600891957 | 0.572774529 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|total|support_face | 420 | spike_peak_over_tail_sd|severe_harm | 0.780797101 | 0.561594203 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|total|delayed_face | 420 | estimated_region_start_fraction|severe_harm | 0.752765178 | 0.688920455 | PATTERN_WEAK | [] |
| per_channel | ANCESTOR|total|both_faces | 840 | srv_tail48_period_filled|severe_harm | 0.755963892 | 0.574545455 | PATTERN_WEAK | [] |
| per_channel | W2_pmc_then_outlier_mad|route|support_face | 420 | srv_period_filled_frac|severe_harm | 0.678787367 | 0.432432432 | PATTERN_WEAK | [] |
| per_channel | W2_pmc_then_outlier_mad|route|delayed_face | 420 | gap_tail_fraction|severe_harm | 0.815627805 | 0.703651685 | PATTERN_INFORMATIVE | [] |
| per_channel | W2_pmc_then_outlier_mad|route|both_faces | 840 | spike_recency|severe_harm | 0.627817079 | 0.564347826 | PATTERN_WEAK | [] |
| per_channel | W2_pmc_then_outlier_mad|ctx|support_face | 345 | spike_peak_over_tail_sd|helped | 0.596642127 | 0.550826699 | PATTERN_UNINFORMATIVE | [] |
| per_channel | W2_pmc_then_outlier_mad|ctx|delayed_face | 378 | gap_longest_run_steps|helped | 0.584010986 | 0.481060606 | PATTERN_WEAK | [] |
| per_channel | W2_pmc_then_outlier_mad|ctx|both_faces | 723 | missing_fraction|severe_harm | 0.937120839 | 0.934210526 | PATTERN_INFORMATIVE | [] |
| per_channel | W2_pmc_then_outlier_mad|total|support_face | 420 | srv_tail48_period_filled|severe_harm | 0.722285602 | 0.48 | PATTERN_WEAK | [] |
| per_channel | W2_pmc_then_outlier_mad|total|delayed_face | 420 | gap_longest_run_steps|severe_harm | 0.857365282 | 0.717105263 | PATTERN_INFORMATIVE | [] |
| per_channel | W2_pmc_then_outlier_mad|total|both_faces | 840 | missing_fraction|severe_harm | 0.81722956 | 0.668128655 | PATTERN_WEAK | [] |
| pooled_R4D_A | ANCESTOR|route|support_face | 420 | srv_donor_dispersion|severe_harm | 0.617481942 | 0.568627451 | PATTERN_WEAK | [] |
| pooled_R4D_A | ANCESTOR|route|delayed_face | 420 | srv_donor_dispersion|helped | 0.579398545 | 0.453877791 | PATTERN_WEAK | [] |
| pooled_R4D_A | ANCESTOR|route|both_faces | 840 | local_robust_z_peak|severe_harm | 0.53426676 | 0.467301587 | PATTERN_WEAK | [] |
| pooled_R4D_A | ANCESTOR|ctx|support_face | 184 | missing_fraction|severe_harm | 0.943049978 | 0.898148148 | PATTERN_INFORMATIVE | [] |
| pooled_R4D_A | ANCESTOR|ctx|delayed_face | 151 | estimated_region_start_fraction|helped | 0.578913459 | 0.527777778 | PATTERN_UNINFORMATIVE | [] |
| pooled_R4D_A | ANCESTOR|ctx|both_faces | 335 | missing_fraction|severe_harm | 0.909801659 | 0.904040404 | PATTERN_INFORMATIVE | [] |
| pooled_R4D_A | ANCESTOR|total|support_face | 420 | local_robust_z_peak|severe_harm | 0.633724765 | 0.525021204 | PATTERN_WEAK | [] |
| pooled_R4D_A | ANCESTOR|total|delayed_face | 420 | local_robust_z_peak|severe_harm | 0.676036124 | 0.59912854 | PATTERN_WEAK | [] |
| pooled_R4D_A | ANCESTOR|total|both_faces | 840 | local_robust_z_peak|severe_harm | 0.650471558 | 0.587191358 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|route|support_face | 420 | spike_peak_over_tail_sd|severe_harm | 0.616312113 | 0.489583333 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|route|delayed_face | 420 | spike_peak_over_tail_sd|severe_harm | 0.608351619 | 0.544 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|route|both_faces | 840 | spike_peak_over_tail_sd|severe_harm | 0.616579857 | 0.538033395 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|ctx|support_face | 345 | srv_period_filled_points|severe_harm | 0.731431773 | 0.646428571 | PATTERN_INFORMATIVE | ['local_robust_z_peak|helped'] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|ctx|delayed_face | 378 | local_robust_z_peak|severe_harm | 0.826536249 | 0.748168498 | PATTERN_INFORMATIVE | ['local_robust_z_peak|severe_harm'] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|ctx|both_faces | 723 | local_robust_z_peak|severe_harm | 0.778596946 | 0.676103368 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|total|support_face | 420 | spike_peak_over_tail_sd|severe_harm | 0.64840367 | 0.543599258 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|total|delayed_face | 420 | local_robust_z_peak|severe_harm | 0.656777469 | 0.541125541 | PATTERN_WEAK | [] |
| pooled_R4D_A | W2_pmc_then_outlier_mad|total|both_faces | 840 | spike_peak_over_tail_sd|severe_harm | 0.655730956 | 0.596 | PATTERN_WEAK | [] |

| consumer | program|group | selection verdict | detail |
| --- | --- | --- | --- |
| per_channel | ANCESTOR|support_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "estimated_level_offset", "best_reading_summary": {"feature": "estimated_level_offset", "n_folds": 4, "folds_beating_both_fixed": 2, "mean_vs_always_p": 0.001920475, "mean_vs_always_identity": 0.069231435, " |
| per_channel | ANCESTOR|delayed_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "local_robust_z_peak", "best_reading_summary": {"feature": "local_robust_z_peak", "n_folds": 4, "folds_beating_both_fixed": 2, "mean_vs_always_p": 0.010369415, "mean_vs_always_identity": 0.024110883, "bindin |
| per_channel | ANCESTOR|both_faces | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 840, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "estimated_level_offset", "best_reading_summary": {"feature": "estimated_level_offset", "n_folds": 4, "folds_beating_both_fixed": 2, "mean_vs_always_p": 0.0005316, "mean_vs_always_identity": 0.041057814, "bi |
| per_channel | W2_pmc_then_outlier_mad|support_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "gap_tail_fraction", "best_reading_summary": {"feature": "gap_tail_fraction", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.000920409, "mean_vs_always_identity": 0.089575011, "binding_me |
| per_channel | W2_pmc_then_outlier_mad|delayed_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "estimated_level_offset", "best_reading_summary": {"feature": "estimated_level_offset", "n_folds": 4, "folds_beating_both_fixed": 3, "mean_vs_always_p": 9.5315e-05, "mean_vs_always_identity": 0.087112637, "b |
| per_channel | W2_pmc_then_outlier_mad|both_faces | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 840, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "spike_recurrence", "best_reading_summary": {"feature": "spike_recurrence", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.0, "mean_vs_always_identity": 0.087835962, "binding_mean_margin" |
| pooled_R4D_A | ANCESTOR|support_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "spike_recurrence", "best_reading_summary": {"feature": "spike_recurrence", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.0, "mean_vs_always_identity": 0.26777252, "binding_mean_margin": |
| pooled_R4D_A | ANCESTOR|delayed_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "gap_tail_fraction", "best_reading_summary": {"feature": "gap_tail_fraction", "n_folds": 4, "folds_beating_both_fixed": 1, "mean_vs_always_p": 0.000548186, "mean_vs_always_identity": 0.231404171, "binding_me |
| pooled_R4D_A | ANCESTOR|both_faces | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 840, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "spike_recurrence", "best_reading_summary": {"feature": "spike_recurrence", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.0, "mean_vs_always_identity": 0.249314253, "binding_mean_margin" |
| pooled_R4D_A | W2_pmc_then_outlier_mad|support_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "spike_recurrence", "best_reading_summary": {"feature": "spike_recurrence", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.0, "mean_vs_always_identity": 0.267159356, "binding_mean_margin" |
| pooled_R4D_A | W2_pmc_then_outlier_mad|delayed_face | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 420, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "spike_tail_count", "best_reading_summary": {"feature": "spike_tail_count", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.0, "mean_vs_always_identity": 0.23923569, "binding_mean_margin": |
| pooled_R4D_A | W2_pmc_then_outlier_mad|both_faces | FIXED_CHOICE_NOT_BEATEN | {"n_rows": 840, "features_with_SELECTION_BEATS_FIXED": [], "best_feature_by_binding_margin": "spike_recurrence", "best_reading_summary": {"feature": "spike_recurrence", "n_folds": 4, "folds_beating_both_fixed": 0, "mean_vs_always_p": 0.0, "mean_vs_always_identity": 0.253197523, "binding_mean_margin" |

## 5. harm shape (pc vs pooled)

| program|face | pc gain | pooled gain | pc hf | pooled hf | pc msh | pooled msh | shape | gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ANCESTOR|support_face | 0.072310116 | 0.245427266 | 0.269047619 | 0.307142857 | 0.365648905 | 1.180821209 | CONSUMER_SHAPES_HARM | GAIN_LOST |
| ANCESTOR|delayed_face | 0.017838252 | 0.224645329 | 0.330952381 | 0.316666667 | 0.769697476 | 2.02152086 | CONSUMER_SHAPES_HARM | GAIN_LOST |
| W2_pmc_then_outlier_mad|support_face | 0.116995598 | 0.272317898 | 0.35952381 | 0.35952381 | 1.186306151 | 1.826279038 | CONSUMER_SHAPES_HARM | GAIN_LOST |
| W2_pmc_then_outlier_mad|delayed_face | 0.101825685 | 0.185002771 | 0.364285714 | 0.357142857 | 0.609590461 | 1.848863568 | CONSUMER_SHAPES_HARM | GAIN_LOST |
