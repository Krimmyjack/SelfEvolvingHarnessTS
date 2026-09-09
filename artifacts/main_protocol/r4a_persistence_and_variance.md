# R4A-b persistence and variance -- machine reading

Evidence class: MECHANISM / NEGATIVE-capable; not a capability claim

Boundary: fits=0, llm=0, held_out_reads=0, max_time_index_read=3863 (frontier 4056), horizon_reads=0.
Loader: `evaluation.main_protocol_p4.preflight_natural_gap_variant.load_variant`, NaN=503712.
Rows built 4400; after dropping the `W3_outlier_mad_then_pmc` alias 3560; 80 unique uids. Dropped: 40 degenerate-uid rows, 8 verifier-failed entries, 33 out-of-scope entries.

## Verdicts per (program, face)

| cell | rows | cross-face Spearman median | position share | uid share (weighted) | uid null (weighted) | residual share | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | 420 | 0.034586 | 0.102973 | 0.201908 | 0.183457 | 0.705300 | MOSTLY_NOISE |
| ANCESTOR\|delayed_face | 420 | 0.034586 | 0.105068 | 0.215059 | 0.183457 | 0.682188 | MOSTLY_NOISE |
| W1_hampel_filter\|support_face | 420 | -0.045113 | 0.060364 | 0.195976 | 0.183457 | 0.733260 | MOSTLY_NOISE |
| W1_hampel_filter\|delayed_face | 420 | -0.045113 | 0.144622 | 0.226196 | 0.183457 | 0.630583 | MOSTLY_NOISE |
| W2_pmc_then_outlier_mad\|support_face | 420 | 0.142857 | 0.104049 | 0.200146 | 0.183457 | 0.710058 | MOSTLY_NOISE |
| W2_pmc_then_outlier_mad\|delayed_face | 420 | 0.142857 | 0.131501 | 0.212886 | 0.183457 | 0.710000 | MOSTLY_NOISE |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face (ref) | 240 | 0.027820 | 0.084711 | 0.152354 | 0.242058 | 0.776320 | MOSTLY_NOISE |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face (ref) | 280 | 0.027820 | 0.143670 | 0.202260 | 0.206841 | 0.629343 | MOSTLY_NOISE |
| CAND_outlier_iqr({})>winsorize({})\|support_face (ref) | 240 | 0.032331 | 0.168711 | 0.202186 | 0.242058 | 0.628635 | MOSTLY_NOISE |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face (ref) | 280 | 0.032331 | 0.091939 | 0.264730 | 0.206841 | 0.649080 | MOSTLY_NOISE |

## Reading 1: cross-face persistence (per program)

| program | pairs | positions | Spearman q25/median/q75 | pooled Spearman | pooled Pearson | P(delayed severe \| support severe) | delayed base rate | helped kappa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR | 420 | 21 | -0.103759 / 0.034586 / 0.133835 | 0.065363 | 0.363516 | 0.107143 | 0.073810 | 0.001664 |
| W1_hampel_filter | 420 | 21 | -0.294737 / -0.045113 / 0.172932 | 0.017177 | 0.281393 | 0.325203 | 0.235714 | 0.072858 |
| W2_pmc_then_outlier_mad | 420 | 21 | -0.139850 / 0.142857 / 0.260150 | 0.108911 | 0.286426 | 0.283333 | 0.178571 | 0.062870 |
| CAND_outlier_iqr({})>hampel_filter({}) | 240 | 12 | -0.120677 / 0.027820 / 0.169173 | 0.076177 | 0.248706 | 0.212121 | 0.141667 | 0.041337 |
| CAND_outlier_iqr({})>winsorize({}) | 240 | 12 | -0.166541 / 0.032331 / 0.184586 | 0.039428 | 0.228161 | 0.058824 | 0.079167 | -0.013208 |

## Reading 2: variance shares inside each cohort

`null` is (levels-1)/(rows-1), the share the factor would take under exchangeable noise. It is a diagnostic and changes no frozen threshold.

| cell | cohort | rows | positions | uid share | uid null | position share | position null | residual | shares sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | [0:40] | 140 | 7 | 0.126387 | 0.136691 | 0.113841 | 0.043165 | 0.759772 | 1.000000 |
| ANCESTOR\|support_face | [120:160] | 60 | 3 | 0.583465 | 0.322034 | 0.071920 | 0.033898 | 0.344615 | 1.000000 |
| ANCESTOR\|support_face | [40:80] | 180 | 9 | 0.076123 | 0.106145 | 0.098938 | 0.044693 | 0.824939 | 1.000000 |
| ANCESTOR\|support_face | [80:120] | 40 | 2 | 0.459927 | 0.487179 | 0.022769 | 0.025641 | 0.517304 | 1.000000 |
| ANCESTOR\|delayed_face | [0:40] | 140 | 7 | 0.175995 | 0.136691 | 0.130036 | 0.043165 | 0.693969 | 1.000000 |
| ANCESTOR\|delayed_face | [120:160] | 60 | 3 | 0.544305 | 0.322034 | 0.033161 | 0.033898 | 0.422535 | 1.000000 |
| ANCESTOR\|delayed_face | [40:80] | 180 | 9 | 0.084825 | 0.106145 | 0.100066 | 0.044693 | 0.815109 | 1.000000 |
| ANCESTOR\|delayed_face | [80:120] | 40 | 2 | 0.443968 | 0.487179 | 0.123742 | 0.025641 | 0.432290 | 1.000000 |
| W1_hampel_filter\|support_face | [0:40] | 140 | 7 | 0.191718 | 0.136691 | 0.070746 | 0.043165 | 0.737535 | 1.000000 |
| W1_hampel_filter\|support_face | [120:160] | 60 | 3 | 0.495799 | 0.322034 | 0.019406 | 0.033898 | 0.484795 | 1.000000 |
| W1_hampel_filter\|support_face | [40:80] | 180 | 9 | 0.040532 | 0.106145 | 0.098943 | 0.044693 | 0.860524 | 1.000000 |
| W1_hampel_filter\|support_face | [80:120] | 40 | 2 | 0.460643 | 0.487179 | 0.021059 | 0.025641 | 0.518298 | 1.000000 |
| W1_hampel_filter\|delayed_face | [0:40] | 140 | 7 | 0.228750 | 0.136691 | 0.092700 | 0.043165 | 0.678550 | 1.000000 |
| W1_hampel_filter\|delayed_face | [120:160] | 60 | 3 | 0.453312 | 0.322034 | 0.068301 | 0.033898 | 0.478387 | 1.000000 |
| W1_hampel_filter\|delayed_face | [40:80] | 180 | 9 | 0.096423 | 0.106145 | 0.215341 | 0.044693 | 0.688236 | 1.000000 |
| W1_hampel_filter\|delayed_face | [80:120] | 40 | 2 | 0.460565 | 0.487179 | 0.107876 | 0.025641 | 0.431559 | 1.000000 |
| W2_pmc_then_outlier_mad\|support_face | [0:40] | 140 | 7 | 0.186572 | 0.136691 | 0.096463 | 0.043165 | 0.716965 | 1.000000 |
| W2_pmc_then_outlier_mad\|support_face | [120:160] | 60 | 3 | 0.606879 | 0.322034 | 0.003992 | 0.033898 | 0.389129 | 1.000000 |
| W2_pmc_then_outlier_mad\|support_face | [40:80] | 180 | 9 | 0.040230 | 0.106145 | 0.130483 | 0.044693 | 0.829287 | 1.000000 |
| W2_pmc_then_outlier_mad\|support_face | [80:120] | 40 | 2 | 0.357175 | 0.487179 | 0.012075 | 0.025641 | 0.630750 | 1.000000 |
| W2_pmc_then_outlier_mad\|delayed_face | [0:40] | 140 | 7 | 0.183063 | 0.136691 | 0.080313 | 0.043165 | 0.736624 | 1.000000 |
| W2_pmc_then_outlier_mad\|delayed_face | [120:160] | 60 | 3 | 0.533461 | 0.322034 | 0.038667 | 0.033898 | 0.427872 | 1.000000 |
| W2_pmc_then_outlier_mad\|delayed_face | [40:80] | 180 | 9 | 0.078874 | 0.106145 | 0.085507 | 0.044693 | 0.835619 | 1.000000 |
| W2_pmc_then_outlier_mad\|delayed_face | [80:120] | 40 | 2 | 0.439463 | 0.487179 | 0.085813 | 0.025641 | 0.474723 | 1.000000 |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face | [120:160] | 40 | 2 | 0.286955 | 0.487179 | 0.134077 | 0.025641 | 0.578968 | 1.000000 |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face | [40:80] | 160 | 8 | 0.073831 | 0.119497 | 0.069967 | 0.044025 | 0.856203 | 1.000000 |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face | [80:120] | 40 | 2 | 0.331848 | 0.487179 | 0.014008 | 0.025641 | 0.654143 | 1.000000 |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face | [120:160] | 60 | 3 | 0.463076 | 0.322034 | 0.071474 | 0.033898 | 0.465449 | 1.000000 |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face | [40:80] | 180 | 9 | 0.071027 | 0.106145 | 0.202385 | 0.044693 | 0.726588 | 1.000000 |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face | [80:120] | 40 | 2 | 0.401582 | 0.487179 | 0.160833 | 0.025641 | 0.437585 | 1.000000 |
| CAND_outlier_iqr({})>winsorize({})\|support_face | [120:160] | 40 | 2 | 0.499424 | 0.487179 | 0.079987 | 0.025641 | 0.420589 | 1.000000 |
| CAND_outlier_iqr({})>winsorize({})\|support_face | [40:80] | 160 | 8 | 0.088231 | 0.119497 | 0.228525 | 0.044025 | 0.683244 | 1.000000 |
| CAND_outlier_iqr({})>winsorize({})\|support_face | [80:120] | 40 | 2 | 0.360772 | 0.487179 | 0.020985 | 0.025641 | 0.618244 | 1.000000 |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face | [120:160] | 60 | 3 | 0.613397 | 0.322034 | 0.040798 | 0.033898 | 0.345805 | 1.000000 |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face | [40:80] | 180 | 9 | 0.087475 | 0.106145 | 0.096742 | 0.044693 | 0.815784 | 1.000000 |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face | [80:120] | 40 | 2 | 0.539380 | 0.487179 | 0.106791 | 0.025641 | 0.353829 | 1.000000 |

## Reading 3: z_peak, strength or direction

| cell | Spearman with \|g\| | Spearman with g | pooled AUC severe_harm (raw) | direction | reads strength not direction |
| --- | --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | -0.107487 | -0.008266 | 0.436316 | higher z_peak -> less severe harm | False |
| ANCESTOR\|delayed_face | -0.077254 | 0.091710 | 0.333071 | higher z_peak -> less severe harm | False |
| W1_hampel_filter\|support_face | -0.230827 | 0.105086 | 0.361816 | higher z_peak -> less severe harm | True |
| W1_hampel_filter\|delayed_face | -0.090302 | 0.137043 | 0.376695 | higher z_peak -> less severe harm | False |
| W2_pmc_then_outlier_mad\|support_face | -0.194563 | 0.074820 | 0.349977 | higher z_peak -> less severe harm | True |
| W2_pmc_then_outlier_mad\|delayed_face | -0.108238 | 0.148804 | 0.316541 | higher z_peak -> less severe harm | False |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face | -0.069299 | 0.049257 | 0.439540 | higher z_peak -> less severe harm | False |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face | -0.013428 | 0.087560 | 0.414238 | higher z_peak -> less severe harm | False |
| CAND_outlier_iqr({})>winsorize({})\|support_face | -0.058830 | 0.029907 | 0.448430 | higher z_peak -> less severe harm | False |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face | -0.060003 | 0.093323 | 0.260536 | higher z_peak -> less severe harm | False |

## Reading 4: unit-level condition against the visible vocabulary

All four AUC columns are pooled and in-sample. The unit-level columns are a ceiling computed with the answer in hand and are **not deployable**.

| cell | unit-level AUC helped | best visible AUC helped | unit-level AUC severe_harm | best visible AUC severe_harm | best visible feature (severe_harm) |
| --- | --- | --- | --- | --- | --- |
| ANCESTOR\|support_face | 0.625736 | 0.544460 | 0.620262 | 0.563684 | local_robust_z_peak |
| ANCESTOR\|delayed_face | 0.640421 | 0.564931 | 0.591218 | 0.666929 | local_robust_z_peak |
| W1_hampel_filter\|support_face | 0.637408 | 0.517644 | 0.619898 | 0.638184 | local_robust_z_peak |
| W1_hampel_filter\|delayed_face | 0.676481 | 0.564128 | 0.657966 | 0.623305 | local_robust_z_peak |
| W2_pmc_then_outlier_mad\|support_face | 0.600446 | 0.599289 | 0.617593 | 0.650023 | local_robust_z_peak |
| W2_pmc_then_outlier_mad\|delayed_face | 0.660000 | 0.561037 | 0.649952 | 0.683459 | local_robust_z_peak |
| CAND_outlier_iqr({})>hampel_filter({})\|support_face | 0.624853 | 0.517443 | 0.721051 | 0.560460 | local_robust_z_peak |
| CAND_outlier_iqr({})>hampel_filter({})\|delayed_face | 0.671131 | 0.588276 | 0.620695 | 0.592287 | estimated_region_start_fraction |
| CAND_outlier_iqr({})>winsorize({})\|support_face | 0.695842 | 0.569666 | 0.660907 | 0.604194 | estimated_region_start_fraction |
| CAND_outlier_iqr({})>winsorize({})\|delayed_face | 0.620262 | 0.577624 | 0.727869 | 0.739464 | local_robust_z_peak |

## Verdict rules (frozen before any number was seen)

- `SERIES_LEVEL_CONDITION`: uid share >= 0.30 and cross-face Spearman median >= 0.50
- `COHORT_LEVEL_CONDITION`: position share >= 0.50 and uid share < 0.30
- `MOSTLY_NOISE`: cross-face Spearman median < 0.30 and uid share < 0.30 and position share < 0.50
- `MIXED`: none of the above
- `note`: the cross-face Spearman median is a program-level quantity by construction -- one pairing serves both faces -- so the two faces of a program share it

Alias check: `W3_outlier_mad_then_pmc` compared on 42 entries, 42 numerically identical to ANCESTOR, max |diff| = 0.000000; skipped from every table; its readings would be ANCESTOR's to the last digit.
