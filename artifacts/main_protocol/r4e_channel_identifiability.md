# R4E per-channel identifiability -- machine reading

Evidence class: MECHANISM / NEGATIVE-capable; development reading; not a capability claim

Boundary: fits=0, llm=0, held_out_reads=0, horizon_reads=0, max_time_index_read=3863 (frontier 4056), existing_files_edited=0, new_sha_or_hash=0.

Rows: 1680 (program, position, face, uid); 80 unique uids over 21 positions and blocks [0:40], [120:160], [40:80], [80:120]; statuses OK, dropped for status 0.  ctx channel keeps the 1058 rows whose serving window was modified and drops the 622 whose ctx is exactly zero.

Features: 27 = 12 vocabulary numbers + 9 mechanism quantities + 6 serving action conditions.  Serving cross-check failed 0 of 840 contexts.

## Reading 1 -- per-channel LODO identifiability

| program | channel | face | rows | helped | severe | best (feature\|target) | LODO mean | LODO min | folds | clears every fold (of which on 4/4 folds) | verdict |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| ANCESTOR | route | support_face | 420 | 274 | 34 | srv_donor_dispersion\|severe_harm | 0.617482 | 0.568627 | [0:40]=0.592308, [120:160]=0.714286, [40:80]=0.594707, [80:120]=0.568627 | 0 (0) | PATTERN_WEAK |
| ANCESTOR | route | delayed_face | 420 | 278 | 38 | srv_donor_dispersion\|helped | 0.579399 | 0.453878 | [0:40]=0.453878, [120:160]=0.725000, [40:80]=0.558124, [80:120]=0.580592 | 0 (0) | PATTERN_WEAK |
| ANCESTOR | route | both_faces | 840 | 552 | 72 | local_robust_z_peak\|severe_harm | 0.534267 | 0.467302 | [0:40]=0.569974, [120:160]=0.467302, [40:80]=0.549444, [80:120]=0.550347 | 0 (0) | PATTERN_WEAK |
| ANCESTOR | ctx | support_face | 184 | 126 | 2 | missing_fraction\|severe_harm | 0.943050 | 0.898148 | [0:40]=0.898148, [120:160]=n/a, [40:80]=0.987952, [80:120]=n/a | 11 (0) | PATTERN_INFORMATIVE |
| ANCESTOR | ctx | delayed_face | 151 | 114 | 1 | estimated_region_start_fraction\|helped | 0.578913 | 0.527778 | [0:40]=0.527778, [120:160]=0.658730, [40:80]=0.560964, [80:120]=0.568182 | 0 (0) | PATTERN_UNINFORMATIVE |
| ANCESTOR | ctx | both_faces | 335 | 240 | 3 | missing_fraction\|severe_harm | 0.909802 | 0.904040 | [0:40]=0.904040, [120:160]=n/a, [40:80]=0.915563, [80:120]=n/a | 10 (0) | PATTERN_INFORMATIVE |
| ANCESTOR | total | support_face | 420 | 291 | 28 | local_robust_z_peak\|severe_harm | 0.633725 | 0.525021 | [0:40]=0.525021, [120:160]=0.596491, [40:80]=0.544966, [80:120]=0.868421 | 0 (0) | PATTERN_WEAK |
| ANCESTOR | total | delayed_face | 420 | 287 | 31 | local_robust_z_peak\|severe_harm | 0.676036 | 0.599129 | [0:40]=0.717803, [120:160]=0.599129, [40:80]=0.702528, [80:120]=0.684685 | 0 (0) | PATTERN_WEAK |
| ANCESTOR | total | both_faces | 840 | 578 | 59 | local_robust_z_peak\|severe_harm | 0.650472 | 0.587191 | [0:40]=0.614516, [120:160]=0.587191, [40:80]=0.616179, [80:120]=0.784000 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | route | support_face | 420 | 277 | 50 | spike_peak_over_tail_sd\|severe_harm | 0.616312 | 0.489583 | [0:40]=0.670895, [120:160]=0.489583, [40:80]=0.616598, [80:120]=0.688172 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | route | delayed_face | 420 | 253 | 70 | spike_peak_over_tail_sd\|severe_harm | 0.608352 | 0.544000 | [0:40]=0.718111, [120:160]=0.544000, [40:80]=0.600983, [80:120]=0.570312 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | route | both_faces | 840 | 530 | 120 | spike_peak_over_tail_sd\|severe_harm | 0.616580 | 0.538033 | [0:40]=0.691684, [120:160]=0.538033, [40:80]=0.604482, [80:120]=0.632120 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | ctx | support_face | 345 | 159 | 37 | srv_period_filled_points\|severe_harm | 0.731432 | 0.646429 | [0:40]=0.894547, [120:160]=0.705882, [40:80]=0.678869, [80:120]=0.646429 | 1 (1) | PATTERN_INFORMATIVE |
| W2_pmc_then_outlier_mad | ctx | delayed_face | 378 | 180 | 39 | local_robust_z_peak\|severe_harm | 0.826536 | 0.748168 | [0:40]=0.802768, [120:160]=0.968750, [40:80]=0.748168, [80:120]=0.786458 | 1 (1) | PATTERN_INFORMATIVE |
| W2_pmc_then_outlier_mad | ctx | both_faces | 723 | 339 | 76 | local_robust_z_peak\|severe_harm | 0.778597 | 0.676103 | [0:40]=0.745849, [120:160]=0.943182, [40:80]=0.676103, [80:120]=0.749254 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | total | support_face | 420 | 269 | 60 | spike_peak_over_tail_sd\|severe_harm | 0.648404 | 0.543599 | [0:40]=0.703210, [120:160]=0.543599, [40:80]=0.662823, [80:120]=0.683983 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | total | delayed_face | 420 | 270 | 75 | local_robust_z_peak\|severe_harm | 0.656777 | 0.541126 | [0:40]=0.724267, [120:160]=0.677560, [40:80]=0.684158, [80:120]=0.541126 | 0 (0) | PATTERN_WEAK |
| W2_pmc_then_outlier_mad | total | both_faces | 840 | 539 | 135 | spike_peak_over_tail_sd\|severe_harm | 0.655731 | 0.596000 | [0:40]=0.734056, [120:160]=0.596000, [40:80]=0.644599, [80:120]=0.648268 | 0 (0) | PATTERN_WEAK |

Support of every INFORMATIVE entry (the frozen rule takes the minimum over the folds that have an AUC, so a fold with no positive instance is skipped, not failed):

- `ANCESTOR\|ctx\|support_face` / `spike_head_count\|severe_harm`: LODO mean 0.876116, min 0.842593 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `spike_peak_over_tail_sd\|severe_harm`: LODO mean 0.765172, min 0.759259 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `gap_recurrence\|severe_harm`: LODO mean 0.775100, min 0.716867 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `gap_longest_run_steps\|severe_harm`: LODO mean 0.900881, min 0.898148 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `missing_fraction\|severe_harm`: LODO mean 0.943050, min 0.898148 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `longest_missing_run_fraction\|severe_harm`: LODO mean 0.900881, min 0.898148 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `estimated_region_end_fraction\|severe_harm`: LODO mean 0.862896, min 0.855422 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `outlier_region_end_fraction\|severe_harm`: LODO mean 0.839804, min 0.703704 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `period_reliability\|severe_harm`: LODO mean 0.767236, min 0.728916 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `srv_gap_points\|severe_harm`: LODO mean 0.943050, min 0.898148 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|support_face` / `srv_donor_dispersion\|severe_harm`: LODO mean 0.828078, min 0.777778 over 2 of 4 folds; positives per fold [1, 0, 1, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `spike_peak_over_tail_sd\|severe_harm`: LODO mean 0.836160, min 0.808081 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `gap_tail_fraction\|severe_harm`: LODO mean 0.853393, min 0.716887 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `gap_recurrence\|severe_harm`: LODO mean 0.770888, min 0.718543 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `missing_fraction\|severe_harm`: LODO mean 0.909802, min 0.904040 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `estimated_region_end_fraction\|severe_harm`: LODO mean 0.764299, min 0.727273 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `outlier_region_end_fraction\|severe_harm`: LODO mean 0.719471, min 0.712121 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `period_reliability\|severe_harm`: LODO mean 0.756522, min 0.725166 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `srv_gap_points\|severe_harm`: LODO mean 0.909802, min 0.904040 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `srv_period_filled_points\|severe_harm`: LODO mean 0.849664, min 0.759934 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `ANCESTOR\|ctx\|both_faces` / `srv_fill_divergence\|severe_harm`: LODO mean 0.780143, min 0.716912 over 2 of 4 folds; positives per fold [1, 0, 2, 0]; folds without a positive instance: [120:160], [80:120]
- `W2_pmc_then_outlier_mad\|ctx\|support_face` / `local_robust_z_peak\|helped`: LODO mean 0.720834, min 0.709091 over 4 of 4 folds; positives per fold [55, 25, 63, 16]; folds without a positive instance: none
- `W2_pmc_then_outlier_mad\|ctx\|delayed_face` / `local_robust_z_peak\|severe_harm`: LODO mean 0.826536, min 0.748168 over 4 of 4 folds; positives per fold [17, 2, 14, 6]; folds without a positive instance: none

## Reading 2 -- select on a pattern vs one fixed choice (utility uses total, denominator is the whole served population)

| program | face | best feature | mean vs always-P | mean vs always-identity | folds beating both | harmed vs P | harmed vs identity | worst vs P | always-P utility | oracle (not deployable) | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ANCESTOR | support_face | spike_recurrence | +0.000000 | +0.267773 | 0/4 | +0.00 | +32.25 | +0.000000 | 0.267773 | 0.333617 | FIXED_CHOICE_NOT_BEATEN |
| ANCESTOR | delayed_face | gap_tail_fraction | +0.000548 | +0.231404 | 1/4 | -0.50 | +32.75 | +0.000000 | 0.230856 | 0.313096 | FIXED_CHOICE_NOT_BEATEN |
| ANCESTOR | both_faces | spike_recurrence | +0.000000 | +0.249314 | 0/4 | +0.00 | +65.50 | +0.000000 | 0.249314 | 0.323357 | FIXED_CHOICE_NOT_BEATEN |
| W2_pmc_then_outlier_mad | support_face | spike_recurrence | +0.000000 | +0.267159 | 0/4 | +0.00 | +37.75 | +0.000000 | 0.267159 | 0.397966 | FIXED_CHOICE_NOT_BEATEN |
| W2_pmc_then_outlier_mad | delayed_face | spike_tail_count | +0.000000 | +0.239236 | 0/4 | +0.00 | +37.50 | +0.000000 | 0.239236 | 0.361946 | FIXED_CHOICE_NOT_BEATEN |
| W2_pmc_then_outlier_mad | both_faces | spike_recurrence | +0.000000 | +0.253198 | 0/4 | +0.00 | +75.25 | +0.000000 | 0.253198 | 0.379956 | FIXED_CHOICE_NOT_BEATEN |

Per-fold rules of the best feature, per program (merged faces):

- `ANCESTOR` / [0:40]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.306622, always-P 0.306622, always-identity 0.000000, oracle 0.366513 (n=280, treated 1.000000)
- `ANCESTOR` / [120:160]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.374165, always-P 0.374165, always-identity 0.000000, oracle 0.438603 (n=120, treated 1.000000)
- `ANCESTOR` / [40:80]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.147710, always-P 0.147710, always-identity 0.000000, oracle 0.207548 (n=360, treated 1.000000)
- `ANCESTOR` / [80:120]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.168760, always-P 0.168760, always-identity 0.000000, oracle 0.280762 (n=80, treated 1.000000)
- `W2_pmc_then_outlier_mad` / [0:40]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.393959, always-P 0.393959, always-identity 0.000000, oracle 0.477444 (n=280, treated 1.000000)
- `W2_pmc_then_outlier_mad` / [120:160]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.340823, always-P 0.340823, always-identity 0.000000, oracle 0.455861 (n=120, treated 1.000000)
- `W2_pmc_then_outlier_mad` / [40:80]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.066524, always-P 0.066524, always-identity 0.000000, oracle 0.235260 (n=360, treated 1.000000)
- `W2_pmc_then_outlier_mad` / [80:120]: deploy P iff spike_recurrence >= 0.000000 (train quantile 0.333) -> utility 0.211484, always-P 0.211484, always-identity 0.000000, oracle 0.351258 (n=80, treated 1.000000)

## Reading 3 -- which channel is more identifiable

- `ANCESTOR`: most identifiable channel = **ctx**; route: 0.534267 (local_robust_z_peak|severe_harm, PATTERN_WEAK), ctx: 0.909802 (missing_fraction|severe_harm, PATTERN_INFORMATIVE), total: 0.650472 (local_robust_z_peak|severe_harm, PATTERN_WEAK); route and ctx share the best feature: False
- `W2_pmc_then_outlier_mad`: most identifiable channel = **ctx**; route: 0.616580 (spike_peak_over_tail_sd|severe_harm, PATTERN_WEAK), ctx: 0.778597 (local_robust_z_peak|severe_harm, PATTERN_WEAK), total: 0.655731 (spike_peak_over_tail_sd|severe_harm, PATTERN_WEAK); route and ctx share the best feature: False

## Reading 4 -- cross-check against the published R4A total reading

| cell | R4E best | R4E LODO mean | R4A best | R4A LODO mean | rows R4E/R4A | agrees |
| --- | --- | ---: | --- | ---: | ---: | --- |
| ANCESTOR\|support_face | local_robust_z_peak\|severe_harm | 0.633725 | local_robust_z_peak\|severe_harm | 0.633725 | 420/420 | True |
| ANCESTOR\|delayed_face | local_robust_z_peak\|severe_harm | 0.676036 | local_robust_z_peak\|severe_harm | 0.676036 | 420/420 | True |
| W2_pmc_then_outlier_mad\|support_face | spike_peak_over_tail_sd\|severe_harm | 0.648404 | spike_peak_over_tail_sd\|severe_harm | 0.648404 | 420/420 | True |
| W2_pmc_then_outlier_mad\|delayed_face | local_robust_z_peak\|severe_harm | 0.656777 | local_robust_z_peak\|severe_harm | 0.656777 | 420/420 | True |

All four cells agree: **True**

## Per-feature detail

The full per-feature LODO tables, frozen-bin stumps and per-fold selection rules are in `artifacts/main_protocol/r4e_channel_identifiability.json`.
