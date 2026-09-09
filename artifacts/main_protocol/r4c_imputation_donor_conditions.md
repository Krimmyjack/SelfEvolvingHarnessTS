# R4C imputation donor conditions -- machine reading

Evidence class: MECHANISM / NEGATIVE-capable; development; not a capability claim

Boundary: fits=0, llm=0, held_out_reads=0, horizon_reads=0, max_time_index_read=3863 (frontier 4056).
Rows: 840 paired (position, face, uid); skipped {'verifier_failed': 0, 'degenerate_uid_rows': 0, 'unpaired_faces': 0}; serving cross-check failed 0/840; training cross-check failed 0 windows of 8400.

## Verdicts

- action condition: **ACTION_CONDITION_CONSISTENT**
- value condition: **VALUE_CONDITION_UNINFORMATIVE** (best feature by mean rule-minus-incumbent: trn_gap_points)

## 3.1 consistency

- rows with no period fill on either side: 0; share |d|<1e-9 among them: None
- rows with |d|>1e-9: 840; share with a period fill somewhere: 1.0
- rows with |d|<1e-9 in total: 0
- basis: no row has zero period-filled points on both sides (every unit's training corpus has some); the check degenerates to: rows with |d| > 1e-9 that have a period-filled point on at least one side

| group | n | mean d | mean abs d | d>0 rate | d<-0.30 rate | exact-zero rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| serving_untouched | 180 | 0.004815 | 0.280934 | 0.527778 | 0.161111 | 0.0 |
| serving_period_filled | 660 | -0.009428 | 0.36577 | 0.469697 | 0.216667 | 0.0 |

## 3.2 Spearman with |d|

| feature | pooled | support | delayed |
| --- | ---: | ---: | ---: |
| srv_fill_divergence | -0.101828 | -0.147014 | -0.057365 |
| srv_period_filled_points | 0.144144 | 0.225588 | 0.06134 |
| srv_donor_dispersion | -0.020808 | 0.066125 | -0.085694 |
| trn_fill_divergence | 0.03156 | -0.012494 | 0.082412 |
| trn_period_filled_frac | -0.036334 | -0.002696 | -0.076111 |

## 3.3 LODO AUC (oriented on training folds)

| feature | target | pooled raw | lodo mean | lodo min | folds |
| --- | --- | ---: | ---: | ---: | --- |
| srv_gap_points | d_positive | 0.538406 | 0.534583 | 0.480861 | [0:40]=0.546192, [120:160]=0.480861, [40:80]=0.555339, [80:120]=0.555941 |
| srv_gap_points | d_severe_harm | 0.477195 | 0.471119 | 0.414773 | [0:40]=0.493683, [120:160]=0.414773, [40:80]=0.43088, [80:120]=0.545139 |
| srv_period_filled_points | d_positive | 0.543539 | 0.473487 | 0.382492 | [0:40]=0.484069, [120:160]=0.464115, [40:80]=0.382492, [80:120]=0.563272 |
| srv_period_filled_points | d_severe_harm | 0.526396 | 0.499305 | 0.413326 | [0:40]=0.413326, [120:160]=0.611742, [40:80]=0.463125, [80:120]=0.509028 |
| srv_period_filled_frac | d_positive | 0.468747 | 0.488861 | 0.401323 | [0:40]=0.401323, [120:160]=0.553828, [40:80]=0.484474, [80:120]=0.515818 |
| srv_period_filled_frac | d_severe_harm | 0.540004 | 0.559624 | 0.524621 | [0:40]=0.549077, [120:160]=0.524621, [40:80]=0.537714, [80:120]=0.627083 |
| srv_donor_dispersion | d_positive | 0.485972 | 0.489606 | 0.433689 | [0:40]=0.457435, [120:160]=0.600478, [40:80]=0.433689, [80:120]=0.466821 |
| srv_donor_dispersion | d_severe_harm | 0.471615 | 0.551735 | 0.496832 | [0:40]=0.538265, [120:160]=0.545455, [40:80]=0.496832, [80:120]=0.626389 |
| srv_fill_divergence | d_positive | 0.514959 | 0.527794 | 0.488426 | [0:40]=0.532679, [120:160]=0.578947, [40:80]=0.511124, [80:120]=0.488426 |
| srv_fill_divergence | d_severe_harm | 0.441202 | 0.583208 | 0.540634 | [0:40]=0.540634, [120:160]=0.560606, [40:80]=0.569092, [80:120]=0.6625 |
| srv_tail48_period_filled | d_positive | 0.564106 | 0.553474 | 0.517943 | [0:40]=0.547152, [120:160]=0.517943, [40:80]=0.602893, [80:120]=0.54591 |
| srv_tail48_period_filled | d_severe_harm | 0.47896 | 0.479171 | 0.395833 | [0:40]=0.463618, [120:160]=0.395833, [40:80]=0.432234, [80:120]=0.625 |
| trn_gap_points | d_positive | 0.549042 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_gap_points | d_severe_harm | 0.453523 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_period_filled_points | d_positive | 0.540414 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_period_filled_points | d_severe_harm | 0.467101 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_period_filled_frac | d_positive | 0.461175 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_period_filled_frac | d_severe_harm | 0.538121 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_fill_divergence | d_positive | 0.549723 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_fill_divergence | d_severe_harm | 0.456656 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_series_touched | d_positive | 0.500795 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |
| trn_series_touched | d_severe_harm | 0.502611 | 0.5 | 0.5 | [0:40]=0.5, [120:160]=0.5, [40:80]=0.5, [80:120]=0.5 |

## 3.4 decision: use W2 iff rule, else incumbent (whole served population)

| feature | mean rule-ANC | mean rule-W2 | mean harmed-ANC | folds beating both | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| srv_gap_points | -0.011948 | -0.015832 | 8.0 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| srv_period_filled_points | -0.02782 | -0.031703 | 11.75 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| srv_period_filled_frac | -0.01098 | -0.014864 | 6.75 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| srv_donor_dispersion | -0.024592 | -0.028476 | 8.0 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| srv_fill_divergence | -0.008691 | -0.012575 | 5.25 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| srv_tail48_period_filled | -0.000495 | -0.004378 | 9.0 | 1/4 | VALUE_CONDITION_UNINFORMATIVE |
| trn_gap_points | 0.021834 | 0.017951 | -1.0 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| trn_period_filled_points | -0.020297 | -0.02418 | 10.25 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| trn_period_filled_frac | -0.017951 | -0.021834 | 10.75 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| trn_fill_divergence | -0.006798 | -0.010681 | 10.25 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |
| trn_series_touched | -0.028632 | -0.032515 | 11.25 | 0/4 | VALUE_CONDITION_UNINFORMATIVE |

## 3.5 unit level (Spearman with unit mean d, 42 units)

- trn_period_filled_frac: -0.209268
- trn_fill_divergence: 0.248635
- trn_period_filled_points: 0.282132
- trn_series_touched: 0.045931

## per unit

| position | face | block | mean d | trn gap | trn period-filled | frac | divergence | series touched | windows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | delayed_face | [0:40] | -0.288466 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 0 | support_face | [0:40] | 0.248802 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 1 | delayed_face | [0:40] | 0.324177 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 1 | support_face | [0:40] | 0.02926 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 2 | delayed_face | [0:40] | 0.00995 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 2 | support_face | [0:40] | 0.0087 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 3 | delayed_face | [0:40] | -0.008171 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 3 | support_face | [0:40] | 0.159561 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 4 | delayed_face | [0:40] | 0.142612 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 4 | support_face | [0:40] | 0.243176 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 5 | delayed_face | [0:40] | 0.002831 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 5 | support_face | [0:40] | 0.179759 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 6 | delayed_face | [0:40] | 0.133662 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 6 | support_face | [0:40] | 0.036863 | 9295.0 | 4207.0 | 0.452609 | 1.443063 | 20.0 | 200.0 |
| 7 | delayed_face | [40:80] | 0.045538 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 7 | support_face | [40:80] | -0.148846 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 8 | delayed_face | [40:80] | -0.104131 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 8 | support_face | [40:80] | -0.002432 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 9 | delayed_face | [40:80] | 0.065525 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 9 | support_face | [40:80] | 0.104342 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 10 | delayed_face | [40:80] | -0.355498 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 10 | support_face | [40:80] | -0.42177 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 11 | delayed_face | [40:80] | -0.164225 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 11 | support_face | [40:80] | 0.213076 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 12 | delayed_face | [40:80] | -0.08419 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 12 | support_face | [40:80] | -0.049487 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 13 | delayed_face | [40:80] | -0.365128 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 13 | support_face | [40:80] | 0.288098 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 14 | delayed_face | [40:80] | -0.350807 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 14 | support_face | [40:80] | 0.017357 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 15 | delayed_face | [40:80] | -0.188557 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 15 | support_face | [40:80] | 0.039783 | 7593.0 | 4120.0 | 0.542605 | 1.35851 | 20.0 | 200.0 |
| 16 | delayed_face | [80:120] | 0.073193 | 7597.0 | 3259.0 | 0.428985 | 1.397833 | 19.0 | 200.0 |
| 16 | support_face | [80:120] | -0.012887 | 7597.0 | 3259.0 | 0.428985 | 1.397833 | 19.0 | 200.0 |
| 17 | delayed_face | [80:120] | 0.152768 | 7597.0 | 3259.0 | 0.428985 | 1.397833 | 19.0 | 200.0 |
| 17 | support_face | [80:120] | -0.042176 | 7597.0 | 3259.0 | 0.428985 | 1.397833 | 19.0 | 200.0 |
| 23 | delayed_face | [120:160] | -0.050881 | 1996.0 | 855.0 | 0.428357 | 1.460613 | 17.0 | 200.0 |
| 23 | support_face | [120:160] | -0.324198 | 1996.0 | 855.0 | 0.428357 | 1.460613 | 17.0 | 200.0 |
| 24 | delayed_face | [120:160] | 0.194035 | 1996.0 | 855.0 | 0.428357 | 1.460613 | 17.0 | 200.0 |
| 24 | support_face | [120:160] | -0.072993 | 1996.0 | 855.0 | 0.428357 | 1.460613 | 17.0 | 200.0 |
| 25 | delayed_face | [120:160] | -0.016732 | 1996.0 | 855.0 | 0.428357 | 1.460613 | 17.0 | 200.0 |
| 25 | support_face | [120:160] | 0.070716 | 1996.0 | 855.0 | 0.428357 | 1.460613 | 17.0 | 200.0 |
