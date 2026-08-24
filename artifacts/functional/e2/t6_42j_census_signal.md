# #42j census + Support signal instrument study

**verdict (main): FIT_POLICY_NOT_QUALIFIED**

r1: Part A/B merged into one 168-fit flow.  Old five loaded from
#42g-b/#42h (not recomputed); only mask-policy fitted fresh (48 fits).
Identity baseline re-fit (10 fits) only for the 10 series with feedback-region
AUPRC (needed for the four Support signal deltas); 14 zero-event series skip
the re-fit because their feedback-region AUPRC is undefined.
Book total 120 (Part A) + 48 (mask) + 10 (identity re-fit) = 178 fits <= 200.
0 LLM.  41 sealed unread.  EXPOSED 24 only.  Not a science claim.

## C1 main verdict (FIT_POLICY_QUALIFIED / NOT_QUALIFIED)

- mask eval macro = 0.001375 (gate > 0.005)
- mask harmed = 8 / 24 (gate <= 2)
- mask worst = -0.166667 (gate >= -0.02)
- harmed series: ['real_10.csv', 'real_12.csv', 'real_16.csv', 'real_20.csv', 'real_21.csv', 'real_23.csv', 'real_27.csv', 'real_30.csv']

## C2 Support signals (descriptive only; not authorized)

| signal | direction rate | policy macro | harmed | worst | qualified |
|---|---|---|---|---|---|
| f1_pooled_delta | 0.417 | 0.005901 | 0 | -0.003623 | SUPPORT_SIGNAL_FOUND |
| auprc_delta | 0.400 | 0.003090 | 0 | -0.003623 | SUPPORT_SIGNAL_NONE |
| score_margin_delta | 0.400 | 0.003968 | 0 | 0.000000 | SUPPORT_SIGNAL_NONE |
| flag_rate_delta | 0.500 | -0.000217 | 2 | -0.166667 | SUPPORT_SIGNAL_NONE |

## Six-program oracle

- delta_oracle_6 = 0.047818
- delta_oracle_5 = 0.037523 (target 0.037523)
- diff (6 - 5) = 0.010294
- choice counts (6-program): {'outlier_mad': 2, 'winsorize': 2, 'identity': 8, 'contamination_mask_refit_v1': 5, 'outlier_iqr': 2, 'hampel_filter': 5}

## B1 global eval (per program)

- outlier_iqr macro=-0.029757 harmed=5 pass=False
- outlier_mad macro=-0.060229 harmed=7 pass=False
- hampel_filter macro=-0.057184 harmed=12 pass=False
- winsorize macro=-0.091949 harmed=14 pass=False
- contamination_mask_refit_v1 macro=0.001375 harmed=8 pass=False

## B2 local (>=5 series improved)

- outlier_iqr n=6 pass=True
- outlier_mad n=5 pass=True
- hampel_filter n=7 pass=True
- winsorize n=4 pass=False
- contamination_mask_refit_v1 n=7 pass=True

## Cost

- Part A (feedback_unit_v1): 120 fits (U0 anchor)
- Part B (mask-policy only): 58 fits
- Book total: 178 / 200
- LLM: 0.  Retrains: 0.
