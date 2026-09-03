# Shared Capability candidate v2 (Source-only revision)

Frozen before any SMD outcome was opened.  v1 is not modified.

```json
{
 "status": "SHARED_CANDIDATE",
 "authorization": "GUIDANCE",
 "target_support_required": true,
 "grants_confirmation_free_try": false,
 "shared_programs": [
  "outlier_iqr",
  "outlier_mad"
 ],
 "guard": {
  "statistic": "min_per_series_gain",
  "window": "delayed",
  "comparator": "lt",
  "threshold": -0.005,
  "actions_allowed": [
   "VETO_AND_FALL_BACK",
   "RESCOPE_MASK_HARMED_SERIES"
  ],
  "operator_independent": "the guard reads the adopted plan's measured per-series vector and names no program; it applies whatever the ladder adopted"
 }
}
```

## What changed

- Shared programs: `['hampel_filter', 'outlier_iqr', 'outlier_mad', 'winsorize']` -> `['outlier_iqr', 'outlier_mad']`.  outlier_iqr, outlier_mad have positive delayed evidence in both traffic and NOAA; hampel_filter, winsorize have it in traffic only, so they may be offered as Source-local contrast or Target exploration but carry no shared recommendation right
- Gates kept: ['missing_fraction', 'local_robust_z_peak'].
- Demoted to evidence: ['outlier_point_fraction'].
- Removed from pre-execution: ['per-eval-series gain dispersion of the adopted plan'].
- Unchanged: the risk guard: min_per_series_gain, delayed window, threshold -0.005, both VETO_AND_FALL_BACK and RESCOPE_MASK_HARMED_SERIES; the four fixed fields; the out-of-scope declaration for imputation and level shift

## Source support, by independent domain

| program | traffic rows / positive | noaa rows / positive | two-domain | role |
| --- | ---: | ---: | --- | --- |
| `outlier_iqr` | 2 / 2 | 1 / 1 | True | shared |
| `outlier_mad` | 2 / 2 | 3 / 3 | True | shared |
| `hampel_filter` | 2 / 2 | 0 / 0 | False | contrast_only |
| `winsorize` | 2 / 2 | 0 / 0 | False | contrast_only |

## Bidirectional LODO on the shared programs

| direction | compiled | target episodes | agrees | verdict |
| --- | --- | ---: | ---: | --- |
| traffic_to_noaa | ['outlier_iqr', 'outlier_mad'] | 4 | 4 | `SUPPORTED` |
| noaa_to_traffic | ['outlier_iqr', 'outlier_mad'] | 4 | 4 | `SUPPORTED` |

the harm line that defines a harmed row is the same line the guard uses, so 'the guard finds the crossing' is true by construction and is not counted here

