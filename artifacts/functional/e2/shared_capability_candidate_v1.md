# Shared Capability candidate v1

**Verdict: `CANDIDATE_COMPILES`** -- the card compiles from 12 deduped evidence rows and both leave-one-domain-out directions replay without a mismatch

SMD is a third-domain CANDIDATE, not an established third domain.  Nothing in this file is evidence about SMD; it appears only as a substrate observation constraining what the card may lean on.

## B1 -- evidence pool

Restricted to `outlier_mad`, `outlier_iqr`, `hampel_filter`, `winsorize` plus an operator-independent per-series harm guard. Dedupe key: `domain x window x plan x outcome`.

| domain | before dedupe | after dedupe |
| --- | ---: | ---: |
| traffic | 8 | 8 |
| noaa | 13 | 4 |
| **total** | 21 | **12** |

Surviving episodes: `traffic/pooled/hampel_filter`, `traffic/pooled/outlier_iqr`, `traffic/pooled/outlier_mad`, `traffic/pooled/winsorize`, `traffic/per_channel/hampel_filter`, `traffic/per_channel/outlier_iqr`, `traffic/per_channel/outlier_mad`, `traffic/per_channel/winsorize`, `noaa/task_A/outlier_mad`, `noaa/task_C/outlier_mad`, `noaa/task_A/outlier_iqr`, `noaa/task_D/outlier_mad`

## The card

```json
{
 "id": "shared_outlier_repair_with_per_series_guard_v1",
 "status": "SHARED_CANDIDATE",
 "authorization": "GUIDANCE",
 "target_support_required": true,
 "grants_confirmation_free_try": false,
 "programs": [
  "hampel_filter",
  "outlier_iqr",
  "outlier_mad",
  "winsorize"
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

**What it recommends.** on a batch whose Context matches the conditions below, put the outlier-repair family on the shortlist ahead of the rest of the menu, and attach the per-series harm guard to whatever the ladder adopts

### Applicability conditions (deployment-observable only)

| feature | requirement | why |
| --- | --- | --- |
| `missing_fraction over the training pool` | may be zero; the capability does not need missing data | traffic's census records missing_region_end_fraction all_zero over its 12 training series, and #30's S1 found 0 of 24 usable channels with any missing value on the third-domain candidate.  A capability that required missing data could not apply to either. |
| `local_robust_z_peak over the training pool` | at least one series at or above 4.0 | this is the public signal the outlier family acts on.  Traffic's census reports per-series peaks from 3.70 to 15.72; NOAA's development block has 3 of 20 series at or above 4.0 |
| `outlier_point_fraction over the training pool` | greater than zero somewhere in the pool | traffic's field stats give a mean of 0.0448 with 12 of 12 series distinct and non-degenerate |
| `per-eval-series gain dispersion of the adopted plan` | the guard is required whenever the minimum per-series delayed gain can fall below -0.005 while the aggregate stays positive | that configuration is the failure this line has recorded six times; it is invisible at aggregate granularity |

### Out of scope

- Families: imputation, level shift.
- imputation has no substrate on two of the three corpora (traffic: missing_region_end_fraction all_zero; smd: 0 of 24 usable channels with any missing value), and level_excursion_score is identically zero on all 20 NOAA development series, so a level-repair capability has no NOAA evidence to be induced from
- Evidence: `artifacts/functional/e2/s1_health_v1.json substrate_shape_warning`
- Evidence: `artifacts/functional/e2/m0a_mask_geometry_census_traffic_v1.json field_stats_train.missing_region_end_fraction`

## C -- leave-one-domain-out dry replay

| arm | episodes | at risk | vacuous | direction agrees (at risk) | hidden harms | caught | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C1: compiled on traffic, checked on NOAA | 4 | 4 | 0 | 4 / 4 | 3 | 3 | `SUPPORTED` |
| C2: compiled on NOAA, checked on traffic | 8 | 4 | 4 | 4 / 4 | 2 | 2 | `SUPPORTED` |
| C3: compiled on both, replayed on both | 12 | 12 | 0 | 12 / 12 | 5 | 5 | `SUPPORTED` |

Only the *at risk* column is evidence: a row whose program the card does not claim passes vacuously.  the harm line that defines a harmed row is the same line the guard uses, so 'the guard catches a harmed row' is true by construction and is not reported as a result.  What is reported is the subset where the aggregate was positive -- the case the guard exists for -- and the count of rows the aggregate alone would have caught.


**C3 is not transfer evidence.** this arm replays a card on the very episodes it was induced from.  It can only fail, never confirm, and it is not transfer evidence.

## Cost

- LLM calls: 0.  Consumer retrains: 0.
- Wall seconds: 0.0.
