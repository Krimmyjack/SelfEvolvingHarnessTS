# #42e census v3

verdict: **SOURCE_RISK_ONLY_TRIGGERABLE**
cohorts: source_aws_cloudwatch, source_known_cause, source_real_traffic, source_real_tweets
cards: 20 old + 20 new = 40
TRY: []
RISK: ['hampel_filter', 'outlier_iqr', 'outlier_mad']
legal scopes: []
LLM 1 / 8; AD fit 170 / 240

## per-file gate

| file | cohort | ok | length | failures |
|---|---|---|---|---|
| TravelTime_387.csv | source_real_traffic | True | 2500 |  |
| TravelTime_451.csv | source_real_traffic | True | 2162 |  |
| occupancy_6005.csv | source_real_traffic | True | 2380 |  |
| occupancy_t4013.csv | source_real_traffic | True | 2500 |  |
| speed_6005.csv | source_real_traffic | True | 2500 |  |
| speed_7578.csv | source_real_traffic | True | 1127 |  |
| speed_t4013.csv | source_real_traffic | True | 2495 |  |
| Twitter_volume_AAPL.csv | source_real_tweets | True | 15902 |  |
| Twitter_volume_AMZN.csv | source_real_tweets | True | 15831 |  |
| Twitter_volume_CRM.csv | source_real_tweets | True | 15902 |  |
| Twitter_volume_CVS.csv | source_real_tweets | True | 15853 |  |
| Twitter_volume_FB.csv | source_real_tweets | True | 15833 |  |
| Twitter_volume_GOOG.csv | source_real_tweets | True | 15842 |  |
| Twitter_volume_IBM.csv | source_real_tweets | True | 15893 |  |
| Twitter_volume_KO.csv | source_real_tweets | True | 15851 |  |
| Twitter_volume_PFE.csv | source_real_tweets | True | 15858 |  |
| Twitter_volume_UPS.csv | source_real_tweets | True | 15866 |  |

## proxy audit

[
  {
    "feature": "level_only_post_shift_support_sufficient",
    "values_to_cohorts": {
      "True": [
        "source_aws_cloudwatch",
        "source_known_cause",
        "source_real_traffic",
        "source_real_tweets"
      ]
    },
    "constant": true,
    "single_cohort_indicator": false,
    "complete_cohort_partition_replica": false,
    "both_boolean_sides_present": false,
    "both_boolean_sides_ge2_cohorts": false,
    "forbidden": false,
    "usable_as_scope": false,
    "note": "no resolving power"
  },
  {
    "feature": "post_shift_support_sufficient",
    "values_to_cohorts": {
      "False": [
        "source_aws_cloudwatch",
        "source_real_traffic",
        "source_real_tweets"
      ],
      "True": [
        "source_known_cause"
      ]
    },
    "constant": false,
    "single_cohort_indicator": true,
    "complete_cohort_partition_replica": false,
    "both_boolean_sides_present": true,
    "both_boolean_sides_ge2_cohorts": false,
    "forbidden": true,
    "usable_as_scope": false,
    "note": "pss forbidden"
  },
  {
    "feature": "period_repair_available",
    "values_to_cohorts": {
      "False": [
        "source_aws_cloudwatch",
        "source_known_cause",
        "source_real_traffic",
        "source_real_tweets"
      ]
    },
    "constant": true,
    "single_cohort_indicator": false,
    "complete_cohort_partition_replica": false,
    "both_boolean_sides_present": false,
    "both_boolean_sides_ge2_cohorts": false,
    "forbidden": false,
    "usable_as_scope": false,
    "note": "no resolving power"
  }
]

## unconditional signed summary

[
  {
    "scope": "unconditional_4_cohort_pool",
    "program": "hampel_filter",
    "positive_cohorts": [],
    "negative_cohorts": [
      "source_aws_cloudwatch",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "conflict_cohorts": [
      "source_known_cause"
    ],
    "immaterial_cohorts": [
      "source_aws_cloudwatch",
      "source_known_cause"
    ],
    "strict_harm_cohorts": [
      "source_aws_cloudwatch",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "extended_harm_cohorts": [
      "source_aws_cloudwatch",
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "authorization": "RISK"
  },
  {
    "scope": "unconditional_4_cohort_pool",
    "program": "outlier_iqr",
    "positive_cohorts": [],
    "negative_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "conflict_cohorts": [
      "source_real_tweets"
    ],
    "immaterial_cohorts": [
      "source_aws_cloudwatch",
      "source_known_cause"
    ],
    "strict_harm_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "extended_harm_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "authorization": "RISK"
  },
  {
    "scope": "unconditional_4_cohort_pool",
    "program": "outlier_mad",
    "positive_cohorts": [],
    "negative_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "conflict_cohorts": [],
    "immaterial_cohorts": [
      "source_aws_cloudwatch",
      "source_known_cause",
      "source_real_tweets"
    ],
    "strict_harm_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "extended_harm_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "authorization": "RISK"
  },
  {
    "scope": "unconditional_4_cohort_pool",
    "program": "winsorize",
    "positive_cohorts": [
      "source_aws_cloudwatch"
    ],
    "negative_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "conflict_cohorts": [],
    "immaterial_cohorts": [
      "source_known_cause",
      "source_real_tweets"
    ],
    "strict_harm_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "extended_harm_cohorts": [
      "source_known_cause",
      "source_real_traffic",
      "source_real_tweets"
    ],
    "authorization": null
  }
]
