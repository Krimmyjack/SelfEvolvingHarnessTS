# #42e2 isolated_dominant × winsorize

verdict: **PATTERN_NO_DISCRIMINATION**

family closed: isolated-extreme × winsorize

may_enter_skill_scope: false

Part 0 sha: `3411bb5b0f9be9506349e2eb43fd194db9059965`

evidence_grade: MECHANISM / development. Does not form a Skill. Not a Capability claim.
positive cohorts: ['source_aws_cloudwatch']
adverse cohorts: ['source_known_cause', 'source_real_traffic', 'source_real_tweets']
excluded NA: []

| episode | delayed | class | isolated_frac | dominant | max_run |
|---|---|---|---|---|---|
| t6_source_aws_cloudwatch_r1_winsorize | POSITIVE | positive | 0.7044534412955465 | True | 172 |
| t6_source_aws_cloudwatch_r2_winsorize | POSITIVE | positive | 0.7890365448504983 | True | 172 |
| t6_source_known_cause_r1_winsorize | NEUTRAL | archive | 0.0861244019138756 | False | 140 |
| t6_source_known_cause_r2_winsorize | NEGATIVE | adverse | 0.09286328460877043 | False | 222 |
| t6_source_real_traffic_r1_winsorize | NEGATIVE | adverse | 0.5679012345679012 | True | 19 |
| t6_source_real_traffic_r2_winsorize | NEGATIVE | adverse | 0.5275590551181102 | True | 22 |
| t6_source_real_tweets_r1_winsorize | NEGATIVE | adverse | 0.600647016534867 | True | 145 |
| t6_source_real_tweets_r2_winsorize | NEUTRAL | archive | 0.6088205128205129 | True | 145 |

## C1
{
  "separates": false,
  "positive_side": true,
  "all_positive_same_side": true,
  "adverse_opposite_rate": 0.25,
  "adverse_opposite_count": 1,
  "positive_n": 2,
  "adverse_n": 4
}

## C2
{
  "values_to_cohorts": {
    "False": [
      "source_known_cause"
    ],
    "True": [
      "source_aws_cloudwatch",
      "source_real_traffic",
      "source_real_tweets"
    ]
  },
  "single_cohort_indicator": true,
  "complete_cohort_partition_replica": false,
  "both_boolean_sides_present": true,
  "both_boolean_sides_ge2_cohorts": false,
  "usable_as_scope": false
}

## C3
{
  "folds": [
    {
      "left_out": "source_aws_cloudwatch",
      "readable": false,
      "reason": "LOCO_UNREADABLE",
      "positive_n": 0,
      "adverse_n": 4
    },
    {
      "left_out": "source_known_cause",
      "readable": true,
      "c1": {
        "separates": false,
        "positive_side": true,
        "all_positive_same_side": true,
        "adverse_opposite_rate": 0.0,
        "adverse_opposite_count": 0,
        "positive_n": 2,
        "adverse_n": 3
      },
      "direction_holds": false
    },
    {
      "left_out": "source_real_traffic",
      "readable": true,
      "c1": {
        "separates": false,
        "positive_side": true,
        "all_positive_same_side": true,
        "adverse_opposite_rate": 0.5,
        "adverse_opposite_count": 1,
        "positive_n": 2,
        "adverse_n": 2
      },
      "direction_holds": false
    },
    {
      "left_out": "source_real_tweets",
      "readable": true,
      "c1": {
        "separates": false,
        "positive_side": true,
        "all_positive_same_side": true,
        "adverse_opposite_rate": 0.3333333333333333,
        "adverse_opposite_count": 1,
        "positive_n": 2,
        "adverse_n": 3
      },
      "direction_holds": false
    }
  ],
  "all_readable": false,
  "direction_holds": false
}
