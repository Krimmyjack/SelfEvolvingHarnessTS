# T6 -- natural A5 vs A3, frozen plan (t6_nab_frozen_plan_v1)

Evidence grade: **NATURAL / provisional**.

## Verdict

**T6_NATURAL_PLAN_READY**

the natural A5-vs-A3 comparison is frozen without any Target outcome being read, and the natural Source carries all three kinds of Action-Response the comparison depends on.  NATURAL / provisional: one Target domain, two cohorts, awaiting replication on an independent dataset; the frozen --evaluate path stays shut until the main line releases it.

## Part 0

- HEAD `5dee103` -- checkpoint: #41b-lite closeout + minimal V10 freeze surface (V10_READY_FOR_T6)
- Part 0 action: verified only -- the #41b-lite closeout was already committed and the tree carries no uncommitted diff

## Exposure

| surface | context | outcome |
| --- | --- | --- |
| Source | INSTANCE_SEEN | EXPOSED |
| Target | INSTANCE_SEEN | SEALED |

Aggregate disclosure: one of the six realAdExchange series carries no anomaly; which one is not disclosed and is not read here

## Part A -- data and shape gate

- upstream https://github.com/numenta/NAB @ `0dcd73007a34` (tag v1.1)
- files gated: 20, all passing: True

| role | cohort | file | n | gate |
| --- | --- | --- | --- | --- |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_24ae8d.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_53ea38.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_5f5533.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_77c1ca.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_825cc2.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_ac20cd.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_c6585a.csv` | 4032 | pass |
| source | source_aws_cloudwatch | `ec2_cpu_utilization_fe7f93.csv` | 4032 | pass |
| source | source_known_cause | `ambient_temperature_system_failure.csv` | 7267 | pass |
| source | source_known_cause | `cpu_utilization_asg_misconfiguration.csv` | 18050 | pass |
| source | source_known_cause | `ec2_request_latency_system_failure.csv` | 4032 | pass |
| source | source_known_cause | `machine_temperature_system_failure.csv` | 22695 | pass |
| source | source_known_cause | `nyc_taxi.csv` | 10320 | pass |
| source | source_known_cause | `rogue_agent_key_hold.csv` | 1882 | pass |
| target | target_cpc | `exchange-2_cpc_results.csv` | 1624 | pass |
| target | target_cpm | `exchange-2_cpm_results.csv` | 1624 | pass |
| target | target_cpc | `exchange-3_cpc_results.csv` | 1538 | pass |
| target | target_cpm | `exchange-3_cpm_results.csv` | 1538 | pass |
| target | target_cpc | `exchange-4_cpc_results.csv` | 1643 | pass |
| target | target_cpm | `exchange-4_cpm_results.csv` | 1643 | pass |

## Part C -- Source Experience bank

| cell | support | delayed | agg gain (delayed) | harmed |
| --- | --- | --- | --- | --- |
| aws_cloudwatch/r1/identity | ABSTAIN | ABSTAIN | +0.0000 | 0 of 8 |
| aws_cloudwatch/r1/outlier_iqr | NEGATIVE | NEUTRAL | +0.0010 | 0 of 8 |
| aws_cloudwatch/r1/outlier_mad | NEUTRAL | NEUTRAL | +0.0036 | 0 of 8 |
| aws_cloudwatch/r1/hampel_filter | NEUTRAL | NEGATIVE | -0.0123 | 1 of 8 |
| aws_cloudwatch/r1/winsorize | CONFLICT | POSITIVE | +0.0071 | 0 of 8 |
| aws_cloudwatch/r2/identity | ABSTAIN | ABSTAIN | +0.0000 | 0 of 8 |
| aws_cloudwatch/r2/outlier_iqr | POSITIVE | NEUTRAL | -0.0008 | 1 of 8 |
| aws_cloudwatch/r2/outlier_mad | NEUTRAL | NEUTRAL | +0.0039 | 0 of 8 |
| aws_cloudwatch/r2/hampel_filter | NEUTRAL | NEUTRAL | -0.0047 | 1 of 8 |
| aws_cloudwatch/r2/winsorize | POSITIVE | POSITIVE | +0.0078 | 0 of 8 |
| known_cause/r1/identity | ABSTAIN | ABSTAIN | +0.0000 | 0 of 6 |
| known_cause/r1/outlier_iqr | CONFLICT | NEUTRAL | +0.0000 | 0 of 6 |
| known_cause/r1/outlier_mad | NEUTRAL | NEUTRAL | +0.0000 | 0 of 6 |
| known_cause/r1/hampel_filter | CONFLICT | NEUTRAL | +0.0002 | 0 of 6 |
| known_cause/r1/winsorize | CONFLICT | NEUTRAL | -0.0004 | 0 of 6 |
| known_cause/r2/identity | ABSTAIN | ABSTAIN | +0.0000 | 0 of 6 |
| known_cause/r2/outlier_iqr | NEGATIVE | NEGATIVE | -0.0473 | 2 of 6 |
| known_cause/r2/outlier_mad | NEGATIVE | NEGATIVE | -0.0550 | 2 of 6 |
| known_cause/r2/hampel_filter | NEGATIVE | CONFLICT | +0.0400 | 2 of 6 |
| known_cause/r2/winsorize | NEGATIVE | NEGATIVE | -0.0747 | 3 of 6 |

### Readiness (C3)

- [x] `C3_identity_consumer_finite` -- every identity cell produced a finite macro F1
- [x] `C3_delayed_non_identity_positive` -- delayed POSITIVE cells: ['t6_source_aws_cloudwatch_r1_winsorize', 't6_source_aws_cloudwatch_r2_winsorize']
- [x] `C3_delayed_negative` -- delayed NEGATIVE cells: ['t6_source_aws_cloudwatch_r1_hampel_filter', 't6_source_known_cause_r2_outlier_iqr', 't6_source_known_cause_r2_outlier_mad', 't6_source_known_cause_r2_winsorize']
- [x] `C3_delayed_conflict` -- delayed CONFLICT cells (aggregate improved, at least one series past the harm line): ['t6_source_known_cause_r2_hampel_filter']

## Frozen evaluate protocol

Released: **False**

--evaluate refuses to run until this artifact carries evaluate_released: true, which only the main line sets after confirming the Target outcome is still sealed

- backend: `gpt-5.6-sol` @ `https://api.agicto.cn/v1`
- budgets: LLM <= 48, AD fits <= 120, forecasting retrains 0

- target_cpc: A3-r1 -> A5-r1 -> A3-r2 -> A5-r2
- target_cpm: A5-r1 -> A3-r1 -> A5-r2 -> A3-r2

## Ambiguities (reported, not self-adjudicated)

- The v2 contract keeps every row NAB shipped, including the four files whose timestamp column is not monotonic. Nothing is sorted, de-duplicated or resampled, and row counts and value bytes are verified identical before and after the read. What this costs is that a duplicated stamp now maps two rows into the same truth event rather than one; that is reported per series rather than hidden.
- The Query's trailing 19 points are read from the raw series, ruled correct by the #42a book: this is a training-data utility experiment, and the Program must not reach the inference input through a Query feature prefix.
- background_alarm_rate stays as implemented: the denominator is the scorable points lying outside every official window.
- The Episode bank writes one Episode per (cohort, round, program), not one per series: the Consumer's primary reading is a macro average over the cohort, and the per-series vector rides inside the same Episode as the harm evidence. identity is written too, as ABSTAIN.
- C3 now reads delayed_relation only. Support relations are kept and the Support-to-delayed flips are reported, but they cannot vote in the three-kind gate.
