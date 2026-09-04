# T6 -- natural A5 vs A3, frozen plan (t6_nab_frozen_plan_v1)

Evidence grade: **NATURAL / provisional**.

## Verdict

**NATURAL_DATA_SHAPE_INELIGIBLE**

2 of the six Target files fail the structural gate: [('exchange-2_cpc_results.csv', ['timestamps_not_strictly_increasing']), ('exchange-2_cpm_results.csv', ['timestamps_not_strictly_increasing'])].  No substitute is drawn -- a replacement chosen after a failure would be one chosen with knowledge the gate is not allowed to have, and the book fixes all six realAdExchange series with no replacement.

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
- files gated: 20, all passing: False

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
| source | source_known_cause | `ec2_request_latency_system_failure.csv` | 4032 | FAIL: timestamps_not_strictly_increasing |
| source | source_known_cause | `machine_temperature_system_failure.csv` | 22695 | FAIL: timestamps_not_strictly_increasing |
| source | source_known_cause | `nyc_taxi.csv` | 10320 | pass |
| source | source_known_cause | `rogue_agent_key_hold.csv` | 1882 | pass |
| target | target_cpc | `exchange-2_cpc_results.csv` | 1624 | FAIL: timestamps_not_strictly_increasing |
| target | target_cpm | `exchange-2_cpm_results.csv` | 1624 | FAIL: timestamps_not_strictly_increasing |
| target | target_cpc | `exchange-3_cpc_results.csv` | 1538 | pass |
| target | target_cpm | `exchange-3_cpm_results.csv` | 1538 | pass |
| target | target_cpc | `exchange-4_cpc_results.csv` | 1643 | pass |
| target | target_cpm | `exchange-4_cpm_results.csv` | 1643 | pass |

## Frozen evaluate protocol

Not written: the run stopped at a Part A first-fault, and a protocol frozen on a surface that failed its own shape gate would be frozen around a defect.


## Ambiguities (reported, not self-adjudicated)

- C3's three-kind gate does not say which lifecycle layer it reads. It is evaluated as 'either the Support or the delayed layer of a cell carries this relation', and both layers are reported separately so the main line can see which one supplied each kind.
- The Query's trailing 19 points are read from the raw series, never from the prepared block. The book says the Query is never processed; taking the trailing window from the prepared block would have let the program reach the query features through the back door. This follows the canon t1b already set for trailing geometry.
- The Episode bank writes one Episode per (cohort, round, program), not one per series: the Consumer's primary reading is a macro average over the cohort, and the per-series vector rides inside the same Episode as the harm evidence. identity is written too, as ABSTAIN, so the bank carries the do-nothing baseline the card channel needs.
- The Episode's final relation is taken from the delayed layer (evidence_level DELAYED), matching online_loop's own semantics; the Support-layer classification is kept alongside it rather than discarded.
- background_alarm_rate is defined here as the share of scored points outside every truth window that were flagged. The book named the metric without fixing the denominator.
- NAB's official window bounds are timestamps that do not always fall on a sample. When a bound does not match a sample exactly, the window is taken as the enclosing sample positions; windows that enclose no sample are dropped. This affects Source only under the wall.
