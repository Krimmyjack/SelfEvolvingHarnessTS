# NOAA health check: is the reserve worth burning?

**Overall: `NOAA_STRUCTURE_FAIL`** -- structure failed (0 of 19 materialized series reach the 5760-point minimum, and 20 are needed); substrate double guard, public phenomenon were never reached, so they are neither passed nor failed;  the frozen screen itself returned REJECTED_STRUCTURE, so the readability probe was not run and no Outcome was opened.

`noaa_global_hourly` is the dataset `e1.SEALED_CONFIRMATION_DATASET` names as the last sealed confirmation set.  Before any of it is spent, one question has to be answered that cannot be answered afterwards: is the instrument readable on this cohort?  Weather was rejected at a 24.4x eval-loss spread, and that failure is indistinguishable from a null result once the Outcome is open.

**Development region only.**  Every series is truncated to `[:3072]` at load time and the truncation is asserted, so no index at or past the boundary can be read by construction.  0 LLM calls.

## Step 0 -- in place

| field | value |
| --- | --- |
| registry | `artifacts/frozen/benchmark_v02/series_registry.jsonl` |
| series in registry | 40 |
| materialized under `data/benchmark_v0_2/clean_base` | 19 |
| not materialized | 21 |
| channel structure | one univariate series per record; 40 distinct entity_id, the NOAA ISD station identifier |
| array shape | `[1024]` |
| registry lengths | {"1024": 40} |
| shortest length | **1024** |
| frequency | hourly |
| exposure_class | certified_virgin |
| overlap family / status | noaa_isd / resolved |
| raw source dir | `data/benchmark_v0/raw/noaa_global_hourly` (74 station files, listed by filesystem metadata only) |

Sealed semantics: `kind = index_sealed_boundary`, boundary 3072 from `evaluation/functional/task_episode_harness/agentic/g3_sourcing.py::SEALED_FROM_INDEX`.  The readability probe's farthest index would be 1848, inside the boundary.  every loaded series is sliced to [:3072] before any consumer sees it, and an assertion refuses to continue if any array is longer

Load report: {"loaded_series": 19, "longest_loaded_length": 1024, "no_index_at_or_past_boundary_was_read": true, "series_truncated_at_the_boundary": 0, "shortest_loaded_length": 1024, "truncated_examples": []}.

The frozen manifest records it as: {"broad_domain": "weather", "claim_tier": "headline", "independence": "independent_entities", "n_series": 40, "network_id": null, "note": "NOAA ISD hourly surface temperature, the sealed U (unseen-domain) pool. Stations are geographically dispersed across the US by a hash-ordered selection. Carries heavy, genuinely natural missingness -- which is the point: it is the one source whose defects nobody injected."}

## Step 1 -- outcome-blind screen

Criteria are read from `g3_sourcing.CRITERIA`, never restated here, and no bar is moved.

| criterion | required | measured | verdict |
| --- | --- | --- | --- |
| structure | at least 12 train plus 8 eval series, each at least 5760 long | 0 of 19 materialized series reach the minimum length; shortest is 1024 | **FAIL** |
| substrate double guard | both the train and the eval substrate guard call the series clean, for at least 20 of them | not reached -- the screen returned REJECTED_STRUCTURE before the guards ran | `NOT_REACHED` |
| public phenomenon | at least 4 training series carry a publicly visible phenomenon | not reached -- the screen returned REJECTED_STRUCTURE before the census ran | `NOT_REACHED` |

The frozen screen's own verdict on the same inputs: `REJECTED_STRUCTURE`.

How far the screen's guards reach on the index axis: the train guard stops at 900 (inside the boundary); the eval guard validates the nine frozen roster windows and reaches 5616 (**past** the boundary).  the frozen roster's first Support origin is the sealed boundary itself, so the eval substrate guard necessarily reads into the sealed region.  Under this check's zero-read rule that guard cannot be completed on any cohort, NOAA included

## Step 2 -- readability probe

**Not run.**  the outcome-blind screen did not pass, so the readability probe was not run and no Outcome was opened

## Exposure ledger

| partition | state after this check | detail |
| --- | --- | --- |
| Context | `INSTANCE_SEEN` | series values on the prefix below index 3072, plus registry metadata |
| development Outcome | `UNTOUCHED` | 0 Consumer retrains -- the outcome-blind screen failed, so no Consumer was fitted and no Judge reading was taken by this check |
| index >= 3072 | `SEALED` | 0 indices read at or past the boundary. no NOAA series in the frozen materialization even reaches index 3072, so the boundary was never approached |

### The reserve was not untouched

these are this project's own NOAA reports; a reserve with outcome reports already written is not untouched.  Reported as found, not resolved here  Artifacts already on disk (10):

- `artifacts/functional/e2/autonomous_natural_workflow_scope_induction_v2_noaa_confirmation_report.json`
- `artifacts/functional/e2/noaa_health_check_v1.json`
- `artifacts/functional/e2/noaa_multichannel_local_repair_2025_report.json`
- `artifacts/functional/e2/noaa_multichannel_local_repair_p0_report.json`
- `artifacts/functional/e2/w1_a5_vs_a3_report_noaa.json`
- `artifacts/functional/e2/w1_noaa_a5_vs_a3_report.json`
- `artifacts/functional/e2/w1_noaa_cross_domain_premise_report.json`
- `artifacts/functional/e2/w1_noaa_impute_census_report.json`
- `artifacts/functional/e2/w1_noaa_impute_fft_census_report.json`
- `artifacts/functional/e2/w1_noaa_impute_linear_census_report.json`

The two records also disagree with each other: the frozen registry marks every series `certified_virgin` with overlap status `resolved`, while `g3_sourcing.EXPOSED_FAMILIES` lists `noaa_global_hourly` under the exposed weather family (noaa_global_hourly, tsl_weather_jena).  g3_sourcing.EXPOSED_FAMILIES lists noaa_global_hourly under the exposed weather family, while the frozen registry marks every one of its series certified_virgin.  Both readings are recorded; this check does not adjudicate between them

## Cost

0 Consumer retrains of a budget of 12; 0 LLM calls.

