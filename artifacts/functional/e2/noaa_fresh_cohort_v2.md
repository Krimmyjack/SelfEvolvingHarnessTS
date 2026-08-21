# NOAA fresh cohort v2

**Overall: `FRESH_COHORT_READY`** -- census v2 confirmed the #14 20-station pool; materialization wrote 20 x 8760 development arrays; 16 stations PASS, roster has 12

Pre-registered verdicts, reported side by side:

| verdict | status |
| --- | --- |
| `FRESH_COHORT_READY` | SELECTED |
| `CENSUS_DRIFT` | NOT_SELECTED |
| `MATERIALIZATION_STRUCTURE_FAIL` | NOT_SELECTED |
| `INSUFFICIENT_HEALTHY_STATIONS` | NOT_SELECTED |
| `JUDGE_UNREADABLE` | NOT_REACHED |

0 LLM calls.  0 Consumer retrains.  2025 csv not opened.

## Floor correction

{
  "from": 24,
  "to": 20,
  "when": "before any csv opened, before any series value read",
  "why": "the floor was slack on a 12-seat roster, not a data-dependent bar.  Under every #14 consumption counting method 24 unconsumed 2024 stations are unreachable (widest remainder 23; delivered #14 count 20).  20 still leaves 8 seats of slack.",
  "hash_on_record_v1": "45b85890c79baadde49cf1e07e5df688ad5346b2e02deaa1db9a4859ab4485ea"
}

## Exposure disclosure (verbatim)

```
family = AGGREGATE_SEEN(旧线 9 份 outcome 报告 + registry 40)
instance(本 20 站)= SCANNED_BY_RETIRED_SCREENING_NO_SURVIVING_READOUT
(旧线 p0 曾扫描全部 64 站,62 个拒绝读数无存留;本线方法开发未用任何 NOAA 数值)
outcome(本 20 站)= SEALED(从无 Consumer 在其上重训;2025 csv 未打开)
```

## Step 0 -- census v2

| field | value |
| --- | --- |
| status | `SUFFICIENT_FRESH_POOL` |
| census_drift | False |
| registry | `artifacts/frozen/benchmark_v02/series_registry.jsonl` (40) |
| consumed | 44 |
| fresh pool | **20** |
| floor (v2 / v1) | 20 / 24 |
| sufficient | True |

Fresh pool (lexicographic):

- `72029953966`
- `72101299999`
- `72203812897`
- `72259003927`
- `72329003935`
- `72351399999`
- `72422093820`
- `72435653866`
- `72438093819`
- `72511654737`
- `72529014768`
- `72605654791`
- `72654014936`
- `72743094850`
- `72793494248`
- `74486514719`
- `99999903062`
- `99999904140`
- `99999923908`
- `99999963862`

### Ambiguities

- **v1_raw_file_count_vs_station_count**: noaa_health_check_v1 counted raw_station_files=74 by rglob('*.csv') under data/benchmark_v0/raw/noaa_global_hourly (74 files: 64 under 2024/, 9 under 2025/, plus isd-history.csv). The fresh pool is unique 11-digit stems of 2024/*.csv (64). 2025 files are the same station ids (metadata listing only); isd-history.csv is not a station series.

- **p0_unnamed_rejected_stations**: noaa_multichannel_local_repair_p0_report.json names two station_id values ['72562624091', '72650014972'] and records source_file_count=64 with 62 rejected as FEWER_THAN_TWO_AFFECTED_TRAINING_ROWS.  Those 62 ids do not appear in the report or in an explicit roster, so they are not added to the consumed set.  If they were, the fresh pool would be empty.

- **2025_config_vs_report_body**: noaa_multichannel_local_repair_2025_report.json contains no station_id.  NOAA_DEWPOINT_FEASIBILITY_STATIONS is the runner config that produced it and is included because the task counts roster/config station numbers.  Those four ids not already in the registry (['72272093026', '72493723289', '72562624091', '72566024028']) are the difference between a 24-station registry-only remainder and the 20-station fresh pool below the pre-registered floor.

- **leap_year_vs_8760**: 2024 is a leap year (8784 hours).  The sealed boundary is the frozen 8760-hour conventional year starting 2024-01-01, not a Feb-29-aware calendar year.  Index 8760 would be 2024-12-31 00:00.  2025 csv contents are not ingested.

## Step 1 -- blind materialization

| field | value |
| --- | --- |
| stations written | 20 |
| length | 8760 |
| finite-count min / median / max | 45 / 8744.0 / 8759 |
| missing-rate min / median / max | 0.0001 / 0.0018 / 0.9949 |
| series with >=5760 finite | 17 |
| 2025 opened | False |
| generator sha256 | `2cc4f4e2c5e5672bad39d796fa4bf5bdf75ead9086e45dec26c0efff8617a654` |

## Step 2 -- per-station health check (development only)

Criteria quoted from `artifacts/functional/e2/noaa_health_check_v1.json`.

- length (verbatim): at least 12 train plus 8 eval series, each at least 5760 long
- length recorded: 8760 by construction (min_series_length=5760)
- missing-rate cap: finite points >= 5760 on the 8760-hour slice (max missing 3000, rate <= 0.342466); artifacts/functional/e2/noaa_health_check_v1.json pre_registered.criteria.min_series_length = 5760, applied as a finite-point minimum on the constructed 8760-hour development slice
- constant/flatline (verbatim): both the train and the eval substrate guard call the series clean, for at least 20 of them

Identity readability probe **not run** (0 Consumer retrain).

| station | length | finite | missing_rate | length | missing | flatline | verdict |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `72029953966` | 8760 | 7898 | 0.0984 | PASS | PASS | FAIL | **FAIL** |
| `72101299999` | 8760 | 228 | 0.9740 | PASS | FAIL | FAIL | **FAIL** |
| `72203812897` | 8760 | 8500 | 0.0297 | PASS | PASS | PASS | **PASS** |
| `72259003927` | 8760 | 8753 | 0.0008 | PASS | PASS | PASS | **PASS** |
| `72329003935` | 8760 | 8757 | 0.0003 | PASS | PASS | PASS | **PASS** |
| `72351399999` | 8760 | 45 | 0.9949 | PASS | FAIL | FAIL | **FAIL** |
| `72422093820` | 8760 | 8744 | 0.0018 | PASS | PASS | PASS | **PASS** |
| `72435653866` | 8760 | 8713 | 0.0054 | PASS | PASS | PASS | **PASS** |
| `72438093819` | 8760 | 8758 | 0.0002 | PASS | PASS | PASS | **PASS** |
| `72511654737` | 8760 | 8759 | 0.0001 | PASS | PASS | PASS | **PASS** |
| `72529014768` | 8760 | 8735 | 0.0029 | PASS | PASS | PASS | **PASS** |
| `72605654791` | 8760 | 8758 | 0.0002 | PASS | PASS | PASS | **PASS** |
| `72654014936` | 8760 | 8758 | 0.0002 | PASS | PASS | PASS | **PASS** |
| `72743094850` | 8760 | 346 | 0.9605 | PASS | FAIL | PASS | **FAIL** |
| `72793494248` | 8760 | 8204 | 0.0635 | PASS | PASS | PASS | **PASS** |
| `74486514719` | 8760 | 8744 | 0.0018 | PASS | PASS | PASS | **PASS** |
| `99999903062` | 8760 | 8758 | 0.0002 | PASS | PASS | PASS | **PASS** |
| `99999904140` | 8760 | 8757 | 0.0003 | PASS | PASS | PASS | **PASS** |
| `99999923908` | 8760 | 8759 | 0.0001 | PASS | PASS | PASS | **PASS** |
| `99999963862` | 8760 | 8707 | 0.0061 | PASS | PASS | PASS | **PASS** |

PASS 16 / 20.  Confirmation roster (lexicographic first 12 PASS):

- `72203812897`
- `72259003927`
- `72329003935`
- `72422093820`
- `72435653866`
- `72438093819`
- `72511654737`
- `72529014768`
- `72605654791`
- `72654014936`
- `72793494248`
- `74486514719`

Substitutes:

- `99999903062`
- `99999904140`
- `99999923908`
- `99999963862`

## Cost

0 Consumer retrains; 0 LLM calls.

generator sha256: `cd2d4720b0409102ba362f39957526a9802f67f9ba44045d15284572a26e0e30`

v1 generator sha on record: `45b85890c79baadde49cf1e07e5df688ad5346b2e02deaa1db9a4859ab4485ea`

