# NOAA fresh cohort v1

**Overall: `INSUFFICIENT_UNCONSUMED_STATIONS`** -- fresh pool has 20 unconsumed 2024 stations; the pre-registered floor is 24.  Materialization did not run.

Pre-registered verdicts, reported side by side:

| verdict | status |
| --- | --- |
| `FRESH_COHORT_READY` | NOT_REACHED |
| `PROCEED_POOLED_ONLY` | NOT_REACHED |
| `INSUFFICIENT_UNCONSUMED_STATIONS` | SELECTED |
| `MATERIALIZATION_STRUCTURE_FAIL` | NOT_REACHED |
| `JUDGE_UNREADABLE` | NOT_REACHED |

0 LLM calls.  2025 confirmation values were not read.

## Step 0 -- consumption census

| field | value |
| --- | --- |
| registry | `artifacts/frozen/benchmark_v02/series_registry.jsonl` (40 entity_id) |
| raw 2024 unique stations | 64 |
| v1 rglob csv files | 74 |
| consumed set | 44 |
| consumed beyond registry | 72272093026, 72493723289, 72562624091, 72566024028 |
| fresh pool | **20** |
| floor | 24 |
| sufficient | False |

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

Consumed-set sources (verbatim):

- `artifacts/frozen/benchmark_v02/series_registry.jsonl`
  - rule: every row with dataset_id=noaa_global_hourly, field entity_id
  - stations (40): `70148626642`, `70232526443`, `70260725378`, `70272026401`, `70305626635`, `70388800111`, `72028304927`, `72031853965`, `72037554844`, `72046599999`, `72049100150`, `72056724180`, `72084499999`, `72090799999`, `72099299999`, `72209253941`, `72215114794`, `72226013895`, `72327199999`, `72341813977`, `72411013741`, `72420014891`, `72541404886`, `72548604906`, `72549504975`, `72583799999`, `72650014972`, `72656804961`, `72658804956`, `72672024061`, `74671693808`, `74693503709`, `74732023002`, `74916799999`, `99730099999`, `99738499999`, `99773999999`, `99816999999`, `99821999999`, `99828199999`

- `artifacts/functional/e2/autonomous_natural_workflow_scope_induction_v2_noaa_confirmation_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (0): 

- `artifacts/functional/e2/noaa_multichannel_local_repair_2025_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (0): 

- `artifacts/functional/e2/noaa_multichannel_local_repair_p0_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (2): `72562624091`, `72650014972`

- `artifacts/functional/e2/w1_a5_vs_a3_report_noaa.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (0): 

- `artifacts/functional/e2/w1_noaa_a5_vs_a3_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (1): `72549504975`

- `artifacts/functional/e2/w1_noaa_cross_domain_premise_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (0): 

- `artifacts/functional/e2/w1_noaa_impute_census_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (2): `72549504975`, `72658804956`

- `artifacts/functional/e2/w1_noaa_impute_fft_census_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (12): `70148626642`, `70388800111`, `72037554844`, `72046599999`, `72226013895`, `72541404886`, `72549504975`, `72658804956`, `74671693808`, `99730099999`, `99738499999`, `99821999999`

- `artifacts/functional/e2/w1_noaa_impute_linear_census_report.json`
  - rule: 11-digit tokens; 64-char series_uid mapped through the registry; unique 8-char series_uid prefix mapped through the registry
  - stations (12): `70148626642`, `70388800111`, `72037554844`, `72046599999`, `72226013895`, `72541404886`, `72549504975`, `72658804956`, `74671693808`, `99730099999`, `99738499999`, `99821999999`

- `evaluation/functional/run_e2_autonomous_natural_workflow_generation.py::_fixed_roster + DATASET_CONFIGS['noaa']`
  - rule: dataset_id=noaa_global_hourly, length >= selection_origin+HORIZON (816), sort by series_uid, take first 20
  - stations (20): `72549504975`, `72658804956`, `99738499999`, `72226013895`, `72541404886`, `72037554844`, `70388800111`, `99730099999`, `72046599999`, `74671693808`, `99821999999`, `70148626642`, `74916799999`, `72341813977`, `72031853965`, `72411013741`, `72056724180`, `72327199999`, `72548604906`, `70305626635`

- `evaluation/functional/run_e2_cross_series_curation.py::NOAA_DEWPOINT_FEASIBILITY_STATIONS`
  - rule: evaluation/functional/run_e2_cross_series_curation.py phase=noaa-multichannel-repair-2025 (station_ids=NOAA_DEWPOINT_FEASIBILITY_STATIONS, year=2025)
  - stations (9): `72493723289`, `72327199999`, `72562624091`, `70232526443`, `72566024028`, `72272093026`, `72411013741`, `72650014972`, `72672024061`

### Ambiguities

- **v1_raw_file_count_vs_station_count**: noaa_health_check_v1 counted raw_station_files=74 by rglob('*.csv') under data/benchmark_v0/raw/noaa_global_hourly (74 files: 64 under 2024/, 9 under 2025/, plus isd-history.csv). The fresh pool is unique 11-digit stems of 2024/*.csv (64). 2025 files are the same station ids (metadata listing only); isd-history.csv is not a station series.

- **p0_unnamed_rejected_stations**: noaa_multichannel_local_repair_p0_report.json names two station_id values ['72562624091', '72650014972'] and records source_file_count=64 with 62 rejected as FEWER_THAN_TWO_AFFECTED_TRAINING_ROWS.  Those 62 ids do not appear in the report or in an explicit roster, so they are not added to the consumed set.  If they were, the fresh pool would be empty.

- **2025_config_vs_report_body**: noaa_multichannel_local_repair_2025_report.json contains no station_id.  NOAA_DEWPOINT_FEASIBILITY_STATIONS is the runner config that produced it and is included because the task counts roster/config station numbers.  Those four ids not already in the registry (['72272093026', '72493723289', '72562624091', '72566024028']) are the difference between a 24-station registry-only remainder and the 20-station fresh pool below the pre-registered floor.

- **leap_year_vs_8760**: 2024 is a leap year (8784 hours).  The sealed boundary is the frozen 8760-hour conventional year starting 2024-01-01, not a Feb-29-aware calendar year.  Index 8760 would be 2024-12-31 00:00.  2025 csv contents are not ingested.

## Step 1 -- blind materialization

Not run.  fresh pool below the pre-registered floor of 24

## Step 2 -- health check v2 (development only)

Not run.

## Exposure ledger

| partition | state | detail |
| --- | --- | --- |
| new-cohort Context | `UNTOUCHED` | nothing: materialization did not run |
| development Outcome | `UNTOUCHED` | 0 retrains -- no Consumer was fitted |
| confirmation (index >= 8760) | `SEALED` | 2025 csv not opened; calendar_sealed_boundary |
| family | `AGGREGATE_SEEN` | the old line already wrote nine NOAA outcome reports and the frozen registry lists 40 NOAA series; this task does not pretend the family is virgin |
| instance | `UNSEEN_REMAINDER_BELOW_FLOOR` | fresh 20 / consumed 44 |

## Cost

0 Consumer retrains of a budget of 12; 0 LLM calls.

generator sha256: `45b85890c79baadde49cf1e07e5df688ad5346b2e02deaa1db9a4859ab4485ea`

