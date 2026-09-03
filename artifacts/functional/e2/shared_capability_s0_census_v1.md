# S0 -- third-domain census for Phase S

**Verdict: `THIRD_DOMAIN_AVAILABLE`** -- 1 of 3 measured candidates cleared structure and the public phenomenon bar

every measurement reads a development prefix of at most 3072 rows, strictly below the sealed index the earlier screening fixed; beyond_17520 and every held-out partition are untouched

## Bars

- This line: 16 series minimum (12 train + 4 eval), 10896 points minimum (windows [1104, 1440, 1800, 9864, 10560], horizon 48).
- Pre-registered g3 bar, reused unchanged: 12 train + 8 eval series, 5760 points, at least 4 series carrying a public phenomenon.

## Exposure ledger

| candidate | family | context | outcome | verdict | evidence |
| --- | --- | --- | --- | --- | --- |
| noaa_global_hourly (this line's cohort) | weather | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_INCUMBENT` | data/benchmark_noaa_fresh_v1/manifest.json consumption_census: consumed_n 44, fresh_n 20; #17 opened the 2025 confirmation partition once (fresh_confirmation_v1.*), and every trajectory since has read task_A/probe/task_B/task_C/task_D inside it |
| noaa_global_hourly (unused stations, Tier2) | weather | `UNSEEN` | `SEALED` | `NOT_MATERIALIZABLE_OFFLINE` | data/benchmark_v0/raw/noaa_global_hourly/2024 holds 64 station csv files, which is exactly consumed_n 44 + fresh_n 20.  isd-history.csv lists the wider station universe but the corresponding hourly csv files are not on disk |
| uci_electricity_load_diagrams / tsquality electricity | energy | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | batch_recipe_electricity_v1.json exposure = G3_DEVELOPMENT_SOURCE_FAMILY_OVERLAP (prior Outcome exposure in this project); g3_candidate_screening_v3 exposed_families.energy |
| T233 | legacy_mixed | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | batch_recipe_T233_v1.json exposure = 'already exposed development data; not fresh'; g1/g2 T233 artifacts |
| monash traffic_hourly / PeMS / metr_la / tsquality traffic | traffic | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | batch_recipe_traffic_v1.json exposure = STRUCTURALLY_ACCEPTED_BUT_SOURCE_FAMILY_EXPOSURE_UNRESOLVED; g3_candidate_screening_v3 exposed_families.traffic lists metr_la and monash:traffic_hourly; m0a_mask_geometry_census_traffic_v1 read the instances |
| tsquality weather (Jena) | weather | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | g3_candidate_screening_v3 exposed_families.weather lists tsl_weather_jena beside noaa_global_hourly |
| kdd2018 | air_quality | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | g3_candidate_screening_v3 exposed_families.air_quality; kdd_historical_policy_skill_memory_target_report.json |
| beijing_multisite (PRSA) | air_quality | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | natural_imputation_prsa_actionable_target_report.json and three natural_missing_window_weighting_prsa_* reports read these stations |
| monash nn5_daily / covid_deaths | finance / epidemiology | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | autonomous_natural_workflow_generation_nn5_* reports; g3_candidate_screening_v3 exposed_families.finance and .epidemiology |
| gefcom2012_load | energy | `INSTANCE_SEEN` | `EXPOSED` | `EXCLUDED_EXPOSED_FAMILY` | autonomous_natural_workflow_generation_gefcom* reports; g3_candidate_screening_v3 exposed_families.energy |

## Unconsumed corpora on this machine

| candidate | family | status |
| --- | --- | --- |
| psm | server telemetry (eBay pooled server metrics) | screened before: FRESH_SOURCE_FAMILY, rejected: 1 series with a public phenomenon against a bar of 4 -- the channels arrive pre-normalized, so the Operator DSL has nothing to act on |
| swat | industrial control (Secure Water Treatment testbed) | screened before: FRESH_SOURCE_FAMILY, rejected: 15 of 52 columns clean under both substrate guards against a bar of 20 -- the actuator channels are near-constant and hit the scale floor |
| exchange_rate | finance | 8 series against this line's bar of 16, and 7588 points against 10896 |
| illness | epidemiology | 7 series and 966 points |
| ETT-small | energy (transformer temperature) | 7 channels per file against a bar of 16; pooling the two hourly files gives 14, and pooling all four mixes hourly with 15-minute resolution.  The energy family is exposed in any case |
| m4 (Hourly / Daily / Monthly / Yearly) | legacy_mixed | many series but each far short of 10896 (Hourly tops out near 960, Daily near 9933); legacy_mixed is an exposed family |
| PEMS-SF | traffic | stored as .ts classification samples of 144 steps, not a continuous batch; traffic is an exposed family |
| UCR archive (BeetleFly, FordA, ... ), and the .ts classification corpora (FaceDetection, Heartbeat, Handwriting, ...) | classification benchmarks | labelled classification samples, not forecasting batches; no origin/horizon structure to map the window shape onto |
| weatherbench_daily, wiki_daily_100k, m5, monash_m3_monthly, electricity_15min, electricity_weekly, energy, healthcare, finance, synthetic, other | various | directory present but empty on this machine; nothing to measure |

## Measured this round (development prefix only)

| candidate | shape | usable / total channels | length bar | public phenomena | binary or constant | verdict |
| --- | --- | --- | --- | --- | ---: | --- |
| `smd` | [708405, 38] | 24 / 38 | pass | 26 / 4 | 9 | `ELIGIBLE` |
| `smap` | [135183, 25] | 1 / 25 **FAIL** | pass | 19 / 4 | 24 | `FAILS_DEGENERATE_CHANNELS` |
| `msl` | [58317, 55] | 1 / 55 **FAIL** | pass | 22 / 4 | 54 | `FAILS_DEGENERATE_CHANNELS` |

## The prior screening, as it stands

- Artifact: `artifacts/functional/e2/g3_candidate_screening_v3.json`, state `OPEN_BACKGROUND_SEARCH`.
- both fresh-family candidates were rejected on pre-registered criteria, not on freshness, and no criterion was moved to let one through. No Outcome was opened on any candidate.
- If resumed: a fresh family needs both an unnormalized signal (so public phenomena exist) and enough non-degenerate channels. Server and industrial-control corpora fail the first and second respectively.

## Cost

- LLM calls: 0.  Consumer retrains: 0.  Outcome opened: none.
- Wall seconds: 120.7.
