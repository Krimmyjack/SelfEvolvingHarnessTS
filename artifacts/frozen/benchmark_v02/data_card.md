# Benchmark benchmark-v0.2 data card

Registry rows: 1919; eligible: 1867.

Frozen role counts: `{'dev_query': 373, 'final_query': 570, 'support_a': 555, 'support_b': 331, 'u': 38}`.

Series per dataset: `{'gefcom2012_load': 20, 'legacy_monash:covid_deaths': 20, 'legacy_monash:fred_md': 20, 'legacy_monash:nn5_daily': 20, 'legacy_monash:saugeenday': 1, 'legacy_monash:sunspot': 1, 'legacy_monash:tourism_monthly': 20, 'legacy_monash:us_births': 1, 'metr_la': 207, 'monash:covid_deaths': 246, 'monash:nn5_daily': 91, 'monash:traffic_hourly': 862, 'noaa_global_hourly': 40, 'uci_electricity_load_diagrams': 370}`.

## What a number on this benchmark is allowed to claim

Claim tiers and domains are declared per dataset in `dataset_manifest.json`, before any method was run. `supplementary` datasets (GEFCom2012, 20 zones) join the pooled roster and the per-cell tables, but a result on a supplementary dataset alone is not a reportable finding -- its Query sample is too small to carry one.

### Traffic: two networks, and only one of them is spatially clean

Traffic series are sensors on a road graph, not independent entities, so a within-dataset traffic result can only ever claim *unseen sensor in this network* -- never cross-network generalization. Cross-network evidence has to come from METR-LA (Los Angeles) and monash:traffic_hourly (San Francisco Bay Area) testing each other, because they are genuinely different road graphs.

**METR-LA is spatially blocked.** Its 207 sensors merge into 137 co-location sites at a 0.2 km radius (largest site: 4 sensors -- these are detectors metres apart, e.g. opposite directions at one milepost), and those sites are grouped into 20 compact blocks by deterministic median bisection. The block, not the sensor, is the atomic split unit, so an entire block always travels to the same role. Minimum distance between sensors in *different* blocks: 0.2062 km. That is the residual: blocking bounds adjacent-sensor leakage, it does not eliminate it.

**monash:traffic_hourly is NOT spatially blocked, and cannot be.** The pinned Monash release ships no sensor coordinates, so its 862 Bay Area sensors are split per series. Adjacent sensors can therefore land on opposite sides of Support/Query and its within-dataset numbers should be read as an optimistic bound. METR-LA is the spatially clean traffic read; this one is not.

## Corruption

The grid in `corruption_grid.json` was pre-registered in full before any method was run against it. It has a **Natural lane** (`natural`, dose 0: no synthetic damage at all, only the missingness the source actually shipped with -- deterministic, so one replicate), the three inherited **Controlled v0** missingness cells, and five new **Controlled v0.1** cells covering outliers, structural break, additive noise, and timestamp disorder. A pipeline that only imputes will score identically to Raw on the last four; that is the point.

## Support-A is two pools, not one

`support_a_subsplit.json` partitions Support-A at the overlap-group level into `support_a_discovery` (search and fit freely) and `support_a_validation` (the development promotion gate). A candidate is never judged on the series used to find it. This is a partition of the Support-A *population* and has nothing to do with the chronological train/validation/test boundaries inside each series.

## U

NOAA Global Hourly is the sealed weather U pool. Natural NaNs are preserved; hourly bins with partial or absent observations remain missing.
