# Benchmark v0 manual source imports

Automatic acquisition handles Monash, METR-LA, UCI Electricity Load Diagrams,
and NOAA Global Hourly. The sources below require an account or acceptance of
portal terms and must be downloaded manually without renaming internal files.

## ENTSO-E Actual Total Load

1. Sign in at <https://transparency.entsoe.eu/>.
2. Export **Actual Total Load** for the preregistered bidding zones and UTC
   interval. Preserve the portal export metadata and timezone.
3. Place the untouched export files under:
   `data/benchmark_v0/incoming/entsoe_transparency/`.

## GEFCom 2012 load forecasting

1. Join <https://www.kaggle.com/competitions/global-energy-forecasting-competition-2012-load-forecasting>.
2. Download the official competition data archive.
3. Place the untouched archive under:
   `data/benchmark_v0/incoming/gefcom2012/`.

## GEFCom 2014 load forecasting

1. Join <https://www.kaggle.com/competitions/global-energy-forecasting-competition-2014-load-forecasting>.
2. Download the official load-track archive and retain its original name.
3. Place it under: `data/benchmark_v0/incoming/gefcom2014/`.

All timestamps must be interpreted or converted to UTC by the importer. Raw
archives are immutable after import; their SHA256 and source revision are
recorded before clean-base conversion. Import status will be available through
`python -m SelfEvolvingHarnessTS.run_benchmark acquire --manual-status` once the
runner task is integrated.
