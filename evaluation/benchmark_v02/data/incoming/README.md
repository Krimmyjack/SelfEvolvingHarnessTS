# Benchmark v0 manual source imports

The benchmark-wide data design, role semantics, split hierarchy, and current
roster are documented in [`../../BENCHMARK_DATA.md`](../../BENCHMARK_DATA.md).

Automatic acquisition handles Monash, METR-LA, UCI Electricity Load Diagrams,
and NOAA Global Hourly. The sources below require an account or acceptance of
portal terms and must be downloaded manually without renaming internal files.

## METR-LA official-object fallback

The automatic downloader first uses the DCRNN authors' Google Drive object and
then `gdown`. If both routes are blocked by local TLS or return a Drive 5xx,
download object `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX` from
<https://drive.google.com/uc?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX> and place
the untouched HDF5 file at:
`data/benchmark_v0/raw/metr_la/metr-la.h5`.

The file must begin with the HDF5 signature `89 48 44 46 0d 0a 1a 0a`.
Re-run `acquire --automatic` afterward so the immutable SHA sidecar and
acquisition manifest are written before probing.

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
