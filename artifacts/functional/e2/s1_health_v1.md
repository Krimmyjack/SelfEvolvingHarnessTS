# S1 -- health check on `smd`

**Verdict: `PROCEED_UNCHANGED`** -- 24 usable channels, 23 of them carrying a public phenomenon against a bar of 4

nothing at or past index 8760 is read.  The windows this line would use at 9864 and 10560 are inside that sealed region, so S1 cannot and does not see them.

## Structure

- Channels: 38 total, **24 usable**, 14 degenerate (cardinality <= 20).
- Development block: 8760 points, 1.2% of the 708405 rows in the file.
- Meets the 12 train + 4 eval roster split: True.

## Prevalence over the usable channels

| family | count | of usable |
| --- | ---: | ---: |
| missing present | 0 | 24 |
| outlier family (z peak >= 4) | 23 | 24 |
| level-shift family | 5 | 24 |
| period evidence readable | 24 | 24 |
| any non-neutral probe direction | 24 | 24 |
| public phenomenon (the g3 test) | 23 | 24 |

## Distributions over the usable channels

| statistic | min | p25 | median | p75 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `missing_fraction` | 0 | 0 | 0 | 0 | 0 |
| `local_robust_z_peak` | 1.706 | 9.037 | 13.29 | 22.6 | 8.205e+07 |
| `level_excursion_score` | 0 | 3.429 | 6.452 | 9.526 | 2.564e+06 |
| `level_region_fraction` | 0 | 0.005365 | 0.009189 | 0.0105 | 0.01096 |
| `cardinality` | 43 | 133 | 423 | 1821 | 4635 |

## The same numbers on NOAA, for scale

noaa_global_hourly, this line's own development block (20 series). already exposed, so re-reading it opens nothing; same extractor, same 8760-point block, so the columns are comparable

| family | `smd` | NOAA |
| --- | ---: | ---: |
| missing present | 0 / 24 | 20 / 20 |
| outlier family (z peak >= 4) | 23 / 24 | 3 / 20 |
| level-shift family | 5 / 24 | 9 / 20 |

| statistic | median here | median on NOAA | max here | max on NOAA |
| --- | ---: | ---: | ---: | ---: |
| `missing_fraction` | 0 | 0.001826 | 0 | 0.9949 |
| `local_robust_z_peak` | 13.29 | 3.344 | 8.205e+07 | 2.8e+09 |
| `level_excursion_score` | 6.452 | 0 | 2.564e+06 | 0 |
| `cardinality` | 423 | 125 | 4635 | 626 |


## Substrate shape warning

the prevalence bar counts channels with any public phenomenon and this corpus clears it.  It does not check that the phenomena are the same ones the incumbent has, and they are not.

- **imputation**: 0 of 24 usable channels carry any missing value, against 20 of 20 series on NOAA.  every imputation operator in the menu is inert on this corpus.  A Shared Capability induced on NOAA that leans on imputation cannot be tested here at all, and a null transfer result would be a property of the substrate rather than of the capability
- **level shift**: level_excursion_score has median 6.452 here and is identically zero on all 20 NOAA series.  the two corpora differ in kind on this axis, not in degree; a level-repair capability has no NOAA evidence to be induced from in the first place

it does not block S1.  It is a pre-condition on the S2 candidate: whatever Shared Capability is compiled must lean on an operator family that both corpora can exercise, or the S3 comparison measures the substrate instead of the capability.


## What this does not do

- no channel was chosen, no threshold was tuned, and no roster was cut.  Choosing 12 train and 4 eval from these channels is S2's job and needs the boundary question below settled first.
- SMD_train.npy is 28 machines concatenated into one array with no boundary index on disk.  The development block read here is the head of that array, which at 8760 of 708405 rows is very likely inside the first machine but is not verified to be.  If a machine boundary falls inside the block, the slice mixes two entities and every per-series reading above is a reading of the mixture.  S2 must recover the per-machine index before any roster is cut.

## Cost

- LLM calls: 0.  Consumer retrains: 0.  Outcome opened: none.
- Wall seconds: 295.2.
