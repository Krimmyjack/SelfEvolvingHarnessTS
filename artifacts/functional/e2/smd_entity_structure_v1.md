# SMD entity structure, recovered from provenance

**28 machine entities, verified block by block against the official per-machine files.**  the 28 blocks tile [0, 708405) with no gap and no overlap; the last block ends exactly at 708405

## What an entity is

the packed array is 708405 x 38, and #30's S1 treated those 38 columns as 38 series.  They are 38 heterogeneous metrics of one machine -- cpu, memory, disk, network -- not 38 comparable series.  A roster cut over them would pool quantities that share no unit, and the batch geometry this line uses (a recipe applied to a training pool of series) would be meaningless.  The series unit is the machine.

## Provenance

Local first, as required:

- `shared_tsq_datasets/SMD`: six packed files only, no machine index
- `SMD_train.pkl`: unpickles to the same (708405, 38) float32 ndarray; carries no per-machine key
- `Time-Series-Library/SMD`: the same six packed files
- `machine-*.txt anywhere under Desktop or Downloads`: none
- `documented_origin`: shared_tsq_datasets/README.md: copied from the thuml/Time-Series-Library data release, which ships SMD pre-concatenated
- `outcome`: local provenance exhausted; priority (ii) taken

Local provenance was exhausted, so the official files were fetched from `https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset/train/machine-<x>-<y>.txt` (28 files).  They are kept in the scratchpad and are not added to the repository.

| check | result |
| --- | --- |
| sum of official train lengths | 708405 |
| packed array rows | 708405 |
| lengths match | **True** |
| column counts | 38 in every official file, matching the packed width |
| block verification | every machine's whole block was then compared with its official file element by element: 28 of 28 identical |
| tiling | the 28 blocks tile [0, 708405) with no gap and no overlap; the last block ends exactly at 708405 |

How the offsets were found: each machine's first row was converted to float32 and looked up in an exact byte index of the packed array's 708405 rows, all of which are distinct.  This is a content match against ground truth, not an inference from the signal: no changepoint detection, no heuristic, no threshold.

**Packing order.** neither numeric nor lexicographic -- the array starts with machine-1-5 and ends with machine-3-1.  The order is recorded below and must not be re-derived by sorting.

## Partition

- Development / held-in: the official train split, in full: each machine's own 23687-to-28743 rows
- Sealed held-out: the official test split.  Not read and not materialised at this stage; the row count below comes from the local file's header, which is metadata (total 708420 rows, header only).

#30's S1 borrowed NOAA's 8760/sealed boundary because SMD had no partition of its own.  It has one -- the official train/test split -- and that supersedes the borrowed index.  In hindsight the borrowed block [0, 8760) sat entirely inside machine-1-5's 23705 training rows, so it opened nothing, but it was one machine's readings reported as twenty-four series.

## The 28 entities

| # | entity | offset | train rows | usable channels | binary or constant | missing |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `machine-1-5` | 0 | 23705 | 25 / 38 | 8 | 0 |
| 2 | `machine-2-7` | 23705 | 23696 | 27 / 38 | 8 | 0 |
| 3 | `machine-2-6` | 47401 | 28743 | 29 / 38 | 8 | 0 |
| 4 | `machine-3-2` | 76144 | 23702 | 26 / 38 | 6 | 0 |
| 5 | `machine-3-11` | 99846 | 28695 | 21 / 38 | 12 | 0 |
| 6 | `machine-2-5` | 128541 | 23688 | 27 / 38 | 8 | 0 |
| 7 | `machine-1-8` | 152229 | 23698 | 28 / 38 | 8 | 0 |
| 8 | `machine-1-3` | 175927 | 23702 | 28 / 38 | 6 | 0 |
| 9 | `machine-3-4` | 199629 | 23687 | 26 / 38 | 8 | 0 |
| 10 | `machine-2-2` | 223316 | 23699 | 25 / 38 | 8 | 0 |
| 11 | `machine-1-4` | 247015 | 23706 | 27 / 38 | 4 | 0 |
| 12 | `machine-3-10` | 270721 | 23692 | 27 / 38 | 9 | 0 |
| 13 | `machine-2-9` | 294413 | 28722 | 25 / 38 | 8 | 0 |
| 14 | `machine-1-7` | 323135 | 23697 | 29 / 38 | 4 | 0 |
| 15 | `machine-3-7` | 346832 | 28705 | 24 / 38 | 6 | 0 |
| 16 | `machine-1-6` | 375537 | 23688 | 30 / 38 | 5 | 0 |
| 17 | `machine-2-3` | 399225 | 23688 | 27 / 38 | 8 | 0 |
| 18 | `machine-3-8` | 422913 | 28703 | 27 / 38 | 8 | 0 |
| 19 | `machine-3-5` | 451616 | 23690 | 24 / 38 | 12 | 0 |
| 20 | `machine-3-3` | 475306 | 23703 | 32 / 38 | 3 | 0 |
| 21 | `machine-1-1` | 499009 | 28479 | 26 / 38 | 8 | 0 |
| 22 | `machine-1-2` | 527488 | 23694 | 25 / 38 | 6 | 0 |
| 23 | `machine-2-1` | 551182 | 23693 | 28 / 38 | 5 | 0 |
| 24 | `machine-2-8` | 574875 | 23702 | 27 / 38 | 7 | 0 |
| 25 | `machine-2-4` | 598577 | 23689 | 28 / 38 | 8 | 0 |
| 26 | `machine-3-9` | 622266 | 28713 | 28 / 38 | 8 | 0 |
| 27 | `machine-3-6` | 650979 | 28726 | 28 / 38 | 8 | 0 |
| 28 | `machine-3-1` | 679705 | 28700 | 27 / 38 | 8 | 0 |

## Roster feasibility

- 28 entities against a requirement of 12 train + 4 eval = 16 entities: **True**.
- Open: every NOAA entity is one univariate hourly series; every SMD entity is 38 heterogeneous channels.  A recipe that this line applies to a pool of univariate series has no unambiguous meaning on a pool of multivariate entities, and which channel (or reduction) plays the role of the series is not decided here.  S1b must settle it before any roster is cut.

## Cost

- LLM calls: 0.  Consumer retrains: 0.  Sealed test split: not read.
- Wall seconds: 0.4.
