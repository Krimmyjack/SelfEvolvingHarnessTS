# HEC-1 Best-Safe-Global baseline

ORACLE_WALL: everything in this module is computed after the course, reads the evaluation face only, and enters no arm.  No value here may be supplied to a Fast prompt, written to an Episode bank, or used to choose a program during a run.

Not an oracle: a Scoped policy can beat it. The reported quantity is each arm's **advantage** over it.

| item | value |
| --- | --- |
| menu size | 36 (18 single, 17 compositions) |
| ordering | forward |
| units available / considered / evaluated | 26 / 26 / 23 |
| fits per unit (estimate) | 70 |
| fits for all units | 1820 |
| fits spent here | 1046 |
| status | COMPLETE |

| unit | best in-budget program | aggregate | identity |
| --- | --- | ---: | --- |
| [0:40] x 1176 | `winsorize` | +0.4356 | False |
| [0:40] x 1896 | `identity` | +0.0000 | True |
| [0:40] x 2136 | `winsorize` | +0.5424 | False |
| [0:40] x 2376 | `period_median_complete>winsorize` | +0.4138 | False |
| [0:40] x 2616 | `outlier_mad` | +0.3014 | False |
| [0:40] x 2856 | `UNSCOREABLE` | — | — |
| [0:40] x 3576 | `period_median_complete>outlier_iqr` | +0.9429 | False |
| [40:80] x 1176 | `outlier_mad` | +0.2392 | False |
| [40:80] x 1416 | `winsorize` | +0.3485 | False |
| [40:80] x 1656 | `outlier_iqr` | +0.2512 | False |
| [40:80] x 2136 | `identity` | +0.0000 | True |
| [40:80] x 2376 | `identity` | +0.0000 | True |
| [40:80] x 2616 | `winsorize` | +0.2940 | False |
| [40:80] x 2856 | `UNSCOREABLE` | — | — |
| [40:80] x 3576 | `period_median_complete>hampel_filter` | +0.3460 | False |
| [40:80] x 3816 | `period_median_complete>winsorize` | +0.3403 | False |
| [80:120] x 1176 | `winsorize` | +0.5809 | False |
| [80:120] x 1416 | `identity` | +0.0000 | True |
| [80:120] x 1896 | `identity` | +0.0000 | True |
| [80:120] x 2136 | `identity` | +0.0000 | True |
| [80:120] x 2376 | `identity` | +0.0000 | True |
| [80:120] x 2616 | `winsorize` | +0.2372 | False |
| [80:120] x 2856 | `identity` | +0.0000 | True |
| [120:160] x 1176 | `identity` | +0.0000 | True |
| [120:160] x 1416 | `outlier_mad` | +0.2536 | False |
| [120:160] x 1656 | `UNSCOREABLE` | — | — |

