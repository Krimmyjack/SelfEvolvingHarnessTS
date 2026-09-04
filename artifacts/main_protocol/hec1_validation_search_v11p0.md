# HEC-1 0-LLM validation-search

Post-course exposed-development comparator; it enters no Harness and opens no held-out Outcome.

| item | value |
| --- | ---: |
| units considered / scoreable | 26 / 23 |
| candidates per unit | 24 |
| any safe Support candidate | 17 |
| Support identity fallback | 9 |
| material gain at evaluation | 15 |
| clears evaluation gate | 7 |
| cumulative evaluation gain | +3.131950 |
| Consumer fits | 899 |
| LLM calls | 0 |

Selection histogram: `{"identity": 9, "impute_ar": 1, "outlier_iqr": 6, "outlier_mad": 5, "winsorize": 5}`.

| unit | selected on Support | safe candidates | evaluation gain | evaluation gate |
| --- | --- | ---: | ---: | --- |
| [0:40] x 1176 | `impute_ar` | 1 | +0.223241 | FAIL |
| [0:40] x 1896 | `outlier_mad` | 1 | +0.131149 | FAIL |
| [0:40] x 2136 | `winsorize` | 2 | +0.365043 | PASS |
| [0:40] x 2376 | `identity` | 0 | +0.000000 | FAIL |
| [0:40] x 2616 | `outlier_mad` | 1 | +0.223272 | PASS |
| [0:40] x 2856 | `outlier_mad` | 2 | — | UNSCOREABLE |
| [0:40] x 3576 | `identity` | 0 | +0.000000 | FAIL |
| [40:80] x 1176 | `outlier_iqr` | 1 | +0.173852 | PASS |
| [40:80] x 1416 | `identity` | 0 | +0.000000 | FAIL |
| [40:80] x 1656 | `identity` | 0 | +0.000000 | FAIL |
| [40:80] x 2136 | `outlier_mad` | 3 | +0.079427 | PASS |
| [40:80] x 2376 | `outlier_iqr` | 2 | +0.109421 | PASS |
| [40:80] x 2616 | `outlier_iqr` | 3 | +0.122494 | FAIL |
| [40:80] x 2856 | `identity` | 0 | — | UNSCOREABLE |
| [40:80] x 3576 | `identity` | 0 | +0.000000 | FAIL |
| [40:80] x 3816 | `outlier_iqr` | 1 | +0.161286 | FAIL |
| [80:120] x 1176 | `outlier_iqr` | 2 | +0.269745 | PASS |
| [80:120] x 1416 | `winsorize` | 2 | +0.048208 | FAIL |
| [80:120] x 1896 | `winsorize` | 1 | +0.047195 | PASS |
| [80:120] x 2136 | `winsorize` | 1 | -0.060131 | FAIL |
| [80:120] x 2376 | `outlier_iqr` | 11 | +0.037708 | FAIL |
| [80:120] x 2616 | `identity` | 0 | +0.000000 | FAIL |
| [80:120] x 2856 | `winsorize` | 1 | +0.504305 | FAIL |
| [120:160] x 1176 | `outlier_mad` | 1 | +0.695735 | FAIL |
| [120:160] x 1416 | `identity` | 0 | +0.000000 | FAIL |
| [120:160] x 1656 | `identity` | 0 | — | UNSCOREABLE |

