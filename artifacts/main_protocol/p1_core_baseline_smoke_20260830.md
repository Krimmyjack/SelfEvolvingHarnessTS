# P1 Core baseline smoke

**Verdict: `P1_CORE_BASELINE_SMOKE_PASS__P2_FORECAST_PILOT_RELEASED`. Overall P1 complete: `True`. P2 release: `True`.**

This is an infrastructure/contract smoke. It makes no performance, headroom, treatment, or capability claim.

Natural Final outcome reads: **0**. Development Query evaluations: **0**.

## Component gates

| task | component pass | Common DSL | methods |
|---|---|---|---:|
| forecast | `True` | `PASS` | 13 |
| classification | `True` | `PASS` | 13 |
| anomaly_detection | `True` | `PASS` | 13 |

## Unified Core rows

| task | method | contract | behavior | selected | logical fits | raw fits |
|---|---|---|---|---|---:|---:|
| forecast | Identity | `PASS` | `EVALUATED` | `identity` | 2 | 2 |
| forecast | Best Fixed Per-task | `PASS` | `EVALUATED` | `winsorize` | 2 | 2 |
| forecast | Fixed Linear-impute | `PASS` | `EVALUATED` | `impute_linear` | 2 | 2 |
| forecast | Fixed Hampel | `PASS` | `EVALUATED` | `hampel_filter` | 2 | 2 |
| forecast | Fixed Winsor | `PASS` | `EVALUATED` | `winsorize` | 2 | 2 |
| forecast | Fixed IQR | `PASS` | `EVALUATED` | `outlier_iqr` | 2 | 2 |
| forecast | Parallel Best-of-N@4 | `PASS` | `EVALUATED` | `winsorize` | 4 | 4 |
| forecast | Sequential Refinement@4 | `PASS` | `EVALUATED` | `winsorize` | 3 | 3 |
| forecast | Frozen H0 | `PASS` | `EVALUATED` | `identity` | 1 | 1 |
| forecast | Static | `PASS` | `EVALUATED` | `identity` | 2 | 2 |
| forecast | A3-reset | `PASS` | `ABSTAINED` | `identity` | 0 | 0 |
| forecast | K0-fixed | `PASS` | `ABSTAINED` | `identity` | 0 | 0 |
| forecast | A5-online | `PASS` | `ABSTAINED` | `identity` | 0 | 0 |
| classification | Identity | `PASS` | `EVALUATED` | `identity` | 2 | 2 |
| classification | Best Fixed Per-task | `PASS` | `EVALUATED` | `outlier_iqr` | 2 | 2 |
| classification | Fixed Linear-impute | `PASS` | `EVALUATED` | `impute_linear` | 2 | 2 |
| classification | Fixed Hampel | `PASS` | `EVALUATED` | `hampel_filter` | 2 | 2 |
| classification | Fixed Winsor | `PASS` | `EVALUATED` | `winsorize` | 2 | 2 |
| classification | Fixed IQR | `PASS` | `EVALUATED` | `outlier_iqr` | 2 | 2 |
| classification | Parallel Best-of-N@4 | `PASS` | `EVALUATED` | `hampel_filter` | 4 | 4 |
| classification | Sequential Refinement@4 | `PASS` | `EVALUATED` | `winsorize` | 3 | 3 |
| classification | Frozen H0 | `PASS` | `EVALUATED` | `identity` | 1 | 1 |
| classification | Static | `PASS` | `EVALUATED` | `identity` | 2 | 2 |
| classification | A3-reset | `PASS` | `ABSTAINED` | `identity` | 0 | 0 |
| classification | K0-fixed | `PASS` | `ABSTAINED` | `identity` | 0 | 0 |
| classification | A5-online | `PASS` | `ABSTAINED` | `identity` | 0 | 0 |
| anomaly_detection | Identity | `PASS` | `EVALUATED` | `identity` | 2 | 2 |
| anomaly_detection | Best Fixed Per-task | `PASS` | `EVALUATED` | `hampel_filter` | 2 | 2 |
| anomaly_detection | Fixed Linear-impute | `PASS` | `EVALUATED` | `impute_linear` | 2 | 2 |
| anomaly_detection | Fixed Hampel | `PASS` | `EVALUATED` | `hampel_filter` | 2 | 2 |
| anomaly_detection | Fixed Winsor | `PASS` | `EVALUATED` | `winsorize` | 2 | 2 |
| anomaly_detection | Fixed IQR | `PASS` | `EVALUATED` | `outlier_iqr` | 2 | 2 |
| anomaly_detection | Parallel Best-of-N@4 | `PASS` | `EVALUATED` | `impute_linear` | 4 | 6 |
| anomaly_detection | Sequential Refinement@4 | `PASS` | `EVALUATED` | `hampel_filter` | 3 | 4 |
| anomaly_detection | Frozen H0 | `PASS` | `EVALUATED` | `winsorize` | 1 | 2 |
| anomaly_detection | Static | `PASS` | `EVALUATED` | `identity` | 2 | 2 |
| anomaly_detection | A3-reset | `PASS` | `ABSTAINED` | `identity` | 2 | 6 |
| anomaly_detection | K0-fixed | `PASS` | `ABSTAINED` | `identity` | 2 | 6 |
| anomaly_detection | A5-online | `PASS` | `ABSTAINED` | `identity` | 2 | 6 |

## Boundary and release

- Forecast component execution by this master: `False`.
- AegisTS-style bounded spike: `STRUCTURALLY_INCOMPATIBLE` (blocking: `False`).
- AD method gate: `NOT_RELEASED_BY_P1`; current first fault: `#44a-r2 PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED / INVERTED_EFFECT_OBSERVED`.
- P2 release authorizes only the Forecast single-flow pilot; it does not authorize AD Evolution or Natural Final.
- P2 complete: `False`.
- Live/Natural-Final outcome release: `False`.

Machine-readable detail: `artifacts/main_protocol/p1_core_baseline_smoke_20260830.json`.
