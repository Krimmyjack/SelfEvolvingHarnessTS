# Forecast P1 Core baseline smoke

**Verdict: `FORECAST_P1_CORE_BASELINE_SMOKE_PASS`. Forecast component pass: `True`. Overall P1 complete: `False`.**

This is an infrastructure/contract smoke on exposed KDD development data. It makes no performance, headroom, treatment, or capability claim.

Natural Final outcome reads: **0**. Development Query evaluations: **0**.

## Core methods

| method | status | selected | fits | LLM calls | tokens |
|---|---|---|---:|---:|---:|
| Identity | `PASS` | `identity` | 2 | 0 | 0 |
| Best Fixed Per-task | `PASS` | `winsorize` | 2 | 0 | 0 |
| Fixed Linear-impute | `PASS` | `impute_linear` | 2 | 0 | 0 |
| Fixed Hampel | `PASS` | `hampel_filter` | 2 | 0 | 0 |
| Fixed Winsor | `PASS` | `winsorize` | 2 | 0 | 0 |
| Fixed IQR | `PASS` | `outlier_iqr` | 2 | 0 | 0 |
| Parallel Best-of-N@4 | `PASS` | `winsorize` | 4 | 0 | 0 |
| Sequential Refinement@4 | `PASS` | `winsorize` | 3 | 0 | 0 |
| Frozen H0 | `PASS` | `identity` | 1 | 2 | 0 |
| Static | `PASS` | `identity` | 2 | 0 | 0 |
| A3-reset | `PASS` | `identity` | 0 | 2 | 0 |
| K0-fixed | `PASS` | `identity` | 0 | 2 | 0 |
| A5-online | `PASS` | `identity` | 0 | 2 | 0 |

## Boundary and release

- Common DSL contract: `PASS`.
- AegisTS-style spike: `STRUCTURALLY_INCOMPATIBLE` (non-blocking).
- P2 is **not** released by this Forecast-only tranche; Classification and AD P1 components remain pending.
- Final outcomes remain sealed.

Machine-readable detail: `artifacts/main_protocol/forecast_p1_core_smoke_20260830.json`.
