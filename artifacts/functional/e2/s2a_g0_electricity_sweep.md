# S2a G0 electricity sweep

**status: S2_HOST_READY_FAIL_BOTH_SOURCES**

reasons: no_near_line_weak_beneficiary, no_identity_field
electricity impulse: 5  producer: 5  strong: 5  weak: 0  identity: 0
merged impulse: 11  producer: 11  strong: 11  weak: 0  identity: 0  gap: 1
fits this sweep: 75  prior traffic: 105  total: 180  elapsed_s: 24.9

## Pre-declared cut

TSL electricity.csv is the in-service UCI-family loader (321 numeric channels). Registry 370x1024 cannot host isomorphic origins 1104/1800. Pre-declared: 5 impulse cells x 60 = 300; leftover unused; gap reused from traffic.
usable=321 leftover=21 gap_reuse=traffic_gap_00 origins=1104/1800

## Merged headroom table

| unit | learnability | oracle | headroom | two_x | near_line |
| --- | --- | --- | ---: | --- | --- |
| `traffic_gap_00` | LEARNABLE | outlier_iqr | 0.1481 | True | False |
| `traffic_impulsive_outlier_00` | LEARNABLE | winsorize | 1.1599 | True | False |
| `traffic_impulsive_outlier_01` | LEARNABLE | winsorize | 1.1012 | True | False |
| `traffic_impulsive_outlier_02` | LEARNABLE | winsorize | 0.8828 | True | False |
| `traffic_impulsive_outlier_03` | LEARNABLE | winsorize | 1.2690 | True | False |
| `traffic_impulsive_outlier_04` | LEARNABLE | winsorize | 1.2617 | True | False |
| `traffic_impulsive_outlier_05` | LEARNABLE | winsorize | 1.1169 | True | False |
| `electricity_impulsive_outlier_00` | LEARNABLE | winsorize | 1.2053 | True | False |
| `electricity_impulsive_outlier_01` | LEARNABLE | winsorize | 1.3905 | True | False |
| `electricity_impulsive_outlier_02` | LEARNABLE | winsorize | 1.1421 | True | False |
| `electricity_impulsive_outlier_03` | LEARNABLE | winsorize | 1.5000 | True | False |
| `electricity_impulsive_outlier_04` | LEARNABLE | winsorize | 1.2807 | True | False |

Oracle files live under `artifacts/functional/e2/s2a_oracle/` and must not enter any arm prompt, store, or retrieval view.

