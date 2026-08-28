# S2a iv decomposition (after metr_la/nn5 expand + repart1)

**status: S2A_CONFLICT_FIELD_UNAVAILABLE**

candidate programs: outlier_mad, winsorize
eligible: none
harm bar M=0.005  fits_total=278  llm=0
metr_la: registry=207 on_disk=89 cells=1 origins=792/888
nn5 on-disk usable=48 < CELL_WIDTH=60; structural skip

Semantics: classify_relation experience_memory.py:411-471; M=0.005 (signed_radius.py:40).

## Four-conjunction table

| P | cell | scope | same_P | pooled+ | harm | 4AND | relation | pooled | n_harm | min |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| \winsorize\ | \electricity_impulsive_outlier_00\ | False | True | True | False | False | POSITIVE | 1.2053 | 0 | 0.7313 |
| \winsorize\ | \electricity_impulsive_outlier_01\ | False | True | True | False | False | POSITIVE | 1.3905 | 0 | 0.0681 |
| \winsorize\ | \electricity_impulsive_outlier_02\ | False | True | True | False | False | POSITIVE | 1.1421 | 0 | 0.2425 |
| \winsorize\ | \electricity_impulsive_outlier_03\ | True | True | True | False | False | POSITIVE | 1.5000 | 0 | 0.8305 |
| \winsorize\ | \electricity_impulsive_outlier_04\ | True | True | True | False | False | POSITIVE | 1.2807 | 0 | 0.8042 |
| \winsorize\ | \traffic_impulsive_outlier_00\ | False | True | True | False | False | POSITIVE | 1.1599 | 0 | 0.4052 |
| \winsorize\ | \traffic_impulsive_outlier_01\ | False | True | True | False | False | POSITIVE | 1.1012 | 0 | 0.6471 |
| \winsorize\ | \traffic_impulsive_outlier_02\ | False | True | True | False | False | POSITIVE | 0.8828 | 0 | 0.5230 |
| \winsorize\ | \traffic_impulsive_outlier_03\ | False | True | True | False | False | POSITIVE | 1.2690 | 0 | 0.8197 |
| \winsorize\ | \traffic_impulsive_outlier_04\ | False | True | True | False | False | POSITIVE | 1.2617 | 0 | 0.3295 |
| \winsorize\ | \traffic_impulsive_outlier_05\ | False | True | True | False | False | POSITIVE | 1.1169 | 0 | 0.5921 |
| \outlier_mad\ | \metr_la_impulsive_outlier_00\ | True | True | True | False | False | POSITIVE | 10.3510 | 0 | 4.5670 |
| \winsorize\ | \metr_la_impulsive_outlier_00\ | True | True | True | False | False | POSITIVE | 8.6822 | 0 | 2.5970 |
| \outlier_mad\ | \electricity_impulsive_outlier_00\ | False | True | True | False | False | POSITIVE | 0.9597 | 0 | 0.5436 |
| \outlier_mad\ | \electricity_impulsive_outlier_01\ | False | True | True | False | False | POSITIVE | 1.0446 | 0 | 0.0578 |
| \outlier_mad\ | \electricity_impulsive_outlier_02\ | False | True | True | False | False | POSITIVE | 0.9163 | 0 | 0.1723 |
| \outlier_mad\ | \electricity_impulsive_outlier_03\ | True | True | True | False | False | POSITIVE | 1.1659 | 0 | 0.5606 |
| \outlier_mad\ | \electricity_impulsive_outlier_04\ | True | True | True | False | False | POSITIVE | 0.9628 | 0 | 0.5318 |
| \outlier_mad\ | \traffic_impulsive_outlier_00\ | False | True | True | False | False | POSITIVE | 0.8648 | 0 | 0.3245 |
| \outlier_mad\ | \traffic_impulsive_outlier_01\ | False | True | True | False | False | POSITIVE | 0.7313 | 0 | 0.3680 |
| \outlier_mad\ | \traffic_impulsive_outlier_02\ | False | True | True | False | False | POSITIVE | 0.6040 | 0 | 0.2337 |
| \outlier_mad\ | \traffic_impulsive_outlier_03\ | False | True | True | False | False | POSITIVE | 0.8616 | 0 | 0.5913 |
| \outlier_mad\ | \traffic_impulsive_outlier_04\ | False | True | True | False | False | POSITIVE | 1.0594 | 0 | 0.3491 |
| \outlier_mad\ | \traffic_impulsive_outlier_05\ | False | True | True | False | False | POSITIVE | 0.9531 | 0 | 0.3439 |

Zero four-conjunction hits after iv + authorized expand. G1/G2 not opened.
