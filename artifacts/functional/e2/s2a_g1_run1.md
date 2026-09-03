# S2a G1/G2 live (reduced course)

**S2a 判词:TREATMENT_EMPTY;自产卡:否;守卫三面:全零;核心数字 LLM 69 / fit 74**

producer did not yield a dual-gate POSITIVE; no forecast card

## Predictions

- **P1**: predicted hold; observed already landed (149 green)
- **P2**: predicted G2 three faces all zero; observed all_zero=True faces=[{'position': 1, 'unit_id': 'electricity_impulsive_outlier_03', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 1, 'unit_id': 'electricity_impulsive_outlier_03', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 2, 'unit_id': 'traffic_clean_identity_00', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 2, 'unit_id': 'traffic_clean_identity_00', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 3, 'unit_id': 'electricity_impulsive_outlier_01', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 3, 'unit_id': 'electricity_impulsive_outlier_01', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 4, 'unit_id': 'electricity_impulsive_outlier_04', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 4, 'unit_id': 'electricity_impulsive_outlier_04', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 5, 'unit_id': 'traffic_gap_00', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 5, 'unit_id': 'traffic_gap_00', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}]
- **P3**: predicted producer hit => strong beneficiary converts; observed forecast_card=False a5_strong_gains=[('electricity_impulsive_outlier_01', 1.552247133197474, {'supplied': 0, 'self_proposed': 1, 'dedup_swallowed': False, 'supplied_ids': [], 'self_proposed_ids': ['mad_broad_extreme_deviation'], 'dedup_detail': {'dedup_swallowed': False}}), ('electricity_impulsive_outlier_04', 3.1667303383627337, {'supplied': 0, 'self_proposed': 2, 'dedup_swallowed': False, 'supplied_ids': [], 'self_proposed_ids': ['outlier_iqr_extreme_deviation', 'outlier_mad_extreme_deviation'], 'dedup_detail': {'dedup_swallowed': False}})]
- **P4**: predicted N/A; observed N/A
- **P5**: predicted harm 0; observed harm_events=0
- **P6**: predicted K0 ≡ A3; observed material_divergences=none
- **P7**: predicted N/A; observed N/A (reduced course)
- **verdict**: predicted S2A_PORTABLE_REDUCED | S2A_PARTIAL | TREATMENT_EMPTY; observed TREATMENT_EMPTY

## Card version chain


## Units

| pos | unit | role | Static gain | A3 | K0 | A5 | A5 G2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | electricity_impulsive_outlier_03 | producer | +0.0000 | +0.0000 | +0.0000 | +0.0000 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 2 | traffic_clean_identity_00 | clean_identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 3 | electricity_impulsive_outlier_01 | strong_beneficiary_1 | +0.0000 | +1.5522 | +1.5522 | +1.5522 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 4 | electricity_impulsive_outlier_04 | strong_beneficiary_2 | +0.0000 | +3.7863 | +3.7863 | +3.1667 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 5 | traffic_gap_00 | gap_out_of_family_guard | +0.0000 | +0.0000 | +0.0000 | +0.0000 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
