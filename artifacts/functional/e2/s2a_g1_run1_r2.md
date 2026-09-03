# S2a G1/G2 live (reduced course)

**S2a 判词:S2A_PORTABLE_REDUCED;自产卡:是;守卫三面:全零;核心数字 LLM 60 / fit 88**

forecast card compiled; at least one strong beneficiary converted; G2 silent; harm 0

## Transport (not part of verdict)

- source: M0_AGENT_*
- request: cpa-gpt-5.6-sol @ https://cpa.cpa-lab.me/v1
- first_returned_model: gpt-5.6-sol
- r1_inspect: api.agicto.cn + gpt-5.6-sol (agicto direct)
- r2_inspect: CPA relay when M0_AGENT_* set (host+model mapped together)
- note: transport-layer difference only; not part of the verdict

## Predictions

- **P1**: predicted hold; observed already landed (149 green)
- **P2**: predicted G2 three faces all zero; observed all_zero=True faces=[{'position': 1, 'unit_id': 'electricity_impulsive_outlier_03', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 1, 'unit_id': 'electricity_impulsive_outlier_03', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 2, 'unit_id': 'traffic_clean_identity_00', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 2, 'unit_id': 'traffic_clean_identity_00', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 3, 'unit_id': 'electricity_impulsive_outlier_01', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 3, 'unit_id': 'electricity_impulsive_outlier_01', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 4, 'unit_id': 'electricity_impulsive_outlier_04', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 4, 'unit_id': 'electricity_impulsive_outlier_04', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 5, 'unit_id': 'traffic_gap_00', 'arm': 'K0-fixed', 'retrieval': 0, 'scope_match': 0, 'supply': 0}, {'position': 5, 'unit_id': 'traffic_gap_00', 'arm': 'A5-online', 'retrieval': 0, 'scope_match': 0, 'supply': 0}]
- **P3**: predicted producer hit => strong beneficiary converts; observed forecast_card=True a5_strong_gains=[('electricity_impulsive_outlier_01', 1.4053390300447028, {'supplied': 0, 'self_proposed': 1, 'dedup_swallowed': False, 'supplied_ids': [], 'self_proposed_ids': ['hampel_local_deviation'], 'dedup_detail': {'skill_id': 's2a_forecast_supply_v0', 'scope_match': False, 'card_in_view': False, 'supplied_in_pool': False, 'self_proposed_same_program': [{'candidate_id': 'hampel_local_deviation', 'operators': ['hampel_filter']}], 'dedup_swallowed': False, 'why': 'not the dedup case: Scope did not match'}}), ('electricity_impulsive_outlier_04', 3.786303416048806, {'supplied': 1, 'self_proposed': 1, 'dedup_swallowed': False, 'supplied_ids': ['cand_skill_s2a_forecast_supply_v0'], 'self_proposed_ids': ['robust_mad_outlier_repair'], 'dedup_detail': {'skill_id': 's2a_forecast_supply_v0', 'scope_match': True, 'card_in_view': True, 'supplied_in_pool': True, 'self_proposed_same_program': [{'candidate_id': 'robust_mad_outlier_repair', 'operators': ['outlier_mad']}], 'dedup_swallowed': False, 'why': 'not the dedup case: the supply did reach the pool'}})]
- **P4**: predicted N/A; observed N/A
- **P5**: predicted harm 0; observed harm_events=0
- **P6**: predicted K0 ≡ A3; observed material_divergences=[{'unit': 'electricity_impulsive_outlier_03', 'abs_delta': 2.317317696753431}, {'unit': 'electricity_impulsive_outlier_01', 'abs_delta': 0.14690810315277125}]
- **P7**: predicted N/A; observed N/A (reduced course)
- **verdict**: predicted S2A_PORTABLE_REDUCED | S2A_PARTIAL | TREATMENT_EMPTY; observed S2A_PORTABLE_REDUCED

## Card version chain

- v0 sha=b5058018ccb8 trigger=ladder_v2_compile_supply_tier

## Units

| pos | unit | role | Static gain | A3 | K0 | A5 | A5 G2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | electricity_impulsive_outlier_03 | producer | +0.0000 | +0.0000 | +2.3173 | +2.3173 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 2 | traffic_clean_identity_00 | clean_identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 3 | electricity_impulsive_outlier_01 | strong_beneficiary_1 | +0.0000 | +1.5522 | +1.4053 | +1.4053 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 4 | electricity_impulsive_outlier_04 | strong_beneficiary_2 | +0.0000 | +3.9069 | +3.9069 | +3.7863 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
| 5 | traffic_gap_00 | gap_out_of_family_guard | +0.0000 | +0.0000 | +0.0000 | +0.0000 | {'retrieval': 0, 'scope_match': 0, 'supply': 0} |
