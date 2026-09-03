# S3 pilot probe-policy

protocol: s3_pilot_probe_policy_v1
gate candidate: S3_EDIT_REJECTED
instrument_stop: None
note: 三臂同课程,菜单 oracle 项对消,Σgain 高 ⟺ 累计 regret 低

## Random-legal-edit draw

{"seed": 20260829, "edit": {"tie_break_rule": "prefer_supplied"}, "param": "tie_break_rule", "value": "prefer_supplied", "policy": {"skill_slot_merge_rule": "draft_does_not_displace_agent", "supply_reserved_probe_slots": 0, "probe_order_rule": "chosen_first_then_pool", "first_positive_stop": true, "winner_compare_rule": "first_positive_in_probe_order", "tie_break_rule": "prefer_supplied", "agent_proposals_kept": 1, "displacement_margin": 0.0}}

## LLM-edit proposal

illegal=False attempts=1

## Arm metrics (beneficiary 2/4/5)

| arm | n | cum_gain | llm_calls | receipts | harm | g2 |
| --- | --- | --- | --- | --- | --- | --- |
| no_edit | 3 | +3.3301 | 13 | 3 | 0 | 0 |
| random_edit | 3 | +1.5334 | 13 | 2 | 0 | 0 |
| llm_edit | 3 | +1.5334 | 11 | 3 | 0 | 0 |

## Units

| pos | unit | role | family | no_edit | random_edit | llm_edit |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | electricity_impulsive_outlier_00 | producer | electricity | +1.1381 | +0.0000 | +1.1381 |
| 2 | electricity_impulsive_outlier_02 | beneficiary | electricity | +1.7967 | +0.0000 | +0.0000 |
| 3 | traffic_impulsive_outlier_00 | producer | traffic | +1.3974 | +1.3974 | +0.0000 |
| 4 | traffic_impulsive_outlier_01 | beneficiary | traffic | +1.5334 | +1.5334 | +1.5334 |
| 5 | traffic_impulsive_outlier_02 | beneficiary | traffic | +0.0000 | +0.0000 | +0.0000 |
