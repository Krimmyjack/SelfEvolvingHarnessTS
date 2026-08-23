# #42e0 Skill v2 development replay

verdict: **RISK_SKILL_NO_TRIGGERING_CANDIDATE**

evidence_grade: DEVELOPMENT / same-context

Part 0 sha: `ad4f7b82373fc25b785d4f5e972517557e279f80`

run_id: `20260823T180847Z`

v2 materialized from frozen JSON entry (no Slow, no Temp h0s_v2).

Cost: LLM **22 / 32**; AD fit **12 / 24**; Slow 0; forecast retrain 0.

Does not count as a Capability or cross-domain claim.

| cell | retrieved v2 | pool | chosen | hampel p/c/pr | non-hampel prop/probe | relation |
|---|---|---|---|---|---|---|
| target_cpc/A3/r1 | False | ['identity'] |  | False/False/0 | 0/0 | — |
| target_cpc/A5/r1 | True | ['identity'] |  | False/False/0 | 0/0 | — |
| target_cpc/A3/r2 | False | ['identity'] |  | False/False/0 | 0/0 | — |
| target_cpc/A5/r2 | True | ['identity'] |  | False/False/0 | 0/0 | — |
| target_cpm/A5/r1 | True | ['identity'] |  | False/False/0 | 0/0 | — |
| target_cpm/A3/r1 | False | ['identity'] |  | False/False/0 | 0/0 | — |
| target_cpm/A5/r2 | True | ['identity', 'mad_tail_deviation'] | mad_tail_deviation | False/False/0 | 1/1 | [('outlier_mad', 'POSITIVE', 'LOCAL_ACTIVE')] |
| target_cpm/A3/r2 | False | ['identity', 'localized_outlier_mad', 'localized_outlier_iqr'] | localized_outlier_mad | False/False/0 | 2/1 | [('outlier_mad', 'POSITIVE', 'LOCAL_ACTIVE')] |

A3 hampel events 0 / probed 0; A5 hampel events 0 / probed 0
A3 non-hampel 3; A5 non-hampel 2
