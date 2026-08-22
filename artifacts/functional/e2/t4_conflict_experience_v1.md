# T4 (#40) task-keyed conflict Experience -- PARTIAL_EXPERIENCE_CONDITIONING

- protocol: `t4_conflict_experience_v1` (evidence grade POSITIVE_CONTROL, permanent)
- Part 0 checkpoint: `fd29501` (5 files)
- backend: `gpt-5.6-sol` at `https://api.agicto.cn/v1`; returned models: gpt-5.6-sol
- cost: 6 LLM calls; 0 forecasting retrains; 0 AD evaluations
- store: `e6d42cc5c9ae` -- the #39 h0 snapshot exactly: 0 Guidance / 0 Experience / 0 learned Skill plus the three always-on bootstrap procedures; this round adds ten Experience episodes and nothing else

## Memory keys

- forecasting: `forecast|pooled_ridge_a1|sMASE`
- anomaly_detection: `anomaly_detection|ad_ridge_train_v3|F1`

## The ten episodes (relation derived mechanically)

| arm | program | relation | aggregate | direction | harmed / read | min per-series |
|-----|---------|----------|-----------|-----------|---------------|----------------|
| forecasting | identity | ABSTAIN | +0.000000 | unchanged | 0 / 4 | +0.000000 |
| forecasting | outlier_iqr | POSITIVE | +0.272266 | improved | 0 / 4 | +0.058555 |
| forecasting | outlier_mad | POSITIVE | +0.325546 | improved | 0 / 4 | +0.167433 |
| forecasting | hampel_filter | CONFLICT | +0.064758 | improved | 1 / 4 | -0.090412 |
| forecasting | winsorize | POSITIVE | +0.405862 | improved | 0 / 4 | +0.332025 |
| anomaly_detection | identity | ABSTAIN | +0.000000 | unchanged | 0 / 12 | +0.000000 |
| anomaly_detection | outlier_iqr | CONFLICT | +0.008580 | improved | 5 / 12 | -0.272727 |
| anomaly_detection | outlier_mad | CONFLICT | +0.041336 | improved | 4 / 12 | -0.200000 |
| anomaly_detection | hampel_filter | CONFLICT | +0.041336 | improved | 4 / 12 | -0.200000 |
| anomaly_detection | winsorize | NEGATIVE | -0.167236 | degraded | 9 / 12 | -0.504202 |

- matches the pre-registered expectation: True
- NEUTRAL produced: none

## Write (Runtime)

- 0 -> 10 episodes on one TTHAMethod; read back from the runtime: True
- evidence levels ['DELAYED']; local statuses ['EPISODE_ONLY']; no promotion: True

## B4 category acceptance: PASSED

- [x] F retrieval returns a locally harmful CONFLICT card
- [x] F retrieval returns a harmless POSITIVE card from the repair family
- [x] AD retrieval returns a NEGATIVE card
- [x] AD retrieval returns a CONFLICT card from a repair program
- [x] no card crosses tasks in either arm
- [x] the card face carries consumer, aggregate direction, harmed count and worst single-series reading, and prescribes nothing
- [x] both arms retrieved something

## C2 three-way prompt assertions: PASSED

- [x] T4-forecasting vs #39-forecasting: the user message is byte-identical
- [x] T4-forecasting vs #39-forecasting: the system message differs only by the experience block
- [x] T4-anomaly_detection vs #39-anomaly_detection: the user message is byte-identical
- [x] T4-anomaly_detection vs #39-anomaly_detection: the system message differs only by the experience block
- [x] T4-F vs T4-AD: the user messages differ exactly at the task_spec bytes
- [x] T4-F vs T4-AD: the system messages differ only by their own experience blocks
- [x] each arm's card is actually present in the bytes that will be sent

## Draws (order forecasting, anomaly_detection, forecasting, anomaly_detection, forecasting, anomaly_detection)

| # | arm | classification | top1 | shortlist | cards |
|---|-----|----------------|------|-----------|-------|
| 1 | forecasting | VALID_PROPOSE | outlier_iqr | outlier_iqr | t4_forecasting_outlier_iqr_replay_v1, t4_forecasting_hampel_filter_replay_v1 |
| 2 | anomaly_detection | VALID_PROPOSE | hampel_filter | hampel_filter | t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1 |
| 3 | forecasting | VALID_PROPOSE | outlier_iqr | outlier_iqr | t4_forecasting_outlier_iqr_replay_v1, t4_forecasting_hampel_filter_replay_v1 |
| 4 | anomaly_detection | VALID_PROPOSE | hampel_filter | hampel_filter | t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1 |
| 5 | forecasting | VALID_PROPOSE | outlier_iqr | outlier_iqr | t4_forecasting_outlier_iqr_replay_v1, t4_forecasting_hampel_filter_replay_v1 |
| 6 | anomaly_detection | VALID_PROPOSE | hampel_filter | hampel_filter | t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1 |

## Displacement against #39

| arm | baseline top-1 | T4 top-1 | moved | baseline Risk | T4 Risk |
|-----|----------------|----------|-------|---------------|---------|
| forecasting | hampel_filter, hampel_filter, hampel_filter | outlier_iqr, outlier_iqr, outlier_iqr | True | 0/3 | 3/3 |
| anomaly_detection | identity, identity, identity | hampel_filter, hampel_filter, hampel_filter | True | 3/3 | 0/3 |

## Distance matrix

- min cross-task: 1.0; max same-task: 0.0; complete separation: True

## Verdict

**PARTIAL_EXPERIENCE_CONDITIONING** -- an intermediate state: F 3/3 safe, AD 0/3 safe, separation kept True, AD risk regression True


## Findings handed back (no LLM cost)

### The store sha moved, the store did not

runtime_bundle_sha = sha(harness_content_sha, operator_bundle_sha, dependency_shas, three compiler versions).  harness_content_sha did not move.  dependency_shas covers ttha:fast_agent and ttha:method by name, and this round's authorized Memory diff edits both, so the bundle sha is obliged to move.  The Harness the Agent read is byte-identical, which C2 proves independently: stripping each arm's experience block from its system prompt returns #39's system bytes exactly.

- the store base is #39's h0 snapshot and nothing was written to it; the ten episodes live in the method instance.  The bundle sha is a runtime-code identity, not a store identity
- methods/ttha/experience_memory.py is NOT in the compiler's dependency list (contracts + runtime/* + ttha:{agent_core, fast_agent, method, public_tools, retrieval, schema_contracts, slow_agent}).  A Memory-only edit that touched no other module would leave runtime_bundle_sha unchanged -- the bundle identity does not currently cover the Memory surface.  Reported, not fixed: the registry is not this round's change surface.

### Why the AD arm regressed

AD moved off identity 3/3 onto hampel_filter 3/3; its aggregate layer stayed 3/3 (hampel_filter is inside the AD aggregate key at +0.0413 macro-F1) and its Risk layer fell from 3/3 to 0/3.  The AD arm made exactly the error F made in #39: an aggregate-appropriate, risk-inappropriate choice.

- cards the AD arm saw: t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1
- cards it could not see: t4_anomaly_detection_identity_replay_v1

identity under AD classifies as ABSTAIN, and ABSTAIN has no channel to the prompt on two independent counts: ContrastPack carries positive / negative / conflict slots only, and SignedEpisodeRetriever._hard_filter drops any episode whose workflow_signature is identity or unknown whenever allowed_operators is non-empty.  So the only AD experience that could reach the Agent was 'winsorize degraded' and 'hampel_filter improved the aggregate'.  The Memory has no way to say that doing nothing was the right call, and the card it could render pointed away from the answer.

the fact sentence leads with the aggregate direction and states the harmed count second.  Under F that ordering was harmless because the safe POSITIVE card was also present to contrast against; under AD there was no such contrast, so 'Aggregate direction: improved' stood alone.

> it does not show the key, the write path or the retrieval gate failed -- all three worked, B4 and C2 passed, zero cards crossed tasks and separation strengthened from 1.0>0.5 to 1.0>0.0.  What failed is the pack's expressive range.

Candidate next surfaces (not adjudicated here):

- give ContrastPack an abstain / do-nothing channel so an ABSTAIN episode can be rendered
- reconsider _hard_filter dropping identity signatures when the task's correct answer may be to leave the batch alone
- card presentation: lead with the risk reading rather than the aggregate direction

## Ambiguities (reported, not self-adjudicated)

- the store snapshot is not byte-identical to #39's: e6d42cc5c9aea28a85dd799b83b00be45aa0f3758fe5d5a0760f6baa4bc68699 vs c8c1e452aac0f2f940df40bdf808fa7187f56ece0723fe024d74064627ef298d
