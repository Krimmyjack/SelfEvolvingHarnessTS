# T3 (#39) task-conditioned proposal exam -- TASK_CONDITIONED_PROPOSALS_CONFIRMED

- protocol: `t3_task_exam_v1` (evidence grade POSITIVE_CONTROL, permanent)
- Part 0 checkpoint: `bd5922d` (6 files)
- backend: `gpt-5.6-sol` at `https://api.agicto.cn/v1`; request carries model+messages only (provider default sampling); returned models: gpt-5.6-sol
- cost: 6 LLM calls (24858 in, 1765 out tokens); 0 forecasting retrains; 0 AD evaluations
- store: `c8c1e452aac0` -- 0 Guidance / 0 Experience / 0 learned Skill; bootstrap procedures always on; one snapshot read by all six draws
- sampling: `second_sample` (first attempt SPENT_WRITE_FAILED)

## Smoke gate (A1, before any LLM call): PASSED

- [x] forecasting task_spec is the T2-wired ssi default, verbatim
- [x] anomaly_detection task_spec is the frozen revision string, verbatim
- [x] public inputs are byte-identical once task_spec is removed
- [x] the two prompts differ exactly at the task_spec bytes
- [x] the system prompt is one byte sequence across the arms
- [x] the AD arm is nowhere redescribed as forecasting
- [x] the F arm carries no anomaly-task wording
- [x] information wall: no T1/T1b gain reading or flip conclusion in either prompt
- [x] store state is 0 Guidance / 0 Experience / 0 learned Skill

## Answer keys (derived in-runner from frozen artifacts)

- aggregate F: hampel_filter, outlier_iqr, outlier_mad, winsorize
- aggregate AD: identity, hampel_filter, outlier_iqr, outlier_mad
- risk F: outlier_iqr, outlier_mad, winsorize
- risk AD: identity (+ abstain credited)
- matches the frozen expectation: True

## Draws (order forecasting, anomaly_detection, forecasting, anomaly_detection, forecasting, anomaly_detection)

| # | arm | classification | top1 | shortlist | retries | returned model |
|---|-----|----------------|------|-----------|---------|----------------|
| 1 | forecasting | VALID_PROPOSE | hampel_filter | hampel_filter, outlier_mad | 0 | gpt-5.6-sol |
| 2 | anomaly_detection | VALID_PROPOSE | identity | identity | 0 | gpt-5.6-sol |
| 3 | forecasting | VALID_PROPOSE | hampel_filter | hampel_filter, outlier_mad | 0 | gpt-5.6-sol |
| 4 | anomaly_detection | VALID_PROPOSE | identity | identity | 0 | gpt-5.6-sol |
| 5 | forecasting | VALID_PROPOSE | hampel_filter | hampel_filter | 0 | gpt-5.6-sol |
| 6 | anomaly_detection | VALID_PROPOSE | identity | identity | 0 | gpt-5.6-sol |

## Distance matrix

- same-task pairs: T3EXAM_F1/T3EXAM_F3=0.0000, T3EXAM_F1/T3EXAM_F5=0.5000, T3EXAM_F3/T3EXAM_F5=0.5000, T3EXAM_AD2/T3EXAM_AD4=0.0000, T3EXAM_AD2/T3EXAM_AD6=0.0000, T3EXAM_AD4/T3EXAM_AD6=0.0000
- cross-task pairs: T3EXAM_F1/T3EXAM_AD2=1.0000, T3EXAM_F1/T3EXAM_AD4=1.0000, T3EXAM_F1/T3EXAM_AD6=1.0000, T3EXAM_F3/T3EXAM_AD2=1.0000, T3EXAM_F3/T3EXAM_AD4=1.0000, T3EXAM_F3/T3EXAM_AD6=1.0000, T3EXAM_F5/T3EXAM_AD2=1.0000, T3EXAM_F5/T3EXAM_AD4=1.0000, T3EXAM_F5/T3EXAM_AD6=1.0000
- min cross-task: 1.0; max same-task: 0.5; complete separation: True

## Verdict

**TASK_CONDITIONED_PROPOSALS_CONFIRMED** -- complete separation and 3/3 aggregate direction on both arms; the Risk layer is not 3/3 (expected to be reachable only with Experience -- the T4 entry evidence)

> task semantics are visible at deployment by construction; this reading proves the proposals are conditioned on the task observation, not that the Agent discovered the task physics from the data, and it says nothing about execution or adoption

## Ambiguities (reported, not self-adjudicated)

- labeled second sample: first live attempt bash-joon3149 spent the six draws and died on mappingproxy json.dumps; draws 1-4 were lost to tail -20; recovered tail is F5=outlier_mad / AD6=identity; the first six draws are not scored
