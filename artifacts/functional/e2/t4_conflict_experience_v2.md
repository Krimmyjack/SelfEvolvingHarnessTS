# T4b (#40b) abstain channel -- TASK_SEPARATION_REGRESSION

- protocol: `t4_conflict_experience_v2` (evidence grade POSITIVE_CONTROL, permanent)
- Part 0 checkpoint: `fbba86f` (10 files)
- only change: the card's expressive range (ContrastPack.abstain); keys, relation classification, card order, Consumer, menu, backend and protocol all frozen at #40
- cost: 6 LLM calls; 0 retrains; 0 AD evaluations; new_independent_evidence = 0

## B1 re-materialization

- 10 episodes, all fields identical to #40: True
- frozen readings identical to #40: True; task keys identical: True

## A4 old-behaviour assertions: PASSED

- [x] A4 forecasting: with abstain removed the card is byte-identical to #40's
- [x] A4 anomaly_detection: with abstain removed the card is byte-identical to #40's
- [x] A4 legacy shape (the other line's episodes): the old sentences are rendered unchanged and no fourth reference appears
- [x] A4 empty pack still renders nothing

## B2 abstain acceptance: PASSED

- [x] the AD pack now carries an abstain card, and it is identity/ABSTAIN
- [x] the AD card face renders the no-action baseline
- [x] the abstain card prescribes nothing
- [x] the F arm's abstain card is archived too (rendered or not)
- [x] no card crosses tasks in either arm, abstain included

## B4 category acceptance (re-run): PASSED

## B3 three-way prompt assertions: PASSED

- [x] #40b-forecasting vs #40-forecasting: the user message is byte-identical
- [x] #40b-forecasting vs #40-forecasting: the system message differs only by the abstain block
- [x] #40b-anomaly_detection vs #40-anomaly_detection: the user message is byte-identical
- [x] #40b-anomaly_detection vs #40-anomaly_detection: the system message differs only by the abstain block
- [x] #40b-F vs #40b-AD: the user messages differ exactly at the task_spec bytes
- [x] #40b-F vs #40b-AD: the system messages differ only by their own experience blocks
- [x] each arm's card is present in the bytes that will be sent

## Draws (order forecasting, anomaly_detection, forecasting, anomaly_detection, forecasting, anomaly_detection)

| # | arm | classification | top1 | shortlist | cards |
|---|-----|----------------|------|-----------|-------|
| 1 | forecasting | VALID_PROPOSE | outlier_iqr | outlier_iqr, hampel_filter, identity | t4_forecasting_outlier_iqr_replay_v1, t4_forecasting_hampel_filter_replay_v1, t4_forecasting_identity_replay_v1 |
| 2 | anomaly_detection | VALID_PROPOSE | identity | identity | t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1, t4_anomaly_detection_identity_replay_v1 |
| 3 | forecasting | VALID_PROPOSE | outlier_iqr | outlier_iqr | t4_forecasting_outlier_iqr_replay_v1, t4_forecasting_hampel_filter_replay_v1, t4_forecasting_identity_replay_v1 |
| 4 | anomaly_detection | VALID_PROPOSE | identity | identity | t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1, t4_anomaly_detection_identity_replay_v1 |
| 5 | forecasting | VALID_PROPOSE | outlier_iqr | outlier_iqr | t4_forecasting_outlier_iqr_replay_v1, t4_forecasting_hampel_filter_replay_v1, t4_forecasting_identity_replay_v1 |
| 6 | anomaly_detection | VALID_PROPOSE | hampel_filter | hampel_filter, identity | t4_anomaly_detection_winsorize_replay_v1, t4_anomaly_detection_hampel_filter_replay_v1, t4_anomaly_detection_identity_replay_v1 |

## Displacement (#39 -> #40 -> #40b)

| arm | #39 top-1 | #40 top-1 | #40b top-1 | Risk 39 | Risk 40 | Risk 40b |
|-----|-----------|-----------|------------|---------|---------|----------|
| forecasting | hampel_filter, hampel_filter, hampel_filter | outlier_iqr, outlier_iqr, outlier_iqr | outlier_iqr, outlier_iqr, outlier_iqr | 0/3 | 3/3 | 3/3 |
| anomaly_detection | identity, identity, identity | hampel_filter, hampel_filter, hampel_filter | identity, identity, hampel_filter | 3/3 | 0/3 | 2/3 |

## Distance matrix

- min cross-task 0.33333333333333337; max same-task 0.6666666666666667; complete separation False

## Verdict

**TASK_SEPARATION_REGRESSION** -- separation lost: min cross 0.33333333333333337 vs max same 0.6666666666666667; Risk F 3/3, AD 2/3

> POSITIVE_CONTROL, permanent: this shows Memory presentation can carry the correction, not that the Agent discovered anything.  Proposal-only; execution and adoption stay in T5


## Findings handed back (no LLM cost)

### Why separation did not hold

> the pre-registered ladder reads separation before the F/AD counts, and separation did not hold; this section explains the reading, it does not change it

- min cross-task 0.33333333333333337 (binding pair `T4EXAM_F1/T4EXAM_AD6`); max same-task 0.6666666666666667 (binding pair `T4EXAM_F1/T4EXAM_F3`)
- the Jaccard distance is computed over the proposed sets.  At the top-1 layer the two tasks stayed completely disjoint: F named ['outlier_iqr'], AD named ['hampel_filter', 'identity'], with no shared entry.

identity entered the shortlists of both arms.  It is the one menu entry the no-action baseline card speaks about, and it is legal under either task, so once both arms could see that evidence they both named it as an option.  Shared vocabulary raises cross-task overlap and same-task spread at once: ['T4EXAM_AD2', 'T4EXAM_AD4', 'T4EXAM_AD6', 'T4EXAM_F1'] carry identity in their shortlist.  Cross-task distance did not collapse because the tasks converged on an answer -- their top-1 choices never overlapped -- but because the shortlists now share one token.

### AD recovery

- Risk layer #39 / #40 / #40b: [3, 0, 2]
- top-1 #39 / #40 / #40b: [['identity', 'identity', 'identity'], ['hampel_filter', 'hampel_filter', 'hampel_filter'], ['identity', 'identity', 'hampel_filter']]

the channel reached the choice: two of three AD draws went back to identity after #40's three-for-three regression, and the third kept identity in its shortlist as the stated fallback.  The recovery is partial, not complete

### A1 reachability

#40b's A1 was written so the abstain channel does not depend on identity being a member of allowed_operators.  That was not a hypothetical: identity is absent from the operator registry entirely, so the rejected formulation would have produced an exam-only channel that the live Runtime could never open

- the registry is not this slice's change surface and nothing was touched; recorded so the main line can route it

> the T5 static seam reconnaissance runs only on CONFLICT_EXPERIENCE_CONDITIONS_PROPOSALS_CONFIRMED; this run did not reach that verdict, so _t5_seam_recon was not called and no recon section is part of this delivery

## Ambiguities (reported, not self-adjudicated)

- B1 asks for a literal episode.to_dict() comparison against the #40 artifact, but #40 persisted a per-episode summary rather than the serialized Episode; the 16 persisted fields, the frozen readings block and both task keys are compared instead, and this run records the full to_dict() so the literal check becomes possible later
