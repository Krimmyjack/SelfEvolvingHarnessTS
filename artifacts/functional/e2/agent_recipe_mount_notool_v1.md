# agent-recipe-mount, no-tool control arm v1

**Verdict: `TOOL_ARM_DOMINATES`**

The tool arm measured behaviour: the Agent routed to the mounted `batch_recipe`, reused its Experience entry on the same cell and searched again when the Consumer changed. This arm prices the other half. Same three requests, same schema, same prompt, no binding: the Agent has only `summarize_series` and `localize_regions`, and every plan it returns is its own.

**Engineering demonstration, not authorization evidence.** No Skill is written, no TRY right is granted, no Episode is promoted, no Fast or Slow update runs, and no snapshot pointer moves.

The tool arm was **not re-run**: `artifacts/functional/e2/agent_recipe_mount_v1.json` (`agent_recipe_mount_v1`, verdict `AGENT_ROUTES_AND_REUSES`, 8 LLM calls) is read verbatim for the comparison.

## 0. What differs from the tool arm, and what does not

Identical, and imported from the tool-arm runner rather than copied:

- request framing of E1/E2/E3 (EPISODE_PLAN)
- batch block, program menu, prior-experience rendering and how_to_read (_public_input)
- workspace tool budget
- stage note, except for the one appended sentence below
- harness view: the same h0 snapshot resolved for role=fast
- offline scoring: the same OfflinePlanEvaluator

Deliberate differences:

- tool supply: no batch_recipe binding, so allowed_local_tools carries the two observation tools only
- decision enum gains OWN_OBSERVATION, and the stage note gains one sentence naming it, because the tool arm's three values leave an Agent with no tool and no prior entry unable to state a treatment plan honestly

Not done: no hint that a recipe exists, no suggestion of a program, no suggestion to exclude any series, no threshold and no target

Tools declared to the Agent in this arm: `summarize_series`, `localize_regions`.

Harness change surface: **none**. no Harness file is edited by this arm. The gateway is used through its frozen binding=None path; the runner subclasses it only to look the batch_plan stage up as inspect, which serves exactly the frozen two-tool supply.

## 1. Verdict

Pre-registered before the first LLM call. Outcome per episode is read off the delayed aggregate gain, recomputed offline for both arms; `TOOL` if the tool arm leads by more than 0.005, `NOTOOL` if the control arm does, `TIE` otherwise.

1. TOOL_ARM_DOMINATES: every compared episode is TOOL
2. HAND_ROLLED_COMPETITIVE: no compared episode is TOOL
3. MIXED_BY_EPISODE: anything else

Matched: **`TOOL_ARM_DOMINATES`** -- every compared episode favours the tool arm by more than 0.005.

## 2. Arm comparison, episode by episode

| episode | arm | behaviour | plan | support | delayed | harmed (support/delayed) | LLM | tool calls (refused) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | tool | `TOOL_CALLED_AND_ADOPTED` | `outlier_iqr` minus 6 | +0.665277 | +1.047075 | 0 / 0 | 4 | 2 (0) |
| E1 | no tool | `IDENTITY_NO_TREATMENT` | `identity` minus nothing | +0.000000 | +0.000000 | 0 / 0 | 8 | 6 (1) |
| E1 | **margin** | `TOOL` | delayed tool - no tool | | +1.047075 | | | |
| E2 | tool | `EXPERIENCE_REUSED_NO_TOOL_CALL` | `outlier_iqr` minus 6 | +0.665277 | +1.047075 | 0 / 0 | 2 | 0 (0) |
| E2 | no tool | `IDENTITY_NO_TREATMENT` | `identity` minus nothing | +0.000000 | +0.000000 | 0 / 0 | 6 | 5 (0) |
| E2 | **margin** | `TOOL` | delayed tool - no tool | | +1.047075 | | | |
| E3 | tool | `TOOL_CALLED_AND_ADOPTED` | `outlier_iqr` minus 3, 8 | +0.303184 | +0.355472 | 0 / 0 | 2 | 1 (0) |
| E3 | no tool | `HAND_ROLLED_PLAN` | `hampel_filter` minus nothing | +0.103591 | +0.130606 | 1 / 0 | 7 | 6 (0) |
| E3 | **margin** | `TOOL` | delayed tool - no tool | | +0.224866 | | | |

## 3. What the no-tool arm actually proposed

### E1 -- traffic x pooled

- decision: `IDENTITY_NO_TREATMENT` (behaviour `IDENTITY_NO_TREATMENT`)
- plan: `identity` minus nothing
- offline readout: support +0.000000, delayed +0.000000, harmed evaluation series 0 (support) / 0 (delayed)
- observation tool calls: 6 [{'series_uid': '0'}, {'series_uid': '1'}, {'series_uid': '2'}, {'series_uid': '3'}, {'series_uid': '4'}, {'series_uid': '5'}, {'series_uid': '6'}]
- experience entries visible: 0; cited: none
- LLM calls: 8, retries: 0 
- reason given: Public observations show localized level and deviation signals in the inspected series, but support is not uniform and six training series could not be inspected within the tool budget. Without batch-wide evidence justifying a common repair, leave the batch untreated.

### E2 -- traffic x pooled

- decision: `IDENTITY_NO_TREATMENT` (behaviour `IDENTITY_NO_TREATMENT`)
- plan: `identity` minus nothing
- offline readout: support +0.000000, delayed +0.000000, harmed evaluation series 0 (support) / 0 (delayed)
- observation tool calls: 5 [{'series_uid': '0'}, {'series_uid': '0'}, {'series_uid': '1'}, {'series_uid': '2'}, {'series_uid': '3'}]
- experience entries visible: 1; cited: none
- LLM calls: 6, retries: 0 
- reason given: Public observations show no missingness, while level and deviation signals vary across inspected series and do not establish a single safe batch-wide repair. Leave the batch untreated rather than apply an uncertain transformation.

### E3 -- traffic x per_channel

- decision: `OWN_OBSERVATION` (behaviour `HAND_ROLLED_PLAN`)
- plan: `hampel_filter` minus nothing
- offline readout: support +0.103591, delayed +0.130606, harmed evaluation series 1 (support) / 0 (delayed)
- observation tool calls: 6 [{'series_uid': '0'}, {'series_uid': '1'}, {'series_uid': '2'}, {'series_uid': '3'}, {'series_uid': '4'}, {'series_uid': '5'}]
- experience entries visible: 2; cited: none
- LLM calls: 7, retries: 0 
- reason given: Public summaries for all inspected series show substantial local robust deviations, while missingness is absent. A Hampel filter is a localized, conservative treatment for these observed outlier signals without excluding any training series.

## 4. Experience entries written

Same episode mechanism as the tool arm, no new store, no Skill. The provenance tag is deliberately different -- `agent_hand_rolled_engineering` here against `batch_recipe_tool_v2_engineering` there -- so a later audit can tell a tool-mediated measurement from a hand-rolled one without reading anything else. Both carry `counts_as_unguided_exploration: false`.

| episode_id | task_consumer_key | plan | support | delayed | relation | local_status | provenance |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| `E1` | `batch:traffic\|consumer:pooled` | `identity` minus nothing | +0.000000 | +0.000000 | ABSTAIN | EPISODE_ONLY | `agent_hand_rolled_engineering` |
| `E2` | `batch:traffic\|consumer:pooled` | `identity` minus nothing | +0.000000 | +0.000000 | ABSTAIN | EPISODE_ONLY | `agent_hand_rolled_engineering` |
| `E3` | `batch:traffic\|consumer:per_channel` | `hampel_filter` minus nothing | +0.103591 | +0.130606 | POSITIVE | EPISODE_ONLY | `agent_hand_rolled_engineering` |

## 5. Cost

| item | this arm | tool arm |
| --- | ---: | ---: |
| LLM calls | 21 (budget 30) | 8 |
| recipe searches executed | 0 (no binding) | 2 |
| wall seconds | 299.5 | -- |

## 6. What this does not say

- It does not authorize anything, in either arm.
- It is one cohort and three episodes on one model. A per-episode outcome is a comparison of two single draws, not a rate.
- The delayed column is in-selection for the tool arm, because the recipe's own adoption gate reads it. For this arm it is out of selection: the Agent never saw a delayed number. That asymmetry is a property of the mounted tool and it favours the tool arm.
- It does not measure whether a hand-rolled plan would improve with a larger Workspace tool budget or more episodes.

## Provenance

- model: `gpt-5.6-luna` at `https://api.agicto.cn/v1`
- scoring: the tool arm's `OfflinePlanEvaluator`, i.e. `run_batch_composition_headroom._evaluate_assignment` + `_gain_rows` on the same windows and Consumer variant, imported not reimplemented
- windows: traffic development origins only: Support 1104/1368, delayed 1800, farthest read 1848, sealed_from_index 3072 never approached
- stage: `batch-plan/1-own-observation`, schema `batch-plan/1-own-observation` declared inside this runner

