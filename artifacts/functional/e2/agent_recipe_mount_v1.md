# agent-recipe-mount micro v1

**Verdict: `AGENT_ROUTES_AND_REUSES`**

The batch recipe was mounted as a Workspace tool the Fast Agent may call, and a three-episode single-arm micro measured whether the Agent routes to it, reuses a prior Experience entry on the same cell, and searches again when the Consumer structure changes.

**Engineering demonstration, not authorization evidence.** No Skill is written, no TRY right is granted, no Episode is promoted, no Fast or Slow update runs, and no snapshot pointer moves.

> **Information wall.** batch_recipe returns delayed-window numbers, which the two pre-existing Workspace tools cannot see. The recipe uses the delayed window inside its own adoption gate, so mounting it widens what the Fast Agent can observe. This run therefore writes no Skill and tags every Experience entry provenance=batch_recipe_tool_v2_engineering; the tool must not be bound in a Task Episode run that produces authorization evidence.

## 0. What was mounted

- surface: Fast Agent Workspace tool supply (`evaluation/functional/task_episode_harness/agentic/gateway.py`)
- change: CohortScopePublicToolGateway gained an optional batch_recipe_binding: one extra tool name, its description, its two bounded argument enums, one extra allowed stage and the binding identity folded into context_sha
- default path: with batch_recipe_binding=None the class serves the same two tools on the same three stages and hashes the same context_sha payload as before
- unchanged: `OBSERVABLE_FEATURES`, `feature_context_sha`, `Judge`, `Metric`, `Operator DSL`, `Source Skill`, `Slow path`, `run_batch_composition_headroom (imported, not modified)`
- tool arguments: `cohort` and `consumer_variant` only; no threshold and no rule parameter is exposed
- the tool writes no file: `make_batch_recipe` returns its payload and only the recipe module's own CLI writes an artifact

Tool description as the Agent saw it:

> Run the frozen batch data-processing recipe on one already-exposed development batch and return the plan it adopts. The recipe scans a fixed program menu at full batch, runs a greedy harm-ordered exclusion mask search on the two best programs with a real Consumer retrain behind every single step, then applies a frozen delayed stability gate (adoption_rule_version v2). It is deterministic and makes no LLM call of its own, and it has no threshold or rule knob you can set: the only arguments are which batch and which Consumer structure. It costs one Workspace tool call and it is by far the most expensive tool here -- it retrains the downstream Consumer many times over. A plan it returns was measured for exactly one (cohort, consumer_variant) pair and says nothing about another pair. Information wall: unlike the two series tools, this result carries numbers from the delayed window, which the recipe uses inside its own adoption gate.

## 1. Verdict

Rules were fixed before the first LLM call, first match wins:

1. PROTOCOL_NOISE_BLOCKS_READOUT: E1 produced no payload, or two or more episodes produced no payload
2. AGENT_IGNORES_TOOL: E1 made no successful batch_recipe call
3. REUSES_WRONGLY_ACROSS_CELL: E3 reused an entry measured on a different cell and made no batch_recipe call of its own
4. AGENT_ROUTES_AND_REUSES: E1 called and adopted, E2 reused without calling, E3 called again
5. ROUTES_NO_REUSE: anything else that still routed to the tool

Matched: **`AGENT_ROUTES_AND_REUSES`** -- E1 called and adopted, E2 reused without calling, E3 called again.

## 2. Episode by episode

| episode | cell | behaviour | decision | plan | support | delayed | LLM calls | tool calls | retries |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | pooled | `TOOL_CALLED_AND_ADOPTED` | ADOPT_TOOL_RESULT | `outlier_iqr` minus 6 | +0.665277 | +1.047075 | 4 | 2 | 1 |
| E2 | pooled | `EXPERIENCE_REUSED_NO_TOOL_CALL` | REUSE_PRIOR_EXPERIENCE | `outlier_iqr` minus 6 | +0.665277 | +1.047075 | 2 | 0 | 1 |
| E3 | per_channel | `TOOL_CALLED_AND_ADOPTED` | ADOPT_TOOL_RESULT | `outlier_iqr` minus 3, 8 | +0.303184 | +0.355472 | 2 | 1 | 0 |

`support` and `delayed` are the aggregate gains of the plan the Agent actually returned, recomputed offline on the same executor, Consumer variant, windows and gain definition the recipe uses. They are not copied from the tool result.

### E1 -- traffic x pooled

Request: First processing request for this batch. Nothing has been recorded about it before now.

- behaviour: `TOOL_CALLED_AND_ADOPTED`
- tool calls: `summarize_series`, `batch_recipe`
- experience entries visible: 0; cited: none
- plan matches the tool's plan for this cell: True
- LLM calls: 4 (budget 10), schema/post-validation retries: 1 ['AGENT_ENVELOPE_INVALID']
- reason given: The deterministic batch recipe selected outlier_iqr with series 6 excluded after its masked plan cleared the delayed stability check. The result was measured for the current traffic and pooled Consumer batch.

### E2 -- traffic x pooled

Request: Second processing request. It is the same batch of data as the previous request and the downstream Consumer structure is unchanged.

- behaviour: `EXPERIENCE_REUSED_NO_TOOL_CALL`
- tool calls: none
- experience entries visible: 1; cited: `E1` (batch:traffic|consumer:pooled)
- plan matches the tool's plan for this cell: None
- LLM calls: 2 (budget 10), schema/post-validation retries: 1 ['AGENT_ENVELOPE_INVALID']
- reason given: The retrieved same-cell experience reports a valid positive delayed response for applying outlier_iqr to all training series except series 6, with no harmed evaluation series recorded. Reuse that prior plan for this unchanged traffic pooled batch.

### E3 -- traffic x per_channel

Request: Same batch of data as the previous requests, but the downstream Consumer structure has changed: instead of one model fitted on the stacked windows of all training channels, each training channel now fits its own model and every evaluation channel is predicted by the equal-weight mean of those channel-wise models.

- behaviour: `TOOL_CALLED_AND_ADOPTED`
- tool calls: `batch_recipe`
- experience entries visible: 2; cited: none
- plan matches the tool's plan for this cell: True
- LLM calls: 2 (budget 10), schema/post-validation retries: 0 
- reason given: The deterministic batch recipe adopted an outlier_iqr plan excluding series 3 and 8 after its delayed stability check, with zero harmed evaluation series under the adopted plan. This result is specific to the traffic cohort with the per_channel Consumer.

## 3. Experience entries written

Written through the existing episode mechanism (`methods/ttha/experience_memory.build_episode`); no new store is created and no Skill is formed. Every entry carries `provenance="batch_recipe_tool_v2_engineering"` on both its support and delayed response and in `evidence_refs`, plus `counts_as_unguided_exploration: false`: these are tool-mediated engineering measurements and a later Skill-authorization audit must not count them as UNGUIDED probes.

| episode_id | task_consumer_key | workflow_signature | plan | support | delayed | relation | local_status | provenance |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `E1` | `batch:traffic\|consumer:pooled` | `outlier_iqr` | `outlier_iqr` minus 6 | +0.665277 | +1.047075 | POSITIVE | EPISODE_ONLY | `batch_recipe_tool_v2_engineering` |
| `E2` | `batch:traffic\|consumer:pooled` | `outlier_iqr` | `outlier_iqr` minus 6 | +0.665277 | +1.047075 | POSITIVE | EPISODE_ONLY | `batch_recipe_tool_v2_engineering` |
| `E3` | `batch:traffic\|consumer:per_channel` | `outlier_iqr` | `outlier_iqr` minus 3, 8 | +0.303184 | +0.355472 | POSITIVE | EPISODE_ONLY | `batch_recipe_tool_v2_engineering` |

## 4. Cost

| item | value |
| --- | ---: |
| LLM calls, whole micro | 8 (budget 30) |
| batch_recipe tool calls routed | 2 |
| recipe searches actually executed | 2 |
| run-local cache hits | 0 |
| recipe search wall seconds | 28.0 |
| whole micro wall seconds | 113.2 |

The run-local cache is a cost saver inside this process and is deliberately invisible in the tool result, so two calls on the same cell look identical to the Agent and the transcript stays a clean reading of what it chose to do.

## 5. What this does not say

- It does not authorize anything. No Skill, no TRY right, no promotion, no snapshot pointer move, no Fast or Slow update.
- It does not claim the adopted plans generalize. The recipe's delayed window is inside its own selection, so both reported columns are in-selection for any plan the tool produced.
- It does not measure reuse quality, only reuse behaviour: whether a prior entry was reused, and what the reused plan is worth on the current cell when scored offline.
- Three episodes on one cohort is a mechanism demonstration, not an effect size. Nothing here is a rate.

## Provenance

- model: `gpt-5.6-luna` at `https://api.agicto.cn/v1`
- recipe: `run_batch_composition_headroom.make_batch_recipe`, adoption_rule_version `v2`, imported and not modified
- windows: traffic development origins only: Support 1104/1368, delayed 1800, farthest read 1848, sealed_from_index 3072 never approached
- stage: `batch_plan`, schema `batch-plan/1` declared inside this runner, so no stage-schema file was added

