# negative-path adaptation v1

**Verdict: `GATE_SAVES_BUT_NO_LEARNING`** -- the identity gate forced the fallback in 2 episodes.

T233 x per_channel is an established low-headroom cell: its W3 full search settles on `identity`, and the plan its W1 run adopted scores +0.0017 Support on W2 and -0.0010 on W3, both under the 0.005 material line. This run asks whether an Agent that keeps failing to re-confirm converges to abstention and stops spending budget, and whether that abstention stays local.

**Engineering effect measurement, not authorization evidence.** no Skill is written, no TRY right is granted, no Episode is promoted, no Fast or Slow path is entered, no snapshot pointer moves.

## 0. Design and what was pre-registered

Two deliberate departures from the earlier warm/cold runs, both fixed before the first call:

- within-run feedback is allowed: episode k's outcome is visible to episode k+1, because learning from one's own failures is the mechanism under test
- the adoption stage gains an identity-incumbent gate: a plan is adopted only if its delayed aggregate gain is at least max(best evaluated full-batch delayed, 0); otherwise the episode falls back to identity

Actions and what they cost:

- `REUSE_CONFIRM`: one charged evaluation: the Support of a plan named from visible experience, measured on this window. Confirmed only at Support >= 0.005
- `SEARCH`: one charged evaluation per shortlisted program, at most 2, plus a free greedy mask round on the highest-Support one
- `ABSTAIN_IDENTITY`: no evaluation is charged and no second stage runs

Identity-incumbent gate: bar = max(delayed of every evaluated full-batch plan, 0.0). A plan whose delayed gain is below the bar is not adopted; the episode falls back to identity. In the reuse path no full-batch plan is evaluated, so the bar is identity at zero. The Agent never sees a delayed number.

Verdict rules, applied in this order:

1. OVERGENERALIZED_ABSTENTION: E4 abstained on the high-headroom control cell
2. ADAPTIVE_ABSTENTION_CONVERGES: E1..E3 charged-evaluation counts are monotonically non-increasing, E3 ends at identity, E3 cites at least one earlier negative or marginal record, and E4 adopts a delayed-positive plan
3. GATE_SAVES_BUT_NO_LEARNING: the identity gate forced the fallback in two or more episodes
4. NO_BEHAVIOR_CHANGE: E2 and E3 both spent the full 2 evaluations and neither abstained
5. MIXED: anything else, reported episode by episode without merging

## 1. Episodes, windows and what each one could see

| episode | cell | window | support origins | delayed origins | reference plan | reference delayed | visible experience |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| E1 | T233 x per_channel | W2 | [3360, 3408, 3456] | [3504, 3552, 3600] | `outlier_iqr` minus T235, T244, T256 | +0.051322 | the low-headroom cell's frozen W1 row only |
| E2 | T233 x per_channel | W3 | [3648, 3696, 3744] | [3792, 3840, 3888] | `identity` minus nothing | +0.000000 | that row plus everything E1 produced |
| E3 | T233 x per_channel | W4 | [3936, 3984, 4032] | [4080, 4128, 4176] | `repair_level_shift` minus T241, T256 | +0.028664 | that row plus E1 and E2 |
| E4 | traffic x per_channel | W3 | [2064, 2328] | [2760] | `outlier_iqr` minus 3 | +0.386972 | the control cell's frozen W1 and W2 rows plus all three earlier episodes, so an over-applied abstention would show |

W2, W3 and the control window are quoted from `batch_recipe_windows_v1`; W4 is the frozen `e1v2_task_04` roster spec and its reference was computed here by the frozen v2 recipe with 0 LLM calls. Scoring references are never shown to the Agent.

Template parity: fields that move between episodes are `episode_id`, `prior_experience`, `public_observation`, `target`, all inside the allowed set `episode_id`, `prior_experience`, `public_observation`, `target` -- **True**. public_observation is a function of the window, so it moves with the window identifier; every other field is fixed by the template

## 2. Episode table

| episode | window | action | evals | confirmation | plan the Agent named | gate | final plan | support | delayed | capture | relation | LLM |
| --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| E1 | W2 | `REUSE_CONFIRM` | 1 | `winsorize` minus T233, T234, T241, T247, T256 -> +0.001731 (**failed**) | `identity` minus nothing | pass (bar +0.000000) | `identity` minus nothing | +0.000000 | +0.000000 | 0.000 | ABSTAIN | 2 |
| E2 | W3 | `SEARCH` | 2 | n/a | `smooth_ma` minus T254 | **fallback** (bar +0.000000) | `identity` minus nothing | +0.000000 | +0.000000 | n/a | ABSTAIN | 2 |
| E3 | W4 | `SEARCH` | 2 | n/a | `repair_level_shift` minus T241, T256 | **fallback** (bar +0.084153) | `identity` minus nothing | +0.000000 | +0.000000 | 0.000 | ABSTAIN | 3 |
| E4 | W3 | `SEARCH` | 2 | n/a | `outlier_iqr` minus 3 | pass (bar +0.380398) | `outlier_iqr` minus 3 | +0.514752 | +0.386972 | 1.000 | POSITIVE | 2 |

Charged evaluations across E1..E3: [1, 2, 2], non-increasing: **False**. E3 ended at identity: **True**. E3 cited ['T233|per_channel|W1', 'run|E2'], of which the negative or marginal records are ['run|E1', 'run|E2'] -- cites at least one: **True**. E4 final program `outlier_iqr`, delayed positive: **True**. Identity gate forced the fallback in 2 episode(s).

## 3. What each episode said

### E1 -- T233 x per_channel @ W2

- visible experience: 1 row(s) -- `T233|per_channel|W1`
- action: `REUSE_CONFIRM`; cited: `T233|per_channel|W1`
- action reason: The prior winsorize plan for this batch and consumer structure had positive Support and delayed-window gains. Current public observations still show strong localized deviations and level excursions, supporting a focused re-confirmation while retaining the previously excluded series.
- re-confirmation: `winsorize` minus T233, T234, T241, T247, T256 scored +0.001731 against a line of 0.005 -- FAILED
- adoption reason: The reconfirmed winsorize plan did not clear the public support line and harmed two evaluation series. Identity is the measured incumbent and is therefore selected.
- gate: bar +0.000000, the named plan's delayed +0.000000, passed True
- retries: negative_path_action=0, negative_path_adoption=0

### E2 -- T233 x per_channel @ W3

- visible experience: 2 row(s) -- `T233|per_channel|W1`, `run|E1`
- action: `SEARCH`; cited: `T233|per_channel|W1`, `run|E1`
- action reason: Public observations show strong level excursions across the training series, making repair_level_shift a direct mechanism candidate; smooth_ma is an effect-distinct alternative with positive prior Support evidence. The prior winsorize plan was not confirmed on the immediately preceding window, so it is not reused.
- shortlist: `repair_level_shift`, `smooth_ma` (mask requested: True)
- adoption reason: The masked smooth_ma plan has the strongest measured support gain (0.029733990728998943), exceeding the full-batch alternatives. Excluding T254 is supported by its unusually strong level excursion and period inconsistency signals, with insufficient post-shift support.
- gate: bar +0.000000, the named plan's delayed -0.063501, passed False
- retries: negative_path_action=0, negative_path_adoption=0

### E3 -- T233 x per_channel @ W4

- visible experience: 3 row(s) -- `T233|per_channel|W1`, `run|E1`, `run|E2`
- action: `SEARCH`; cited: `run|E2`, `T233|per_channel|W1`
- action reason: Public evidence shows strong level excursions and localized robust deviations with no missingness, so repair_level_shift is directly motivated and outlier_iqr provides an effect-distinct local alternative. Earlier smooth_ma search fell back to identity, while prior winsorize reuse was not confirmed on support.
- shortlist: `repair_level_shift`, `outlier_iqr` (mask requested: True)
- adoption reason: Public features show pronounced level excursions across the training series, and the measured repair_level_shift plan had positive support gain with no harmed evaluation series. The exclusion mask further improved measured support while retaining zero harmed evaluation series.
- gate: bar +0.084153, the named plan's delayed +0.028664, passed False
- retries: negative_path_action=1['AGENT_ENVELOPE_INVALID'], negative_path_adoption=0

### E4 -- traffic x per_channel @ W3

- visible experience: 5 row(s) -- `traffic|per_channel|W1`, `traffic|per_channel|W2`, `run|E1`, `run|E2`, `run|E3`
- action: `SEARCH`; cited: `traffic|per_channel|W1`, `traffic|per_channel|W2`
- action reason: The public features show repeated strong local deviations with no missingness, while level-shift support is insufficient for most series. Earlier traffic/per-channel records provide public support for outlier_iqr and winsorize, so evaluating both with the free mask search is justified.
- shortlist: `outlier_iqr`, `winsorize` (mask requested: True)
- adoption reason: The measured outlier_iqr plan with series 3 excluded had the highest support aggregate gain among the measured non-identity plans and harmed no evaluation series. Public features show pronounced local deviations across the training series, supporting a localized outlier treatment.
- gate: bar +0.380398, the named plan's delayed +0.386972, passed True
- retries: negative_path_action=0, negative_path_adoption=0

## 4. Experience rows this run produced

Written through the existing episode mechanism with `provenance="negative_path_engineering"`, `counts_as_unguided_exploration: false`. Unlike the earlier runs these rows **are** fed forward inside this run, which is the point: each episode sees whether the previous one's re-confirmation cleared the line and whether the identity incumbent refused its plan.

| episode | cell | window | action | confirmation | final plan | support | delayed | gate fallback | relation |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| `E1` | `batch:T233\|consumer:per_channel` | W2 | `REUSE_CONFIRM` | +0.001731 (failed) | `identity` minus nothing | +0.000000 | +0.000000 | False | ABSTAIN |
| `E2` | `batch:T233\|consumer:per_channel` | W3 | `SEARCH` | n/a | `identity` minus nothing | +0.000000 | +0.000000 | True | ABSTAIN |
| `E3` | `batch:T233\|consumer:per_channel` | W4 | `SEARCH` | n/a | `identity` minus nothing | +0.000000 | +0.000000 | True | ABSTAIN |
| `E4` | `batch:traffic\|consumer:per_channel` | W3 | `SEARCH` | n/a | `outlier_iqr` minus 3 | +0.514752 | +0.386972 | False | POSITIVE |

## 5. What this does not say

- It does not authorize anything, and abstaining is not a Skill.
- Four episodes on two cells with one model. Every label here is a single draw, not a rate.
- The identity gate is a Harness-side backstop that reads the delayed window. Where it fired, the episode's final plan is the gate's choice, not the Agent's, and the table separates the two.
- Within-run feedback means the episodes are not independent by construction. That is the mechanism under test, not a confound to be removed, but it does mean these four rows cannot be pooled with the earlier warm/cold runs.
- The low-headroom cell being low-headroom is itself an established fact from the frozen artifacts, not something this run discovered.

## Provenance

- model: `gpt-5.6-luna` at `https://api.agicto.cn/v1`
- instrument, observation table, corpus and stage driver: imported from `run_e2_warm_vs_cold_recipe_search`, which is not modified
- W4 reference: `run_batch_composition_headroom.make_batch_recipe` with `adoption_rule_version="v2"`, computed here, 0 LLM calls
- LLM calls: 9 of 30
- wall seconds: 275.1

