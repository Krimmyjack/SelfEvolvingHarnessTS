# recipe experience -> Source Skill -> Fast v1

**Overall: `SKILL_LOSES_SIGNAL`** -- A5 lost on T2; the clauses it cited there were T2:['R1-1', 'R1-2', 'R3-1'].

The warm/cold rotation showed a table of other-batch records helps, but it reached the Agent as a pasted table and the cost account left out the mask round's internal retrains. This run sends the same signal through a deterministic Slow compilation into a Source-derived Skill card, gives Fast only that card, and counts every Consumer retrain.

**Engineering effect measurement, not authorization evidence.** no Skill is promoted, no TRY right is granted, no Episode is promoted, and no Fast or Slow path of the real Harness runs; the Skill card is a rendered text object, installed nowhere.

## 0. The v2 gate, read before it was copied

- what v2 actually does: best_full_program = ranked[0] where ranked sorts the full-batch programs by descending Support aggregate gain; bar_delayed = full_batch_delayed[best_full]; delayed_bar = max(bar_delayed, 0.0). So the bar is the delayed gain of exactly one program -- the highest-Support full-batch one -- floored at identity's zero, and the sign of that program's Support is never checked
- what this runner does: bar = max(0.0, max delayed over the evaluated full-batch plans whose Support gain is strictly positive)
- the difference, and why: v2 reads one program; this reads the max over the Support-positive ones, which is never lower than identity and never lets a plan nobody could adopt set the bar. The negative-path run copied a wider 'max over all evaluated full-batch delayed' with no eligibility check: on its E3 a plan with Support -0.002312 and delayed +0.084153 set the bar and knocked out a plan that equalled the frozen W4 reference

## 1. Compiled Skill cards

| target | cell | status | clauses | rules that produced nothing |
| --- | --- | --- | ---: | --- |
| T1 | T233 x pooled | `COMPILED` | 3 | R2 |
| T2 | electricity x per_channel | `COMPILED` | 3 | R2 |
| T3 | traffic x pooled | `COMPILED` | 3 | R2 |

Rendered text, exactly as A5 saw it:

**T1**

> Source-derived Skill for this batch, compiled from other-cohort records only (every record measured on this cohort was withheld). Clauses are guidance, not instructions; the measurements you take on this window decide.
>
> - [R1-1] Try `outlier_mad` early on this Consumer structure: it holds a delayed-positive record on 2 cohorts (electricity, traffic), cross-cohort mean delayed +0.480164.
> - [R1-2] Try `outlier_iqr` early on this Consumer structure: it holds a delayed-positive record on 2 cohorts (electricity, traffic), cross-cohort mean delayed +0.442983.
> - [R3-1] Do not reuse a historical exclusion mask. Across the leave-one-cohort-out cells only 0 of 4 kept their mask across windows (share 0.0000, at or below the 0.3333 threshold), so a mask has to be re-searched on the window in front of you.

**T2**

> Source-derived Skill for this batch, compiled from other-cohort records only (every record measured on this cohort was withheld). Clauses are guidance, not instructions; the measurements you take on this window decide.
>
> - [R1-1] Try `outlier_iqr` early on this Consumer structure: it holds a delayed-positive record on 2 cohorts (T233, traffic), cross-cohort mean delayed +0.216691.
> - [R1-2] Try `winsorize` early on this Consumer structure: it holds a delayed-positive record on 2 cohorts (T233, traffic), cross-cohort mean delayed +0.213245.
> - [R3-1] Do not reuse a historical exclusion mask. Across the leave-one-cohort-out cells only 1 of 4 kept their mask across windows (share 0.2500, at or below the 0.3333 threshold), so a mask has to be re-searched on the window in front of you.

**T3**

> Source-derived Skill for this batch, compiled from other-cohort records only (every record measured on this cohort was withheld). Clauses are guidance, not instructions; the measurements you take on this window decide.
>
> - [R1-1] Try `outlier_iqr` early on this Consumer structure: it holds a delayed-positive record on 2 cohorts (T233, electricity), cross-cohort mean delayed +0.121066.
> - [R1-2] Try `winsorize` early on this Consumer structure: it holds a delayed-positive record on 2 cohorts (T233, electricity), cross-cohort mean delayed +0.098648.
> - [R3-1] Do not reuse a historical exclusion mask. Across the leave-one-cohort-out cells only 1 of 4 kept their mask across windows (share 0.2500, at or below the 0.3333 threshold), so a mask has to be re-searched on the window in front of you.

Full clause provenance -- which artifact and which key each clause came from -- is in `artifacts/functional/e2/recipe_skill_cards_v1.json`.

## 2. Targets and windows

| target | cell | window | support | delayed | origin source | reference plan | reference delayed |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| T1 | T233 x pooled | e1v2_task_04 | [3936, 3984, 4032] | [4080, 4128, 4176] | quoted from the frozen roster | `winsorize` minus T244 | +0.277200 |
| T2 | electricity x per_channel | e1v2_task_04 | [3936, 3984, 4032] | [4080, 4128, 4176] | quoted from the frozen roster | `denoise_median` minus 10, 3, 4 | +0.134243 |
| T3 | traffic x pooled | W4_traffic_shift | [2208, 2472] | [2904] | chosen | `outlier_iqr` minus nothing | +1.165099 |

T3's window is chosen, not quoted. Its sealed boundary was verified before the run against both the code constant (`evaluation/functional/task_episode_harness/agentic/g3_sourcing.py::SEALED_FROM_INDEX` = 3072) and the screening artifacts ({"g3_candidate_screening_v2": 3072, "g3_candidate_screening_v3": 3072}): farthest index read 2952, tightest boundary 3072, inside: **True**.

Prompt parity, per target: T1 -> `arm`, `source_skill`; T2 -> `arm`, `source_skill`; T3 -> `arm`, `source_skill`. All targets pass: **True**.

## 3. Per-target result

| target | card | delivery | clauses A5 cited | A3 delayed | A5 delayed | paired delta | label |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| T1 | `COMPILED` (3) | `DELIVERED` | `R1-1`, `R1-2`, `R3-1` | +0.000000 | +0.244845 | +0.244845 | `A5_WINS` |
| T2 | `COMPILED` (3) | `DELIVERED` | `R1-1`, `R1-2`, `R3-1` | +0.024745 | +0.000000 | -0.024745 | `A5_LOSES` |
| T3 | `COMPILED` (3) | `DELIVERED` | `R1-1`, `R1-2`, `R3-1` | +0.000000 | +1.165099 | +1.165099 | `A5_WINS` |

Counts: A5_WINS 2, A5_LOSES 1, TIE 0, unreadable 0.

## 4. The twelve-row arm table

| target | arm | shortlist | mask | evals | retrains | plan named | gate | final plan | support | delayed | capture | LLM |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| T1 | A3 | `repair_level_shift`, `hampel_filter` | True | 2 | 75 | `repair_level_shift` minus T233, T236, T239, T244 | **fallback** (bar +0.000000) | `identity` minus nothing | +0.000000 | +0.000000 | 0.000 | 2 |
| T1 | A5 | `outlier_mad`, `outlier_iqr` | True | 2 | 66 | `outlier_iqr` minus nothing | pass (bar +0.244845) | `outlier_iqr` minus nothing | +0.107139 | +0.244845 | 0.883 | 2 |
| T2 | A3 | `repair_level_shift`, `hampel_filter` | True | 2 | 81 | `repair_level_shift` minus 0, 1, 10, 11, 3 | pass (bar +0.000000) | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.052073 | +0.024745 | 0.184 | 2 |
| T2 | A5 | `outlier_iqr`, `winsorize` | True | 2 | 69 | `winsorize` minus 0, 1 | **fallback** (bar +0.040939) | `identity` minus nothing | +0.000000 | +0.000000 | 0.000 | 2 |
| T3 | A3 | `repair_level_shift`, `hampel_filter` | True | 2 | 38 | `hampel_filter` minus nothing | **fallback** (bar +0.251583) | `identity` minus nothing | +0.000000 | +0.000000 | 0.000 | 2 |
| T3 | A5 | `outlier_iqr`, `winsorize` | True | 2 | 40 | `outlier_iqr` minus nothing | pass (bar +1.165099) | `outlier_iqr` minus nothing | +1.335389 | +1.165099 | 1.000 | 2 |

## 5. Cost, first measurement, no criterion attached

| arm | total Consumer retrains | first delayed-positive adoption | cumulative retrains to get there |
| --- | ---: | --- | ---: |
| A3 | 194 | T2_A3 | 156 |
| A5 | 175 | T1_A5 | 66 |

A retrain is one fit of the downstream Consumer. One evaluation over three origins is three retrains, and the mask round's per-series work is retrains too -- that is the part the earlier runs did not count.

## 6. What each arm said

**T1_A3** -- shortlist `repair_level_shift`, `hampel_filter` (mask True), cited none

  shortlist reason: The public observations show repeated localized level excursions with substantial estimated offsets and robust local deviations across the training series. Shortlist level-shift repair first, with Hampel filtering as an effect-distinct alternative for the prominent local deviations; request the free exclusion round to assess whether a subset of exposed series is driving support.

  adopted `repair_level_shift` minus T233, T236, T239, T244: The public measurements show that repair_level_shift improves aggregate Support when applied with the measured exclusion mask, while full-batch repair_level_shift and hampel_filter reduce aggregate Support. The selected plan is the measured positive-gain plan and preserves the identity fallback if its delayed reading is not better than identity.

**T1_A5** -- shortlist `outlier_mad`, `outlier_iqr` (mask True), cited `R1-1`, `R1-2`, `R3-1`

  shortlist reason: The public features show strong local deviations and level excursions without missingness, so robust outlier treatments are the most relevant programs to evaluate first. A fresh exclusion-mask search is requested because the public guidance indicates historical masks are not reliable across windows.

  adopted `outlier_iqr` minus nothing: Adopt outlier_iqr because it had the highest measured support aggregate gain (0.1071) and harmed no evaluation series in the full-batch Support evaluation. The mask round accepted no revert.

**T2_A3** -- shortlist `repair_level_shift`, `hampel_filter` (mask True), cited none

  shortlist reason: The public features show strong level excursions across multiple channels, with several coincident extreme local deviations and no missingness, making level-shift repair the primary candidate and Hampel filtering an effect-distinct alternative. A mask search is warranted because the evidence is heterogeneous across channels, including channels without sufficient post-shift support.

  adopted `repair_level_shift` minus 0, 1, 10, 11, 3: The greedy exclusion-mask round showed positive aggregate Support gain for repair_level_shift and no harmed evaluation series after reverting series 0, 1, 10, 11, and 3. The full-batch level-shift plan was harmful, so the masked plan is the measured option with the strongest public support.

**T2_A5** -- shortlist `outlier_iqr`, `winsorize` (mask True), cited `R1-1`, `R1-2`, `R3-1`

  shortlist reason: Several channels show strong localized robust deviations and level excursions, while missingness is absent, so robust outlier treatments are the most directly supported options. The exclusion mask should be re-searched on this window because historical mask stability is weak.

  adopted `winsorize` minus 0, 1: The measured winsorize plan with exclusions for series 0 and 1 had the highest support aggregate gain (0.04307120325796349) among the measured plans and harmed no evaluation series. It also improved over the full-batch winsorize plan.

**T3_A3** -- shortlist `repair_level_shift`, `hampel_filter` (mask True), cited none

  shortlist reason: Public observations show strong level excursions across channels and substantial local robust deviations, with no missingness. The shortlist prioritizes a level-shift repair, followed by a localized robust outlier filter; a free mask search can refine the highest-Support option.

  adopted `hampel_filter` minus nothing: The Hampel filter had the larger measured full-batch Support gain (0.4487 versus 0.1037 for repair_level_shift), although both non-identity plans harmed two evaluation series. It is therefore the strongest measured candidate under the available public evidence.

**T3_A5** -- shortlist `outlier_iqr`, `winsorize` (mask True), cited `R1-1`, `R1-2`, `R3-1`

  shortlist reason: The public observations show no missingness but widespread extreme local deviations and level excursions, so robust outlier treatments are the most relevant shortlist. I will try outlier_iqr first and winsorize second, with a fresh mask search rather than reusing a historical exclusion mask.

  adopted `outlier_iqr` minus nothing: The full-batch outlier_iqr plan had the largest measured support aggregate gain among the measured non-identity plans, with no harmed evaluation series. The public observations show strong local robust deviations across the training series, supporting this localized outlier treatment.

## 7. Experience rows written

`provenance="skill_bridge_engineering"`, `counts_as_unguided_exploration: false`, and fed into nothing: the three targets are independent and no row from this run reaches another target.

| episode | cell | plan | support | delayed | relation |
| --- | --- | --- | ---: | ---: | --- |
| `T1_A3` | `batch:T233\|consumer:pooled` | `identity` minus nothing | +0.000000 | +0.000000 | ABSTAIN |
| `T1_A5` | `batch:T233\|consumer:pooled` | `outlier_iqr` minus nothing | +0.107139 | +0.244845 | POSITIVE |
| `T2_A3` | `batch:electricity\|consumer:per_channel` | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.052073 | +0.024745 | POSITIVE |
| `T2_A5` | `batch:electricity\|consumer:per_channel` | `identity` minus nothing | +0.000000 | +0.000000 | ABSTAIN |
| `T3_A3` | `batch:traffic\|consumer:pooled` | `identity` minus nothing | +0.000000 | +0.000000 | ABSTAIN |
| `T3_A5` | `batch:traffic\|consumer:pooled` | `outlier_iqr` minus nothing | +1.335389 | +1.165099 | POSITIVE |

## 8. What this does not say

- It authorizes nothing. The Skill card is compiled text handed to a prompt; no snapshot, no Fast/Slow path and no execution right is touched.
- Three targets, one draw each, one model. Every label is a comparison of two single runs.
- The compiler's rules are three hand-frozen thresholds, not a learned policy, and a rule that produced nothing here says something about the provenance available, not about the mechanism.
- The delayed column is out of selection for both arms -- neither saw it -- but the reference plans it is compared against were themselves selected on their own delayed windows.
- The retrain count is the instrument's cost, not the Agent's: both arms are charged for whatever the instrument had to fit.

## Provenance

- model: `gpt-5.6-luna` at `https://api.agicto.cn/v1`
- instrument, observation table and stage driver: imported from `run_e2_warm_vs_cold_recipe_search`, which is not modified
- references: `run_batch_composition_headroom.make_batch_recipe` with `adoption_rule_version="v2"`, run here per target, 0 LLM calls
- Skill cards and every clause's provenance: `artifacts/functional/e2/recipe_skill_cards_v1.json`
- full prompt bodies for all six arm-targets are in the JSON under `episodes[].prompt_body`
- LLM calls: 12 of 40
- wall seconds: 919.8

