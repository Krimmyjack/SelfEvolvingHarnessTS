# #45-Frep-b: symmetric zero-feedback deployment, terminal re-adjudication

**Verdict: `A5A3_TERMINAL_READJUDICATED`**

both arms deployed from frozen state only, with zero delayed openings used for adoption, zero candidate evaluations and identical deployment cost (9 retrains each). On the primary pooled cell the held-out terminal utility is A5 +0.059385 vs A3 -0.216513 (difference +0.275898, A5_HIGHER); first-positive cost 84 vs 123 is carried over from the #45-Frep held-in ledger unchanged.

Development data, already-exposed outcomes, one paired draw. Held-in was not re-run: the #45-Frep frozen snapshots and held-in ledger are reused unchanged.

## The repaired deployment semantics

**F1** -- the deployed Workflow is fixed from frozen state before any outcome is touched. The inherited v2 adoption ladder, with its held-out delayed `bar` and `confirmation` reads, is never called. Exactly one delayed opening happens per arm, after the bytes are already fixed, and it is the evaluator's one-shot scoring.

**F2** -- both arms deploy recall-only. An arm with an applicable ACTIVE Skill deploys the Workflow that Skill governs; an arm without one deploys the standing incumbent of its frozen ledger (the last held-in round's adopted plan), or identity if none stands. Forbidden on the scored block: shortlist or any LLM call; candidate evaluation, full-batch Support evaluation, mask round; Support confirmation used as an adoption gate; the v2 adoption ladder and its delayed bar/confirmation reads; any adoption driven by a reading taken on the scored block; Skill formation, approval, limitation or revocation.

## Frozen snapshot verification

| slot | active sha | recompiles to itself | matches #45-Frep | skill ids match |
| --- | --- | --- | --- | --- |
| `a5_pooled` | `f66c8e323937a33b` | `True` | `True` | `True` |
| `a3_pooled` | `836d5e768f6f79cc` | `True` | `True` | `True` |
| `a5_per_channel` | `39af15db974c8b29` | `True` | `True` | `True` |
| `a3_per_channel` | `f70abc9d8bf4b16f` | `True` | `True` | `True` |

## Four-arm deployment ledger (F1/F2 discharge and cost symmetry)

| slot | delayed openings | used for adoption | charged Support evals | candidate evals | LLM | deploy retrains | pure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `a5_pooled` | 1 | 0 | 0 | 0 | 0 | 9 | `True` |
| `a3_pooled` | 1 | 0 | 0 | 0 | 0 | 9 | `True` |
| `a5_per_channel` | 1 | 0 | 0 | 0 | 0 | 9 | `True` |
| `a3_per_channel` | 1 | 0 | 0 | 0 | 0 | 9 | `True` |

Deployment cost is identical across all four arms: `True` (values [9], spread 0). each arm pays 3 identity support-baseline retrains + 3 identity delayed-baseline retrains, both computed by the instrument's own constructor and used by no decision, plus 3 retrains for the one scoring read of a non-identity applied plan (0 for identity, which reuses the cached identity baseline)

## Paired four-readout table

| cell | arm | first-pos cost (retrains) | first-pos LLM | total LLM/fit | deploy source | applied Workflow | held-out terminal utility | harm |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| `pooled` | `A5` | 84 | 2 | 2 LLM / 108 fit | `FROZEN_ACTIVE_SKILL_RECALL` | `outlier_iqr` full batch | +0.059385 | 1 |
| `pooled` | `A3` | 123 | 5 | 5 LLM / 132 fit | `FROZEN_LEDGER_INCUMBENT` | `repair_level_shift` full batch | -0.216513 | 4 |
| `per_channel` | `A5` | 75 | 2 | 2 LLM / 102 fit | `FROZEN_ACTIVE_SKILL_RECALL` | `repair_level_shift` full batch | +0.096496 | 1 |
| `per_channel` | `A3` | 75 | 2 | 2 LLM / 102 fit | `FROZEN_ACTIVE_SKILL_RECALL` | `repair_level_shift` full batch | +0.096496 | 1 |

## Re-adjudicated terminal comparison, against the voided reading

| cell | A5 terminal | A3 terminal | difference | direction | same Workflow | voided #45-Frep difference |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `pooled` | +0.059385 | -0.216513 | +0.275898 | `A5_HIGHER` | `False` | -0.152549 |
| `per_channel` | +0.096496 | +0.096496 | +0.000000 | `WITHIN_MATERIAL_BAND` | `True` | +0.000000 |

### Why the voided reading differed

- `pooled`: under the defective protocol A5 deployed via `DIRECT_RECALL` for +0.000000 at 9 retrains and A3 via `FULL_PRICE_SEARCH` for +0.152549 at 69 retrains, raw cell verdict `A5_LOSES`. Voided because F1: the voided deployment took 2 delayed reading(s) of the scored block while choosing what to deploy; F2: the two arms were not given the same deployment treatment (DIRECT_RECALL at 9 retrains vs FULL_PRICE_SEARCH at 69 retrains).
- `per_channel`: under the defective protocol A5 deployed via `DIRECT_RECALL` for +0.096496 at 15 retrains and A3 via `DIRECT_RECALL` for +0.096496 at 15 retrains, raw cell verdict `A5_TIE_TRANSFER_BOUNDARY`. Voided because F1: the voided deployment took 4 delayed reading(s) of the scored block while choosing what to deploy.

## Held-in carry-over

First-positive cost is unchanged from #45-Frep on every arm: `True`. every arm reached its first delayed-positive adoption inside held-in, so the voided deployment row never entered the first-positive cost in the first place

## Budget

- this book's LLM calls: **0** (the repaired deployment is recall-only, so it spends no LLM call at all; the increment over #45-Frep is exactly 0)
- this book's Consumer retrains: 36, by arm {'A5': 18, 'A3': 18}
- carried over from #45-Frep held-in: 11 LLM / 408 retrains
- downloads: 0; sealed reads: 0; held-in re-runs: 0

## Freeze and binding

- frozen snapshot unchanged across deployment: `True`
- scored bytes == applied bytes on every arm: `True`

## Wall

4.8 seconds.

## Post-run annotation (0 evaluation)

after the run, by a 0-evaluation deterministic pass. Changes no measured number: `True`.

### What the repair changed, mechanically

Under the repaired protocol every arm's deployed Workflow was fixed before a single outcome of the scored block was touched, and every arm paid exactly the same 9 Consumer retrains and 0 LLM calls to deploy. The instrument's own counters carry the proof: 1 delayed opening per arm, 0 of them used for adoption, 0 charged Support evaluations, 0 candidate evaluations. Under the voided protocol the pooled cell had 9 retrains on one arm against 69 plus 2 LLM calls on the other, and the deployed program on that second arm was chosen by reading the scored block.

### The re-adjudicated terminal reading

Primary pooled cell: A5 +0.059385 against A3 -0.216513, difference +0.275898, direction A5_HIGHER. per_channel: both arms deployed the identical Workflow from identical frozen Skills and scored +0.096496 each, difference +0.000000 -- a clean tie inside the 0.005 material band, so the transfer boundary the original adjudication recorded on this Consumer still stands. The voided #45-Frep pooled difference was -0.152549, i.e. the direction of the pooled terminal comparison flips once the two defects are removed.

### Why the pooled arms separated, stated plainly

This is not A5 scoring better with the same Workflow. The two arms deployed different Workflows and one of them failed to carry to the tail block. A3's frozen incumbent repair_level_shift was delayed-POSITIVE inside held-in (+0.162837 on task_B) and turns to -0.216513 on the held-out tail, harming all 4 of 4 evaluation series for a total harm of 0.866053. A5's frozen Skill outlier_iqr was the weaker held-in performer (+0.066941) and holds at +0.059385 with 1 harmed series and 0.187138 total harm. So the pooled reading says the Source-primed arm converged on a Workflow that generalised while the cold arm's standing incumbent did not -- and it says nothing about A5 being better at the same Workflow.

### How much the voided protocol was actually leaking

The two defects are now quantified rather than asserted. Given a free search on the scored block, and permitted to read that block's delayed outcomes while choosing, the A3 pooled arm replaced a -0.216513 incumbent with a +0.152549 fresh find. That 0.369062 swing is the size of the leak the voided reading was carrying, and it is larger than either arm's honest terminal utility.

### Sensitivity of the A3 reading to the incumbent rule

The incumbent rule was pinned before scoring as 'the final plan of the last held-in round'. The one arm it governs, a3_pooled, therefore deployed repair_level_shift. Under the looser reading 'an arm with no ACTIVE Skill deploys identity', a3_pooled would have scored exactly +0.000000, because the gain metric is defined against the identity baseline -- no new measurement is needed to state that. The pooled difference would then be +0.059385 instead of +0.275898, still above the 0.005 material band and still A5_HIGHER. The direction of the pooled re-adjudication is therefore robust to that protocol choice; only its magnitude is not.

### Claim ceiling for this artifact

Development data, already-exposed outcomes, one paired draw, two Consumers, 16 series. This may be cited as INFRASTRUCTURE/MECHANISM evidence that the forecasting held-out deployment can be run with zero feedback and symmetric arm cost, and that under that clean protocol the #45-Frep pooled terminal comparison reads A5_HIGHER rather than A5_LOSES. It may not be cited as fresh evidence, as generalisation, as a replication of FRESH_A5_DELIVERS, or as settled evidence that accumulated knowledge improves final held-out quality: a single development block with one LLM draw cannot support that, and the per_channel cell shows no marginal contribution at all.

### Out-of-book findings (reported, not fixed)

- **G1 -- fc.stage_4 still carries both defects** -- The repair is driver-local, as the scope pin required. evaluation/functional/run_e2_fresh_confirmation.py::stage_4 is unmodified and still runs the delayed adoption ladder and the asymmetric full-price fallback. Any other caller of that entry inherits F1 and F2. Reported, not fixed.
- **G2 -- the #45-Frep artifact still shows the voided numbers** -- t6_45_frep_a5a3_replay.json/.md were not edited and still carry the voided held-out terminal columns and the A5_LOSES / FRESH_A5_FAILS raw verdicts in their own tables. They remain valid for the chain and held-in readouts only. Anyone quoting a terminal number must quote this artifact instead.
- **G3 -- 3 decision-free Support-block fits per arm remain** -- fc.FreshSearch.__init__ computes identity baselines on both the support and the delayed origins of whatever window it is given, so each arm pays 3 Consumer fits on the held-out support origins that no decision reads. It is identical across arms and feeds nothing, but it is not literally zero contact with the scored block. Removing it needs an edit to the in-service constructor, which is outside this book's scope pin. Reported, not fixed.
- **G4 -- a held-in-positive Workflow harmed 4 of 4 series** -- repair_level_shift was delayed-positive on held-in task_B and harmed every evaluation series on the tail block. This is a concrete instance of held-in delayed feedback failing to predict held-out safety, which is the risk the Scope/Risk and abstain machinery exists for. It is a development-block observation on one draw, not a measured Scope defect, and it belongs to the Scope/overfitting row of the first-fault table rather than to this book. Reported, not investigated.
- **G5 -- per_channel gives accumulation no marginal role** -- a3_per_channel promoted the same Skill id as a5_per_channel during held-in, so both arms deploy identical bytes and the cell is an exact tie. On this Consumer the Target held-in feedback alone reached the same place the Source-primed arm reached, which is the transfer boundary the original adjudication already recorded. It is not evidence against accumulation; it is one Consumer where accumulation bought nothing measurable.

### Obligation self-report

- conda_env_project_and_interpreter_printed: `True`
- interpreter: `D:\Anaconda_envs\envs\project\python.exe`
- subagents_spawned: `0`
- scope_pin_respected: `only F1 and F2 were repaired; no historical debt was fixed and no instrument was extended`
- fix_location: `evaluation/functional/run_e2_t6_45_frep_a5a3_replay.py deployment path only`
- methods_modified: `False`
- in_service_runner_lines_edited: `0`
- held_in_re_run: `False`
- frozen_snapshots_reused_and_verified: `True`
- downloads: `0`
- noaa_2025_read: `False`
- beyond_17520_read: `False`
- yahoo_read: `False`
- other_line_files_touched: `none; AGENTS.md, README, PROJECT_STATE_AND_DATA_MAP and SUCCESSOR_BRIEF were neither edited nor committed`
- frep_a_artifacts_modified: `False`
- this_book_llm_calls: `0`
- this_book_consumer_retrains: `36`
- scored_runs: `2`
- second_run_disclosed: `the deployment is recall-only and fully deterministic, so it was re-run once after a per-cell correction to the 'voided because' attribution text; every measured number was verified byte-identical across the two passes before and after that change, so no draw was selected`
- positive_bar_imposed: `False`
- artifacts_committed: `False`
- code_committed: `True`

