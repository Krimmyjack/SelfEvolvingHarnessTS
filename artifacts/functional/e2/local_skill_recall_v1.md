# Target-local Skill persistence and task_06 recall

**Overall: `LOCAL_LIFECYCLE_CLOSES`** -- A5 had zero RECALL_MISS across T1/T2, reused at least one local Skill, and no reused plan crossed the -0.005 harm guard

Only the lifecycle surface changed. Source Guidance, recipe/compiler, program menu, Consumer/Metric/Support budget, #10 prompt templates and ADOPTION_RULE_V2 stayed frozen. task_06 is quoted verbatim from the frozen E1-v2 roster.

## Per arm-target

| cell | verdict | local retrieval | support confirmation | adopted | support | delayed | task_04 cost | task_06 cost | lifecycle |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `A3_T1` | `NO_LOCAL_SKILL` | None (`--`) | -- | `hampel_filter full batch` | +0.030430 | +0.022936 | 69 | 63 | -- -> --; episode LOCAL_DRAFT |
| `A5_T1` | `RECALLED_REUSED_CHEAPER` | True (`fast_winner_e1v2_outlier_iqr`) | +0.092136 | `outlier_iqr full batch` | +0.092136 | +0.032863 | 60 | 15 | LOCAL_ACTIVE -> LOCAL_ACTIVE; episode LOCAL_DRAFT |
| `A3_T2` | `RECALLED_REUSED_CHEAPER` | True (`fast_winner_e1v2_repair_level_shift`) | +0.013525 | `repair_level_shift minus 0, 1, 10, 11, 3` | +0.013525 | +0.005271 | 75 | 15 | LOCAL_ACTIVE -> LOCAL_ACTIVE; episode LOCAL_DRAFT |
| `A5_T2` | `RECALLED_REUSED_CHEAPER` | True (`fast_winner_e1v2_outlier_iqr`) | +0.021251 | `outlier_iqr full batch` | +0.021251 | +0.017907 | 69 | 15 | LOCAL_ACTIVE -> LOCAL_ACTIVE; episode LOCAL_DRAFT |

Costs are total Consumer retrains, not shortlist counts: identity baselines, direct confirmation/shortlist, mask internals, Support bookkeeping and delayed reads are all included.

## Learning curve

| arm | target | task_04 | task_06 | delta |
| --- | --- | ---: | ---: | ---: |
| `A3` | `T1` | 69 | 63 | -6 |
| `A5` | `T1` | 60 | 15 | -45 |
| `A3` | `T2` | 75 | 15 | -60 |
| `A5` | `T2` | 69 | 15 | -54 |

## Persistence evidence

- `a3_t1`: `REGISTERED`; Guidance `--`; local `--`; status `--`, evidence `--`, #11 probe `--` gain --.
- `a5_t1`: `REGISTERED`; Guidance `recipe_batch_guidance_t1_v1`; local `fast_winner_e1v2_outlier_iqr`; status `LOCAL_ACTIVE`, evidence `DELAYED`, #11 probe `e1v2_task_05` gain +0.093583.
- `a3_t2`: `REGISTERED`; Guidance `--`; local `fast_winner_e1v2_repair_level_shift`; status `LOCAL_ACTIVE`, evidence `DELAYED`, #11 probe `e1v2_task_05` gain +0.056720.
- `a5_t2`: `REGISTERED`; Guidance `recipe_batch_guidance_t2_v1`; local `fast_winner_e1v2_outlier_iqr`; status `LOCAL_ACTIVE`, evidence `DELAYED`, #11 probe `e1v2_task_05` gain +0.040501.

## Provenance and stopping conditions

Every task_06 Experience records `provenance=local_skill_recall` and `counts_as_unguided_exploration=false`. The #11 probe is referenced read-only and was not remeasured during persistence.

LLM calls: 2 / 30.

No unresolved ambiguity stopped the run. The task_06 delayed reading participated in its own adoption gate, so it did not replace #11's independent task_05 activation evidence.
