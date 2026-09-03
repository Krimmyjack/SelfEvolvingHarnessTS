# SA-1 minimal r2 -- the one authorised sampling replicate

**CAP-1b 出口:A + G0 立 / G1 立 / G2 立**

capstone 形态 = **A3-reset vs A5-adaptive (scope-v2 card + R1-R3)**;头条主张 = the end-to-end gain of a Scope-scoped experience card plus feedback revision。

本跑 runner 判词 `SA1_DEVELOPMENT_SIGNAL`。**无 r3**(sol 令):r2 收口后无论出口即停,capstone 由主线按 CAP-1b 出口另书发车。

## Protocol identity

- **Same course freeze, byte for byte.**  r2 read `artifacts/functional/e2/sa1_course_freeze.json` unchanged -- six positions including the PowerCons #2 re-encounter slot, the same three arms, the same delta_material 0.08846153846153847 and harm bar 0.05.  The freeze records `git_head: 22256a9`, i.e. it was written before the r1 code commit and has not been regenerated since.

- **Same seed card.**  The v0 card is the one the gate artifact holds, compiled from the v4 record's GunPointMaleVersusFemale Episode under Scope rule v2; content sha `00503481edac9d90d49262fe53dd2e636ec2f063178da2ddc2632bf303ba3c7e`, identical to r1's v0.

- **Experiment face identical to `cf2eb12`.**  `git diff cf2eb12 HEAD` is empty over `methods/`, `contracts/`, `runtime/`, `operators/`, `evaluation/minipipe/`, `run_e2_s1_curriculum_four_arms.py` and `task_episode_harness/` (which holds `source_skill.py` and `skill_revision.py`).  Nothing that proposes, retrieves, probes, revises or deploys moved.

- **Correction to one sentence of the CAP-1b freeze.**  The amendment states "现 HEAD 代码面未变".  Taken literally that is not true: commit `5ff76b5` changed `run_e2_sa1_minimal.py` by +182/-8 lines.  Every hunk is in the post-run scoring and reporting region -- `_attribution`, `_headline`, `_honest_boundaries`, `_markdown`, the `--finalize` carry-over, and one stricter criterion in `_verdict` (mechanism difference must be attributable to the narrowing, not merely raw).  The live loop, the revision step and the per-unit scoring are untouched.  Decisively: **r1's committed artifact was itself re-rendered by this same HEAD code**, so r1 and r2 are scored by identical bytes and the side-by-side table below is apples to apples.  Reported as a factual correction to the freeze premise, not as a protocol change.

- **Artifact paths were relocated, not parameterised.**  The runner hardcodes the r1 output paths and changing that would be a code edit this book forbids, so r2 wrote to the r1 filenames and the products were moved to `sa1_minimal_r2.*` afterwards; the committed r1 files were then restored from `5ff76b5` and verified byte-identical to a pre-run copy.

- **Only difference from r1 = LLM sampling.**  This substrate's injection has no RNG to seed; the `--seed r2` label selects a fresh snapshot store root and a run id, nothing about the data.  `replicate_kind` stays `sampling`, as in r1.

## CAP-1b mechanism gates

### G0 safety -- pass

harm events 0; worst-class minimum by arm {'A3-reset': 0.0, 'K0-fixed': 0.0, 'A5-adaptive': 0.0}; Scope widened at nowhere; non-ladder rules none; versions not produced by a write-back none; PATCHes outside the two authorised surfaces none; guidance conditioning unrecorded at nowhere.

### G1 feedback-driven update -- pass

4 write-backs, version chain length 5, all content shas distinct: True.

| after # | unit | rules |
|---|---|---|
| 1 | GunPoint | R1 |
| 2 | GunPointOldVersusYoung | R1 |
| 3 | PowerCons | R2 |
| 4 | Herring | R2 |

### G2 behaviour change -- pass

| # | unit | scope match | refused | relations | R2/R3 fired |
|---|---|---|---|---|---|
| 3 | PowerCons | True | 1 | CONFLICT | True |
| 4 | Herring | True | 1 | CONFLICT | True |

Re-encounter slot (PowerCons #2): A5-adaptive scope=False supplied=0 refused=0; K0-fixed scope=True supplied=1 refused=1 -- holds: **True**.


## Three-way exit

| exit | triggered | capstone shape |
|---|---|---|
| A | **yes** | A3-reset vs A5-adaptive (scope-v2 card + R1-R3) |
| B | no | - |
| C | no | - |

## Card version chain

v0 `00503481edac` -> v1 `0eae563f76ce` -> v2 `3ef7202e73ce` -> v3 `89728a4af566` -> v4 `64c1cb5923f4`

## Per unit, per arm (candidate source split out)

| # | unit | arm | deploy | regret | probes | supplied | self-proposed probes | converted | refused | harm |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GunPoint | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.4067 | 1 | 0 | 1 | 0 | 0 | False |
| 1 | GunPoint | A5-adaptive | FROZEN_ACTIVE_SKILL_RECALL | -0.0667 | 1 | 1 | 0 | 1 | 0 | False |
| 1 | GunPoint | K0-fixed | FROZEN_ACTIVE_SKILL_RECALL | -0.0667 | 1 | 1 | 0 | 1 | 0 | False |
| 2 | GunPointOldVersusYoung | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1841 | 0 | 0 | 0 | 0 | 0 | False |
| 2 | GunPointOldVersusYoung | A5-adaptive | FROZEN_ACTIVE_SKILL_RECALL | -0.0286 | 2 | 1 | 1 | 1 | 0 | False |
| 2 | GunPointOldVersusYoung | K0-fixed | FROZEN_ACTIVE_SKILL_RECALL | -0.0286 | 2 | 1 | 1 | 1 | 0 | False |
| 3 | PowerCons | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 1 | 0 | 1 | 0 | 0 | False |
| 3 | PowerCons | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 1 | 1 | 0 | 1 | False |
| 3 | PowerCons | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 1 | 1 | 0 | 1 | False |
| 4 | Herring | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0469 | 2 | 0 | 2 | 0 | 0 | False |
| 4 | Herring | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0469 | 2 | 1 | 1 | 0 | 1 | False |
| 4 | Herring | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0469 | 2 | 1 | 1 | 0 | 1 | False |
| 5* | PowerCons | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 1 | 0 | 1 | 0 | 0 | False |
| 5* | PowerCons | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 1 | 0 | 1 | 0 | 0 | False |
| 5* | PowerCons | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 1 | 1 | 0 | 1 | False |
| 6 | BirdChicken | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0000 | 2 | 0 | 2 | 0 | 0 | False |
| 6 | BirdChicken | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0000 | 2 | 0 | 2 | 0 | 0 | False |
| 6 | BirdChicken | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0000 | 2 | 0 | 2 | 0 | 0 | False |

`*` = re-encounter slot; a mechanism readout that does not count toward cumulative regret.

## r1 vs r2, side by side

numbers not reproducing is not a failure; CAP-1b asks only for the mechanism gates.  This table exists so that what did and did not repeat is on the record either way

### Headline

| reading | r1 | r2 | reproduced |
|---|---|---|---|
| probes_saved_a5_vs_k0 | 2 | 1 | **no** |
| refusals_avoided_a5_vs_k0 | 2 | 1 | **no** |
| probes_saved_by_narrowing | 0 | 1 | **no** |
| refusals_avoided_by_narrowing | 1 | 1 | yes |
| regret_gap_a5_vs_k0_distinct_units | 0.0 | 0.0 | yes |
| card_vs_no_card_regret_gap | 0.6860317460317461 | 0.6860317460317461 | yes |
| harm_all_zero | True | True | yes |

### Arm totals

| arm | reading | r1 | r2 |
|---|---|---|---|
| A3-reset | cumulative_regret_distinct_units | 0.7710019841269842 | 0.7710019841269842 |
| A3-reset | probes | 7 | 7 |
| A3-reset | supplied_in_pool | 0 | 0 |
| A3-reset | supplied_converted | 0 | 0 |
| A3-reset | supplied_refused | 0 | 0 |
| A3-reset | harm_events | 0 | 0 |
| A3-reset | llm | 28 | 26 |
| A3-reset | consumer_fits | 15 | 16 |
| K0-fixed | cumulative_regret_distinct_units | 0.08497023809523807 | 0.08497023809523807 |
| K0-fixed | probes | 12 | 11 |
| K0-fixed | supplied_in_pool | 5 | 5 |
| K0-fixed | supplied_converted | 2 | 2 |
| K0-fixed | supplied_refused | 3 | 3 |
| K0-fixed | harm_events | 0 | 0 |
| K0-fixed | llm | 25 | 25 |
| K0-fixed | consumer_fits | 28 | 28 |
| A5-adaptive | cumulative_regret_distinct_units | 0.08497023809523807 | 0.08497023809523807 |
| A5-adaptive | probes | 10 | 10 |
| A5-adaptive | supplied_in_pool | 3 | 4 |
| A5-adaptive | supplied_converted | 2 | 2 |
| A5-adaptive | supplied_refused | 1 | 2 |
| A5-adaptive | harm_events | 0 | 0 |
| A5-adaptive | llm | 25 | 29 |
| A5-adaptive | consumer_fits | 26 | 27 |

### Per cell

| # | unit | arm | r1 regret / probes / supplied / refused | r2 same | identical |
|---|---|---|---|---|---|
| 1 | GunPoint | A3-reset | +0.4067 / 1 / 0 / 0 | +0.4067 / 1 / 0 / 0 | yes |
| 1 | GunPoint | A5-adaptive | -0.0667 / 2 / 1 / 0 | -0.0667 / 1 / 1 / 0 | no (probes) |
| 1 | GunPoint | K0-fixed | -0.0667 / 2 / 1 / 0 | -0.0667 / 1 / 1 / 0 | no (probes) |
| 2 | GunPointOldVersusYoung | A3-reset | +0.1841 / 1 / 0 / 0 | +0.1841 / 0 / 0 / 0 | no (probes) |
| 2 | GunPointOldVersusYoung | A5-adaptive | -0.0286 / 2 / 1 / 0 | -0.0286 / 2 / 1 / 0 | yes |
| 2 | GunPointOldVersusYoung | K0-fixed | -0.0286 / 2 / 1 / 0 | -0.0286 / 2 / 1 / 0 | yes |
| 3 | PowerCons | A3-reset | +0.1333 / 2 / 0 / 0 | +0.1333 / 1 / 0 / 0 | no (probes) |
| 3 | PowerCons | A5-adaptive | +0.1333 / 2 / 1 / 1 | +0.1333 / 2 / 1 / 1 | yes |
| 3 | PowerCons | K0-fixed | +0.1333 / 2 / 1 / 1 | +0.1333 / 2 / 1 / 1 | yes |
| 4 | Herring | A3-reset | +0.0469 / 1 / 0 / 0 | +0.0469 / 2 / 0 / 0 | no (probes) |
| 4 | Herring | A5-adaptive | +0.0469 / 1 / 0 / 0 | +0.0469 / 2 / 1 / 1 | no (probes, refused, supplied) |
| 4 | Herring | K0-fixed | +0.0469 / 2 / 1 / 1 | +0.0469 / 2 / 1 / 1 | yes |
| 5 | PowerCons | A3-reset | +0.1333 / 1 / 0 / 0 | +0.1333 / 1 / 0 / 0 | yes |
| 5 | PowerCons | A5-adaptive | +0.1333 / 2 / 0 / 0 | +0.1333 / 1 / 0 / 0 | no (probes) |
| 5 | PowerCons | K0-fixed | +0.1333 / 2 / 1 / 1 | +0.1333 / 2 / 1 / 1 | yes |
| 6 | BirdChicken | A3-reset | +0.0000 / 1 / 0 / 0 | +0.0000 / 2 / 0 / 0 | no (probes) |
| 6 | BirdChicken | A5-adaptive | +0.0000 / 1 / 0 / 0 | +0.0000 / 2 / 0 / 0 | no (probes) |
| 6 | BirdChicken | K0-fixed | +0.0000 / 2 / 0 / 0 | +0.0000 / 2 / 0 / 0 | yes |

### Pre-registered predictions

| id | r1 held | r2 held | r2 observed |
|---|---|---|---|
| P1 | True | True | matched ['GunPointOldVersusYoung__impulse_v2', 'GunPoint__impulse_v2', 'Herring__impulse_v2', 'PowerCons__impulse_v2'] |
| P2 | True | True | A5-adaptive conversions at positions [1, 2] |
| P3 | True | True | PowerCons#1 refused=1, narrowing PATCHes at position 3 = 1, version chain = ['v0', 'v1', 'v2', 'v3', 'v4'] |
| P4 | False | True | Herring scope_match=True supplied=1 refused=1 |
| P5 | True | True | A5 supplied=0 refused=0 / K0 supplied=1 refused=1 |
| P6 | True | True | probes saved 1 raw / 1 attributable to the narrowing, regret gap +0.0000 |
| P7 | True | True | harm events {'A3-reset': 0, 'K0-fixed': 0, 'A5-adaptive': 0} |

Card version chain identical to r1: **False**; shared prefix identical: **True** (r1 ['00503481edac', '0eae563f76ce', '3ef7202e73ce', '89728a4af566'], r2 ['00503481edac', '0eae563f76ce', '3ef7202e73ce', '89728a4af566', '64c1cb5923f4']).  r2's chain is not identical to r1's, but it does not diverge from it: the first 4 versions are byte-identical content shas in the same order, and r2 carries one extra version because Herring's refusal -- which r1 never saw, its supply having been deduplicated away -- drove a second narrowing.  Identical shas across two runs mean the revision content is a deterministic function of the readings that triggered it.

## What broke, and what the numbers do not say

- **The r1 P4 third path did not recur.**  In r1 Herring matched, the card was in view, and the mechanical supply was deduplicated against the agent's own identical proposal, so nothing was supplied and nothing was refused.  Here Herring took one of the two branches P4 did allow for: the supplied candidate reached the pool and the Target refused it (CONFLICT).  The deduplication census is empty (0 units).  Deduplication is not a refusal either way, and the G2 conditional does not read it.

- **This time the probe saving is the narrowing's.**  Both raw readings -- 1 probe saved, 1 refusal avoided -- are attributable to the narrowing, and both land at the pre-registered re-encounter slot.  In r1 the narrowing removed the refusal but the Fast agent immediately spent the freed slot on a proposal of its own, so r1's attributable probe saving was zero.  That is the one place r2 is cleaner than r1, and it is a sampling difference in the agent's proposal set, not a protocol difference.

- **Two narrowings this course, against r1's one.**  Herring's refusal drove a second R2 (card v4, clause `estimated_region_start_fraction==low, level_excursion_score==medium`), which r1 never reached because r1's Herring supply was deduplicated away.  Each clause is selective on the course units it could see: after #3 excludes exactly ['PowerCons'] (selective: True); after #4 excludes exactly ['Herring'] (selective: True).  But v4 arrived at position 4, and the only later family unit was PowerCons #2, which v3 had already excluded -- so **v4's freedom from over-exclusion is untested in-course**, established only against the six frozen Pattern views, not against a live unit that v4 alone would have kept.

- **Regret between the two card arms is again exactly +0.0000.**  With one refusal avoided, one probe saved and no conversion gained or lost, revision bought cost and not quality on this course.  Non-inferiority is what was pre-registered and it holds; nothing stronger is claimed.

- **The large regret number again belongs to Part 0.5, not to revision.**  Both card arms finish at +0.0850 cumulative regret over the five distinct units against +0.7710 with no card, a gap of +0.6860 -- identical to r1 to four decimals.  The family Pattern axis, not the revision loop, is what moved Scope coverage from 1-of-5 (L1's n=1 card) to 4-of-5.

- **R3 was not exercised live, in either run.**  Harm was zero everywhere and no supplied candidate was judged NEGATIVE, so the negative/harm branch rests on the offline historical replay only.  CAP-1b anticipated this and does not read R3.

- **Q7 stands unchanged.**  Three family leaves the edit schema cannot carry are declared on the card and absent from its machine AST, so the effective Scope is wider than the recorded one and every exclusion compiled on this surface inherits the same blind spot.

- **Two runs, and there will be no third** (sol's ruling).  Evidence grade stays development: every unit was already exposed, the seed Episode is a recorded one rather than a live re-earn, and the comparison arm for the card claim is a within-course control rather than a sealed target.  What two runs buy is the mechanism gates holding twice in the same direction, which is what CAP-1b asked for -- not the numbers repeating, which it explicitly did not.

## Cost

LLM 80 (book cap 120, runner constant 150 unchanged), consumer fits 71/300, wall 2580 s / 10800 s book cap, downloads 0.

## Obligations

- **capstone**: not in this book; the mainline issues it against the CAP-1b exit recorded here
- **code_changes**: zero -- experiment face byte-identical to cf2eb12; the one file differing from cf2eb12 at HEAD differs only in post-run reporting and was frozen before r2 ran; nothing was edited for this book and no code is committed
- **downloads**: 0
- **fault_routes_and_router_unmodified**: true
- **full_repo_pytest_not_run**: true
- **other_lines_files_untouched**: AGENTS.md, README, PROJECT_STATE*, SUCCESSOR_BRIEF*, ROADMAP untouched
- **parameter_or_course_changes**: zero -- same freeze file, same thresholds, same budget semantics, same prompt/model/backend protocol
- **scoring_script**: the CAP-1b gate readings in this artifact were computed by an uncommitted _scratch analysis script from fields already present in the two run artifacts, so every verdict is recheckable from the JSON alone; keeping it out of the tree is what preserves the zero-code-diff obligation
- **sealed_material_not_opened**: Epilepsy2 and the s1_oracle keys were not read; every Pattern view comes from extract_public_features on the built cell
- **subagents_spawned**: 0
- **third_run**: none, and none will be run -- sol's ruling is one r2 and then stop regardless of outcome
