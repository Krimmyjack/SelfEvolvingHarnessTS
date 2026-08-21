# recipe experience -> Source Skill -> Fast, v2 ladder replay

**Instrument-corrected reading: `SKILL_BRIDGE_DELIVERS`** -- delivery holds on every non-abstaining target, no target is A5_LOSES, and A5 wins on T1, T2, T3.

**Filed beside `artifacts/functional/e2/recipe_skill_bridge_v1.json`, which read `SKILL_LOSES_SIGNAL`.**  That artifact is not overwritten and not amended; both readings stand, and this one carries the corrected adoption ladder.

The bridge run's adoption gate dropped to identity the moment a named plan missed the bar.  The frozen v2 rule has a rung in between: fall back to the best full-batch plan when its delayed gain is positive, and only then to identity.  That run also set the bar from the highest delayed among the eligible full-batch plans, where v2 sets it from exactly one plan, the Support winner.  This replay re-decides the adoption stage of the six recorded arm-targets from numbers that run already measured.  0 LLM calls, 0 new Consumer retrains, no data touched.

**Engineering instrument correction, not authorization evidence.**  No Skill is promoted, no TRY right is granted, and no Fast or Slow path of the real Harness runs.

## What changed

| arm-target | old adopted | old delayed | new adopted | new delayed | path | changed |
| --- | --- | ---: | --- | ---: | --- | --- |
| `T1_A3` | `identity` full batch | +0.000000 | `identity` full batch | +0.000000 | `GATE_FAIL_FALLBACK_IDENTITY` | no |
| `T1_A5` | `outlier_iqr` full batch | +0.244845 | `outlier_iqr` full batch | +0.244845 | `GATE_PASS_ADOPT_NAMED` | no |
| `T2_A3` | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.024745 | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.024745 | `GATE_PASS_ADOPT_NAMED` | no |
| `T2_A5` | `identity` full batch | +0.000000 | `winsorize` full batch | +0.040939 | `GATE_FAIL_FALLBACK_SUPPORT_WINNER` | **yes** |
| `T3_A3` | `identity` full batch | +0.000000 | `hampel_filter` full batch | +0.048274 | `GATE_PASS_ADOPT_NAMED` | **yes** |
| `T3_A5` | `outlier_iqr` full batch | +1.165099 | `outlier_iqr` full batch | +1.165099 | `GATE_PASS_ADOPT_NAMED` | no |

## Recomputed labels

| target | A3 delayed | A5 delayed | paired delta | label | old label | delivery |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `T1` | +0.000000 | +0.244845 | **+0.244845** | `A5_WINS` | `A5_WINS` | `DELIVERED` |
| `T2` | +0.024745 | +0.040939 | **+0.016194** | `A5_WINS` | `A5_LOSES` | `DELIVERED` |
| `T3` | +0.048274 | +1.165099 | **+1.116825** | `A5_WINS` | `A5_WINS` | `DELIVERED` |

Capture against each target's full-search reference: T1 A3 0.000 -> A5 0.883; T2 A3 0.184 -> A5 0.305; T3 A3 0.041 -> A5 1.000.

## Cost, re-attributed

Total retrains are a property of what was run and do not move: A3 194, A5 175, 369 in all.  Only which adoption counts as the first delayed-positive one is recomputed.

| arm | first delayed-positive adoption | cumulative retrains | old reading |
| --- | --- | ---: | --- |
| A3 | `T2_A3` | 156 | `T2_A3` at 156 |
| A5 | `T1_A5` | 66 | `T1_A5` at 66 |

Delayed-positive adoptions: A3 2 of 3 (was 1), A5 3 of 3 (was 2).

## The ladder, against the frozen rule

| rung | relation to v2 | note |
| --- | --- | --- |
| step_1_support_winner | `PORTED_WITH_ONE_DECLARED_ADDITION` | the ordering and the menu-order tie-break are the same; v2 never checks the sign of that program's Support, this replay requires Support > 0.  That addition is the correction carried forward from the negative-path run, and literal_v2_sensitivity on every arm-target reports what the unchecked reading would have adopted |
| step_2_gate | `IDENTICAL` | bar = max(0, one program's full-batch delayed).  v1 read the max over every eligible full-batch plan's delayed instead, which is the forbidden comparison of several candidates' delayed |
| step_3_fallback | `IDENTICAL` | this is the rung v1 skipped: it went from a failed gate straight to the else branch |
| candidate set | `NARROWED_BY_DESIGN` | v2 walks its own list of masked plans.  The replay re-scores the one plan the Agent named, masked or full-batch, because the object under test is the Agent's decision, not the recipe's search.  When the named plan is itself the Support winner at full batch the gate is satisfied by equality |
| full-batch pool | `NARROWED_BY_BUDGET` | the arms paid for at most 2 full-batch evaluations, so the pool is what each arm actually measured; nothing unmeasured is estimated |

Selection uses Support only.  Delayed is consulted at most twice per arm-target -- once for the Support winner, to set the bar, and once for the named plan, to confirm it -- and every read is recorded on that arm-target's `delayed_reads` tape.

## Per arm-target

### `T1_A3`

- full-batch pool: `repair_level_shift` support -0.008419 delayed -0.250678; `hampel_filter` support -0.115115 delayed -0.307956
- Support winner: none -- the highest-Support full-batch plan is 'repair_level_shift' at -0.008419, which is not positive, so nothing here is a plan a deployer could adopt
- bar +0.000000 (no Support winner, so the bar is identity at zero); named plan `repair_level_shift` minus T233, T236, T239, T244 at delayed -0.098916, margin -0.098916
- the named plan missed the bar by -0.098916 and there is no Support winner with a positive full-batch delayed, so the ladder fell to identity
- adopted `identity` full batch at support +0.000000, delayed +0.000000, capture 0.000 (the old run adopted `identity` full batch at delayed +0.000000 under a bar of +0.000000)
- delayed numbers consulted: 1 (confirmation <the plan the Agent named> -0.098916)
- literal v2, without the Support-sign check, would adopt `identity` full batch at +0.000000 (`GATE_FAIL_FALLBACK_IDENTITY`) -- same

### `T1_A5`

- full-batch pool: `outlier_mad` support +0.076984 delayed +0.196464; `outlier_iqr` support +0.107139 delayed +0.244845
- Support winner: `outlier_iqr`
- bar +0.244845 (max(0, `outlier_iqr` full-batch delayed), and `outlier_iqr` is the Support winner); named plan `outlier_iqr` full batch at delayed +0.244845, margin +0.000000
- the named plan cleared the bar (+0.244845 >= +0.244845)
- adopted `outlier_iqr` full batch at support +0.107139, delayed +0.244845, capture 0.883 (the old run adopted `outlier_iqr` full batch at delayed +0.244845 under a bar of +0.244845)
- delayed numbers consulted: 2 (bar outlier_iqr +0.244845; confirmation <the plan the Agent named> +0.244845)
- literal v2, without the Support-sign check, would adopt `outlier_iqr` full batch at +0.244845 (`GATE_PASS_ADOPT_NAMED`) -- same

### `T2_A3`

- full-batch pool: `repair_level_shift` support -0.003122 delayed -0.026275; `hampel_filter` support -0.101649 delayed -0.089815
- Support winner: none -- the highest-Support full-batch plan is 'repair_level_shift' at -0.003122, which is not positive, so nothing here is a plan a deployer could adopt
- bar +0.000000 (no Support winner, so the bar is identity at zero); named plan `repair_level_shift` minus 0, 1, 10, 11, 3 at delayed +0.024745, margin +0.024745
- the named plan cleared the bar (+0.024745 >= +0.000000)
- adopted `repair_level_shift` minus 0, 1, 10, 11, 3 at support +0.052073, delayed +0.024745, capture 0.184 (the old run adopted `repair_level_shift` minus 0, 1, 10, 11, 3 at delayed +0.024745 under a bar of +0.000000)
- delayed numbers consulted: 1 (confirmation <the plan the Agent named> +0.024745)
- literal v2, without the Support-sign check, would adopt `repair_level_shift` minus 0, 1, 10, 11, 3 at +0.024745 (`GATE_PASS_ADOPT_NAMED`) -- same

### `T2_A5`

- full-batch pool: `outlier_iqr` support +0.037366 delayed +0.034455; `winsorize` support +0.042821 delayed +0.040939
- Support winner: `winsorize`
- bar +0.040939 (max(0, `winsorize` full-batch delayed), and `winsorize` is the Support winner); named plan `winsorize` minus 0, 1 at delayed +0.040678, margin -0.000260
- the named plan missed the bar by -0.000260, so the ladder fell back to the Support winner `winsorize`, whose full-batch delayed is positive
- adopted `winsorize` full batch at support +0.042821, delayed +0.040939, capture 0.305 (the old run adopted `identity` full batch at delayed +0.000000 under a bar of +0.040939)
- delayed numbers consulted: 2 (bar winsorize +0.040939; confirmation <the plan the Agent named> +0.040678)
- literal v2, without the Support-sign check, would adopt `winsorize` full batch at +0.040939 (`GATE_FAIL_FALLBACK_BEST_FULL_BATCH`) -- same

### `T3_A3`

- full-batch pool: `repair_level_shift` support +0.103687 delayed +0.251583; `hampel_filter` support +0.448739 delayed +0.048274
- Support winner: `hampel_filter`
- bar +0.048274 (max(0, `hampel_filter` full-batch delayed), and `hampel_filter` is the Support winner); named plan `hampel_filter` full batch at delayed +0.048274, margin +0.000000
- the named plan cleared the bar (+0.048274 >= +0.048274)
- adopted `hampel_filter` full batch at support +0.448739, delayed +0.048274, capture 0.041 (the old run adopted `identity` full batch at delayed +0.000000 under a bar of +0.251583)
- delayed numbers consulted: 2 (bar hampel_filter +0.048274; confirmation <the plan the Agent named> +0.048274)
- literal v2, without the Support-sign check, would adopt `hampel_filter` full batch at +0.048274 (`GATE_PASS_ADOPT_NAMED`) -- same

### `T3_A5`

- full-batch pool: `outlier_iqr` support +1.335389 delayed +1.165099; `winsorize` support +1.125221 delayed +0.925549
- Support winner: `outlier_iqr`
- bar +1.165099 (max(0, `outlier_iqr` full-batch delayed), and `outlier_iqr` is the Support winner); named plan `outlier_iqr` full batch at delayed +1.165099, margin +0.000000
- the named plan cleared the bar (+1.165099 >= +1.165099)
- adopted `outlier_iqr` full batch at support +1.335389, delayed +1.165099, capture 1.000 (the old run adopted `outlier_iqr` full batch at delayed +1.165099 under a bar of +1.165099)
- delayed numbers consulted: 2 (bar outlier_iqr +1.165099; confirmation <the plan the Agent named> +1.165099)
- literal v2, without the Support-sign check, would adopt `outlier_iqr` full batch at +1.165099 (`GATE_PASS_ADOPT_NAMED`) -- same

## Standing

- the corrected reading is filed beside the v1 verdict as the instrument-corrected reading of the bridge experiment; v1 is not overwritten and not amended.
- the six arm-targets carry no cross-target experience feedback, and every LLM decision was taken before the gate and without seeing its result, so re-deciding the gate deterministically cannot have changed what the Agent would have said.
- run_e2_negative_path_adaptation feeds each episode's outcome into the next, so a changed gate result would change later decisions; a replay there would be invalid and its labels stand as filed.
- the harm accounts of newly adopted plans are not reproduced here: the bridge run persisted the aggregate gains of every full-batch plan but the full harm account only of the plan it adopted, and this replay estimates nothing it does not have.

