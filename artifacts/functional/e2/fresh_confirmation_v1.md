# Fresh confirmation: NOAA 2024 -> 2025

**Overall: `FRESH_A5_DELIVERS`** -- pooled is the primary cell: A5_WINS; per_channel: A5_WINS

One opening of the sealed confirmation domain. Roster, program menu, recipe compiler, ADOPTION_RULE_V2, the 0.005 material and -0.005 harm lines, the #10 prompt templates, both Consumers, the Support budget, the e1v2 triple-window syntax and the lifecycle path were frozen; the Slow Agent was off throughout.

## Roster split amendment

12 train + 4 eval = all 16 PASS stations. Ruled before stage 0, before any Consumer was fitted on this cohort and before one 2025 index was read, no measured value taking part.

- the training batch of 12 is the calibrating geometry of every source evidence row -- the Guidance card, the mask geometry and the 0.005 material and -0.005 harm lines were all read off a 12-series batch, so it is a load-bearing dimension for comparability and is not scaled.
- cutting eval from 8 to 4 adds symmetric noise only; the variance lift is disclosed rather than hidden.
- per-station eval share 1/4 = 0.25 and eval fraction 4/16 = 0.25 both sit inside [0.20, 0.40].

TRAIN_SERIES_COUNT / EVAL_SERIES_COUNT appear only in cohort construction (v6._fixed_roster, run_w2_focus_recheck). The evaluation path -- bch._evaluate_assignment, bch._evaluate_variant, bch._gain_rows, wvc.BudgetedSearch -- reads train/eval off the roster rows and takes eval_uids as a parameter, so 12 + 4 is supplied, not patched in. No instrument line was changed.

## Stage 0 -- is the Judge readable

| Consumer | pass | block | spread | share | min | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `pooled` | True | support | 1.352 | 0.2913 | 1.1509 | 1.5563 |
| `pooled` | True | delayed | 2.291 | 0.3325 | 1.0167 | 2.3293 |
| `per_channel` | True | support | 1.562 | 0.2852 | 0.8564 | 1.3378 |
| `per_channel` | True | delayed | 1.526 | 0.2896 | 0.9206 | 1.4050 |

Caps quoted from `artifacts/functional/e2/noaa_health_check_v1.json`: spread <= 5.0, single-series share <= 0.40. 12 Consumer retrains, 0 LLM.

## Stage 1 -- the merged Guidance cards

- `pooled`: COMPILED, 4 clauses ['R1-1', 'R1-2', 'R1-3', 'R3-1']; bytes sha256 `5e55df667b46f3c2`.
- `per_channel`: COMPILED, 3 clauses ['R1-1', 'R1-2', 'R3-1']; bytes sha256 `c221c2163d30b0f4`.

- `pooled` store parity outside the skill library: True.
- `per_channel` store parity outside the skill library: True.

## Stage 1.5 -- does 2025 exist

ROSTER_LOCKED. Checked 16 stations, missing [], 0 bytes downloaded.

## Per cell

| cell | verdict | first-positive cost A5/A3 | total cost A5/A3 | task_C delayed A5/A3 | difference | harm A5/A3 |
| --- | --- | --- | --- | --- | ---: | --- |
| `pooled` | `A5_WINS` | 69 / 123 | 99 / 195 | +0.029688 / +0.029688 | +0.000000 | 1 / 1 |
| `per_channel` | `A5_WINS` | 75 / 78 | 102 / 105 | +0.000000 / +0.000000 | +0.000000 | 0 / 0 |

## Trajectory

| cell | arm | step | mode | plan | support | delayed | harm | retrains | LLM |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pooled` | `A5` | task_A | FULL_PRICE_SEARCH | `outlier_mad` full batch | +0.072486 | +0.306380 | 0 | 69 | 2 |
| `pooled` | `A5` | probe | OUT_OF_SELECTION_PROBE | -- | -- | +0.205806 | 1 | 6 | 0 |
| `pooled` | `A5` | task_B | DIRECT_RECALL | `identity` full batch | +0.000000 | +0.000000 | 0 | 9 | 0 |
| `pooled` | `A5` | task_C | DIRECT_RECALL | `outlier_mad` full batch | +0.191203 | +0.029688 | 1 | 15 | 0 |
| `pooled` | `A3` | task_A | FULL_PRICE_SEARCH | `identity` full batch | +0.000000 | +0.000000 | 0 | 57 | 2 |
| `pooled` | `A3` | probe | OUT_OF_SELECTION_PROBE | -- | -- | -- | -- | 0 | 0 |
| `pooled` | `A3` | task_B | FULL_PRICE_SEARCH | `repair_level_shift` full batch | +0.110951 | +0.162837 | 1 | 66 | 2 |
| `pooled` | `A3` | task_C | FULL_PRICE_SEARCH | `outlier_mad` full batch | +0.191203 | +0.029688 | 1 | 72 | 2 |
| `per_channel` | `A5` | task_A | FULL_PRICE_SEARCH | `repair_level_shift` full batch | +0.011824 | +0.046298 | 0 | 75 | 2 |
| `per_channel` | `A5` | probe | OUT_OF_SELECTION_PROBE | -- | -- | +0.023846 | 1 | 6 | 0 |
| `per_channel` | `A5` | task_B | DIRECT_RECALL | `identity` full batch | +0.000000 | +0.000000 | 0 | 12 | 0 |
| `per_channel` | `A5` | task_C | DIRECT_RECALL | `identity` full batch | +0.000000 | +0.000000 | 0 | 9 | 0 |
| `per_channel` | `A3` | task_A | FULL_PRICE_SEARCH | `repair_level_shift` minus 72203812897, 72259003927, 72422093820, 72605654791, 72793494248 | +0.053583 | +0.070321 | 0 | 78 | 2 |
| `per_channel` | `A3` | probe | OUT_OF_SELECTION_PROBE | -- | -- | +0.061088 | 1 | 6 | 0 |
| `per_channel` | `A3` | task_B | DIRECT_RECALL | `identity` full batch | +0.000000 | +0.000000 | 0 | 12 | 0 |
| `per_channel` | `A3` | task_C | DIRECT_RECALL | `identity` full batch | +0.000000 | +0.000000 | 0 | 9 | 0 |

## Exposure ledger

| partition | index | instance | outcome | opened by |
| --- | --- | --- | --- | --- |
| `development_2024` | [0, 8760] | `SEEN` | `EXPOSED` | stage 0 identity baselines and the stage 2 adaptation episodes fitted Consumers on this partition |
| `confirmation_2025` | [8760, 17520] | `SEEN` | `EXPOSED` | stage 4 read the confirmation task on this partition |
| `beyond_17520` | [17520, None] | `SEALED` | `SEALED` | nothing |

The confirmation partition is now open. The frozen version has no second reading: a repaired instrument may not be re-run on it, only a 0-evaluation deterministic replay of what is already here

## Cost and integrity

- LLM calls: 12 / 40.
- Consumer retrains: 513.
- Frozen surface: 19 files, drift [].
- Wall seconds: 397.8.
