# recipe Skill through the real store, retrieval and lifecycle

**Overall: `INTEGRATION_DELIVERS`** -- the card was retrieved and cited on every target, every target is non-inferior on delayed with A5 ahead on T1, T2, T3, and T1, T2, T3 formed a LOCAL_DRAFT.

The bridge run put the compiled card into the prompt by hand.  This slice keeps the signal and the three targets fixed and changes only the channel: each leave-one-cohort-out card is registered into the Skill store under the store's own `skill-entry/1` schema, Fast reads whatever `resolve_harness_view` returns for the Task Context, and the adopted plan is written through the existing Experience lifecycle.  The two arms send byte-identical public input; the only difference is the store they resolve against.

**It is not new evidence that the signal transfers.**  These are the same three targets the bridge already measured, so what is measured here is channel loss.

**Engineering integration measurement, not authorization evidence.**

## Source-class and authorization, as registered

Every card is registered with `source_class = "deterministic_recipe_compilation"` and `authorization_scope = "GUIDANCE"`.  The three reasons that class is claimed, recorded on each entry's `risk_guards`:

- the whole program menu was enumerated on every source cell, so no program reached the card by being the one somebody happened to try.
- no proposal step and no model chose which records to keep: the compiler is a fixed rule over committed rows, so there is no proposal-selection bias to inherit.
- every row it reads was measured on already-exposed development origins inside the sealed boundary, and the compiler reads committed artifacts only -- it never touches data.

What GUIDANCE means here: the card is retrievable knowledge for the proposal stage.  It carries no frozen program and no allowed tool, so it can never enter the candidate pool on its own; a plan it suggests is adopted only after this batch's own Support evaluation and the delayed gate, exactly as an unguided plan would be.

What is **not** granted: no confirmation-free TRY right, no execution right, no promotion to an active Skill, and no standing beyond the store namespace it was registered into.

The carrier guarantees are checked at registration, not asserted: allowed_tools is empty; the body carries no 'Frozen program steps:' marker, so _skill_frozen_candidates yields nothing from it; risk_guards records advises_the_proposal_stage_only, never_supplies_a_candidate and requires_target_support.

Leave-one-cohort-out: each card was compiled with every record measured on its own target cohort withheld, and each is registered into a store namespace of its own so that a target can never resolve another target's card.

## Registration

| slot | status | skill id | runtime bundle | skills in snapshot |
| --- | --- | --- | --- | ---: |
| `A3` | `REGISTERED` | `--` | `c4cb24b84092` | 3 |
| `T1` | `REGISTERED` | `recipe_batch_guidance_t1_v1` | `46a1010fc7ab` | 4 |
| `T2` | `REGISTERED` | `recipe_batch_guidance_t2_v1` | `69da26537db8` | 4 |
| `T3` | `REGISTERED` | `recipe_batch_guidance_t3_v1` | `9f34fdf2d3e4` | 4 |

Store namespace `recipe_batch_guidance_v1` under `_scratch/skill_store/recipe_batch_guidance_v1`; `methods/ttha/harness/h0` is read and never written.

- `T1` vs the empty-store arm: 12 files compared, differing `resolved.snapshot.json`, `skills/learned/.gitkeep`, `skills/learned/recipe_batch_guidance_t1_v1.json`, `snapshot.lock.json` -- nothing unexpected
- `T2` vs the empty-store arm: 12 files compared, differing `resolved.snapshot.json`, `skills/learned/.gitkeep`, `skills/learned/recipe_batch_guidance_t2_v1.json`, `snapshot.lock.json` -- nothing unexpected
- `T3` vs the empty-store arm: 12 files compared, differing `resolved.snapshot.json`, `skills/learned/.gitkeep`, `skills/learned/recipe_batch_guidance_t3_v1.json`, `snapshot.lock.json` -- nothing unexpected

## Per target

| target | delivery | retrieved | clauses cited | A3 plan | A3 delayed | A5 plan | A5 delayed | delta | direction | lifecycle |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `T1` | `DELIVERED` | `recipe_batch_guidance_t1_v1` | R1-1, R1-2, R3-1 | `identity` full batch | +0.000000 | `outlier_iqr` full batch | +0.244845 | **+0.244845** | `A5_WINS` | `LOCAL_DRAFT` |
| `T2` | `DELIVERED` | `recipe_batch_guidance_t2_v1` | R1-1, R3-1 | `repair_level_shift` minus 0, 1, 10, 11, 3 | +0.024745 | `outlier_iqr` full batch | +0.034455 | **+0.009710** | `A5_WINS` | `LOCAL_DRAFT` |
| `T3` | `DELIVERED` | `recipe_batch_guidance_t3_v1` | R1-1, R1-2, R3-1 | `hampel_filter` full batch | +0.048274 | `outlier_iqr` full batch | +1.165099 | **+1.116825** | `A5_WINS` | `LOCAL_DRAFT` |

Capture against each target's full-search reference: T1 A3 0.000 -> A5 0.883; T2 A3 0.184 -> A5 0.257; T3 A3 0.041 -> A5 1.000.

### What A5 cited

- **T1** cited R1-1, R1-2, R3-1: "The public observations show strong localized robust deviations and level excursions, with no missingness, so robust outlier programs are the most directly supported choices. The retrieved guidance supports trying outlier_mad and outlier_iqr early, while the unstable historical masks justify a fresh greedy mask search."
- **T2** cited R1-1, R3-1: "Several series show strong level-excursion and local-deviation signals, with no missingness observed, so level-shift repair is directly supported by the public features. An outlier-IQR alternative is also worth one evaluation based on the retrieved guidance; the mask should be re-searched on this window rather than reused."
- **T3** cited R1-1, R1-2, R3-1: "The public observations show repeated strong local deviations and level excursions, with no missingness, so robust outlier-focused programs are the most relevant mechanisms to evaluate. Retrieved guidance supports trying outlier_iqr and winsorize early and requires a fresh mask search rather than reusing a historical exclusion mask."

## Checks

| check | passed | detail |
| --- | --- | --- |
| delivery | True | {"T1": "DELIVERED", "T2": "DELIVERED", "T3": "DELIVERED"} |
| direction | True | 3 of 3 non-inferior, winners T1, T2, T3 |
| lifecycle | True | drafts T1, T2, T3 |

## Lifecycle

| arm-target | status | relation | evidence level | promotion |
| --- | --- | --- | --- | --- |
| `T1_A3` | `EPISODE_ONLY` | `ABSTAIN` | `SUPPORT` | not attempted |
| `T1_A5` | `LOCAL_DRAFT` | `POSITIVE` | `SUPPORT` | not attempted |
| `T2_A3` | `LOCAL_DRAFT` | `POSITIVE` | `SUPPORT` | not attempted |
| `T2_A5` | `LOCAL_DRAFT` | `POSITIVE` | `SUPPORT` | not attempted |
| `T3_A3` | `LOCAL_DRAFT` | `POSITIVE` | `SUPPORT` | not attempted |
| `T3_A5` | `LOCAL_DRAFT` | `POSITIVE` | `SUPPORT` | not attempted |

Promotion is not attempted: LOCAL_ACTIVE needs a delayed probe that did not take part in selection; this instrument spends its delayed reading on the adoption gate, so the Draft stands.

## Cost and parity

Consumer retrains: A3 182, A5 167, 349 in all.  First delayed-positive adoption: A3 at `T2_A3` after 144 retrains; A5 at `T1_A5` after 60 retrains.

Prompt parity: identical on every target.  {"T1": [], "T2": [], "T3": []}

LLM calls 12 of 40.  No early stop.

