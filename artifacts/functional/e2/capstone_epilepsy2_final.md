# CAP-1 capstone: Epilepsy2, single shot

**CAPSTONE 判词:CAPSTONE_NEUTRAL;A5−A3 accuracy = +0.000000;harm = 0**

CAP-1b exit **A** executed.  CAP-1 section 7 is void; section 3's A5 pool is replaced by the SA-1 scope-v2 supply card with the R1-R3 revision loop open.  Everything else is the CAP-1 freeze verbatim.

## Unseal record

| field | value |
|---|---|
| unsealed at (UTC) | 2026-08-28T06:28:56Z |
| zip | `data/ucr_conf_downloaded/D3_reserve/EpilepticSeizures.zip` |
| zip bytes | 16220082 |
| zip sha256 | `72ebe5b2be9756967110593170c89bb440d33a793a8f362f9d85a325d853f2bb` |
| seal verdict | SEAL_INTACT |
| first read scope | EpilepticSeizures/EpilepticSeizures_TRAIN.ts and EpilepticSeizures/EpilepticSeizures_TEST.ts, parsed in full to float arrays; val.ts, EpilepticSeizures.txt and the .png were not opened |
| TRAIN / TEST shape | [80, 178] / [11420, 178] |
| authorised by | CAP-1b (frozen before r2 results existed) + the mainline adjudication entry of 2026-08-28 13:5x + sol's ruling that one r2 is followed by the capstone regardless of outcome |

### Seal re-check

| check | pass | expected | observed |
|---|---|---|---|
| zip file present | True | True | True |
| zip byte count matches the ROSTER record | True | 16220082 | 16220082 |
| member listing matches CAP-0 | True | ['EpilepticSeizures/', 'EpilepticSeizures/EpilepticSeizures. | ['EpilepticSeizures/', 'EpilepticSeizures/EpilepticSeizures. |
| TRAIN member uncompressed size matches CAP-0 | True | 272040 | 272040 |
| TEST member uncompressed size matches CAP-0 | True | 38806349 | 38806349 |
| val.ts size matches CAP-0 | True | 68077 | 68077 |
| TRAIN raw newline count matches CAP-0 | True | 87 | 87 |
| TRAIN data-record count matches CAP-0 | True | 80 | 80 |
| TEST raw newline count matches CAP-0 | True | 11427 | 11427 |
| TEST data-record count matches CAP-0 | True | 11420 | 11420 |
| CAP-0 verdict on record is MATCH | True | MATCH | MATCH |

**CAP-0 recorded the ROSTER byte count, the member listing, the member sizes and the row counts, but no sha256 of the zip.  That one comparison therefore could not be made; the digest above is computed here for the first time and is the baseline for any future check.  Every fact CAP-0 did record was re-checked and matches.**

## Preflight (0 LLM, 0 fit, pre-unseal)

| check | pass |
|---|---|
| TEST subset regenerates from seed 20260827 and matches the frozen sha | True |
| menu names sha matches the freeze | True |
| h0 runtime_bundle_sha matches the freeze | True |
| h0 harness_content_sha matches the freeze | True |
| mod-4 quarters reproduce the frozen half-protocol indices | True |
| injection condition is the course's fit_only_artifact | True |

## Verdict arithmetic (CAP-1 section 6)

| reading | value |
|---|---|
| Static accuracy | 0.533613 |
| A3 accuracy | 0.533613 |
| A5 accuracy | 0.533613 |
| **A5 − A3** | **+0.000000** |
| held-out material line | 0.005000 |
| worst-class delta (A5 − A3) | +0.000000 |
| A5 harm | False |
| A5 − Static | +0.000000 |
| A3 − Static | +0.000000 |

A5/A3 against Static are reported per `AGENTS.md` section 2.1 and do not replace the verdict.

## Per arm

| arm | deploy | applied | accuracy | identity accuracy | gain | supplied | self-proposed | dedup swallowed | probes | llm | fit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.533613 | 0.533613 | +0.000000 | 0 | 0 | 0 | 0 | 0 | 1 |
| A3 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.533613 | 0.533613 | +0.000000 | 0 | 2 | 0 | 2 | 4 | 4 |
| A5 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.533613 | 0.533613 | +0.000000 | 0 | 2 | 0 | 2 | 5 | 4 |

Per-class held-out recall:

- **Static**: recall {'0': 0.46511627906976744, '1': 0.5487179487179488}, delta vs identity {'0': 0.0, '1': 0.0}
- **A3**: recall {'0': 0.46511627906976744, '1': 0.5487179487179488}, delta vs identity {'0': 0.0, '1': 0.0}
- **A5**: recall {'0': 0.46511627906976744, '1': 0.5487179487179488}, delta vs identity {'0': 0.0, '1': 0.0}

## The card on Epilepsy2 (ITT)

Scope machine match: **False** (applicability score 8).  a non-match is a legal reading, not a failure: the card's claim is conditional and ITT counts the condition as the card stated it

Leaves the Target does not carry at the card's value:

| leaf | card | Epilepsy2 |
|---|---|---|
| estimated_level_offset | low | zero |

## Card version chain

v0 `00503481edac`

Revision on the exam unit: rules none, card sha `00503481edac`.  CAP-1b: with one unit there is no later unit for a revision to change, so the revision loop's capstone readings are a descriptive record and the headline rests on the exit-A table, not on them

## Governance readout

| item | value |
|---|---|
| harm_event | False |
| harmed_classes | [] |
| card_authority_unchanged | {'grants_execution': False, 'reorders_supplied_candidates': False, 'supplies_candidates': True, 'suppresses_operators': False} |
| card_was_not_re_minted | True |
| no_tier_promotion | True |
| no_scope_widening | every revision in this book narrows or appends; narrow_applicability nests the old AST under all(old, not(...)) and no code path widens it |
| guided_positive_counts_zero | every evidence row the revision appends carries counts_toward_authorization=false, so a positive earned under the card buys the card nothing |

## What this verdict does and does not license

- **The NEUTRAL comes from a Scope non-match, not from a card that matched and then failed.**  Exactly one leaf separates the card from this Target: estimated_level_offset: card wants low, Epilepsy2 reads zero.  The card is scoped on the S1a hampel *family* intersection, and Epilepsy2 sits outside it, so retrieval never put the card in A5's Fast view, nothing was supplied, and the dedup mark correctly reads 'Scope did not match' rather than claiming a swallowed supply.  CAP-1b pre-declared this exact outcome a legal reading of a conditional claim, and ITT records it as the condition the card itself stated.

- **So the capability half of the claim was never exercised.**  A5 and A3 differ in base -- A5 starts from the K0 origin (h0 plus the three bootstrap procedures plus the inert Slow card) and A3 from cold h0 -- but the one piece of knowledge A5 carried that A3 did not, the supply card, was out of Scope here.  A5 - A3 is therefore exactly +0.000000 by construction, and the verdict arithmetic reduces to 'no arm changed anything'.  Reading this as evidence about the card's content would be reading a number the exam did not produce.

- **Collateral finding: Epilepsy2 offered no headroom to either adaptive arm.**  Both independently proposed the same two program families (outlier_mad, repair_level_shift) and both got negative Support readings (remove_remaining_extreme_deviation -0.0500; repair_local_level_excursion -0.2750), so no Draft formed, both abstained, and all three arms deployed identity at 0.5336 accuracy.  The abstention is the correct behaviour, not a failure to try.

- **The card's own program was never probed here, so its value on Epilepsy2 is unmeasured.**  Nobody proposed hampel_filter, which means this run cannot say whether supplying it would have helped or wasted a probe.  The exam establishes that the card declined; it does not establish that declining was optimal.

- **This is not the L1 failure mode repeating.**  L1's misses were decided by an incidental leaf a single Episode happened to carry; the leaf that decides this one is a member of the pre-frozen S1a family intersection.  The family axis behaved exactly as specified -- it is Epilepsy2 that is outside the family, which is the axis doing its job rather than failing at it.

- **The revision loop correctly did nothing.**  With no supply and no refusal attributable to the card, R1, R2 and R3 all declined to write and the version chain stays at v0.  Combined with zero harm, unchanged authority and no re-mint, the safety readout is clean.

- **What the capstone licenses.**  Not a capability claim: the conditional was not entered.  Not a refutation of SA-1 either: the exit-A mechanism evidence from r1 and r2 is untouched by a Target the card never claimed.  What it does demonstrate, on sealed material opened once, is the safety half -- a single-Episode card bought at the lowest rung on the ladder declined a domain its evidence never covered, cost nothing, and harmed nothing.  A capability verdict needs a sealed Target inside the card's family, which this line does not currently hold.

## Pre-registered predictions

| id | claim | held | observed |
|---|---|---|---|
| P1 | the seal re-check passes in full | yes | verdict SEAL_INTACT; 11 of 11 checks passed |
| P2 | A5 harm = 0 and no over-reach | yes | harmed classes []; card re-minted: False |
| P3 | the card's family Scope axis matches Epilepsy2 under the course injection family, so the card is retrieved and supplies | **no** | machine_match False; supplied into the pool 0; dedup_swallowed 0 |
| P4 | the headline A5-A3 reading is the exam question itself and was not predicted | not predicted | reported as the three-way verdict, not scored against a prediction |

## Cost

LLM 9 (cap 15 per arm), consumer fits 9 (cap 25 per arm), wall 5.58 min (cap 90), downloads 0.

## Obligations

- **cap1_section3_a5_pool_substituted**: True
- **cap1_section7_void_cap1b_substituted**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True
- **methods_contracts_runtime_operators_unmodified**: True
- **other_lines_files_untouched**: True
- **s1_oracle_not_touched**: True
- **seal_sha_gap_declared**: CAP-0 recorded no zip sha256, so that one comparison could not be made; every fact it did record was re-checked and matched, and the digest is now on record
- **single_shot_no_rerun**: the TEST subset was opened once; no verdict in CAP-1 section 6 authorises a second pass and none was run
- **subagents_spawned**: 0
- **thresholds_menu_template_prompt_model_unmodified**: every numeric constant is read from cap1_capstone_protocol_freeze.json; the injection template is cls._inject_v2 at helpers['positions'], unchanged
