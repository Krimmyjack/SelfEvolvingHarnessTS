# S1b smoke -- four arms on curriculum unit 1

protocol: `s1_curriculum_four_arms_v1`  entry: `--smoke`  backend: **scripted_sealed_probe**  git: `4cca78518b1cf6eb3394f16988e7fe2954fbf0cd`

**S1B_SMOKE_WIRED**

wiring only, on curriculum unit 1, one round per adaptive arm.  No Capability claim; the course was not run.

curriculum revision under test: **r2** (forward order frozen in `artifacts/functional/e2/s1_curriculum_frozen.json`)

unit under test: `MiddlePhalanxOutlineCorrect__impulse_v2` (harm_evidence, PhalanxFamily; smallest held-in slice 45 rows)

## Gates

- **four_arm_state_isolation**: True
- **domain_binding_hooks_live**: True
- **judging_produced_a_regret_table**: True
- **oracle_isolation_holds**: True
- **oracle_wall_proved_armed**: True
- **within_budget**: True
- **deploy_purity_clean**: True
- **feedback_surface_readable**: True
- **no_delayed_rejected_winner_deployed**: True

## Four-arm readout

| arm | deploy | program | held-out utility | menu-oracle | regret | worst-class | harm | wrong promo | LLM | fits | probes | wasted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 |
| A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 3 | 6 | 2 | 1 |
| K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 3 | 6 | 2 | 1 |
| A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 3 | 6 | 2 | 1 |

## Feedback surface: is it readable? (**live**)

at least one Support receipt came back non-NEUTRAL: the feedback surface is readable on live evidence

- smallest held-in slice: 45 rows, resolution 0.0222
- programs the proposal stage actually probed: ['denoise_median', 'winsorize']
- non-NEUTRAL Support receipts: **3**

| arm | round | program | relation | support gain | delayed gain |
|---|---|---|---|---|---|
| A3-reset | r1 | winsorize | NEGATIVE | 0.1333333333333333 | -0.11111111111111116 |
| K0-fixed | r1 | winsorize | NEGATIVE | 0.1333333333333333 | -0.11111111111111116 |
| A5-online | r1 | winsorize | NEGATIVE | 0.1333333333333333 | -0.11111111111111116 |

Arithmetic side-evidence from the sealed oracle:

| program | legal | pooled held-in | |m| >= 1/slice | material | probed in this smoke |
|---|---|---|---|---|---|
| outlier_iqr | True | -0.09444444444444444 | True | True | False |
| outlier_mad | True | 0.13888888888888895 | True | True | False |
| repair_level_shift | True | 0.0 | False | False | False |

- the oracle headroom is measured on the pooled held-in surface (all four slices concatenated) while a round reads one slice, so the arithmetic is a necessary condition on the surface, not a guarantee about any single round

## State at the unit boundary

next unit would be `DistalPhalanxOutlineCorrect__burst_cls2`

| arm | end-of-unit sha | store evolved | episodes at end | next base | episodes carried | skills carried |
|---|---|---|---|---|---|---|
| Static | `4abf3bec40f1` | False | 0 | h0 (never adapts, never deploys a learned Workflow) | 0 | - |
| A3-reset | `4abf3bec40f1` | False | 2 | h0 (cold start, zero carry) | 0 | - |
| K0-fixed | `127ecb868f63` | False | 2 | K0 (reset; no write-back) | 0 | - |
| A5-online | `127ecb868f63` | False | 2 | K0 + everything the domain-binding wall admits | 2 | - |

## Four-arm state isolation

- **static_ran_no_round**: True
- **static_wrote_no_episode**: True
- **static_store_unchanged**: True
- **static_deployed_identity**: True
- **a3_started_from_h0**: True
- **a3_started_with_no_episode**: True
- **a3_next_base_is_h0**: True
- **k0fixed_started_from_k0**: True
- **k0fixed_next_base_resets_to_k0**: True
- **k0fixed_carries_no_episode**: True
- **a5_started_from_k0**: True
- **a5_carries_episode_memory**: True
- **a5_and_k0fixed_share_the_same_start**: True
- **a5_next_base_differs_from_k0_or_states_why**: True
- **k0_card_in_the_store_of_k0fixed_and_a5**: True
- **k0_card_never_entered_a_fast_view**: True
- **a3_store_never_held_the_k0_card**: True

## Domain binding

- hook 1, every minted Skill stamped: True
- hook 1, stamped Skills: none minted
- hook 2, next unit `DistalPhalanxOutlineCorrect__burst_cls2`; foreign Target-local dropped: True
- hook 2, decisions: none
- hook 3, Source card decision: no card minted
- Episode domain namespaces observed: ['MiddlePhalanxOutlineCorrect__impulse_v2']

### Synthetic probe of the two walls

synthetic entries through the live decision functions; the arms minted none of these

- hook 2, a capability stamped with unit 1 offered to unit 2: carried = False
- hook 2, a capability stamped with unit 2 offered to unit 2: carried = True
- hook 2 behaves as specified: **True**
- hook 3, matching five-axis Scope admits: True
- hook 3, empty pattern intersection admits: False
- hook 3, wrong consumer admits: False
- hook 3, pattern mismatch admits: False (axes that differ between unit 1 and unit 2: ['estimated_region_start_fraction', 'period_change_score'])
- hook 3 behaves as specified: **True**

## Instrument finding (blocks S1c: **False**)

the r2 readability floor holds across the whole course: every unit's smallest held-in slice is at least 7 rows, none is empty, and the frozen two-round protocol has a delayed surface everywhere.  On unit 1 the observed Episode relations were ['NEGATIVE', 'NEUTRAL'], so a harm Episode was written on live evidence rather than inferred.

- relations observed on unit 1: ['NEGATIVE', 'NEUTRAL']
- harm Episode formed on unit 1: True
- guard minted on unit 1: False
- the frozen two-round protocol has rows on every unit: True
- units with an empty held-in slice: none

nothing outstanding on the readability axis.  The remaining question is not whether harm can be *read* but whether the proposal stage samples the same harmful program on both harm units -- see guard_channel_feasibility.

| # | unit | group | fit rows | support pool | slice rows (r1s/r1d/r2s/r2d) | smallest slice | smallest expressible gain |
|---|---|---|---|---|---|---|---|
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | harm_evidence | 420 | 180 | 45/45/45/45 | 45 | 0.022 |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | learnable_positive | 420 | 180 | 46/45/45/44 | 44 | 0.023 |
| 3 | PowerCons__impulse_v2 | harm_evidence | 126 | 54 | 14/14/14/12 | 12 | 0.083 |
| 4 | FreezerRegularTrain__burst_cls2 | identity | 106 | 44 | 12/12/10/10 | 10 | 0.100 |
| 5 | GunPointOldVersusYoung__impulse_v2 | learnable_positive | 95 | 41 | 11/10/10/10 | 10 | 0.100 |
| 6 | ECG200__impulse_v2 | heldout_only_temptation | 70 | 30 | 9/7/7/7 | 7 | 0.143 |
| 7 | Ham__impulse_v2 | identity | 76 | 33 | 9/8/8/8 | 8 | 0.125 |

- the harm channel needs NEGATIVE Episodes on two distinct units before risk_skill can compile a guard, and classify_relation needs an aggregate move of at least 0.005 to call anything but NEUTRAL.  A slice of n rows moves accuracy only in steps of 1/n, so a one- or two-row slice reports 0.0 for every candidate and a zero-row slice leaves the round with no surface to read at all.  This table is the readability precondition, not a result.

## Guard channel feasibility

a guard can form on ['outlier_iqr']: legal and readably harmful on both harm units.  Formation still requires the proposal stage to sample it on both, which is an agent behaviour and not an arithmetic guarantee.

- programs readably harmful on **every** harm unit: ['outlier_iqr']
- after forward position 3, the second harm unit

| harm unit | # | min slice | resolution | readably harmful legal programs (held-in) |
|---|---|---|---|---|
| MiddlePhalanxOutlineCorrect__impulse_v2 | 1 | 45 | 0.0222 | outlier_iqr -0.0944 |
| PowerCons__impulse_v2 | 3 | 12 | 0.0833 | outlier_iqr -0.1852, repair_level_shift -0.2778 |

## Deploy rule: refused winners (regression check)

- rule: Support drafts, delayed approves; a refused winner is not adopted as the ledger incumbent and the arm falls back to the last approved Workflow, or to identity when there is none
- rule lives in: `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:_incumbent_after_delayed`
- **a refused winner reached a deployment: False**
- 3 round(s) had a Support winner the delayed gate refused; none of them reached a deployment

| arm | round | Support winner | delayed relation | Skill approved | incumbent after round | deployed | deploy source |
|---|---|---|---|---|---|---|---|
| A3-reset | r1 | ['winsorize'] | ['NEGATIVE'] | none | cleared | identity | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY |
| K0-fixed | r1 | ['winsorize'] | ['NEGATIVE'] | none | cleared | identity | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY |
| A5-online | r1 | ['winsorize'] | ['NEGATIVE'] | none | cleared | identity | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY |

## A5 Slow integration at the boundary

- probe rows: 2; census rows: 2
- authorized TRY operators: none
- risk-authorized operators: none
- Skill written: False; execution right granted: False
- Slow LLM: 0 / 6

## Oracle isolation

- builtins.open, io.open, os.open, Path.open, Path.read_text and Path.read_bytes are wrapped at module import; any path containing artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach while the phase is 'arm'
- deliberate arm-phase probe fired the wall on every reader surface: **True** on `BeetleFly__burst_cls2.json` -- {'builtins.open': 'blocked', 'pathlib.Path.read_text': 'blocked', 'pathlib.Path.open': 'blocked'}
- keys the judging component read (after every arm closed): ['MiddlePhalanxOutlineCorrect__impulse_v2.json', 'PowerCons__impulse_v2.json']
- arm-phase attempts 3, blocked 3, leaks 0
- unblocked reads by phase: {'setup': 0, 'select': 0, 'judge': 2}

## Cost

- proposal-backend calls: 9 fast + 0 slow = 9 / 15  (**scripted** backend: these are sealed-probe calls, real LLM spend is 0)
- Consumer fits: 19
- wall clock: 33.3 s / 1800 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **shared_runner_unmodified**: True
- **full_curriculum_not_run**: True
- **units_run**: 1
- **rounds_per_adaptive_arm**: 1
- **oracle_isolation_mechanism**: builtins.open, io.open, os.open, Path.open, Path.read_text and Path.read_bytes are wrapped at module import; any path containing artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach while the phase is 'arm'
- **oracle_isolation_holds**: True
- **oracle_wall_selftest_fired**: True
- **oracle_wall_surfaces_probed**: {'builtins.open': 'blocked', 'pathlib.Path.read_text': 'blocked', 'pathlib.Path.open': 'blocked'}
- **harm_channel_exercised_on_unit_1**: True
- **instrument_blocker_reported_for_s1c**: False
- **feedback_surface_evidence_mode**: live
- **guard_formable_in_principle_on_this_course**: True
- **refused_winner_reached_a_deployment**: False
- **rounds_with_a_delayed_refused_winner**: 3
- **support_delayed_thresholds_and_relation_semantics_unchanged**: True
- **curriculum_revision**: r2
- **k0_has_no_target_local_capability**: True
- **k0_card_carries_no_frozen_steps**: True
- **live_llm_backend**: False
- **downloads**: 0
- **full_repo_pytest_not_run**: True
- **sealed_artifacts_not_rewritten**: True
