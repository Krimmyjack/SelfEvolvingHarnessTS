# S1b smoke -- four arms on curriculum unit 1

protocol: `s1_curriculum_four_arms_v1`  entry: `--smoke`  backend: **scripted_sealed_probe**  git: `e64c68444923725351fcf99ad7652e87cb884690`

**S1B_SMOKE_WIRED**

wiring only, on curriculum unit 1, one round per adaptive arm.  No Capability claim; the course was not run.

unit under test: `MoteStrain__impulse_v2` (harm_evidence, MoteStrain)

## Gates

- **four_arm_state_isolation**: True
- **domain_binding_hooks_live**: True
- **judging_produced_a_regret_table**: True
- **oracle_isolation_holds**: True
- **oracle_wall_proved_armed**: True
- **within_budget**: True
- **deploy_purity_clean**: True

## Four-arm readout

| arm | deploy | program | held-out utility | menu-oracle | regret | worst-class | harm | wrong promo | LLM | fits | probes | wasted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1142 | +0.1142 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 |
| A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1142 | +0.1142 | +0.0000 | False | 0 | 3 | 4 | 2 | 1 |
| K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1142 | +0.1142 | +0.0000 | False | 0 | 3 | 4 | 2 | 1 |
| A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1142 | +0.1142 | +0.0000 | False | 0 | 3 | 4 | 2 | 1 |

## State at the unit boundary

next unit would be `ECGFiveDays__impulse_v2`

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
- hook 2, next unit `ECGFiveDays__impulse_v2`; foreign Target-local dropped: True
- hook 2, decisions: none
- hook 3, Source card decision: no card minted
- Episode domain namespaces observed: ['MoteStrain__impulse_v2']

### Synthetic probe of the two walls

synthetic entries through the live decision functions; the arms minted none of these

- hook 2, a capability stamped with unit 1 offered to unit 2: carried = False
- hook 2, a capability stamped with unit 2 offered to unit 2: carried = True
- hook 2 behaves as specified: **True**
- hook 3, matching five-axis Scope admits: True
- hook 3, empty pattern intersection admits: False
- hook 3, wrong consumer admits: False
- hook 3, pattern mismatch admits: False (axes that differ between unit 1 and unit 2: ['period_change_score'])
- hook 3 behaves as specified: **True**

## Instrument finding (blocks S1c: **True**)

the course selected by the declared smallest-total-points rule gives 6 of its seven units a held-in slice of at most two rows, and 3 of those slices are empty outright, so the frozen two-round protocol has no r2 delayed surface to open there.  On unit 1 every probe read gain 0.0 and every Episode came back NEUTRAL: no harm evidence was written and the guard channel had nothing to compile.  The wiring is correct; the material is too coarse to exercise it.

- relations observed on unit 1: ['NEUTRAL']
- harm Episode formed on unit 1: False
- guard minted on unit 1: False
- the frozen two-round protocol has rows on every unit: False
- units with an empty held-in slice: [{'unit_id': 'MoteStrain__impulse_v2', 'empty_slices': ['r2_delayed']}, {'unit_id': 'BeetleFly__burst_cls2', 'empty_slices': ['r2_delayed']}, {'unit_id': 'MoteStrain__burst_cls2', 'empty_slices': ['r2_delayed']}]

the selection rule optimises for cheapness (smallest total points), which is the opposite of what a readable Support surface needs.  Either the rule changes -- e.g. a floor on support_pool_rows, or the largest-points qualifier instead of the smallest -- or the four held-in slices stop being quarters of an already tiny support pool.  Both are protocol changes and are outside this book; they are reported for the main line to arbitrate before S1c runs.

| # | unit | group | fit rows | support pool | slice rows (r1s/r1d/r2s/r2d) | smallest slice | smallest expressible gain |
|---|---|---|---|---|---|---|---|
| 1 | MoteStrain__impulse_v2 | harm_evidence | 14 | 6 | 2/2/2/0 | 0 | empty slice |
| 2 | ECGFiveDays__impulse_v2 | learnable_positive | 16 | 7 | 2/2/2/1 | 1 | 1.000 |
| 3 | Coffee__impulse_v2 | harm_evidence | 20 | 8 | 2/2/2/2 | 2 | 0.500 |
| 4 | SonyAIBORobotSurface1__burst_cls2 | identity | 13 | 7 | 2/2/2/1 | 1 | 1.000 |
| 5 | GunPoint__impulse_v2 | learnable_positive | 35 | 15 | 4/4/4/3 | 3 | 0.333 |
| 6 | BeetleFly__burst_cls2 | heldout_only_temptation | 14 | 6 | 2/2/2/0 | 0 | empty slice |
| 7 | MoteStrain__burst_cls2 | identity | 14 | 6 | 2/2/2/0 | 0 | empty slice |

- the harm channel needs NEGATIVE Episodes on two distinct units before risk_skill can compile a guard.  A one- or two-row Support slice cannot produce one: every candidate reads gain 0.0 and classify_relation returns NEUTRAL.  A zero-row slice is worse -- the frozen two-round protocol has no delayed surface to open on that unit at all.

## A5 Slow integration at the boundary

- probe rows: 2; census rows: 2
- authorized TRY operators: none
- risk-authorized operators: none
- Skill written: False; execution right granted: False
- Slow LLM: 0 / 6

## Oracle isolation

- builtins.open, io.open, os.open, Path.open, Path.read_text and Path.read_bytes are wrapped at module import; any path containing artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach while the phase is 'arm'
- deliberate arm-phase probe fired the wall on every reader surface: **True** on `BeetleFly__burst_cls2.json` -- {'builtins.open': 'blocked', 'pathlib.Path.read_text': 'blocked', 'pathlib.Path.open': 'blocked'}
- keys the judging component read (after every arm closed): ['MoteStrain__impulse_v2.json']
- arm-phase attempts 3, blocked 3, leaks 0
- unblocked reads by phase: {'setup': 0, 'select': 0, 'judge': 1}

## Cost

- proposal-backend calls: 9 fast + 0 slow = 9 / 15  (**scripted** backend: these are sealed-probe calls, real LLM spend is 0)
- Consumer fits: 13
- wall clock: 10.7 s / 1800 s
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
- **harm_channel_exercised_on_unit_1**: False
- **instrument_blocker_reported_for_s1c**: True
- **k0_has_no_target_local_capability**: True
- **k0_card_carries_no_frozen_steps**: True
- **live_llm_backend**: False
- **downloads**: 0
- **full_repo_pytest_not_run**: True
- **sealed_artifacts_not_rewritten**: True
