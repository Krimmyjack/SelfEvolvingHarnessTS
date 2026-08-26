# S1c -- four-arm evolution course, forward order

protocol: `s1_curriculum_four_arms_v1`  run-id: `s1_course_fwd_run1`  entry: `--run-course --order forward`  backend: **live_fast_agent**  returned_model: `gpt-5.6-sol`  git: `987a13c0c06eed2f37e123d3f75a3272977c8fc7`

**NEGATIVE_TRANSFER**

A5-online is worse than cold-start A3-reset on quality or harm.  first-fault: loop_1: same-operator harm was not sampled on both harm units, so the two-Task guard floor cannot compile from live Episodes; loop_2: no guard entered the A5 Fast view; A5-online is worse than A3-reset on quality or harm

ceiling for a single order / single run: `S1_DEVELOPMENT_EVOLUTION_SIGNAL`.  Regret is never cited without harm / worst-class.

## Frozen course (r2, forward)

`['MiddlePhalanxOutlineCorrect__impulse_v2', 'DistalPhalanxOutlineCorrect__burst_cls2', 'PowerCons__impulse_v2', 'FreezerRegularTrain__burst_cls2', 'GunPointOldVersusYoung__impulse_v2', 'ECG200__impulse_v2', 'Ham__impulse_v2']`

## Per-unit per-arm readout

| # | unit | group | arm | deploy | program | held-out | oracle | regret | worst-class | harm | wrong promo | LLM | fits | probes | wasted | Support=delayed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | harm_evidence | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | harm_evidence | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 10 | 8 | 3 | 0 | 0/1 |
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | harm_evidence | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 10 | 5 | 2 | 1 | 0/0 |
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | harm_evidence | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0103 | +0.0103 | +0.0000 | False | 0 | 10 | 9 | 4 | 0 | 0/1 |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | learnable_positive | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0181 | +0.0181 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | learnable_positive | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0181 | +0.0181 | +0.0000 | False | 0 | 11 | 7 | 2 | 0 | 1/1 |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | learnable_positive | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0181 | +0.0181 | +0.0000 | False | 0 | 11 | 5 | 2 | 0 | 0/0 |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | learnable_positive | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0181 | +0.0181 | +0.0000 | False | 0 | 10 | 7 | 2 | 0 | 1/1 |
| 3 | PowerCons__impulse_v2 | harm_evidence | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1333 | +0.1333 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 3 | PowerCons__impulse_v2 | harm_evidence | A3-reset | FROZEN_ACTIVE_SKILL_RECALL | hampel_filter | +0.0833 | +0.1333 | +0.0500 | +0.0444 | False | 0 | 8 | 8 | 2 | 0 | 1/1 |
| 3 | PowerCons__impulse_v2 | harm_evidence | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1333 | +0.1333 | +0.0000 | False | 0 | 10 | 6 | 3 | 0 | 0/0 |
| 3 | PowerCons__impulse_v2 | harm_evidence | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1333 | +0.1333 | +0.0000 | False | 0 | 6 | 1 | 0 | 0 | 0/0 |
| 4 | FreezerRegularTrain__burst_cls2 | identity | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 4 | FreezerRegularTrain__burst_cls2 | identity | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 8 | 5 | 2 | 1 | 0/0 |
| 4 | FreezerRegularTrain__burst_cls2 | identity | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 8 | 5 | 2 | 1 | 0/0 |
| 4 | FreezerRegularTrain__burst_cls2 | identity | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 9 | 3 | 1 | 0 | 0/0 |
| 5 | GunPointOldVersusYoung__impulse_v2 | learnable_positive | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1841 | +0.1841 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 5 | GunPointOldVersusYoung__impulse_v2 | learnable_positive | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1841 | +0.1841 | +0.0000 | False | 0 | 7 | 1 | 1 | 1 | 0/0 |
| 5 | GunPointOldVersusYoung__impulse_v2 | learnable_positive | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1841 | +0.1841 | +0.0000 | False | 0 | 9 | 5 | 4 | 3 | 1/1 |
| 5 | GunPointOldVersusYoung__impulse_v2 | learnable_positive | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.1841 | +0.1841 | +0.0000 | False | 0 | 11 | 1 | 2 | 2 | 0/0 |
| 6 | ECG200__impulse_v2 | heldout_only_temptation | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0400 | +0.0400 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 6 | ECG200__impulse_v2 | heldout_only_temptation | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0400 | +0.0400 | +0.0000 | False | 0 | 10 | 3 | 3 | 2 | 0/0 |
| 6 | ECG200__impulse_v2 | heldout_only_temptation | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0400 | +0.0400 | +0.0000 | False | 0 | 9 | 3 | 3 | 3 | 0/0 |
| 6 | ECG200__impulse_v2 | heldout_only_temptation | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0400 | +0.0400 | +0.0000 | False | 0 | 9 | 1 | 1 | 1 | 0/0 |
| 7 | Ham__impulse_v2 | identity | Static | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 0 | 1 | 0 | 0 | 0/0 |
| 7 | Ham__impulse_v2 | identity | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 9 | 1 | 1 | 1 | 0/0 |
| 7 | Ham__impulse_v2 | identity | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 9 | 7 | 4 | 2 | 1/1 |
| 7 | Ham__impulse_v2 | identity | A5-online | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | +0.0000 | +0.0000 | +0.0000 | +0.0000 | False | 0 | 9 | 7 | 3 | 1 | 1/1 |

## Cumulative totals (regret must be read with harm)

| arm | units | cum. regret | cum. held-out | harm events | worst-class min | wrong promo | wasted | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|
| Static | 7 | +0.3859 | +0.0000 | 0 | +0.0000 | 0 | 0 | 0 | 7 |
| A3-reset | 7 | +0.3026 | +0.0833 | 0 | +0.0000 | 0 | 5 | 63 | 33 |
| K0-fixed | 7 | +0.3859 | +0.0000 | 0 | +0.0000 | 0 | 10 | 66 | 36 |
| A5-online | 7 | +0.3859 | +0.0000 | 0 | +0.0000 | 0 | 4 | 64 | 29 |

A5 Slow integration LLM (included in A5 total cost): 6

### Cumulative regret curve

| # | unit | Static | A3-reset | K0-fixed | A5-online |
|---|---|---|---|---|---|
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | +0.0103 | +0.0103 | +0.0103 | +0.0103 |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | +0.0284 | +0.0284 | +0.0284 | +0.0284 |
| 3 | PowerCons__impulse_v2 | +0.1618 | +0.0784 | +0.1618 | +0.1618 |
| 4 | FreezerRegularTrain__burst_cls2 | +0.1618 | +0.0784 | +0.1618 | +0.1618 |
| 5 | GunPointOldVersusYoung__impulse_v2 | +0.3459 | +0.2626 | +0.3459 | +0.3459 |
| 6 | ECG200__impulse_v2 | +0.3859 | +0.3026 | +0.3859 | +0.3859 |
| 7 | Ham__impulse_v2 | +0.3859 | +0.3026 | +0.3859 | +0.3859 |

## Guard timeline and the two untested loops

forward order places the second harm unit at position 3; the guard is compilable after that unit if the same Program family was sampled as harmful on both harm units

### Loop 1 -- same-operator harm sampled on both harm units?

no: A5-online did not write a same-operator NEGATIVE Episode on both harm units

- **A3-reset**: {'MiddlePhalanxOutlineCorrect__impulse_v2': ['repair_level_shift'], 'PowerCons__impulse_v2': []}  shared=none
- **K0-fixed**: {'MiddlePhalanxOutlineCorrect__impulse_v2': [], 'PowerCons__impulse_v2': ['outlier_iqr', 'repair_level_shift']}  shared=none
- **A5-online**: {'MiddlePhalanxOutlineCorrect__impulse_v2': ['repair_level_shift'], 'PowerCons__impulse_v2': []}  shared=none

### Loop 2 -- after a guard is in view, does the proposer avoid it?

guard never entered the A5-online Fast view

- expected earliest position: 3
- A5 guard formed at position: None
- later rounds that respected the guard: 0
- later rounds that probed a guarded operator: 0

## Winsorize Support-positive / delayed-negative

no winsorize episode was written on this run

- count: **0**  by arm: {}

## Backend identity

- family: shared runner _live_backend / _live_agent
- expected: `gpt-5.6-sol` @ `https://api.agicto.cn/v1`
- probe ok: **True**  returned_model: `gpt-5.6-sol`
- probe charged to course cap: False

## Oracle isolation

- arm-phase wall fired: **True** on `BeetleFly__burst_cls2.json`
- arm-phase attempts 3, blocked 3, leaks 0
- judge-phase keys read: ['DistalPhalanxOutlineCorrect__burst_cls2.json', 'ECG200__impulse_v2.json', 'FreezerRegularTrain__burst_cls2.json', 'GunPointOldVersusYoung__impulse_v2.json', 'Ham__impulse_v2.json', 'MiddlePhalanxOutlineCorrect__impulse_v2.json', 'PowerCons__impulse_v2.json']

## Cost

- Fast LLM: 193
- Slow LLM: 6
- total LLM: 199 / 400
- Consumer fits: 105 / 900
- wall clock: 3509.3 s / 10800 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **shared_runner_unmodified**: True
- **course_and_budgets_unmodified**: True
- **live_llm_backend**: True
- **backend_identity**: gpt-5.6-sol @ https://api.agicto.cn/v1 (shared runner _live_agent / SLOW_MODEL)
- **no_backend_swap**: True
- **two_untested_loops_reported**: True
- **no_a3_a5_adaptation_outside_this_course**: True
- **no_injection_scan**: True
- **oracle_isolated**: True
- **sealed_oracles_not_rewritten**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True

## Outside the book

- A5 carry rebuilds from K0 plus every skill the wall has admitted so far, not only the last unit's additions; otherwise a guard minted at position 3 would vanish at position 5.  carry_decision / hook 2 / hook 3 are unchanged.
- The Slow-authored source card still uses skill_id source_investigation_cls_v1, which is already in K0, so carry_into_next_unit drops it (pre-existing).  The tested channel is the risk-guard path, not TRY.
