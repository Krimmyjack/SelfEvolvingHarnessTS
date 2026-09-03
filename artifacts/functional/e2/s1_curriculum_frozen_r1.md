# S1 four-arm evolution curriculum -- frozen course

protocol: `s1_curriculum_four_arms_v1`  evidence grade: **DEVELOPMENT**  git: `e64c68444923725351fcf99ad7652e87cb884690`

The course below was selected mechanically from the sealed census and oracle artifacts by the rules in section 1.  No unit was chosen by hand and no outcome reordered anything after the rules ran.

## 1. Selection rules (declared before scoring)

- **harm_evidence**: an oracle-scored unit qualifies when outlier_mad or outlier_iqr is legal on it (verifier passed and cohort modified fraction <= 0.10) and its held-in headroom is <= -0.005, i.e. materially harmful on held-in.  Take the two smallest-total-points qualifiers whose family_key differ.
- **learnable_positive**: learnability == LEARNABLE, family_key distinct from each other and from both harm units.  Take the two largest held-in headroom.
- **identity**: oracle_set is exactly identity (or empty).  Take the two smallest-total-points units whose family_key differ.
- **heldout_only_temptation**: learnability == HELDOUT_ONLY, not already selected.  Prefer a family_key no other selected unit uses; take the smallest total points.  If every HELDOUT_ONLY family is already used, drop the family preference and take the smallest total points.
- **total_points**: cell.official_train_rows x series_length, the same point count the S1a-r3 census enumerated the pool with
- **tie_break**: ascending (total_points, unit_id); no outcome may reorder
- **unit_disjointness**: no unit id may appear twice in the course
- **forward_order**: harm A -> learnable A -> harm B -> identity A -> learnable B -> HELDOUT_ONLY -> identity B.  Design intent: the guard should be compilable after the second harm unit, so every unit after it tests whether the guard actually fires.
- **reverse_order**: the exact reverse of the forward order
- **domain_namespace**: unit_id (dataset__injection).  The whole course runs at one condition (fit_only_artifact), so dataset alone would collapse two curriculum units of the same substrate into one counted Task and the guard census (risk_skill._task_of) would undercount.

## 2. The seven units, forward order

| # | unit | group | family | learnability | oracle set | points | held-in n | held-out n | why |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MoteStrain__impulse_v2 | harm_evidence | MoteStrain | HELDOUT_ONLY | hampel_filter | 1680 | 6 | 1252 | materially harmful legal outlier program on held-in: outlier_mad(held-in -0.1667), outlier_iqr(held-in -0.1667); smallest-points qualifier in its family |
| 2 | ECGFiveDays__impulse_v2 | learnable_positive | ECGFamily | LEARNABLE | repair_burst_segment | 3128 | 7 | 861 | LEARNABLE with held-in headroom 0.5714, the largest available in a family unused by the harm units |
| 3 | Coffee__impulse_v2 | harm_evidence | Coffee | N/A | identity | 8008 | 8 | 28 | materially harmful legal outlier program on held-in: outlier_mad(held-in -0.2500), outlier_iqr(held-in -0.2500); smallest-points qualifier in its family |
| 4 | SonyAIBORobotSurface1__burst_cls2 | identity | SonyAIBOFamily | N/A | identity | 1400 | 7 | 601 | oracle set is identity, so the correct end state is to change nothing; smallest-points identity unit in its family |
| 5 | GunPoint__impulse_v2 | learnable_positive | GunPointFamily | LEARNABLE | hampel_filter | 7500 | 15 | 150 | LEARNABLE with held-in headroom 0.4667, the largest available in a family unused by the harm units |
| 6 | BeetleFly__burst_cls2 | heldout_only_temptation | BeetleFly | HELDOUT_ONLY | outlier_iqr,hampel_filter | 10240 | 6 | 20 | HELDOUT_ONLY: the oracle-set program helps held-out (0.1000) but held-in cannot approve it (headroom 0.0000) -- the abstention temptation |
| 7 | MoteStrain__burst_cls2 | identity | MoteStrain | N/A | identity | 1680 | 6 | 1252 | oracle set is identity, so the correct end state is to change nothing; smallest-points identity unit in its family |

forward: `['MoteStrain__impulse_v2', 'ECGFiveDays__impulse_v2', 'Coffee__impulse_v2', 'SonyAIBORobotSurface1__burst_cls2', 'GunPoint__impulse_v2', 'BeetleFly__burst_cls2', 'MoteStrain__burst_cls2']`

reverse: `['MoteStrain__burst_cls2', 'BeetleFly__burst_cls2', 'GunPoint__impulse_v2', 'SonyAIBORobotSurface1__burst_cls2', 'Coffee__impulse_v2', 'ECGFiveDays__impulse_v2', 'MoteStrain__impulse_v2']`

Design intent of the forward order: the guard should become compilable after the second harm unit, so every unit after it is a test of whether the guard actually fires.

## 3. Families and substrates

- distinct families in course: **6** (BeetleFly, Coffee, ECGFamily, GunPointFamily, MoteStrain, SonyAIBOFamily)
- repeated families: ['MoteStrain']
- repeated substrates: ['MoteStrain']

## 4. Arms

- **Static**: no adaptation; identity frozen and deployed on every unit; only the scoring fit is spent
- **A3-reset**: cold start from h0 on every unit; zero carry between units
- **K0-fixed**: every unit starts from the same K0; normal in-unit held-in adaptation; no write-back between units
- **A5-online**: same K0 start and in-unit protocol; full Slow integration between units including the risk lifecycle; the pool evolves with the course

## 5. K0

- base: methods/ttha/harness/h0 (the three bootstrap Skills)
- bootstrap Skills: inspect_and_localize, build_contrastive_candidates, select_or_identity_and_verify
- inert Slow card: `source_investigation_cls_v1` from `artifacts/functional/e2/t6_cls_op_r2_three_arms.json`; TRY = `NO_AUTHORIZED_ACTIVE_RECOMMENDATION`; allowed_tools = []; carries frozen steps = False
- **excluded on purpose**: the C40 Target-local hampel capability is NOT in K0.  It is a frozen-steps capability bound to one Source domain; placing it in K0 would leak an answer across domains and contaminate both K0-fixed and A5-online.

## 6. Domain-binding hooks

- **1_stamp**: unit_id (dataset__injection).  The whole course runs at one condition (fit_only_artifact), so dataset alone would collapse two curriculum units of the same substrate into one counted Task and the guard census (risk_skill._task_of) would undercount.
- **2_carry_wall**: a Target-local capability (frozen program steps; not an experience card) is dropped at the unit boundary unless its stamp equals the next unit's domain_namespace
- **3_scope_v1**: a Source-derived experience card reaches the next unit's Fast surface only when task_kind, consumer_id and metric match, its authorizing pattern-view intersection is non-empty and is satisfied by the next unit's binned deployment-visible pattern view, and its Program geometry is a real operator.  Dataset name is not an axis.

## 7. Budgets

- llm_per_unit_per_adaptive_arm: 15
- fit_per_unit_per_arm: 25
- llm_per_slow_integration: 6
- llm_total_cap: 400
- fit_total_cap: 900
- wall_seconds_cap: 10800
- over_cap_verdict: COMPUTE_BUDGET_EXCEEDED

## 8. Pre-registered readout (judged in S1c, not here)

- **primary**: A5-online must be non-inferior to both A3-reset and K0-fixed on quality (cumulative held-out utility) and on harm (harm events, worst-class harm), and must improve at least one of cumulative regret or cumulative total cost by a material margin
- **material_threshold**: 0.005
- **verdict_ceiling_for_a_single_order_single_run**: S1_DEVELOPMENT_EVOLUTION_SIGNAL
- **judged_in**: S1c.  This book freezes the readout and does not judge it.

## Oracle isolation

- builtins.open, io.open, os.open, Path.open, Path.read_text and Path.read_bytes are wrapped at module import; any path containing artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach while the phase is 'arm'
- arm-phase attempts: 0 (blocked 0, leaks 0)

## Not in this book

- the full course is not run here.  This entry freezes the course and the readout only; S1c runs it.

