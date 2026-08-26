# S1a-r3 remaining-pool census

protocol: `s1a_r3_pool_census_v1`  parent r1: `837b537`  parent r2: `e74c021`  evidence grade: **development**

0 LLM.  One-shot take-what-comes.  No r4.  Sealed oracles for new units only; r1/r2 artifacts not overwritten.

## Isolation

本文件不得进入任何臂的 prompt/store/检索视野

## 1. Pool enumeration (frozen before scoring)

every zip in data/ucr_task_context; keep iff binary AND loadable AND official TRAIN rows*length <= 100000 AND dataset not among the 8 substrates that appear in the r1 9 units.  Each kept substrate x {impulse_v2, burst_cls2}.  Roster is frozen before the first oracle score.

zip count=40; included substrates=19; excluded=21; declared units=38

### Included substrates

| dataset | family | TRAIN rows | L | points |
|---|---|---|---|---|
| BeetleFly | BeetleFly | 20 | 512 | 10240 |
| BirdChicken | BirdChicken | 20 | 512 | 10240 |
| Coffee | Coffee | 28 | 286 | 8008 |
| DistalPhalanxOutlineCorrect | PhalanxFamily | 600 | 80 | 48000 |
| ECGFiveDays | ECGFamily | 23 | 136 | 3128 |
| FreezerRegularTrain | FreezerFamily | 150 | 301 | 45150 |
| FreezerSmallTrain | FreezerFamily | 28 | 301 | 8428 |
| GunPointMaleVersusFemale | GunPointFamily | 135 | 150 | 20250 |
| GunPointOldVersusYoung | GunPointFamily | 136 | 150 | 20400 |
| HouseTwenty | HouseTwenty | 40 | 2000 | 80000 |
| MiddlePhalanxOutlineCorrect | PhalanxFamily | 600 | 80 | 48000 |
| MoteStrain | MoteStrain | 20 | 84 | 1680 |
| PowerCons | PowerCons | 180 | 144 | 25920 |
| ProximalPhalanxOutlineCorrect | PhalanxFamily | 600 | 80 | 48000 |
| ShapeletSim | ShapeletSim | 20 | 500 | 10000 |
| SonyAIBORobotSurface1 | SonyAIBOFamily | 20 | 70 | 1400 |
| SonyAIBORobotSurface2 | SonyAIBOFamily | 27 | 65 | 1755 |
| ToeSegmentation2 | ToeSegmentationFamily | 36 | 343 | 12348 |
| TwoLeadECG | TwoLeadECG | 23 | 82 | 1886 |

### Excluded substrates

| dataset | reason | rows | L | points |
|---|---|---|---|---|
| Computers | train_points_over_100000 | 250 | 720 | 180000 |
| DodgerLoopWeekend | not_finite | 20 | 288 | 5760 |
| Earthquakes | train_points_over_100000 | 322 | 512 | 164864 |
| ECG200 | in_r1_tested_units | 100 | 96 | 9600 |
| FordA | train_points_over_100000 | 3601 | 500 | 1800500 |
| FordB | train_points_over_100000 | 3636 | 500 | 1818000 |
| GunPoint | in_r1_tested_units | 50 | 150 | 7500 |
| GunPointAgeSpan | in_r1_tested_units | 135 | 150 | 20250 |
| Ham | in_r1_tested_units | 109 | 431 | 46979 |
| HandOutlines | train_points_over_100000 | 1000 | 2709 | 2709000 |
| Herring | in_r1_tested_units | 64 | 512 | 32768 |
| KeplerLightCurves | no_TRAIN_member | None | None | None |
| Lightning2 | in_r1_tested_units | 60 | 637 | 38220 |
| PhalangesOutlinesCorrect | train_points_over_100000 | 1800 | 80 | 144000 |
| SemgHandGenderCh2 | train_points_over_100000 | 300 | 1500 | 450000 |
| Strawberry | train_points_over_100000 | 613 | 235 | 144055 |
| ToeSegmentation1 | in_r1_tested_units | 40 | 277 | 11080 |
| Wafer | train_points_over_100000 | 1000 | 152 | 152000 |
| Wine | in_r1_tested_units | 57 | 234 | 13338 |
| WormsTwoClass | train_points_over_100000 | 181 | 900 | 162900 |
| Yoga | train_points_over_100000 | 300 | 426 | 127800 |

## 2. Unit oracle + learnability + family

Learnability = r2 predicate on the sealed held-in pool (`classify_relation == POSITIVE`; experience_memory.py:411-451; method.py:742-757 / 1466-1492).  Family = name prefix + pattern_view byte-equality merge.

| unit | src | family | oracle set | learnability | held-in | held-out | construction |
|---|---|---|---|---|---|---|---|
| GunPointAgeSpan__impulse_v2 | r1_sealed | GunPointFamily | hampel_filter | **LEARNABLE** | 0.37499999999999994 | 0.2626582278481012 |  |
| GunPoint__impulse_v2 | r1_sealed | GunPointFamily | hampel_filter | **LEARNABLE** | 0.4666666666666667 | 0.4066666666666667 |  |
| ECG200__impulse_v2 | r1_sealed | ECGFamily | repair_burst_segment | **HELDOUT_ONLY** | 0.0 | 0.040000000000000036 |  |
| Wine__impulse_v2 | r1_sealed | Wine | identity | **N/A** | 0.0 | 0.0 |  |
| ToeSegmentation1__impulse_v2 | r1_sealed | ToeSegmentationFamily | repair_burst_segment | **LEARNABLE** | 0.08333333333333326 | 0.030701754385964952 |  |
| Lightning2__impulse_v2 | r1_sealed | Lightning2 | repair_burst_segment | **LEARNABLE** | 0.16666666666666663 | 0.09836065573770492 |  |
| Herring__impulse_v2 | r1_sealed | Herring | hampel_filter | **HELDOUT_ONLY** | 0.0 | 0.046875 |  |
| Ham__impulse_v2 | r1_sealed | Ham | identity | **N/A** | 0.0 | 0.0 |  |
| GunPoint__burst_cls2 | r1_sealed | GunPointFamily | outlier_iqr | **HELDOUT_ONLY** | 0.0 | 0.013333333333333308 |  |
| BeetleFly__impulse_v2 | r3_sealed_reused | BeetleFly | identity | **N/A** | 0.0 | 0.0 |  |
| BeetleFly__burst_cls2 | r3_sealed_reused | BeetleFly | outlier_iqr,hampel_filter | **HELDOUT_ONLY** | 0.0 | 0.10000000000000003 |  |
| BirdChicken__impulse_v2 | r3_sealed_reused | BirdChicken | identity | **N/A** | 0.0 | 0.0 |  |
| BirdChicken__burst_cls2 | r3_sealed_reused | BirdChicken | identity | **N/A** | 0.0 | 0.0 |  |
| Coffee__impulse_v2 | r3_sealed_reused | Coffee | identity | **N/A** | 0.0 | 0.0 |  |
| Coffee__burst_cls2 | r3_sealed_reused | Coffee | identity | **N/A** | 0.0 | 0.0 |  |
| DistalPhalanxOutlineCorrect__impulse_v2 | r3_sealed_reused | PhalanxFamily | outlier_mad | **HELDOUT_ONLY** | 0.005555555555555536 | 0.1123188405797102 |  |
| DistalPhalanxOutlineCorrect__burst_cls2 | r3_sealed_reused | PhalanxFamily | outlier_iqr,outlier_mad | **LEARNABLE** | 0.033333333333333326 | 0.018115942028985588 |  |
| ECGFiveDays__impulse_v2 | r3_sealed_reused | ECGFamily | repair_burst_segment | **LEARNABLE** | 0.5714285714285714 | 0.321718931475029 |  |
| ECGFiveDays__burst_cls2 | r3_sealed_reused | ECGFamily | identity | **N/A** | 0.0 | 0.0 |  |
| FreezerRegularTrain__impulse_v2 | r3_sealed_reused | FreezerFamily | identity | **N/A** | 0.0 | 0.0 |  |
| FreezerRegularTrain__burst_cls2 | r3_sealed_reused | FreezerFamily | identity | **N/A** | 0.0 | 0.0 |  |
| FreezerSmallTrain__impulse_v2 | r3_sealed_reused | FreezerFamily | identity | **N/A** | 0.0 | 0.0 |  |
| FreezerSmallTrain__burst_cls2 | r3_sealed_reused | FreezerFamily | identity | **N/A** | 0.0 | 0.0 |  |
| GunPointMaleVersusFemale__impulse_v2 | r3_sealed_reused | GunPointFamily | hampel_filter | **LEARNABLE** | 0.15000000000000002 | 0.19303797468354422 |  |
| GunPointMaleVersusFemale__burst_cls2 | r3_sealed_reused | GunPointFamily | outlier_iqr,hampel_filter,repair_level_shift | **LEARNABLE** | 0.09999999999999998 | 0.06645569620253167 |  |
| GunPointOldVersusYoung__impulse_v2 | r3_sealed_reused | GunPointFamily | hampel_filter | **LEARNABLE** | 0.41463414634146345 | 0.18412698412698414 |  |
| GunPointOldVersusYoung__burst_cls2 | r3_sealed_reused | GunPointFamily | identity | **N/A** | 0.0 | 0.0 |  |
| HouseTwenty__impulse_v2 | r3_sealed_reused | HouseTwenty | identity | **N/A** | 0.0 | 0.0 |  |
| HouseTwenty__burst_cls2 | r3_sealed_reused | HouseTwenty | identity | **N/A** | 0.0 | 0.0 |  |
| MiddlePhalanxOutlineCorrect__impulse_v2 | r3_sealed_reused | PhalanxFamily | repair_level_shift | **HELDOUT_ONLY** | 0.0 | 0.010309278350515483 |  |
| MiddlePhalanxOutlineCorrect__burst_cls2 | r3_sealed_reused | PhalanxFamily | identity | **N/A** | 0.0 | 0.0 |  |
| MoteStrain__impulse_v2 | r3_sealed_reused | MoteStrain | hampel_filter | **HELDOUT_ONLY** | 0.0 | 0.1142172523961662 |  |
| MoteStrain__burst_cls2 | r3_sealed_reused | MoteStrain | identity | **N/A** | 0.0 | 0.0 |  |
| PowerCons__impulse_v2 | r3_sealed_reused | PowerCons | hampel_filter | **LEARNABLE** | 0.20370370370370372 | 0.1333333333333333 |  |
| PowerCons__burst_cls2 | r3_sealed_reused | PowerCons | hampel_filter | **LEARNABLE** | 0.18518518518518523 | 0.11666666666666659 |  |
| ProximalPhalanxOutlineCorrect__impulse_v2 | r3_sealed_reused | PhalanxFamily | identity | **N/A** | 0.0 | 0.0 |  |
| ProximalPhalanxOutlineCorrect__burst_cls2 | r3_sealed_reused | PhalanxFamily | winsorize,outlier_iqr,outlier_mad | **LEARNABLE** | 0.011111111111111072 | 0.027491408934707917 |  |
| ShapeletSim__impulse_v2 | r3_sealed_reused | ShapeletSim | identity | **N/A** | 0.0 | 0.0 |  |
| ShapeletSim__burst_cls2 | r3_sealed_reused | ShapeletSim | identity | **N/A** | 0.0 | 0.0 |  |
| SonyAIBORobotSurface1__impulse_v2 | r3_construction_failed | SonyAIBOFamily | — | **N/A** | None | None | v2_segment_length_zero_at_L=70 |
| SonyAIBORobotSurface1__burst_cls2 | r3_sealed_reused | SonyAIBOFamily | identity | **N/A** | 0.0 | 0.0 |  |
| SonyAIBORobotSurface2__impulse_v2 | r3_construction_failed | SonyAIBOFamily | — | **N/A** | None | None | v2_segment_length_zero_at_L=65 |
| SonyAIBORobotSurface2__burst_cls2 | r3_sealed_reused | SonyAIBOFamily | hampel_filter | **HELDOUT_ONLY** | 0.0 | 0.022035676810073457 |  |
| ToeSegmentation2__impulse_v2 | r3_sealed_reused | ToeSegmentationFamily | identity | **N/A** | 0.0 | 0.0 |  |
| ToeSegmentation2__burst_cls2 | r3_sealed_reused | ToeSegmentationFamily | hampel_filter,repair_burst_segment | **HELDOUT_ONLY** | 0.0 | 0.023076923076923106 |  |
| TwoLeadECG__impulse_v2 | r3_sealed_reused | TwoLeadECG | outlier_mad | **LEARNABLE** | 0.14285714285714285 | 0.06760316066725197 |  |
| TwoLeadECG__burst_cls2 | r3_sealed_reused | TwoLeadECG | identity | **N/A** | 0.0 | 0.0 |  |

## 3. Program clusters (oracle operator + Scope v1)

### `hampel_filter`

- LEARNABLE **6** / oracle-members 12; HELDOUT_ONLY 6; independent families **2** (GunPointFamily, PowerCons); name-prefix families 2 (GunPointFamily, PowerCons)
- all-learnable Scope-v1 intersection nonempty: **True**
- learnable: GunPointAgeSpan__impulse_v2, GunPoint__impulse_v2, GunPointMaleVersusFemale__impulse_v2, GunPointOldVersusYoung__impulse_v2, PowerCons__impulse_v2, PowerCons__burst_cls2
- held-out only: Herring__impulse_v2, BeetleFly__burst_cls2, GunPointMaleVersusFemale__burst_cls2, MoteStrain__impulse_v2, SonyAIBORobotSurface2__burst_cls2, ToeSegmentation2__burst_cls2

### `repair_burst_segment`

- LEARNABLE **3** / oracle-members 5; HELDOUT_ONLY 2; independent families **3** (ECGFamily, Lightning2, ToeSegmentationFamily); name-prefix families 3 (ECGFamily, Lightning2, ToeSegmentationFamily)
- all-learnable Scope-v1 intersection nonempty: **True**
- learnable: ToeSegmentation1__impulse_v2, Lightning2__impulse_v2, ECGFiveDays__impulse_v2
- held-out only: ECG200__impulse_v2, ToeSegmentation2__burst_cls2

### `outlier_iqr`

- LEARNABLE **3** / oracle-members 5; HELDOUT_ONLY 2; independent families **2** (GunPointFamily, PhalanxFamily); name-prefix families 2 (GunPointFamily, PhalanxFamily)
- all-learnable Scope-v1 intersection nonempty: **True**
- learnable: DistalPhalanxOutlineCorrect__burst_cls2, GunPointMaleVersusFemale__burst_cls2, ProximalPhalanxOutlineCorrect__burst_cls2
- held-out only: GunPoint__burst_cls2, BeetleFly__burst_cls2

### `outlier_mad`

- LEARNABLE **3** / oracle-members 4; HELDOUT_ONLY 1; independent families **2** (PhalanxFamily, TwoLeadECG); name-prefix families 2 (PhalanxFamily, TwoLeadECG)
- all-learnable Scope-v1 intersection nonempty: **True**
- learnable: DistalPhalanxOutlineCorrect__burst_cls2, ProximalPhalanxOutlineCorrect__burst_cls2, TwoLeadECG__impulse_v2
- held-out only: DistalPhalanxOutlineCorrect__impulse_v2

### `repair_level_shift`

- LEARNABLE **1** / oracle-members 2; HELDOUT_ONLY 1; independent families **1** (GunPointFamily); name-prefix families 1 (GunPointFamily)
- all-learnable Scope-v1 intersection nonempty: **True**
- learnable: GunPointMaleVersusFemale__burst_cls2
- held-out only: MiddlePhalanxOutlineCorrect__impulse_v2

### `winsorize`

- LEARNABLE **1** / oracle-members 1; HELDOUT_ONLY 0; independent families **1** (PhalanxFamily); name-prefix families 1 (PhalanxFamily)
- all-learnable Scope-v1 intersection nonempty: **True**
- learnable: ProximalPhalanxOutlineCorrect__burst_cls2
- held-out only: —

## 4. Verdict

**POOL_EXHAUSTED_FOR_TRY_CHANNEL**

the pre-declared local pool is exhausted.  No Program cluster has 3 independent-family LEARNABLE positives plus a fourth LEARNABLE matching field.  Closest: program=repair_burst_segment LEARNABLE=3 families=3.  have 3 independent LEARNABLE families but no trio has a non-empty Scope-v1 pattern intersection plus a fourth LEARNABLE matching field.

### Closest miss per program

| program | LEARNABLE | families | 3+1 hits | blocked | note |
|---|---|---|---|---|---|
| repair_burst_segment | 3 | 3 | 0 | fewer_than_4_learnable_units_need_3_plus_exam_field;no_3_independent_families_with_nonempty_scope_and_exam_field | have 3 independent LEARNABLE families but no trio has a non-empty Scope-v1 pattern intersection plus a fourth LEARNABLE matching field. |
| hampel_filter | 6 | 2 | 0 | fewer_than_3_independent_learnable_families | independent LEARNABLE families=2 (need 3); LEARNABLE units=6 (need >=4).  Short by 1 family and/or 0 learnable unit(s). |
| outlier_iqr | 3 | 2 | 0 | fewer_than_4_learnable_units_need_3_plus_exam_field;fewer_than_3_independent_learnable_families | independent LEARNABLE families=2 (need 3); LEARNABLE units=3 (need >=4).  Short by 1 family and/or 1 learnable unit(s). |
| outlier_mad | 3 | 2 | 0 | fewer_than_4_learnable_units_need_3_plus_exam_field;fewer_than_3_independent_learnable_families | independent LEARNABLE families=2 (need 3); LEARNABLE units=3 (need >=4).  Short by 1 family and/or 1 learnable unit(s). |
| repair_level_shift | 1 | 1 | 0 | fewer_than_4_learnable_units_need_3_plus_exam_field;fewer_than_3_independent_learnable_families | independent LEARNABLE families=1 (need 3); LEARNABLE units=1 (need >=4).  Short by 2 families and/or 3 learnable unit(s). |
| winsorize | 1 | 1 | 0 | fewer_than_4_learnable_units_need_3_plus_exam_field;fewer_than_3_independent_learnable_families | independent LEARNABLE families=1 (need 3); LEARNABLE units=1 (need >=4).  Short by 2 families and/or 3 learnable unit(s). |

## Cost

- Fast LLM: 0
- Slow LLM: 0
- Consumer fits: 342 / 600 (this pass 0)
- wall clock: 435.52 s / 5400 s
- downloads: 0
- units scored / declared: 36 / 38

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **no_fast_llm**: True
- **no_slow_llm**: True
- **no_a3_a5_adaptation_arm**: True
- **no_injection_scan**: True
- **no_pool_edit_after_declaration**: True
- **no_r4**: True
- **r1_artifacts_not_overwritten**: True
- **r2_artifacts_not_overwritten**: True
- **r1_sealed_oracles_not_rewritten**: True
- **oracle_isolated**: True
- **downloads**: 0
- **ucr_conf_downloaded_not_opened**: True
- **fit_budget_held**: True
- **wall_clock_held**: True
- **full_repo_pytest_not_run**: True
- **learnability_reuses_r2_predicate**: True

## Outside the book

- SonyAIBO L=65/70 makes v2 segment=round(L/150)=0; those impulse units are construction failures, not silent drops.
- Independence keys are union-find over LEARNABLE members only (name prefix OR byte-equal pattern_view).  A first draft that unioned every name-family sharing any unit's pattern_view collapsed GunPoint/PowerCons/ECG into BeetleFly via identity rows; that merge was rejected before the verdict was filed.  Sealed oracle numbers were not rescored.
- ECG200 (r1) and ECGFiveDays share name prefix ECG → ECGFamily; TwoLeadECG does not.
- Phalanx OutlineCorrect trio is one family; Freezer* is one family; GunPoint MaleVersusFemale/OldVersusYoung join the existing GunPointFamily and cannot add independence.
- classification online_loop still does not write task_episode_id; Fast-guard stays off.  Unchanged from r1/r2.
