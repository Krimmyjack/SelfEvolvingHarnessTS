# CAP-2 Stage 1: structural blind selection

**Verdict: CAP2_CANDIDATE_POOL_EMPTY.**  Frozen candidates: **none**

Written before any dataset zip was fetched (downloads so far: 0).  The pool is the official published table at `https://timeseriesclassification.com/aeon-toolkit/metadata.csv` -- 190 rows, sha256 `1e33662990291fc43568f13057684afe124859225164f46ac81249475a3fe528`.  the fresh fetch is byte-identical to the copy the 2026-08-25 CLS-CONF-dl book archived (same 7253 bytes, same sha256), so this census is stable and the selection below is reproducible from the same public table

## Filters

| filter | rule |
|---|---|
| binary | NumberClasses == 2 |
| equal_length | Length != 0 (site convention) |
| univariate | Channels == 1 |
| train_rows | 40 <= TrainSize <= 400 |
| total_points | (TrainSize + TestSize) * Length <= 100000 |
| template_gate | derived from the live injection template and the modification cap; see per-row citations |
| name_exclusion | not among the local archive stems nor any name the download roster mentions |

### Template compatibility gate, derived not invented

| clause | source |
|---|---|
| segment_rule | `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:430` |
| segment_positive_or_raise | `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:455` |
| overflow_or_raise | `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:464` |
| position_geometry | `evaluation/functional/run_e2_task_context_label_evidence_witness.py:88` |
| spike_fractions | `evaluation/functional/run_e2_task_context_label_evidence_witness.py:37` |
| modification_cap | `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:301` |

Spike fractions are the frozen four, so the artifact occupies four segments of `round(length/150)` samples each; the modification cap is 0.10.  At every admitted length the footprint is far inside the cap, so this clause never decides a candidate on its own -- the binding clause is the segment length and the end-margin geometry.

## Stop, before any download

`STOPPED_BEFORE_ANY_DOWNLOAD` -- CAP-2 section 1's conjunction admits zero names, so there is no C1/C2/C3 to freeze and Stage 2 has nothing to open

- **why_not_scope_coverage_limited**: section 3.6's SCOPE_COVERAGE_LIMITED is a verdict about the card's Scope, reached after three candidates are unsealed and their Pattern views computed.  Nothing was unsealed here and no Scope decision was made, so reusing that label would claim evidence this run does not have

- **nearest_miss_deliberately_not_taken**: DodgerLoopGame fails exactly one clause -- TRAIN 20 against the frozen [40,400] band -- and passes every other, including the template gate and the point budget.  Relaxing the band to admit it is precisely the move this book forbids, so it was not made and is reported instead

- **requires**: a mainline decision: widen a section 1 clause, change the pool, or record the gap as structural.  The executor has no discretion over any of the three

## Every genuinely fresh name the archive offers

binary, equal-length, univariate, and neither already local nor named in the download roster -- 5 rows.  these are the only names in the published archive that CAP-2 could newly seal; each fails at least one section 1 clause on public metadata alone, with no value or label read

| dataset | train | test | length | total points | template gate | fails |
|---|---|---|---|---|---|---|
| Chinatown | 20 | 345 | 24 | 8760 | False | train_rows_20_outside_[40,400]; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| DodgerLoopGame | 20 | 138 | 288 | 45504 | True | train_rows_20_outside_[40,400] |
| ElectricDeviceDetection | 623 | 3767 | 256 | 1123840 | True | train_rows_623_outside_[40,400]; total_points_1123840_over_100000 |
| RightWhaleCalls | 10934 | 1962 | 4000 | 51584000 | True | train_rows_10934_outside_[40,400]; total_points_51584000_over_100000 |
| SharePriceIncrease | 965 | 965 | 60 | 115800 | False | train_rows_965_outside_[40,400]; total_points_115800_over_100000; template_gate:segment_length_0_below_template_minimum |

## Is the empty pool real?

| name exclusions applied | eligible |
|---|---|
| as frozen (local + roster) | none |
| roster exclusion dropped | none |
| both dropped | ECG200, GunPoint, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, Ham, Herring, Lightning2, PowerCons, ToeSegmentation1, Wine |

the conservative roster match changes nothing -- dropping it entirely still admits nobody.  Dropping both name exclusions re-admits only names this line already holds locally, which is what makes the exhaustion real rather than an artifact of how the roster was parsed

## Counts

| stage | n |
|---|---|
| pool | 190 |
| binary | 58 |
| eligible | 0 |

Eligible after the full conjunction, in lexicographic order: **none**.

Eligible but not selected (4th onward, recorded so the cut is auditable): none.

## Exclusion lists

- local archive stems (40): BeetleFly, BirdChicken, Coffee, Computers, DistalPhalanxOutlineCorrect, DodgerLoopWeekend, ECG200, ECGFiveDays, Earthquakes, FordA, FordB, FreezerRegularTrain, FreezerSmallTrain, GunPoint, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, Ham, HandOutlines, Herring, HouseTwenty, KeplerLightCurves, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, PhalangesOutlinesCorrect, PowerCons, ProximalPhalanxOutlineCorrect, SemgHandGenderCh2, ShapeletSim, SonyAIBORobotSurface1, SonyAIBORobotSurface2, Strawberry, ToeSegmentation1, ToeSegmentation2, TwoLeadECG, Wafer, Wine, WormsTwoClass, Yoga

- names the download roster mentions: BinaryHeartbeat, CatsDogs, Epilepsy2, ItalyPowerDemand

## Admitted rows

| dataset | train | test | length | classes | channels | total points | template gate |
|---|---|---|---|---|---|---|---|

## Full trajectory (every pool row, with its mechanical reason)

| dataset | admitted | excluded because |
|---|---|---|
| Adiac | False | not_binary:classes=37; total_points_137456_over_100000 |
| ArrowHead | False | not_binary:classes=3; train_rows_36_outside_[40,400] |
| Beef | False | not_binary:classes=5; train_rows_30_outside_[40,400] |
| BeetleFly | False | train_rows_20_outside_[40,400]; name_already_local |
| BirdChicken | False | train_rows_20_outside_[40,400]; name_already_local |
| Car | False | not_binary:classes=4 |
| CBF | False | not_binary:classes=3; train_rows_30_outside_[40,400]; total_points_119040_over_100000 |
| ChlorineConcentration | False | not_binary:classes=3; train_rows_467_outside_[40,400]; total_points_714962_over_100000 |
| CinCECGTorso | False | not_binary:classes=4; total_points_2327380_over_100000 |
| Coffee | False | train_rows_28_outside_[40,400]; name_already_local |
| Computers | False | total_points_360000_over_100000; name_already_local |
| CricketX | False | not_binary:classes=12; total_points_234000_over_100000 |
| CricketY | False | not_binary:classes=12; total_points_234000_over_100000 |
| CricketZ | False | not_binary:classes=12; total_points_234000_over_100000 |
| DiatomSizeReduction | False | not_binary:classes=4; train_rows_16_outside_[40,400]; total_points_111090_over_100000 |
| DistalPhalanxOutlineAgeGroup | False | not_binary:classes=3 |
| DistalPhalanxOutlineCorrect | False | train_rows_600_outside_[40,400]; name_already_local |
| DistalPhalanxTW | False | not_binary:classes=6 |
| Earthquakes | False | total_points_236032_over_100000; name_already_local |
| ECG200 | False | name_already_local |
| ECG5000 | False | not_binary:classes=5; train_rows_500_outside_[40,400]; total_points_700000_over_100000 |
| ECGFiveDays | False | train_rows_23_outside_[40,400]; total_points_120224_over_100000; name_already_local |
| ElectricDevices | False | not_binary:classes=7; train_rows_8926_outside_[40,400]; total_points_1597152_over_100000 |
| FaceAll | False | not_binary:classes=14; train_rows_560_outside_[40,400]; total_points_294750_over_100000 |
| FaceFour | False | not_binary:classes=4; train_rows_24_outside_[40,400] |
| FacesUCR | False | not_binary:classes=14; total_points_294750_over_100000 |
| FiftyWords | False | not_binary:classes=50; train_rows_450_outside_[40,400]; total_points_244350_over_100000 |
| Fish | False | not_binary:classes=7; total_points_162050_over_100000 |
| FordA | False | train_rows_3601_outside_[40,400]; total_points_2460500_over_100000; name_already_local |
| FordB | False | train_rows_3636_outside_[40,400]; total_points_2223000_over_100000; name_already_local |
| GunPoint | False | name_already_local |
| Ham | False | name_already_local |
| HandOutlines | False | train_rows_1000_outside_[40,400]; total_points_3711330_over_100000; name_already_local |
| Haptics | False | not_binary:classes=5; total_points_505596_over_100000 |
| Herring | False | name_already_local |
| InlineSkate | False | not_binary:classes=7; total_points_1223300_over_100000 |
| ItalyPowerDemand | False | template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError; name_already_in_roster |
| LargeKitchenAppliances | False | not_binary:classes=3; total_points_540000_over_100000 |
| Lightning2 | False | name_already_local |
| Lightning7 | False | not_binary:classes=7 |
| Mallat | False | not_binary:classes=8; total_points_2457600_over_100000 |
| Meat | False | not_binary:classes=3 |
| MedicalImages | False | not_binary:classes=10; total_points_112959_over_100000 |
| MiddlePhalanxOutlineAgeGroup | False | not_binary:classes=3 |
| MiddlePhalanxOutlineCorrect | False | train_rows_600_outside_[40,400]; name_already_local |
| MiddlePhalanxTW | False | not_binary:classes=6 |
| MoteStrain | False | train_rows_20_outside_[40,400]; total_points_106848_over_100000; name_already_local |
| NonInvasiveFetalECGThorax1 | False | not_binary:classes=42; train_rows_1800_outside_[40,400]; total_points_2823750_over_100000 |
| NonInvasiveFetalECGThorax2 | False | not_binary:classes=42; train_rows_1800_outside_[40,400]; total_points_2823750_over_100000 |
| OliveOil | False | not_binary:classes=4; train_rows_30_outside_[40,400] |
| OSULeaf | False | not_binary:classes=6; total_points_188734_over_100000 |
| PhalangesOutlinesCorrect | False | train_rows_1800_outside_[40,400]; total_points_212640_over_100000; name_already_local |
| Phoneme | False | not_binary:classes=39; total_points_2160640_over_100000 |
| Plane | False | not_binary:classes=7 |
| ProximalPhalanxOutlineAgeGroup | False | not_binary:classes=3 |
| ProximalPhalanxOutlineCorrect | False | train_rows_600_outside_[40,400]; name_already_local |
| ProximalPhalanxTW | False | not_binary:classes=6 |
| RefrigerationDevices | False | not_binary:classes=3; total_points_540000_over_100000 |
| ScreenType | False | not_binary:classes=3; total_points_540000_over_100000 |
| ShapeletSim | False | train_rows_20_outside_[40,400]; name_already_local |
| ShapesAll | False | not_binary:classes=60; train_rows_600_outside_[40,400]; total_points_614400_over_100000 |
| SmallKitchenAppliances | False | not_binary:classes=3; total_points_540000_over_100000 |
| SonyAIBORobotSurface1 | False | train_rows_20_outside_[40,400]; template_gate:segment_length_0_below_template_minimum; name_already_local |
| SonyAIBORobotSurface2 | False | train_rows_27_outside_[40,400]; template_gate:segment_length_0_below_template_minimum; name_already_local |
| StarLightCurves | False | not_binary:classes=3; train_rows_1000_outside_[40,400]; total_points_9457664_over_100000 |
| Strawberry | False | train_rows_613_outside_[40,400]; total_points_231005_over_100000; name_already_local |
| SwedishLeaf | False | not_binary:classes=15; train_rows_500_outside_[40,400]; total_points_144000_over_100000 |
| Symbols | False | not_binary:classes=6; train_rows_25_outside_[40,400]; total_points_405960_over_100000 |
| SyntheticControl | False | not_binary:classes=6; template_gate:segment_length_0_below_template_minimum |
| ToeSegmentation1 | False | name_already_local |
| ToeSegmentation2 | False | train_rows_36_outside_[40,400]; name_already_local |
| Trace | False | not_binary:classes=4 |
| TwoLeadECG | False | train_rows_23_outside_[40,400]; name_already_local |
| TwoPatterns | False | not_binary:classes=4; train_rows_1000_outside_[40,400]; total_points_640000_over_100000 |
| UWaveGestureLibraryAll | False | not_binary:classes=8; train_rows_896_outside_[40,400]; total_points_4231710_over_100000 |
| UWaveGestureLibraryX | False | not_binary:classes=8; train_rows_896_outside_[40,400]; total_points_1410570_over_100000 |
| UWaveGestureLibraryY | False | not_binary:classes=8; train_rows_896_outside_[40,400]; total_points_1410570_over_100000 |
| UWaveGestureLibraryZ | False | not_binary:classes=8; train_rows_896_outside_[40,400]; total_points_1410570_over_100000 |
| Wafer | False | train_rows_1000_outside_[40,400]; total_points_1088928_over_100000; name_already_local |
| Wine | False | name_already_local |
| WordSynonyms | False | not_binary:classes=25; total_points_244350_over_100000 |
| Worms | False | not_binary:classes=5; total_points_232200_over_100000 |
| WormsTwoClass | False | total_points_232200_over_100000; name_already_local |
| Yoga | False | total_points_1405800_over_100000; name_already_local |
| ACSF1 | False | not_binary:classes=10; total_points_292000_over_100000 |
| AllGestureWiimoteX | False | not_binary:classes=10; variable_length; template_gate:variable_length_no_template_geometry |
| AllGestureWiimoteY | False | not_binary:classes=10; variable_length; template_gate:variable_length_no_template_geometry |
| AllGestureWiimoteZ | False | not_binary:classes=10; variable_length; template_gate:variable_length_no_template_geometry |
| BME | False | not_binary:classes=3; train_rows_30_outside_[40,400] |
| EthanolLevel | False | not_binary:classes=4; train_rows_504_outside_[40,400]; total_points_1758004_over_100000 |
| FreezerRegularTrain | False | total_points_903000_over_100000; name_already_local |
| FreezerSmallTrain | False | train_rows_28_outside_[40,400]; total_points_866278_over_100000; name_already_local |
| GunPointAgeSpan | False | name_already_local |
| GunPointMaleVersusFemale | False | name_already_local |
| GunPointOldVersusYoung | False | name_already_local |
| InsectEPGRegularTrain | False | not_binary:classes=3; total_points_186911_over_100000 |
| InsectEPGSmallTrain | False | not_binary:classes=3; train_rows_17_outside_[40,400]; total_points_159866_over_100000 |
| PickupGestureWiimoteZ | False | not_binary:classes=10; variable_length; template_gate:variable_length_no_template_geometry |
| PigAirwayPressure | False | not_binary:classes=52; total_points_624000_over_100000 |
| PigArtPressure | False | not_binary:classes=52; total_points_624000_over_100000 |
| PigCVP | False | not_binary:classes=52; total_points_624000_over_100000 |
| PLAID | False | not_binary:classes=11; variable_length; train_rows_537_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| PowerCons | False | name_already_local |
| ShakeGestureWiimoteZ | False | not_binary:classes=10; variable_length; template_gate:variable_length_no_template_geometry |
| SmoothSubspace | False | not_binary:classes=3; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| UMD | False | not_binary:classes=3; train_rows_36_outside_[40,400] |
| Fungi | False | not_binary:classes=18; train_rows_18_outside_[40,400] |
| GesturePebbleZ1 | False | not_binary:classes=6; variable_length; template_gate:variable_length_no_template_geometry |
| GesturePebbleZ2 | False | not_binary:classes=6; variable_length; template_gate:variable_length_no_template_geometry |
| HouseTwenty | False | train_rows_34_outside_[40,400]; total_points_405000_over_100000; name_already_local |
| DodgerLoopDay | False | not_binary:classes=7 |
| DodgerLoopWeekend | False | train_rows_20_outside_[40,400]; name_already_local |
| DodgerLoopGame | False | train_rows_20_outside_[40,400] |
| SemgHandGenderCh2 | False | total_points_1350000_over_100000; name_already_local |
| SemgHandMovementCh2 | False | not_binary:classes=6; train_rows_450_outside_[40,400]; total_points_1350000_over_100000 |
| SemgHandSubjectCh2 | False | not_binary:classes=5; train_rows_450_outside_[40,400]; total_points_1350000_over_100000 |
| MixedShapes | False | not_binary:classes=5; train_rows_500_outside_[40,400]; total_points_2995200_over_100000 |
| MixedShapesSmallTrain | False | not_binary:classes=5; total_points_2585600_over_100000 |
| EOGHorizontalSignal | False | not_binary:classes=12; total_points_905000_over_100000 |
| EOGVerticalSignal | False | not_binary:classes=12; total_points_905000_over_100000 |
| GestureMidAirD1 | False | not_binary:classes=26; total_points_121680_over_100000 |
| GestureMidAirD2 | False | not_binary:classes=26; total_points_121680_over_100000 |
| GestureMidAirD3 | False | not_binary:classes=26; total_points_121680_over_100000 |
| Rock | False | not_binary:classes=4; train_rows_20_outside_[40,400]; total_points_199080_over_100000 |
| Crop | False | not_binary:classes=24; train_rows_7200_outside_[40,400]; total_points_1104000_over_100000; template_gate:segment_length_0_below_template_minimum |
| Chinatown | False | train_rows_20_outside_[40,400]; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| MelbournePedestrian | False | not_binary:classes=10; train_rows_1194_outside_[40,400]; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| ArticularyWordRecognition | False | not_binary:classes=25; not_univariate:channels=9 |
| AtrialFibrillation | False | not_binary:classes=3; not_univariate:channels=2; train_rows_15_outside_[40,400] |
| BasicMotions | False | not_binary:classes=4; not_univariate:channels=6 |
| CharacterTrajectories | False | not_binary:classes=20; variable_length; not_univariate:channels=3; train_rows_1422_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| Cricket | False | not_binary:classes=12; not_univariate:channels=6; total_points_215460_over_100000 |
| DuckDuckGeese | False | not_binary:classes=5; not_univariate:channels=1345 |
| EigenWorms | False | not_binary:classes=5; not_univariate:channels=6; total_points_4657856_over_100000 |
| Epilepsy | False | not_binary:classes=4; not_univariate:channels=3 |
| EthanolConcentration | False | not_binary:classes=4; not_univariate:channels=3; total_points_917524_over_100000 |
| ERing | False | not_binary:classes=6; not_univariate:channels=4; train_rows_30_outside_[40,400]; template_gate:segment_length_0_below_template_minimum |
| FaceDetection | False | not_univariate:channels=144; train_rows_5890_outside_[40,400]; total_points_583668_over_100000; template_gate:segment_length_0_below_template_minimum |
| FingerMovements | False | not_univariate:channels=28; template_gate:segment_length_0_below_template_minimum |
| HandMovementDirection | False | not_binary:classes=4; not_univariate:channels=10 |
| Handwriting | False | not_binary:classes=26; not_univariate:channels=3; total_points_152000_over_100000 |
| Heartbeat | False | not_univariate:channels=61; total_points_165645_over_100000 |
| InsectWingbeat | False | not_binary:classes=10; variable_length; not_univariate:channels=200; train_rows_25000_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| JapaneseVowels | False | not_binary:classes=9; not_univariate:channels=12; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| Libras | False | not_binary:classes=15; not_univariate:channels=2; template_gate:segment_length_0_below_template_minimum |
| LSST | False | not_binary:classes=14; not_univariate:channels=6; train_rows_2459_outside_[40,400]; total_points_177300_over_100000; template_gate:segment_length_0_below_template_minimum |
| MotorImagery | False | not_univariate:channels=64; total_points_1134000_over_100000 |
| NATOPS | False | not_binary:classes=6; not_univariate:channels=24; template_gate:segment_length_0_below_template_minimum |
| PenDigits | False | not_binary:classes=10; not_univariate:channels=2; train_rows_7494_outside_[40,400]; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| PEMS-SF | False | not_binary:classes=7; not_univariate:channels=963 |
| PhonemeSpectra | False | not_binary:classes=39; not_univariate:channels=11; train_rows_3315_outside_[40,400]; total_points_1446956_over_100000 |
| RacketSports | False | not_binary:classes=4; not_univariate:channels=6; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| SelfRegulationSCP1 | False | not_univariate:channels=6; total_points_502656_over_100000 |
| SelfRegulationSCP2 | False | not_univariate:channels=7; total_points_437760_over_100000 |
| SpokenArabicDigits | False | not_binary:classes=10; not_univariate:channels=13; train_rows_6599_outside_[40,400]; total_points_818214_over_100000 |
| StandWalkJump | False | not_binary:classes=3; not_univariate:channels=4; train_rows_12_outside_[40,400] |
| UWaveGestureLibrary | False | not_binary:classes=8; not_univariate:channels=3; train_rows_2238_outside_[40,400]; total_points_1410885_over_100000 |
| AsphaltObstacles | False | not_binary:classes=4; variable_length; template_gate:variable_length_no_template_geometry |
| AsphaltPavementType | False | not_binary:classes=3; variable_length; train_rows_1055_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| AsphaltRegularity | False | variable_length; train_rows_751_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| AsphaltObstaclesCoordinates | False | not_binary:classes=4; variable_length; not_univariate:channels=3; template_gate:variable_length_no_template_geometry |
| AsphaltPavementTypeCoordinates | False | not_binary:classes=3; variable_length; not_univariate:channels=3; train_rows_1055_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| AsphaltRegularityCoordinates | False | variable_length; train_rows_751_outside_[40,400]; template_gate:variable_length_no_template_geometry |
| EyesOpenShut | False | not_univariate:channels=14 |
| Colposcopy | False | not_binary:classes=6 |
| CounterMovementJump | False | not_binary:classes=3; not_univariate:channels=3; train_rows_419_outside_[40,400]; total_points_2541500_over_100000 |
| Tiselac | False | not_binary:classes=9; not_univariate:channels=10; train_rows_81714_outside_[40,400]; total_points_2292801_over_100000; template_gate:segment_length_0_below_template_minimum; template_gate:bound_positions_rejected:ValueError |
| RightWhaleCalls | False | train_rows_10934_outside_[40,400]; total_points_51584000_over_100000 |
| SharePriceIncrease | False | train_rows_965_outside_[40,400]; total_points_115800_over_100000; template_gate:segment_length_0_below_template_minimum |
| CatsDogs | False | total_points_4062575_over_100000; name_already_in_roster |
| BinaryHeartbeat | False | total_points_7578770_over_100000; name_already_in_roster |
| DucksAndGeese | False | not_binary:classes=5; total_points_23678400_over_100000 |
| UrbanSound | False | not_binary:classes=10; train_rows_2713_outside_[40,400]; total_points_239242500_over_100000 |
| FruitFlies | False | not_binary:classes=3; train_rows_17259_outside_[40,400]; total_points_172590000_over_100000 |
| InsectSound | False | not_binary:classes=10; train_rows_25000_outside_[40,400]; total_points_30000000_over_100000 |
| MosquitoSound | False | not_binary:classes=6; train_rows_139883_outside_[40,400]; total_points_1049122500_over_100000 |
| AbnormalHeartbeat | False | not_binary:classes=5; total_points_1850118_over_100000 |
| ElectricDeviceDetection | False | train_rows_623_outside_[40,400]; total_points_1123840_over_100000 |
| MindReading | False | not_binary:classes=5; not_univariate:channels=204; train_rows_727_outside_[40,400]; total_points_276000_over_100000 |
| MotionSenseHAR | False | not_binary:classes=6; not_univariate:channels=12 |
| EMOPain | False | not_binary:classes=3; not_univariate:channels=30; train_rows_1093_outside_[40,400]; total_points_205740_over_100000 |
| Blink | False | not_univariate:channels=4; train_rows_500_outside_[40,400]; total_points_484500_over_100000 |
| KeplerLightCurves | False | not_binary:classes=7; train_rows_920_outside_[40,400]; total_points_6287673_over_100000; name_already_local |
| WalkingSittingStanding | False | not_binary:classes=6; not_univariate:channels=3; train_rows_7352_outside_[40,400]; total_points_2121594_over_100000 |
| Sleep | False | not_binary:classes=5; train_rows_478785_outside_[40,400]; total_points_101299800_over_100000 |
| FaultDetectionA | False | not_binary:classes=3; train_rows_10912_outside_[40,400]; total_points_69836800_over_100000 |
| FaultDetectionB | False | not_binary:classes=3; total_points_69836800_over_100000 |
| NerveDamage | False | not_binary:classes=3; total_points_306000_over_100000 |
| CardiacArrhythmia | False | not_binary:classes=3; train_rows_43673_outside_[40,400]; total_points_68365500_over_100000 |
| Epilepsy2 | False | total_points_2047000_over_100000; name_already_in_roster |

## Obligations

- **downloads_so_far**: 0
- **no_value_or_family_based_preselection**: every clause is a public metadata field or a mechanical consequence of the frozen template; nothing about the expected defect family entered the filter
- **zero_llm**: True
- **zero_values_or_labels_read**: only the published metadata table was read; no dataset zip has been fetched at the time this artifact is written
