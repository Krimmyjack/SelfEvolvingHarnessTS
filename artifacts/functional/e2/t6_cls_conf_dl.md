# CLS-CONF-dl -- filter trajectory audit gate (before any dataset zip)

protocol: 	6_cls_conf_downloaded_target_v1  evidence grade: **DEVELOPMENT**

## Verdict

**AUDIT_GATE_PASSED** — filter trajectory written; dataset zips not yet downloaded; arms not run.

pool = official UCR univariate archive as published on timeseriesclassification.com (aeon-toolkit/metadata.csv); keep rows with NumberClasses=2 AND Length!=0 (equal length; site lists variable length as 0) AND Channels=1 (univariate) AND official TrainSize in [40,400] AND name not among mechanically enumerated zip stems in local data/ucr_task_context; take lexicographic first 3 as D1/D2/D3. No extra archive-generation filter. Outcome-blind. Written before any dataset zip download.

## Local zip enumeration (mechanical)

- directory: data/ucr_task_context
- n_zips: **40**
- stems: BeetleFly, BirdChicken, Coffee, Computers, DistalPhalanxOutlineCorrect, DodgerLoopWeekend, ECG200, ECGFiveDays, Earthquakes, FordA, FordB, FreezerRegularTrain, FreezerSmallTrain, GunPoint, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, Ham, HandOutlines, Herring, HouseTwenty, KeplerLightCurves, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, PhalangesOutlinesCorrect, PowerCons, ProximalPhalanxOutlineCorrect, SemgHandGenderCh2, ShapeletSim, SonyAIBORobotSurface1, SonyAIBORobotSurface2, Strawberry, ToeSegmentation1, ToeSegmentation2, TwoLeadECG, Wafer, Wine, WormsTwoClass, Yoga

## Official metadata table

- URL: https://timeseriesclassification.com/aeon-toolkit/metadata.csv
- fetched_utc: 2026-08-25T13:20:15Z
- rows: **190**

## Filter steps

### Step 1 — Channels == 1 (univariate)

- pass 150: Adiac, ArrowHead, Beef, BeetleFly, BirdChicken, Car, CBF, ChlorineConcentration, CinCECGTorso, Coffee, Computers, CricketX, CricketY, CricketZ, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, Earthquakes, ECG200, ECG5000, ECGFiveDays, ElectricDevices, FaceAll, FaceFour, FacesUCR, FiftyWords, Fish, FordA, FordB, GunPoint, Ham, HandOutlines, Haptics, Herring, InlineSkate, ItalyPowerDemand, LargeKitchenAppliances, Lightning2, Lightning7, Mallat, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, OliveOil, OSULeaf, PhalangesOutlinesCorrect, Phoneme, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, SonyAIBORobotSurface1, SonyAIBORobotSurface2, StarLightCurves, Strawberry, SwedishLeaf, Symbols, SyntheticControl, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, TwoPatterns, UWaveGestureLibraryAll, UWaveGestureLibraryX, UWaveGestureLibraryY, UWaveGestureLibraryZ, Wafer, Wine, WordSynonyms, Worms, WormsTwoClass, Yoga, ACSF1, AllGestureWiimoteX, AllGestureWiimoteY, AllGestureWiimoteZ, BME, EthanolLevel, FreezerRegularTrain, FreezerSmallTrain, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, InsectEPGRegularTrain, InsectEPGSmallTrain, PickupGestureWiimoteZ, PigAirwayPressure, PigArtPressure, PigCVP, PLAID, PowerCons, ShakeGestureWiimoteZ, SmoothSubspace, UMD, Fungi, GesturePebbleZ1, GesturePebbleZ2, HouseTwenty, DodgerLoopDay, DodgerLoopWeekend, DodgerLoopGame, SemgHandGenderCh2, SemgHandMovementCh2, SemgHandSubjectCh2, MixedShapes, MixedShapesSmallTrain, EOGHorizontalSignal, EOGVerticalSignal, GestureMidAirD1, GestureMidAirD2, GestureMidAirD3, Rock, Crop, Chinatown, MelbournePedestrian, AsphaltObstacles, AsphaltPavementType, AsphaltRegularity, AsphaltRegularityCoordinates, Colposcopy, RightWhaleCalls, SharePriceIncrease, CatsDogs, BinaryHeartbeat, DucksAndGeese, UrbanSound, FruitFlies, InsectSound, MosquitoSound, AbnormalHeartbeat, ElectricDeviceDetection, KeplerLightCurves, Sleep, FaultDetectionA, FaultDetectionB, NerveDamage, CardiacArrhythmia, Epilepsy2
- fail 40: ArticularyWordRecognition, AtrialFibrillation, BasicMotions, CharacterTrajectories, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, EthanolConcentration, ERing, FaceDetection, FingerMovements, HandMovementDirection, Handwriting, Heartbeat, InsectWingbeat, JapaneseVowels, Libras, LSST, MotorImagery, NATOPS, PenDigits, PEMS-SF, PhonemeSpectra, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, StandWalkJump, UWaveGestureLibrary, AsphaltObstaclesCoordinates, AsphaltPavementTypeCoordinates, EyesOpenShut, CounterMovementJump, Tiselac, MindReading, MotionSenseHAR, EMOPain, Blink, WalkingSittingStanding

### Step 2 — NumberClasses == 2

- pass 58: BeetleFly, BirdChicken, Coffee, Computers, DistalPhalanxOutlineCorrect, Earthquakes, ECG200, ECGFiveDays, FordA, FordB, GunPoint, Ham, HandOutlines, Herring, ItalyPowerDemand, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, PhalangesOutlinesCorrect, ProximalPhalanxOutlineCorrect, ShapeletSim, SonyAIBORobotSurface1, SonyAIBORobotSurface2, Strawberry, ToeSegmentation1, ToeSegmentation2, TwoLeadECG, Wafer, Wine, WormsTwoClass, Yoga, FreezerRegularTrain, FreezerSmallTrain, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, PowerCons, HouseTwenty, DodgerLoopWeekend, DodgerLoopGame, SemgHandGenderCh2, Chinatown, FaceDetection, FingerMovements, Heartbeat, MotorImagery, SelfRegulationSCP1, SelfRegulationSCP2, AsphaltRegularity, AsphaltRegularityCoordinates, EyesOpenShut, RightWhaleCalls, SharePriceIncrease, CatsDogs, BinaryHeartbeat, ElectricDeviceDetection, Blink, Epilepsy2
- fail 132: Adiac, ArrowHead, Beef, Car, CBF, ChlorineConcentration, CinCECGTorso, CricketX, CricketY, CricketZ, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistalPhalanxTW, ECG5000, ElectricDevices, FaceAll, FaceFour, FacesUCR, FiftyWords, Fish, Haptics, InlineSkate, LargeKitchenAppliances, Lightning7, Mallat, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxTW, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, OliveOil, OSULeaf, Phoneme, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapesAll, SmallKitchenAppliances, StarLightCurves, SwedishLeaf, Symbols, SyntheticControl, Trace, TwoPatterns, UWaveGestureLibraryAll, UWaveGestureLibraryX, UWaveGestureLibraryY, UWaveGestureLibraryZ, WordSynonyms, Worms, ACSF1, AllGestureWiimoteX, AllGestureWiimoteY, AllGestureWiimoteZ, BME, EthanolLevel, InsectEPGRegularTrain, InsectEPGSmallTrain, PickupGestureWiimoteZ, PigAirwayPressure, PigArtPressure, PigCVP, PLAID, ShakeGestureWiimoteZ, SmoothSubspace, UMD, Fungi, GesturePebbleZ1, GesturePebbleZ2, DodgerLoopDay, SemgHandMovementCh2, SemgHandSubjectCh2, MixedShapes, MixedShapesSmallTrain, EOGHorizontalSignal, EOGVerticalSignal, GestureMidAirD1, GestureMidAirD2, GestureMidAirD3, Rock, Crop, MelbournePedestrian, ArticularyWordRecognition, AtrialFibrillation, BasicMotions, CharacterTrajectories, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, EthanolConcentration, ERing, HandMovementDirection, Handwriting, InsectWingbeat, JapaneseVowels, Libras, LSST, NATOPS, PenDigits, PEMS-SF, PhonemeSpectra, RacketSports, SpokenArabicDigits, StandWalkJump, UWaveGestureLibrary, AsphaltObstacles, AsphaltPavementType, AsphaltObstaclesCoordinates, AsphaltPavementTypeCoordinates, Colposcopy, CounterMovementJump, Tiselac, DucksAndGeese, UrbanSound, FruitFlies, InsectSound, MosquitoSound, AbnormalHeartbeat, MindReading, MotionSenseHAR, EMOPain, KeplerLightCurves, WalkingSittingStanding, Sleep, FaultDetectionA, FaultDetectionB, NerveDamage, CardiacArrhythmia

### Step 3 — Length != 0 (equal length; TSC lists variable as 0)

- pass 174: Adiac, ArrowHead, Beef, BeetleFly, BirdChicken, Car, CBF, ChlorineConcentration, CinCECGTorso, Coffee, Computers, CricketX, CricketY, CricketZ, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistalPhalanxOutlineCorrect, DistalPhalanxTW, Earthquakes, ECG200, ECG5000, ECGFiveDays, ElectricDevices, FaceAll, FaceFour, FacesUCR, FiftyWords, Fish, FordA, FordB, GunPoint, Ham, HandOutlines, Haptics, Herring, InlineSkate, ItalyPowerDemand, LargeKitchenAppliances, Lightning2, Lightning7, Mallat, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, MiddlePhalanxTW, MoteStrain, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, OliveOil, OSULeaf, PhalangesOutlinesCorrect, Phoneme, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapeletSim, ShapesAll, SmallKitchenAppliances, SonyAIBORobotSurface1, SonyAIBORobotSurface2, StarLightCurves, Strawberry, SwedishLeaf, Symbols, SyntheticControl, ToeSegmentation1, ToeSegmentation2, Trace, TwoLeadECG, TwoPatterns, UWaveGestureLibraryAll, UWaveGestureLibraryX, UWaveGestureLibraryY, UWaveGestureLibraryZ, Wafer, Wine, WordSynonyms, Worms, WormsTwoClass, Yoga, ACSF1, BME, EthanolLevel, FreezerRegularTrain, FreezerSmallTrain, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, InsectEPGRegularTrain, InsectEPGSmallTrain, PigAirwayPressure, PigArtPressure, PigCVP, PowerCons, SmoothSubspace, UMD, Fungi, HouseTwenty, DodgerLoopDay, DodgerLoopWeekend, DodgerLoopGame, SemgHandGenderCh2, SemgHandMovementCh2, SemgHandSubjectCh2, MixedShapes, MixedShapesSmallTrain, EOGHorizontalSignal, EOGVerticalSignal, GestureMidAirD1, GestureMidAirD2, GestureMidAirD3, Rock, Crop, Chinatown, MelbournePedestrian, ArticularyWordRecognition, AtrialFibrillation, BasicMotions, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, EthanolConcentration, ERing, FaceDetection, FingerMovements, HandMovementDirection, Handwriting, Heartbeat, JapaneseVowels, Libras, LSST, MotorImagery, NATOPS, PenDigits, PEMS-SF, PhonemeSpectra, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, StandWalkJump, UWaveGestureLibrary, EyesOpenShut, Colposcopy, CounterMovementJump, Tiselac, RightWhaleCalls, SharePriceIncrease, CatsDogs, BinaryHeartbeat, DucksAndGeese, UrbanSound, FruitFlies, InsectSound, MosquitoSound, AbnormalHeartbeat, ElectricDeviceDetection, MindReading, MotionSenseHAR, EMOPain, Blink, KeplerLightCurves, WalkingSittingStanding, Sleep, FaultDetectionA, FaultDetectionB, NerveDamage, CardiacArrhythmia, Epilepsy2
- fail 16: AllGestureWiimoteX, AllGestureWiimoteY, AllGestureWiimoteZ, PickupGestureWiimoteZ, PLAID, ShakeGestureWiimoteZ, GesturePebbleZ1, GesturePebbleZ2, CharacterTrajectories, InsectWingbeat, AsphaltObstacles, AsphaltPavementType, AsphaltRegularity, AsphaltObstaclesCoordinates, AsphaltPavementTypeCoordinates, AsphaltRegularityCoordinates

### Step 4 — TrainSize in [40, 400]

- pass 100: Adiac, Car, CinCECGTorso, Computers, CricketX, CricketY, CricketZ, DistalPhalanxOutlineAgeGroup, DistalPhalanxTW, Earthquakes, ECG200, FacesUCR, Fish, GunPoint, Ham, Haptics, Herring, InlineSkate, ItalyPowerDemand, LargeKitchenAppliances, Lightning2, Lightning7, Mallat, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxTW, OSULeaf, Phoneme, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW, RefrigerationDevices, ScreenType, SmallKitchenAppliances, SyntheticControl, ToeSegmentation1, Trace, Wine, WordSynonyms, Worms, WormsTwoClass, Yoga, ACSF1, AllGestureWiimoteX, AllGestureWiimoteY, AllGestureWiimoteZ, FreezerRegularTrain, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, InsectEPGRegularTrain, PickupGestureWiimoteZ, PigAirwayPressure, PigArtPressure, PigCVP, PowerCons, ShakeGestureWiimoteZ, SmoothSubspace, GesturePebbleZ1, GesturePebbleZ2, DodgerLoopDay, SemgHandGenderCh2, MixedShapesSmallTrain, EOGHorizontalSignal, EOGVerticalSignal, GestureMidAirD1, GestureMidAirD2, GestureMidAirD3, ArticularyWordRecognition, BasicMotions, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, EthanolConcentration, FingerMovements, HandMovementDirection, Handwriting, Heartbeat, JapaneseVowels, Libras, MotorImagery, NATOPS, PEMS-SF, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, AsphaltObstacles, AsphaltObstaclesCoordinates, EyesOpenShut, Colposcopy, CatsDogs, BinaryHeartbeat, DucksAndGeese, AbnormalHeartbeat, MotionSenseHAR, FaultDetectionB, NerveDamage, Epilepsy2
- fail 90: ArrowHead, Beef, BeetleFly, BirdChicken, CBF, ChlorineConcentration, Coffee, DiatomSizeReduction, DistalPhalanxOutlineCorrect, ECG5000, ECGFiveDays, ElectricDevices, FaceAll, FaceFour, FiftyWords, FordA, FordB, HandOutlines, MiddlePhalanxOutlineCorrect, MoteStrain, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, OliveOil, PhalangesOutlinesCorrect, ProximalPhalanxOutlineCorrect, ShapeletSim, ShapesAll, SonyAIBORobotSurface1, SonyAIBORobotSurface2, StarLightCurves, Strawberry, SwedishLeaf, Symbols, ToeSegmentation2, TwoLeadECG, TwoPatterns, UWaveGestureLibraryAll, UWaveGestureLibraryX, UWaveGestureLibraryY, UWaveGestureLibraryZ, Wafer, BME, EthanolLevel, FreezerSmallTrain, InsectEPGSmallTrain, PLAID, UMD, Fungi, HouseTwenty, DodgerLoopWeekend, DodgerLoopGame, SemgHandMovementCh2, SemgHandSubjectCh2, MixedShapes, Rock, Crop, Chinatown, MelbournePedestrian, AtrialFibrillation, CharacterTrajectories, ERing, FaceDetection, InsectWingbeat, LSST, PenDigits, PhonemeSpectra, SpokenArabicDigits, StandWalkJump, UWaveGestureLibrary, AsphaltPavementType, AsphaltRegularity, AsphaltPavementTypeCoordinates, AsphaltRegularityCoordinates, CounterMovementJump, Tiselac, RightWhaleCalls, SharePriceIncrease, UrbanSound, FruitFlies, InsectSound, MosquitoSound, ElectricDeviceDetection, MindReading, EMOPain, Blink, KeplerLightCurves, WalkingSittingStanding, Sleep, FaultDetectionA, CardiacArrhythmia

### Step 5 — Dataset name not in local data/ucr_task_context zip stems

- pass 150: Adiac, ArrowHead, Beef, Car, CBF, ChlorineConcentration, CinCECGTorso, CricketX, CricketY, CricketZ, DiatomSizeReduction, DistalPhalanxOutlineAgeGroup, DistalPhalanxTW, ECG5000, ElectricDevices, FaceAll, FaceFour, FacesUCR, FiftyWords, Fish, Haptics, InlineSkate, ItalyPowerDemand, LargeKitchenAppliances, Lightning7, Mallat, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxTW, NonInvasiveFetalECGThorax1, NonInvasiveFetalECGThorax2, OliveOil, OSULeaf, Phoneme, Plane, ProximalPhalanxOutlineAgeGroup, ProximalPhalanxTW, RefrigerationDevices, ScreenType, ShapesAll, SmallKitchenAppliances, StarLightCurves, SwedishLeaf, Symbols, SyntheticControl, Trace, TwoPatterns, UWaveGestureLibraryAll, UWaveGestureLibraryX, UWaveGestureLibraryY, UWaveGestureLibraryZ, WordSynonyms, Worms, ACSF1, AllGestureWiimoteX, AllGestureWiimoteY, AllGestureWiimoteZ, BME, EthanolLevel, InsectEPGRegularTrain, InsectEPGSmallTrain, PickupGestureWiimoteZ, PigAirwayPressure, PigArtPressure, PigCVP, PLAID, ShakeGestureWiimoteZ, SmoothSubspace, UMD, Fungi, GesturePebbleZ1, GesturePebbleZ2, DodgerLoopDay, DodgerLoopGame, SemgHandMovementCh2, SemgHandSubjectCh2, MixedShapes, MixedShapesSmallTrain, EOGHorizontalSignal, EOGVerticalSignal, GestureMidAirD1, GestureMidAirD2, GestureMidAirD3, Rock, Crop, Chinatown, MelbournePedestrian, ArticularyWordRecognition, AtrialFibrillation, BasicMotions, CharacterTrajectories, Cricket, DuckDuckGeese, EigenWorms, Epilepsy, EthanolConcentration, ERing, FaceDetection, FingerMovements, HandMovementDirection, Handwriting, Heartbeat, InsectWingbeat, JapaneseVowels, Libras, LSST, MotorImagery, NATOPS, PenDigits, PEMS-SF, PhonemeSpectra, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, StandWalkJump, UWaveGestureLibrary, AsphaltObstacles, AsphaltPavementType, AsphaltRegularity, AsphaltObstaclesCoordinates, AsphaltPavementTypeCoordinates, AsphaltRegularityCoordinates, EyesOpenShut, Colposcopy, CounterMovementJump, Tiselac, RightWhaleCalls, SharePriceIncrease, CatsDogs, BinaryHeartbeat, DucksAndGeese, UrbanSound, FruitFlies, InsectSound, MosquitoSound, AbnormalHeartbeat, ElectricDeviceDetection, MindReading, MotionSenseHAR, EMOPain, Blink, WalkingSittingStanding, Sleep, FaultDetectionA, FaultDetectionB, NerveDamage, CardiacArrhythmia, Epilepsy2
- fail 40: BeetleFly, BirdChicken, Coffee, Computers, DistalPhalanxOutlineCorrect, Earthquakes, ECG200, ECGFiveDays, FordA, FordB, GunPoint, Ham, HandOutlines, Herring, Lightning2, MiddlePhalanxOutlineCorrect, MoteStrain, PhalangesOutlinesCorrect, ProximalPhalanxOutlineCorrect, ShapeletSim, SonyAIBORobotSurface1, SonyAIBORobotSurface2, Strawberry, ToeSegmentation1, ToeSegmentation2, TwoLeadECG, Wafer, Wine, WormsTwoClass, Yoga, FreezerRegularTrain, FreezerSmallTrain, GunPointAgeSpan, GunPointMaleVersusFemale, GunPointOldVersusYoung, PowerCons, HouseTwenty, DodgerLoopWeekend, SemgHandGenderCh2, KeplerLightCurves

## Eligible after conjunction (lexicographic)

- eligible (4): BinaryHeartbeat, CatsDogs, Epilepsy2, ItalyPowerDemand

## D1 / D2 / D3 roles

| role | dataset | TRAIN | TEST | length | classes | channels | sealed |
|---|---|---|---|---|---|---|---|
| D1 | **BinaryHeartbeat** | 204 | 205 | 18530 | 2 | 1 | False |
| D2_sealed | **CatsDogs** | 138 | 137 | 14773 | 2 | 1 | True |
| D3_reserve | **Epilepsy2** | 80 | 11420 | 178 | 2 | 1 | True |

## All binary-class rows

| dataset | TRAIN | length | classes | channels | excluded because |
|---|---|---|---|---|---|
| BeetleFly | 20 | 512 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| BirdChicken | 20 | 512 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Coffee | 28 | 286 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Computers | 250 | 720 | 2 | 1 | name_in_local_ucr_task_context_zip |
| DistalPhalanxOutlineCorrect | 600 | 80 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Earthquakes | 322 | 512 | 2 | 1 | name_in_local_ucr_task_context_zip |
| ECG200 | 100 | 96 | 2 | 1 | name_in_local_ucr_task_context_zip |
| ECGFiveDays | 23 | 136 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| FordA | 3601 | 500 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| FordB | 3636 | 500 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| GunPoint | 50 | 150 | 2 | 1 | name_in_local_ucr_task_context_zip |
| Ham | 109 | 431 | 2 | 1 | name_in_local_ucr_task_context_zip |
| HandOutlines | 1000 | 2709 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Herring | 64 | 512 | 2 | 1 | name_in_local_ucr_task_context_zip |
| ItalyPowerDemand | 67 | 24 | 2 | 1 | **ELIGIBLE** |
| Lightning2 | 60 | 637 | 2 | 1 | name_in_local_ucr_task_context_zip |
| MiddlePhalanxOutlineCorrect | 600 | 80 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| MoteStrain | 20 | 84 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| PhalangesOutlinesCorrect | 1800 | 80 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| ProximalPhalanxOutlineCorrect | 600 | 80 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| ShapeletSim | 20 | 500 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| SonyAIBORobotSurface1 | 20 | 70 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| SonyAIBORobotSurface2 | 27 | 65 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Strawberry | 613 | 235 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| ToeSegmentation1 | 40 | 277 | 2 | 1 | name_in_local_ucr_task_context_zip |
| ToeSegmentation2 | 36 | 343 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| TwoLeadECG | 23 | 82 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Wafer | 1000 | 152 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| Wine | 57 | 234 | 2 | 1 | name_in_local_ucr_task_context_zip |
| WormsTwoClass | 181 | 900 | 2 | 1 | name_in_local_ucr_task_context_zip |
| Yoga | 300 | 426 | 2 | 1 | name_in_local_ucr_task_context_zip |
| FreezerRegularTrain | 150 | 301 | 2 | 1 | name_in_local_ucr_task_context_zip |
| FreezerSmallTrain | 28 | 301 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| GunPointAgeSpan | 135 | 150 | 2 | 1 | name_in_local_ucr_task_context_zip |
| GunPointMaleVersusFemale | 135 | 150 | 2 | 1 | name_in_local_ucr_task_context_zip |
| GunPointOldVersusYoung | 135 | 150 | 2 | 1 | name_in_local_ucr_task_context_zip |
| PowerCons | 180 | 144 | 2 | 1 | name_in_local_ucr_task_context_zip |
| HouseTwenty | 34 | 3000 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| DodgerLoopWeekend | 20 | 288 | 2 | 1 | train_rows_outside_40_400, name_in_local_ucr_task_context_zip |
| DodgerLoopGame | 20 | 288 | 2 | 1 | train_rows_outside_40_400 |
| SemgHandGenderCh2 | 300 | 1500 | 2 | 1 | name_in_local_ucr_task_context_zip |
| Chinatown | 20 | 24 | 2 | 1 | train_rows_outside_40_400 |
| FaceDetection | 5890 | 62 | 2 | 144 | not_univariate_channels_144, train_rows_outside_40_400 |
| FingerMovements | 316 | 50 | 2 | 28 | not_univariate_channels_28 |
| Heartbeat | 204 | 405 | 2 | 61 | not_univariate_channels_61 |
| MotorImagery | 278 | 3000 | 2 | 64 | not_univariate_channels_64 |
| SelfRegulationSCP1 | 268 | 896 | 2 | 6 | not_univariate_channels_6 |
| SelfRegulationSCP2 | 200 | 1152 | 2 | 7 | not_univariate_channels_7 |
| AsphaltRegularity | 751 | 0 | 2 | 1 | unequal_length_official_length_0, train_rows_outside_40_400 |
| AsphaltRegularityCoordinates | 751 | 0 | 2 | 1 | unequal_length_official_length_0, train_rows_outside_40_400 |
| EyesOpenShut | 56 | 128 | 2 | 14 | not_univariate_channels_14 |
| RightWhaleCalls | 10934 | 4000 | 2 | 1 | train_rows_outside_40_400 |
| SharePriceIncrease | 965 | 60 | 2 | 1 | train_rows_outside_40_400 |
| CatsDogs | 138 | 14773 | 2 | 1 | **ELIGIBLE** |
| BinaryHeartbeat | 204 | 18530 | 2 | 1 | **ELIGIBLE** |
| ElectricDeviceDetection | 623 | 256 | 2 | 1 | train_rows_outside_40_400 |
| Blink | 500 | 510 | 2 | 4 | not_univariate_channels_4, train_rows_outside_40_400 |
| Epilepsy2 | 80 | 178 | 2 | 1 | **ELIGIBLE** |

Downloads of dataset zips have **not** started.

## RUN TERMINATED

**Verdict (primary): `COMPUTE_BUDGET_EXCEEDED`.** Secondary: `INSTRUMENT_SCALE_MISMATCH`.

This is **not a scientific negative**. CLS-CONF remains **OPEN**. The A3 r1 print `winner=None delayed=None` is a local observation only and **must not be cited as non-replication evidence**.

### Timeline

| event | local time (UTC+8) |
|---|---|
| `--conf-dl-run` started | 2026-08-25 21:36 |
| A3 arm started | 2026-08-25 22:02 |
| mainline terminated the process | 2026-08-26 09:55:06 |
| wall clock | 12.3 h |
| CPU (approx.) | ~11 h |

Process was PID 15748 / terminal 931012. Termination followed the user + sol wall-clock ruling. A3 completed only r1 (`A3 BinaryHeartbeat r1 probes=1 winner=None delayed=None`). The final two-arm payload was not written.

### Cause

BinaryHeartbeat is ~3.78 million total points (`(204+205)×18530`). The shared CLS-OP pipeline has hot paths that grow with point count. The frozen selection rule had no compute-scale gate, so an eligible-but-infeasible D1 entered the two-arm run.

### Retained artifacts (wrap-up only; no rerun)

- Part A selection / filter-trajectory census in this file and `t6_cls_conf_dl.json` (unchanged; still a valid audit)
- `data/ucr_conf_downloaded/ROSTER.md`
- terminal trajectory of `--conf-dl-run`

