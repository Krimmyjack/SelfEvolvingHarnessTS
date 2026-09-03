# S2a #31 natural-pool sweep

**branch: reduced (zero remounts, zero four-conjunction hits)**

Pool = 12 #31 episodes (traffic 8 + noaa 4). Old #31 numbers locate the pool only.
Capacity gate: TRAIN>=40, half>=20. All 12 fail (traffic 12+8, NOAA 12+4). Structural exclusion, not selection.
Hidden-harm series (traffic 14/16/17; NOAA 99999904140 / 99999923908 / 99999963862) sit on the original eval face and cannot enter a Support/delayed pool of a TRAIN>=40 remount without adding series.
Fits: 0. Four-conjunction table: empty (no remounted cell to score).

## Structural exclusions

- `traffic/pooled/hampel_filter`: n_train=12 n_eval=8 hidden=14,16,17 reasons=n_train=12 < 40; n_half=6 < 20; hidden-harm series are on the original eval face; cannot place them in Support/delayed of a TRAIN>=40 remount without adding series outside this pool member
- `traffic/pooled/outlier_iqr`: n_train=12 n_eval=8 hidden=14 reasons=n_train=12 < 40; n_half=6 < 20; hidden-harm series are on the original eval face; cannot place them in Support/delayed of a TRAIN>=40 remount without adding series outside this pool member
- `traffic/pooled/outlier_mad`: n_train=12 n_eval=8 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `traffic/pooled/winsorize`: n_train=12 n_eval=8 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `traffic/per_channel/hampel_filter`: n_train=12 n_eval=8 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `traffic/per_channel/outlier_iqr`: n_train=12 n_eval=8 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `traffic/per_channel/outlier_mad`: n_train=12 n_eval=8 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `traffic/per_channel/winsorize`: n_train=12 n_eval=8 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `noaa/task_A/outlier_mad`: n_train=12 n_eval=4 hidden=none reasons=n_train=12 < 40; n_half=6 < 20
- `noaa/task_C/outlier_mad`: n_train=12 n_eval=4 hidden=99999904140 reasons=n_train=12 < 40; n_half=6 < 20; hidden-harm series are on the original eval face; cannot place them in Support/delayed of a TRAIN>=40 remount without adding series outside this pool member
- `noaa/task_A/outlier_iqr`: n_train=12 n_eval=4 hidden=99999923908 reasons=n_train=12 < 40; n_half=6 < 20; hidden-harm series are on the original eval face; cannot place them in Support/delayed of a TRAIN>=40 remount without adding series outside this pool member
- `noaa/task_D/outlier_mad`: n_train=12 n_eval=4 hidden=99999904140,99999963862 reasons=n_train=12 < 40; n_half=6 < 20; hidden-harm series are on the original eval face; cannot place them in Support/delayed of a TRAIN>=40 remount without adding series outside this pool member

## Reduced course

R2 forecast 未考(冲突场在注入与 #31 自然池下均不可得)

- producer: `electricity_impulsive_outlier_03` (injected)
- clean_identity: `traffic_clean_identity_00` (clean)
- boundary_compile: `electricity_impulsive_outlier_03` (injected)
- strong_beneficiary_1: `electricity_impulsive_outlier_01` (injected)
- strong_beneficiary_2: `electricity_impulsive_outlier_04` (injected)
- gap_out_of_family_guard: `traffic_gap_00` (injected)
