# R4C non-author check (Grok, 2026-09-08)

Role: read-only independent recompute from `_scratch/m_r0k_prediction_store.json`.
Did not run `audit_r4c_imputation_donor_conditions.run()`, did not load the with-missing variant, did not read horizon truth, 0 Consumer fits, 0 LLM.
Did import `evaluation.main_protocol_p4.audit_r4c_imputation_donor_conditions` helpers plus `SelfEvolvingHarnessTS.operators.s1_impute` / `s1_outlier` / `_common` for item 4 only (three synthetic arrays; no series I/O).
Labels (taskbook + script): `g = raw_per_view - program_per_view`; `d = g(W2) - g(ANCESTOR)`; pair key `(position, face, uid)`; `d_positive = d>0`; `d_severe_harm = -d > 0.30`; exact-zero `|d|<1e-9`.
Rounding: six decimals, `round(x, 6)`. Match if `|a-b| < 5e-7` after that rounding.

**Grid tally (recompute vs artifact / report): 25 / 27 match. Mismatches: 2.**

---

## 1. Pairing

| item | recompute | artifact | match |
| --- | --- | --- | --- |
| paired rows | 840 | 840 | Y |
| 21 pos × 2 faces × 20 uid | 21 × 2 × 20 = 840 | 840 | Y |
| `eval_uids` identical per (position, face) | True (42/42) | claimed identical | Y |
| `raw_per_view` identical (atol 1e-12) | True (42/42) | claimed identical | Y |
| skipped verifier / degenerate / unpaired | 0 / 0 / 0 | 0 / 0 / 0 | Y |

Store entries: W2 42, ANCESTOR 42. No unpaired face.

---

## 2. d statistics (840 rows; faces; §3.1 weighted merge)

Independent overall / per-face from the store. Artifact §3.1 only prints the two serving-action groups; the check is that their n-weighted merge equals the independent overall.

| item | recompute | artifact / merge | match |
| --- | ---: | ---: | --- |
| n | 840 | 180+660=840 | Y |
| mean d | -0.006376 | merge -0.006376 | Y |
| d>0 rate | 0.482143 | merge 0.482143 | Y |
| −d>0.30 rate | 0.204762 | merge 0.204762 | Y |
| \|d\|<1e-9 rows | 0 | 0 (md/JSON) | Y |

§3.1 groups (md = JSON at 6 d.p.; not independently re-derived — that needs serving-window flags / raw series, not done):

| group | n | mean d | mean \|d\| | d>0 | −d>0.30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| serving_untouched | 180 | 0.004815 | 0.280934 | 0.527778 | 0.161111 |
| serving_period_filled | 660 | -0.009428 | 0.365770 | 0.469697 | 0.216667 |

Per-face (independent; no §3.1 counterpart):

| face | n | mean d | mean \|d\| | d>0 | −d>0.30 | \|d\|<1e-9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| support_face | 420 | 0.026891 | 0.366139 | 0.509524 | 0.192857 | 0 |
| delayed_face | 420 | -0.039643 | 0.329042 | 0.454762 | 0.216667 | 0 |

---

## 3. always-W2 minus always-ANCESTOR

`U = mean(g)`; `U(W2)-U(ANC) = mean(d)` on the same rows. Folds from artifact `position_block` (also the per-unit table): n = 280 / 360 / 80 / 120.

| item | recompute | artifact | match |
| --- | ---: | ---: | --- |
| pooled | -0.006376 | = overall mean d / §3.1 merge | Y |
| fold [0:40] | 0.087337 | JSON 0.393959−0.306622 = 0.087337 | Y |
| fold [40:80] | -0.081186 | JSON 0.066524−0.147710 = -0.081186 | Y |
| fold [80:120] | 0.042724 | JSON 0.211484−0.168760 = 0.042724 | Y |
| fold [120:160] | -0.033342 | JSON 0.340823−0.374165 = -0.033342 | Y |
| unweighted 4-fold mean vs JSON | 0.003883 | 0.003883 | Y |
| unweighted 4-fold mean vs result report | 0.003883 | **−0.003883** | **N** |

JSON / machine `.md` do not print a single “4-fold always-W2−ANC” field; the fold `always_w2` / `always_anc` utilities imply **+0.003883**.
`docs/R4C_IMPUTATION_DONOR_CONDITIONS_RESULT_2026-09-07.md` §3.4 writes **−0.003883** (sign flip). Same file §6 “整体略负（−0.0039）” is that flipped 4-fold figure, not the pooled −0.006376.

---

## 4. Operator assertions

Source (read-only):

- `SelfEvolvingHarnessTS/operators/s1_impute.py` `period_median_complete` (lines 149–166): donors = `raw[i − k*period]` for `k=1..cycles` with index ≥ 0 and finite; `>= min_donors` → `median(donors)`, else `interp_nan(raw)[i]`. Observed values preserved. W2 params `period=24, cycles=3, min_donors=2` (`smoke_m_r0k_scope_workflow.PMC_PARAMS` = `r4c.PMC`).
- `SelfEvolvingHarnessTS/operators/s1_outlier.py` `outlier_mad` (lines 43–50): `y = interp_nan(as_1d(x))`, then global median / MAD, clip at `med ± k * 1.4826 * mad`, default `k=3.5`.

Three synthetic arrays → `r4c.pmc_point_flags(arr, **r4c.PMC)` + `r4c.cross_check_flags` vs real `period_median_complete`:

| id | construction | flags period / fallback | cross_check | class vs source rule |
| --- | --- | --- | --- | --- |
| A | len 80, NaN at 50, donors at 26=10 and 2=14 | [50] / [] | True | Y (median fill) |
| B | len 30, NaN at 8 (no in-window donor) | [] / [8] | True | Y (`interp_nan`) |
| C | len 72; NaNs at 10 (no donor), 48 (2 donors), 36 (0 donors), 60 (1 donor) | [48] / [10,36,60] | True | Y |

Manual `outlier_mad` (interp then MAD clip k=3.5) matches `SelfEvolvingHarnessTS.operators.s1_outlier.outlier_mad` (atol 1e-12). Both assertions hold.

---

## 5. Training-side observables constant in-block; anchors

JSON `per_unit` (42 units). Within each block, `trn_gap_points` / `trn_period_filled_points` / `trn_n_windows` are bit-identical across every (position, face). Also `trn_period_filled_frac` / `trn_fill_divergence` / `trn_series_touched`.

| block | n units | trn_gap | trn_period_filled | trn_n_windows | constant |
| --- | ---: | ---: | ---: | ---: | --- |
| [0:40] | 14 | 9295 | 4207 | 200 | Y |
| [40:80] | 18 | 7593 | 4120 | 200 | Y |
| [80:120] | 4 | 7597 | 3259 | 200 | Y |
| [120:160] | 6 | 1996 | 855 | 200 | Y |

Anchors chain (source text, not a Consumer call):

`evaluation/main_protocol_p4/run_forecast_p4_performance.py::_config` → `dict(forecast_p1._config())` → `evaluation/main_protocol_p1/run_forecast_p1.py::_config` → `dict(kdd._config())` →

`evaluation/functional/run_v1_kdd2018_natural_slow_update.py::_config`:

```
"anchors": list(range(312, 853, 60)),
```

Expanded: `[312, 372, 432, 492, 552, 612, 672, 732, 792, 852]` (matches JSON `provenance.geometry.per_block.*.anchors_all`).
`max(anchors)+48 = 852+48 = 900`. Store min origin among W2/ANCESTOR entries = **1176**. `900 ≤ 1176` holds for every origin, so all 10 anchors pass `anchor+48 ≤ origin`.

---

## 6. JSON `boundary`

| field | JSON | check | match |
| --- | ---: | --- | --- |
| consumer_fits | 0 | 0 | Y |
| llm_calls | 0 | 0 | Y |
| held_out_reads | 0 | 0 | Y |
| horizon_reads | 0 | 0 | Y |
| max_time_index_read | 3863 | 3863 ≤ store max origin 3864 − 1 | Y |

This reviewer did not re-execute the 9240 raw-series reads; the inequality uses store origins only.

---

## 7. git (`git status --porcelain`)

R4C **new** (untracked) files:

- `artifacts/main_protocol/r4c_imputation_donor_conditions.json`
- `artifacts/main_protocol/r4c_imputation_donor_conditions.md`
- `docs/R4C_IMPUTATION_DONOR_CONDITIONS_RESULT_2026-09-07.md`
- `docs/R4C_IMPUTATION_DONOR_CONDITIONS_TASKBOOK_2026-09-07.md`
- `evaluation/main_protocol_p4/audit_r4c_imputation_donor_conditions.py`

Modified **tracked** files with R4C-related diff: **1** (expected 0).

- `docs/STATE_ONE_PAGE_2026-09-03.md` — dirty hunk adds “R4C 收口” and points at the result/artifacts.

Other-line dirty counts (not listed): modified tracked 59 besides that file; untracked 131 besides the 5 R4C deliverables (plus this check file after write). No new SHA created here.

---

## Inconsistencies (not corrected)

1. Result report §3.4 “always-W2 相对 always-ANC 四折均值 −0.003883”: independent unweighted mean of the four fold `mean(g_W2)-mean(g_ANC)` is **+0.003883**, and the JSON fold utilities give the same positive value. Sign error in the report. §6 “整体略负（−0.0039）” repeats that flipped 4-fold number; pooled is −0.006376.
2. Dirty tracked `docs/STATE_ONE_PAGE_2026-09-03.md` contains an R4C write-up. Task constraint was 0 R4C edits on already-tracked files.
