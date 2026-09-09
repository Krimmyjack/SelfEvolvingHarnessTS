# R4A non-author check (Grok, 2026-09-07)

Role: read-only independent recompute from `_scratch/m_r0k_prediction_store.json` (263 entries).
Did not run the audit script, did not load series, did not compute features, 0 Consumer fits, 0 LLM.
Labels (taskbook §2 and `audit_r4a_pattern_identifiability.py`): `g = raw_per_view - program_per_view`; `helped = g > 0`; `severe_harm = (program - raw) > 0.30`; `transport_flip` = support helped and delayed harmed; pairing key `(program, position, uid)` requiring both faces.
Row construction matched the script (store-native, not feature code): skip programs outside `{ANCESTOR, W1_hampel_filter, W2_pmc_then_outlier_mad, W3_outlier_mad_then_pmc, two CAND}`; drop `verifier_passed=false` (8 CAND entries); drop `degenerate_uids` (40 uid-rows). Result: 4400 rows, 0 ties `g==0`.
Rounding: six decimals, `round(x, 6)` vs `.md` `%.6f`. Match if `|a-b| < 5e-7` after that rounding.

**Grid tally (recompute vs `.md` and vs JSON): 90 / 90 match. Mismatches: none.**

---

## 1. Per (program, face) B1

| cell | field | recompute | .md | JSON | match |
| --- | --- | --- | --- | --- | --- |
| `ANCESTOR|support_face` | n_rows | 420 | 420 | 420 | Y |
| `ANCESTOR|support_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `ANCESTOR|support_face` | helped_rate | 0.692857 | 0.692857 | 0.692857 | Y |
| `ANCESTOR|support_face` | severe_harm_rate | 0.066667 | 0.066667 | 0.066667 | Y |
| `ANCESTOR|support_face` | mean_g | 0.245427 | 0.245427 | 0.245427 | Y |
| `ANCESTOR|delayed_face` | n_rows | 420 | 420 | 420 | Y |
| `ANCESTOR|delayed_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `ANCESTOR|delayed_face` | helped_rate | 0.683333 | 0.683333 | 0.683333 | Y |
| `ANCESTOR|delayed_face` | severe_harm_rate | 0.073810 | 0.073810 | 0.073810 | Y |
| `ANCESTOR|delayed_face` | mean_g | 0.224645 | 0.224645 | 0.224645 | Y |
| `W1_hampel_filter|support_face` | n_rows | 420 | 420 | 420 | Y |
| `W1_hampel_filter|support_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `W1_hampel_filter|support_face` | helped_rate | 0.459524 | 0.459524 | 0.459524 | Y |
| `W1_hampel_filter|support_face` | severe_harm_rate | 0.292857 | 0.292857 | 0.292857 | Y |
| `W1_hampel_filter|support_face` | mean_g | -0.031355 | -0.031355 | -0.031355 | Y |
| `W1_hampel_filter|delayed_face` | n_rows | 420 | 420 | 420 | Y |
| `W1_hampel_filter|delayed_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `W1_hampel_filter|delayed_face` | helped_rate | 0.509524 | 0.509524 | 0.509524 | Y |
| `W1_hampel_filter|delayed_face` | severe_harm_rate | 0.235714 | 0.235714 | 0.235714 | Y |
| `W1_hampel_filter|delayed_face` | mean_g | 0.045469 | 0.045469 | 0.045469 | Y |
| `W2_pmc_then_outlier_mad|support_face` | n_rows | 420 | 420 | 420 | Y |
| `W2_pmc_then_outlier_mad|support_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `W2_pmc_then_outlier_mad|support_face` | helped_rate | 0.640476 | 0.640476 | 0.640476 | Y |
| `W2_pmc_then_outlier_mad|support_face` | severe_harm_rate | 0.142857 | 0.142857 | 0.142857 | Y |
| `W2_pmc_then_outlier_mad|support_face` | mean_g | 0.272318 | 0.272318 | 0.272318 | Y |
| `W2_pmc_then_outlier_mad|delayed_face` | n_rows | 420 | 420 | 420 | Y |
| `W2_pmc_then_outlier_mad|delayed_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `W2_pmc_then_outlier_mad|delayed_face` | helped_rate | 0.642857 | 0.642857 | 0.642857 | Y |
| `W2_pmc_then_outlier_mad|delayed_face` | severe_harm_rate | 0.178571 | 0.178571 | 0.178571 | Y |
| `W2_pmc_then_outlier_mad|delayed_face` | mean_g | 0.185003 | 0.185003 | 0.185003 | Y |
| `W3_outlier_mad_then_pmc|support_face` | n_rows | 420 | 420 | 420 | Y |
| `W3_outlier_mad_then_pmc|support_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `W3_outlier_mad_then_pmc|support_face` | helped_rate | 0.692857 | 0.692857 | 0.692857 | Y |
| `W3_outlier_mad_then_pmc|support_face` | severe_harm_rate | 0.066667 | 0.066667 | 0.066667 | Y |
| `W3_outlier_mad_then_pmc|support_face` | mean_g | 0.245427 | 0.245427 | 0.245427 | Y |
| `W3_outlier_mad_then_pmc|delayed_face` | n_rows | 420 | 420 | 420 | Y |
| `W3_outlier_mad_then_pmc|delayed_face` | n_unique_uids | 80 | 80 | 80 | Y |
| `W3_outlier_mad_then_pmc|delayed_face` | helped_rate | 0.683333 | 0.683333 | 0.683333 | Y |
| `W3_outlier_mad_then_pmc|delayed_face` | severe_harm_rate | 0.073810 | 0.073810 | 0.073810 | Y |
| `W3_outlier_mad_then_pmc|delayed_face` | mean_g | 0.224645 | 0.224645 | 0.224645 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|support_face` | n_rows | 240 | 240 | 240 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|support_face` | n_unique_uids | 60 | 60 | 60 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|support_face` | helped_rate | 0.616667 | 0.616667 | 0.616667 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|support_face` | severe_harm_rate | 0.137500 | 0.137500 | 0.137500 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|support_face` | mean_g | 0.211554 | 0.211554 | 0.211554 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|delayed_face` | n_rows | 280 | 280 | 280 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|delayed_face` | n_unique_uids | 60 | 60 | 60 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|delayed_face` | helped_rate | 0.600000 | 0.600000 | 0.600000 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|delayed_face` | severe_harm_rate | 0.153571 | 0.153571 | 0.153571 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|delayed_face` | mean_g | 0.184201 | 0.184201 | 0.184201 | Y |
| `CAND_outlier_iqr({})>winsorize({})|support_face` | n_rows | 240 | 240 | 240 | Y |
| `CAND_outlier_iqr({})>winsorize({})|support_face` | n_unique_uids | 60 | 60 | 60 | Y |
| `CAND_outlier_iqr({})>winsorize({})|support_face` | helped_rate | 0.720833 | 0.720833 | 0.720833 | Y |
| `CAND_outlier_iqr({})>winsorize({})|support_face` | severe_harm_rate | 0.070833 | 0.070833 | 0.070833 | Y |
| `CAND_outlier_iqr({})>winsorize({})|support_face` | mean_g | 0.331931 | 0.331931 | 0.331931 | Y |
| `CAND_outlier_iqr({})>winsorize({})|delayed_face` | n_rows | 280 | 280 | 280 | Y |
| `CAND_outlier_iqr({})>winsorize({})|delayed_face` | n_unique_uids | 60 | 60 | 60 | Y |
| `CAND_outlier_iqr({})>winsorize({})|delayed_face` | helped_rate | 0.700000 | 0.700000 | 0.700000 | Y |
| `CAND_outlier_iqr({})>winsorize({})|delayed_face` | severe_harm_rate | 0.067857 | 0.067857 | 0.067857 | Y |
| `CAND_outlier_iqr({})>winsorize({})|delayed_face` | mean_g | 0.247086 | 0.247086 | 0.247086 | Y |

12 cells x 5 fields = 60. All match.

---

## 2. treat-all utility vs mean g

Script writes treat-all as `b1.mean_g`. Independent recompute of mean g equals both the B1 mean-g column and the stump table treat-all column.

| cell | recompute mean g | .md treat-all | .md mean g | JSON mean g | treat-all==mean g | match |
| --- | --- | --- | --- | --- | --- | --- |
| `ANCESTOR|support_face` | 0.245427 | 0.245427 | 0.245427 | 0.245427 | Y | Y |
| `ANCESTOR|delayed_face` | 0.224645 | 0.224645 | 0.224645 | 0.224645 | Y | Y |
| `W1_hampel_filter|support_face` | -0.031355 | -0.031355 | -0.031355 | -0.031355 | Y | Y |
| `W1_hampel_filter|delayed_face` | 0.045469 | 0.045469 | 0.045469 | 0.045469 | Y | Y |
| `W2_pmc_then_outlier_mad|support_face` | 0.272318 | 0.272318 | 0.272318 | 0.272318 | Y | Y |
| `W2_pmc_then_outlier_mad|delayed_face` | 0.185003 | 0.185003 | 0.185003 | 0.185003 | Y | Y |
| `W3_outlier_mad_then_pmc|support_face` | 0.245427 | 0.245427 | 0.245427 | 0.245427 | Y | Y |
| `W3_outlier_mad_then_pmc|delayed_face` | 0.224645 | 0.224645 | 0.224645 | 0.224645 | Y | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|support_face` | 0.211554 | 0.211554 | 0.211554 | 0.211554 | Y | Y |
| `CAND_outlier_iqr({})>hampel_filter({})|delayed_face` | 0.184201 | 0.184201 | 0.184201 | 0.184201 | Y | Y |
| `CAND_outlier_iqr({})>winsorize({})|support_face` | 0.331931 | 0.331931 | 0.331931 | 0.331931 | Y | Y |
| `CAND_outlier_iqr({})>winsorize({})|delayed_face` | 0.247086 | 0.247086 | 0.247086 | 0.247086 | Y | Y |

12 / 12: treat-all utility equals that cell mean g.

---

## 3. transport_flip

Unit = `(position, uid)` within program; both faces must exist. Flip = support `g>0` and delayed `g<0`. Rate (support helped) denominator = units with support `g>0`.

| program | field | recompute | .md | JSON | match |
| --- | --- | --- | --- | --- | --- |
| `ANCESTOR` | n_units | 420 | 420 | 420 | Y |
| `ANCESTOR` | flip_rate_over_all_units | 0.219048 | 0.219048 | 0.219048 | Y |
| `ANCESTOR` | flip_rate_over_support_helped | 0.316151 | 0.316151 | 0.316151 | Y |
| `W1_hampel_filter` | n_units | 420 | 420 | 420 | Y |
| `W1_hampel_filter` | flip_rate_over_all_units | 0.207143 | 0.207143 | 0.207143 | Y |
| `W1_hampel_filter` | flip_rate_over_support_helped | 0.450777 | 0.450777 | 0.450777 | Y |
| `W2_pmc_then_outlier_mad` | n_units | 420 | 420 | 420 | Y |
| `W2_pmc_then_outlier_mad` | flip_rate_over_all_units | 0.214286 | 0.214286 | 0.214286 | Y |
| `W2_pmc_then_outlier_mad` | flip_rate_over_support_helped | 0.334572 | 0.334572 | 0.334572 | Y |
| `W3_outlier_mad_then_pmc` | n_units | 420 | 420 | 420 | Y |
| `W3_outlier_mad_then_pmc` | flip_rate_over_all_units | 0.219048 | 0.219048 | 0.219048 | Y |
| `W3_outlier_mad_then_pmc` | flip_rate_over_support_helped | 0.316151 | 0.316151 | 0.316151 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})` | n_units | 240 | 240 | 240 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})` | flip_rate_over_all_units | 0.229167 | 0.229167 | 0.229167 | Y |
| `CAND_outlier_iqr({})>hampel_filter({})` | flip_rate_over_support_helped | 0.371622 | 0.371622 | 0.371622 | Y |
| `CAND_outlier_iqr({})>winsorize({})` | n_units | 240 | 240 | 240 | Y |
| `CAND_outlier_iqr({})>winsorize({})` | flip_rate_over_all_units | 0.225000 | 0.225000 | 0.225000 | Y |
| `CAND_outlier_iqr({})>winsorize({})` | flip_rate_over_support_helped | 0.312139 | 0.312139 | 0.312139 | Y |

JSON also has `n_support_helped_units` (not in the .md table): ANCESTOR 291, W1 193, W2 269, W3 291, CAND hampel 148, CAND winsorize 173 — matches independent recompute.
6 programs x 3 fields = 18. All match.

---

## 4. JSON `boundary`

| field | JSON | check |
| --- | --- | --- |
| consumer_fits (`fits`) | 0 | required 0 |
| llm_calls (`llm`) | 0 | required 0 |
| held_out_reads | 0 | required 0 |
| max_time_index_read | 3911 | see bound below |
| held_out_frontier | 4056 | 4056 |

Store origin range: min 1176, max 3864 (support max 3816, delayed max 3864).

User bound: `max_time_index_read <= 库内最大 origin + 48 (delayed) + 48 (horizon) - 1`

- 库内最大 origin = 3864 (this *is* a delayed_face origin)
- bound = 3864 + 48 + 48 - 1 = **3959**
- JSON `max_time_index_read` = **3911** <= 3959? **PASS**

Tighter geometry (ORACLE reads `[origin, origin+48)` so last index = origin+47): delayed max 3864 + 48 - 1 = **3911**. JSON equals this. Equivalently support max 3816 + 48 + 48 - 1 = 3911. 3911 < held-out frontier 4056.

This check did not re-execute series reads; it only verified the claimed index against store origins.

---

## 5. ANCESTOR|delayed_face `local_robust_z_peak` vs severe_harm (JSON read, not recomputed)

Feature AUCs depend on series/feature code; **not independently recomputed**. JSON vs `.md` only.

| quantity | JSON | .md | match |
| --- | --- | --- | --- |
| best visible | `local_robust_z_peak|severe_harm` | `local_robust_z_peak\|severe_harm` | Y |
| LODO mean (oriented) | 0.676036 | 0.676036 | Y |
| LODO folds (oriented) | [0:40]=0.717803, [120:160]=0.599129, [40:80]=0.702528, [80:120]=0.684685 | [0:40]=0.717803, [120:160]=0.599129, [40:80]=0.702528, [80:120]=0.684685 | Y |
| pooled_auc (raw, not max(v,1-v)) | 0.333071 | (not in .md table) | — |
| direction_consistent | True | (implicit in WEAK verdict) | — |

Per-fold raw AUC (JSON `auc_raw`, **not** oriented):

| fold | auc_raw | auc_oriented | train_auc |
| --- | --- | --- | --- |
| [0:40] | 0.282197 | 0.717803 | 0.352563 |
| [120:160] | 0.400871 | 0.599129 | 0.296194 |
| [40:80] | 0.297472 | 0.702528 | 0.356818 |
| [80:120] | 0.315315 | 0.684685 | 0.329393 |

**Direction:** JSON stores `auc_raw` / `train_auc` (not only `max(v, 1-v)`). Pooled raw AUC = 0.333071 < 0.5; every fold `auc_raw` < 0.5 and every `train_auc` < 0.5, so oriented = 1 - raw. **High `local_robust_z_peak` corresponds to *less* severe_harm.**

---

## 6. git

`git status --porcelain`: 172 lines. `git diff --stat`: 11 files, +1711 / -126 (many `artifacts/functional/e2/**` diffs failed with `Filename too long`).

### Modified existing (tracked) files: 60, not 0

None of these 60 is an R4A artifact. R4A json/md/script/docs are all untracked (`??`). The working tree nevertheless has 60 dirty tracked files from other lines:

Non-e2 (13):

- `AGENTS.md`
- `docs/FABLE_FINAL_REVIEW_AND_SUCCESSOR_BRIEF_2026-09-03.md`
- `docs/HEC1_INDEPENDENT_REVIEW_CHECKLIST_2026-09-03.md`
- `docs/HEC_EVOLUTION_MAINLINE_PLAN_2026-09-02.md`
- `docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`
- `docs/STATE_ONE_PAGE_2026-09-03.md`
- `evaluation/main_protocol_p4/audit_hec1_best_safe_global.py`
- `evaluation/main_protocol_p4/outer_loop.py`
- `evaluation/main_protocol_p4/restricted_draft.py`
- `evaluation/main_protocol_p4/run_hec1.py`
- `evaluation/main_protocol_p4/run_source_line_v3.py`
- `methods/ttha/online_loop.py`
- `methods/ttha/scope_executor.py`

e2 snapshot JSON (47): all under `artifacts/functional/e2/` (harness_snapshot_provenance / proposer_micro / wrong_prior_micro / t6_nab). `git diff --stat` did not print them (`Filename too long`).

R4A-related tracked modifications: **0**.

### New untracked (porcelain `??`): 112 entries

R4A-related:

- `artifacts/main_protocol/r4a_pattern_identifiability.json`
- `artifacts/main_protocol/r4a_pattern_identifiability.md`
- `docs/R4A_IDENTIFIABILITY_EVIDENCE_LEDGER_2026-09-07.md`
- `docs/R4A_PATTERN_IDENTIFIABILITY_RESULT_2026-09-07.md`
- `docs/R4A_PATTERN_IDENTIFIABILITY_TASKBOOK_2026-09-07.md`
- `evaluation/main_protocol_p4/audit_r4a_pattern_identifiability.py`

Other untracked (compact): `.dev_auto1_runs/` `.dev_auto2_runs/` `.dev_auto3_runs/` `.dev_know1_runs/` `idea-stage/` `refine-logs/`; `artifacts/main_protocol/` dev_auto* / dev_know* / m_r0* / hec1_course_pytest* / r2_revision*; `docs/` DEV_* / M_* / NEXT_* / OVERALL_* / V2_* / SelfEvolvingHarnessTS literature md; `evaluation/main_protocol_p4/` remaining audit_*/run_*/smoke_* scripts listed in porcelain; `tests/main_protocol/test_hec1_relation_and_lineage_repair.py`.

This check file `artifacts/main_protocol/r4a_nonauthor_check.md` is the only file written by the reviewer (new).

---

## 7. Inconsistencies

**None** among independently recomputable B1 / treat-all / transport_flip grids (90/90), nor JSON-vs-.md LODO AUC for `local_robust_z_peak|severe_harm` on ANCESTOR delayed.

Notes, not mismatches:

- W3 rows are numerically identical to ANCESTOR on every B1 and transport number (store-level; script provenance already flags alias).
- Working tree has 60 modified *pre-existing* tracked files; R4A delivery itself added only untracked files.
- AUC direction and fold AUCs were read from JSON, not recomputed from series.
