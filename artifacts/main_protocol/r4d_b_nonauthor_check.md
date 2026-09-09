# R4D-B non-author check (Grok, 2026-09-08)

Role: read-only independent recompute from the two three-cell stores.
Did **not** run `run_r4d_b_perchannel_three_cell.py` / `run_r4d_a_action_decomposition.py`, did not call `_serve`, 0 Consumer fits, 0 Harness LLM, 0 sub-agents, no git commit, no new SHA, no edits to existing files.
numpy 2.1.3 in the current shell. Spearman = average-rank Pearson (mergesort average ranks; no scipy).
Match rule: `|recompute − artifact| < 1e-9` counts as consistent.

Frozen verdict (R4D-A §4.1, reused): `ROUTE_DOMINANT` iff route algebraic share `Σ|route|/(Σ|route|+Σ|ctx|) ≥ 0.6` **and** among `total > 0.30`, share of instances with `route > ctx` ≥ 0.6.

**Tally: 8 / 8 numbered items consistent on stored channels / JSON boundary / git. Three rounding notes (item 1 algebra at the 9-decimal bound; item 3 ANCESTOR Spearman and item 4 ANCESTOR delayed `hf` if channels are rebuilt from rounded `L_*`) — no verdict change, artifacts not edited.**

---

## 0. Store structure (one full instance printed before any arithmetic)

pc store `_scratch/r4d_b_perchannel_three_cell_store.json` (237281 bytes). Top-level: `task`, `generated_at`, `note`, `definitions`, `boundary`, `consumer`, `entries`. `task` = `R4D_B_PERCHANNEL_THREE_CELL_LOSSES`. `entries`: 84 keys `program|position|face`.

One complete instance (`ANCESTOR|0|delayed_face`, uid `T1`): fields `program, position, face, origin, block, eval_uids, L_rr, L_pr, L_pp, route, ctx, total, status, serving_moved_points, serving_unmodified, store_raw_per_view, store_program_per_view`.

| field | pc value | pooled same key |
| --- | ---: | ---: |
| L_rr | 2.308973147 | 1.4159302193895233 |
| L_pr / L_pp | 2.391619162 | 1.1608783091301715 |
| route / total | 0.082646015 | −0.2550519102593518 |
| ctx | 0.0 | 0.0 |
| status | OK | OK |
| serving_unmodified | true | true |
| store_raw_per_view | 1.415930219 (= pooled L_rr) | 1.4159302193895233 |
| store_program_per_view | 1.160878309 (= pooled L_pp) | 1.1608783091301715 |

pc `store_*` are the paired pooled `L_rr` / `L_pp` (not a reconciliation target). pc numeric fields are `_round(..., 9)` except `|v| < 1e-6` nonzero leftovers kept raw. pooled store keeps full floats. Expanded rows: 1680 / 1680.

---

## 1. pc internal consistency

`route ≟ L_pr − L_rr`, `ctx ≟ L_pp − L_pr`, `total ≟ L_pp − L_rr`; status.

| item | recompute | artifact | match |
| --- | ---: | ---: | --- |
| rows | 1680 | 1680 | Y |
| status OK | 1680 / 1680 | 1680 / 1680 | Y |
| max \|stored − (L_* diff)\| route / ctx / total | 1.0000005e-9 / 1.0000004e-9 / 1.0000005e-9 | (identity) | note |
| rows with any \|Δ\| ≥ 1e-9 | 358 | 0 expected if unrounded | note |
| rows with any \|Δ\| > 1.5e-9 | 0 | 0 | Y |

Note: 358 rows sit at the 9-decimal rounding bound (`_round` applied independently to `L_*` and to channels). No live algebra break. Counted consistent.

---

## 2. Baseline quality (ANCESTOR; delta = pc `L_rr` − pooled `L_rr`; dedup `(position, face, uid)`)

| scope | n | delta mean | delta median | share pc worse | vs json (9 dp) |
| --- | ---: | ---: | ---: | ---: | --- |
| support_face | 420 | −0.469732832 | −0.255676629 | 0.280952381 | Y (\|Δ\| < 1e-9) |
| delayed_face | 420 | −0.281209258 | −0.184632772 | 0.380952381 | Y |
| both_faces | 840 | −0.375471045 | −0.223985062 | 0.330952381 | Y |

Matches the request’s 6-dp headline (−0.375471 / −0.223985 / 0.330952; support −0.469733 / −0.255677 / 0.280952; delayed −0.281209 / −0.184633 / 0.380952) and `.md` §1 / JSON `readout_1`.

---

## 3. pc six-cell decomposition vs `.md` §2 + frozen re-judgement

Primary numbers use the store’s own `route` / `ctx` / `total` (item 1 identity holds to display precision).

| program | scope | n | route share | sev n | route>ctx \| sev | ρ(route,total) | ρ(ctx,total) | verdict | vs §2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ANCESTOR | support_face | 420 | 0.727289000 | 3 | 0.666666667 (2/3) | 0.755415357 | 0.433600067 | ROUTE_DOMINANT | Y |
| ANCESTOR | delayed_face | 420 | 0.791368246 | 9 | 1.000000000 (9/9) | 0.879168021 | 0.234632563 | ROUTE_DOMINANT | Y |
| ANCESTOR | both_faces | 840 | 0.753710679 | 12 | 0.916666667 (11/12) | 0.816480533 | 0.352889693 | ROUTE_DOMINANT | Y |
| W2 | support_face | 420 | 0.733194720 | 8 | 0.875000000 (7/8) | 0.880168216 | 0.310515432 | ROUTE_DOMINANT | Y |
| W2 | delayed_face | 420 | 0.732890161 | 6 | 0.833333333 (5/6) | 0.831814246 | 0.413020756 | ROUTE_DOMINANT | Y |
| W2 | both_faces | 840 | 0.733052175 | 14 | 0.857142857 (12/14) | 0.856912228 | 0.360796139 | ROUTE_DOMINANT | Y |

**Re-judgement (frozen rule):** all six `ROUTE_DOMINANT`. Same as artifact.

If channels are rebuilt from rounded `L_*` instead of stored channels, shares / severe n / severe-route share / verdicts still match; ANCESTOR Spearmans move (ties + one leftover, see §4 note): support ρ(ctx,total) 0.436533 vs 0.433600; delayed ρ(route,total) 0.879299 vs 0.879168; delayed ρ(ctx,total) 0.242231 vs 0.234633; both ρ(route,total) 0.816536 vs 0.816481; both ρ(ctx,total) 0.358285 vs 0.352890. W2 Spearmans stay inside 1e-9. Not a table error.

---

## 4. Harm shape vs `.md` §5

`aggregate_gain = −mean(total)`, `harmed_fraction = mean(total>0)`, `max_single_series_harm = max(total)`. Stored `total`.

| program\|face | pc gain | pooled gain | pc hf | pooled hf | pc msh | pooled msh | vs §5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ANCESTOR\|support_face | 0.072310116 | 0.245427266 | 0.269047619 | 0.307142857 | 0.365648905 | 1.180821209 | Y |
| ANCESTOR\|delayed_face | 0.017838252 | 0.224645329 | 0.330952381 | 0.316666667 | 0.769697476 | 2.021520860 | Y |
| W2\|support_face | 0.116995598 | 0.272317898 | 0.359523810 | 0.359523810 | 1.186306151 | 1.826279038 | Y |
| W2\|delayed_face | 0.101825685 | 0.185002771 | 0.364285714 | 0.357142857 | 0.609590461 | 1.848863568 | Y |

Note (L_* rebuild only): `ANCESTOR|17|delayed_face|T189` has stored `total = 2.22e-16` (`_round` keeps `|v|<1e-6`) while rounded `L_pp − L_rr = 0`. That flips `total>0` (139 → 138 / 420), so L_* `hf` = 0.328571429 vs §5 0.330952381. Shape/gain verdicts unchanged. Artifacts not edited.

---

## 5. Persistence (pc `total`)

Cross-face Spearman: per position, pair the same `(uid)` on both faces, then median over positions. uid share: per-block one-way ANOVA `SSB/SST` grouped by uid, weighted by block row count.

| item | recompute | artifact / request | match |
| --- | ---: | ---: | --- |
| ANCESTOR cross-face Spearman median | 0.073282443 | 0.073282443 (request 0.073282) | Y |
| W2 cross-face Spearman median | 0.144360902 | 0.144360902 (request 0.144361) | Y |
| W2 support uid share (weighted) | 0.317412734 | 0.317412734 (request 0.317413) | Y |

---

## 6. JSON `boundary`

| field | JSON | claim | match |
| --- | ---: | ---: | --- |
| physical_fits | 240 | 240 | Y |
| physical_fits_hard_cap | 260 | 260 | Y |
| held_out_reads | 0 | 0 | Y |
| max_time_index_read | 3911 | 3911 = 3864+48−1 | Y |
| feature_reads_max_time_index | 3863 | 3863 | Y |

This check spent 0 physical fits (read-only).

---

## 7. §4 pc `PATTERN_INFORMATIVE` cells

All four pc INFORMATIVE cells have `full-support INFORMATIVE = []` (JSON `full_support_informative` and `.md` last column). They are **not** backed by a feature that clears every fold with full class support.

Severe-harm **channel** positives recomputed from the pc store (`channel > 0.30`; ctx population = `|ctx| > 1e-12`, n=723, same count on all 840):

| cell | n_rows (artifact) | channel > 0.30 | total > 0.30 (reference) |
| --- | ---: | ---: | ---: |
| ANCESTOR\|route\|support_face | 420 | **3** (route) | 3 |
| W2\|route\|delayed_face | 420 | **5** (route) | 6 |
| W2\|ctx\|both_faces | 723 | **3** (ctx) | 14 |
| W2\|total\|delayed_face | 420 | **6** (total) | 6 |

Empty full-support lists + these counts (3 / 5 / 3 / 6) are the “缺正例折撑起” flag: INFORMATIVE here is a thin positive class, not a fully supported pattern.

---

## 8. `git status --porcelain` (R4D-B slice)

New untracked R4D-B files:

- `artifacts/main_protocol/r4d_b_perchannel_three_cell.md`
- `artifacts/main_protocol/r4d_b_perchannel_three_cell.json`
- `evaluation/main_protocol_p4/run_r4d_b_perchannel_three_cell.py`
- `docs/R4D_B_PERCHANNEL_RESULT_2026-09-08.md`

`_scratch/r4d_b_perchannel_three_cell_store.json` exists (237281 bytes) but `_scratch/` is gitignored — not in porcelain.

Modified **already-tracked** files with R4D-B content: **1**, the status page `docs/STATE_ONE_PAGE_2026-09-03.md` (expected exception). `AGENTS.md` is modified but has **0** R4D-B strings. Other modified tracked paths (e2 snapshots, HEC docs/scripts, `scope_executor.py`, …) have **0** R4D-B content.

Expected “0 R4D-B edits to pre-existing tracked files except the status page”: **held**.

---

## Inconsistencies (explicit; artifacts not corrected)

No verdict-level mismatch. Rounding notes only:

1. Item 1: 358/1680 rows have `|stored channel − L_* diff| ≈ 1.000e-9` (independent 9-decimal `_round`). None > 1.5e-9.
2. Item 3: rebuilding channels from rounded `L_*` moves ANCESTOR Spearmans (listed in §3); stored-channel path matches §2 inside 1e-9.
3. Item 4: `ANCESTOR|17|delayed_face|T189` leftover `total=2.22e-16` vs `L_pp−L_rr=0` flips one `total>0` if rebuilt from `L_*`.
