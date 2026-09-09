# R4D-A non-author check (Grok, 2026-09-08)

Role: read-only independent recompute from the two prediction stores.
Did **not** run `run_r4d_a_action_decomposition.py`, did not call `_serve`, 0 Consumer fits, 0 Harness LLM, 0 sub-agents, no git commit, no new SHA, no edits to existing files.
numpy 2.1.3 in the current shell. Spearman = average-rank Pearson, implemented here (no scipy).
Match rule: six-decimal display; `|recompute − artifact| < 1e-9` counts as consistent.

Frozen verdict (`docs/R4D_PERCHANNEL_STAGE_A_AND_ACTION_DECOMPOSITION_REQUEST_2026-09-07.md` §4.1, loss convention as specified for this check):
`ROUTE_DOMINANT` iff route algebraic share `Σ|route|/(Σ|route|+Σ|ctx|) ≥ 0.6` **and** among `total > 0.30`, share of instances with route the largest positive contribution (`route > ctx`) ≥ 0.6.

**Grid tally (recompute vs artifact `.md` / `.json` six-cell table): 6 / 6 match. Verdicts: all six `ROUTE_DOMINANT` (re-judged). Headline recon and unmodified counts match. One wording / exact-zero flag mismatch listed in §3 / §6 (not a verdict change).**

---

## 0. New-store structure (confirmed before any arithmetic)

File: `_scratch/r4d_a_three_cell_store.json` (447054 bytes). Top-level keys: `task`, `generated_at`, `note`, `definitions`, `boundary`, `entries`.

- `task` = `R4D_A_THREE_CELL_LOSSES`
- `entries`: 84 keys of the form `program|position|face` (2 programs × 21 positions × 2 faces)
- One complete record (`ANCESTOR|0|delayed_face`) fields: `program`, `position`, `face`, `origin`, `block`, `eval_uids` (20), `L_rr`, `L_pr`, `L_pp`, `route`, `ctx`, `total`, `status`, `serving_moved_points`, `serving_unmodified`, `store_raw_per_view`, `store_program_per_view`
- Per-sequence losses are aligned with `eval_uids`. `route` / `ctx` / `total` stored in the file equal `L_pr−L_rr` / `L_pp−L_pr` / `L_pp−L_rr` with max |Δ| = 0 (1680/1680 bit-identical). This check **recomputes** the three differences from `L_*` and does **not** trust the cached `store_*` fields for reconciliation — those come from the old store.

Old store `_scratch/m_r0k_prediction_store.json`: 263 top-level keys `program|position|face`; fields `eval_uids`, `raw_per_view`, `program_per_view`, `origin`, … . `eval_uids` order matches the new store on every compared face (0 mismatches).

---

## 1. Reconciliation (new `L_rr`/`L_pp` vs old `raw_per_view`/`program_per_view`)

Compared every (program ∈ {ANCESTOR, W2_pmc_then_outlier_mad}, position, face, uid) by uid. Tolerance 1e-9.

| item | recompute | artifact `.md` / `.json` | match |
| --- | ---: | ---: | --- |
| rows compared | 1680 / 1680 | 1680 | Y |
| `L_rr` reproduced (`|Δ|<1e-9`) | 1680 / 1680 | 1680 / 1680 | Y |
| `L_pp` reproduced (`|Δ|<1e-9`) | 1680 / 1680 | 1680 / 1680 | Y |
| max `|Δ|` `L_rr` | 1.3322676295501878e-15 | 1.3322676295501878e-15 | Y |
| max `|Δ|` `L_pp` | 8.881784197001252e-16 | 8.881784197001252e-16 | Y |
| offenders | 0 | 0 | Y |

Per program:

| program | n | L_rr ok | bit-id L_rr | max \|Δ\| L_rr | L_pp ok | bit-id L_pp | max \|Δ\| L_pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | 840 | 840 | 788 | 1.3322676295501878e-15 | 840 | 788 | 6.661338147750939e-16 |
| W2_pmc_then_outlier_mad | 840 | 840 | 788 | 1.3322676295501878e-15 | 840 | 791 | 8.881784197001252e-16 |

Bit-identical counts match the artifact table exactly. Residual 1e-16–1e-15; no row outside 1e-9.

---

## 2. Six-cell decomposition (recomputed from `L_*`) vs artifact `.md` table

`route = L_pr − L_rr`, `ctx = L_pp − L_pr`, `total = L_pp − L_rr`. Cells = 2 programs × {support, delayed, both}.

| program | scope | n | route share | sev n | route>ctx \| sev | route mean | route med | ctx mean | ctx med | ρ(route,total) | ρ(ctx,total) | verdict | vs md |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ANCESTOR | support_face | 420 | 0.768009429 | 28 | 0.964285714 (27/28) | -0.163761133 | -0.083867398 | -0.081666133 | 0.0 | 0.901856263 | 0.214287040 | ROUTE_DOMINANT | Y |
| ANCESTOR | delayed_face | 420 | 0.803766880 | 31 | 0.967741935 (30/31) | -0.161654515 | -0.105627802 | -0.062990815 | 0.0 | 0.937469357 | 0.245078368 | ROUTE_DOMINANT | Y |
| ANCESTOR | both_faces | 840 | 0.784907566 | 59 | 0.966101695 (57/59) | -0.162707824 | -0.093033272 | -0.072328474 | 0.0 | 0.920054511 | 0.228908554 | ROUTE_DOMINANT | Y |
| W2 | support_face | 420 | 0.714483157 | 60 | 0.650000000 (39/60) | -0.256276681 | -0.112454517 | -0.016041217 | 0.0 | 0.867865464 | 0.291901220 | ROUTE_DOMINANT | Y |
| W2 | delayed_face | 420 | 0.703445421 | 75 | 0.786666667 (59/75) | -0.174467469 | -0.142246309 | -0.010535302 | 0.0 | 0.846592425 | 0.255882915 | ROUTE_DOMINANT | Y |
| W2 | both_faces | 840 | 0.709106337 | 135 | 0.725925926 (98/135) | -0.215372075 | -0.119121277 | -0.013288260 | 0.0 | 0.857872308 | 0.273619555 | ROUTE_DOMINANT | Y |

Every numeric field vs artifact `.json` / `.md`: `|Δ| < 1e-9` (in practice the printed 9-decimal values are identical). Tightest verdict cell: W2 support, severe route share = 0.65 exactly (≥ 0.6).

**Re-judgement:** all six cells `ROUTE_DOMINANT`. Same as artifact. Result-report §5 four-row table is the same numbers rounded to 3 decimals; not a second population.

---

## 3. Serving-window unmodified / ctx == 0

New store has both a per-sequence boolean `serving_unmodified` and a numeric `ctx`. This check reports three predicates.

| program | face | n | `serving_unmodified` | `\|ctx\|<1e-12` | `ctx == 0.0` exact | sev n | sev among `\|ctx\|<1e-12` | sev share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | delayed_face | 420 | **269** | **269** | 268 | **31** | **24** | **0.774194** (24/31) |
| ANCESTOR | support_face | 420 | **236** | **236** | 235 | **28** | **16** | **0.571429** (16/28) |
| W2 | delayed_face | 420 | 42 | 42 | 42 | 75 | 8 | 0.106667 |
| W2 | support_face | 420 | 75 | 75 | 74 | 60 | 10 | 0.166667 |

Headline three-number check (ANCESTOR delayed 269/420, 24, 0.774; support 236/420, 16/28): **match** under `serving_unmodified` and under `|ctx| < 1e-12`.

**Inconsistency (wording / exact-zero flag, not the headline counts):** report §5.1 and JSON `ctx_exactly_zero_on_unmodified` claim ctx is bit-identical 0 on all unmodified instances (`L_pp = L_pr` 逐位相等). Independent: 3 unmodified rows have a 1-ULP residual, so `ctx == 0.0` is false:

| key | uid | ctx | total | severe? |
| --- | --- | ---: | ---: | --- |
| `ANCESTOR\|17\|delayed_face` | T195 | +2.220446049250313e-16 | 0.039534367 | N |
| `ANCESTOR\|16\|support_face` | T188 | -4.440892098500626e-16 | -0.257845817 | N |
| `W2_pmc_then_outlier_mad\|16\|support_face` | T188 | -2.220446049250313e-16 | -0.069801056 | N |

None of the three is severe, so 24/31 and 16/28 are unchanged whether the predicate is exact-0 or `|ctx|<1e-12`. JSON `ctx_exactly_zero_on_unmodified` equals the unmodified count, not the bit-identical-zero count. Not corrected here.

---

## 4. JSON `boundary`

| item | JSON / new-store boundary | independent | match |
| --- | ---: | ---: | --- |
| physical_fits (this run) | 12 | cannot re-count fits (0 `_serve` in this check); JSON + new-store both 12 | as written |
| hard cap | 24 | 24 | Y |
| report cumulative | 24 / 24 (result §2 / §6: two runs × 12) | 12 + 12 = 24 | Y (report text) |
| llm_calls | 0 | 0 observed in this check | Y |
| held_out_reads | 0 | 0 (this check did not read held-out; JSON claims 0) | Y as written |
| max_time_index_read | 3911 | old-store max `origin` = **3864**; 3864 + 48 − 1 = **3911** | Y |

---

## 5. `git status --porcelain` (R4D-A slice)

R4D-A delivery files (untracked unless noted):

| path | git |
| --- | --- |
| `evaluation/main_protocol_p4/run_r4d_a_action_decomposition.py` | `??` |
| `artifacts/main_protocol/r4d_a_action_decomposition.json` | `??` |
| `artifacts/main_protocol/r4d_a_action_decomposition.md` | `??` |
| `docs/R4D_A_ACTION_DECOMPOSITION_RESULT_2026-09-08.md` | `??` |
| `_scratch/r4d_a_three_cell_store.json` | gitignored (`.gitignore:26` `_scratch/`) |

Also untracked, preregistration (not this run's output): `docs/R4D_PERCHANNEL_STAGE_A_AND_ACTION_DECOMPOSITION_REQUEST_2026-09-07.md`.

Modified tracked files with R4D-A content: **`docs/STATE_ONE_PAGE_2026-09-03.md` only** (mainline intentional update; not a defect). Other modified tracked files (`AGENTS.md`, HEC docs, `evaluation/main_protocol_p4/{outer_loop,run_hec1,…}`, `methods/ttha/*`, e2 snapshot JSON) have no R4D-A delivery content. Expected 0 R4D-A edits to pre-existing tracked files besides the state page: **holds**.

---

## 6. Inconsistencies (explicit; artifacts not edited)

1. Report / JSON claim “ctx 恰为 0 / `L_pp = L_pr` 逐位相等” on all unmodified rows. False for 3 rows at 1 ULP (see §3). Headline 269/420, 24, 0.774 and 236/420, 16/28 still hold under `|ctx|<1e-12` and `serving_unmodified`.
2. No other numeric disagreement at 1e-9 on recon, six-cell table, verdicts, or boundary identity `3911 = 3864+48-1`.

---

## 7. Boundary self-check (this review)

physical_fits = **0**; llm = 0; held_out_reads = 0; `run_r4d_a_action_decomposition.py` not executed; `_serve` not imported; no sub-agent; no git commit; no new SHA; only new file is this check.
