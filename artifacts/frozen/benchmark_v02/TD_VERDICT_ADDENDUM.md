# benchmark-v0.2 — verdict addendum: squeezing the water out

**Written 2026-07-14, after `TD_VERDICT.md` and after the results were read.** Everything below
is therefore **post-hoc** and is labelled as such. It changes no decision rule, no threshold,
no frozen artifact, and no split. Final-Query remains sealed.

Recomputation script: `diagnostics/v02_covid_sensitivity.py`.
Machine-readable output: `covid_sensitivity.json`.

The script refuses to run unless it first reproduces the published ladder from
`dev_repeat_losses.jsonl` to within `1e-9` — so the corrected numbers below come out of the
same code path as the numbers they correct, and the only thing that changed is the roster.

---

## Why this exists

`TD_VERDICT.md` got every **direction** right and most **magnitudes** wrong.

Two cells — `monash:covid_deaths|trend_high` (sMASE **122.1**) and
`monash:covid_deaths|seasonal_high` (**59.2**), against ~1.0 everywhere else — enter the frozen
macro fold with weight 1/3 inside their regime and 1/4 across regimes. COVID deaths are a
monotone cumulative count, so the sMASE denominator (a seasonal-naive difference) is nearly
zero and the ratio explodes. **One single series carries a raw sMASE of 1507.6.**

The result is that four of the verdict's headline numbers are 80–100% COVID. Any reviewer who
recomputes from the persisted losses finds this in ten minutes. Better that we find it first.

**None of this reverses the verdict.** The Gate still passes. But the numbers we may *quote*
shrink by 4–6×, two of §2's exemplars are artifacts and are withdrawn, and one of §1's
rhetorical set-pieces collapses entirely.

---

## 1. COVID exclusion sensitivity ladder

| role | all 6 datasets | ex-COVID |
| --- | --- | --- |
| floor: Raw | 11.4815 | **1.0300** |
| floor: `best_fixed` = `denoise_stl` | 10.9784 | **0.9401** |
| incumbent: H_ref | 11.5583 | **1.0300** |
| **CEILING (Gate): `oracle_transfer_retrained`** | 10.7881 | **0.8981** |
| envelope: `oracle_insample_retrained` | 10.5410 | 0.8897 |

| quantity | all 6 datasets | ex-COVID | COVID share |
| --- | --- | --- | --- |
| headroom, Raw → ceiling | +0.6934 | **+0.1319** | **81.0%** |
| C1 space, `best_fixed` → ceiling | +0.1903 | **+0.0419** | **78.0%** |
| winner's curse (envelope − ceiling) | +0.2471 | **+0.0084** | **96.6%** |
| H_ref − Raw | +0.0768 | **+0.0000006** | **~100%** |

**The Gate does not flip.** Both Gate quantities still clear ε = 0.02 ex-COVID
(0.1319 > 0.02; 0.0419 > 0.02), and — checked, not assumed — **all six readable-and-material
cells are non-COVID** (`traffic_hourly|seasonal_high`, `uci|seasonal_high`,
`metr_la|low_structure`, `metr_la|structured_mixed`, `traffic_hourly|structured_mixed`,
`gefcom|seasonal_high`). COVID contributed **zero** readable cells — its effects are enormous
but its `mde_80` is larger still. Excluding it removes no evidence, only inflation.

**Report the relative figures, not the absolute ones.** Against a non-degenerate floor
(Raw = 1.0300) the honest statements are:

- data preparation buys **12.8%** over doing nothing (Raw → ceiling);
- **conditioning** buys **4.5%** over the best single program (`best_fixed` → ceiling).

These are the numbers that should appear in any write-up. `+0.69` and `+0.19` should not.

### Two claims in §1 that do not survive

**The winner's curse set-piece collapses.** `TD_VERDICT.md` §1 argued: *"the winner's curse
(+0.2471) is larger than the entire honest conditioning gain (+0.1903)"* — presented as the
empirical vindication of the transfer oracle. Ex-COVID the curse is **+0.0084**, which is
**below ε** and **five times smaller** than the conditioning gain (+0.0419). **The claim is
withdrawn.** The transfer oracle remains methodologically correct — an in-sample oracle is a
biased ceiling whether or not the bias is large here — but it is no longer true that the curse
dominates the conditioning space. It was a COVID artifact.

**"H_ref is net worse than Raw" is entirely COVID.** Ex-COVID, H_ref − Raw = **+0.0000006**.
This is not a new finding — TB already established that H_ref selects `impute_linear`, which
*is* canonical ingestion, on 329 of 373 series — but the bolded ladder row "H_ref 11.5583 —
worse than doing nothing" reads as an independent result in isolation, and it is not one. **H_ref
and Raw are numerically the same object outside COVID.**

*(Footnote, reported and not adopted: re-selecting the floor ex-COVID would still pick
`denoise_stl`. The floor stays as selected on Support-A regardless — re-selecting a floor after
choosing which dataset to drop is exactly the post-hoc move this file exists to catch.)*

---

## 2. §2's exemplars: two withdrawn, two replacements

`TD_VERDICT.md` §2 made the right argument with the wrong evidence. Both of its most vivid
exemplars are COVID artifacts:

| withdrawn claim | as printed | ex-COVID | status |
| --- | --- | --- | --- |
| "`winsorize` **harms clean data** (+0.845)" | +0.845 | **−0.033** | **false** — it mildly *helps* |
| "`forward_fill` is **catastrophic** under heavy missingness (+1.259)" | +1.259 | **+0.001** | **false** — it does nothing |

Ex-COVID, `winsorize` is mildly beneficial in *all three* headline conditions (natural −0.033,
block 0.24 −0.024, spike 0.03 −0.162). The "winsorize harms clean data" story was one
cumulative-count series being clipped.

`forward_fill`'s collapse to ≈0 has a structural cause worth recording: **Raw is not
un-imputed.** Programs run *before* canonical ingestion, and ingestion resolves any remaining
NaNs by linear interpolation. So `forward_fill` vs `raw` is not "filled vs unfilled" — it is
**"forward-fill vs linear interpolation"**, a distinction that is nearly free on ordinary
series (+0.001) and only bites on a monotone cumulative count, where forward-filling a block
creates a plateau-then-jump while interpolation tracks the trend. This is the same mechanism
TB found behind H_ref.

### Replacement exemplars (these survive ex-COVID)

| program | natural | block 0.24 | spike 0.03 |
| --- | --- | --- | --- |
| `denoise_stl` | −0.052 help | **+0.110 harm** | **−0.227 help** |
| `denoise_median` | **+0.020 harm** | +0.069 harm | **−0.158 help** |

*(mean sMASE minus Raw; negative = better than doing nothing)*

### C1 survives — and here is its strongest form

The C1 claim is not "some program helps on average". It is **"the program you should run
changes with the condition"**. That claim lives or dies on whether the *argmin* moves. Ex-COVID,
across all nine grid conditions:

| condition | best program | gain vs Raw |
| --- | --- | --- |
| block 0.12 | **`winsorize`** | −0.0290 |
| block 0.24 | **`winsorize`** | −0.0236 |
| gaussian 0.5 | **`winsorize`** | −0.0199 |
| natural | `denoise_stl` | −0.0519 |
| scattered 0.12 | `denoise_stl` | −0.0527 |
| level_shift 0.05 | `denoise_stl` | −0.0598 |
| local_permutation 0.05 | `denoise_stl` | −0.0648 |
| spike 0.01 | `denoise_stl` | −0.1765 |
| spike 0.03 | `denoise_stl` | −0.2274 |

**Two distinct winners across nine conditions. No fixed program wins everywhere.** `denoise_stl`
— the frozen `best_fixed` — is *actively harmful* on block 0.24 (+0.110), where `winsorize`
wins. **C1 holds ex-COVID**, on non-degenerate data, at honest magnitudes.

---

## 3. §3 re-attributed: the confound is real, but it is one-fifth the size and it is not the
main story

`TD_VERDICT.md` §3 was the verdict's most-emphasised caveat: smoothing buys 0.84 sMASE on
*defect-free* data, therefore *"a weak judge manufactures apparent value for smoothing"*, and
therefore the judge ladder must precede TTHA-0. **The direction is right. The attribution was
wrong, and the size was wrong by 16×.**

**93.9% of that 0.84 is COVID.** On the natural lane, `denoise_stl` vs Raw:

| roster | `denoise_stl` on natural |
| --- | --- |
| all 6 datasets | **−0.8449** |
| ex-COVID | **−0.0519** |

And the weak-judge hypothesis makes a falsifiable prediction that **fails**. If the mechanism
were "smoothing deletes a high-frequency component the ridge DLinear cannot fit", then *every*
smoother should help on clean data. Ex-COVID, three of four **harm** it:

| smoother | natural, ex-COVID | |
| --- | --- | --- |
| `denoise_stl` | **−0.0519** | helps |
| `denoise_median` | +0.0200 | **harms** |
| `denoise_wavelet` | +0.1083 | **harms** |
| `denoise_savgol` | **+0.2065** | **harms** |

Two smoothers moving in opposite directions on the same clean data is not a story about the
judge's frequency response. **The magnitude generator was COVID's sMASE denominator**, not the
judge. Both extremes of the natural lane — `denoise_stl` at −0.8449 and `winsorize` at +0.8449,
suspiciously equal and opposite — trace to **the same single series** (`3b127a51…`, raw sMASE
**1507.6**), which STL moves by −193 and `winsorize` by +51. The symmetry is a coincidence of
the fold weights, not a sign error.

**Corrected statement.** A weak-judge confound is still present and still unexplained: STL buys
−0.052 on data with no defect to repair, which is 2.6× ε and not nothing, and STL is the
program that wins six of the nine conditions. So **Task G (the judge ladder) is still
warranted** — but it is no longer true that *"a large part of the measured space is an artifact
of the judge"*. The bounded exposure is ≈0.05 sMASE against a 0.13 headroom. **Urgency
downgraded from blocking-and-urgent to warranted-and-scheduled.** It should still run before
TTHA-0, because it is cheap and because a −0.05 unexplained gain on defect-free data is a real
crack — but it is no longer plausible that it invalidates the premise wholesale.

---

## 4. Governance note on the `power.py` fix

`TD_VERDICT.md` §6 disclosed that I found and fixed a bug in preregistered code, and that the
fix ran in the **permissive** direction (SHELVE → LAUNCH) — the direction that deserves the most
suspicion. That disclosure stands. Adjudicating it here, on the record:

**The fix is correct.** The strict (buggy) reading marked a cell `diagnostic_unavailable`
whenever `mde_80 > ε`, *including cells that had cleanly detected a large effect*. That makes
`is_detectable` and `readable` contradict each other, and it contradicts prereg §6's own
criterion, which reads `|effect| > mde_80` — a cell satisfying that clause cannot coherently be
called unreadable for failing to resolve something 60× smaller. The corrected rule is the rule
the preregistration *states*; the buggy rule is internally incoherent. The defence is not that
the new answer is nicer.

**And it currently drives no behaviour.** Under the strict reading, 15/17 cells are unreadable
and TTHA-0 is shelved pending a better instrument. Under the corrected reading, TTHA-0 launches
but §3 (even at its corrected size) says the judge ladder goes first. **Both readings produce
the same next action.** The disagreement between them is therefore not yet load-bearing, and no
decision is being bought with the permissive fix.

**Formal restatement of the Gate result:**

> **conditional LAUNCH — execution deferred until after the judge ladder.**

That is the phrasing that should be used from here. "LAUNCH" unqualified overstates what the
evidence licenses.

---

## 5. Erratum — preregistration §0 asserts a false premise

Prereg §0 (`docs/superpowers/specs/2026-07-14-benchmark-v0_2-prereg.md`) states:

> The frozen spec **never specified the downstream training pool's scope.** […] A search of all
> three frozen documents for `shared model / per-config / training pool / one model / per
> dataset / joint` returns nothing on the subject.

**This is false.** `idea/Benchmark_v0_Forecast_Design.md:156`, in the fixed downstream training
protocol, states:

> `→ per-config 从头训练 shared downstream model（不逐序列、不跨 config 混训）`
> *("train the shared downstream model from scratch per-config — not per-series, and no
> cross-config mixed training")*

The clause is directly on point, and it is on the very subject §0 declares absent.

**Root cause: a document-set error, not a search-string error.** The search string *included*
`per-config` and would have matched. What failed is the corpus: the search covered
`docs/superpowers/specs/` (pipeline-design + amendment-1), where `per-config` genuinely has zero
hits. The v0 forecast design document lives in `idea/` and **was never in the searched set**.
Verified: `grep` for `per-config|混训` returns zero hits across `docs/superpowers/specs/` and
exactly one file in `idea/`.

**But the clause was never operationalized, so it was unenforceable.** `config_id` is declared in
that document's §1.2 registry schema (`broad_domain / dataset_id / config_id / entity_id /
series_uid`) — and it appears:

- **0 times** in the 373 emitted `series_registry.jsonl` rows,
- **0 times** in every `.py` file in the repository.

The field was specified and never implemented. No code could have honoured a clause keyed on an
identifier that does not exist in any schema.

**What `config` means, and what follows.** §1.2 places `config_id` *between* `dataset_id` and
`entity_id`, so a config is a sub-division of a dataset; and §1.3's pool table lists exactly one
config per dataset (`nn5_daily`, `covid_deaths`, `traffic_hourly`, …). In v0-core, therefore,
**config ≡ dataset**. Under that reading:

| version | training unit | revised judgement |
| --- | --- | --- |
| v0 | per `dataset × regime` | still **illegal**, on the independent §5 regime-leak ground. Finer than the prescribed unit, but does not itself mix across configs. |
| v0.1 | per role (all datasets pooled) | **violates the clause.** It mixes across configs, which `不跨 config 混训` forbids under any reasonable reading of `config`. Re-judged from the prereg's "legal but coupling". |
| v0.2 | per `dataset` | **exactly the prescribed unit.** |

**Honest accounting of what this does and does not mean.** The v0.2 training-unit decision is
unchanged; it turns out to have had a contractual leg the preregistration did not know it had.
**This is not a vindication.** The defect is that a preregistration — the one artifact whose
entire value is that it was written before the numbers — **asserted a false premise about the
frozen corpus**, and did so while explicitly congratulating itself for correcting an earlier
false premise on the same subject. "Declaration ≠ execution" is this project's recurring failure
family, and §0 is a fresh instance of it: a search was *declared* exhaustive that was not
*executed* exhaustively.

**Timing, stated plainly.** This erratum was written *after* the results were read. The
pre-results window is gone and cannot be recovered. It is admissible only because it touches no
decision rule, no threshold, and no number: it corrects a factual claim about which documents say
what. It should not be read as preregistered.

**One thing it strengthens:** the v0.1 ↔ v0.2 incomparability notice. v0.1's numbers are not
merely produced under a different-but-legal training unit — they were produced under a unit that
**violates the design document's explicit prohibition on cross-config mixed training**. The
existing "v0.1 numbers are NOT comparable to v0.2" statement in `benchmark/__init__.py` holds a
fortiori.

---

## 6. What stands, and what is retracted

**Stands:**

- The Gate passes. Headroom is material (+0.1319 ex-COVID, 6.6× ε) and readable (6 non-COVID
  cells, from 4 datasets in 3 domains).
- **C1 survives at honest magnitudes**: two distinct argmins across nine conditions ex-COVID;
  `best_fixed` is actively harmful where `winsorize` wins. Conditioning buys **4.5%** over the
  best single program.
- The instrument works: 0 of 54 mechanism cells are dead (v0.1: 5 of 9 lanes dead).
- The retrained oracle was necessary — the untrained counterfactual still loses to `best_fixed`.
- TB's H_ref attribution, which this analysis independently reconfirms (H_ref ≡ Raw ex-COVID to
  7 decimals).

**Retracted or corrected:**

| # | claim in `TD_VERDICT.md` | status |
| --- | --- | --- |
| §1 | winner's curse (+0.2471) exceeds the honest conditioning gain | **retracted** — ex-COVID it is +0.0084, below ε |
| §1 | headroom +0.6934 / C1 space +0.1903 | **corrected** — +0.1319 / +0.0419 |
| §1 | "H_ref is net worse than Raw" as a standalone ladder result | **corrected** — ≈0 outside COVID |
| §2 | `winsorize` harms clean data | **retracted** — it mildly helps |
| §2 | `forward_fill` is catastrophic under heavy missingness | **retracted** — it does nothing (+0.001) |
| §3 | "a large part of the measured space is a judge artifact" | **corrected** — bounded at ≈0.05 of 0.13; 93.9% of the headline 0.84 was COVID |
| §6 | Gate result = "LAUNCH" | **restated** — "conditional LAUNCH, deferred until after the judge ladder" |

**Next action is unchanged: the judge ladder (Task G) runs before TTHA-0.** Its justification is
now narrower — a bounded −0.05 unexplained gain on defect-free data, rather than a wholesale
threat to the premise — but it is cheap, and the crack is real.

**A standing methodological note for every future report from this benchmark:** the macro fold
is not robust to a degenerate sMASE denominator. `monash:covid_deaths` will inflate any
magnitude it touches by roughly 4–6×. Every headline number this benchmark emits should be
reported **with and without COVID**, or as a **relative** improvement against a non-degenerate
floor. Absolute sMASE deltas folded across all six datasets are not interpretable quantities.
