# benchmark-v0.2 — Dev-Query verdict

> **⚠ READ `TD_VERDICT_ADDENDUM.md` ALONGSIDE THIS DOCUMENT.**
> The verdict below is directionally correct but its **magnitudes are inflated 4–6×** by two
> degenerate `monash:covid_deaths` cells. The addendum recomputes every figure with and without
> COVID, **retracts two of §2's exemplars and §1's winner's-curse claim**, re-attributes §3's
> confound (93.9% of it was COVID, not the judge), and restates the Gate result as
> **"conditional LAUNCH — deferred until after the judge ladder."** The Gate itself does not
> flip. Quote the addendum's numbers, not this document's.

Final-Query was **not** read. It remains sealed.

Preregistered before any of these numbers existed:
`docs/superpowers/specs/2026-07-14-benchmark-v0_2-prereg.md`.

---

## Verdict

**The Gate passes. TTHA-0 launches** — with one caveat that changes what may be claimed
(§3), and one correction to the preregistered code that must be disclosed (§6).

The instrument now works. In v0.1, five of the nine corruption cells were dead: every
program in the pool scored identically to four decimal places, because every program filled
missing values and none could touch an outlier, a noise floor, a break, or a shuffle. C1 was
untestable against that pool. Under the v0.2 mechanism-covering pool, **0 of 54
(dataset × scenario × dose) cells are dead**, and the spread between programs on the spike
lane alone reaches 3.2 sMASE.

---

## 1. The ladder (headline macro fold)

| role | baseline | sMASE |
| --- | --- | --- |
| floor | Raw (No-op + canonical ingestion) | 11.4815 |
| floor | **best_fixed = `denoise_stl`** | 10.9784 |
| incumbent | H_ref | **11.5583 — worse than doing nothing** |
| **CEILING (Gate)** | **`oracle_transfer_retrained`** | **10.7881** |
| envelope | `oracle_insample_retrained` | 10.5410 |
| descriptive only | untrained counterfactual (transfer) | 10.9861 |

- **Headroom, Raw → ceiling: +0.6934.** Confounded — see §3. Do not report this as the value
  of data readiness.
- **C1 conditioning space, best_fixed → ceiling: +0.1903.** This is the clean number.
- **Winner's curse (envelope − ceiling): +0.2471.**

Three things in that table are worth stopping on.

**The winner's curse is larger than the entire honest conditioning gain.** Selecting the
policy on the query set inflates the ceiling by +0.247, while the true conditioning space is
+0.190. Any oracle selected in-sample — which is what v0 and v0.1 reported — would have more
than doubled the apparent value of conditioning. The transfer oracle is not a nicety.

**The untrained counterfactual oracle (10.9861) loses to `best_fixed` (10.9784).** An
"oracle" that scores worse than a fixed program is the signature of a world no model was
fitted to: it picks a program per cell, but reads each cell's loss off a model trained on a
corpus prepared with one program throughout. Retraining the oracle was necessary, not
fastidious.

**H_ref is net worse than Raw** (+0.077), and on the heavy-missing lane it is worse by
+0.939. See the separate H_ref attribution (`h_ref_self_harm_diagnosis.json`): the damage is
one COVID series, and outside COVID H_ref's self-harm is exactly 0.0000 because it selects
`impute_linear` — which *is* canonical ingestion — on 329 of 373 series. H_ref and Raw are
not two independent baselines.

---

## 2. C1, shown in raw program means

No oracle, no aggregation. The same operator flips sign across conditions:

| program | natural | block 0.24 | spike 0.03 |
| --- | --- | --- | --- |
| `winsorize` | **+0.845 harm** | +1.354 harm | **−1.195 help** |
| `forward_fill` | −0.000 | **+1.259 harm** | −0.000 |
| `denoise_stl` | −0.845 help | **+0.779 harm** | −3.189 help |
| `denoise_median` | −0.029 | +0.166 harm | **−2.872 help** |

*(mean sMASE minus Raw; negative = better than doing nothing)*

**No fixed program wins everywhere.** `winsorize` harms clean data and helps spiked data.
`forward_fill` helps light missingness (block 0.12: −0.062) and is catastrophic under heavy
missingness (block 0.24: +1.259). Even `best_fixed` — `denoise_stl` — is actively harmful on
block 0.24.

This is the C1 claim, and it is not touched by the confound in §3, because it is a statement
about *which* program to run, not about whether running one helps on average.

---

## 3. THE CAVEAT — most of the gain over Raw is not defect repair

On the Natural lane, **359 of 373 series have zero native missingness** (roster mean NaN
fraction: 0.08%). There is nothing to repair. And yet:

| program | mechanism | vs Raw on Natural |
| --- | --- | --- |
| `forward_fill` | missing | −0.0003 |
| `seasonal_fill` | missing | −0.0001 |
| **`denoise_stl`** | additive noise | **−0.8449** |
| `denoise_savgol` | additive noise | −0.5515 |
| `winsorize` | outlier | +0.8449 |

The fill programs correctly do nothing, because there is nothing to fill. But smoothing buys
0.84 sMASE **on clean data with no injected defect**.

That is not data readiness. It is **smoothing acting as regularization for a weak judge**.
The frozen closed-form judge is a ridge DLinear mapping 48 → 48; it cannot fit the
high-frequency component, so deleting that component improves it. The project already knows
the dual of this — *a weak judge masks the value of data-quality improvement* — and this is
the inverse: **a weak judge manufactures apparent value for smoothing.**

Consequences:

- **`gain_over_raw` must not be reported as the value of data preparation.** The headline
  +0.6934 is substantially this artifact.
- **`gain over best_fixed` (+0.1903) is the defensible C1 number**, because the floor already
  smooths — the artifact is inside the floor and cancels.
- **Every conclusion here is conditional on the closed-form judge** and must be re-checked
  under Adam-DLinear, LSTM-scratch, and Chronos. If the smoothing gain collapses under a
  judge that can fit high frequencies, then the pool's additive-noise mechanism is measuring
  the judge, not the data. **This is now the highest-value next experiment**, ahead of
  TTHA-0.

---

## 4. Detectability (cluster bootstrap over overlap groups)

**6 of 17 cells are readable and material.** The resampling unit is the overlap group, not
the series.

| cell | n | clusters | effect | mde_80 | status |
| --- | --- | --- | --- | --- | --- |
| `monash:traffic_hourly \| seasonal_high` | 140 | **140** | +0.0981 | 0.0142 | **readable** |
| `uci_electricity \| seasonal_high` | 60 | **60** | +0.1317 | 0.0566 | **readable** |
| `metr_la \| structured_mixed` | 12 | 5 | +0.3701 | 0.1599 | readable |
| `metr_la \| low_structure` | 41 | **5** | +0.3138 | 0.2944 | readable (marginal) |
| `monash:traffic_hourly \| structured_mixed` | 35 | 35 | +0.0578 | 0.0536 | readable (marginal) |
| `gefcom2012_load \| seasonal_high` | 2 | 2 | +0.1164 | 0.0083 | readable (thin; supplementary) |
| `monash:covid_deaths \| trend_high` | 43 | 43 | +6.3545 | **7.5223** | **not detected** + scale warning |
| `monash:covid_deaths \| seasonal_high` | 6 | 6 | +3.6172 | **4.0134** | **not detected** + scale warning |

The two solid cells are `traffic_hourly|seasonal_high` (140 independent clusters, effect 7×
the detection limit) and `uci|seasonal_high` (60 clusters). **They are in different domains
and different datasets**, which is the closest thing this roster offers to independent
confirmation.

**COVID is excluded by its own power, not by hand.** Its effects are enormous (+6.35, +3.62)
but its MDE is larger still, and it carries a `scale_warning` — its sMASE denominator is a
tiny seasonal difference of a monotone cumulative count. The verdict does not rest on it.
This is what the panel is for.

**METR-LA shows why the cluster bootstrap mattered.** `metr_la|low_structure` has 41 series
but only **5 spatial blocks**. An IID bootstrap would have reported an interval roughly
`sqrt(41/5) ≈ 2.9×` too narrow and made a marginal cell look decisive.

---

## 5. Natural vs Controlled, judged separately (prereg §6)

| lane | Raw | best_fixed | ceiling | gain over Raw | C1 gain over best_fixed | readable+material cells | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| natural | 18.2770 | 17.4320 | 17.3687 | +0.9082 | **+0.0633** (mde 0.0371) | 5/17 | **headroom** |
| controlled_v0 | 13.5062 | 13.4263 | 13.0539 | +0.4523 | **+0.3724** (mde 0.3239) | 4/17 | **headroom** |
| controlled_v0_1 | 16.9496 | 15.8656 | 15.7676 | +1.1820 | +0.0980 (mde 0.1932) | 7/17 | **headroom** |

**Quadrant: Natural has headroom → real-world data preparation has value** (the strongest
conclusion the prereg allows) — subject entirely to §3, which says a large share of the
"value" on the Natural lane is the judge's weakness, not the data's condition.

C1 conditioning is **measurable on natural and controlled_v0**; on controlled_v0_1 the
conditioning gain (+0.098) sits below its detection limit (0.193).

---

## 6. Disclosure — a bug in the preregistered code, fixed in the permissive direction

`power.py` applied `diagnostic_unavailable` in **both** directions: a cell was marked
unreadable whenever `mde_80 > ε`, even when it had cleanly detected a large effect. That made
`is_detectable` and `readable` contradict each other (`controlled_v0_1`: effect +1.18, CI90
[+0.67, +1.84] excluding zero, marked "unreadable" for being unable to see 0.02).

The preregistration's own rationale is explicit that the rule guards the **null** direction
only:

> "A `diagnostic_unavailable` cell contributes nothing to a **saturation claim**. Absence of
> a detected effect is not evidence of absence when `mde_80 > ε`."

The code was corrected to match that rationale: the guard now fires only when **no effect was
detected**. This is disclosed prominently because **it flips the verdict from SHELVE to
LAUNCH**, and a post-hoc change in the permissive direction is exactly the change that
deserves the most suspicion. The defence is that the corrected rule is the one the
preregistration *states*, and that the buggy rule is internally incoherent — not that the
new answer is nicer.

The strict (buggy) reading is recorded for the reader who wants it: it marked 15/17 cells
unreadable and shelved TTHA-0.

---

## 7. What happens next

**TTHA-0 six-arm pilot: LAUNCH** — but headroom is a **necessary** condition, not a
sufficient one. It says an action difference exists to be chosen between. It says nothing
about whether a harness can learn to choose it. If TTHA-0 then fails, the failure is located
in selection, not in the benchmark.

**Ahead of TTHA-0, however: the judge ladder.** §3 says a large part of the measured space is
an artifact of a judge that cannot fit high frequencies. Running the pool under Adam-DLinear
and Chronos is cheaper than TTHA-0 and could invalidate a chunk of its premise. It should go
first.

---

### Artifacts

`dev_discrimination_report.json`, `dev_program_losses.jsonl` (one row per uid × program),
`dev_repeat_losses.jsonl` (the unfolded per-uid × scenario × dose × replicate measurements),
`dev_per_dose_report.json`, `baseline_report.md`, `h_ref_self_harm_diagnosis.json`,
`TD_VERDICT.json`, `program_pool.json`.

Runtime: 13,634 s (3.79 h), single-threaded. `denoise_stl` is ~99% of it.
