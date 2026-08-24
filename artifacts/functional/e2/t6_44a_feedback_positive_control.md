# #44a -- AD feedback positive control

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  a development positive control on injected NOAA copies; its conclusions never enter a natural-Yahoo capability claim and say nothing about whether natural data carries removable contamination

## Verdict

- **PROGRAM_CONSUMER_LAYER_FAULT**
- routing: no rate produced a readable repair effect, so the first fault is at the Program/Consumer layer; this is NOT an injection failure: the contamination did change the fitted object (median std inflation 1.076x) and the oracle mask did restore it (median std residual 1.001x), so the exam's mechanics worked and the delayed reading still did not move; the delayed estimand is the suspect: the Consumer flags a median 88% of the Qcal region even in the clean arm, so its event F1 is decided by where a near-continuous flag run breaks rather than by what it learned from the substrate
- B1 by rate: {'r05': 'EFFECT_NOT_CONFIRMED', 'r15': 'EFFECT_NOT_CONFIRMED'}
- predictive Support signals: none evaluated

## Geometry, as understood and verified

- series length 8760 (hourly year), inside the 17520 wall; 12 NOAA stations from the T1 injection family.
- training substrate / P's action region: [120, 900]
- Qcal (the only region scored here): [2600, 3060], four known events per station.
- Qf: [2100, 2560] -- **not opened by this book**, kept as an independent confirmation surface.
- Support window: [744, 900].  T1's own triple-window geometry (support origins 1104/1152/1200, delayed 1248/1296/1344) is a forecasting origin geometry sited outside the training block and carries no AD event labels, so it cannot serve as the AD Support window; the book's stated fallback -- the last 20% of the training region -- is used
- substrate choice: the qcal copy is the substrate: its training block is pristine (byte-identical to the qf copy there, and the t1 copy differs from it exactly at the union of the t1 and qcal ledger points), and it already carries the four known Qcal events this book scores against
- verification: all 12 stations pass -- the t1 copy differs from the qcal copy exactly at the union of their ledger points, the qf and qcal copies are byte-identical on the training block, and no injected event point is missing.

## Three arms x two rates: Qcal macro event F1

| rate | spikes | window rate | point rate | clean | contaminated | repaired |
|---|---|---|---|---|---|---|
| r05 | 2 | 0.0526 | 0.0026 | 0.4354 | 0.4023 | 0.4110 |
| r15 | 6 | 0.1577 | 0.0077 | 0.4354 | 0.4359 | 0.4293 |

## B1: did contamination hurt, and did the oracle repair recover it?

| rate | harm (contaminated − clean) | recovery (repaired − contaminated) | harmed | worst | recovered share of loss | verdict |
|---|---|---|---|---|---|---|
| r05 | -0.033142 | +0.008714 | 4 | -0.1538 | 0.263 | EFFECT_NOT_CONFIRMED |
| r15 | +0.000516 | -0.006676 | 7 | -0.1714 | n/a | EFFECT_NOT_CONFIRMED |

gate: macro Δ ≥ +0.005, harmed ≤ 1/12, worst ≥ -0.020

## Per-station Qcal event F1

### rate r05

| station | clean | contaminated | repaired | repair − contaminated | contaminated − clean |
|---|---|---|---|---|---|
| 72203812897 | 0.4000 | 0.4211 | 0.4444 | +0.0234 | +0.0211 |
| 72259003927 | 0.3333 | 0.3333 | 0.3333 | +0.0000 | +0.0000 |
| 72329003935 | 0.2500 | 0.2222 | 0.2857 | +0.0635 | -0.0278 |
| 72422093820 | 0.3750 | 0.2667 | 0.3333 | +0.0667 | -0.1083 |
| 72435653866 | 0.7273 | 0.8000 | 0.7273 | -0.0727 | +0.0727 |
| 72438093819 | 0.6667 | 0.5000 | 0.6000 | +0.1000 | -0.1667 |
| 72511654737 | 0.6667 | 0.5714 | 0.6667 | +0.0952 | -0.0952 |
| 72529014768 | 0.4444 | 0.3333 | 0.5000 | +0.1667 | -0.1111 |
| 72605654791 | 0.4000 | 0.4000 | 0.4000 | +0.0000 | +0.0000 |
| 72654014936 | 0.3333 | 0.4000 | 0.3333 | -0.0667 | +0.0667 |
| 72793494248 | 0.1667 | 0.1176 | 0.0000 | -0.1176 | -0.0490 |
| 74486514719 | 0.4615 | 0.4615 | 0.3077 | -0.1538 | +0.0000 |

### rate r15

| station | clean | contaminated | repaired | repair − contaminated | contaminated − clean |
|---|---|---|---|---|---|
| 72203812897 | 0.4000 | 0.2581 | 0.4211 | +0.1630 | -0.1419 |
| 72259003927 | 0.3333 | 0.5000 | 0.3333 | -0.1667 | +0.1667 |
| 72329003935 | 0.2500 | 0.2857 | 0.2222 | -0.0635 | +0.0357 |
| 72422093820 | 0.3750 | 0.5455 | 0.4615 | -0.0839 | +0.1705 |
| 72435653866 | 0.7273 | 0.6667 | 0.6154 | -0.0513 | -0.0606 |
| 72438093819 | 0.6667 | 0.5455 | 0.5455 | +0.0000 | -0.1212 |
| 72511654737 | 0.6667 | 0.7500 | 0.6667 | -0.0833 | +0.0833 |
| 72529014768 | 0.4444 | 0.2000 | 0.5714 | +0.3714 | -0.2444 |
| 72605654791 | 0.4000 | 0.5714 | 0.4000 | -0.1714 | +0.1714 |
| 72654014936 | 0.3333 | 0.4000 | 0.2857 | -0.1143 | +0.0667 |
| 72793494248 | 0.1667 | 0.1333 | 0.1667 | +0.0333 | -0.0333 |
| 74486514719 | 0.4615 | 0.3750 | 0.4615 | +0.0865 | -0.0865 |

## B2: not reached

B1 did not confirm an effect at either rate, so the Support signal stream was not run -- the book stops the round at the Program/Consumer layer rather than hunting for a signal that predicts an effect that is not there.

## Did the contamination reach the fit, and did the mask undo it?

| rate | median std inflation | max | median std residual after repair | reached fit | repair restored |
|---|---|---|---|---|---|
| r05 | 1.0221 | 1.1372 | 1.0007 | True | True |
| r15 | 1.1298 | 1.2743 | 1.0014 | True | True |

## Can the delayed estimand resolve anything? (Qcal flag saturation)

| rate | median flagged share (clean arm) | min | max | series with ≥50% flagged |
|---|---|---|---|---|
| r05 | 0.877 | 0.306 | 1.000 | 11/12 |
| r15 | 0.877 | 0.306 | 1.000 | 11/12 |

## Determinism

- substrate: two independent constructions byte-identical: **True**
- model/reading recheck on 2 stations x 2 rates x 3 arms identical: **True**
- adapter equals the frozen Consumer when nothing is masked: **ADAPTER_EQUALS_FROZEN_CONSUMER**

## Budget

- LLM: 0; AD fits: 74 / 100
- by arm: instrument_check=4, clean=12, contaminated_identity=24, contaminated_repaired=24, determinism_recheck=10

## Obligation self-report

- adapter_equivalence: ADAPTER_EQUALS_FROZEN_CONSUMER
- data_directory_opened: False
- fit_budget_cap: 100
- fit_budget_respected: True
- fit_budget_used: 74
- fits_by_arm: {'instrument_check': 4, 'clean': 12, 'contaminated_identity': 24, 'contaminated_repaired': 24, 'determinism_recheck': 10}
- frozen_injection_tree_written: False
- llm_calls: 0
- model_determinism: True
- nab_smd_beyond_17520_reads: 0
- noaa_2025_new_reads: 0
- qf_opened: False
- rate_cherry_picking: False
- rates_reported: ['r05', 'r15']
- substrate_determinism: True
- yahoo_reads: 0
- yahoo_sealed_41_reads: 0

---

*Everything above this line is generated by the runner from the artifact.
Everything below it is the executor's written reading, appended by hand.*

## Geometry, as I understood it before building

The T1 family is three independent injections over one pristine NOAA year
(8760 hourly points, comfortably inside the 17520 wall), not a chain:

- `t1/` carries training-block events in [431, 900);
- `t1b_query/qcal/` carries Query events in [2600, 3060);
- `t1b_query/qf/` carries Query events in [2100, 2560).

Each copy is pristine everywhere except its own region.  I verified rather
than assumed this: for all 12 stations, the t1 copy differs from the qcal
copy at exactly the union of the two ledgers' point sets, the qf copy
differs from the qcal copy at exactly the union of *its* ledger with qcal's,
and the qcal and qf copies are byte-identical across the whole training
block.  That is what licenses the choice of substrate: **the qcal copy has a
pristine training block and already carries the four known Qcal events**, so
it is the one array that lets me inject my own training-side contamination
and still score against a known answer.

Three boundaries therefore govern this book: the training substrate
[120, 900) is the only region any arm may alter; Qcal [2600, 3060) is the
delayed scoring region and its bytes are never touched; Qf [2100, 2560) is
left closed as an independent confirmation surface for a later book.  Note
that Qf sits *before* Qcal in index order — the two Query regions are not
"near" and "far", they are two disjoint windows, both roughly 1200–2200
hours after the fit block.

T1 does define a support/delayed geometry, but it is a *forecasting* origin
triple (support 1104/1152/1200, delayed 1248/1296/1344, horizon 48) sited
outside the training block and carrying no AD event labels.  It cannot serve
as an AD Support window, so the book's stated fallback applies: the last 20%
of the training region, [744, 900).

## Executor's reading

### What happened

Question (A) — can removing training-side contamination produce a measurable
delayed effect under this framework — is answered **no, not under this
Consumer on this Query region**.  Question (B) was not reached, exactly as
the book instructs: B1 failed at both rates, so the Support signal stream
was not run.

The two rates disagree in a way that is itself the finding:

- at **r05** (2 spikes, 5.3% of training windows) the contamination *did*
  hurt — macro −0.0331, which clears the readability line — and the oracle
  repair recovered **26% of the loss**, macro +0.0087.  That clears the
  gate's macro bar.  It fails the two safety conditions badly: 4 of 12
  stations are harmed by the repair (bar is 1) and the worst station loses
  −0.154 (floor is −0.02);
- at **r15** (6 spikes, 15.8% of windows) the contamination did **not hurt
  at all** — macro +0.0005 — despite three times the dose and a larger
  measured effect on the fit.  The repair then reads −0.0067.

Three times the contamination doing less damage is not a dose response.  It
is what a noise-dominated estimand looks like.

### This is not an injection failure, and that is the important part

Every mechanical link in the exam is verified working:

- the contamination reached the object the Consumer actually fits: the
  standardization scale inflates by a median 1.022x at r05 and 1.130x at
  r15, up to 1.274x on individual stations;
- the oracle mask undid it: the repaired arm's scale lands within 0.1% of
  the clean arm's (median residual 1.0007x / 1.0014x).  The mask removes
  exactly the intended windows — 40 at r05 and 120 at r15 per station, i.e.
  20 per spike, as designed;
- the injection is not too weak: the median injected magnitude is **4.5
  times the training block's own standard deviation** (range 2.9x–8.6x).
  These are large spikes by any reading;
- the whole path is deterministic (substrate byte-identical across two
  independent constructions; models and readings identical on the recheck),
  and the adapter provably reduces to the frozen Consumer bitwise when
  nothing is masked.

So the exam did what it was built to do, and the delayed reading still did
not move coherently.  That routes the fault past the injection and past the
repair, onto the Consumer and its estimand.

### The specific first fault

**The in-service AD Consumer is saturated on this Query region.**  Fitted on
[120, 900) and scored at [2600, 3060), it flags a median **88%** of the
Query — 11 of 12 stations flag more than half of it, and one station flags
100% of it.  When a detector alarms on almost everything, `merge_events`
collapses the region into a handful of runs and the event F1 is decided by
where a near-continuous flag run happens to break, not by what the detector
learned from its training substrate.  That is why per-station deltas swing
by ±0.17 to ±0.37 while the macro effects being tested are ±0.03: the
per-station noise is roughly ten times the effect.

In the AGENTS.md ladder this is `no readable positive effect → Consumer /
evaluator / training protocol`, and within that, the *evaluator* end: the
delayed estimand cannot resolve a training-side effect here, so no amount of
Program work would show up.  This is a different diagnosis from "IForest is
blind to point spikes", and the two should not be conflated — the r05 harm
reading (−0.0331) shows the Consumer does respond to training contamination;
what fails is the ability to read that response through a saturated
event-F1.

### A defect I introduced, found, and fixed before the final reading

The pre-registered nesting rule was "positions are drawn once for the larger
rate; the smaller rate takes the ordered prefix", where *ordered* means draw
order, so that both rates sample positions uniformly over the block.  My
first implementation sorted the accepted positions by index before storing
them, so r05 systematically took the **two earliest positions in the block**
rather than a uniform sample.

I did not spot this by inspection.  I spotted it because the first run left
all 12 Support windows with zero injected events, which is about a 1-in-120
outcome under uniform placement — implausible enough to check.  The
implementation was corrected to keep draw order and sort only for
application.

The correction changes r05 and must be reported as changing it: r05's
contamination harm went from +0.016 (the buggy placement made contamination
look *helpful*) to −0.033.  It does not change r15, where all six drawn
positions are used and order is irrelevant — and indeed the r15 macro
figures are bitwise identical across the two executions, which is an
independent confirmation that the defect was confined to the prefix rule.
The verdict is unchanged by the fix: both rates failed B1 before and after.

## Out-of-book findings (reported, not fixed)

1. **The frozen AD Consumer cannot run on this data at all without an
   abstention rule.**  Five of the twelve stations carry missing points in
   the training block (up to 40 windows' worth) and five carry them inside
   Qcal; scikit-learn refuses to fit or score a NaN row, and
   `aegists_iforest_v1` has no abstention semantics.  The runner supplied
   the project's existing T0/v3 canon at the adapter layer (never fit,
   forced not to flag, excluded from the AUPRC ranking) rather than editing
   a Consumer that is in service for the Yahoo line.  Any future book that
   puts this Consumer on real-world data outside Yahoo needs that rule as a
   first-class Consumer feature, with its own gate.  Not fixed here.

2. **Query saturation is invisible on the Yahoo geometry and severe on
   this one.**  On Yahoo the eval region begins immediately where the
   training block ends, so the fit stays roughly in calibration; here the
   scored region starts about 1700 hours later, across seasonal drift, and
   the same frozen Consumer alarms on 88% of it.  Every reading the AD line
   has taken with this Consumer has been taken under the first geometry.
   This is a scope condition on the Consumer that was not previously on the
   record.

3. **Support-window starvation is a design trap at low contamination
   rates.**  At r05 only 4 of 12 stations end up with any injected event in
   the [744, 900) Support window; the other 8 can supply false-alarm
   evidence only.  Had B1 passed at r05, B2's "≥70% of event-bearing
   series" test would have run on four series, which is not a test.  A
   future book must size the injection against the *Support window*, not
   only against the fit region — or site the Support window where the
   contamination is.

4. **The robust-z detector appears far better calibrated for this data than
   the in-service IForest.**  The ledger records the T1b line's oracle event
   F1 at roughly 0.7 on this family; the IForest's clean-arm macro here is
   0.435, with the saturation described above.  If a later book wants a
   working AD positive control on NOAA, the Consumer choice — not the
   injection — is the variable to reconsider first.  Only reported; nothing
   was swapped in mid-round.

5. **`r15` contamination raised macro F1 on 6 of 12 stations.**  Inflating
   the standardization scale shrinks the standardized Query, which flags
   fewer points, which can accidentally raise precision on a saturated
   detector.  This is the mechanism that makes "more contamination is
   harmless" possible, and it is a reason to distrust any AD utility reading
   taken while the detector is saturated — including, potentially, some of
   the historical readings on other geometries.  Flagged for the mainline,
   not investigated here.

## Executor's obligation self-report (beyond the machine section)

- Executions: two full runs (74 fits each; the first under the placement
  defect described above, the second after the fix) plus one
  `--verify-geometry` pass that spends no fit.  Only the corrected run's
  artifact is delivered.  The budget cap is per run; 74 of 100.
- B2 was **not** run, by rule, because B1 confirmed nothing.  No Support
  signal is reported as predictive or unpredictive — the round simply does
  not know.
- Both pre-registered rates are reported in full; neither was dropped and
  no third rate was added after seeing the readings.
- The clean arm is called a **reference** throughout, never an upper bound,
  per the short canon.  As it happens the reading would not have supported
  the word "bound" anyway: contamination raised macro F1 above clean at r15.
- Injection positions and the clean reference are evaluator-side only.  No
  Agent ran this book (0 LLM), so the question of them leaking into an
  Observation does not arise, but the runner keeps them out of every Support
  feature it computes.
- Scratch isolation: substrates and the injection ledger are written under
  `_scratch/phase_t/m44a_v1/` with an explicit guard that refuses to write
  inside `t1/`, `t1b_query/qcal/` or `t1b_query/qf/`.  The frozen T1 copies
  are byte-unchanged.
- Zero LLM calls, zero network access, zero reads under `data/`, zero reads
  of the 41 sealed Yahoo series, zero new NOAA 2025 / NAB / SMD /
  beyond_17520 reads.  Qf was declared and left closed.
- This report is **not** committed, per Part C.  The runner is committed.
- Part 0 note: the two ledger docs I was asked to commit
  (`ROADMAP_POST_V1_2026-08-22.md`,
  `STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`) already carried the
  mainline's own #43/#44a entries in the working tree.  I committed that
  text as authored rather than rewriting it, and left `AGENTS.md`,
  `README.md`, `PROJECT_STATE_AND_DATA_MAP_2026-08-23.md` and
  `SUCCESSOR_BRIEF_2026-08-22.md` uncommitted, since they carry other
  in-flight edits and were not on this book's allowlist.

