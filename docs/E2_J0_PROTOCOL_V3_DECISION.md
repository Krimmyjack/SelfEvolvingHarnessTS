# E2-J0 protocol-v3 decision memo

## Project context

The project studies whether a receipt-grounded time-series data-preparation Harness can
turn Source-dataset interventions into scope- and risk-aware capabilities that reduce
adaptation regret on unseen Target datasets under the same feedback budget. The intended
causal chain is:

```text
contractual correctness
  -> Judge readability
  -> intervention headroom
  -> observable ApplicabilityWitness selectivity
  -> fresh Source promotion
  -> A3/A4/A5 Target adaptation
```

Natural E2 work has not yet promoted a Capability. Periodic-missing and outlier families
failed local/program or cross-dataset gates. Provenance-key rebind exactly repaired a
structural defect, but downstream utility conflicted between Traffic and COVID; it was
compiled to `ABSTAIN_DO_NOT_REGISTER`. UCI Target Query remains unopened.

## What E2-J0 tested

J0 isolated `Judge readability` from Pattern, Witness, Memory and transfer. It reused two
previously exposed Source rosters only for calibration:

- Traffic Hourly: 12 train series and 8 group-disjoint eval series;
- FRED-MD: 12 train series and 8 group-disjoint eval series.

The fixed Consumer was `Ridge(alpha=1, solver=SVD)`. The primary loss was per-series
sMASE computed in original units. In 4/7/14 of 72 training rows, the frozen corruption
added `+2` standardized units to target indices `[18,30)`. Three row-selection seeds were
averaged within each eval UID. Exact-Repaired copied the clean target block back exactly.
P0 and four Clean/Exact controls had to pass before eighteen corruption fits were allowed.

## Real result

| Dataset | Mean delta sMASE at d05/d10/d20 | Endpoint positive UIDs | q05 | MDE80 |
|---|---:|---:|---:|---:|
| Traffic | 0.175126 / 0.250154 / 0.328540 | 8/8 | 0.177720 | 0.227442 |
| FRED-MD | 0.120937 / 0.289602 / 0.501220 | 7/8 | 0.248500 | 0.381644 |

Both response sub-gates passed and the aggregate dose curves were nondecreasing. The
pre-registered absolute sMASE resolution gate required `MDE80 <= 0.02`; both datasets
failed. The immutable classification is therefore:

```text
READABLE_AT_INJECTED_DOSE_BUT_UNDERPOWERED_FOR_EPSILON
```

The run used exactly 22 fits. Support-B, UCI, Target and Query values were not read.

The positive statement is narrow: Ridge+sMASE reads this strong synthetic corruption on
these exposed Source rosters after seed/UID aggregation. It is not true for every seed or
every series, and it does not generalize to pooled original-unit MAE. There is no
Capability, Pattern/Witness, Memory, promotion or transfer evidence.

## Why adding a few samples cannot repair the old gate

Under the observed variance and the same one-sided design, reaching absolute sMASE
MDE80 `0.02` requires approximately 1,035 independent Traffic-like groups and 2,914
FRED-like groups. Available exposed Source groups are far below this. Seeds, anchors and
multiple origins from the same series are not independent cross-series units. A
development-only log-ratio diagnostic reduced scale heterogeneity but still had endpoint
MDE-like values around 0.116 for Traffic and 0.260 for FRED; it is not a hidden fix.

## Decision options

### A. Strong-effect readability instrument (recommended)

Keep the original J0 verdict as FAIL, but define protocol-v3 prospectively: J0 proves
only that the Judge reads strong, direct interventions. Remove the universal absolute
`0.02` MDE requirement from the *instrument* gate. Each future Capability family must
instead pre-register its own oracle-headroom minimum, paired confidence rule, harm rule
and abstention behavior before a deployable Witness is tested. Small effects that the
available independent groups cannot resolve remain unavailable, not negative.

This is the fastest route to testing the actual Harness mechanism without pretending the
old power gate passed.

### B. New dimensionless effect protocol

Pre-register a paired relative/log sMASE effect or dose-response estimand, define a new
material threshold in that scale, and validate it once on different exposed Source
calibration data. Current J0 numbers may motivate the design but cannot validate it.
Because the development-only log diagnostic did not remove most variance, this option
needs a concrete statistical reason beyond choosing the metric with the nicest result.

### C. Preserve absolute epsilon=0.02

If absolute sMASE resolution `0.02` is scientifically non-negotiable, stop the present
Source campaign until a substantially larger independent calibration population exists.
Continuing to add Witness fields, seeds, or operators would not solve the measurement
constraint.

## Proposed next experiment if A is accepted

Use a non-integrity training-data intervention with a shorter causal path to the shared
Consumer. A practical positive-control family is **target-reliability weighting**:

1. create a frozen, graded training-target corruption on Source data;
2. use an oracle reliability weight only to establish P1 headroom;
3. build a deployment-visible Witness from training-side temporal consistency, without
   clean Query future;
4. compare conservative downweighting against Identity and matched genuine-event risks;
5. only after two-Source P1/P2 success, allocate a new fresh promotion cohort.

The existing series/origin geometry is already uniform (12 series x 6 anchors), so plain
inverse-multiplicity weighting has no natural headroom unless a multiplicity defect is
explicitly injected; it should be treated as a calibration control rather than the
headline natural Capability.

## Questions for external review

1. Is option A scientifically defensible if the old `epsilon=0.02` verdict remains an
   explicit failure and future capabilities use prospectively frozen family-specific
   headroom, confidence and harm gates?
2. If a universal resolution criterion is still required, what effect scale and
   independently estimable material threshold would be preferable to absolute sMASE
   `0.02`, given the observed cross-series heterogeneity?
3. Is target-reliability weighting the best next non-integrity positive-control family,
   or would censoring/saturation recovery provide a better bridge to the eventual
   Pattern/Witness transfer claim?
4. What minimum evidence should authorize moving from this strong-effect Judge
   calibration to P1 without turning the project into a conventional numerical router?

Until one option is frozen, no new Capability fit, Source promotion, A3/A4/A5 run or UCI
Target Query opening is authorized.
