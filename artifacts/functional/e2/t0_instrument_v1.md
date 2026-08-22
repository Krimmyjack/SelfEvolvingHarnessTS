# T0: AD Consumer v1, the same-byte contract, and the calibrated substrate

**T0_READY.**  0 LLM calls, 0 forecasting retrains, 176 of 200 AD evaluations.

## Part B -- the same-byte contract

one injected block B, one Program P, one action geometry, exactly one P(B); the forecasting Consumer trains on P(B) and scores on the task's native future window, and the AD Consumer scores the injection ledger inside that same P(B).  No fork is permitted on the data-processing side.

- Processing-side boundary: the processing side is v6._apply_program and ends at its return value.  _center_scale, the stacking and the ridge are Consumer-internal representation, as the detector's own rolling median and MAD are; both act on the same P(B).
- Action geometry: P acts once per (train series, anchor) on the 240-point window raw[anchor-192 : anchor+48].  P(B) is unique at the window, which is the unit B names: the operators are window-local, so a series-level P(B) would not be well defined under overlapping anchors.
- Asymmetry declaration: the forecasting Consumer reads the window's last HORIZON points as a future to predict; the AD Consumer reads the block it is given and flags inside it.  That difference is the task semantics under test, not a geometric confound, and PROGRAM_GEOMETRY_UNALIGNED must not be called on it.

| check | result |
| --- | --- |
| P(B) comparisons | 600 |
| `np.array_equal` on every one | **True** |
| same bytes on a repeat call | True |
| identity == `_linear_integrity` baseline | True |
| detector flags invariant under `_center_scale` | True |
| forecasting retrains spent | 0 |

Verdict: **SAME_BYTE_CONTRACT_HOLDS**.

## Part D -- the calibration block and the frozen ledger

- Rule: the earliest contiguous development segment of the same length as an in-service Support/delayed triple-window that overlaps none of them.
- Literal earliest segment `[0, 288)` is **not executable**: at start=0 the first legal position is 25 and its frozen sigma_local prefix [t-168, t) begins at -143, off the front of the array.  The rule as written selects a segment on which one of its own constants cannot be evaluated.
- Taken: **`[143, 431)`** -- the earliest such segment on which every frozen constant is evaluable as written.  No other degree of freedom was used and the choice took no measured value into account.
- Zero overlap with the in-service triple-windows [[1104, 1392], [1800, 2088]]: True.
- Calibration seed 20260822; T1's seed 20260823 is reserved and not materialised by this book.

| series | events | skips | spacing rejections | indices |
| --- | ---: | ---: | ---: | --- |
| `72203812897` | 2 | 0 | 0 | 173, 394 |
| `72259003927` | 2 | 0 | 0 | 234, 318 |
| `72329003935` | 2 | 0 | 0 | 314, 373 |
| `72422093820` | 2 | 0 | 3 | 285, 335 |
| `72435653866` | 2 | 0 | 0 | 210, 291 |
| `72438093819` | 2 | 0 | 1 | 216, 338 |
| `72511654737` | 2 | 0 | 0 | 294, 393 |
| `72529014768` | 2 | 0 | 0 | 244, 294 |
| `72605654791` | 2 | 0 | 0 | 203, 303 |
| `72654014936` | 2 | 0 | 1 | 262, 386 |
| `72793494248` | 2 | 0 | 1 | 169, 247 |
| `74486514719` | 2 | 0 | 0 | 180, 374 |
| `99999903062` | 2 | 0 | 1 | 292, 388 |
| `99999904140` | 2 | 0 | 0 | 189, 297 |
| `99999923908` | 2 | 0 | 0 | 185, 287 |
| `99999963862` | 2 | 0 | 0 | 224, 369 |

Total 32 events, 0 skips, written to `_scratch/phase_t/injected`.

## Part C -- the acceptance

| condition | reading | pass |
| --- | --- | --- |
| (i) determinism | full path twice, identical | True |
| (ii) background alarm rate (twin block) | median 13.88888888888889 per 1000 points | reported, not gated |
| (iii) calibrated block | P 0.34444444444444444 / R 0.96875 / F1 0.5081967213114753 | True |

Frozen detector setting: window 49, threshold 3.5.  Fallback taken: True.

| series | ledger | predicted | matched | precision | recall | F1 | background /1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `72203812897` | 2 | 4 | 2 | 0.500 | 1.000 | 0.667 | 10.417 |
| `72259003927` | 2 | 5 | 2 | 0.400 | 1.000 | 0.571 | 10.417 |
| `72329003935` | 2 | 8 | 2 | 0.250 | 1.000 | 0.400 | 20.833 |
| `72422093820` | 2 | 4 | 2 | 0.500 | 1.000 | 0.667 | 6.944 |
| `72435653866` | 2 | 4 | 2 | 0.500 | 1.000 | 0.667 | 10.417 |
| `72438093819` | 2 | 5 | 1 | 0.200 | 0.500 | 0.286 | 13.889 |
| `72511654737` | 2 | 3 | 2 | 0.667 | 1.000 | 0.800 | 6.944 |
| `72529014768` | 2 | 7 | 2 | 0.286 | 1.000 | 0.444 | 17.361 |
| `72605654791` | 2 | 6 | 2 | 0.333 | 1.000 | 0.500 | 13.889 |
| `72654014936` | 2 | 10 | 2 | 0.200 | 1.000 | 0.333 | 27.778 |
| `72793494248` | 2 | 7 | 2 | 0.286 | 1.000 | 0.444 | 17.361 |
| `74486514719` | 2 | 3 | 2 | 0.667 | 1.000 | 0.800 | 3.472 |
| `99999903062` | 2 | 8 | 2 | 0.250 | 1.000 | 0.400 | 17.361 |
| `99999904140` | 2 | 5 | 2 | 0.400 | 1.000 | 0.571 | 6.944 |
| `99999923908` | 2 | 6 | 2 | 0.333 | 1.000 | 0.500 | 13.889 |
| `99999963862` | 2 | 5 | 2 | 0.400 | 1.000 | 0.571 | 13.889 |

the NOAA substrate carries unlabelled natural anomalies, so an alarm here may be correct.  This level bounds nothing and is reported as context for the precision figure below.

computed against the ledger only; a correct flag on an unlabelled natural anomaly counts in the denominator.

## What T0 hands to T1

### The injection-placement conflict (blocking)

- Measured training block **B = [120, 900)**.
- In-service triple-windows, context inclusive: [[912, 1392], [1608, 2088]].
- Intersection empty: **True**.

P acts only where the forecasting Consumer builds training rows, which is the train windows at _config()'s anchors -- [120, 900).  The eval side is read through _linear_integrity and the truth window is read raw, so P never touches the triple-window region at all.  If T1 injects only inside the triple-windows then P(B) carries no injection: the Program acts on clean data and the AD Consumer's ledger inside that same P(B) is empty.  Nothing would flip, and nothing would have been tested.

These two cannot both hold:

- #35 v2 D2: T1's injection goes only inside the triple-windows
- #35 errata (d): injection and AD detection happen inside the 12 train series' blocks, both Consumers consuming one P(train block)

a main-line ruling on where T1's injection lives, before T1 runs  And T0's calibration block [143, 431) lies inside [120, 900), so D2's hard T0/T1 isolation would then require T1's block to avoid it.

### What the acceptance gate actually gated on

| setting | recall | predicted events |
| --- | ---: | ---: |
| primary (25 / 4.0) | 0.9688 | 121 |
| fallback (49 / 3.5) | 0.9688 | 90 |

recall did not move between the two detector settings: the detector saw the same injections either way.  The whole F1 difference came from precision, whose denominator is the count of predicted events, most of which the ledger does not name.  So the fallback was selected on the background alarm level -- the quantity C5(ii) says bounds nothing.  The verdict follows the frozen protocol, but the gate as built reads background, not readability, and the main line should know that before T1 leans on the frozen setting.

Margin over the bar: **+0.0082**.

### Resolution of the AD gain vector

with two ledger events a per-series F1 moves in steps of roughly 0.2 to 0.3 when one event changes hands.  The pre-registered material line of 0.005 cannot resolve anything below that step, so a per-series AD gain fed to the guard's min_per_series_gain is coarse by construction.  The errata's reading -- flip judged at the aggregate layer, per-series comparison only within each task -- is the one this resolution supports.

### Sigma-scale mismatch

a nominal 6-sigma_local injection is not guaranteed to be a detector-scale exceedance, because the two sigmas are measured over different spans.  One of the 32 calibration events was missed for exactly this reason.

## Cost

- LLM calls 0.  Forecasting retrains 0.  AD evaluations 176 of 200.
- Sealed: NOAA 2025, everything beyond 17520, SMD official test and labels.
- Wall seconds 1.9.
