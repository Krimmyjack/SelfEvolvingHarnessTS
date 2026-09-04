# A5 vs A3 power precheck v1

Instrument only. No claim that a current AB is significant.
Zero LLM. T5口径 names are used throughout.

Version 1 uses assumed discordant-pair rates plus the already-exposed
D0 / D1 / Weather paired binaries. Version 2 (below) replaces the
variance assumption with T1's A3 single-arm distribution once that
calibration finishes.

## Estimand

Paired Task Episode, A5 vs A3, same frozen protocol. Binary outcomes:

- `LOCAL_ACTIVE` formed (yes/no)
- first material-positive probe occurred on that task (yes/no)
- at least one **material-harm** probe (`support_gain < -0.005`)
- at least one **all-negative** probe (`support_gain < 0`)

Primary test: McNemar / sign test on discordant pairs. Let

- `p10` = P(A3 yes, A5 no)
- `p01` = P(A3 no, A5 yes)
- `ψ = p10 + p01` discordant-pair rate
- `Δ = p01 − p10` A5 advantage (or A3 advantage if negative)

Two-sided α = 0.05, target power 80%. Sample size (paired tasks)

```
n = ((z_{1-α/2} √ψ + z_{1-β} √(ψ − Δ²)) / Δ)²
```

with `z_{1-α/2} = 1.960`, `z_{1-β} = 0.842`. Requires `ψ ≥ |Δ|`.
Independence across Task pairs is an assumption; prequential tasks on one
cohort are **not** independent replicates. n below is therefore a lower
bound on roster size, not a licence to treat 9 electricity tasks as 9 iid
draws.

## n(Δ, ψ) for 80% power

| Δ | ψ=Δ (all discordant in one direction) | ψ=1.5Δ | ψ=2Δ | ψ=0.30 | ψ=0.40 | ψ=0.50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 77 | 116 | 155 | 234 | 312 | 391 |
| 0.15 | 50 | 77 | 103 | 103 | 138 | 172 |
| 0.20 | 37 | 57 | 77 | 57 | 77 | 96 |
| 0.25 | 29 | 45 | 61 | 36 | 48 | 61 |
| 0.30 | 24 | 37 | 50 | 24 | 33 | 42 |
| 0.40 | 18 | 27 | 37 | — | 18 | 22 |
| 0.50 | 14 | 21 | 29 | — | — | 14 |

Empty cells: `ψ < Δ` is impossible. Round up; these are continuous-n
formula values.

Reading the table: to detect a 25-point paired advantage (Δ=0.25) at a
discordant rate ψ=0.40 you need about **48 paired tasks**. Nine electricity
tasks cannot do that. That is the E2-J0 lesson applied to this AB.

## Observed paired binaries (not tests)

Mechanical-exit pairs dropped. D0 drops e1v2_task_05 (A3
`AGENT_PROTOCOL_ERROR`). D1 drops none. Weather w1 drops none.

| source | n usable | LOCAL_ACTIVE A3only / A5only / both / neither | first material-positive A3only / A5only / both / neither | material-harm-any A3only / A5only / both / neither |
| --- | ---: | --- | --- | --- |
| D0 electricity (raw-Episode A5, rejected interface) | 8 | 3 / 0 / 0 / 5 | 5 / 1 / 0 / 2 | 1 / 5 / 2 / 0 |
| D1 electricity skill-only | 9 | 0 / 0 / 0 / 9 | 1 / 1 / 0 / 7 | 1 / 1 / 7 / 0 |
| Weather w1 A5A3 (autonomous guidance) | 19 | 1 / 2 / 7 / 9 | 0 / 2 / 11 / 6 | 1 / 4 / 8 / 6 |
| Weather g2 shakedown | 9 | 2 / 0 / 6 / 1 | 3 / 0 / 6 / 0 | 0 / 1 / 0 / 8 |

Implied Δ and ψ on D1 (skill-only, the relevant protocol):

- `LOCAL_ACTIVE`: Δ=0, ψ=0. Under this binary the current 9-task AB has
  **no discordant pairs** and cannot reject anything.
- first material-positive: Δ=0, ψ=2/9≈0.22. n=9 vs ~77 required for
  Δ=0.20 at ψ=0.40.
- material-harm-any: Δ=0, ψ=2/9≈0.22, with 7/9 both-harm. A harm-reduction
  claim of Δ=0.25 at ψ=0.40 still wants ~48 pairs.

Weather n=19 is larger but still below the Δ=0.20 / ψ=0.40 line (n≈77),
and it is a different A5 (guidance patch / source card), not G3-D1
skill-only.

These rows are planning inputs. They are not “current A5 is better/worse”.

## Assumptions (all of them)

1. McNemar two-sided, α=0.05, power 80%, no multiplicity correction
   across the four binaries.
2. Task pairs independent. False for sequential flywheel tasks on one
   cohort; treat n as a floor.
3. Version 1 does **not** use T1's run-to-run variance. Binary McNemar
   ignores within-arm Gaussian noise on continuous harm. Version 2 will
   fold T1 in.
4. D0 A5 is the rejected raw-Episode inlet. Its Δ is not a planning
   target for skill-only A5.
5. Instrument-unreadable or protocol-error pairs are dropped, not tied.

## G3-F roster suggestion (v1, before T1)

Do not run a confirmation AB with 9 paired tasks and expect to read a
modest A5 vs A3 difference.

- If the intended Δ on `LOCAL_ACTIVE` or first material-positive is ~0.25
  and discordant pairs are common (ψ≈0.40): **≥48 paired tasks**, split
  across several small targets so one flywheel does not eat the sample.
  A concrete shape: 3 targets × 16–20 frozen tasks each, same Consumer
  and budget, independently reset memory.
- If only a large effect (Δ≥0.40, ψ≈0.50) would count as a go: **≥22
  paired tasks**, e.g. 2 targets × 12. Still report per-target, do not
  pool as iid.
- A single 9-task electricity rerun is a development smoke, not a
  powered confirmation. That is already what D1 showed: 0–0 on
  `LOCAL_ACTIVE`, 1–1 on first material-positive.

Fresh sourcing stays outcome-blind screening. This file does not pick
datasets.

## Version 2 (T1 empirical A3 noise, 2026-08-19)

Source: `calib_a3_variance_report_v1.md`. Seven A3-config trajectories on
the frozen electricity 9-task protocol (D1 A3 + 6 calib arms). Not a
significance claim about any current AB.

Observed single-arm spread:

| quantity | min | max | mean | same-config pair \|Δ\| |
| --- | ---: | ---: | ---: | --- |
| LOCAL_ACTIVE per 9-task arm | 0 | 3 | 0.57 | 0, 1, 3 |
| material-harm n | 3 | 9 | 5.9 | 1, 2, 3 |
| first material-positive task index | 2 | never | — | unstable |

5/7 A3-config arms formed zero LOCAL_ACTIVE. Identical-config concurrent
arms already differed by up to 3 LOCAL_ACTIVE and 3 material-harm probes.

What this does to the v1 n(Δ) table:

- A 9-task electricity AB cannot resolve a LOCAL_ACTIVE advantage of 1–3.
  That is the observed null band, not a detectable Δ.
- McNemar on “LOCAL_ACTIVE formed on this task” is even worse: most tasks
  are neither-arm-active, so ψ is tiny and Δ is smaller still. n from the
  v1 table for Δ=0.10 (77–391) is the relevant order if the per-task
  success rate stays near zero.
- material-harm-any as a 9-task binary is also inside a 3-count noise
  band. Detecting a 25-point paired harm reduction (Δ=0.25, ψ=0.40) still
  wants ~48 pairs, unchanged from v1.
- first material-positive index is not a powered confirmation readout on
  9 tasks.

G3-F roster (v2, replacing v1's last bullet):

- Do not use a 9-task electricity rerun as the confirmation AB. T1 shows
  the instrument noise on that roster is already as large as the effects
  one would hope to read.
- To read Δ≈0.25 on a per-task binary (LOCAL_ACTIVE or first
  material-positive or material-harm-any) at ψ≈0.40: **≥48 paired tasks**,
  split across several small targets with independently reset memory
  (v1: 3 × 16–20).
- If only a large, roster-level count is acceptable — e.g. A5 forming
  ≥4 more LOCAL_ACTIVE than A3 on the same 9 tasks — T1's range says that
  bar is outside current A3 noise, but it is also near the ceiling of a
  9-task run. Prefer more tasks over raising the bar.
- Fresh sourcing stays outcome-blind. This file still does not pick
  datasets.

Independence caveat from v1 still applies: sequential tasks on one
cohort share a flywheel. T1's same-config pair diffs already include that
flywheel, which is why the 9-task noise band is wide.
