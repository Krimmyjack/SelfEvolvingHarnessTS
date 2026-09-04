# T1 A3 single-arm variance (post-c166b63) v1

Measurement instrument for later AB readout. Not a Capability, not a Claim,
no Gate. T5口径 names are used throughout.

Protocol: frozen electricity 9-task roster, `paired_arms=True`,
`run_slow=False`, `warm_arm_snapshot=None`. Both arms of each calib run are
the current A3 configuration. Driver:
`evaluation/functional/task_episode_harness/agentic/calib_a3.py`.

Smoke (run01) passed before the batch: `calib_a3_run01.json` written,
`.calib_a3_state/run01/` exists, 9/9 `source_prior_retrieval.matched=false`,
mainline `.g1_pipeline_state/` and `g1_agentic_pipeline_report.json` mtime
unchanged (still 2026-08-19 03:32 / 03:55). Batch run02/run03 used the same
isolated paths. All three calib runs stayed 9/9 `matched=false`, so no WARM
row is dropped.

## Sample

7 A3-config trajectories = 6 new (3 runs × 2 arms) + D1 A3 from
`g3d1_electricity_skill_only_ab.json`. D1 A5 is **not** in this sample (it
carried Source-derived guidance). Historical D0 A3 is **not** in this sample
(pre-c166b63 Fast Episode channel).

Relay models actually returned: `gpt-5.6-luna` on every trajectory;
run03_WARM also saw `gpt-5.6-luna-2026-07-09`. That routing is a variance
source, not a protocol change.

run02_WARM task_02 exited `AGENT_PROTOCOL_ERROR` (0 probes on that task).
Mechanical, not a method event. The trajectory stays in the sample; that
task is not a behavioural readout.

## Per-trajectory readout

| trajectory | LOCAL_ACTIVE | `real_support_probe_count` | material-harm n / sum | all-negative n / sum | first material-positive task | abstention | returned_models |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| D1_A3 (existing) | 0 | 11 | 9 / 0.721 | 9 / 0.721 | 6 | 9 | gpt-5.6-luna |
| run01_COLD | 0 | 8 | 4 / 0.410 | 4 / 0.410 | 7 | 8 | gpt-5.6-luna |
| run01_WARM | 3 | 9 | 3 / 0.206 | 3 / 0.206 | 2 | 5 | gpt-5.6-luna |
| run02_COLD | 0 | 10 | 7 / 0.608 | 7 / 0.608 | 6 | 8 | gpt-5.6-luna |
| run02_WARM | 1 | 8 | 4 / 0.682 | 4 / 0.682 | 5 | 5 | gpt-5.6-luna |
| run03_COLD | 0 | 9 | 8 / 0.657 | 8 / 0.657 | 6 | 9 | gpt-5.6-luna |
| run03_WARM | 0 | 9 | 6 / 0.418 | 6 / 0.418 | none | 9 | gpt-5.6-luna + gpt-5.6-luna-2026-07-09 |

`charged_probe_cost` is not a probe count. For the six calib arms it was 36
whenever LOCAL_ACTIVE=0, and 27 when LOCAL_ACTIVE was 3 (run01_WARM) or 1
plus the protocol-error task (run02_WARM). Do not cite 36 as probes.

## Empirical distribution (n=7)

| quantity | values | min | max | mean |
| --- | --- | ---: | ---: | ---: |
| LOCAL_ACTIVE | 0,0,3,0,1,0,0 | 0 | 3 | 0.57 |
| `real_support_probe_count` | 11,8,9,10,8,9,9 | 8 | 11 | 9.1 |
| material-harm n | 9,4,3,7,4,8,6 | 3 | 9 | 5.9 |
| all-negative n | same as material-harm n in this sample | 3 | 9 | 5.9 |
| abstention | 9,8,5,8,5,9,9 | 5 | 9 | 7.6 |

Five of seven trajectories formed **zero** LOCAL_ACTIVE. One formed 1, one
formed 3. First material-positive task index, when it exists, ranged from 2
to 7; one trajectory never got a material-positive probe.

## Same-config paired difference (WARM − COLD, identical A3)

This is the noise of two concurrent A3 arms on the same Task Episode, not
an A5 vs A3 effect.

| run | Δ LOCAL_ACTIVE | Δ material-harm n | Δ material-harm sum | first-pos COLD / WARM |
| ---: | ---: | ---: | ---: | --- |
| 01 | +3 | −1 | −0.205 | 7 / 2 |
| 02 | +1 | −3 | +0.074 | 6 / 5 |
| 03 | 0 | −2 | −0.238 | 6 / none |

Observed |Δ LOCAL_ACTIVE| already reached **3** with no Source Skill and no
guidance difference. Observed |Δ material-harm n| reached **3**. Observed
|Δ material-harm sum| reached **0.24**.

## Reference rule for later AB readout

Use this as a noise band on the frozen 9-task electricity protocol, current
config, n=7 A3-config trajectories. Small-n: it is a reference, not a test.

1. **LOCAL_ACTIVE.** An official A5−A3 difference of 1 or 2 on this 9-task
   roster does **not** exceed the same-config band (observed pair diffs 0, 1,
   3; single-arm range 0–3). A difference of ≥4 would sit outside this
   sample's range. Zero vs zero is the modal A3 outcome (5/7).
2. **material-harm count (Draft gate `g < -0.005`).** |Δ| ≤ 3 harmful probes
   is inside the band. |Δ| ≥ 5 would be outside the observed same-config
   pair diffs, still inside the single-arm range 3–9, so treat it as weak.
3. **material-harm sum.** |Δ| ≲ 0.25 is inside the same-config band.
   Cite the口径; all-negative coincided with material-harm on these seven
   arms (no extra immaterial negatives except D1's `0.0`, which is neither).
4. **first material-positive task index.** Too unstable on 9 tasks to be a
   confirmation readout (2, 5, 6, 7, or never). Do not rank arms by it
   unless the gap survives this spread.
5. **abstention.** 5–9. A difference of 1–3 abstentions is noise.
6. **`real_support_probe_count` vs `charged_probe_cost`.** Never quote charged
   as probes. Real probes here live in 8–11.

This file does not say the current D1 skill-only AB is significant or not.
It only says what size of difference this instrument can currently see.

## Cost (calib runs, not D1)

| run | wall | LLM calls COLD+WARM | prompt tokens COLD+WARM | matched true |
| ---: | ---: | --- | --- | ---: |
| 01 | 1066 s | 61+63 | 581091+583401 | 0/9 |
| 02 | 1433 s | 46+57 | 406497+535649 | 0/9 |
| 03 | 1917 s | 55+54 | 512047+496310 | 0/9 |

Mainline `.g1_pipeline_state/` and `g1_agentic_pipeline_report.json` mtimes
were not written.
