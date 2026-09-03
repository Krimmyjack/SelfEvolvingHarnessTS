# SMD target readiness v2 -- index-aligned mapping

**Verdict: `JUDGE_UNREADABLE_ALIGNED_MAPPING`** -- identity readings fail the reused NOAA bar on the aligned mapping; there is no backup mapping and no threshold is moved

The only variable this round: the entity-to-series mapping.  #32 let each machine choose its own highest-cardinality channel; this round uses one channel index eligible on all 28.  The Candidate v2, the Consumer, the Metric, the thresholds, the roster, the windows, the menu, the Program bindings, the risk line, the guard and the Observation surface are all untouched.

Exposure: SMD selected train development windows `context = INSTANCE_SEEN`, `outcome = EXPOSED`; SMD official test and labels `outcome = SEALED`, never read.

## A -- the aligned mapping

- Eligibility: cardinality > 20 on that machine's own [0, 1104)
- Intersection over all 28 machines: `[1, 2, 5, 13, 14, 15, 18, 19, 20, 21, 27, 30]` (|I| = 12).
- Chosen channel index: **18** (median cardinality 1069.5; ties [18]).
- `mapping_semantics = CHANNEL_INDEX_ALIGNED_PROXY`.  aligned by column position, not by known physical meaning.  SMD ships no channel dictionary, so nothing here establishes that column 18 is the same quantity on every machine; it establishes only that it is the same column and that it varies enough on all 28 to be a candidate.  Any claim that the cohort is physically homogeneous would be unsupported.

| entity | eligible channels | cardinality at the chosen index |
| --- | ---: | ---: |
| `machine-1-5` | 24 | 991 |
| `machine-2-7` | 24 | 1077 |
| `machine-2-6` | 26 | 1090 |
| `machine-3-2` | 24 | 1100 |
| `machine-3-11` | 18 | 1103 |
| `machine-2-5` | 25 | 1069 |
| `machine-1-8` | 25 | 1094 |
| `machine-1-3` | 25 | 1070 |
| `machine-3-4` | 21 | 972 |
| `machine-2-2` | 24 | 1001 |
| `machine-1-4` | 25 | 1059 |
| `machine-3-10` | 18 | 426 |
| `machine-2-9` | 20 | 889 |
| `machine-1-7` | 26 | 1047 |
| `machine-3-7` | 22 | 1095 |
| `machine-1-6` | 27 | 1095 |
| `machine-2-3` | 15 | 499 |
| `machine-3-8` | 22 | 904 |
| `machine-3-5` | 21 | 1058 |
| `machine-3-3` | 30 | 1088 |
| `machine-1-1` | 24 | 929 |
| `machine-1-2` | 22 | 853 |
| `machine-2-1` | 23 | 1092 |
| `machine-2-8` | 24 | 1043 |
| `machine-2-4` | 24 | 1102 |
| `machine-3-9` | 26 | 1094 |
| `machine-3-6` | 26 | 1090 |
| `machine-3-1` | 25 | 1096 |

## B -- Judge at identity

| episode | block | spread (cap 5.0) | share (cap 0.40) | finite | non-degenerate | pass |
| --- | --- | ---: | ---: | --- | --- | --- |
| episode_1 (s=1104) | support | 9.8681 | 0.5261 | True | True | False |
| episode_1 (s=1104) | delayed | 2.5845 | 0.4010 | True | True | False |
| episode_2 (s=1800) | support | 4.5855 | 0.4090 | True | True | False |
| episode_2 (s=1800) | delayed | 5.1337 | 0.3737 | True | True | False |

## Cost

- LLM calls: 0.  Consumer retrains: 12 / 100.
- Wall seconds: 0.3.

## Consequence

SMD is closed as a Target under the current fixed forecasting family.  No Judge calibration and no on-site repair is attempted.
