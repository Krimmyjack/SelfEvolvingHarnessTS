# SMD target readiness (S1b)

**Verdict: `JUDGE_UNREADABLE`** -- identity readings fail the reused NOAA readability bar on both the primary and the backup mapping

Exposure: SMD selected train development windows `context = INSTANCE_SEEN`, `outcome = EXPOSED`; SMD official test and labels `outcome = SEALED`, never read.

## Machine -> series mapping

Entity = machine, channel = a metric inside it.  Channel choice reads [0, 1104) of each machine's own train block only.  highest cardinality, ties to the lowest channel index

Roster (first 16 of the frozen packing order, no substitution): train `['1-5', '2-7', '2-6', '3-2', '3-11', '2-5', '1-8', '1-3', '3-4', '2-2', '1-4', '3-10']`; eval `['2-9', '1-7', '3-7', '1-6']`.

| entity | train rows | eligible channels | primary ch | card | backup ch | card |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1-5` | 23705 | 24 | 22 | 1028 | 18 | 991 |
| `2-7` | 23696 | 24 | 18 | 1077 | 15 | 1069 |
| `2-6` | 28743 | 26 | 19 | 1094 | 18 | 1090 |
| `3-2` | 23702 | 24 | 18 | 1100 | 19 | 1096 |
| `3-11` | 28695 | 18 | 18 | 1103 | 19 | 1099 |
| `2-5` | 23688 | 25 | 15 | 1071 | 10 | 1069 |
| `1-8` | 23698 | 25 | 18 | 1094 | 13 | 1093 |
| `1-3` | 23702 | 25 | 22 | 1094 | 19 | 1081 |
| `3-4` | 23687 | 21 | 13 | 1083 | 22 | 1014 |
| `2-2` | 23699 | 24 | 18 | 1001 | 19 | 879 |
| `1-4` | 23706 | 25 | 22 | 1091 | 19 | 1068 |
| `3-10` | 23692 | 18 | 13 | 1058 | 10 | 529 |
| `2-9` | 28722 | 20 | 13 | 1049 | 19 | 931 |
| `1-7` | 23697 | 26 | 22 | 1102 | 19 | 1062 |
| `3-7` | 28705 | 22 | 18 | 1095 | 19 | 1094 |
| `1-6` | 23688 | 27 | 18 | 1095 | 19 | 1092 |
| `2-3` | 23688 | 15 | 13 | 820 | 10 | 796 |
| `3-8` | 28703 | 22 | 13 | 1066 | 19 | 913 |
| `3-5` | 23690 | 21 | 19 | 1084 | 13 | 1066 |
| `3-3` | 23703 | 30 | 18 | 1088 | 36 | 1085 |
| `1-1` | 28479 | 24 | 13 | 1053 | 22 | 1022 |
| `1-2` | 23694 | 22 | 5 | 1066 | 15 | 907 |
| `2-1` | 23693 | 23 | 19 | 1098 | 18 | 1092 |
| `2-8` | 23702 | 24 | 19 | 1074 | 18 | 1043 |
| `2-4` | 23689 | 24 | 18 | 1102 | 27 | 1064 |
| `3-9` | 28713 | 26 | 19 | 1098 | 18 | 1094 |
| `3-6` | 28726 | 26 | 19 | 1100 | 18 | 1090 |
| `3-1` | 28700 | 25 | 18 | 1096 | 19 | 1094 |

## Judge and headroom -- primary mapping

Judge readable: **False**.

| episode | support spread | support share | delayed spread | delayed share | readable |
| --- | ---: | ---: | ---: | ---: | --- |
| episode_1 (s=1104) | 4.93 | 0.409 | 4.78 | 0.486 | False |
| episode_2 (s=1800) | 4.66 | 0.43 | 18.3 | 0.463 | False |
## Judge and headroom -- backup mapping

Judge readable: **False**.

| episode | support spread | support share | delayed spread | delayed share | readable |
| --- | ---: | ---: | ---: | ---: | --- |
| episode_1 (s=1104) | 5.07 | 0.561 | 2.5 | 0.355 | False |
| episode_2 (s=1800) | 7.47 | 0.486 | 11.1 | 0.381 | False |
## Cost

- LLM calls: 0.  Consumer retrains: 24 / 100.
- Wall seconds: 0.9.

Nothing modified: Harness, Consumer, Metric, menu, risk line, v2 adoption ladder, the frozen v1 candidate, the Observation surface.
