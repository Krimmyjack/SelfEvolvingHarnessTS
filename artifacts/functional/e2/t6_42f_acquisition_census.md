# #42f acquisition census

verdict: **NO_LOCAL_SEALED_CANDIDATE** (Part A stop, kept)

**After user authorization:** see `t6_42f_yahoo_a1_freeze.md` — **TARGET_FROZEN_SEALED**.

Part 0 sha: `d594fcfaae756f6ab70cc84b50693de9eb1d9aa4`

0 LLM / 0 AD fit / 0 retrain / 0 download. No label values opened. No NAB / NOAA 2025 / SMD / beyond_17520 reads.

This book stops. Whether to download Yahoo S5 (preferred) or UCR-AD (fallback) is a **user decision**.

## Eligibility (r1)

- context may be UNSEEN / AGGREGATE_SEEN / INSTANCE_SEEN
- outcome must be SEALED
- length gate if freezing: T6 `MIN_LENGTH=1000`
- do not mix two libraries in one exam

## Candidates

| id | local? | context | outcome | eligible | why not |
|---|---|---|---|---|---|
| Yahoo S5 A1Benchmark (prefer) | **no** | AGGREGATE_SEEN | SEALED | no | files absent |
| UCR-AD Anomaly Archive (fallback) | **no** | AGGREGATE_SEEN | SEALED | no | files absent |
| `data/ucr_task_context/*.zip` | yes (40 zips) | INSTANCE_SEEN (names only) | SEALED | no | UCR **classification** archive, not UCR-AD |

Search (name-only): `data/`, shared `tsquality`, sibling `SelfEvolvingHarnessTS-deepseek/data`, Desktop, Downloads, `/mnt/d`, `$HOME`. No `A1Benchmark`, `ydata-labeled-time-series`, `Yahoo_S5`, `UCR_AnomalyArchive`.

Exposure citations for the two planned libraries are planning text only (`ROADMAP`, `STAGE_REPORT`, `PROJECT_STATE_AND_DATA_MAP`). Context-seen ≠ outcome leak.

## Part B

Not issued. No roster, no 0.7n wall, no vaults, no SHA lock.

## Residual

None. Stop is the pre-registered empty-local cell.
