# #45-Frep: A5 vs A3 forecasting development replay

**Chain verdict: `CHAIN_REPRODUCED`**

both arms walked Source Skill -> held-in calibration -> Target-local Skill -> freeze -> held-out Fast-only scoring; on the primary pooled cell A5 first-positive cost 84 vs 123 (cheaper=True) and held-out delayed difference -0.152549 (better=False)

**Raw A5-vs-A3 instrument verdict on the same readings: `FRESH_A5_FAILS`** (pooled `A5_LOSES`, per_channel `A5_TIE_TRANSFER_BOUNDARY`).

The two verdicts answer different questions and must be quoted together. The chain verdict asks whether the forward chain still walks and whether A5 keeps a directional edge on *either* primary readout; the original pre-registered A5-vs-A3 clauses ask for a joint cost-and-terminal-utility win, and on this replay they are not met.

Development data, already-exposed outcomes. This is a chain regression check on current HEAD, not capability evidence.

## Binding

- run id: `frep45_r1`
- reused entry: `evaluation/functional/run_e2_fresh_confirmation.py`
- original verdict being replayed: `FRESH_A5_DELIVERS`, first-positive cost 69 vs 123 (pooled)
- Target: noaa_fresh, 16 series, partition development_2024, index [0, 8760)
- Source Skill overlap with Target series: `none`
- LLM budget both arms: 40 (task-book cap 200)

## Windows

| block | window | support origins | delayed origins | farthest |
| --- | --- | --- | --- | ---: |
| held-in task_A | `fresh_task_A` | [1104, 1152, 1200] | [1248, 1296, 1344] | 1392 |
| held-in task_B | `fresh_task_B` | [1800, 1848, 1896] | [1944, 1992, 2040] | 2088 |
| held-in probe | `fresh_probe` | [1440, 1488, 1536] | -- | 1584 |
| held-out | `frep45_held_out` | [8472, 8520, 8568] | [8616, 8664, 8712] | 8760 |

held-in / held-out disjoint: `True`; held-out farthest 8760 <= 8760: `True`; 2025 bytes read: `False`

## Instrument drift since FRESH_A5_DELIVERS

5 of 19 frozen-surface files moved between the original reading and HEAD.

- `evaluation/functional/run_e2_skill_store_integration.py`: 37d31cb85e9b -> 0dbe61d98def
- `evaluation/functional/run_e2_warm_vs_cold_recipe_search.py`: bab5feb8e6ad -> ffd532752100
- `evaluation/functional/task_episode_harness/e1.py`: 389110c5fad6 -> e5501fe94ad7
- `methods/ttha/method.py`: e9c27af3d43d -> a16d5257570f
- `methods/ttha/harness/h0/snapshot.lock.json`: 1e54a67ea021 -> b29ac33966be

## Paired readouts

| cell | arm | first-pos cost (retrains) | first-pos LLM | total retrains | total LLM | held-in delayed | held-in harm | held-out plan | held-out delayed | held-out harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `pooled` | `A5` | 84 | 2 | 108 | 2 | +0.066941 | 1 | `identity` full batch | +0.000000 | 0 |
| `pooled` | `A3` | 123 | 5 | 192 | 7 | +0.162837 | 1 | `hampel_filter` full batch | +0.152549 | 0 |
| `per_channel` | `A5` | 75 | 2 | 108 | 2 | +0.046298 | 0 | `repair_level_shift` full batch | +0.096496 | 1 |
| `per_channel` | `A3` | 75 | 2 | 108 | 2 | +0.046298 | 0 | `repair_level_shift` full batch | +0.096496 | 1 |

## Per cell (raw instrument clauses, unchanged from the original)

| cell | raw verdict | first-pos A5/A3 | total A5/A3 | held-out delayed diff | harm A5/A3 |
| --- | --- | --- | --- | ---: | --- |
| `pooled` | `A5_LOSES` | 84 / 123 | 108 / 192 | -0.152549 | 0 / 0 |
| `per_channel` | `A5_TIE_TRANSFER_BOUNDARY` | 75 / 75 | 108 / 108 | +0.000000 | 1 / 1 |

## Lifecycle receipts

- Skills formed (Drafts written): 3
- Store approvals: 3
- Promotions to LOCAL_ACTIVE: 3
- Drafts written but not promoted: 0
- Revocations observed: 0
- Deploy abstentions on reuse: 1

## Freeze and deploy binding

- frozen snapshot unchanged across deployment: `True`
- scored bytes == applied bytes on every arm: `True`

## Budget

- LLM calls: 13 / 40 (task-book cap 200)
- LLM by arm: {'A5': 4, 'A3': 9}
- Consumer retrains: 528
- Consumer retrains by arm: {'A5': 216, 'A3': 300}
- Downloads: 0; sealed reads: 0

## Wall

483.8 seconds.

## Post-run annotation (0 evaluation)

after the run, by a 0-evaluation deterministic pass. Changes no measured number: `True`.

### What reproduced

The whole forward chain walked on current HEAD. The Source Guidance cards recompiled to byte-identical sha256 prefixes 5e55df667b46 (pooled) and c221c2163d30 (per_channel), the same bytes the FRESH_A5_DELIVERS run registered. Stage 0 readability reproduced digit for digit (pooled spread 1.3522555977523183 / 2.291102717863714, share 0.2913 / 0.3325). Three Target-local Drafts were written through handle_fast_winner, all three were approved through handle_feedback_delayed and all three were promoted to LOCAL_ACTIVE through e1._update_delayed. All four arm stores froze to a stable runtime_bundle_sha and active pointer, and all four were byte-identical after deployment. The deploy binding assertion held on all four arms: the scored aggregate is the delayed_gate reading of exactly the applied bytes.

### The load-bearing directional reading

On the primary pooled cell the first-positive cost ordering reproduced: A5 reached its first delayed-positive adoption at 84 Consumer retrains against A3's 123, a 31.7% reduction, versus 69 vs 123 (-43.9%) in the original. A5 also spent 2 LLM calls against A3's 5 to get there. The direction of the original load-bearing readout therefore survives the #42i/#42k/#42l wiring surgery.

### The reading that did not reproduce, and why

The original pooled cell had a held-out delayed difference of exactly +0.000000 (both arms landed on the same mechanism). Here the pooled difference is -0.152549 against A5: A5's frozen Target-local Skill (outlier_iqr) failed the +0.005 current-window Support confirmation on the held-out tail and the harness abstained to identity for +0.000000, while A3, which carried no ACTIVE Skill into deployment, was routed into a fresh full-price search on the held-out block and landed hampel_filter at +0.152549. The pooled A5_LOSES clause is therefore driven by an abstention against a fresh search, not by A5 degrading anything: A5's held-out harm count is 0 against A3's 0, and A5's deployment cost 9 retrains and 0 LLM against A3's 69 retrains and 2 LLM. On per_channel both arms converged to the identical Skill id and identical readings, giving a clean tie.

### Claim ceiling for this artifact

This is a development replay on already-EXPOSED NOAA 2024 outcomes. It may be cited as INFRASTRUCTURE/MECHANISM evidence that the forecasting forward chain -- Source Skill injection, held-in multi-round calibration, Target-local Skill formation/approval/promotion, freeze, held-out Fast-only deployment and one-shot scoring -- is unbroken at HEAD, and that the first-positive-cost direction survives. It may not be cited as fresh evidence, as generalisation, as a replication of the FRESH_A5_DELIVERS magnitudes, or as evidence that A5 does or does not improve final held-out quality: the pre-registered A5-vs-A3 clauses were not met here, and a single unreplicated development block cannot settle that either way.

### Out-of-book findings (reported, not fixed)

- **F1 -- held-out deployment reads held-out delayed outcomes** -- The inherited protocol's deployment episode consults the delayed (held-out) reading as an adoption gate. The artifact records it explicitly: a3_pooled's adoption ladder took two newly measured delayed reads on the held-out block, roles 'bar' and 'confirmation', and its path is GATE_FAIL_FALLBACK_SUPPORT_WINNER -- i.e. the held-out outcome selected the deployed program. AGENTS.md section 3 forbids open_delayed in held-out. This is a pre-existing property of run_e2_fresh_confirmation stage 4, inherited verbatim, not introduced here. Reported, not fixed.
- **F2 -- deployment cost is asymmetric by construction** -- An arm that carries an ACTIVE Target-local Skill into deployment pays a recall plus one Support confirmation (9-15 retrains, 0 LLM); an arm that carries none is routed into a full-price search at deployment (69 retrains, 2 LLM). The arm with less accumulated knowledge therefore gets more search on the scored block. This structurally biases the held-out terminal-utility comparison against A5 and is the direct cause of the pooled A5_LOSES clause above. It also existed in the original run (pooled A3 spent 72 retrains and 2 LLM at task_C) but did not bite there. Reported, not fixed.
- **F3 -- frozen-surface drift since the original reading** -- 5 of the 19 files the original run froze have moved at HEAD: run_e2_skill_store_integration.py, run_e2_warm_vs_cold_recipe_search.py, task_episode_harness/e1.py, methods/ttha/method.py and methods/ttha/harness/h0/snapshot.lock.json. The chain still runs and the Guidance card bytes and stage 0 numbers still reproduce exactly, so the drift is compatible; but any future citation pairing this replay with fresh_confirmation_v1 must disclose that the two readings were taken on different instrument surfaces.
- **F4 -- h0 snapshot lock is bypassed, not satisfied** -- Every store build in this chain calls compile_snapshot(H0_ROOT, verify_lock=False). The #42k INSTRUMENT_UNREADABLE h0 lock mismatch is therefore routed around rather than resolved on this path; the lock file's own sha is one of the five drifted entries in F3. Reported, not fixed.
- **F5 -- LLM non-determinism moves the magnitudes** -- The Fast Agent shortlisted outlier_iqr for pooled A5 where the original shortlisted outlier_mad, which is what moves the pooled first-positive cost from 69 to 84 and the held-out recall from a passing reuse to an abstention. Any future quantitative comparison against fresh_confirmation_v1 needs repeated draws; a single paired run pins the direction only.

### Obligation self-report

- read_repo_AGENTS_md_first: `True`
- conda_env_project_and_interpreter_printed: `True`
- interpreter: `D:\Anaconda_envs\envs\project\python.exe`
- subagents_spawned: `0`
- downloads: `0`
- new_data_selected: `False`
- new_instrument_built: `no new Consumer, program, prompt, schema, gate, threshold or candidate; one thin driver that supplies windows, the isolated store root, the freeze receipt and the deploy assertion, and calls the in-service runner's own stage callables`
- methods_modified: `False`
- in_service_runner_lines_edited: `0`
- noaa_2025_read: `False`
- beyond_17520_read: `False`
- yahoo_read: `False`
- other_line_files_touched: `none; AGENTS.md, README, PROJECT_STATE_AND_DATA_MAP and SUCCESSOR_BRIEF were read only`
- fresh_confirmation_v1_artifacts_modified: `False`
- runs_executed: `1`
- rerolls: `0`
- llm_calls: `13`
- llm_cap_task_book: `200`
- consumer_retrains: `528`
- preflight_disclosed: `one 0-LLM preflight of stage 0 / stage 1 / the missing gate ran first under a throwaway store root and is not part of the scored run's ledger; it cost 12 additional development-partition Consumer retrains`
- committed: `False`

