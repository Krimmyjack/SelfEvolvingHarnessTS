# Harness v2 freeze

Book = S3-R1. 2026-08-29. Scientific verdict **S3_EDIT_REJECTED**. Production policy = **DEFAULT**. v2 ≡ v1 behaviorally.

## Freeze declaration

Harness v2 ≡ v1 DEFAULT exploration policy. LLM-edit was legal but failed the acceptance gate (beneficiary cum_gain not non-inferior). Random-legal-edit never enters production. Parameterization remains in tree, fail-closed, DEFAULT-only unless a later sol-authorized revision installs a non-DEFAULT policy.

## Pilot

- verdict: `S3_EDIT_REJECTED` (gain not non-inferior vs no_edit; harm 0; G2 0)
- LLM-edit: `supply_reserved_probe_slots=1` (legal, 1 attempt)
- Random-edit: seed 20260829 `tie_break_rule=prefer_supplied` (never production)
- resumed: yes; first_returned_model=`gpt-5.6-sol`; LLM 62 / fit 97 / wall 1047.1s

## Inventory

- file_count: **30**
- prior_inventory_sha256: `68ea0f07dc12e57b4c623d94979dfbff7b7a926fc6a9b1d05fccb7bc04f11494`
- inventory_sha256: `dc11b19c63ce1519dc29571d727fed3ea6b971bc86872a9526b4e16611af2e34`

| path | bytes | sha256 |
|---|---:|---|
| `methods/ttha/retrieval.py` | 12631 | `d438149f6cc3feaa04628788c4e8122ac355e342e35cf32b462c139bdf1a4ef1` |
| `methods/ttha/fast_agent.py` | 68926 | `0e19cd2d18f48a93b2c2a6840e9b9e18580396644edb6bac247d740878509e7d` |
| `methods/ttha/online_loop.py` | 53863 | `536df2ba750bd58b2f1e8bf594ff80e79196f40cc06944656cf5b7d424b24eaf` |
| `methods/ttha/experience_memory.py` | 40146 | `d479bf4a5d71fa892168134d53d95d42c1102cfbc7d1036a538296537c3b3740` |
| `methods/ttha/ordering_card.py` | 11965 | `2bf85c0e8747dc510c29558689c68965421a290bd000c28aa6504fd7ac34a1b1` |
| `methods/ttha/method.py` | 76021 | `a16d5257570febce37662929e7e0379c04dba0a003f19064171d9a647e34177a` |
| `methods/ttha/signed_radius.py` | 21763 | `b911f75de4333e23b6e7342ded4767a1e29fb05e6a73c62621d219343cf3bdfa` |
| `methods/ttha/agent_core.py` | 22851 | `5c82b722db02a2859a9b7a605f067a4d3a9efcff46d0ae2163790e9979e69b7b` |
| `methods/ttha/public_tools.py` | 16034 | `10aa9a6c15ced59b026d60d997dbdf4502e01c418a7c6d9df9db71bcf5bf03b9` |
| `methods/ttha/slow_agent.py` | 19177 | `d4bbdb770ae5424a0f3e755dc4cd64cfbb71342028435f24f51479b6626afed9` |
| `methods/ttha/schema_contracts.py` | 15766 | `4ec56a613d4e5e06dab07126e290ab6e32f7836b32f27ddda257644a897bc408` |
| `methods/ttha/harness/compiler.py` | 44288 | `78e6772ea874b79b55b26ffe35c229de6a7bc371abab5505175bc70fdefc7e09` |
| `methods/ttha/harness/store.py` | 8652 | `04ab77279df6a15e31b5d2b8b85329e22584e3bfa3ec76760b6e85b839112e66` |
| `methods/ttha/harness/harness_surfaces.json` | 9027 | `e222daff89026ded3a111141a056ba3cd86c21a64bc32b350483ef11b5882824` |
| `contracts/harness.py` | 16837 | `a9d48a12db8747287842f99165716e6a79bafd33f309e3aaae0e650868ada4c2` |
| `contracts/observables.py` | 7901 | `08052cb06191ff2ffdfd5a930b98007ca23313abb3a5b8c2ca07dcac3cd559b4` |
| `contracts/canonical.py` | 5036 | `bdabde345fd17548579cdeef6840774ac03b78793b04541800310d717c9805c1` |
| `contracts/schemas/skill_entry_v1.json` | 871 | `aaf8473a829d2a988ec4d16d42552f6f9453ff1f95661b7c371554ee1b7a8e50` |
| `contracts/schemas/memory_entry_v1.json` | 629 | `d4aea29946d4df2120e7f76ad5f46808ad9731b4c01c67c9b770b68dbfe530a6` |
| `contracts/schemas/observable_feature_v1.json` | 4530 | `9af721ceff7d5b37191c928ecb72f14e491fc3abc4603750943bbaf5e584b6bc` |
| `evaluation/functional/task_episode_harness/agentic/source_skill.py` | 46833 | `d4062eb6fbda3b3a0dffc700966efef1cc70ff7a948f2f26e9f7808c27e0f06e` |
| `evaluation/functional/task_episode_harness/agentic/risk_skill.py` | 10490 | `e8b64943cdb29ab2b066487724173a009ce33151d8eb4242c2d616d23a48299c` |
| `evaluation/functional/task_episode_harness/agentic/skill_revision.py` | 13776 | `f1c856e811e7a43115a73bbf922a0798d3ff7b28fbb45e37401e2a464a5badef` |
| `evaluation/functional/task_episode_harness/agentic/runner.py` | 95898 | `816426ca0900b167c30b4daed131a366f7068d699b2236b4e11ca1f59140f1dc` |
| `evaluation/functional/task_episode_harness/e1.py` | 99786 | `e5501fe94ad7efd777ed9c67e30dfc6d7eed7df4b0ab2616f7857fe45341097f` |
| `evaluation/minipipe/feedback/fault_routes.json` | 4313 | `b9e05a9535c941f0da7cfc3295c8266b82658066da5e7b44f648ba33f5857e36` |
| `evaluation/minipipe/feedback/router.py` | 4432 | `6e0aa79e065d5f7b00fd6c6acc8c03b727b06c0e8c7ea3a3e0f5ccc2019e7272` |
| `evaluation/minipipe/replay/edit_controller.py` | 32638 | `14597c758febab68262d644f68ff7fbd11eda46e88b0412f330ebc5dad3c7100` |
| `runtime/public_features.py` | 13692 | `c9bd9ef36c054f69c3e6e8225bced13b9ec6954b5a47d9e74db9a59e412d31da` |
| `methods/ttha/exploration_policy.py` | 2876 | `e2ff07f0263ef2bd43a6984387cdf8fabb43bb6ffe66b5ca5a00b3fb6b78898b` |

## This book

- Exploration policy parameterized; DEFAULT reproduces chosen-first + first-positive-stop.
- Non-DEFAULT values fail-closed; only live under `install_policy` in the pilot runner.
- Instrument stops (`BACKEND_UNAVAILABLE` etc.) are not scientific verdicts.
- `runner.py` hash differs from v1.1 because of the earlier S2a inspect-path unification, already in git.

