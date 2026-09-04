# #42f-local-r2 local AD candidate census

verdict: **LOCAL_TARGET_REQUIRES_MULTIVARIATE_ADAPTER**

0 LLM / 0 fit / 0 retrain / 0 download. Label **values** were not read, counted, or printed. Only existence, path, format, shape/header, and prior exposure records.

Old Forecasting screens (G3 / S0 usable-channel and public-phenomenon bars) are **not** AD vetoes.

Current Consumer `aegists_iforest_v1` takes one univariate series and truth as event row-sets (`macro_event_f1`).

## Per-candidate table

| | SMD | MSL | SMAP | PSM | SWaT |
|---|---|---|---|---|---|
| Native task | OmniAnomaly server machines | NASA MSL telemetry AD | NASA SMAP telemetry AD | eBay pooled-server AD | iTrust water-treatment AD |
| Local | TSLib packed npy/pkl | TSLib npy | TSLib npy | TSLib csv | iTrust xlsx + csv |
| Redistribute | research dump; official 28-machine files were scratchpad-only | TSLib/NASA dump, untracked | same | eBay dump, untracked | iTrust license, untracked |
| Entity × channel | **28 machines × 38 metrics** | packed stream × **55** | packed stream × **25** | **1** plant × **25** features | **1** plant × **51** sensors |
| Train / test shape | 708405×38 / 708420×38 | 58317×55 / 73729×55 | 135183×25 / 427617×25 | 132481 / 87841 rows, 25 features | swat2 449919×52; train2 495000×52 |
| Label file | `SMD_test_label.npy` shape [708420] | `MSL_test_label.npy` [73729] bool | `SMAP_test_label.npy` [427617] bool | `test_label.csv` header `timestamp_(min),label` 87841 rows | inline `Normal/Attack` column |
| Label grain | **machine / packed-row**, not channel | **stream-level** | **stream-level** | **system-level** | **system-level** |
| Official train/test | yes (train vs test files) | yes | yes | yes | Normal_v1 vs Attack_v0 (not 0.7n) |
| Train context / outcome | INSTANCE_SEEN / **EXPOSED** (S1 + readiness) | INSTANCE_SEEN / EXPOSED (S0 prefix) | INSTANCE_SEEN / EXPOSED (S0 prefix) | INSTANCE_SEEN / EXPOSED (G3 screen) | INSTANCE_SEEN / EXPOSED (G3 on swat2 values) |
| Test values | INSTANCE_SEEN / **SEALED** | INSTANCE_SEEN / SEALED | INSTANCE_SEEN / SEALED | INSTANCE_SEEN / SEALED | n/a as a separate test csv |
| Test/attack labels | AGGREGATE_SEEN / **SEALED** | AGGREGATE_SEEN / SEALED | AGGREGATE_SEEN / SEALED | AGGREGATE_SEEN / SEALED | header name seen / **SEALED** (no value counts) |
| Old screen | Forecasting S0/S1 eligible after entity fix | Forecasting **FAILS_DEGENERATE_CHANNELS** | same | Forecasting **REJECTED_NO_PUBLIC_PHENOMENON** | Forecasting **REJECTED_SUBSTRATE** |
| Old Judge | forecast windows / phenomenon bars | same | same | same | same |
| Transplant as AD veto? | **no** | **no** | **no** | **no** | **no** |
| `aegists_iforest_v1` | only via frozen channel + event-map; weak labels | no honest multi-entity map | no | one-entity only | one-entity only |
| Split for #42g | official train fit; official test still one sealed block | official files exist; no entity roster | same | official files; one timeline | official Normal/Attack; **incompatible with 0.7n** |
| Adapter path | **C** weak-label channel proxy | **D** multivariate adapter | **D** | **D** (C would still be 1 entity) | **D** |
| Enter univariate #42g now? | **no** | **no** | **no** | **no** | **no** |

## Path notes

- **A AS_IS_UNIVARIATE**: none of the five.
- **B FIXED_CHANNEL_PER_ENTITY**: SMD is the only multi-entity case and already has an outcome-blind rule (channel **18**, cardinality on train `[0,1104)`, intersection of 28 machines). That can *build* 28 univariate series, but labels stay machine-level → this book records it as **C**, not B.
- **C CHANNEL_AS_SERIES_WEAK_LABEL**: SMD (and PSM/SWaT if one channel were forced). Forbidden to treat copied labels as independent samples.
- **D MULTIVARIATE_CONSUMER_ADAPTER_REQUIRED**: MSL, SMAP, PSM, SWaT; also the honest reading of SMD if the 38 metrics stay together.

## Exam geometry

`#42g` wants official train / base fit + test prefix held-in + test suffix sealed held-out. Official train/test exists for SMD/MSL/SMAP/PSM. Forcing the Yahoo-style **0.7n on the concatenated file** would ignore official partitions → layout stop, not a silent remap. SWaT’s Normal vs Attack protocol is also not 0.7n.

## Verdict ladder

1. INSTRUMENT_UNREADABLE — no. Shapes rechecked; SMD entity tiling already proven in `smd_entity_structure_v1`.
2. EXPOSURE_UNCLEAR_STOP — no. Test/label outcomes are recorded SEALED in readiness v1/v2 and were not opened here.
3. LOCAL_UNIVARIATE_PROXY_TARGET_AVAILABLE — no. B is not satisfied without weak-label overclaim.
4. **LOCAL_TARGET_REQUIRES_MULTIVARIATE_ADAPTER** — yes. Sealed local AD targets exist; legality needs a multivariate Consumer (or an explicit weak-label protocol this book does not authorize).
5. NO_LOCAL_COMPATIBLE_TARGET — not reached. Yahoo is **not** auto-resumed; download remains a user decision if the exam must stay univariate without a new adapter.

## Ambiguities (not self-adjudicated)

- SMD packed test labels are 1-D over 708420 rows. Historical OmniAnomaly per-machine test files were not re-fetched this round; machine tiling on **test** is not re-proven here (train tiling is).
- S0 once marked SMD/MSL/SMAP train `UNSEEN/SEALED`; later S1/readiness superseded SMD train to EXPOSED. This census uses the later record.
- SWaT `data.py` historically counted attack rows; that script was not executed.
