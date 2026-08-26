# CLS-CONF-dl downloaded UCR roster

Isolation directory for the authorized ≤3-dataset download. Not a general SHA,
dual-warehouse, or Evidence Ledger. Roles were assigned by the frozen
lexicographic rule before any dataset zip was fetched.

Filter-trajectory pointer (written before any dataset zip):
`artifacts/functional/e2/t6_cls_conf_dl.json` field `selection`
(audit-gate snapshot also copied to `_scratch/cls_conf_dl_selection.json`).

Official metadata source:
`https://timeseriesclassification.com/aeon-toolkit/metadata.csv`
fetched 2026-08-25T13:20:15Z (190 rows). Local zip stems enumerated from
`data/ucr_task_context` (40 zips, mechanical `iterdir`). Eligible after the
conjunction: BinaryHeartbeat, CatsDogs, Epilepsy2, ItalyPowerDemand.

| role | dataset | source URL | downloaded_utc | bytes | sealed | values loaded |
|---|---|---|---|---|---|---|
| D1 | BinaryHeartbeat | https://timeseriesclassification.com/aeon-toolkit/BinaryHeartbeat.zip | 2026-08-25T13:23:26Z | 73002597 | no (CLS-CONF target); **run terminated (compute-infeasible)** | yes, after format conversion |
| D2_sealed | CatsDogs | https://timeseriesclassification.com/aeon-toolkit/CatsDogs.zip | 2026-08-25T13:23:37Z | 54080451 | yes — **sol 2026-08-26: remain sealed; do not force-use because already downloaded** | no (zip open + member names only) |
| D3_reserve | Epilepsy2 | https://timeseriesclassification.com/aeon-toolkit/EpilepticSeizures.zip (equivalent official name; `aeon-toolkit/Epilepsy2.zip` returned 404.php) | 2026-08-25T13:26:20Z | 16220082 | yes — **sol 2026-08-26: remain sealed; do not force-use because already downloaded** | no (zip open + member names only) |

ItalyPowerDemand was eligible but was the 4th name; it was not downloaded.

## D1

- Original download retained: `data/ucr_conf_downloaded/D1/BinaryHeartbeat_aeon_source.zip`
- Loader zip: `data/ucr_conf_downloaded/D1/BinaryHeartbeat.zip` (`BinaryHeartbeat_TRAIN.txt`, `BinaryHeartbeat_TEST.txt`)
- Conversion: `_scratch/convert_ucr_ts_to_txt_zip.py` from ARFF (the `.ts` members are concatenated multi-problem dumps). Value tokens copied; ARFF class names `{Abnormal,Normal}` mapped to `0/1` in declared order.
- Source members (original zip): BinaryHeartbeat.pdf, BinaryHeartbeat.png, BinaryHeartbeat.txt, BinaryHeartbeatEq.arff, BinaryHeartbeat_TEST.arff, BinaryHeartbeat_TEST.ts, BinaryHeartbeat_TRAIN.arff, BinaryHeartbeat_TRAIN.ts
- **`--conf-dl-run` terminated 2026-08-26 09:55:06 (compute-infeasible).** Verdict `COMPUTE_BUDGET_EXCEEDED` / `INSTRUMENT_SCALE_MISMATCH`. Not a scientific negative; do not treat D1 as a completed CLS-CONF cell.

## D2_sealed

- Path: `data/ucr_conf_downloaded/D2_sealed/CatsDogs.zip`
- Member names only: CatsDogs.arff, CatsDogs.jpg, CatsDogs.pdf, CatsDogs.txt, CatsDogs_TEST.arff, CatsDogs_TEST.ts, CatsDogs_TRAIN.arff, CatsDogs_TRAIN.ts
- Reserved for a future A5 vs A3 book. Do not load values.
- **sol 2026-08-26:** remain sealed. Already-downloaded status does not authorize opening or substituting this zip for the terminated D1 run.

## D3_reserve

- Path: `data/ucr_conf_downloaded/D3_reserve/EpilepticSeizures.zip`
- Official metadata row name is Epilepsy2 (TRAIN 80 / TEST 11420 / length 178 / 2 classes / 1 channel). The site's published per-dataset zip uses the equivalent name EpilepticSeizures (same TRAIN/TEST/length/classes on the website table).
- Member names only: EpilepticSeizures/, EpilepticSeizures/EpilepticSeizures.png, EpilepticSeizures/EpilepticSeizures.txt, EpilepticSeizures/EpilepticSeizures_TEST.ts, EpilepticSeizures/EpilepticSeizures_TRAIN.ts, EpilepticSeizures/val.ts
- Structural-failure reserve only. Do not load values unless a future book promotes it after a classified structural failure of D1, and only before the first LLM spend.
- **sol 2026-08-26:** remain sealed. Compute-infeasible D1 is not a structural/load failure; do not promote or force-use this zip because it is already on disk.

## Isolation

- Yahoo / NOAA / NAB / SMD: not opened.
- Local `data/ucr_task_context`: name enumeration only; no values loaded.
- No fourth dataset was fetched.
