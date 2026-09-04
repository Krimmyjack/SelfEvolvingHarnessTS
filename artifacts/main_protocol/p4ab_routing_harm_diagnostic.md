# p4ab routing-harm diagnostic

0 LLM. Ridge fits 106 / 400 (pooled 14, per-channel 92).

## (1) Per-window S / E / harm

| kind | origin | read | |S| | |E| | harmed_m | harmed_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| delayed | 1896 | 1944 | 6 | 6 | 0 | 0 |
| re_encounter | 1896 | 2136 | 9 | 7 | 4 | 3 |
| delayed | 2376 | 2424 | 5 | 7 | 2 | 0 |
| re_encounter | 2376 | 2616 | 2 | 9 | 1 | 0 |
| delayed | 2616 | 2664 | 7 | 5 | 1 | 0 |
| re_encounter | 2616 | 2856 | 15 | 10 | 3 | 2 |
| delayed | 2856 | 2904 | 2 | 9 | 1 | 0 |

## (2) AUC and harmed_severe

- S1_div: AUC=0.6495 CI=[0.5177, 0.8079] **DOES_NOT_SEPARATE** (n=46, harmed=12)
- S2_dist: AUC=0.511 CI=[0.3289, 0.7026] **DOES_NOT_SEPARATE** (n=46, harmed=12)
- S3_beh_dist: AUC=0.5968 CI=[0.4436, 0.75] **DOES_NOT_SEPARATE** (n=46, harmed=12)
- S4_mod_fraction: AUC=0.451 CI=[0.2511, 0.6261] **DOES_NOT_SEPARATE** (n=46, harmed=12)
- S4_mod_magnitude: AUC=0.4032 CI=[0.2544, 0.5548] **DOES_NOT_SEPARATE** (n=46, harmed=12)

- severe T262 @2136 gain=-0.5018 moved=2 NEW_ENTRANT
- severe T269 @2136 gain=-0.4209 moved=9 CONTINUING
- severe T270 @2136 gain=-0.3369 moved=1 CONTINUING
- severe T260 @2856 gain=-0.8834 moved=6 NEW_ENTRANT
- severe T263 @2856 gain=-0.6446 moved=1 NEW_ENTRANT

routing_harm_check: **ROUTING_HARM_NOT_DOMINANT** (moved<=1 share=0.4)

## (3) pooled vs per-channel (re-encounter)

- 2136 pooled new-harmed=1 msh=0.5018 | per-channel new-harmed=4 msh=0.0814
- 2616 pooled new-harmed=0 msh=0.0904 | per-channel new-harmed=1 msh=0.0562
- 2856 pooled new-harmed=3 msh=0.8834 | per-channel new-harmed=5 msh=0.1403

## (4) Verdicts

- separation/total: **NO_OUTCOME_FREE_SEPARATOR**
- routing: **ROUTING_HARM_NOT_DOMINANT**
- counterfactual: **NO_CLEAR_DIFFERENCE**

## (5) Fits

{"llm_calls": 0, "consumer_fits": 106, "consumer_fits_pooled": 14, "consumer_fits_per_channel": 92, "held_out_reads": 0, "thresholds_changed": 0, "operators_added": 0, "artifacts_overwritten": 0, "fit_cap": 400}

AUC is in-sample separation on already-exposed windows, not a claim that harm is predictable at deployment.

## (6) Deviations

- S1 prediction capture: `scoped_evaluate` computes `raw_prediction`/`program_prediction` but does not return them; this audit copies the 2-fit path in-script and does not edit `scoped_serving_evaluator.py`.
- 22-d binned features: this checkout's numeric observable vocabulary is 12 names; S2 uses those frozen bins. Public card is 21 keys, not 22.
- Optional Support-A secondary window set: not run; main window set completed first and is the registered reading.

## (7) Spec tensions

- `scoped_evaluate` return dict omits predictions — captured in-script; evaluator left untouched.
- Round 2856 `delayed_gate.passes=False` vs `delayed_event` approved — this audit uses `delayed_gate.per_series_gain`, not `delayed_event`.

