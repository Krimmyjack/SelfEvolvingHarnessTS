# R4D-A three-cell action decomposition -- machine reading

Evidence class: MECHANISM / INSTRUMENT; development only; not a capability or generalisation claim

Boundary: physical_fits=12 / hard cap 24 (target 12), llm=0, held_out_reads=0, max_time_index_read=3911 (frontier 4056), existing files edited=0, m_r0k store written=False.

Population: 4 blocks, 42 faces, 1680 rows (840 per program), 0 NOT_EVALUABLE.

## Reconciliation against the prediction store

- rows compared: 1680; L_rr reproduction rate 1.0; L_pp reproduction rate 1.0 (tolerance |delta| < 1e-09)
- max |delta|: L_rr 1.3322676295501878e-15, L_pp 8.881784197001252e-16; offenders 0
- route / ctx readings are reported as registered

| program | n | L_rr ok | bit-identical | max d | L_pp ok | bit-identical | max d |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | 840 | 840 | 788 | 1.3322676295501878e-15 | 840 | 788 | 6.661338147750939e-16 |
| W2_pmc_then_outlier_mad | 840 | 840 | 788 | 1.3322676295501878e-15 | 840 | 791 | 8.881784197001252e-16 |

## Three-cell decomposition

| program | scope | n | route share | severe (total>0.30) | route worst share | route mean | route med | ctx mean | ctx med | rho(route,total) | rho(ctx,total) | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ANCESTOR | support_face | 420 | 0.768009429 | 28 | 0.964285714 | -0.163761133 | -0.083867398 | -0.081666133 | 0.0 | 0.901856263 | 0.21428704 | **ROUTE_DOMINANT** |
| ANCESTOR | delayed_face | 420 | 0.80376688 | 31 | 0.967741935 | -0.161654515 | -0.105627802 | -0.062990815 | 0.0 | 0.937469357 | 0.245078368 | **ROUTE_DOMINANT** |
| ANCESTOR | both_faces | 840 | 0.784907566 | 59 | 0.966101695 | -0.162707824 | -0.093033272 | -0.072328474 | 0.0 | 0.920054511 | 0.228908554 | **ROUTE_DOMINANT** |
| W2_pmc_then_outlier_mad | support_face | 420 | 0.714483157 | 60 | 0.65 | -0.256276681 | -0.112454517 | -0.016041217 | 0.0 | 0.867865464 | 0.29190122 | **ROUTE_DOMINANT** |
| W2_pmc_then_outlier_mad | delayed_face | 420 | 0.703445421 | 75 | 0.786666667 | -0.174467469 | -0.142246309 | -0.010535302 | 0.0 | 0.846592425 | 0.255882915 | **ROUTE_DOMINANT** |
| W2_pmc_then_outlier_mad | both_faces | 840 | 0.709106337 | 135 | 0.725925926 | -0.215372075 | -0.119121277 | -0.01328826 | 0.0 | 0.857872308 | 0.273619555 | **ROUTE_DOMINANT** |

Sign agreement (share of instances whose component sign equals total's):

| program | scope | route vs total | ctx vs total | route vs ctx |
| --- | --- | ---: | ---: | ---: |
| ANCESTOR | support_face | 0.94047619 | 0.276190476 | 0.216666667 |
| ANCESTOR | delayed_face | 0.945238095 | 0.257142857 | 0.202380952 |
| ANCESTOR | both_faces | 0.942857143 | 0.266666667 | 0.20952381 |
| W2_pmc_then_outlier_mad | support_face | 0.871428571 | 0.507142857 | 0.378571429 |
| W2_pmc_then_outlier_mad | delayed_face | 0.869047619 | 0.519047619 | 0.388095238 |
| W2_pmc_then_outlier_mad | both_faces | 0.870238095 | 0.513095238 | 0.383333333 |

D5 (11 windows, quoted, not recomputed, not covered by this package): ROUTE_DOMINANT, route share 0.73, severe route-worst 8/10.

## Serving window unmodified by the Program (astra 2.1 recomputed)

_apply_program(context, compiled) equals _linear_integrity(context) value by value, i.e. _prepare's moved count is 0 on the 192-step served window

| program | scope | evaluable | unmodified | share | total mean (unmod) | total median (unmod) | total>0.30 (unmod) | severe total | severe share on unmod | ctx==0 on unmod |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ANCESTOR | support_face | 420 | 236 | 0.561904762 | -0.156326715 | -0.083867398 | 0.06779661 | 28 | 0.571428571 | 236 |
| ANCESTOR | delayed_face | 420 | 269 | 0.64047619 | -0.165911426 | -0.106531812 | 0.089219331 | 31 | 0.774193548 | 269 |
| ANCESTOR | both_faces | 840 | 505 | 0.601190476 | -0.161432234 | -0.088251758 | 0.079207921 | 59 | 0.677966102 | 505 |
| W2_pmc_then_outlier_mad | support_face | 420 | 75 | 0.178571429 | -0.042357851 | -0.046277591 | 0.133333333 | 60 | 0.166666667 | 75 |
| W2_pmc_then_outlier_mad | delayed_face | 420 | 42 | 0.1 | -0.160801183 | -0.183461843 | 0.19047619 | 75 | 0.106666667 | 42 |
| W2_pmc_then_outlier_mad | both_faces | 840 | 117 | 0.139285714 | -0.08487597 | -0.063425959 | 0.153846154 | 135 | 0.133333333 | 117 |

## Block training-corpus invariance (R4C section 4, re-checked)

- [0:40]: 200 windows from 20 train series, anchors retained 10/10, origins checked 14, identical window sets True, identical train series True
- [120:160]: 200 windows from 20 train series, anchors retained 10/10, origins checked 6, identical window sets True, identical train series True
- [40:80]: 200 windows from 20 train series, anchors retained 10/10, origins checked 18, identical window sets True, identical train series True
- [80:120]: 200 windows from 20 train series, anchors retained 10/10, origins checked 4, identical window sets True, identical train series True

## Fits

{"physical_fits": 12, "physical_fits_hard_cap": 24, "physical_fits_target": 12, "physical_fits_by_model": {"[0:40]|ANCESTOR": 1, "[0:40]|W2_pmc_then_outlier_mad": 1, "[0:40]|raw": 1, "[120:160]|ANCESTOR": 1, "[120:160]|W2_pmc_then_outlier_mad": 1, "[120:160]|raw": 1, "[40:80]|ANCESTOR": 1, "[40:80]|W2_pmc_then_outlier_mad": 1, "[40:80]|raw": 1, "[80:120]|ANCESTOR": 1, "[80:120]|W2_pmc_then_outlier_mad": 1, "[80:120]|raw": 1}, "fit_counting_rule": "one _serve call = one physical Ridge fit", "physical_fits_scope": "this run only; the ledger cannot see earlier runs, so a package that replays the script must add the runs up itself when reporting against the cap", "llm_calls": 0, "held_out_reads": 0, "held_out_frontier": 4056, "held_out_origins": [4056, 4296, 4536, 4776, 5016], "max_time_index_read": 3911, "raw_series_window_reads": 1640, "existing_files_edited": 0, "m_r0k_prediction_store_written": false, "new_prediction_store": "_scratch/r4d_a_three_cell_store.json", "new_sha_or_hash": 0, "git_commits": 0, "sub_agents_spawned": 0}

