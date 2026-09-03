# #42e1 v3 behavior acceptance

verdict: **RISK_PRIOR_EFFECT_AMBIGUOUS**

L2: **CLOSED** (v3 archived; #42g L2 does not open)

Part 0 sha: `20218007b9f7695065567339199543510dcaeb8d`

run_id: `20260823T213436Z`

evidence_grade: DEVELOPMENT / same-context. Not a Capability claim.
h0s_v3: `f2054da1d18e2059457ed62282b7f7ff972ae219aedf98b39204ba2009bd7914`
LLM 22 / 32; AD fit 12 / 24; retrain 0
wall breached: False; target_key_requests: 6

| cell | v3? | pool | chosen | ham/iqr/mad prop | probes | win p/s | non-id | delayed | harmS/D | status |
|---|---|---|---|---|---|---|---|---|---|---|
| target_cpc/A3/r1 | False | ['identity'] |  | 0/0/0 | 0/0/0 | 0/0 | 0 | None | 0/0 | — |
| target_cpc/A5/r1 | True | ['identity'] |  | 0/0/0 | 0/0/0 | 0/0 | 0 | None | 0/0 | — |
| target_cpc/A3/r2 | False | ['identity'] |  | 0/0/0 | 0/0/0 | 0/0 | 0 | None | 0/0 | — |
| target_cpc/A5/r2 | True | ['identity'] |  | 0/0/0 | 0/0/0 | 0/0 | 0 | None | 0/0 | — |
| target_cpm/A5/r1 | True | ['identity'] |  | 0/0/0 | 0/0/0 | 0/0 | 0 | None | 0/0 | — |
| target_cpm/A3/r1 | False | ['identity'] |  | 0/0/0 | 0/0/0 | 0/0 | 0 | None | 0/0 | — |
| target_cpm/A5/r2 | True | ['identity', 'localized_outlier_mad', 'localized_hampel_filter'] | localized_outlier_mad | 1/0/1 | 0/0/1 | 0/0 | 1 | 0.011111111111111113 | 0/0 | [('outlier_mad', 'POSITIVE', 'LOCAL_ACTIVE')] |
| target_cpm/A3/r2 | False | ['identity', 'outlier_mad_local_extreme_deviation'] | outlier_mad_local_extreme_deviation | 0/0/1 | 0/0/1 | 0/0 | 1 | 0.011111111111111113 | 0/0 | [('outlier_mad', 'POSITIVE', 'LOCAL_ACTIVE')] |

## CPM r2 (sharp cell)

{
  "A3": {
    "cell": "target_cpm/A3/r2",
    "arm": "A3",
    "cohort": "target_cpm",
    "round": "r2",
    "retrieved_v3": false,
    "retrieved_skill_ids": [
      "build_contrastive_candidates",
      "inspect_and_localize",
      "select_or_identity_and_verify"
    ],
    "held": 0,
    "pool": [
      "identity",
      "outlier_mad_local_extreme_deviation"
    ],
    "chosen": "outlier_mad_local_extreme_deviation",
    "probe_order": [
      "outlier_mad_local_extreme_deviation"
    ],
    "probes": [
      {
        "candidate_id": "outlier_mad_local_extreme_deviation",
        "kind": "probe",
        "gain": 0.05925925925925926
      }
    ],
    "proposed": {
      "hampel_filter": 0,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 1
    },
    "shortlisted": {
      "hampel_filter": 0,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 1
    },
    "probed": {
      "hampel_filter": 0,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 0
    },
    "selected": {
      "hampel_filter": 0,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 0
    },
    "vacated_slot_occupants": [
      "identity"
    ],
    "risk_positive_events": [
      {
        "workflow": "outlier_mad",
        "relation": "POSITIVE",
        "status": "LOCAL_ACTIVE",
        "layer": "DELAYED"
      }
    ],
    "override": "NO_OVERRIDE_OCCASION",
    "harm_support": 0,
    "harm_delayed": 0,
    "delayed_utility": 0.011111111111111113,
    "delayed_event": {
      "stage": "approved",
      "delayed_gain": 0.011111111111111113,
      "delayed_relation": "POSITIVE",
      "delayed_evidence": {
        "relation": "POSITIVE",
        "classification_basis": "aggregate >= +0.005 and every per-series reading >= -0.005",
        "material_threshold": 0.005,
        "consumer_id": "aegists_iforest_v1",
        "aggregate_gain": 0.011111111111111113,
        "aggregate_direction": "improved",
        "series_read": 3,
        "harmed_series_count": 0,
        "harmed_series": [],
        "min_per_series_gain": 0.0
      },
      "snapshot_updated": true
    },
    "approved_skill_id": "fast_winner_anomaly_detection_aegists_iforest_v1_macro_event_f1_outlier_mad",
    "activated": true,
    "episode_rows": [
      {
        "episode_id": "target_cpm_target_outlier_mad_a3_r2_p1",
        "task_consumer_key": "anomaly_detection|aegists_iforest_v1|macro_event_f1",
        "domain_namespace": "target_cpm",
        "workflow_signature": "outlier_mad",
        "relation": "POSITIVE",
        "evidence_level": "DELAYED",
        "local_status": "LOCAL_ACTIVE"
      }
    ],
    "non_identity_trials": 1,
    "abstained": false,
    "winsorize_proposed": 0,
    "winsorize_selected": 0,
    "cite": {
      "retrieved_v3": false,
      "cites_risk_knowledge": false,
      "names_risk_operator_in_propose": false,
      "excerpt": "",
      "cell_blob_cites": false
    }
  },
  "A5": {
    "cell": "target_cpm/A5/r2",
    "arm": "A5",
    "cohort": "target_cpm",
    "round": "r2",
    "retrieved_v3": true,
    "retrieved_skill_ids": [
      "build_contrastive_candidates",
      "inspect_and_localize",
      "select_or_identity_and_verify",
      "source_investigation_ad_v3"
    ],
    "held": 0,
    "pool": [
      "identity",
      "localized_outlier_mad",
      "localized_hampel_filter"
    ],
    "chosen": "localized_outlier_mad",
    "probe_order": [
      "localized_outlier_mad",
      "localized_hampel_filter"
    ],
    "probes": [
      {
        "candidate_id": "localized_outlier_mad",
        "kind": "probe",
        "gain": 0.05925925925925926
      }
    ],
    "proposed": {
      "hampel_filter": 1,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 1
    },
    "shortlisted": {
      "hampel_filter": 1,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 1
    },
    "probed": {
      "hampel_filter": 0,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 0
    },
    "selected": {
      "hampel_filter": 0,
      "outlier_iqr": 0,
      "outlier_mad": 1,
      "winsorize": 0,
      "identity": 0
    },
    "vacated_slot_occupants": [
      "identity"
    ],
    "risk_positive_events": [
      {
        "workflow": "outlier_mad",
        "relation": "POSITIVE",
        "status": "LOCAL_ACTIVE",
        "layer": "DELAYED"
      }
    ],
    "override": "NO_OVERRIDE_OCCASION",
    "harm_support": 0,
    "harm_delayed": 0,
    "delayed_utility": 0.011111111111111113,
    "delayed_event": {
      "stage": "approved",
      "delayed_gain": 0.011111111111111113,
      "delayed_relation": "POSITIVE",
      "delayed_evidence": {
        "relation": "POSITIVE",
        "classification_basis": "aggregate >= +0.005 and every per-series reading >= -0.005",
        "material_threshold": 0.005,
        "consumer_id": "aegists_iforest_v1",
        "aggregate_gain": 0.011111111111111113,
        "aggregate_direction": "improved",
        "series_read": 3,
        "harmed_series_count": 0,
        "harmed_series": [],
        "min_per_series_gain": 0.0
      },
      "snapshot_updated": true
    },
    "approved_skill_id": "fast_winner_anomaly_detection_aegists_iforest_v1_macro_event_f1_outlier_mad",
    "activated": true,
    "episode_rows": [
      {
        "episode_id": "target_cpm_target_outlier_mad_a5_r2_p1",
        "task_consumer_key": "anomaly_detection|aegists_iforest_v1|macro_event_f1",
        "domain_namespace": "target_cpm",
        "workflow_signature": "outlier_mad",
        "relation": "POSITIVE",
        "evidence_level": "DELAYED",
        "local_status": "LOCAL_ACTIVE"
      }
    ],
    "non_identity_trials": 1,
    "abstained": false,
    "winsorize_proposed": 0,
    "winsorize_selected": 0,
    "cite": {
      "retrieved_v3": true,
      "cites_risk_knowledge": false,
      "names_risk_operator_in_propose": false,
      "excerpt": "",
      "cell_blob_cites": true
    }
  }
}

## verdict detail

{
  "verdict": "RISK_PRIOR_EFFECT_AMBIGUOUS",
  "associated_misses_not_causal": [],
  "l2_opens": false,
  "v3": "archived",
  "note": "arms differ but no pre-registered effect cell fired"
}
