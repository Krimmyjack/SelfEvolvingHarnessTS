# AD Skill v3

verdict: **SOURCE_RISK_ONLY_TRIGGERABLE**

skill_id: `source_investigation_ad_v3`

h0s_v3: `f2054da1d18e2059457ed62282b7f7ff972ae219aedf98b39204ba2009bd7914`

delivery: True (A5 retrieves v3; A3 does not; both Memory empty at construction)

v1/v2: superseded, not deleted, not in h0s_v3

entry is in `t6_nab_42e_source_skill_v3.json` for deterministic rebuild. Temp Store not committed.

audit: containment + temporal both pass (1/2 Slow calls)

## sections

### WHEN

Apply only when task_kind is anomaly_detection.

### OBSERVE

Confirm that the current Workspace identifies task_kind as anomaly_detection before using this guidance.

### TRY

NO_AUTHORIZED_ACTIVE_RECOMMENDATION

### RISK

Lower proposal priority for hampel_filter, outlier_iqr, and outlier_mad by default. Under strong public Pattern evidence, each may still be a restricted probe candidate.

### VERIFY

Current Target Support relation POSITIVE forms a Draft; later delayed relation POSITIVE approves or keeps Active.

### FALLBACK

If task_kind is not anomaly_detection or is not established, do not apply this entry and retain the current approach.

