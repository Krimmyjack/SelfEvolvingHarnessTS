# AD Skill v2

verdict: SKILL_V2_FROZEN_PENDING_BEHAVIOR_REPLAY

v1 `source_investigation_ad_v1` is superseded, not deleted, and is not in h0s_v2.

h0s_v2 runtime_bundle_sha: `9cd6ade6daa4a613b9057b3dbf55cf09d009f1dd6240b9cd0f96bf6dd903804a`

## sections

### WHEN

Apply when the proposal-time task_kind is anomaly_detection.

### OBSERVE

Observe the proposal-time task_kind and any strong public Pattern evidence before deciding.

### TRY

NO_AUTHORIZED_ACTIVE_RECOMMENDATION

### RISK

Lower the proposal priority of hampel_filter. Under strong public Pattern evidence, it may still be a restricted probe candidate.

### VERIFY

The current Target Support relation must be POSITIVE to form a Draft; the later delayed relation must be POSITIVE to approve or keep it Active.

### FALLBACK

When the observation does not support a guarded hypothesis, use the ordinary unmodified proposal path and gather further Target Support.


## delivery assert

A5 retrieves v2: True
A3 does not: True
both Memory empty: True / True
This proves delivery only. Behavior replay is a #42e/#42g precondition, not this book.
