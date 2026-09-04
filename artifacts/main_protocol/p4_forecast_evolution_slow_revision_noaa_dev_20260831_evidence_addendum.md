# Forecast P4 Slow-revision evidence addendum (2026-08-31)

This addendum does not rewrite or upgrade the immutable terminal artifact
`p4_forecast_evolution_slow_revision_noaa_dev_20260831.json`.  It records two
reporting interpretations found during the post-run result-to-claim review.

1. `stage_1.local_fault=true` in the terminal artifact means that the Runtime
   probe Episode associated with the old Skill had a `NEGATIVE` or `CONFLICT`
   relation.  It is not a qualifying causal old-Skill fault because the same
   record has `fast_selected_v1=false` and `runtime_executed_v1=false`.  The
   qualifying causal `local_fault` for any future execution is therefore
   false unless those causal-use gates are also true.
2. The Stage-3 `BUDGET_EXHAUSTED_ABSTAIN_IDENTITY` text is an unreachable-stage
   placeholder in this artifact, not an observed budget exhaustion.  Stage 3
   was not authorized because no v2 was approved.  The authoritative budget
   fields are `llm_budget_exhausted=false`, 6/8 A5 calls, 5/8 K0 calls, and
   zero blocked calls.
3. `constrained_proposal_succeeds=false` means that no non-Skill candidate
   became a qualifying Runtime winner after the full safety relation was
   applied.  It does not mean every non-Skill aggregate probe was numerically
   non-positive: `localized_outlier_mad` had aggregate probe gain
   `+0.2539601691162441` but the winner remained empty.

The scientific disposition remains
`OLD_SKILL_NOT_CAUSALLY_USED__H3_HELD`.  No Slow PATCH, pending candidate,
Support-B decision, promotion, revision 2, or causal re-encounter occurred.
The pre-frozen repeat is ineligible.  Natural Final reads remain zero.
