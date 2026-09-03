# Slow Scope/Risk self-update

**Overall: `SLOW_ABSTAINS`** -- the Slow Agent returned the no_proposal envelope on all 2 attempts (no_authorized_minimal_edit, insufficient_public_evidence); this outcome is not in the pre-registered set and is reported as it stands

The closing run adopted `outlier_mad` on pooled task_C in both arms: aggregate delayed +0.029688, evaluation series 99999904140 down 0.125557. This slice runs Runtime attribution over that episode, lets the Slow Agent patch one authorized Scope/Risk surface through the deterministic EditController, and replays the same already-exposed window. The replay is development-level and claims nothing about held-out performance.

## B1 -- where the fold puts the fault

| episode | with the per-series risk reading | aggregate only |
| --- | --- | --- |
| `a5_pooled` | `RISK_GAP` at OUTCOME_RISK | `NO_ACTIONABLE_FAULT` |
| `a3_pooled` | `SELECTION_MISS` at CANDIDATE_SELECTION | `SELECTION_MISS` |

Primary cell `a5_pooled`, cause `RISK_GAP`. The fold, the route table and the 0.005 risk epsilon are the old line's, unchanged.

## B2 -- the patch

- Verdict: `SLOW_ABSTAINS`.
- the Slow Agent returned the no_proposal envelope on all 2 attempts (no_authorized_minimal_edit, insufficient_public_evidence); this outcome is not in the pre-registered set and is reported as it stands

## Cost and integrity

- LLM calls: 2 / 15.
- Consumer retrains: 0 / 200.
- Experience rows written (provenance `slow_scope_update`): None.
- Frozen surface: 32 files, drift [].
- Wall seconds: 59.4.
