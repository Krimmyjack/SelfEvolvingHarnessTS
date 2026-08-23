# #42g L1 Static vs A3

verdict: **ADAPTATION_HARMS_HELDOUT**

Part 0 sha: `21bff0431a88d9724b86fd7b09db2d6707b96ab7`

evidence_grade: DEVELOPMENT / first sealed Yahoo A1 exam. Not a Capability claim.

## Protocol facts

- Roster: freeze-list lexicographic first 24; lengths match r1 (sum 34507, held-in 24140, held-out 10367). Remaining 41 series stayed SEALED.
- Smoke `--deploy-fast-only-smoke` on Source fixture: `DEPLOY_FAST_ONLY_SMOKE_OK` (`open_delayed=0`, Slow=0, store hash unchanged, 0 LLM, Yahoo unread).
- Official held-in run `20260824T002018Z`: two fixed rounds, A3 started from h0. Leftover store still h0 (no learned Skill file). Fast-only deploy store also h0.
- First-run crash: Part D treated winner alias `extreme-deviation-mad` as a menu name (`INSTRUMENT` / mapping). Held-out vault had **not** been opened yet. Repair mapped alias → `outlier_mad` and scored Part D only (`--l1-score-heldout-only`). Held-in was **not** rerun.

## Missing pre-registrations (self-declared)

- Per-round held-in cell table (pool / chosen / probes / delayed / first-positive cost) was in memory and lost when the first process died before writing `OUT_L1`.
- Held-in LLM call count and AD-fit count of run `20260824T002018Z` were not persisted.
- Direction-consistency (held-in delayed sign vs held-out) cannot be computed without those cells.
- These are instrument gaps, not scientific retries.

## Held-out macros (24 series)

| arm | macro event-F1 | Δ vs identity |
|---|---:|---:|
| identity (primary Static) | 0.3227 | 0 |
| hampel_filter (pressure baseline) | 0.2655 | −0.0572 |
| A3* = outlier_mad | 0.2624 | **−0.0602** |

best_static = identity 0.3227.

A3* harmed vs identity (Δ < −0.005): 7/24 = real_10, real_11, real_12, real_19, real_23, real_25, real_3.  
worst-series Δ = **−0.6667** (real_11).

hampel harmed 12/24 (pressure baseline did harm; A3* did not beat identity).

Ladder: not PROTOCOL_BREACH / budget / unreadable → **ADAPTATION_HARMS_HELDOUT** (Δ vs identity −0.060 < −0.005 and harmed 7 > 2).

## Cost that is on disk

Part D scoring fits = 72 (24 × 3 arms). LLM 0 on the scoring pass. First-run held-in LLM/fit unknown.
