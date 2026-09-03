# P3 Evidence Addendum and P4 gate amendment

Date: 2026-08-30. This addendum does not replace or modify the historical P3
receipt. Its verdict remains `P3_UNIFIED_VERTICAL_INTEGRATION_PASS__P4_HELD`.

## Classification evidence clarification

All values below are post-run diagnostics on exposed official TRAIN surfaces.
Confusion matrices use rows=true class and columns=predicted class. Prediction
flips compare identity-trained Ridge to Hampel-on-fit-trained Ridge; both score
the same unmodified Support rows.

| Surface | fit n | scored n | class n | Identity | Hampel | changed predictions | wrong→right | right→wrong |
|---|---:|---:|---|---|---|---:|---:|---:|
| Epilepsy2 Support-A | 40 | 20 | 10/10 | `[[7,3],[1,9]]` | `[[8,2],[1,9]]` | 1 | 1 | 0 |
| Epilepsy2 Support-B | 40 | 20 | 10/10 | `[[6,4],[0,10]]` | `[[8,2],[0,10]]` | 2 | 2 | 0 |
| PowerCons Support-A | 90 | 44 | 22/22 | `[[17,5],[1,21]]` | `[[17,5],[2,20]]` | 7 | 3 | 4 |

The unique scored surface count is 84, not the sum of repeated calls. P3
revalidation reused the same Epilepsy2 A/B rows. Conflict, K0 replay and the
controlled narrowed-policy replay reused the same PowerCons A rows.

On PowerCons, `A5−K0=+0.02233` means same-surface harm avoidance: changing from
K0 Hampel to controlled identity changes the same seven predictions in the
reverse direction (4 wrong→right, 3 right→wrong), for one additional correct
sample. It is not an independent performance gain. The Epilepsy2 revalidation
is same-surface deterministic replay, not independent generalization.

## AD evidence clarification

The primary #44a interpretation is `INVERTED_EFFECT_OBSERVED`: the fixed
Consumer detects the intervention, but cleaning moves Event-F1 in the opposite
direction from the intended improvement. AD therefore carries no positive
performance claim. Its permitted role is limited to Task/Consumer
conditioning, signal protection, safe refusal and no-negative-transfer
evidence. The Consumer, metric and event matching remain frozen and will not be
changed to seek a positive sign.

## Minimal operational amendment to v1.2

This amendment restores the separation already stated in v1.2 sections 12.2,
13.1 and 14:

1. `P4-Performance` collects H1/H2 evidence. An unexercised RQ3 event chain
   does not block Forecast or Classification performance collection.
2. `P4-Evolution` is the independent H3 gate. It remains held until a natural
   production chain reaches pending, independent Support-B approval,
   promotion, versioned revision, independent later re-encounter and material
   improvement over K0-fixed.
3. `P4-AD` is restricted to conditioning and safety. It cannot support a
   positive Event-F1 or online-revision claim.
4. Historical split-1 used 8 Evolution episodes per Task,
   Forward/Reverse/Interleaved replicas, Static/A3-reset/K0-fixed/A5-online and
   the Primary B=4 vector. K0 and A5 started with identical task-local
   knowledge; only A5 could write back across units. The Forecast Performance
   component froze eight natural exposed-KDD origins (600 through 936 in
   horizon-sized steps); no controlled or injected witness was Performance
   evidence. `Parallel Best-of-N@4` was the split-1 equal-budget H2 comparator,
   not a fifth Harness arm.
5. Evolution/Validation may run, but Natural Final, UCR TEST and the sealed AD
   series remain closed.

### Prospective split-2 amendment: Forecast P4 at B=8

The user-directed split-2 amendment supersedes split-1 only for future Forecast
P4-Performance collection. It is a disclosed operating-point change made after
the B=4 launch and budget-instrument failures; it does not retroactively turn
B=8 into the original v1.2.1 Primary point, and B=4 and B=8 rows must not be
pooled.

- The Forecast B=8 vector is the already specified v1.2.1 Budget-Curve vector:
  8 full Support Consumer evaluations, 24 cheap statistical/verifier probes,
  6 LLM calls, 60,000 input+output tokens, at most 1 accepted update, and 2700
  wall-clock seconds per stochastic method-cell.
- The face split is frozen prospectively as at most 7 Support-A full
  evaluations plus at most 1 independent Support-B promotion evaluation.
- A3-reset, K0-fixed and A5-online receive the exact same vector. A5 receives
  no extra calls, fits, tokens, probes, updates or time; its only existing
  difference remains permission to write back across units.
- The matched H2 comparator becomes `Parallel Best-of-N@8`: Support-A evaluates
  exactly these seven existing Common-DSL programs in frozen order:
  `impute_linear`, `hampel_filter`, `winsorize`, `outlier_iqr`, `impute_fft`,
  `impute_ema`, `period_complete`; only the Support-A winner is evaluated once
  on Support-B. No new Operator, Prompt, threshold or duplicate candidate is
  introduced.
- This amendment does not change the Classification operating point and does
  not expand AD beyond conditioning/safety. A separate prospective amendment
  is required before either Task can use B=8.
- All B=4 FAILED artifacts and their split-1 gate remain historical instrument
  records. A B=8 run must start with a new output and a fresh counter ledger.
  Natural Final, UCR TEST and sealed AD outcomes remain unread.

### Prospective split-3 instrument correction: eight LLM calls and local abstain

The split-2 live artifact remains an unmodified, non-scientific instrument
failure: its six-call cell cap stopped the whole run. For the next clean
Forecast P4-Performance run only, the user-directed correction is:

- A3-reset, K0-fixed and A5-online each have the same maximum of 8 LLM calls
  per method-cell; A5 has no exception. All other B=8 resource ceilings and
  the frozen methods, data, seeds, Prompt, Consumer and thresholds are
  unchanged.
- A ninth attempted call is rejected before reaching the backend and is not
  charged. The completed eight calls and their actual tokens/time remain cost.
- `LLM_CELL_BUDGET_EXHAUSTED` ends thinking for that cell only. The cell
  atomically discards partial state, selects identity, records
  `BUDGET_EXHAUSTED_ABSTAIN_IDENTITY`, and the runner continues with later
  arms, units and replicas.
- Global LLM exhaustion, token/time overflow, transport failures and all other
  protocol or data errors still fail closed.
- Exhaustion count and rate are reported separately for A3-reset, K0-fixed and
  A5-online as cost/efficiency outcomes. Identity fallback rows remain in the
  pre-registered paired H1/H2 estimands; they are not deleted as missing data.

This is a prospective operating/instrument correction, not a rewrite of the
historical split-2 gate or failed output. It adds no SHA, manifest or hash
infrastructure and does not open Natural Final, UCR TEST or sealed AD outcomes.

The first split-3 launch attempt stopped before any completed unit because its
transient credential was duplicated by the launch terminal, producing a 401 on
the first backend request. Its JSON/log/receipt remain a non-scientific
authentication-instrument record. The clean retry therefore uses the distinct
`p4_forecast_performance_b8_llm8_run2_20260830.json` output without changing
any experimental field.
