# #42g-b menu headroom

verdict: **PARTIAL_SERIES_HEADROOM_ONLY**

flags: none (no `SELECTION_OR_LIFECYCLE_MISS_UNRESOLVED`; B1 failed so that flag is not eligible)

Part 0 sha: `92d5bf22825a1afd14d6f5efe87390578141475c`

eval zone: **development_exposed_eval** (the 24 EXPOSED series). True held-out = remaining **41 sealed** series; unread this book.

0 LLM / 0 retrain / 120 AD fits (cap 150). `methods/` not edited.

evidence_grade: DEVELOPMENT / first-fault localization. Not a Capability claim.

Claim cap if someone later quotes NO_MENU_HEADROOM (this book did **not** land that verdict): *under the in-service Consumer, five-program menu, and the pre-registered global/local bars, there is no actionable headroom.* Not “AD data has no treatable space.”

## Part 0

0a. Six-file allowlist committed as above. Secret-scan hits were scientific vocabulary (`authorization=` / runner params). Yahoo raw/vaults untracked.

0b. Three mechanical fixes in the runner (self-tested, not a new official L1 run):

- (a) A3* scoring uses only `deploy.applied_program`; Part D asserts `scored_program == deploy.applied_program`. Winner-op path removed from the scoring branch.
- (b) held-in / freeze / deploy each write their own artifact (`t6_42g_l1_held_in.json` / `_freeze.json` / `_deploy.json`) before any development_exposed_eval open.
- (c) `allow_slow` / `slow_agent` come from `L1_ALLOW_SLOW` / `L1_SLOW_AGENT` (this book: False / None). No hardcoded `allow_slow=False` at the L1 call site.

## B1 global (development_exposed_eval Δ vs identity)

Any program with macro Δ > +0.005 and harmed ≤ 2/24? **No.**

| program | macro Δ | harmed | pass |
|---|---:|---:|---|
| outlier_iqr | −0.0298 | 5 | no |
| outlier_mad | −0.0602 | 7 | no |
| hampel_filter | −0.0572 | 12 | no |
| winsorize | −0.0919 | 14 | no |

## B2 local (≥5 series Δ > +0.005)

Headroom is **not** tied to feature separability. Three programs clear the local bar.

| program | n improved | series | public-feature association |
|---|---:|---|---|
| outlier_iqr | **6** | real_10, 16, 24, 29, 3, 30 | `PUBLIC_FEATURE_ASSOCIATION_NOT_SEEN` |
| outlier_mad | **5** | real_1, 17, 2, 24, 26 | `PUBLIC_FEATURE_ASSOCIATION_SEEN` |
| hampel_filter | **7** | real_15, 17, 2, 21, 28, 29, 30 | `PUBLIC_FEATURE_ASSOCIATION_SEEN` |
| winsorize | 4 | real_10, 17, 20, 29 | not applicable (bar missed) |

Association is a side reading on in-register boolean public features. No threshold scan. No Scope emitted.

## B3 feedback preference (Agent-visible estimand = union of four feedback windows [.30n,.70n))

B1 did not pass, so miss-flag is **not** applied. Recorded anyway:

| subset | n | iqr | mad | hampel | winsorize |
|---|---:|---:|---:|---:|---:|
| all series (in-service macro) | 24 | −0.0567 | −0.0519 | −0.0541 | −0.0731 |
| event-bearing | 10 | −0.0362 | −0.0246 | −0.0298 | −0.0754 |
| zero-event | 14 | −0.0714 | −0.0714 | −0.0714 | −0.0714 |

No program is preferred on the in-service feedback estimand. Event-bearing subset also does not prefer any menu entry above +0.005. Zero-event subset is uniformly negative (silent-window F1 arithmetic). Therefore: **not** `SELECTION_OR_LIFECYCLE_MISS_UNRESOLVED`, and **not** `FEEDBACK_EVENT_STARVATION_OR_TARGET_MISMATCH` as a B1-winner mismatch. Sparsity itself is a fact: 14/24 series have **zero** feedback-window events.

## Event sparsity (EXPOSED, legal)

| series | r1 Support | r1 delayed | r2 Support | r2 delayed | development_exposed_eval | any feedback event |
|---|---:|---:|---:|---:|---:|---|
| real_1.csv | 0 | 0 | 0 | 0 | 2 | no |
| real_10.csv | 0 | 0 | 0 | 0 | 1 | no |
| real_11.csv | 0 | 0 | 0 | 0 | 1 | no |
| real_12.csv | 0 | 0 | 0 | 0 | 1 | no |
| real_13.csv | 0 | 0 | 0 | 1 | 1 | yes |
| real_14.csv | 0 | 0 | 0 | 0 | 0 | no |
| real_15.csv | 0 | 0 | 0 | 1 | 1 | yes |
| real_16.csv | 0 | 0 | 0 | 0 | 1 | no |
| real_17.csv | 0 | 0 | 1 | 1 | 2 | yes |
| real_18.csv | 0 | 0 | 0 | 0 | 0 | no |
| real_19.csv | 0 | 0 | 1 | 1 | 2 | yes |
| real_2.csv | 0 | 0 | 0 | 0 | 2 | no |
| real_20.csv | 1 | 0 | 0 | 0 | 2 | yes |
| real_21.csv | 0 | 0 | 0 | 0 | 2 | no |
| real_22.csv | 0 | 0 | 0 | 0 | 1 | no |
| real_23.csv | 0 | 3 | 2 | 0 | 2 | yes |
| real_24.csv | 0 | 0 | 0 | 0 | 2 | no |
| real_25.csv | 0 | 0 | 0 | 0 | 1 | no |
| real_26.csv | 0 | 0 | 0 | 0 | 5 | no |
| real_27.csv | 0 | 0 | 0 | 0 | 2 | no |
| real_28.csv | 0 | 0 | 0 | 1 | 2 | yes |
| real_29.csv | 0 | 0 | 1 | 0 | 3 | yes |
| real_3.csv | 1 | 0 | 0 | 0 | 1 | yes |
| real_30.csv | 1 | 0 | 0 | 0 | 1 | yes |

r1 Support events exist on only three series (20, 3, 30). Two series (14, 18) have no events in either zone.

## First-fault reading

Global menu has no actionable headroom on this Consumer + bar. Local headroom exists for iqr / mad / hampel, and current public features do **not** have to see it (iqr: association not seen). First official L1 “A3* = outlier_mad harmed held-out” is consistent with **no global headroom**; it does not by itself prove a selection miss, because (1) L1 trajectory was lost, (2) deploy mapping was wrong, (3) in-service feedback macros also dislike every non-identity program.

## Deliverables (not committed)

- `artifacts/functional/e2/t6_42g_b_menu_headroom.json`
- `artifacts/functional/e2/t6_42g_b_menu_headroom.md`
- runner wiring for 0b + `--menu-headroom-v1` (dirty vs 92d5bf2)
