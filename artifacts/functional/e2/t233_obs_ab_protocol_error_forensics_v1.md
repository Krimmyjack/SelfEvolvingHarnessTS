# T233 OBS-AB protocol-error forensics, first-fault classification

Read-only forensics over the delivered run of `evaluation/functional/run_t233_supply_obs_ab.py`
(`artifacts/functional/e2/t233_supply_obs_ab_v1.json`, mtime 2026-08-20 12:04:15).
Zero LLM calls, no experiment re-run, no source or artifact modified. This file is
the only write.

## Headline

13 `AGENT_PROTOCOL_ERROR` arm-runs out of 38 (34.2%). **All 13 fail at the same
stage (`inspect`) and all 13 classify as C** — the Agent emitted a payload that
violated a contract it had been given verbatim, and the harness rejected it
correctly. **Zero of the 13 are caused by this morning's driver (class A).**

There are two message families, not one, but they collapse to **one root
signature** for 11 of the 13, plus a **second, distinct signature** for the
remaining 2.

A genuine class-A defect does exist in the driver, but it is an **accounting**
defect that mis-reports these errors rather than causing any of them. It is
specified in "Minimal fix" below, and it invalidates one evidence claim in
`t233_supply_obs_ab_v1.md`.

## The contract that was violated

`methods/ttha/schemas/fast_inspect_v1.json` — frozen stage schema, `$comment`
self-dated `SKILL_CONTENT_MIGRATION 2026-08-14`, last touched by commit `a1d879a`
(long before today). Two sibling fields differ by exactly one level of nesting:

```12:24:methods/ttha/schemas/fast_inspect_v1.json
  "inspected_region_fractions": {
   "type": "array",
   "items": {
    "type": "array",
    "minItems": 2,
    "maxItems": 2,
    "items": {
     "type": "number",
```

```68:77:methods/ttha/schemas/fast_inspect_v1.json
     "region_fractions": {
      "type": "array",
      "minItems": 2,
      "maxItems": 2,
      "items": {
       "type": "number",
       "minimum": 0.0,
       "maximum": 1.0
      }
     },
```

So `inspected_region_fractions` is a **list of pairs**, while
`pattern_hypotheses[].region_fractions` is a **single flat pair** `[start, end]`
— one region per hypothesis, never a list of regions. A conforming value looks
like `t233_supply_obs_ab_v1.json:1635-1638` (`[0.0, 0.9856770833333334]`).

The whole schema, not just its name, is placed in the Agent's prompt:
`methods/ttha/agent_core.py:266` — `"stage_payload_schema": _plain(output_schema)`.
The Agent therefore had `minItems: 2 / maxItems: 2 / items.type: number` in front
of it on every call.

## Signature S1 — flat-pair field filled with a list of pairs (11 errors)

Both S1 messages are produced by the same validator on the same field, and the
choice between them is fully determined by how many nested pairs the Agent
emitted:

- `methods/ttha/schema_contracts.py:314` → `f"{path} has too few items"`, raised
  when `len(value) < minItems`. With `region_fractions = [[a, b]]` the length is
  1 < 2 → **"has too few items"**.
- `methods/ttha/schema_contracts.py:283` → `f"{path} has wrong type"`, raised when
  the element type does not match. With `region_fractions = [[a, b], [c, d]]` the
  length passes `minItems: 2`, then `items.type: number` is checked against
  element 0, which is an array → **"region_fractions[0] has wrong type"**.

One defect, two surface strings, split purely by arity. The 2026-08-19 evidence
below closes this inference: a single task produced *both* strings in its two
arms.

## Signature S2 — ungrounded `key=value` evidence citation (2 errors)

Raised by `methods/ttha/fast_agent.py:436-440`
(`StagePostValidationError("HYPOTHESIS_EVIDENCE_UNGROUNDED", ...)`, `retryable=True`),
reached through the `post_validator` hook at
`evaluation/functional/task_episode_harness/agentic/fast_path.py:444`.

The Agent cited `local_robust_z_peak=37.09699177121275` and
`estimated_level_offset=35.6` — the `key=value` spelling. `fast_path.py:221-268`
(`_normalize_evidence_citations`, committed as `3cf6a5a "Bounded lexical
compatibility: accept the spelling, never widen the evidence"`) strips `key=value`
down to `key` **only when the value string exactly matches a served value**
(`fast_path.py:258-260`). Neither value matched exactly, so the literal string
survived into the grounding check and was rejected as a non-public feature name.

Both cited keys are legitimate public features — they appear in accepted
hypotheses elsewhere in the same run (`t233_supply_obs_ab_v1.json:1640-1642`).
The rejection is about value-string exactness, not about the feature being hidden.
Neither key is one of the four masked M0b names
(`level_region_fraction`, `level_region_end_fraction`,
`outlier_region_end_fraction`, `level_only_post_shift_support_sufficient`,
listed at `t233_supply_obs_ab_v1.json:64-69`), and the NEW_OBS instance occurred
with `mask_active: false` (`:5256`).

## Why the errors were fatal rather than repaired

`fast_path.py:292` pins `validation_retries=1`, i.e. one repair attempt. The
retry loop at `methods/ttha/agent_core.py:450-457` (schema) and `:461-468`
(post-validation) feeds the Agent the exact validator message and asks for a
corrected envelope; only when that second attempt also fails does
`raise_with_validation_context(exc)` turn it into a fatal `AgentProtocolError`.

The retry demonstrably works when it works: NEW_OBS `task_09`
(`t233_supply_obs_ab_v1.json:4793-4800`) recorded
`first_pass_valid: false`, `validation_retry_count: 1`,
`validation_error_codes: ["STAGE_SCHEMA_INVALID"]` and then completed. The 13
fatal rows are the cases where the Agent repeated the same malformation after
being shown the error.

## Classification table

`census` = per-task census line in `t233_supply_obs_ab_v1.json`;
`rows` = full row line carrying the message; `summary` = the driver's own
`mask_artifact_check.protocol_errors` block.

| # | task | arm | stage | signature | class | census | rows | summary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | task_09 | OLD_OBS | inspect | S1 too few items | C | 562 | 4727 | 88 |
| 2 | task_10 | OLD_OBS | inspect | S1 too few items | C | 575 | 5019 | 92 |
| 3 | task_11 | OLD_OBS | inspect | S1 too few items | C | 588 | 5180 | 96 |
| 4 | task_12 | OLD_OBS | inspect | S1 `[0]` wrong type | C | 601 | 5341 | 100 |
| 5 | task_13 | OLD_OBS | inspect | S1 `[0]` wrong type | C | 614 | 5619 | 104 |
| 6 | task_15 | OLD_OBS | inspect | S1 `[0]` wrong type | C | 653 | 6172 | 108 |
| 7 | task_16 | OLD_OBS | inspect | S2 ungrounded `local_robust_z_peak=37.09699177121275` | C | 666 | 6333 | 112 |
| 8 | task_17 | OLD_OBS | inspect | S1 `[0]` wrong type | C | 679 | 6612 | 116 |
| 9 | task_10 | NEW_OBS | inspect | S1 too few items | C | 1222 | 5083 | 122 |
| 10 | task_11 | NEW_OBS | inspect | S2 ungrounded `estimated_level_offset=35.6` | C | 1235 | 5244 | 126 |
| 11 | task_14 | NEW_OBS | inspect | S1 `[0]` wrong type | C | 1307 | 6075 | 130 |
| 12 | task_15 | NEW_OBS | inspect | S1 `[0]` wrong type | C | 1320 | 6236 | 134 |
| 13 | task_19 | NEW_OBS | inspect | S1 `[0]` wrong type | C | 1420 | 7415 | 138 |

Class counts: **A = 0, B = 0, C = 13, D = 0.**
Signature counts: **S1 = 11** (4 "too few items" + 7 "`[0]` has wrong type`"),
**S2 = 2**. Stage: **13/13 at `inspect`**, 0 at `propose`, 0 at `select`.

### Why not A (our driver's glue)

- `run_t233_supply_obs_ab.py` touches `region_fractions` in exactly one place,
  `:292` — `"region_fractions": raw.get("region_fractions")` — inside
  `_slim_arm_result`, which **reads** the finished payload for the census. It
  never assembles an inspect payload, never names a stage, and never builds the
  schema.
- Stage naming, sequencing, retry policy and validation all live in committed
  harness modules: `fast_path.py` (last touched `31ffd77`, with
  `validation_retries=1` introduced by `8806745 "G1: connect the core Pipeline in
  one Runner package"`), `agent_core.py`, `fast_agent.py`,
  `schema_contracts.py`, `schemas/fast_inspect_v1.json`.
- `git status` confirms `run_t233_supply_obs_ab.py` is untracked (written today)
  while every module above is tracked and unmodified.

### Why not B (the pre-existing `stage_result names the wrong stage` drift)

Different error code, different message, different branch.
`agent_core.py:436-446` raises `AgentProtocolError("stage_result names the wrong
stage")` under error code `WRONG_STAGE`, on an **envelope-level** mismatch
between the requested stage and the returned `envelope["stage"]`. That is the
`test_ttha_agent` replay-backend drift counted among the 16 pre-existing failures
in `m0b_field_wiring_report_v1.md:84-88`.

The 13 errors here are raised at `agent_core.py:449-457` under
`STAGE_SCHEMA_INVALID` (S1) and `:461-468` under
`HYPOTHESIS_EVIDENCE_UNGROUNDED` (S2), on **payload** content, after the stage
name has already validated. No overlap.

### Why C, on the merits

The Agent received the full schema in-prompt, received the exact validator
message on retry, and re-emitted a malformed value anyway. The harness rejected a
genuinely non-conforming payload. That is the definition of C.

One qualification, recorded as-is rather than promoted to a class: the
`inspect_and_localize` bootstrap Skill in the h0 the run used
(`methods/ttha/harness/h0/skills/bootstrap/inspect_and_localize.json:6`,
`revision: 2`, byte-identical to the compiled copy at
`.t233_supply_obs_ab_state/OLD_OBS/e1v2_task_09/snapshots/c4cb24b8…/skills/bootstrap/inspect_and_localize.json`)
instructs: *"Preserve disjoint narrow hypotheses when a high robust-z signal spans
a broad low-density region instead of merging the whole span."* It names
`region_fractions` without stating its shape. That text pulls toward expressing
several sub-regions while the schema admits exactly one pair per hypothesis — a
standing tension between guidance prose and frozen schema. It is committed
harness content, added by `2db2f02 "fix: harden minipipe feedback and edit
lifecycle"`, long predating both M0b and today's driver. It is a **contributing
condition, not the first fault**: the schema was still supplied verbatim and was
still not followed.

## Judgement question 1 — one signature? which class is the first fault?

**Not one signature; two, at 11 + 2.** They share the stage (`inspect`), the
run-level outcome (`AGENT_PROTOCOL_ERROR`), and the class (C), but they are
distinct validations with distinct error codes:

| | S1 | S2 |
| --- | --- | --- |
| count | 11 | 2 |
| exception | `AgentProtocolError` | `StagePostValidationError` |
| error code | `STAGE_SCHEMA_INVALID` | `HYPOTHESIS_EVIDENCE_UNGROUNDED` |
| raised at | `agent_core.py:449` via `schema_contracts.py:283`/`:314` | `fast_agent.py:436` via `fast_path.py:444` |
| field | `pattern_hypotheses[0].region_fractions` | `pattern_hypotheses[].evidence_features` |

Within S1 the two message strings **are** one signature: same field, same
validator, same nested-list malformation, differing only by element count.

**The single first-fault class is C for both signatures**, and S1 is the
first fault by volume (11/13, 85%).

## Judgement question 2 — same origin as v1 `task_08` and as the test-suite drift?

**Same origin as the historical `task_08` error: yes, and it is the strongest
evidence in this report.**

The brief locates that error in `t233_independent_source_supply.json`. It is not
there: that file's 19 rows all carry `"protocol_error": null` (lines 30, 78, 136,
184, 232, 294, 342, 390, 438, 486, 548, 606, 663, 711, 759, 817, 875, 923, 981),
and `task_08`'s stop is `AGENT_ABSTAIN` (`:389`), not a protocol error. **The v1
independent supply run had zero protocol errors.**

The `task_08` protocol error is in `g3d1_t233_source_extension.json`
(mtime 2026-08-19 17:05, ~19 hours before this run), which carries a `first_fault`
block naming it outright:

```1789:1806:artifacts/functional/e2/g3d1_t233_source_extension.json
  "first_fault": {
    "first_fault": "AGENT_PROTOCOL_ERROR",
    "cause": "MECHANICAL_EXIT",
    "layer": "MECHANICAL",
    "editable": false,
    "occurrences": [
      {
        "task_episode_id": "e1v2_task_08",
        "arm": "A3",
        "error": "inspect: AgentProtocolError: payload.pattern_hypotheses[0].region_fractions has too few items"
      },
      {
        "task_episode_id": "e1v2_task_08",
        "arm": "A5",
        "error": "inspect: AgentProtocolError: payload.pattern_hypotheses[0].region_fractions[0] has wrong type"
      }
    ],
```

Character-for-character the same two strings, on the same `e1v2` T233 cohort, on
the same `A3` arm token, before today's driver existed. The repository had already
classified it `layer: MECHANICAL`, `editable: false`. That the *same task* threw
"too few items" in one arm and "`[0]` has wrong type" in the other is direct
confirmation that the two strings are one defect at different arity.

The lineage is older still: the same nesting confusion on the sibling field is
recorded as `payload.inspected_region_fractions[0] has wrong type` in
`w1_guidance_evolution_report.json:4774` (2026-08-16, with a
`STAGE_SCHEMA_INVALID` retry envelope) and `calib_a3_run02.json:965, 2019`
(2026-08-19 22:49).

**Same origin as the test-suite `stage_result names the wrong stage` drift: no.**
Different error code (`WRONG_STAGE` vs `STAGE_SCHEMA_INVALID`), different branch
(`agent_core.py:437-446` vs `:449-457`), envelope-level vs payload-level. Not
related.

## Judgement question 3 — random, or concentrated?

**Sharply concentrated, on a task-geometry covariate.** Every row records
`task_signature.estimated_region_start_fraction`, which partitions the roster
cleanly: `"zero"` for `task_01..task_09`, `"low"` for `task_10..task_19`
(`t233_supply_obs_ab_v1.json:1597, 2016, 2407, 2825, 3200, 3580, 3981, 4321, 4716`
= zero; `:5008, 5169, 5330, 5608, 5894, 6161, 6322, 6601, 6879, 7249` = low).

| geometry | tasks | arm-runs | errors | rate | S1 | S2 |
| --- | --- | --- | --- | --- | --- | --- |
| `estimated_region_start_fraction = "zero"` | task_01–09 | 18 | 1 | **5.6%** | 1 | 0 |
| `estimated_region_start_fraction = "low"` | task_10–19 | 20 | 12 | **60.0%** | 10 | 2 |

Tasks 01–08 produced **0 errors in 16 arm-runs**. The single "zero"-geometry
error is `task_09`, the last task before the boundary. This is mechanistically
coherent with S1: when the region starts at 0 the natural answer is one flat pair
`[0.0, x]`, and when the region starts at a low nonzero fraction the Agent has a
genuine interior sub-interval and the Skill text explicitly asks it to keep
disjoint sub-regions — which is exactly the nested-list shape the schema refuses.
The ~5.6% "zero" rate also matches the ~5% historical rate cited in the brief.

Per arm and per task:

| task | geometry | OLD_OBS | NEW_OBS |
| --- | --- | --- | --- |
| task_01–08 | zero | ok | ok |
| task_09 | zero | **S1** | ok (recovered on retry) |
| task_10 | low | **S1** | **S1** |
| task_11 | low | **S1** | **S2** |
| task_12 | low | **S1** | ok |
| task_13 | low | **S1** | ok |
| task_14 | low | ok | **S1** |
| task_15 | low | **S1** | **S1** |
| task_16 | low | **S2** | ok |
| task_17 | low | **S1** | ok |
| task_18 | low | ok | ok |
| task_19 | low | ok | **S1** |

| arm | errors / 19 | rate |
| --- | --- | --- |
| OLD_OBS | 8 | 42.1% |
| NEW_OBS | 5 | 26.3% |
| both arms | 13 / 38 | 34.2% |

Both arms failed on the same task 3 times (task_10, task_11, task_15); exactly
one arm failed on 7 tasks; neither arm failed on 9 tasks. Distribution across
arms is not the concentrated dimension — geometry is.

## Minimal fix / mitigation

### The one real class-A defect: the driver under-counts ungrounded rejections

Not a cause of any of the 13 errors, but it corrupts a load-bearing claim in the
delivered report, and it is a self-defeating safeguard: the driver's own comment
at `run_t233_supply_obs_ab.py:324-326` states the counter exists *"so a
mask-induced grounding failure cannot hide inside a protocol error"* — and it
hides exactly that.

```359:364:evaluation/functional/run_t233_supply_obs_ab.py
        "ungrounded_citation_rejections": sum(
            1
            for stage in stage_validation
            for code in stage["validation_error_codes"]
            if str(code) == "HYPOTHESIS_EVIDENCE_UNGROUNDED"
        ),
```

`stage_validation` is derived from `result["stages"]`, and `fast_path.py:306-315`
appends a stage entry **only after `core.run_stage` returns**. When the stage dies,
nothing is appended. Both S2 rows therefore read
`"stage_validation": []` with `"ungrounded_citation_rejections": 0`
(`t233_supply_obs_ab_v1.json:5246-5247` and `:6335-6336`). The counter can only
ever observe *recovered* rejections, never fatal ones — the case it was written
for. This propagates to `reads_as_mask_artifact` at
`run_t233_supply_obs_ab.py:936` (`bool(ungrounded.get(OLD_OBS, 0))` → `False`),
and into the delivered claim at `t233_supply_obs_ab_v1.md:33`, "ungrounded-citation
rejections were 0 in *both* arms". True count: **1 fatal per arm** (OLD_OBS
task_16, NEW_OBS task_11).

Minimal fix, two edits, no behaviour change to any episode:

1. `fast_path.py`, in the inspect `except _AGENT_FAULTS` handler at `:446-452`
   (which already reaches for `last_assistant_text` at `:449-451`), also record
   the terminal code — one added statement plus one `FastPathTrace` field:
   `trace.terminal_validation_error_code = getattr(exc, "error_code", None)`.
   `StagePostValidationError` already carries `error_code`
   (`fast_agent.py:436`), so no new plumbing is needed.
2. `run_t233_supply_obs_ab.py:359-364`, add that terminal code to the same sum.

Why this and not a text match on `protocol_error`: the message prose does not
contain the code string, so a text match would key on
`"StagePostValidationError: evidence feature"` and would silently drift the next
time the message is reworded.

Correcting this does **not** overturn the mask-artifact conclusion. Neither cited
key is one of the four masked names, and the NEW_OBS instance ran with
`mask_active: false`. The fix restores the evidence chain rather than reversing
the finding.

### Mitigation for the first fault (C), and whether it is worth doing now

The first fault is C, so there is no driver bug to repair. Three options, in
increasing cost:

1. **Raise the repair budget — recommended, one line.**
   `fast_path.py:292`, `validation_retries=1` → `2`. The retry already hands the
   Agent the exact validator string, and it demonstrably converts failures into
   passes (NEW_OBS task_09, `:4793-4800`). Cost is bounded at one extra LLM call
   per failing stage; the per-Task-per-arm guardrail of 20 never bound in this run
   (`t233_supply_obs_ab_v1.json:41-42`, 173 calls over 38 arm-runs). Caveat: this
   is shared harness glue, so it changes every driver that uses it; if that is
   unacceptable, thread it as a `_run_stage` keyword defaulting to 1 and pass 2
   from the T233 driver only.
   **Worth doing** — it recovers effective sample from a known-recoverable
   failure mode without touching any contract or threshold.

2. **State the shape in the Skill text — do not do this now.**
   Adding "`region_fractions` is exactly two numbers, `[start, end]`; use a
   second hypothesis for a second region" to
   `methods/ttha/harness/h0/skills/bootstrap/inspect_and_localize.json:6` would
   remove the guidance/schema tension at the source. Two reasons to refuse it
   today: it mutates h0, which invalidates `snapshot.lock.json` `dependency_shas`
   and `harness_content_sha` and cascades into the same lock-regeneration
   breakage documented at `m0b_field_wiring_report_v1.md:92-103`; and it edits the
   guidance substrate that the guidance-evolution experiment is measuring, which
   would contaminate the comparison against every prior run on this h0.

3. **Loosen the schema to accept a list of pairs — do not do this.**
   `fast_inspect_v1.json` is a frozen contract and its own `$comment` records that
   `pattern_hypotheses` was made optional specifically to avoid breaking
   `SealedProbeBackend` and frozen replay bands. Widening it would invalidate
   frozen replays for a formatting convenience.

For S2 specifically, no change is warranted: `_normalize_evidence_citations` is
deliberately bounded ("accept the spelling, never widen the evidence",
`3cf6a5a`), and relaxing the value-exactness test to tolerate float
reformatting would widen accepted evidence, which is the one thing that
normalizer was written not to do. 2 of 38 arm-runs is not worth that trade.

## Incidental — `REQUEST_OBSERVATION` share (count only, no interpretation)

| arm | `REQUEST_OBSERVATION` | of tasks | share |
| --- | --- | --- | --- |
| OLD_OBS | 6 | 19 | 0.316 |
| NEW_OBS | 11 | 19 | 0.579 |
| both | 17 | 38 | 0.447 |

Full stop-reason census for context (`t233_supply_obs_ab_v1.json:355-1449`):

| stop_reason | OLD_OBS | NEW_OBS |
| --- | --- | --- |
| `REQUEST_OBSERVATION` | 6 | 11 |
| `AGENT_PROTOCOL_ERROR` | 8 | 5 |
| `TRUST_DRAFT_GATE_PASS` | 3 | 1 |
| `AGENT_ABSTAIN` | 2 | 2 |
| total | 19 | 19 |

## Evidence limits

- **The raw offending Agent output is not recoverable.** `fast_path.py:449-451`
  captures `last_assistant_text` into `trace.protocol_error_output`, but
  `run_t233_supply_obs_ab.py` never serializes that field (no occurrence in the
  driver), and `.t233_supply_obs_ab_state/` holds only snapshot, lock, retrieval,
  verification and skill files — no LLM transcript. The S1 payload shape is
  therefore *deduced*, not quoted: it is forced by the two validator predicates
  at `schema_contracts.py:283` and `:314` against the schema at
  `fast_inspect_v1.json:68-77`, and corroborated by the 2026-08-19 `task_08`
  pair that produced both strings on one task. The S2 offending values are
  quoted verbatim from the messages themselves.
- **The v1-vs-today rate difference is not explained here.** v1
  (`t233_independent_source_supply.json`, 2026-08-19 23:20) ran the same 19-task
  roster with the identical geometry split (`:26`–`:977`, tasks 01–09 `"zero"`,
  tasks 10–19 `"low"`) and produced 0 protocol errors, versus 13/38 today.
  Geometry locates today's errors but cannot be the whole cause. Neither v1 nor
  `g3d1_t233_source_extension.json` records a `model` field, so the model era
  cannot be compared from the artifacts; today's run pins
  `gpt-5.6-luna` (`t233_supply_obs_ab_v1.json:43`). Attributing the difference
  would need a run, which is out of scope for read-only forensics.
- **Two accuracy notes on the delivered report.**
  `t233_supply_obs_ab_v1.md:24-25` describes the errors as "almost all the same
  schema slip (`pattern_hypotheses[0].region_fractions has too few items`)". That
  exact variant is 4 of 13; the majority variant is `[0] has wrong type` at 7 of
  13, and 2 are not schema slips at all. The characterization at `:37-38` — that
  these are ordinary schema slips and not mask damage — holds for the 11 S1
  errors and, on the merits, for the 2 S2 errors, but the "0 rejections in both
  arms" evidence offered at `:33` is an artifact of the accounting blind spot
  above, not a measurement.
