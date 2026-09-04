# M1a static prep: insertion-point review

Scope: static code reading only. No Runtime behaviour was changed, no experiment
was run, no Outcome was opened. This is the frozen M1a-prep deliverable and is
not method progress.

The gating semantics being located are the frozen ones:

* a Source Skill's TRY clause may influence candidate *ordering* only;
* each `skill-version x Context cell x Workflow family` key gets at most **one**
  unconfirmed Source-triggered Support probe;
* after that probe fails, the Runtime deterministically silences the clause for
  that key inside this Domain;
* the Target-only candidate channel is always preserved;
* a positive Support result yields a LOCAL_DRAFT only; activation still requires
  a second, delayed positive.

Everything below is stated against the agentic pipeline
(`evaluation/functional/task_episode_harness/agentic/`), which is the path the
M1a experiment will run on. The older `fast_agent` / E1 proposal path is noted
only where it differs, because it is a different candidate-supply mechanism.

---

## (a) Where the clause enters, and where the probe result comes back

### a.1 Entry: the Source Skill reaches the Agent as prompt text, not as a candidate

The Source-derived Skill is `skill_id="source_investigation_v1"`,
`skill_kind="capability"`, with `observable_applicability` fixed to
`task_kind == forecast`
(`evaluation/functional/task_episode_harness/agentic/source_skill.py:48`,
`:57-59`, `:441-456`). Its six sections, TRY included, are flattened into the
entry `body` at `source_skill.py:438-440`.

It is retrieved by the ordinary view resolver, not by a Source-specific path:

* `methods/ttha/retrieval.py:145-202` — `resolve_harness_view`. A CAPABILITY
  entry is applicability-matched at `retrieval.py:177-183` and then truncated by
  the capability `top_k` rule at `retrieval.py:186-199`.
* `evaluation/functional/task_episode_harness/agentic/runner.py:775-779` — the
  arm resolves its view for this Task's `task_fast_features`.

From there the clause reaches the Agent through **two independent carriers**,
and both matter for M1a:

1. **The system prompt.** `methods/ttha/agent_core.py:178-192` serializes the
   whole resolved view into the system message; each Skill is rendered at
   `agent_core.py:180`. This is the carrier that actually delivers the TRY
   sentence to the model.
2. **The `retrieved_knowledge` payload.** `runner.py:798-820` builds the
   `retrieved` mapping; the Source Skill's id appears in
   `retrieved["general"]["retrieved_skill_ids"]` at `runner.py:804`. That mapping
   is injected into INSPECT at
   `agentic/fast_path.py:428`, carried into PROPOSE by the `**inspect_input`
   splat at `fast_path.py:467`, and re-injected into every SELECT round at
   `fast_path.py:618`.

The Source Skill deliberately supplies **no** executable candidate: it carries no
`Frozen program steps:` marker and an empty `allowed_tools`
(`source_skill.py:428-437`, `:448`), which is what makes
`_skill_frozen_candidates` (`methods/ttha/fast_agent.py:325`, called at
`fast_agent.py:995`) yield nothing for it. Independently of that, the agentic
Fast Path never calls `_skill_frozen_candidates` at all — every candidate in this
pipeline originates in the Agent's PROPOSE payload, read at
`fast_path.py:510-511` and compiled at `fast_path.py:521-562`.

**Consequence, and this is the load-bearing finding for M1a:** in the agentic
pipeline there is currently *no per-candidate Source attribution*. A compiled row
(`fast_path.py:557-561`) carries `attempt_index`, `candidate_id`,
`addresses_hypothesis_id`, `status`, `steps`, `workflow` — nothing that records
"this candidate exists because a Source clause named that family". Any M1a gate
must therefore derive the attribution, and the only mechanism already frozen in
this repository for doing so is the operator-structure family string.

### a.2 The exact insertion point for the gate: `fast_path.py:571`

```569:575:evaluation/functional/task_episode_harness/agentic/fast_path.py
    # ---- SUPPORT + SELECT ------------------------------------------------
    probed: list[dict[str, Any]] = []
    pending = _deprioritized_probe_order(compiled_rows, harness_view, trace)
    while pending:
        current = pending.pop(0)
        try:
            support = support_probe(current["workflow"])
```

`fast_path.py:571` is the single line between "the candidate list exists" and
"a Support probe is spent". It is already the point where R1 applies its
order-only intervention, so it is the one place where a channel partition can be
inserted without touching COMPILE, SELECT, the Judge, or candidate supply. The
family key it would have to use is exactly the one R1 already computes:

```369:371:evaluation/functional/task_episode_harness/agentic/fast_path.py
    def named_by_a_risk_skill(row: Mapping[str, Any]) -> bool:
        family = "+".join(str(op) for op, _params in row["steps"])
        return "target_risk_" + family.replace("+", "_") in deprioritized
```

and which `risk_skill.family_of` defines canonically at
`agentic/risk_skill.py:57-64` ("operator structure, parameters discarded").

### a.3 Where the probe result flows back

Two reflow sites, at two different lifetimes.

**Within the Task (the one that decides silencing).**
`fast_path.py:572-602`: `support_probe` is called at `fast_path.py:575`, the gain
is read at `fast_path.py:586`, the pass/fail verdict against the material
threshold is computed at `fast_path.py:596`
(`"meets_material_threshold": bool(gain >= material_threshold)`), and the row is
appended to `trace.probes` at `fast_path.py:600-602`. This is the only place in
the agentic path where a Source-triggered probe's result becomes known, and
therefore the only place a `record_probe_outcome` call can sit.

**Across Tasks (the lifetime silencing must survive).**
`runner.py:860-882`: every `status == "PROBED"` row is turned into an Episode by
`_make_episode` (`e1.py:896-954`) and appended to `arm_state.episodes`
(`runner.py:882`). `_ArmState` (`e1.py:451-459`) is the only object in this
pipeline with per-arm, cross-Task lifetime that is never merged across A3/A5;
it is constructed once per arm at `runner.py:1732-1736`. A silence that must hold
"in this Domain, in this Context" for the rest of the run has to be derived from
`arm_state.episodes` or held on `_ArmState`. It must **not** be written into the
Skill entry — see (b) below.

Note also `fast_path.py:521` — only the first `probe_budget` proposals are ever
compiled — and `fast_path.py:687-696`, where a probed candidate below threshold is
rejected by the Runtime's own mechanical gate. M1a's gate is upstream of that one
and does not replace it.

---

## (b) Interaction with R1 and R4b

### b.1 R1 (risk Skill, order-only deprioritization)

Implementation: `fast_path.py:319-331` (`_risk_deprioritized_skill_ids`, reading
the *resolved view*, not the snapshot) and `fast_path.py:334-397`
(`_deprioritized_probe_order`). Minting: `runner.py:585-633`, from
`risk_skill.risk_candidates` (`risk_skill.py:153-184`).

Four interaction points, in order of how much they can bite:

1. **Same list, same line, same key.** R1 and the M1a gate would both act on
   `compiled_rows` at `fast_path.py:571`, and both key on the `"+".join(op)`
   family string (`fast_path.py:370`; `risk_skill.py:57-64`). Composition order
   is a real decision and the frozen M1a spec does not state it. R1's early-exit
   guard at `fast_path.py:374` (`if not held_back or len(held_back) == len(rows)`)
   is sensitive to which list it sees, so the two orderings are not equivalent.
   Reading the spec literally — M1a partitions *channels*, R1 orders *within* a
   list — partitioning first and ordering the surviving channel afterwards is the
   only composition that leaves both invariants intact, but this is an inference,
   not something the spec says. Flagged, not resolved.
2. **R1's "never dropped" invariant must not be broken.** `fast_path.py:351-358`
   states explicitly that a deprioritized candidate "is never dropped, never
   blocked, and stays selectable". M1a silences a *clause's ability to trigger a
   probe*, not a candidate. If an implementation ever silenced the candidate
   itself, it would contradict a documented and tested R1 property. This is
   exactly why the frozen semantics keep the Target-only channel open, and why
   the third contract test below asserts it.
3. **Different evidence thresholds, so they fire at different times.** R1 needs
   `>= 2` distinct negative Tasks *and* no positive anywhere
   (`risk_skill.py:49`, `:171`). M1a silences after **one** failed probe on a key.
   So on a first failure M1a is active and R1 is not; they are complementary, not
   duplicative.
4. **R1 has a lift path, M1a does not.** `risk_skill.contradicted_risk_families`
   (`risk_skill.py:187-201`), invoked at `runner.py:716-726`, retires a risk Skill
   once its family later earns a material positive in this Domain. The frozen M1a
   semantics describe silencing with no stated lift condition. Whether an M1a
   silence should be liftable by later in-domain positive evidence is **not
   covered by the spec**; flagged as an ambiguity, deliberately left out of the
   contract test.

Also note the recording discipline at `fast_path.py:378-396`: R1 only appends to
`trace.probe_order_deprioritizations` when the order actually changed, precisely
so the "times the reorder acted" readout is not inflated. Any M1a receipt must be
a separate field for the same reason; sharing that list would silently corrupt an
existing readout.

### b.2 R4b (delayed disconfirmation removes a Skill from all Fast channels)

Chain, end to end:

* `e1.py:976-981` — `_update_delayed` grades a delayed window `<= -tau` as
  `STATUS_RESTRICTED` / `RELATION_CONFLICT`.
* `runner.py:945-952` — the arm collects the disconfirmed skill id, but only when
  `delayed_event["stage"] == "existing_skill_revalidated"` **and** the winner's
  `local_status == "RESTRICTED"`.
* `runner.py:690-740` — `run_risk_skill_lifecycle`, step 1 at `runner.py:711-715`.
* `runner.py:636-687` — `_restrict_skill` PATCHes the Skill's own `risk_guards`,
  setting `restricted_by_target_feedback = True` at `runner.py:649-652`. The entry
  is not deleted (`runner.py:641-643`).
* Both Fast channels then drop it: `methods/ttha/retrieval.py:173-174` (resolved
  view, both roles) and `e1.py:536-537` (the second, independent Target-local read,
  whose comment at `e1.py:530-535` records why one channel was not enough).

Three interaction points:

1. **Do not reuse the R4b carrier.** R4b's guard is skill-wide: one flag on the
   entry removes it from every Context, in both roles. M1a's silence is scoped to
   one `skill-version x cell x family` key. Writing an M1a silence into
   `risk_guards[restricted_by_target_feedback]` would silently promote a
   single-cell silence into a global restriction and destroy the "Target-only
   channel stays available" clause. M1a state must live outside the Skill entry —
   `_ArmState` (`e1.py:451-459`) is the natural holder, and it is also the only
   per-arm cross-Task object.
2. **R4b structurally cannot cover the M1a case.** R4b requires a *winner* that
   reached a delayed window (`runner.py:887-909`, gated on `winner is not None`).
   A Source clause that triggered a probe which failed at Support never produces a
   winner and never reaches a delayed window, so R4b will never silence it. That
   gap is precisely what M1a exists to close, which confirms the new gate is not
   duplicating an existing lifecycle.
3. **If the Source Skill itself is ever restricted**, `retrieval.py:173-174`
   removes it from the view before anything downstream sees it, so it stops
   producing gate keys altogether. Keying the gate on the skill version — rather
   than on a candidate or a family alone — is what makes that degrade cleanly.

---

## (c) What "unconfirmed probe" already corresponds to in the code

There is an exact existing representation; M1a does not need a new status.

An Episode written from a Support probe alone is created at `e1.py:911-953`:

* `delayed_response={"evaluated": False, "gain": None, ...}` — `e1.py:948-949`;
* `evidence_level=EVIDENCE_SUPPORT` — `e1.py:951`;
* `local_status = STATUS_LOCAL_DRAFT if positive else STATUS_EPISODE_ONLY` —
  `e1.py:952`, where `positive` is `gain >= MATERIAL_THRESHOLD` (`e1.py:910`).

Confirmation happens only in `_update_delayed` (`e1.py:957-994`), which sets
`evidence_level=EVIDENCE_DELAYED` (`e1.py:991`) and grants
`STATUS_LOCAL_ACTIVE` only when **both** windows clear `+tau`
(`e1.py:973-977`). The three-band rule and why it was tightened are documented at
`e1.py:964-972`.

So "unconfirmed" is any of these three equivalent reads, and they agree by
construction:

* `episode.delayed_response["evaluated"] is False`;
* `episode.evidence_level == EVIDENCE_SUPPORT`;
* `episode.local_status in {STATUS_LOCAL_DRAFT, STATUS_EPISODE_ONLY}`.

Two consequences worth recording:

* The delayed window is only ever run for the **winner** (`runner.py:887-909`).
  Every probed-but-not-chosen candidate therefore stays permanently unconfirmed.
  "At most one unconfirmed Source-triggered probe per key" is consequently a hard,
  binding budget, not a soft one: within a Task of budget `B`, at most one of
  those probes may be charged to a given Source clause key.
* The "Support positive -> LOCAL_DRAFT only, delayed-positive -> active" half of
  the M1a semantics is **already implemented** and needs no new code:
  `e1.py:952` writes the draft, `method.handle_fast_winner` returns stage
  `"pending"` (`e1.py:1351-1365`), and the active pointer is written only after
  the delayed stage returns `"approved"` (`e1.py:1388-1391`).

### c.1 Where "Context cell" identity lives today

Recorded for completeness, because M0 is about to redefine it and the contract
test must not depend on any of it:

* `public_context["task_signature"]` — stored on the Episode at `e1.py:920`;
* `public_context["scope_bin"]` and the projection bin — `e1.py:929-934`;
* `G1_CONDITION_FEATURE = C1_POST_SHIFT_SUPPORT_FEATURE` (`g1.py:97`), which is
  what the Source census uses as its `context_condition`
  (`source_skill.py:169`, `:219-221`).

Since M0 will re-cut these, the contract test treats the cell identity as an
opaque string and never constructs, parses or compares its internals.

---

## (d) Ambiguities stopped on, not resolved

Listed here rather than decided, per the frozen prep task.

1. **No `skill_version` field exists.** The Source entry carries
   `revision: 1` (`source_skill.py:445`); the closest run-level identifiers are the
   snapshot's `harness_content_sha` / `runtime_bundle_sha` (e.g.
   `runner.py:555-556`). The spec's "skill-version" has no single existing
   binding. No new hash or identity scheme was introduced (AGENTS.md §1); the
   contract test takes it as an opaque string.
2. **Composition order of the M1a partition and R1's reorder is unstated** —
   see (b.1) item 1.
3. **Whether an M1a silence is liftable** by later in-domain positive evidence is
   unstated, whereas R1's analogue explicitly is (`risk_skill.py:187-201`) — see
   (b.1) item 4.
4. **"Source-triggered" has no existing marker.** Given (a.1), the attribution
   must be derived, and the only frozen mechanism is the operator-structure family
   string. Deriving it that way makes the gate approximate in one direction: an
   Agent that proposes the same family for its own reasons would be counted as
   Source-triggered. The spec does not say how to distinguish these.
5. **Module path and API for the gate were not specified.** The contract test has
   to name something, so it targets
   `evaluation/functional/task_episode_harness/agentic/source_probe_gate.py`
   (the package that already holds `risk_skill.py`), and skips itself until that
   module exists. This is a test-side placeholder, not a design decision, and no
   implementation was written.

### Note on prior partial output

An earlier interrupted attempt at this task left
`artifacts/functional/e2/m1a_static_review_v1.md` (2026-08-20 01:57) and a first
version of the contract test in the working tree. Both were untracked. This
review is the deliverable named by the frozen task; the earlier `_v1` file was
left in place untouched rather than deleted, since deleting it is outside the
task's mandate. It should be removed or superseded by whoever owns M1a.

---

No implementation code and no schedule are included, by instruction.
