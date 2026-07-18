# Agent-Centric Minipipe M0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable, two-cycle TTHA Harness-evolution loop in which one Agent core supplies and selects preparation programs on the fast path, proposes one-surface Harness edits on the slow path, and receives mechanically attributed paired-replay feedback.

**Architecture:** Add typed Candidate/Harness contracts and a generic runtime above the existing canonical operator registry and executor. TTHA owns an immutable, domain-naive H0 and one permission-separated Agent core; `evaluation/minipipe` owns synthetic cases, the private Chronos valuator, public fixed probes, fault attribution, sanitization, edit replay, and lineage. Complete the new vertical path before severing the remaining H_ref imports, then remove H_ref from the active tree. Any historical H_ref reproduction artifact is an inert grader-private fossil under `_frozen_reference/`, never an active baseline, candidate supplier, or TTHA dependency.

**Tech Stack:** Python 3.10.19, NumPy 2.2.6, PyTorch 2.12.0+cu126, `chronos-forecasting` 2.3.0, Transformers 5.12.1, OpenAI Python SDK 2.45.0, an OpenAI-compatible Chat Completions relay using model alias `gpt-5.5`, stdlib `dataclasses`/`enum`/`hashlib`/`json`/`pathlib`/`typing`, pytest.

**Source specification:** `docs/superpowers/specs/2026-07-17-agent-centric-minipipe-m0-design.md`, including the approved 2026-07-18 contract amendments.

## Global Constraints

- Run Python tests from `/mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence`, using `/mnt/d/Anaconda_envs/envs/project/python.exe` and an explicit package-local `--basetemp`.
- Before every task, run `git status --short --branch`; preserve all pre-existing user changes and stage only the files named by that task.
- New production code may import `contracts`, `operators`, `runtime`, and `methods/ttha`; TTHA must never import `evaluation/minipipe`.
- No new operator implementation, copied operator registry, arbitrary Python edit, downstream-model search, Harness Tree, or candidate-J-assisted selection belongs in M0.
- The same `TTHAAgentCore` instance type serves fast and slow roles. Role prompts, visible inputs, tools, and permissions differ; the underlying Agent backend and HarnessSnapshot are shared.
- Fast-path prompts receive only typed public inputs. Clean values, injection labels/indices, candidate utilities, exact rankings, private receipts, and oracle witnesses remain grader-private. The sole evaluator-derived exception is the fixed, deployment-computable `R_public` probe panel defined below.
- The Agent never certifies its own PASS/FAIL status. Runtime facts enter `DecisionTrace`; deterministic grader rules produce `StageAssessment` and `FaultAttribution`.
- The runtime injects one tagged identity candidate, identity occupies one configured slot and cannot be filtered, and absence of `chosen_candidate_id` is a protocol failure.
- `K=3` is the M0 configuration default: identity plus at most two Agent PROGRAM candidates. It is not encoded as a permanent contract invariant.
- H0 is procedurally complete and domain-naive: three always-injected bootstrap procedure skills, no capability SkillEntry, no memory, and no defect-family-to-program mapping.
- Slow-path edits are restricted to one declared textual, structured-rule, or scalar-config surface. Python/runtime/compiler/operator edits are rejected.
- A SkillEntry ADD is one atomic source edit; file existence is membership and the retrieval index is a deterministic derived artifact.
- A surface is only confirmed after single-surface paired replay verifies the predicted behavior change. Narrative attribution alone never confirms responsibility.
- Candidate utility and arbitrary evaluator queries stay private. The Agent's only evaluator-derived fast-path evidence is the fixed versioned public ProbeAPI; private clean-future probe value never crosses the wall.
- The deployment-observable vocabulary, its schema, and observation-tool code are read-only wall substrate in M0. The slow Agent cannot edit them.
- Every semantic content hash uses versioned `m0-c14n/1` canonicalization; raw file bytes, platform newlines, and JSON key order do not define semantic identity.
- Tests and default acceptance runs make no network calls. Live Agent calls use Chat Completions at `https://api.agicto.cn/v1`, model alias `gpt-5.5`, and an API key supplied only through `AGICTO_API_KEY`; Chronos loads the pinned local Hugging Face snapshot with `local_files_only=True`.
- At most three edits enter replay and at most one edit is promoted per cycle. M0 must execute two consecutive cycles from one command.
- Commit after every task. Do not combine tasks into one commit even when implementation is fast.
- Execute these dependent tasks inline and sequentially unless the user explicitly requests sub-agent delegation; the plan header names compatible execution skills but does not itself authorize delegation.

## Locked M0 Decisions

### Frozen valuator

Use deterministic zero-shot forecasting with:

```json
{
  "schema_version": "m0-valuator/1",
  "model_id": "amazon/chronos-bolt-small",
  "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
  "chronos_forecasting": "2.3.0",
  "torch": "2.12.0+cu126",
  "transformers": "5.12.1",
  "device": "cpu",
  "dtype": "float32",
  "context_length": 192,
  "prediction_length": 48,
  "point_forecast": "mean",
  "loss": "nrmse_clean_context_scale",
  "utility": "negative_loss"
}
```

The evaluator linearly fills remaining NaNs before inference, records the fill fraction, and uses

```text
scale = max(std(finite clean context), 1e-8)
J = sqrt(mean((forecast - clean future)^2)) / scale
U = -J
```

No downstream model is trained inside M0.

### Agent backend

Production code defines one `AgentBackend.complete(AgentRequest) -> AgentResponse` protocol. Automated tests and the default smoke command use an immutable replay backend. Live runs use the relay-safe Chat Completions adapter described below; the local immutable cache remains the replay authority.

The live adapter uses `openai==2.45.0` and constructs `OpenAI(api_key=..., base_url="https://api.agicto.cn/v1")`, then calls `client.chat.completions.create(messages=..., model="gpt-5.5")`. The key is read by the CLI from `AGICTO_API_KEY`, passed explicitly to the adapter, never written into a config/artifact/cache, and never logged. `M0_AGENT_MODEL` and `M0_AGENT_BASE_URL` may explicitly override the defaults for a non-reference run; both values, the SDK version, and capability flags enter the cache record and `run_context_sha`.

The relay has only demonstrated ordinary Chat Completions, so M0 does **not** assume Responses API support, provider-native function calling, `response_format`, strict Structured Outputs, reasoning controls, or provider seed support. Instead, resolved messages include a local `agent-envelope/1` contract. The assistant must emit exactly one JSON object with either:

```json
{"schema_version":"agent-envelope/1","kind":"tool_request","call_id":"call-1","tool_name":"inspect_public_series","arguments":{}}
```

or:

```json
{"schema_version":"agent-envelope/1","kind":"stage_result","stage":"select","payload":{"chosen_candidate_id":"identity"}}
```

The runtime validates this envelope, executes allowlisted public tools locally, appends the assistant envelope and one typed `tool-result/1` user message, and calls Chat Completions again. Invalid JSON or schema is cached as Agent behavior. Transient connection/time-out errors, HTTP 408/409/429, and 5xx failures are infrastructure errors and are not cached.

`gpt-5.5` is a relay model alias, not a verifiable dated provider snapshot. M0 therefore makes no snapshot-level model claim. Reproducibility comes from the exact effective-request identity, immutable raw-response cache, offline replay backend, and recorded provider metadata when present.

### Corpus and verdict constants

Check in `evaluation/minipipe/config/m0_rules.json` with these exact values:

```json
{
  "schema_version": "m0-rules/1",
  "utility_tolerance": 0.000001,
  "critic_damage_min": 0.01,
  "candidate_gain_min": 0.01,
  "selection_regret_min": 0.01,
  "risk_epsilon": 0.005,
  "localization_fail_iou_max": 0.10,
  "localization_pass_iou_min": 0.30,
  "probe_gain_min": 0.01,
  "probe_margin_min": 0.005,
  "target_recovery_fraction": 0.67,
  "target_median_gain_min": 0.01,
  "max_edits_per_cycle": 3,
  "max_promotions_per_cycle": 1,
  "candidate_pool_size": 3,
  "agent_program_slots": 2,
  "infrastructure_retries": 1,
  "probe_betas": [0.25, 0.50, 0.75],
  "public_probe_origins": [96, 120, 144, 168],
  "public_probe_horizon": 24,
  "public_probe_min_finite_targets": 12,
  "public_probe_round_decimals": 6,
  "corpus": {
    "base_seeds": [101, 202, 303],
    "context_length": 192,
    "future_length": 48,
    "severities": ["mild", "severe"],
    "target_families": [
      "missing",
      "impulsive_outlier",
      "level_shift",
      "period_change"
    ]
  }
}
```

The deterministic core corpus uses three base seeds (`101`, `202`, `303`), context length `192`, future length `48`, two severities per target family, and one matched clean/genuine-event risk case per seed. This yields 24 target cases and 12 risk cases before replay expansion.

### Probe strength mappings

All probes expose `beta in {0.25, 0.50, 0.75}` and larger beta is always more aggressive:

- imputation: replace the first `ceil(beta * n_detected_missing)` detected missing positions with the full `impute_linear` result;
- clipping: apply local robust-z clipping with thresholds `8.0`, `5.0`, and `3.0` respectively;
- denoising: blend raw and `denoise_median(window=5)` as `(1-beta)*raw + beta*denoised` on the detected region;
- level correction: subtract `beta * estimated_excursion_offset` on the detected excursion region;
- period: diagnostic only, with no repair transformation.

## File Map

### Canonical contracts and runtime

- `requirements-m0.txt` — incremental M0 dependency lock containing `openai==2.45.0`.
- `contracts/canonical.py` — `m0-c14n/1` text/JSON/JSONL normalization and SHA helpers.
- `contracts/candidate.py` — tagged identity/PROGRAM candidate identity.
- `contracts/harness.py` — SkillEntry, HarnessSnapshot, EditManifest, and semantic hashes.
- `contracts/observables.py` — closed deployment-observable feature vocabulary and applicability-AST validator.
- `contracts/schemas/skill_entry_v1.json` — deployable SkillEntry shape.
- `contracts/schemas/memory_entry_v1.json` — scoped deployable memory shape.
- `contracts/schemas/observable_feature_v1.json` — versioned feature names, types, and allowed predicates.
- `contracts/schemas/harness_snapshot_v1.json` — compiled snapshot receipt shape.
- `contracts/schemas/edit_manifest_v2.json` — PATCH/ADD edit contract shape.
- `runtime/candidate_pool.py` — identity injection, deduplication, risk filtering, selection validation, and effect-equivalence.
- `runtime/decision_trace.py` — runtime fact records and normalized behavior signatures.
- `runtime/agent_backend.py` — provider-independent request/response protocol, relay-safe GPT-5.5 Chat Completions adapter, and immutable replay backend.
- `runtime/llm_cache.py` — effective-request cache and immutable response records.

### TTHA method

- `methods/ttha/agent_core.py` — shared role-aware Agent invocation and schema parsing.
- `methods/ttha/schemas/agent_envelope_v1.json` — local tool-request/stage-result response envelope.
- `methods/ttha/schemas/tool_result_v1.json` — typed local public-tool result message.
- `methods/ttha/public_tools.py` — deployment-observable tool gateway.
- `methods/ttha/retrieval.py` — deterministic bootstrap/capability retrieval and effective Harness view.
- `methods/ttha/fast_agent.py` — inspect, propose, select, compile, execute orchestration.
- `methods/ttha/slow_agent.py` — sanitized FailurePatternCard to EditManifest proposal.
- `methods/ttha/method.py` — canonical `Method.prepare` adapter.
- `methods/ttha/harness/compiler.py` — H0/snapshot compilation and content/bundle hashes.
- `methods/ttha/harness/store.py` — immutable copy-on-write snapshot roots and active pointer.
- `methods/ttha/harness/harness_surfaces.json` — single-owner editable surface map.
- `methods/ttha/harness/h0/**` — checked-in H0 authoring files and lock.

### Minipipe

- `evaluation/minipipe/contracts.py` — private/public case, feedback, assessment, pattern, replay, and lineage records.
- `evaluation/minipipe/config/m0_rules.json` — executable thresholds and cycle limits.
- `evaluation/minipipe/corpus/generate.py` — deterministic target and risk cases.
- `evaluation/minipipe/corpus/injections.py` — four private injection mechanisms.
- `evaluation/minipipe/valuation/chronos.py` — pinned frozen Chronos adapter.
- `evaluation/minipipe/valuation/rolling_observed.py` — deployment-computable rolling-origin public probe valuator.
- `evaluation/minipipe/valuation/model_manifest.json` — exact model/runtime identity.
- `evaluation/minipipe/valuation/outcomes.py` — U, D, G, NRR, collateral, and regret.
- `evaluation/minipipe/probes/features.py` — closed public feature vocabulary and extraction.
- `evaluation/minipipe/probes/panel.py` — separated public rolling-origin and private clean-future response panels.
- `evaluation/minipipe/probes/expressibility.py` — oracle/observable witnesses and three-valued status.
- `evaluation/minipipe/baselines/fixed_program_baseline_v1.json` — independently authored grader-private positive witnesses.
- `evaluation/minipipe/feedback/first_fault.py` — ordered deterministic StageAssessment rules.
- `evaluation/minipipe/feedback/router.py` — cause-to-surface/skill-kind authorization.
- `evaluation/minipipe/feedback/sanitize.py` — private-to-public wall.
- `evaluation/minipipe/feedback/patterns.py` — buckets, contrasts, purity, and cards.
- `evaluation/minipipe/replay/edit_controller.py` — manifest validation and single-surface application.
- `evaluation/minipipe/replay/paired.py` — Stage A/B replay and verdict derivation.
- `evaluation/minipipe/replay/lineage.py` — append-only snapshot/edit lineage.
- `evaluation/minipipe/cycle.py` — one deterministic M0 cycle.
- `cli/minipipe.py` — two-cycle composition root.

### Tests

- `tests/contracts/test_candidate_contract.py`
- `tests/contracts/test_harness_contract.py`
- `tests/runtime/test_candidate_pool.py`
- `tests/runtime/test_agent_backend.py`
- `tests/runtime/test_llm_cache.py`
- `tests/methods/test_ttha_h0.py`
- `tests/methods/test_ttha_agent.py`
- `tests/minipipe/test_corpus.py`
- `tests/minipipe/test_valuator.py`
- `tests/minipipe/test_probes.py`
- `tests/minipipe/test_first_fault.py`
- `tests/minipipe/test_information_walls.py`
- `tests/minipipe/test_edit_controller.py`
- `tests/minipipe/test_paired_replay.py`
- `tests/integration/test_minipipe_two_cycles.py`
- `tests/architecture/test_ttha_dependency_rules.py`

---

### Task 1: Define Candidate and Harness Contracts

**Files:**

- Create: `contracts/canonical.py`
- Create: `contracts/candidate.py`
- Create: `contracts/harness.py`
- Create: `contracts/observables.py`
- Create: `contracts/schemas/skill_entry_v1.json`
- Create: `contracts/schemas/memory_entry_v1.json`
- Create: `contracts/schemas/observable_feature_v1.json`
- Create: `contracts/schemas/harness_snapshot_v1.json`
- Create: `contracts/schemas/edit_manifest_v2.json`
- Modify: `contracts/__init__.py`
- Test: `tests/contracts/test_candidate_contract.py`
- Test: `tests/contracts/test_harness_contract.py`
- Test: `tests/contracts/test_canonical.py`

**Interfaces:**

- Consumes: `contracts.program.Program`, canonical JSON-native data.
- Produces: `CandidateKind`, `Candidate`, `SkillKind`, `SkillEntry`, `MemoryEntry`, `HarnessSnapshot`, `EditOperation`, `EditManifest`, `canonical_text_bytes`, `parse_json_document`, `canonical_json_bytes`, `canonical_json_document_bytes`, `canonical_jsonl_bytes`, `canonical_sha256`, Skill/Memory loaders, `OBSERVABLE_FEATURES`, and `validate_applicability(ast)`.

- [ ] **Step 1: Write failing Candidate contract tests**

```python
from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.program import Program


def test_identity_is_tagged_and_has_no_program():
    candidate = Candidate.identity()
    assert candidate.candidate_id == "identity"
    assert candidate.kind is CandidateKind.IDENTITY
    assert candidate.program is None


def test_program_candidate_requires_non_empty_program():
    program = Program.from_steps([("impute_linear", {})], source="agent")
    candidate = Candidate.program_candidate("agent-0", program, source="agent")
    assert candidate.kind is CandidateKind.PROGRAM
    assert candidate.program is program
```

- [ ] **Step 2: Run the Candidate tests and verify the missing module failure**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/contracts/test_candidate_contract.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t01_candidate
```

Expected: collection fails with `ModuleNotFoundError: No module named 'SelfEvolvingHarnessTS.contracts.candidate'`.

- [ ] **Step 3: Implement the tagged Candidate contract**

Create `contracts/candidate.py` with this public shape:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .program import Program


class CandidateKind(str, Enum):
    IDENTITY = "identity"
    PROGRAM = "program"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: CandidateKind
    program: Program | None
    source: str
    downstream_model_id: str = "fixed:m0"

    def __post_init__(self) -> None:
        if not self.candidate_id or self.candidate_id != self.candidate_id.strip():
            raise ValueError("candidate_id must be canonical")
        if self.kind is CandidateKind.IDENTITY:
            if self.candidate_id != "identity" or self.program is not None or self.source != "runtime":
                raise ValueError("identity is runtime-owned and program-free")
        elif self.program is None:
            raise ValueError("PROGRAM candidate requires Program")

    @classmethod
    def identity(cls) -> "Candidate":
        return cls("identity", CandidateKind.IDENTITY, None, "runtime")

    @classmethod
    def program_candidate(
        cls, candidate_id: str, program: Program, *, source: str
    ) -> "Candidate":
        return cls(candidate_id, CandidateKind.PROGRAM, program, source)
```

Export `Candidate` and `CandidateKind` from `contracts/__init__.py`.

- [ ] **Step 4: Write failing Harness contract tests**

```python
import pytest

from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    SkillKind,
    load_skill_entry,
)


VALID_SKILL = {
    "schema_version": "skill-entry/1",
    "skill_id": "local_outlier_repair_v1",
    "skill_kind": "capability",
    "revision": 1,
    "body": "Prefer a local robust repair when public evidence supports it.",
    "observable_applicability": {
        "all": [{"feature": "local_robust_z_peak", "op": ">=", "value": 5.0}]
    },
    "allowed_tools": ["hampel_filter"],
    "risk_guards": {"max_modified_fraction": 0.05},
}


def test_skill_entry_rejects_private_fields():
    with pytest.raises(ValueError, match="forbidden deployable field"):
        load_skill_entry({**VALID_SKILL, "injection_type": "spike"})


def test_applicability_rejects_unknown_or_oracle_features():
    from SelfEvolvingHarnessTS.contracts.observables import validate_applicability

    with pytest.raises(ValueError, match="unknown observable feature"):
        validate_applicability(
            {"all": [{"feature": "injection_type", "op": "==", "value": "spike"}]}
        )


def test_deployable_memory_rejects_source_pattern_provenance():
    from SelfEvolvingHarnessTS.contracts.harness import load_memory_entry

    with pytest.raises(ValueError, match="forbidden deployable field"):
        load_memory_entry({
            "schema_version": "memory-entry/1",
            "memory_id": "local-repair-caution-v1",
            "revision": 1,
            "body": "Keep the change local.",
            "observable_applicability": {"const": True},
            "risk_guards": {},
            "pattern_id": "pattern-private-source",
        })


def test_add_manifest_requires_absent_precondition():
    with pytest.raises(ValueError, match="ABSENT"):
        EditManifest(
            edit_id="e1",
            base_harness_sha="a" * 64,
            target_pattern_id="p1",
            target_surface_id="skill_library.entries/local_outlier_repair_v1",
            operation=EditOperation.ADD,
            surface_precondition={"kind": "SHA", "sha": "b" * 64},
            dependency_precondition_shas={},
            new_value=VALID_SKILL,
            predicted_agent_behavior_change=("retrieve_new_skill",),
            predicted_data_effect=("target_gain",),
            falsification_condition=("skill_not_retrieved",),
        )
```

Also create `tests/contracts/test_canonical.py` with fixtures proving that LF versus CRLF, an optional UTF-8 BOM, JSON key order, insignificant JSON whitespace, and canonically equivalent Unicode strings produce the same semantic digest. Assert that duplicate JSON keys, NaN/Infinity, NUL-containing text, and invalid UTF-8 are rejected. Assert JSONL order is stable and meaningful: individual rows are canonicalized but permuting rows changes the digest.

- [ ] **Step 5: Run the Harness tests and verify the missing symbols failure**

Run the two contract test files. Expected: Candidate tests pass and Harness tests fail during import because `contracts.harness` does not exist.

- [ ] **Step 6: Implement Harness contracts and checked-in schemas**

Implement immutable dataclasses with these exact field names:

```python
class SkillKind(str, Enum):
    BOOTSTRAP_PROCEDURE = "bootstrap_procedure"
    CAPABILITY = "capability"
    SAFETY = "safety"


@dataclass(frozen=True)
class SkillEntry:
    schema_version: str
    skill_id: str
    skill_kind: SkillKind
    revision: int
    body: str
    observable_applicability: Mapping[str, object]
    allowed_tools: tuple[str, ...]
    risk_guards: Mapping[str, object]


@dataclass(frozen=True)
class MemoryEntry:
    schema_version: str
    memory_id: str
    revision: int
    body: str
    observable_applicability: Mapping[str, object]
    risk_guards: Mapping[str, object]


@dataclass(frozen=True)
class HarnessSnapshot:
    schema_version: str
    instruction: str
    skills: tuple[SkillEntry, ...]
    memories: tuple[MemoryEntry, ...]
    retrieval: Mapping[str, object]
    candidate_policy: Mapping[str, object]
    verification: Mapping[str, object]
    dependency_shas: Mapping[str, str]
    harness_content_sha: str
    runtime_bundle_sha: str


class EditOperation(str, Enum):
    PATCH = "PATCH"
    ADD = "ADD"


@dataclass(frozen=True)
class EditManifest:
    edit_id: str
    base_harness_sha: str
    target_pattern_id: str
    target_surface_id: str
    operation: EditOperation
    surface_precondition: Mapping[str, object]
    dependency_precondition_shas: Mapping[str, str]
    minimal_patch: Mapping[str, object] | None = None
    new_value: Mapping[str, object] | None = None
    observable_applicability: Mapping[str, object] | None = None
    predicted_agent_behavior_change: tuple[str, ...] = ()
    predicted_data_effect: tuple[str, ...] = ()
    automatically_selected_risk_cases: tuple[str, ...] = ()
    falsification_condition: tuple[str, ...] = ()
```

`load_skill_entry` must reject keys outside the eight schema fields and reject `case_id`, `injection_type`, `injection_indices`, `D`, `G`, `J`, `pattern_id`, and `private_receipt` explicitly. It accepts the three declared SkillKind values so the H0 compiler can load bootstrap and safety entries. `load_learned_skill_entry` wraps it and additionally requires `skill_kind=capability`.

Implement `contracts/canonical.py` as the only semantic hashing path, with `CANONICALIZATION_VERSION = "m0-c14n/1"`:

- text: require valid UTF-8, remove one leading BOM, reject NUL, normalize `CRLF` and bare `CR` to `LF`, normalize Unicode to NFC, trim only trailing newline code points, then append exactly one `LF`;
- JSON: `parse_json_document(raw_bytes)` decodes UTF-8 after optional BOM removal, detects duplicate keys, and rejects non-finite numbers; `canonical_json_bytes(parsed_value)` recursively NFC-normalizes string keys/values, rejects a collision created by normalization, then serializes with `sort_keys=True`, `ensure_ascii=False`, `separators=(",", ":")`, and `allow_nan=False` to UTF-8 without BOM; `canonical_json_document_bytes` composes the two without treating a raw JSON document as a JSON string value;
- JSONL: canonicalize every nonblank row as JSON, preserve declared row order, join rows with one `LF`, and end with one `LF`;
- `canonical_sha256(value)` accepts a parsed JSON-native value and hashes its canonical JSON bytes.

The canonicalization version and implementation-source SHA are dependency locks. A raw provider response keeps both its original-byte hash and, when parseable, its semantic JSON hash; it is never silently rewritten in the cache.

`contracts/observables.py` owns this closed M0 vocabulary and no other module may extend it at runtime:

```python
OBSERVABLE_FEATURES = {
    "task_kind": "string",
    "missing_fraction": "number",
    "longest_missing_run_fraction": "number",
    "local_robust_z_peak": "number",
    "estimated_region_start_fraction": "number",
    "estimated_region_end_fraction": "number",
    "level_excursion_score": "number",
    "period_change_score": "number",
    "period_repair_available": "boolean",
    "imputation_probe_direction": "string",
    "clipping_probe_direction": "string",
    "denoising_probe_direction": "string",
    "level_probe_direction": "string"
}
```

`validate_applicability` recursively accepts only `all`, `any`, and `not` boolean nodes, the explicit node `{"const": true|false}`, and leaves of `{feature, op, value}`. Numeric features allow `>`, `>=`, `<`, `<=`, `==`; booleans allow `==`; strings allow `==` and `in`. It rejects empty boolean nodes, non-finite numbers, unknown fields, and all private/oracle names.

`load_memory_entry` applies the same private/oracle-field and applicability checks, requires a canonical `memory_id`, and forbids source case/pattern provenance in deployable content. Memory lineage stays on EditManifest/lineage records.

Write the five JSON schema files with `additionalProperties: false`, the same field names, and required lists matching the dataclasses. Schema files are identity artifacts even though runtime validation remains explicit Python.

- [ ] **Step 7: Run contract tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/contracts -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t01_contracts
```

Expected: all contract tests pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add contracts tests/contracts
git commit -m "feat: define TTHA candidate and harness contracts"
```

---

### Task 2: Compile the Domain-Naive H0 and Immutable Snapshot Store

**Files:**

- Create: `methods/ttha/__init__.py`
- Create: `methods/ttha/harness/__init__.py`
- Create: `methods/ttha/harness/compiler.py`
- Create: `methods/ttha/harness/store.py`
- Create: `methods/ttha/harness/harness_surfaces.json`
- Create: `methods/ttha/harness/h0/instruction.md`
- Create: `methods/ttha/harness/h0/skills/bootstrap/inspect_and_localize.json`
- Create: `methods/ttha/harness/h0/skills/bootstrap/build_contrastive_candidates.json`
- Create: `methods/ttha/harness/h0/skills/bootstrap/select_or_identity_and_verify.json`
- Create: `methods/ttha/harness/h0/skills/learned/.gitkeep`
- Create: `methods/ttha/harness/h0/memories.jsonl`
- Create: `methods/ttha/harness/h0/retrieval.json`
- Create: `methods/ttha/harness/h0/candidate_policy.json`
- Create: `methods/ttha/harness/h0/verification.json`
- Create: `methods/ttha/harness/h0/snapshot.lock.json`
- Test: `tests/methods/test_ttha_h0.py`

**Interfaces:**

- Consumes: Task 1 Harness contracts, `operators.registry.OPERATOR_METADATA`, checked-in schema bytes.
- Produces: `compile_snapshot(root, verify_lock=True) -> HarnessSnapshot`, `write_lock(root) -> Path`, `MaterializedSnapshot`, `SnapshotStore.materialize(snapshot, parent_sha=None) -> MaterializedSnapshot`, and `SnapshotStore.set_active(runtime_bundle_sha)`.

- [ ] **Step 1: Write failing H0 tests**

```python
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.contracts.harness import SkillKind
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot


H0 = Path(__file__).resolve().parents[2] / "methods" / "ttha" / "harness" / "h0"


def test_h0_is_stable_procedural_and_domain_naive():
    first = compile_snapshot(H0)
    second = compile_snapshot(H0)
    assert first.harness_content_sha == second.harness_content_sha
    assert first.runtime_bundle_sha == second.runtime_bundle_sha
    assert first.memories == ()
    assert [skill.skill_kind for skill in first.skills] == [
        SkillKind.BOOTSTRAP_PROCEDURE,
        SkillKind.BOOTSTRAP_PROCEDURE,
        SkillKind.BOOTSTRAP_PROCEDURE,
    ]
    forbidden = ("missing ->", "outlier ->", "impute_linear", "winsorize")
    resolved = first.instruction + "\n" + "\n".join(skill.body for skill in first.skills)
    assert not any(token in resolved for token in forbidden)


def test_h0_lock_mismatch_fails_loudly(tmp_path):
    root = tmp_path / "h0"
    root.mkdir()
    for source in H0.rglob("*"):
        if source.is_file():
            target = root / source.relative_to(H0)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    (root / "instruction.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="lock mismatch"):
        compile_snapshot(root)
```

- [ ] **Step 2: Run the H0 tests and verify they fail because TTHA is absent**

Run the new test file. Expected: collection fails with `ModuleNotFoundError` for `methods.ttha`.

- [ ] **Step 3: Add exact H0 authoring content**

Write `instruction.md` as a protocol-only instruction containing these rules:

```text
You are the TTHA preparation Agent. Use only public observations, declared tools,
canonical operator contracts, retrieved Harness content, and the typed output schema.
Never request or infer clean references, injection metadata, candidate utility, or
private rankings. Inspect before proposing. Supply at most the configured number of
effect-distinct PROGRAM candidates. Select exactly one candidate, including identity.
Keep modifications local and abstain when public evidence does not justify a repair.
```

Each bootstrap JSON is a `skill-entry/1` object with `skill_kind="bootstrap_procedure"`, revision 1, no family/operator names, `observable_applicability={"const": true}`, `allowed_tools=[]`, and one of these bodies:

```text
inspect_and_localize: Inspect public missingness, robust local deviation, level continuity,
and period-consistency evidence; report bounded candidate regions and uncertainty.

build_contrastive_candidates: Construct at most two minimal, behavior-distinct typed
PROGRAM candidates from currently allowed tools; retain runtime identity as the control.

select_or_identity_and_verify: Compare candidates using deployment-visible evidence and
risk guards only; choose one ID explicitly and record verification actions and scope.
```

Write:

```json
// retrieval.json
{"schema_version":"retrieval/1","bootstrap":"always","capability":{"kind":"rule_lexical","top_k":2}}

// candidate_policy.json
{"schema_version":"candidate-policy/1","total_k":3,"identity_slots":1,"agent_program_slots":2,"proposal_guidance":"Supply only minimal effect-distinct candidates justified by public evidence.","selection_guidance":"Choose one candidate explicitly and prefer identity when public evidence is insufficient."}

// verification.json
{"schema_version":"verification/1","identity_unfilterable":true,"require_explicit_choice":true,"max_modified_fraction":0.25,"preserve_outside_candidate_region":true}
```

`memories.jsonl` is zero bytes and `skills/learned/` contains no `.json` files.

- [ ] **Step 4: Implement deterministic snapshot compilation**

`compile_snapshot` must:

1. read authoring files in sorted relative-path order and canonicalize them with `m0-c14n/1` before semantic hashing;
2. parse every bootstrap/learned SkillEntry;
3. verify exactly the three required bootstrap IDs and no duplicate ID;
4. reject capability skills or memory rows in H0;
5. derive a retrieval index sorted by `skill_id`;
6. compute `harness_content_sha` from resolved semantic objects only;
7. compute an operator implementation bundle hash from sorted `m0-c14n/1` text of `operators/*.py` plus canonical `OPERATOR_METADATA`;
8. compute `runtime_bundle_sha` from content SHA, operator bundle SHA, schema hashes, compiler source hash, and retrieval compiler version;
9. compare all values with `snapshot.lock.json` when `verify_lock=True`.

Import canonicalization helpers from `contracts.canonical`; do not define a second JSON hashing recipe in the compiler. Include `canonicalization_version` and the canonicalizer source SHA in `snapshot.lock.json`.

`SnapshotStore.materialize` creates `runs/minipipe/harness_snapshots/<runtime_bundle_sha>/` by writing resolved JSON and copied source entries, refuses an existing directory with different bytes, and writes provenance outside semantic content. `set_active` atomically replaces a small `active.json` pointer; it never edits checked-in H0.

The store returns a separate reference rather than putting filesystem state into `HarnessSnapshot`:

```python
@dataclass(frozen=True)
class MaterializedSnapshot:
    root: Path
    snapshot: HarnessSnapshot
    parent_runtime_bundle_sha: str | None

    @property
    def harness_content_sha(self) -> str:
        return self.snapshot.harness_content_sha

    @property
    def runtime_bundle_sha(self) -> str:
        return self.snapshot.runtime_bundle_sha
```

- [ ] **Step 5: Generate and verify the H0 lock**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m \
  SelfEvolvingHarnessTS.methods.ttha.harness.compiler \
  --root SelfEvolvingHarnessTS/methods/ttha/harness/h0 --write-lock
```

Expected: `snapshot.lock.json` is rewritten with `harness_content_sha`, `runtime_bundle_sha`, `operator_bundle_sha`, dependency hashes, and compiler version; a second invocation without `--write-lock` exits 0 and prints the same two snapshot hashes.

Add a test that copies H0 while changing Markdown to CRLF/BOM and reorders keys/whitespace in JSON authoring files. The copied H0 must compile to the same `harness_content_sha`; a semantic field change must not.

- [ ] **Step 6: Run H0 tests**

Run the H0 test file twice. Expected: all tests pass both times with identical printed hashes.

- [ ] **Step 7: Commit Task 2**

```bash
git add methods/ttha tests/methods/test_ttha_h0.py
git commit -m "feat: compile immutable domain-naive H0"
```

---

### Task 3: Build the Generic Candidate Pool and Decision Trace

**Files:**

- Create: `runtime/candidate_pool.py`
- Create: `runtime/decision_trace.py`
- Modify: `runtime/__init__.py`
- Test: `tests/runtime/test_candidate_pool.py`

**Interfaces:**

- Consumes: `Candidate`, `Program`, canonical `runtime.executor.run_pipeline`.
- Produces: `CandidatePool.build(program_candidates, total_k)`, `CandidatePool.require_choice(candidate_id)`, `execute_selected(candidate, values)`, `effect_equivalent_to_identity(raw, prepared, tolerance=None)`, `DecisionTrace`, and `BehaviorSignature.from_trace(trace)`.

- [ ] **Step 1: Write failing identity, risk, and choice tests**

```python
import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.runtime.candidate_pool import CandidatePool, ProtocolChoiceError


def _program_candidate(candidate_id="agent-0"):
    program = Program.from_steps([("impute_linear", {})], source="agent")
    return Candidate.program_candidate(candidate_id, program, source="agent")


def test_identity_is_injected_and_never_filtered():
    pool = CandidatePool.build([_program_candidate()], total_k=3)
    kept = pool.apply_risk(lambda candidate: False)
    assert kept.ids[0] == "identity"
    assert "identity" in kept.ids


def test_missing_choice_is_protocol_failure():
    pool = CandidatePool.build([_program_candidate()], total_k=3)
    with pytest.raises(ProtocolChoiceError, match="chosen_candidate_id"):
        pool.require_choice("")


def test_effect_equivalence_uses_shape_dtype_and_bytes():
    raw = np.asarray([1.0, 2.0], dtype=np.float64)
    assert pool_effect(raw, raw.copy()) is True
    assert pool_effect(raw, raw.astype(np.float32)) is False
```

Import `effect_equivalent_to_identity as pool_effect` in the test.

- [ ] **Step 2: Run the test and verify missing runtime module failure**

Expected: import fails for `runtime.candidate_pool`.

- [ ] **Step 3: Implement pool invariants and execution**

Implement `CandidatePool` as an immutable tuple with these rules:

```python
@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[Candidate, ...]
    requested_k: int

    @classmethod
    def build(cls, programs: Iterable[Candidate], *, total_k: int) -> "CandidatePool":
        if total_k < 1:
            raise ValueError("total_k must include identity")
        merged = [Candidate.identity()]
        seen_program_sha: set[str] = set()
        for candidate in programs:
            if candidate.kind is not CandidateKind.PROGRAM:
                raise ValueError("suppliers may only submit PROGRAM candidates")
            assert candidate.program is not None
            if candidate.program.sha() in seen_program_sha:
                continue
            seen_program_sha.add(candidate.program.sha())
            merged.append(candidate)
            if len(merged) == total_k:
                break
        return cls(tuple(merged), total_k)

    def apply_risk(self, keep: Callable[[Candidate], bool]) -> "CandidatePool":
        kept = tuple(c for c in self.candidates if c.kind is CandidateKind.IDENTITY or keep(c))
        return CandidatePool(kept, self.requested_k)
```

`require_choice` rejects blank or unknown IDs. `execute_selected(identity, values)` returns an immutable float64 copy and no Program; PROGRAM execution delegates to `run_pipeline(program.execution_steps(), values, source=...)`.

`effect_equivalent_to_identity` requires exact shape, dtype, and bytes when tolerance is `None`; when tolerance is configured it requires equal shape plus `np.allclose(..., atol=tolerance, rtol=0, equal_nan=True)`.

- [ ] **Step 4: Implement runtime-owned DecisionTrace**

Use JSON-native immutable records rather than free-form reasoning:

```python
@dataclass(frozen=True)
class DecisionTrace:
    case_id: str
    public_observation_ids: tuple[str, ...]
    inspected_regions: tuple[tuple[int, int], ...]
    tool_calls: tuple[Mapping[str, object], ...]
    retrieved_skill_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    applicability_matches: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_program_shas: tuple[str | None, ...]
    chosen_candidate_id: str
    compilation_status: str
    execution_status: str
    modified_indices: tuple[int, ...]
    verification_actions: tuple[str, ...]
    effect_equivalent_to_identity: bool
```

`BehaviorSignature.from_trace` hashes only normalized semantic behavior: inspected region fractions rounded to six decimals, tool names, skill/memory IDs, program SHAs, chosen ID, compile/execute status, modified-region fractions, verification actions, and identity-equivalence. It excludes timestamps, latencies, raw prose, provider request IDs, and Harness provenance.

- [ ] **Step 5: Run runtime tests**

Run `tests/runtime/test_candidate_pool.py` plus existing `tests/runtime/test_executor_contract.py`. Expected: all pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add runtime tests/runtime/test_candidate_pool.py
git commit -m "feat: add generic candidate pool and decision trace"
```

---

### Task 4: Add the Relay-Safe GPT-5.5 Chat Backend and Effective-Request Cache

**Files:**

- Create: `requirements-m0.txt`
- Create: `runtime/agent_backend.py`
- Create: `runtime/llm_cache.py`
- Modify: `runtime/__init__.py`
- Test: `tests/runtime/test_agent_backend.py`
- Test: `tests/runtime/test_llm_cache.py`

**Interfaces:**

- Consumes: exact resolved Chat Completions messages, local envelope/tool schemas, `AGICTO_API_KEY`, optional `M0_AGENT_MODEL`/`M0_AGENT_BASE_URL`.
- Produces: `AgentRequest`, `AgentResponse`, `AgentTransportError`, `AgentBackend`, `AgictoChatCompletionsBackend`, `ReplayAgentBackend`, `EffectiveRequestCache`, and `CachedAgentBackend`.

- [ ] **Step 1: Pin the SDK and write failing relay request-shape tests**

```python
from types import SimpleNamespace

from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgentRequest,
    AgictoChatCompletionsBackend,
)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content='{"schema_version":"agent-envelope/1",'
                                          '"kind":"stage_result","stage":"select",'
                                          '"payload":{"chosen_candidate_id":"identity"}}')
        return SimpleNamespace(
            id="chatcmpl-m0-1",
            model="gpt-5.5",
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=8),
            model_dump=lambda **_: {
                "id": "chatcmpl-m0-1",
                "model": "gpt-5.5",
                "choices": [{"message": {"content": message.content}, "finish_reason": "stop"}],
            },
        )


def test_chat_request_uses_relay_alias_and_no_unproven_provider_features():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    backend = AgictoChatCompletionsBackend(client=client)
    request = AgentRequest.for_stage(
        case_id="case-1",
        role="fast",
        stage="select",
        call_index=0,
        replicate_id="r0",
        messages=(
            {"role": "system", "content": "Return agent-envelope/1 JSON only."},
            {"role": "user", "content": "Select from the public candidate pool."},
        ),
        envelope_schema_sha="4" * 64,
        tool_schema_sha="5" * 64,
        tool_result_schema_sha="6" * 64,
        stage_schema_sha="7" * 64,
        public_case_view_sha="1" * 64,
        effective_harness_view_sha="2" * 64,
        tool_context_sha="3" * 64,
    )

    result = backend.complete(request)
    payload = completions.calls[0]
    assert payload == {"model": "gpt-5.5", "messages": list(request.messages)}
    assert result.parsed_envelope["payload"]["chosen_candidate_id"] == "identity"
```

Write `requirements-m0.txt` with exactly `openai==2.45.0`. Unit tests inject a fake client and make no network call. A separate constructor test monkeypatches `openai.OpenAI` and asserts `api_key`, `base_url="https://api.agicto.cn/v1"`, and the timeout are passed without exposing the key in `repr`, exceptions, or request/cache records.

- [ ] **Step 2: Write failing cache-identity and malformed-response tests**

```python
from dataclasses import replace

from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse, ReplayAgentBackend
from SelfEvolvingHarnessTS.runtime.llm_cache import CachedAgentBackend, EffectiveRequestCache


def test_full_snapshot_provenance_does_not_change_semantic_request_hash(tmp_path, request):
    changed_provenance = replace(request, source_harness_snapshot_sha="f" * 64)
    assert request.semantic_request_hash() == changed_provenance.semantic_request_hash()


def test_effective_view_change_invalidates_cache(tmp_path, request):
    envelope = {"schema_version": "agent-envelope/1", "kind": "stage_result",
                "stage": "select", "payload": {"chosen_candidate_id": "identity"}}
    backend = ReplayAgentBackend([AgentResponse.valid(envelope, raw_response={"id": "r1"})])
    cached = CachedAgentBackend(backend, EffectiveRequestCache(tmp_path))
    cached.complete(request)
    changed = replace(request, effective_harness_view_sha="9" * 64)
    with pytest.raises(KeyError, match="replay response exhausted"):
        cached.complete(changed)


def test_successful_malformed_response_is_cached(tmp_path, request):
    malformed = AgentResponse(
        transport_ok=True,
        raw_response={"id": "r-bad", "choices": [{"message": {"content": "not-json"}}]},
        assistant_text="not-json",
        parsed_envelope=None,
        parse_status="INVALID_AGENT_ENVELOPE",
        provider_metadata={"model": "gpt-5.5"},
    )
    backend = ReplayAgentBackend([malformed])
    cached = CachedAgentBackend(backend, EffectiveRequestCache(tmp_path))
    assert cached.complete(request).parse_status == "INVALID_AGENT_ENVELOPE"
    assert cached.complete(request).raw_response["id"] == "r-bad"
    assert backend.call_count == 1
```

- [ ] **Step 3: Run the new runtime tests and verify missing-symbol failures**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/runtime/test_agent_backend.py \
  SelfEvolvingHarnessTS/tests/runtime/test_llm_cache.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t04_red
```

Expected: collection fails because `runtime.agent_backend` and `runtime.llm_cache` do not exist.

- [ ] **Step 4: Implement the provider-independent request and response contracts**

Use immutable JSON-native fields and make the exact relay request part of semantic identity:

```python
DEFAULT_AGENT_MODEL = "gpt-5.5"
DEFAULT_AGENT_BASE_URL = "https://api.agicto.cn/v1"
OPENAI_SDK_VERSION = "2.45.0"


@dataclass(frozen=True)
class AgentRequest:
    case_id: str
    role: str
    stage: str
    call_index: int
    replicate_id: str
    messages: tuple[Mapping[str, object], ...]
    envelope_schema_sha: str
    tool_schema_sha: str
    tool_result_schema_sha: str
    stage_schema_sha: str
    public_case_view_sha: str
    effective_harness_view_sha: str
    tool_context_sha: str
    source_harness_snapshot_sha: str = ""
    model: str = DEFAULT_AGENT_MODEL
    base_url: str = DEFAULT_AGENT_BASE_URL
    sdk_version: str = OPENAI_SDK_VERSION
    capability_flags: Mapping[str, bool] = field(default_factory=lambda: {
        "native_tools": False,
        "structured_outputs": False,
        "reasoning_controls": False,
        "provider_seed": False,
    })
    cache_schema_version: str = "effective-request/1"

    def semantic_request_hash(self) -> str:
        return canonical_sha256({
            "provider": "agicto-chat-completions",
            "base_url": self.base_url,
            "model": self.model,
            "sdk_version": self.sdk_version,
            "capability_flags": self.capability_flags,
            "messages": self.messages,
            "envelope_schema_sha": self.envelope_schema_sha,
            "tool_schema_sha": self.tool_schema_sha,
            "tool_result_schema_sha": self.tool_result_schema_sha,
            "stage_schema_sha": self.stage_schema_sha,
            "public_case_view_sha": self.public_case_view_sha,
            "effective_harness_view_sha": self.effective_harness_view_sha,
            "tool_context_sha": self.tool_context_sha,
            "cache_schema_version": self.cache_schema_version,
        })
```

`source_harness_snapshot_sha` is provenance and is intentionally excluded. Validate role as `fast|slow`, stage/name as canonical strings, SHA fields as full lowercase digests, the HTTPS base URL as a canonical origin ending in `/v1`, and all messages as finite JSON-native values. `AgentRequest.for_stage` never reads environment variables.

`AgentResponse` stores `transport_ok`, exact raw response, assistant text, parsed local envelope or null, `parse_status`, finish reason, and provider metadata. `ReplayAgentBackend` accepts either an ordered sequence for focused tests or an immutable semantic-hash mapping loaded from JSONL. It raises on a replay miss, exposes `clone()` and `call_count`, and is the backend used by default tests and offline acceptance runs.

- [ ] **Step 5: Implement the Chat Completions adapter**

For every request, call only:

```python
completion = client.chat.completions.create(
    model=request.model,
    messages=list(request.messages),
)
```

Do not send `tools`, `tool_choice`, `response_format`, reasoning fields, seed, temperature, `store`, or undocumented relay parameters in M0. Extract `choices[0].message.content`; require a nonempty string containing exactly one JSON value and validate it locally against `agent-envelope/1`. Preserve `completion.model_dump(mode="json")` as the raw response plus response ID, returned model, finish reason, and usage as provenance.

Map OpenAI SDK connection/time-out errors, rate limits, and API status 408/409/429/5xx to `AgentTransportError`. A successful response with absent, non-JSON, multiple-JSON, or schema-invalid content becomes `parse_status="INVALID_AGENT_ENVELOPE"`; it is Agent behavior, not infrastructure failure.

The adapter constructor accepts either an injected client or explicit `api_key`, `base_url="https://api.agicto.cn/v1"`, and `timeout_seconds=120`. Only the CLI reads `AGICTO_API_KEY`; production modules never read or log it themselves. Reject an empty key before constructing the SDK client.

- [ ] **Step 6: Implement the immutable effective-request cache**

Use this namespace record:

```python
@dataclass(frozen=True)
class CacheKey:
    case_id: str
    role: str
    stage: str
    call_index: int
    replicate_id: str
    semantic_request_hash: str
```

Store each response at `<cache_root>/<sha256(canonical CacheKey)>.json`. The record contains key, source snapshot SHA, relay origin, requested model alias, SDK version, capability flags, exact messages, local schema hashes, raw response, assistant text, parsed envelope, parse status, raw-response hash, semantic-response hash when parseable, provider metadata, and `created_at` provenance. It never contains the API key or authorization header. Writes use a temporary file plus `Path.replace`; an existing key with different response bytes raises `ValueError("immutable cache collision")`.

`CachedAgentBackend.complete` returns a hit before invoking its delegate. It writes every transport-success response, including malformed output, and never writes an `AgentTransportError`. Expose `CacheReceipt(hit, key_sha, response_hash)` and attach it to the returned response so paired replay can prove eligible reuse.

- [ ] **Step 7: Run runtime backend and cache tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/runtime/test_agent_backend.py \
  SelfEvolvingHarnessTS/tests/runtime/test_llm_cache.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t04_green
```

Expected: all tests pass, `FakeCompletions` records exactly one SDK call, and no test requires a key or network access.

- [ ] **Step 8: Commit Task 4**

```bash
git add requirements-m0.txt runtime tests/runtime/test_agent_backend.py tests/runtime/test_llm_cache.py
git commit -m "feat: add relay-safe GPT-5.5 agent backend"
```

---

### Task 5: Implement the Shared TTHA Agent Core, Retrieval, and Fast/Slow Roles

**Files:**

- Create: `methods/ttha/agent_core.py`
- Create: `methods/ttha/public_tools.py`
- Create: `methods/ttha/retrieval.py`
- Create: `methods/ttha/fast_agent.py`
- Create: `methods/ttha/slow_agent.py`
- Create: `methods/ttha/method.py`
- Create: `methods/ttha/schemas/agent_envelope_v1.json`
- Create: `methods/ttha/schemas/tool_result_v1.json`
- Create: `methods/ttha/schemas/fast_inspect_v1.json`
- Create: `methods/ttha/schemas/fast_propose_v1.json`
- Create: `methods/ttha/schemas/fast_select_v1.json`
- Create: `methods/ttha/schemas/slow_edit_v1.json`
- Modify: `methods/ttha/harness/compiler.py`
- Modify: `methods/ttha/__init__.py`
- Test: `tests/methods/test_ttha_agent.py`

**Interfaces:**

- Consumes: Tasks 1–4 contracts/runtime, `PreparationRequest`, `HarnessSnapshot`, an injected `PublicToolGateway`, and an injected `AgentBackend`.
- Produces: `AgentRole`, `PublicAgentInput`, `EffectiveHarnessView`, `TTHAAgentCore.run_stage`, `TTHAFastAgent.prepare`, `TTHASlowAgent.propose_edit`, and `TTHAMethod.prepare`.

- [ ] **Step 1: Write failing same-core, identity, and retrieval-scope tests**

```python
import numpy as np

from SelfEvolvingHarnessTS.contracts.method import PreparationStatus
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent


def test_fast_and_slow_paths_share_the_same_agent_core(replay_backend, h0, tool_gateway):
    core = TTHAAgentCore(replay_backend, tool_gateway)
    fast = TTHAFastAgent(core)
    slow = TTHASlowAgent(core)
    assert fast.core is core
    assert slow.core is core


def test_fast_path_explicit_identity_maps_to_abstention(identity_replay, h0, public_request):
    result, trace = TTHAFastAgent(identity_replay.core).prepare(public_request, h0)
    assert result.status is PreparationStatus.ABSTAINED
    np.testing.assert_array_equal(result.prepared.values, public_request.values)
    assert result.program is None
    assert trace.chosen_candidate_id == "identity"


def test_out_of_scope_skill_does_not_change_effective_view(h0, capability_skill, public_features):
    from dataclasses import replace
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    baseline = resolve_harness_view(h0, public_features)
    edited_snapshot = replace(h0, skills=(*h0.skills, capability_skill))
    edited = resolve_harness_view(edited_snapshot, public_features)
    assert capability_skill.skill_id not in edited.skill_ids
    assert baseline.effective_harness_view_sha == edited.effective_harness_view_sha
```

- [ ] **Step 2: Run the Agent tests and verify missing modules**

Run the test file. Expected: collection fails for `methods.ttha.agent_core`.

- [ ] **Step 3: Implement deterministic retrieval and effective-view identity**

`EffectiveHarnessView` contains only the content consumed by a call:

```python
@dataclass(frozen=True)
class EffectiveHarnessView:
    instruction: str
    skills: tuple[SkillEntry, ...]
    memories: tuple[MemoryEntry, ...]
    controls: Mapping[str, object]
    effective_harness_view_sha: str

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(skill.skill_id for skill in self.skills)
```

Always include bootstrap skills. Evaluate capability applicability with `validate_applicability` against the public feature map, score matching capability entries deterministically by the count of satisfied leaves followed by `skill_id`, and retain `top_k`. Safety entries follow their declared applicability. An unmatched new skill must not displace a matched old skill and must not enter the effective SHA. Hash exact instruction, selected skills, selected memories, and role-relevant controls only.

- [ ] **Step 4: Implement the public tool gateway and immutable tool receipts**

Define a narrow injected protocol:

```python
class PublicToolGateway(Protocol):
    @property
    def context_sha(self) -> str:
        raise NotImplementedError

    def schemas_for(
        self, *, role: AgentRole, stage: str
    ) -> tuple[Mapping[str, object], ...]:
        raise NotImplementedError

    def call(self, name: str, arguments: Mapping[str, object]) -> PublicToolReceipt:
        raise NotImplementedError
```

`PublicToolReceipt` records tool name, normalized public arguments/result, context SHA, and receipt SHA. The complete fixed probe panel is computed and attached to `PublicAgentInput` before the Agent call; it is not calculated in response to a model query. If the gateway exposes `read_fixed_probe_panel`, that zero-argument tool only returns the same immutable attached receipt and cannot select probes/beta values or trigger evaluation. The default gateway otherwise exposes only observable summary/localization tools. It rejects paths, arbitrary evaluator queries, candidate IDs, `J`, absolute `U`, `R_private`, clean values, injection metadata, and undeclared tool names. Slow role exposes no series/evaluator tool; it receives only the sanitized card and surface descriptions.

- [ ] **Step 5: Implement one role-aware Agent core and local-envelope tool loop**

Use one class, not separate model clients. Its concrete public method is
`run_stage(*, role: AgentRole, stage: str, case_id: str, public_input: Mapping[str, object], harness_view: EffectiveHarnessView, output_schema_name: str, output_schema: Mapping[str, object], source_snapshot_sha: str) -> AgentStageResult`.

```python
class AgentRole(str, Enum):
    FAST = "fast"
    SLOW = "slow"


class TTHAAgentCore:
    def __init__(self, backend: AgentBackend, tools: PublicToolGateway):
        self.backend = backend
        self.tools = tools
```

Resolve the system instruction, public input, allowed local tool descriptions, expected stage schema, and `agent-envelope/1` contract into exact Chat Completions messages before constructing `AgentRequest`. Do not pass provider-native tool or structured-output parameters. For a validated `kind="tool_request"` envelope, require a fresh canonical `call_id`, invoke the gateway, append the assistant's exact JSON envelope, then append one user message containing a `tool-result/1` JSON object with the matching call ID and immutable receipt. Increment `call_index` and call the same backend again. A `kind="stage_result"` envelope must name the requested stage and its payload must validate against that stage's local schema. Stop after eight tool rounds with an Agent protocol failure. Never accept model-emitted tool receipts or PASS/FAIL judgments as runtime truth.

The two envelope schemas use `additionalProperties: false`. `tool-result/1` contains only `schema_version`, `call_id`, `tool_name`, `ok`, `public_result`, and `receipt_sha`; it cannot carry arbitrary prose or private handles. Duplicate call IDs, mismatched tool names, undeclared tools, and a stage result for the wrong stage are protocol failures recorded in the trace.

Load the four checked-in strict schemas. `fast_inspect_v1` returns inspected region fractions, requested public tools, and uncertainty; `fast_propose_v1` returns zero to two candidate objects with non-empty operator steps; `fast_select_v1` returns one `chosen_candidate_id` plus public verification actions; `slow_edit_v1` returns one EditManifest-shaped object and no filesystem patch text.

- [ ] **Step 6: Implement fast orchestration and mechanical tracing**

`TTHAFastAgent.prepare` performs exactly:

```text
public feature extraction
→ effective Harness retrieval
→ inspect stage
→ propose stage
→ canonical Program validation/compilation
→ runtime identity injection and deterministic deduplication
→ public risk filtering (identity retained)
→ select stage
→ explicit choice validation
→ canonical executor
→ runtime-built DecisionTrace
```

Program validation rejects unknown/deprecated operators, task-incompatible operators, shape-changing operators, non-finite parameters, and programs longer than four steps. The runtime, not the model, computes program SHAs, compilation/execution status, modified indices, and effect-equivalence. Invalid/malformed Agent output becomes a failed `PreparationResult` with a trace; it is not infrastructure `INCONCLUSIVE`.

`TTHAMethod(method_id="ttha_m0")` wraps `TTHAFastAgent` and the active snapshot store to satisfy the canonical `Method.prepare` protocol. Identity returns `ABSTAINED`; a successful PROGRAM returns `PREPARED`; protocol/compile/execute failure returns `FAILED`.

- [ ] **Step 7: Implement slow orchestration over public cards only**

`TTHASlowAgent.propose_edit(card, surface_catalog, snapshot)` accepts typed/mapping values already sanitized by minipipe. It rejects absolute paths, private artifact handles, and cards with unvalidated applicability ASTs. It resolves the slow-role view from the same snapshot, calls `core.run_stage(role=SLOW, stage="edit")`, and returns an untrusted `EditManifest` object for the controller to validate. It cannot write files, apply its edit, or choose its verdict.

- [ ] **Step 8: Add deterministic replay fixtures and run Agent tests**

Check in test-local replay responses that cover:

1. inspect → no tool call → propose no program → select identity;
2. inspect → one public tool call → propose two programs → select one;
3. malformed propose output → Agent behavior failure;
4. sanitized pattern card → one ADD SkillEntry manifest;
5. unmatched capability skill → identical effective view.

After creating the prompt schemas, Agent-core resolver, and retrieval compiler, extend `harness/compiler.py`'s runtime dependency list to hash those exact source/schema bytes, then regenerate `methods/ttha/harness/h0/snapshot.lock.json`. The Task 2 lock must fail before regeneration because the runtime dependency set changed, then pass after `--write-lock`. Stage the compiler and updated lock in this task's commit; this is the intentional finalization of H0's runtime bundle, not an implicit mutation.

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/methods/test_ttha_agent.py \
  SelfEvolvingHarnessTS/tests/methods/test_ttha_h0.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t05
```

Expected: all tests pass without `AGICTO_API_KEY` and without network access.

- [ ] **Step 9: Commit Task 5**

```bash
git add methods/ttha tests/methods/test_ttha_agent.py
git commit -m "feat: add shared TTHA fast and slow agent core"
```

---

### Task 6: Define Minipipe Case Contracts and Generate the Deterministic Public/Private Corpus

**Files:**

- Create: `evaluation/minipipe/__init__.py`
- Create: `evaluation/minipipe/contracts.py`
- Create: `evaluation/minipipe/config/m0_rules.json`
- Create: `evaluation/minipipe/corpus/__init__.py`
- Create: `evaluation/minipipe/corpus/injections.py`
- Create: `evaluation/minipipe/corpus/generate.py`
- Create: `evaluation/minipipe/artifacts/.gitignore`
- Test: `tests/minipipe/test_corpus.py`

**Interfaces:**

- Consumes: Task 1 canonical hashing and observable schema.
- Produces: `CasePurpose`, `PrivateSyntheticCase`, `PublicCaseView`, `ArtifactRoots`, `build_core_corpus(rules)`, and `write_case_artifacts(cases, roots)`.

- [ ] **Step 1: Write failing corpus-count, determinism, and wall tests**

```python
import json

import numpy as np

from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus


def test_core_corpus_is_24_targets_plus_12_risks_and_reproducible(m0_rules):
    first = build_core_corpus(m0_rules)
    second = build_core_corpus(m0_rules)
    assert len(first.targets) == 24
    assert len(first.risks) == 12
    assert [case.private_sha for case in first.all_cases] == [
        case.private_sha for case in second.all_cases
    ]
    assert {case.private_family for case in first.targets} == {
        "missing", "impulsive_outlier", "level_shift", "period_change"
    }


def test_public_view_contains_only_corrupt_context_and_opaque_identity(m0_rules):
    case = build_core_corpus(m0_rules).targets[0]
    public = case.to_public_view()
    assert public.case_id.startswith("m0-")
    assert case.private_family not in json.dumps(public.to_json())
    assert "injection" not in json.dumps(public.to_json()).lower()
    np.testing.assert_array_equal(public.values, case.corrupt_context, equal_nan=True)
    assert "clean_context" not in public.to_json()
    assert "clean_future" not in public.to_json()


def test_public_and_private_artifacts_are_written_to_disjoint_roots(tmp_path, m0_rules):
    from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import write_case_artifacts

    corpus = build_core_corpus(m0_rules)
    roots = write_case_artifacts(corpus.all_cases, tmp_path)
    assert roots.public.resolve().parent == roots.private.resolve().parent
    assert roots.public.resolve() != roots.private.resolve()
    public_text = "\n".join(path.read_text() for path in roots.public.rglob("*.json"))
    assert "private_family" not in public_text
    assert "clean_future" not in public_text
```

- [ ] **Step 2: Run the corpus tests and verify missing-module failures**

Expected: collection fails for `evaluation.minipipe`.

- [ ] **Step 3: Implement immutable public/private case contracts**

Use opaque public IDs allocated after deterministic sorting. Store oracle facts only on the private object:

```python
class CasePurpose(str, Enum):
    TARGET = "target"
    RISK_CLEAN = "risk_clean"
    RISK_GENUINE_EVENT = "risk_genuine_event"


@dataclass(frozen=True)
class PublicCaseView:
    schema_version: str
    case_id: str
    values: np.ndarray
    task_kind: str
    public_features: Mapping[str, object]
    public_probe_panel: Mapping[str, object] | None
    public_case_view_sha: str


@dataclass(frozen=True)
class PrivateSyntheticCase:
    case_id: str
    seed: int
    purpose: CasePurpose
    private_family: str
    private_severity: str
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray
    oracle_affected_indices: tuple[int, ...]
    observable_counterpart_id: str | None
    private_sha: str

    def to_public_view(self) -> PublicCaseView:
        return PublicCaseView.create(
            case_id=self.case_id,
            values=self.corrupt_context,
            task_kind="forecast",
            public_features={},
        )
```

Copy arrays, make them read-only, reject non-one-dimensional or wrong-length data, and hash float64 bytes plus structural metadata. `PublicCaseView.create` computes the hash with `public_probe_panel=None`; `with_features(features)` and `with_probe_panel(public_receipt)` return rehashed immutable views after Task 8. `with_probe_panel` accepts only the strict public receipt serializer and refuses private receipt types. `PublicCaseView.to_json()` serializes NaNs as explicit missing-position runs plus finite values, not non-standard JSON NaN literals. Public IDs are `m0-0001` onward and reveal neither purpose nor family.

- [ ] **Step 4: Implement exact deterministic base series and four injections**

For each seed, generate 240 values (`192` context + `48` future) with a local `np.random.default_rng(seed)`:

```python
t = np.arange(240, dtype=np.float64)
period = 24.0 + (seed % 3) * 2.0
base = (
    0.0025 * t
    + 0.8 * np.sin(2.0 * np.pi * t / period)
    + 0.25 * np.sin(2.0 * np.pi * t / 7.0 + 0.3)
    + rng.normal(0.0, 0.04, size=t.size)
)
```

Apply target injections to context only, using scale `max(std(clean_context), 1e-8)`:

| Family | Mild | Severe | Oracle affected region |
| --- | --- | --- | --- |
| missing | NaN run length 12 from index 108 | NaN run length 30 from index 102 | missing run |
| impulsive/outlier | spikes at `[119, 147]` with alternating `±6*scale` | spikes at `[111, 128, 149, 166]` with alternating `±10*scale` | spike indices |
| level shift | add `1.5*scale` on `[128, 168)` | add `3.0*scale` on `[112, 176)` | shifted interval |
| period change | on `[96, 192)`, replace the primary seasonal component with period `0.75*period` | same with period `0.55*period` | `[96, 192)` |

Create one matched risk case per family and seed: unchanged clean for missing; a genuine two-point pulse whose continuation is present in the clean target for outlier; a genuine sustained level transition propagated through the future for level shift; and a genuine frequency-regime transition propagated through the future for period change. These risks are not treated as corruptions and therefore have no repair target.

- [ ] **Step 5: Check in executable M0 constants and physical artifact roots**

Write `m0_rules.json` exactly as locked near the top of this plan. Add corpus seeds, lengths, beta values, and family list under a nested `corpus` key without changing the locked thresholds. Parse with a strict loader that rejects unknown top-level keys and records the rules SHA.

`ArtifactRoots.create(run_root)` creates only:

```text
<run_root>/artifacts/public/
<run_root>/artifacts/private/
```

The public writer accepts only `PublicCaseView`; the private writer accepts only private records. Add `evaluation/minipipe/artifacts/.gitignore` containing `public/`, `private/`, and `!.gitignore` so runtime artifacts cannot be committed accidentally.

- [ ] **Step 6: Run corpus tests twice**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_corpus.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t06
```

Expected: all tests pass; a repeated run produces the same 36 private SHAs.

- [ ] **Step 7: Commit Task 6**

```bash
git add evaluation/minipipe tests/minipipe/test_corpus.py
git commit -m "feat: add deterministic minipipe corpus contracts"
```

---

### Task 7: Implement the Pinned Frozen-Chronos Valuator and Outcome Ledger

**Files:**

- Create: `evaluation/minipipe/valuation/__init__.py`
- Create: `evaluation/minipipe/valuation/chronos.py`
- Create: `evaluation/minipipe/valuation/rolling_observed.py`
- Create: `evaluation/minipipe/valuation/outcomes.py`
- Create: `evaluation/minipipe/valuation/model_manifest.json`
- Test: `tests/minipipe/test_valuator.py`

**Interfaces:**

- Consumes: private case contexts/futures and the locally cached Chronos snapshot.
- Produces: `FrozenChronosValuator`, `ValuationReceipt`, `RollingObservedValuator`, `RollingObservedReceipt`, `OutcomeView`, `evaluate_outcome`, and `evaluate_candidate_regret`.

- [ ] **Step 1: Write failing deterministic-utility and sign-convention tests**

```python
import numpy as np

from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import FrozenChronosValuator
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.outcomes import evaluate_outcome


class FakeChronos:
    def predict_quantiles(self, contexts, *, prediction_length, quantile_levels):
        import torch
        mean = torch.zeros((len(contexts), prediction_length), dtype=torch.float32)
        quantiles = mean[:, :, None]
        return quantiles, mean


def test_utility_is_negative_clean_scaled_nrmse_and_fill_is_recorded():
    context = np.arange(192, dtype=float)
    context[10:12] = np.nan
    clean_context = np.arange(192, dtype=float)
    future = np.ones(48, dtype=float)
    receipt = FrozenChronosValuator(pipeline=FakeChronos()).evaluate(
        context, future, scale_context=clean_context
    )
    expected_j = 1.0 / np.std(np.arange(192, dtype=float))
    assert receipt.loss_j == pytest.approx(expected_j)
    assert receipt.utility_u == pytest.approx(-expected_j)
    assert receipt.fill_fraction == pytest.approx(2 / 192)


def test_outcome_definitions_use_higher_is_better_utility():
    outcome = evaluate_outcome(
        clean_u=-0.10,
        corrupt_u=-0.40,
        prepared_u=-0.20,
        identity_u=-0.40,
        candidate_utilities={"identity": -0.40, "agent-0": -0.20},
        chosen_candidate_id="agent-0",
        damage_noise_floor=0.01,
    )
    assert outcome.damage_d == pytest.approx(0.30)
    assert outcome.repair_gain_g == pytest.approx(0.20)
    assert outcome.nrr == pytest.approx(2 / 3)
    assert outcome.selection_regret == 0.0
```

- [ ] **Step 2: Run the tests and verify the valuator module is absent**

Expected: collection fails for `evaluation.minipipe.valuation`.

- [ ] **Step 3: Check in and enforce the frozen model manifest**

Write `model_manifest.json` exactly from the locked valuator JSON near the top of this plan. Add `manifest_sha` computed over the remaining fields. Runtime rejects a different model ID, revision, device, dtype, context length, prediction length, or dependency version. Loading must call:

```python
BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    device_map="cpu",
    torch_dtype=torch.float32,
    local_files_only=True,
)
```

If the local snapshot is absent, raise a typed `FrozenModelUnavailable` naming the exact model/revision; never download or substitute another model.

- [ ] **Step 4: Implement deterministic evaluation receipts**

`FrozenChronosValuator.evaluate(context, clean_future, *, scale_context)`:

1. requires exactly 192 context and 48 future values;
2. linearly interpolates non-finite context positions with nearest endpoint behavior;
3. records missing count/fraction and SHA of both raw and filled context;
4. passes one float32 CPU tensor to `predict_quantiles(..., prediction_length=48, quantile_levels=[0.5])`;
5. uses the returned mean, converted to float64, as point forecast;
6. computes the shared scale from the explicitly supplied private clean context and `J/U` exactly as locked;
7. rejects non-finite forecasts instead of silently coercing them.

`ValuationReceipt` contains model manifest SHA, input/future/forecast SHAs, `loss_j`, `utility_u`, fill fraction, scale, prediction length, and evaluator status. It contains no Agent-facing serializer.

Implement `RollingObservedValuator` as a separate deployment-computable adapter over the same frozen model. For origins `(96, 120, 144, 168)` and horizon `24`, it forecasts each held-out slice using only the prefix before that origin, scores against the already-observed corrupt slice, and normalizes by `max(std(finite prefix), 1e-8)`. A probe transformation may change only the prefix; the held-out target slice is always read from the unchanged public corrupt series. Exclude an origin with fewer than 12 finite target values; if no origin survives, return `status="UNKNOWN"` rather than a fabricated score. Batch the surviving origins per arm and record origin list, model/input/target/forecast SHAs, per-origin losses, mean public utility, and status. This receipt contains no clean future, injection metadata, or private outcome value.

- [ ] **Step 5: Implement the private outcome ledger**

Define:

```python
@dataclass(frozen=True)
class OutcomeView:
    clean_u: float
    corrupt_u: float
    prepared_u: float
    damage_d: float
    repair_gain_g: float
    nrr: float | None
    over_restoration: bool
    selection_regret: float
    target_window_gain: float | None
    outside_window_change: float | None
    counterpart_change: float | None
    non_target_collateral: float | None
    agent_decision_status: str
    system_capability_status: str
```

Set `over_restoration = prepared_u - clean_u > utility_tolerance`. Candidate regret includes identity with `U(identity)=U(corrupt)` and is private. NRR is null unless `D > critic_damage_min`. Keep Agent decision and system capability as separate fields; the period `CORRECT_IDENTITY` rule is applied after expressibility in Task 10.

- [ ] **Step 6: Run fake-pipeline unit tests and one pinned-local smoke test**

Mark the real model test `@pytest.mark.frozen_model` and assert two private evaluations and two rolling-observed evaluations of one case have identical forecast SHA and utility within `1e-12`. Add tests proving an intervention never changes its held-out target slice, invalid origins are excluded, all-invalid origins yield `UNKNOWN`, and no private field appears in `RollingObservedReceipt.to_public_dict()`. Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_valuator.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t07
```

Expected: unit and local-model tests pass without network access. If the pinned local snapshot is missing, the real-model test must fail with `FrozenModelUnavailable`, not skip or download.

- [ ] **Step 7: Commit Task 7**

```bash
git add evaluation/minipipe/valuation tests/minipipe/test_valuator.py
git commit -m "feat: add private and rolling-observed Chronos valuation"
```

---

### Task 8: Add Observable Features, Fixed Probe Curves, and Expressibility Witnesses

**Files:**

- Create: `evaluation/minipipe/probes/__init__.py`
- Create: `evaluation/minipipe/probes/features.py`
- Create: `evaluation/minipipe/probes/panel.py`
- Create: `evaluation/minipipe/probes/expressibility.py`
- Create: `evaluation/minipipe/probes/transformation_classes.json`
- Create: `evaluation/minipipe/baselines/fixed_program_baseline_v1.json`
- Test: `tests/minipipe/test_probes.py`

**Interfaces:**

- Consumes: canonical operators/executor, public corrupt context, private cases for grader-only witnesses, and Task 7 valuator.
- Produces: `extract_public_features`, `ProbeSpec`, `ProbePanel.run_public`, `ProbePanel.run_private`, `PublicProbePanelReceipt`, `PrivateProbePanelReceipt`, `PeriodDiagnostic`, `ExpressibilityStatus`, and `evaluate_expressibility`.

- [ ] **Step 1: Write failing monotonic-beta, diagnostic-only, and proof-asymmetry tests**

```python
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.expressibility import ExpressibilityStatus
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import M0_PROBE_SPECS


def test_every_repair_probe_has_three_monotonic_strengths():
    for name in ("imputation", "clipping", "denoising", "level_correction"):
        spec = M0_PROBE_SPECS[name]
        assert spec.betas == (0.25, 0.50, 0.75)
        assert spec.aggressiveness == tuple(sorted(spec.aggressiveness))


def test_period_is_diagnostic_only_and_declares_repair_unavailable(period_case, panel):
    result = panel.run_public(period_case.to_public_view())
    assert result.period_diagnostic.repair_available is False
    assert "period" not in result.response_curves


def test_oracle_yes_existing_feature_not_derived_is_procedure_gap(expressibility_fixture):
    result = expressibility_fixture(oracle_succeeds=True, observable_succeeds=False,
                                    required_feature_is_in_closed_vocabulary=True)
    assert result.oracle_witness.succeeded is True
    assert result.status is ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
    assert result.cause_code == "OBSERVABLE_DERIVATION_PROCEDURE_GAP"


def test_missing_required_public_feature_is_noneditable_schema_gap(expressibility_fixture):
    result = expressibility_fixture(oracle_succeeds=True, observable_succeeds=False,
                                    required_feature_is_in_closed_vocabulary=False)
    assert result.status is ExpressibilityStatus.EXPRESSIBILITY_UNKNOWN
    assert result.cause_code == "OBSERVABLE_FEATURE_SCHEMA_GAP"


def test_absent_complete_period_class_proves_unavailable(period_case, expressibility):
    result = expressibility.evaluate(period_case)
    assert result.status is ExpressibilityStatus.PROVEN_UNAVAILABLE
    assert result.missing_transformation_class == "period_correction"
```

- [ ] **Step 2: Run the tests and verify missing probe modules**

Expected: collection fails for `evaluation.minipipe.probes`.

- [ ] **Step 3: Implement the closed public feature extractor**

Compute only features declared in `contracts/observables.py`:

- missing fraction and longest contiguous missing-run fraction;
- median/MAD robust-z peak, with MAD floor `1e-8`;
- a candidate region from thresholded robust-z points expanded by two neighbors;
- level-excursion score from the maximum absolute two-sided median difference over split points `24..168`, normalized by MAD;
- estimated region fractions from the union of missing/outlier/level candidate regions;
- pre/post dominant ACF period over integer lags `4..48` using the first and last 80 finite/interpolated points;
- period-change score `abs(post-pre)/max(pre, 1)` and `period_repair_available=false`;
- probe direction labels populated only after the fixed panel: `positive`, `flat`, `overdose_collapse`, or `harmful`.

The extractor returns a mapping restricted to the closed vocabulary plus a `feature_context_sha`. The composition root calls `structural_view.with_features(extracted.mapping)` before any Agent request. The extractor never accepts clean data or private injection parameters.

- [ ] **Step 4: Implement separate public and private probe panels exactly as locked**

`ProbeSpec` stores name, beta tuple, human-independent aggressiveness tuple, mapping version, required detected region, operator IDs, and implementation SHA. Implement the four transformations exactly from the locked mappings. All operator execution goes through `runtime.executor`; no copied operator functions are allowed.

`run_public` is always computed before the fast Agent. It applies each of the four repair probes at all three beta values to the prefix at each fixed rolling origin, invokes `RollingObservedValuator`, and records `R_public(beta)=U_public(probe)-U_public(corrupt)`. Use one shared corrupt baseline batch plus exactly twelve probe-arm batches. The schedule, origins, horizon, finite-target rule, evaluator manifest, and probe specs are fixed, so the Agent cannot choose queries or timing. Round only the public serialized deltas to six decimals; internal scoring retains full precision.

`PublicProbePanelReceipt` contains only panel/spec/evaluator/input/feature-context SHAs, period diagnostic, status, and for each fixed point: probe ID, beta, rounded `R_public`, modified fraction, response shape, and receipt SHA. It contains no absolute `J/U`, `R_private`, clean reference, candidate ID/program/value, injection metadata, or arbitrary operator parameters. The panel is a deliberately accepted bounded leakage surface: an Agent may reproduce a fixed panel transformation and infer its measured public score, but the panel point is an instrument, not a candidate-utility query.

`run_private` executes the same fixed probe arms after the case is graded and records `R_private(beta)=U_private(probe)-U_private(corrupt)` against clean future. It shares deterministic model/config inputs but never serializes through ProbeAPI. Record public/private curve agreement privately as an instrumentation field; disagreement does not authorize exposing private values.

Classify both response shapes with `probe_gain_min` and `probe_margin_min`; for example, a positive middle response followed by a drop greater than the margin is `overdose_collapse`. `run_public` and `run_private` must have distinct receipt types and serializers so an accidental private field cannot cross the wall.

The period diagnostic returns pre/post estimates, change score, ACF/spectral consistency, and `repair_available=false`; it performs no transformation and no candidate utility query for the Agent.

- [ ] **Step 5: Add the private baseline and proof-safe expressibility logic**

Check in a schema-versioned private positive-witness catalog independently authored from the canonical registry:

```json
[
  [["impute_linear", {}]],
  [["hampel_filter", {}]],
  [["repair_level_shift", {}]]
]
```

Declare bounded observable witness grammars for missing (`impute_linear` plus mechanism-distinct registry alternatives), impulsive/outlier (`hampel_filter` and `winsorize`), and level shift (`repair_level_shift`). Locations and parameters must come from `extract_public_features`; the oracle arm may separately use private affected indices. The file lives under `evaluation/minipipe/baselines`, is loaded only by the grader, and is never imported by `methods/ttha`. It is not an H_ref artifact; historical reproduction material remains isolated under `evaluation/benchmark_v02/_frozen_reference/`.

Co-design the witness grammar and four-family corpus through a checked coverage matrix. Each repairable family/signature must have at least one observable witness route; `period_change` must have an unavailable-class receipt. Hash the corpus definition, witness catalog, parameterization rules, and coverage matrix into one joint instrument SHA so either side cannot drift silently.

Write `transformation_classes.json` with a complete category declaration derived from canonical `OPERATOR_METADATA` and this required-family map:

```json
{
  "missing": "impute",
  "impulsive_outlier": "outlier",
  "level_shift": "structural",
  "period_change": "period_correction"
}
```

Use the transformation-class map in both directions. Family → required class selects witnesses and proves a declared class unavailable. Operator category → implied mechanism claim supplies a supplemental stage-4 mechanism signal from a selected Program. It is not the sole mechanism rule because a supply gap may have no Program at all.

`PROVEN_EXPRESSIBLE` requires a successful observable-parameterized witness. Oracle-only success never signs a library gap. `PROVEN_UNAVAILABLE` requires the complete category declaration and an absent required class; this is expected only for period correction. Failed finite search without such proof remains `EXPRESSIBILITY_UNKNOWN`. Oracle success plus failed public parameterization routes to `OBSERVABLE_DERIVATION_PROCEDURE_GAP` when the needed feature already exists in the closed vocabulary, or `OBSERVABLE_FEATURE_SCHEMA_GAP` when it does not. The former may edit only `inspect_and_localize`; the latter is an M0 backlog item. Neither result falsely claims an operator class is unavailable.

- [ ] **Step 6: Run probe tests with the fake valuator and canonical registry**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_probes.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t08
```

Expected: all tests pass; the public panel always contains exactly twelve fixed `R_public` points, the private panel never enters its serializer, no canonical operator has category `period_correction`, and all four repair probes have strictly ordered aggressiveness.

- [ ] **Step 7: Commit Task 8**

```bash
git add evaluation/minipipe/probes evaluation/minipipe/baselines tests/minipipe/test_probes.py
git commit -m "feat: add fixed diagnostic probes and witnesses"
```

---

### Task 9: Derive Four-View CaseFeedback and the First Actionable Fault

**Files:**

- Modify: `evaluation/minipipe/contracts.py`
- Create: `evaluation/minipipe/feedback/__init__.py`
- Create: `evaluation/minipipe/feedback/first_fault.py`
- Create: `evaluation/minipipe/feedback/router.py`
- Create: `evaluation/minipipe/feedback/fault_routes.json`
- Test: `tests/minipipe/test_first_fault.py`

**Interfaces:**

- Consumes: private outcome/probe/expressibility receipts, runtime DecisionTrace, snapshot/retrieval facts, and M0 rules.
- Produces: `Stage`, `AssessmentStatus`, `StageAssessment`, `FaultAttribution`, `OutcomeFeedback`, `MechanismFeedback`, `BehaviorFeedback`, `UpdateAttributionFeedback`, `CaseFeedback`, `assess_case`, and `FaultRouter`.

- [ ] **Step 1: Write failing selection, library, retrieval, operator, and UNKNOWN fixtures**

```python
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import assess_case


def test_effective_candidate_present_but_unchosen_is_selection_miss(case_facts):
    result = assess_case(case_facts(candidate_gains={"identity": 0.0, "agent-0": 0.12},
                                    chosen="identity", all_earlier_stages_pass=True))
    assert result.attribution.first_stage == "CANDIDATE_SELECTION"
    assert result.attribution.fault_code == "SELECTION_MISS"
    assert result.attribution.cause_code == "SELECTION_MISS"


def test_observable_witness_without_capability_skill_is_library_gap(case_facts):
    result = assess_case(case_facts(no_effective_candidate=True,
                                    expressibility="PROVEN_EXPRESSIBLE",
                                    capability_skill_exists=False,
                                    all_earlier_stages_pass=True))
    assert result.attribution.first_stage == "CANDIDATE_SUPPLY"
    assert result.attribution.cause_code == "SKILL_LIBRARY_GAP"
    assert result.attribution.suspect_surface_templates == (
        "skill_library.entries/{skill_id}",
    )


def test_forced_existing_skill_success_is_retrieval_miss(case_facts):
    result = assess_case(case_facts(normal_retrieval=False, forced_skill_succeeds=True,
                                    capability_skill_exists=True,
                                    all_earlier_stages_pass=True))
    assert result.attribution.first_stage == "RETRIEVAL_POLICY"
    assert result.attribution.cause_code == "RETRIEVAL_MISS"


def test_period_unavailable_identity_is_agent_success_and_system_gap(period_facts):
    result = assess_case(period_facts(expressibility="PROVEN_UNAVAILABLE",
                                      chosen="identity", upstream_pass=True))
    assert result.feedback.outcome.agent_decision_status == "CORRECT_IDENTITY"
    assert result.feedback.outcome.system_capability_status == "OPERATOR_GAP"
    assert result.attribution.actionability == "CAPABILITY_BACKLOG"


def test_uncertain_localization_stops_false_downstream_attribution(case_facts):
    result = assess_case(case_facts(localization_iou=0.20, downstream_selection_miss=True))
    assert result.assessments[2].status.value == "UNKNOWN"
    assert result.attribution.fault_code == "LOCALIZATION_UNKNOWN"
    assert result.attribution.actionability == "EVIDENCE_BACKLOG"
```

- [ ] **Step 2: Run the tests and verify feedback symbols are absent**

Expected: imports fail for `feedback.first_fault`.

- [ ] **Step 3: Implement typed four-view feedback records**

Extend `contracts.py` with immutable JSON-native records. `CaseFeedback` is always private and has exactly:

```python
@dataclass(frozen=True)
class CaseFeedback:
    schema_version: str
    case_id: str
    outcome: OutcomeFeedback
    mechanism: MechanismFeedback
    behavior: BehaviorFeedback
    update_attribution: UpdateAttributionFeedback
    assessments: tuple[StageAssessment, ...]
    fault_attribution: FaultAttribution
    private_receipt_refs: tuple[str, ...]
```

Outcome carries Task 7 fields and candidate utilities/regret privately. Mechanism carries localization IoU, observable features, separate `R_public`/`R_private` curves, their agreement receipt, period diagnostic, witness receipts, transformation-class implied claims, and expressibility status. Only the sanitized public probe receipt may later cross the wall. Behavior is mechanically projected from DecisionTrace and forced-replay facts. Update attribution initially contains only `suspect_surface_templates`; its `confirmed_surface` must be null before paired replay.

- [ ] **Step 4: Implement the ordered deterministic assessments**

Emit exactly ten stages in this order:

```python
STAGE_ORDER = (
    "ELIGIBILITY",
    "OBSERVATION",
    "LOCALIZATION",
    "MECHANISM",
    "RETRIEVAL_POLICY",
    "CANDIDATE_SUPPLY",
    "CANDIDATE_SELECTION",
    "COMPILATION",
    "EXECUTION",
    "OUTCOME_RISK",
)
```

Each `StageAssessment` stores `PASS|FAIL|UNKNOWN|NOT_APPLICABLE`, evidence receipt IDs, a versioned `decision_rule_id`, and suspect surface templates. Apply these rules:

1. `ELIGIBILITY`: target `D < critic_damage_min` → `CRITIC_BLIND`/instrumentation; risk cases are eligible for risk checks but have damage `NOT_APPLICABLE`.
2. `OBSERVATION`: discriminative public evidence absent → `OBSERVATION_GAP`; evidence exists but trace did not inspect/call it → `OBSERVATION_PROCEDURE_GAP`.
3. `LOCALIZATION`: non-local cases → `NOT_APPLICABLE`; IoU `<=0.10` → `LOCALIZATION_MISS`; IoU `>=0.30` → PASS; the open interval is `LOCALIZATION_UNKNOWN`.
4. `MECHANISM`: no public repair probe exceeds `probe_gain_min`, or best-versus-second margin is below `probe_margin_min`, → `MECHANISM_UNKNOWN`; discriminative public evidence exists but Agent behavior contradicts it → `MECHANISM_AMBIGUITY`. The selected Program's operator category supplies a supplemental implied claim through the same transformation-class mapping used by witnesses; absence of a Program does not itself fail mechanism. Period is the explicit diagnostic-only exception: a versioned period-change diagnostic above its declared threshold passes mechanism identification without pretending a repair response exists.
5. `RETRIEVAL_POLICY`: an existing capability skill succeeds when forced but normal effective view lacks it → `RETRIEVAL_MISS`.
6. `CANDIDATE_SUPPLY`: no effect-distinct candidate reaches `candidate_gain_min`; choose the cause using the expressibility/skill/forced replay decision tree below.
7. `CANDIDATE_SELECTION`: an effective candidate exists and chosen regret is at least `selection_regret_min` → `SELECTION_MISS`.
8. `COMPILATION`: proposed candidate exists but canonical validation/compilation failed → `IMPLEMENTATION_MISMATCH`.
9. `EXECUTION`: compiled candidate failed or modified outside its declared execution contract → `EXECUTION_MISMATCH`.
10. `OUTCOME_RISK`: chosen target gains less than threshold → outcome gap; clean/genuine risk loses more than epsilon or scope expands → `RISK_GAP`.

Fold assessments by stopping at the first `FAIL` or `UNKNOWN` after eligibility. An earlier `UNKNOWN` cannot be skipped to claim a later causal failure. `NOT_APPLICABLE` and `PASS` continue. Pre-Agent `CRITIC_BLIND` never routes to a Harness edit.

Load every numerical rule from `m0-rules/1`; do not embed another threshold in Python. A PROGRAM is effective only when it is effect-distinct from identity and `U(candidate)-U(identity) >= candidate_gain_min`. `SELECTION_MISS` requires at least one effective candidate and `max_c U(c)-U(chosen) >= selection_regret_min`. Risk stability uses `delta U >= -risk_epsilon` for in-scope clean/genuine cases and exact effective-view/cache/behavior equality for out-of-scope cases. Every `StageAssessment.decision_rule_id` is `<rule-name>@<m0_rules_sha>` so historical decisions remain auditable after a threshold revision.

- [ ] **Step 5: Implement the supply-cause decision tree and proof asymmetry**

Use this order:

```text
PROVEN_UNAVAILABLE required class
  → OPERATOR_GAP / CAPABILITY_BACKLOG
EXPRESSIBILITY_UNKNOWN
  → EXPRESSIBILITY_UNKNOWN / EVIDENCE_BACKLOG
OBSERVABLE_DERIVATION_PROCEDURE_GAP (needed feature already declared)
  → bootstrap inspect_and_localize procedure surface
OBSERVABLE_FEATURE_SCHEMA_GAP (needed feature absent from closed vocabulary)
  → OBSERVATION_CAPABILITY_BACKLOG / non-editable in M0
PROVEN_EXPRESSIBLE + no capability skill
  → SKILL_LIBRARY_GAP / ADD SkillEntry
skill exists + forced succeeds + normal retrieval absent
  → RETRIEVAL_MISS
skill retrieved + forced use still supplies no effective candidate
  → SKILL_CONTENT_GAP
skill works when explicitly constrained but free proposal does not
  → PROPOSAL_CONTROL_GAP
otherwise
  → CANDIDATE_SUPPLY_UNKNOWN / EVIDENCE_BACKLOG
```

Oracle witness success is recorded but never used in the `PROVEN_EXPRESSIBLE` branch.

- [ ] **Step 6: Add the executable fault-to-surface authorization table**

Write `fault_routes.json` with exactly the cause classes from design §7. `FaultRouter.allowed_targets(cause_code)` returns target classes, allowed skill kinds, allowed operations, and actionability. It must reject:

- bootstrap procedure edits for family/capability faults;
- capability edits for `RISK_GAP` unless the target is an existing capability risk guard;
- all M0 edits for `OPERATOR_GAP` and `EXPRESSIBILITY_UNKNOWN`;
- all M0 edits for `OBSERVABLE_FEATURE_SCHEMA_GAP`;
- the observable feature schema/vocabulary and observation-tool Python code; only `OBSERVABLE_DERIVATION_PROCEDURE_GAP` may route to the declared `inspect_and_localize` bootstrap surface.

Because H0 has no capability skills, build `RETRIEVAL_MISS` and `SKILL_CONTENT_GAP` acceptance fixtures from immutable test-only snapshots seeded with one capability SkillEntry. Never weaken H0 merely to make those branches reachable in tests.

- [ ] **Step 7: Run attribution and router tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_first_fault.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t09
```

Expected: fixtures distinguish library, retrieval, skill-content, proposal-control, selection, operator, observable-derivation, observable-schema, and evidence-unknown causes; an oracle-only witness cannot produce `SKILL_LIBRARY_GAP`.

- [ ] **Step 8: Commit Task 9**

```bash
git add evaluation/minipipe/contracts.py evaluation/minipipe/feedback tests/minipipe/test_first_fault.py
git commit -m "feat: derive first actionable minipipe faults"
```

---

### Task 10: Enforce Information Walls and Mine Sanitized FailurePatternCards

**Files:**

- Create: `evaluation/minipipe/feedback/sanitize.py`
- Create: `evaluation/minipipe/feedback/patterns.py`
- Create: `evaluation/minipipe/schemas/failure_pattern_card_v1.json`
- Create: `evaluation/minipipe/schemas/public_case_view_v1.json`
- Test: `tests/minipipe/test_information_walls.py`
- Test: `tests/architecture/test_ttha_dependency_rules.py`

**Interfaces:**

- Consumes: private CaseFeedback records and public case views.
- Produces: `FailurePatternCard`, `sanitize_case_feedback`, `mine_failure_patterns`, `PublicArtifactReader`, and private `ClusterPurityReceipt`.

- [ ] **Step 1: Write failing leak, path, and purity tests**

```python
import json

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.sanitize import sanitize_case_feedback


FORBIDDEN_PUBLIC_KEYS = {
    "private_family", "private_severity", "oracle_affected_indices",
    "clean_context", "clean_future", "candidate_utilities", "loss_j",
    "utility_u", "R_private", "injection_type", "confirmed_surface",
}


def test_sanitizer_removes_oracle_and_judge_fields(private_feedback):
    public = sanitize_case_feedback(private_feedback)
    encoded = json.dumps(public.to_json(), sort_keys=True)
    assert not any(key in encoded for key in FORBIDDEN_PUBLIC_KEYS)
    assert public.confirmed_surface is None


def test_applicability_in_public_card_uses_closed_vocabulary(private_feedback):
    card = sanitize_case_feedback(private_feedback)
    assert card.observable_signature
    with pytest.raises(ValueError, match="unknown observable feature"):
        card.with_applicability({"all": [{"feature": "injection_type", "op": "==", "value": "x"}]})


def test_public_reader_rejects_private_or_parent_paths(artifact_roots):
    reader = PublicArtifactReader(artifact_roots.public)
    with pytest.raises(PermissionError):
        reader.read_json(artifact_roots.private / "case_feedback.jsonl")
    with pytest.raises(PermissionError):
        reader.read_json("../private/case_feedback.jsonl")
```

- [ ] **Step 2: Run wall tests and verify missing sanitizer symbols**

Expected: the sanitizer import fails.

- [ ] **Step 3: Implement allowlist-only sanitization**

Never recursively delete a blacklist from a private object. Instead construct `FailurePatternEvidence` from an explicit allowlist:

```text
opaque case_id
first_stage/fault_code/cause_code/actionability
observable feature bins and probe direction/shape labels
fixed ProbeAPI point IDs, beta values, rounded R_public, modified fractions, and receipt SHAs
normalized BehaviorSignature
public tool/operator/skill IDs already visible to the Agent
sanitized intervention receipt IDs
suspect surface templates
success/counterexample opaque IDs
```

Do not expose numeric `U/J/D/G/NRR/R_private`, candidate rankings, oracle locations/family, clean values, private witness parameters, or confirmed surfaces. The sole judge-derived numeric exception is the already authorized fixed `R_public` receipt. Quantize other public numeric signatures to declared bins before cards are made. Validate every applicability AST through Task 1.

`PublicArtifactReader` accepts a configured public root and resolves every path with `Path.resolve()`. It requires `candidate_path.is_relative_to(public_root)` and `.json|.jsonl|.md` suffixes. Prompt constructors receive typed objects from this reader; they do not accept arbitrary paths.

- [ ] **Step 4: Implement deterministic buckets, contrasts, and private purity receipts**

Bucket by `(fault_code, cause_code, suspect_surface_template, observable_signature_hash)` and sort by pattern ID. A bucket becomes a recurring pattern at support `>=2`. For each failure choose a matched success using the same binned observable signature, then minimum Euclidean distance over normalized public numeric features, then opaque case ID. If none exists, record an explicit missing-contrast flag.

`FailurePatternCard` contains the exact public fields from design §7 and names patterns as `pattern-<12-char public content hash>`; optional slow-Agent prose may summarize only these fields and cannot change the coarse stage/cause.

Separately compute `ClusterPurityReceipt` from private labels: oracle mechanism purity, best-intervention purity, and target-surface purity. A low mechanism purity event enters `instrumentation_backlog.jsonl`; it is not copied into the public card and is not automatically labeled observation failure.

- [ ] **Step 5: Add mechanical architecture checks**

`tests/architecture/test_ttha_dependency_rules.py` parses imports and filesystem calls to assert:

1. `methods/ttha/**/*.py` never imports `evaluation` or `methods.h_ref_v02`;
2. `runtime/**/*.py` never imports either method package;
3. files defining prompt/request constructors contain no `open`, `Path.read_*`, or private-root literals;
4. `evaluation/minipipe/baselines` is absent from the transitive imports of `methods/ttha`;
5. all applicability construction calls route through `validate_applicability`;
6. public schema property names are disjoint from `FORBIDDEN_PUBLIC_KEYS`.
7. the only evaluator-derived public fields match `PublicProbePanelReceipt` exactly and no `R_private` serializer is reachable;
8. neither the slow Agent nor `EditManifest` targets can name `observable_feature_v1.json`, `contracts/observables.py`, feature-extractor code, or public-tool code;
9. H_ref reproduction code is reachable only from `evaluation/benchmark_v02/_frozen_reference/`, never from TTHA, minipipe, generic runtime, or `evaluation/minipipe/baselines`.

Add a runtime contract test that serializes every object passed to a replay backend and searches recursively for private keys and candidate utility/loss fields.

- [ ] **Step 6: Run wall, pattern, and architecture tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_information_walls.py \
  SelfEvolvingHarnessTS/tests/architecture/test_ttha_dependency_rules.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t10
```

Expected: all tests pass and a deliberately injected `candidate_utilities` key is caught before backend invocation.

- [ ] **Step 7: Commit Task 10**

```bash
git add evaluation/minipipe/feedback evaluation/minipipe/schemas \
  tests/minipipe/test_information_walls.py tests/architecture/test_ttha_dependency_rules.py
git commit -m "feat: enforce minipipe information walls"
```

---

### Task 11: Validate and Apply One-Surface Harness Edits to Isolated Snapshots

**Files:**

- Create: `evaluation/minipipe/replay/__init__.py`
- Create: `evaluation/minipipe/replay/edit_controller.py`
- Modify: `methods/ttha/harness/harness_surfaces.json`
- Modify: `methods/ttha/harness/compiler.py`
- Modify: `methods/ttha/harness/store.py`
- Test: `tests/minipipe/test_edit_controller.py`

**Interfaces:**

- Consumes: untrusted EditManifest, confirmed cause code, public pattern card, active immutable snapshot, surface registry, fault router, schema/dependency SHAs.
- Produces: `SurfaceRegistry`, `EditController.validate`, `EditController.apply_to_fork`, `AppliedEditReceipt`, and isolated candidate HarnessSnapshot.

- [ ] **Step 1: Write failing atomic-ADD, stale-precondition, wrong-route, and isolation tests**

```python
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import EditController


def test_add_skill_is_one_source_surface_and_index_is_derived(controller, h0_snapshot, add_skill_manifest):
    receipt = controller.apply_to_fork(
        h0_snapshot,
        add_skill_manifest,
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    assert receipt.source_surfaces_changed == (
        "skill_library.entries/local_outlier_repair_v1",
    )
    assert receipt.derived_outputs_changed == ("retrieval_index",)
    assert receipt.parent_runtime_bundle_sha == h0_snapshot.runtime_bundle_sha
    assert receipt.candidate_runtime_bundle_sha != h0_snapshot.runtime_bundle_sha


def test_add_does_not_mutate_parent_or_checked_in_h0(controller, h0_snapshot, add_skill_manifest):
    before = controller.tree_digest(h0_snapshot.root)
    receipt = controller.apply_to_fork(h0_snapshot, add_skill_manifest,
                                       confirmed_cause="SKILL_LIBRARY_GAP")
    assert controller.tree_digest(h0_snapshot.root) == before
    assert not (H0_ROOT / "skills" / "learned" / "local_outlier_repair_v1.json").exists()
    assert receipt.candidate_root != h0_snapshot.root


def test_stale_surface_or_dependency_precondition_requires_replay(
    controller, h0_snapshot, add_skill_manifest
):
    stale = replace(add_skill_manifest,
                    dependency_precondition_shas={"operator_registry": "0" * 64})
    with pytest.raises(StaleEditError, match="operator_registry"):
        controller.apply_to_fork(h0_snapshot, stale, confirmed_cause="SKILL_LIBRARY_GAP")


def test_capability_fault_cannot_edit_bootstrap(controller, h0_snapshot, bootstrap_patch):
    with pytest.raises(EditAuthorizationError, match="SKILL_LIBRARY_GAP"):
        controller.apply_to_fork(h0_snapshot, bootstrap_patch,
                                 confirmed_cause="SKILL_LIBRARY_GAP")
```

- [ ] **Step 2: Run controller tests and verify missing module failure**

Expected: collection fails for `replay.edit_controller`.

- [ ] **Step 3: Finish the single-owner surface registry**

Make `harness_surfaces.json` a versioned list with these M0 target classes:

```text
instruction.core                              PATCH text
bootstrap_skills.entries/{skill_id}.body      PATCH text
skill_library.entries/{skill_id}              ADD structured_entry, precondition ABSENT
skill_library.entries/{skill_id}.body         PATCH text
skill_library.entries/{skill_id}.observable_applicability PATCH structured_rule
skill_library.entries/{skill_id}.risk_guards  PATCH structured_rule
retrieval.capability.top_k                     PATCH scalar
candidate_policy.agent_program_slots           PATCH scalar
candidate_policy.proposal_guidance              PATCH text
candidate_policy.selection_guidance             PATCH text
verification.rules                             PATCH structured_rule
memory.entries/{memory_id}                     ADD structured_entry, memory-entry/1
```

Each entry declares owner path/JSON pointer, target class, allowed operation, precondition kind, value schema, mutually exclusive parent/child relationship, allowed skill kind, and derived outputs. Python, compiler, operator, prompt-builder, and retrieval-index paths are declared read-only. Each writable byte region has one owner; registry load fails on overlapping path/pointer ownership.

The dynamic ADD entry is exactly:

```json
{
  "surface_template_id": "skill_library.entries/{skill_id}",
  "owner": "skills/learned",
  "path_template": "skills/learned/{skill_id}.json",
  "target_class": "capability",
  "surface_type": "structured_entry",
  "allowed_operations": ["ADD"],
  "precondition": "ABSENT",
  "value_schema": "skill-entry/1",
  "atomic": true,
  "derived_outputs": ["retrieval_index"]
}
```

- [ ] **Step 4: Implement manifest and dependency validation**

Before touching a snapshot, verify:

1. `base_harness_sha == snapshot.harness_content_sha`;
2. target matches one and only one surface template;
3. operation and precondition kind match the surface;
4. the fault router authorizes target class and skill kind for the confirmed cause;
5. all dependency precondition SHAs equal the snapshot lock;
6. SkillEntry and MemoryEntry applicability validate against the closed public vocabulary; the observable vocabulary/schema, feature extraction, and public-tool surfaces are categorically non-editable;
7. tools are canonical, non-deprecated, and task-compatible;
8. deployable `new_value`/`minimal_patch` strings contain no private field names, opaque case IDs, source pattern IDs, filesystem paths, candidate values, or code blocks; manifest provenance may retain `target_pattern_id` outside deployable content. This same forbidden-field scan applies recursively to SkillEntry and MemoryEntry bodies, applicability, and risk guards;
9. PATCH supplies `minimal_patch` only and ADD supplies `new_value` only;
10. behavior-prediction strings use the M0 predicate DSL:
    `retrieve_skill:<id>`, `supply_operator:<id>`, `supply_effect_distinct`,
    `choose_candidate_kind:identity|program`, `identity_retained`,
    `effective_view_unchanged_out_of_scope`, and `scope_modified_fraction<=<number>`.

Reject arbitrary narrative as a machine behavior predicate. Natural-language rationale may live in provenance, but it cannot sign `prediction_verified`. The controller ignores any Agent-proposed risk-case IDs and deterministically replaces `automatically_selected_risk_cases` with the Task 12 builder's signed output before replay.

- [ ] **Step 5: Implement copy-on-write application and exact diff ownership**

`SnapshotStore.fork(parent_runtime_bundle_sha, edit_id)` copies the resolved authoring tree into a new temporary sibling directory. Apply the edit only there, deterministically rebuild the retrieval index, compile a general snapshot lock, then compare source bytes against the parent.

Map every changed canonical semantic path/JSON pointer back through `SurfaceRegistry` and require exactly one semantic surface. Formatting-only differences that canonicalize identically are not edits. Derived index/lock/provenance changes are recorded separately and cannot satisfy the one-surface rule. Atomically rename the compiled candidate into `runs/minipipe/harness_snapshots/<runtime_bundle_sha>/`; leave both roots intact. A collision with different bytes fails loudly.

For ADD, validate canonical `skill_id`, path containment beneath `skills/learned`, file absence, exact schema, `skill_kind=capability`, and deterministic index order. File presence is membership; do not add or edit a separate member list.

For MemoryEntry ADD, require the strict `memory-entry/1` loader from Task 1, a canonical ID, public-only applicability, and the identical oracle/private-field prohibition used for SkillEntry. Memory is never a looser free-text escape hatch through the information wall.

- [ ] **Step 6: Run edit-controller and H0 regression tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_edit_controller.py \
  SelfEvolvingHarnessTS/tests/methods/test_ttha_h0.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t11
```

Expected: all tests pass; parent tree digest is unchanged and exactly one new learned-skill source file exists only in the candidate snapshot.

- [ ] **Step 7: Commit Task 11**

```bash
git add evaluation/minipipe/replay methods/ttha/harness tests/minipipe/test_edit_controller.py
git commit -m "feat: apply isolated one-surface harness edits"
```

---

### Task 12: Run Paired Replay and Derive Falsifiable Edit Verdicts

**Files:**

- Create: `evaluation/minipipe/replay/paired.py`
- Create: `evaluation/minipipe/replay/risk_sets.py`
- Create: `evaluation/minipipe/schemas/paired_replay_report_v1.json`
- Test: `tests/minipipe/test_paired_replay.py`

**Interfaces:**

- Consumes: parent/candidate snapshots, AppliedEditReceipt, target pattern, core cases, cached case-runner, frozen evaluator, rules.
- Produces: `ReplayEvaluationStatus`, `EditVerdict`, `ReplayFacts`, `PairedReplayReport`, `AutomaticRiskSetBuilder`, and `PairedReplayRunner.run`.

- [ ] **Step 1: Write failing verdict-table and scope tests**

```python
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.paired import EditVerdict, derive_verdict


@pytest.mark.parametrize(
    "facts, expected",
    [
        (facts(prediction=False, target=False), EditVerdict.DEAD_EDIT),
        (facts(prediction=True, behavior=True, target=False),
         EditVerdict.BEHAVIOR_CHANGED_NO_GAIN),
        (facts(prediction=True, behavior=True, target=True, risk=False),
         EditVerdict.TARGET_RECOVERED_WITH_HARM),
        (facts(prediction=True, behavior=True, target="partial", risk=True, scope=True),
         EditVerdict.PARTIAL_RECOVERY),
        (facts(prediction=True, behavior=True, target=True, risk=True, scope=True),
         EditVerdict.SUPPORTED_EDIT),
        (facts(prediction=False, behavior=True, target=True, risk=True, scope=True),
         EditVerdict.UNEXPECTED_GAIN),
        (facts(evaluation="infrastructure_failure"), EditVerdict.INCONCLUSIVE),
    ],
)
def test_verdict_truth_table(facts, expected):
    assert derive_verdict(facts) is expected


def test_out_of_scope_requires_equal_view_cache_reuse_and_behavior(out_of_scope_pair):
    pair = out_of_scope_pair(
        effective_view_equal=True,
        all_eligible_calls_reused=False,
        behavior_equal=True,
    )
    assert pair.scope_status == "FAIL"


def test_top_k_displacement_is_scope_failure_even_when_new_skill_does_not_match(
    displacement_pair,
):
    assert displacement_pair.new_skill_applicability_match is False
    assert displacement_pair.effective_view_equal is False
    assert displacement_pair.scope_status == "FAIL"
```

- [ ] **Step 2: Run paired tests and verify missing replay symbols**

Expected: imports fail for `replay.paired`.

- [ ] **Step 3: Build the automatic five-part risk set**

`AutomaticRiskSetBuilder.build(pattern, corpus, baseline_feedback)` returns stable opaque IDs from these categories, in order:

1. clean risk cases with the same binned observable signature;
2. genuine-event cases nearest in observable feature space;
3. target cases in the same observable bucket at adjacent severity;
4. cases whose baseline first-fault result is PASS and target/risk outcome is acceptable;
5. cases whose best probe direction is opposite to the proposed behavior action.

Deduplicate by case ID and retain category membership on the receipt. The slow Agent may not remove automatically selected cases. If Stage B later finds an unselected regression, append a separate `RISK_SET_MISS` instrumentation record in addition to the harmful verdict.

- [ ] **Step 4: Materialize the baseline arm first and enforce paired identities**

Define an injected `CaseRunner.run(snapshot, case, cache) -> CaseRunReceipt`. `PairedReplayRunner`:

1. runs H_t for every Stage A case first;
2. then runs H_t+edit with the same case bytes, evaluator/probe manifest, replicate ID, and effective-request cache; the relay exposes no relied-upon provider seed;
3. records content/bundle/run-context SHAs for both arms;
4. retries a typed infrastructure failure once with the identical request;
5. treats a second infrastructure failure as `INCONCLUSIVE`, keeps the edit pending, and appends infrastructure backlog;
6. treats malformed model output, invalid program, compilation failure, and execution failure as Agent behavior, not infrastructure.

For an out-of-scope case, require all of:

```text
baseline.effective_harness_view_sha == edit.effective_harness_view_sha
every eligible edit-arm Agent call has cache_receipt.hit == true
baseline.behavior_signature_sha == edit.behavior_signature_sha
```

For in-scope clean/genuine-event risks, require `delta_U >= -risk_epsilon`.

- [ ] **Step 5: Verify behavior predicates before target utility**

Evaluate the Task 11 predicate DSL from mechanical receipts. `prediction_verified` is true only when every declared predicate passes. `behavior_change_status` separately says whether normalized behavior changed at all. Then evaluate targets:

- full recovery: at least `target_recovery_fraction` of target cases improve over the baseline arm by at least `candidate_gain_min`, and median improvement is at least `target_median_gain_min`;
- partial: positive median improvement but either threshold is missed;
- no gain: median improvement `<= utility_tolerance`;
- risk/scope failure overrides full/partial recovery as harmful.

Derive the verdict only after storing these orthogonal facts:

```text
evaluation_status
prediction_verified
behavior_change_status
target_outcome_status
risk_status
scope_status
```

If outcome improves but predicted behavior did not occur, label `UNEXPECTED_GAIN`; retain evidence, do not promote, return the pattern for a rewritten explanation and fresh replay. If the suspected surface changed but did not cause predicted behavior, add `UPDATE_MISATTRIBUTION` to the attribution receipt; do not infer that code before intervention. Write `confirmed_surface=target_surface_id` only when the predicted behavior is mechanically verified. This may confirm behavioral control even for no-gain/harmful outcomes, but only `SUPPORTED_EDIT` confirms the full behavior→outcome contract and is eligible for promotion.

Load all named thresholds from the exact `decision_rule_id`/`m0_rules_sha` used during StageAssessment. Do not substitute Python defaults during replay. This keeps supply, selection, target-majority, and risk decisions on one versioned threshold family.

- [ ] **Step 6: Run Stage B and record missed or interaction regressions**

Only Stage-A eligible candidates run the complete 36-case core suite. A new Stage-B regression changes the verdict to harmful and records `RISK_SET_MISS`. Multiple individually supported edits are not combined inside this task; Task 13 sequentially promotes and reruns Stage B on the final snapshot. Composition-only regression is `EDIT_INTERACTION_REGRESSION`, never `UPDATE_MISATTRIBUTION`.

On period-change cases, if upstream assessments pass, expressibility is `PROVEN_UNAVAILABLE`, and identity is chosen, record Agent success plus system backlog without counting the case in Agent-failure recovery. Earlier observation/localization/mechanism faults remain earlier faults.

- [ ] **Step 7: Run paired replay tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/minipipe/test_paired_replay.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t12
```

Expected: fixtures cover dead, no-gain, harmful, partial, supported, unexpected-gain, and infrastructure-inconclusive verdicts; no unexpected gain is promoted.

- [ ] **Step 8: Commit Task 12**

```bash
git add evaluation/minipipe/replay evaluation/minipipe/schemas \
  tests/minipipe/test_paired_replay.py
git commit -m "feat: add falsifiable paired harness replay"
```

---

### Task 13: Compose Two M0 Cycles, Immutable Lineage, and Live/Offline CLI Backends

**Files:**

- Create: `evaluation/minipipe/replay/lineage.py`
- Create: `evaluation/minipipe/cycle.py`
- Create: `evaluation/minipipe/fixtures/__init__.py`
- Create: `evaluation/minipipe/fixtures/contract_policy.py`
- Create: `evaluation/minipipe/fixtures/build_offline_replay.py`
- Create: `evaluation/minipipe/fixtures/m0_offline_replay_v1.jsonl`
- Create: `cli/__init__.py`
- Create: `cli/minipipe.py`
- Create: `runs/.gitignore`
- Test: `tests/integration/test_minipipe_two_cycles.py`

**Interfaces:**

- Consumes: Tasks 1–12, one backend, one valuator, H0, corpus/rules, and a run root.
- Produces: `RunContext`, `HarnessLineage`, `M0CycleRunner.run_cycle`, `run_cycles`, backend factory, artifacts, and `python -m SelfEvolvingHarnessTS.cli.minipipe`.

- [ ] **Step 1: Write the failing two-cycle acceptance test**

```python
import json

from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import run_cycles


def test_two_cycles_promote_at_most_one_edit_and_reproduce_scientific_outputs(
    tmp_path, contract_replay_backend, deterministic_test_valuator
):
    first = run_cycles(
        cycles=2,
        run_root=tmp_path / "first",
        backend=contract_replay_backend.clone(),
        valuator=deterministic_test_valuator,
    )
    second = run_cycles(
        cycles=2,
        run_root=tmp_path / "second",
        backend=contract_replay_backend.clone(),
        valuator=deterministic_test_valuator,
    )
    assert len(first.cycles) == 2
    assert all(len(cycle.promoted_edit_ids) <= 1 for cycle in first.cycles)
    assert first.cycles[1].starting_snapshot_sha == first.cycles[0].ending_snapshot_sha
    assert first.normalized_behavior_shas == second.normalized_behavior_shas
    assert first.scientific_verdicts == second.scientific_verdicts
    assert first.lineage.verify_hash_chain() is True
    for event in first.lineage.promotions:
        assert event.parent_snapshot_sha
        assert event.edit_manifest_sha
        assert event.paired_replay_report_sha
        assert event.final_core_regression_sha


def test_primary_artifacts_exist_in_their_correct_visibility_roots(two_cycle_run):
    assert (two_cycle_run.private_root / "case_feedback.jsonl").is_file()
    assert (two_cycle_run.public_root / "failure_patterns.json").is_file()
    assert (two_cycle_run.public_root / "failure_patterns.md").is_file()
    assert (two_cycle_run.public_root / "edit_manifest.json").is_file()
    assert (two_cycle_run.private_root / "paired_replay_report.json").is_file()
    assert (two_cycle_run.run_root / "harness_lineage.jsonl").is_file()
    assert (two_cycle_run.private_root / "operator_capability_backlog.jsonl").is_file()
```

- [ ] **Step 2: Run the integration test and verify missing cycle module failure**

Expected: collection fails for `evaluation.minipipe.cycle`.

- [ ] **Step 3: Implement an append-only, hash-chained lineage ledger**

Each canonical JSONL row contains:

```text
schema_version
event_index
event_kind: GENESIS | EDIT_EVALUATED | PROMOTED | REJECTED | PENDING
cycle_id
parent_snapshot_sha
candidate_snapshot_sha
active_snapshot_sha
edit_manifest_sha
paired_replay_report_sha
final_core_regression_sha
verdict
scope_kind
previous_event_sha
event_sha
```

Compute `event_sha` over the row without itself, include no timestamp in scientific identity, flush and `os.fsync` each append, and reject a broken chain on resume. GENESIS binds H0 content/bundle SHA and run-context SHA. A promotion row requires parent, manifest, replay report, and final core-regression receipts.

- [ ] **Step 4: Implement exact pattern/edit prioritization and one-cycle orchestration**

Sort actionable recurring patterns by this stable tuple:

```text
(-support_count, -median_damage_d, -attribution_pass_count, pattern_id)
```

Take at most three. For each, ask the same slow Agent core for one edit, validate and fork it, build the automatic risk set, and run paired replay. Only `SUPPORTED_EDIT` reports are promotion-eligible. Keep `PARTIAL_RECOVERY` as revision evidence and require a new manifest/replay before it can be promoted. Sort supported reports by:

```text
(-target_recovery_fraction, -median_target_improvement, edit_id)
```

Promote at most one. Immediately before promotion recheck base/surface/dependency preconditions; mismatch returns the edit to replay rather than applying a stale receipt. Set the active pointer only after a complete Stage-B receipt. After any promotion, rerun the complete core suite on the final active snapshot and record composition regression separately.

`M0CycleRunner.run_cycle` executes the twelve design §12 steps in order and writes artifacts atomically. Non-actionable `OPERATOR_GAP`, expressibility unknown, critic blind, and infrastructure incidents go to separate private backlogs and never consume an edit slot.

Within each case, construct the public view and complete fixed `R_public` panel before the first fast-Agent call. Run the fast Agent next, then compute private clean-future outcome, candidate regret, `R_private`, expressibility, and CaseFeedback. A composition root must never accidentally compute a private receipt and pass the same object into the public tool gateway.

- [ ] **Step 5: Compute snapshot, runtime, and run-context identities**

`RunContext` hashes:

```text
runtime_bundle_sha
backend kind (agicto-chat-completions or offline-replay)
relay base URL and API style
requested model alias (default gpt-5.5)
OpenAI Python SDK version and provider-capability flags
reported response model/fingerprint set when available
Python/NumPy/Torch/Transformers/Chronos versions
valuator manifest SHA
probe specs/rules/corpus SHAs
platform/device determinism flags
```

`gpt-5.5` is deliberately recorded as a relay alias, never upgraded in the report to a dated snapshot claim. If `M0_AGENT_MODEL` or `M0_AGENT_BASE_URL` overrides a default, the run context necessarily changes. Offline fixture runs have backend identity `offline-contract-replay/1` and never claim their responses came from the relay or OpenAI.

- [ ] **Step 6: Build an immutable offline response tape without oracle leakage**

`contract_policy.py` is an evaluation-only fixture author, not a production Agent and not scientific evidence. It receives the already-resolved **public AgentRequest only** and emits deterministic contract-valid responses:

- inspect: returns the estimated public region and no self-assigned PASS/FAIL;
- propose: without a matching capability skill, returns no PROGRAM; with a matching learned skill, emits one allowed local PROGRAM;
- select: chooses the sole effect-distinct PROGRAM, otherwise identity;
- edit: for a public `SKILL_LIBRARY_GAP` card, emits one capability SkillEntry whose applicability uses the observed missing/robust-z/level signature and whose allowed tool is respectively `impute_linear`, `hampel_filter`, or `repair_level_shift`;
- period/operator gaps emit no edit.

The fixture builder runs the two-cycle composition once with this policy while recording `(CacheKey, AgentResponse)` rows, then reruns from the written JSONL using only `ReplayAgentBackend` and asserts zero author-policy calls. Check in the resulting canonical tape with its own SHA and `fixture_source="contract_policy_not_openai"`. A changed prompt/H0/schema produces a replay miss and forces intentional tape regeneration.

- [ ] **Step 7: Implement relay Chat Completions backend selection with GPT-5.5 as the live default**

Expose:

```bash
python -m SelfEvolvingHarnessTS.cli.minipipe run \
  --cycles 2 \
  --run-dir runs/minipipe/reference
```

The default `--backend agicto` builds `AgictoChatCompletionsBackend`. It reads the key only from `AGICTO_API_KEY`, uses `M0_AGENT_BASE_URL` defaulting to `https://api.agicto.cn/v1`, and uses `M0_AGENT_MODEL` defaulting to `gpt-5.5`. A missing key fails before corpus execution with a concise configuration error that names only the environment variable. `--backend replay --replay-file <path>` is explicit, makes no network calls, and is used by CI/default smoke. `--model` and `--base-url` may override the environment but are always recorded. Do not add a silent fallback from the relay to fixture replay, and never accept an API key as a command-line flag.

Other required flags are `--h0-root`, `--rules`, `--valuator-manifest`, `--resume`, and `--overwrite-empty-run-dir`. Resume verifies lineage and all snapshot locks. Never overwrite a non-empty run directory unless it is the same resumable run.

- [ ] **Step 8: Generate the tape and run two-cycle integration twice**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m \
  SelfEvolvingHarnessTS.evaluation.minipipe.fixtures.build_offline_replay \
  --out SelfEvolvingHarnessTS/evaluation/minipipe/fixtures/m0_offline_replay_v1.jsonl

/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/integration/test_minipipe_two_cycles.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t13
```

Expected: fixture generation proves a pure replay rerun; integration passes twice with equal behavior signatures/verdicts and no network calls.

- [ ] **Step 9: Run the offline CLI smoke**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m \
  SelfEvolvingHarnessTS.cli.minipipe run \
  --cycles 2 \
  --backend replay \
  --replay-file SelfEvolvingHarnessTS/evaluation/minipipe/fixtures/m0_offline_replay_v1.jsonl \
  --run-dir SelfEvolvingHarnessTS/_m0_cli_smoke
```

Expected: exit 0, two cycle summaries, active snapshot SHA, and all primary artifacts. Remove only this explicitly named disposable smoke directory after recording the result.

- [ ] **Step 10: Commit Task 13**

```bash
git add evaluation/minipipe cli runs/.gitignore tests/integration/test_minipipe_two_cycles.py
git commit -m "feat: compose reproducible two-cycle minipipe"
```

---

### Task 14: Retire H_ref from the Active Method/Runtime and Run Final Acceptance

**Files:**

- Create: `evaluation/benchmark_v02/_frozen_reference/__init__.py`
- Move: `methods/h_ref_v02/config.py` → `evaluation/benchmark_v02/_frozen_reference/config.py`
- Move: `runtime/fast_path.py` → `evaluation/benchmark_v02/_frozen_reference/fast_path.py`
- Modify: `evaluation/benchmark_v02/method_compat.py`
- Modify: `evaluation/benchmark_v02/baselines.py`
- Modify: `evaluation/benchmark_v02/dev_eval.py`
- Modify: `evaluation/benchmark_v02/__init__.py`
- Modify: `methods/__init__.py`
- Modify: `runtime/__init__.py`
- Delete: `methods/h_ref_v02/__init__.py`
- Delete: `methods/h_ref_v02/method.py`
- Delete: `tests/integration/test_h_ref_method.py`
- Delete: `tests/runtime/test_fast_path_equivalence.py`
- Create: `tests/integration/test_ttha_method.py`
- Create: `tests/frozen_protocol/test_legacy_reference_equivalence.py`
- Create: `tests/architecture/test_no_active_h_ref.py`
- Modify: `README.md`

**Interfaces:**

- Consumes: completed TTHA method and existing byte-frozen benchmark evidence.
- Produces: active method surface containing TTHA only, generic runtime without H_ref imports, and an evaluation-private frozen benchmark compatibility fossil that cannot influence TTHA/minipipe.

- [ ] **Step 1: Verify benchmark relocation attributes and frozen bytes before any move**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence/SelfEvolvingHarnessTS
git check-attr text -- \
  evaluation/benchmark_v02/data/acquisition_manifest.json \
  evaluation/benchmark_v02/data/legacy/monash_clean.npz \
  artifacts/frozen/benchmark_v02/*

cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_benchmark_v02_smoke.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t14_prefreeze
```

Expected: every listed binary/frozen path reports `text: unset`, and frozen manifest digest tests pass byte-for-byte. If either check fails, stop this task and repair the relocation guard before moving H_ref code.

- [ ] **Step 2: Write failing active-tree and TTHA adapter tests**

```python
from pathlib import Path

from SelfEvolvingHarnessTS.evaluation.benchmark_v02.method_compat import BenchmarkMethodAdapter
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod


ROOT = Path(__file__).resolve().parents[2]


def test_no_active_h_ref_method_or_runtime_remains():
    assert not (ROOT / "methods" / "h_ref_v02").exists()
    assert not (ROOT / "runtime" / "fast_path.py").exists()
    assert not hasattr(importlib.import_module("SelfEvolvingHarnessTS.methods"), "HRefV02Method")


def test_benchmark_adapter_accepts_ttha(ttha_method, method_series_view, task_spec):
    prepared = BenchmarkMethodAdapter(ttha_method).prepare(method_series_view, task_spec, {})
    assert prepared.series_uid == method_series_view.series_uid
```

Run the new tests before the move. Expected: the no-active-H_ref assertions fail.

- [ ] **Step 3: Move, do not copy, the frozen benchmark reference implementation**

The old selector remains necessary only to reproduce benchmark-v0.2's already-frozen `h_ref` incumbent rows. Move its configuration and fast-path implementation beneath `evaluation/benchmark_v02/_frozen_reference/`, update their relative imports, and rename public Python symbols exposed by the benchmark wrapper to `LegacyReference*` / `run_legacy_reference_batch`. Preserve the frozen report field/program ID string `h_ref` because changing an existing frozen protocol is not an Agent improvement.

This compatibility fossil has strict boundaries:

- it is importable only from `evaluation/benchmark_v02`;
- it is absent from `methods`, `runtime`, active registry, TTHA, minipipe, H0, prompts, candidate pools, and skill creation;
- it cannot be selected as an M0 method or baseline;
- the minipipe's positive-witness catalog is independently authored from canonical operator contracts and contains no import or provenance dependency on this fossil;
- architecture tests fail if any non-benchmark module imports `_frozen_reference`.

Delete the active method adapter rather than wrapping or aliasing it. This satisfies the user's requirement that H_ref provide no capability to the Agent while preserving old benchmark byte/numeric reproducibility.

- [ ] **Step 4: Rewire only benchmark-v0.2 internal compatibility imports**

`method_compat.py` keeps `BenchmarkMethodAdapter` generic and makes the legacy batch helper private to the benchmark. `baselines.py` renames `HRefBaseline` to `LegacyReferenceBaseline` without exporting it from the package root. `dev_eval.py` imports the local legacy materializer; its frozen schema strings remain unchanged. `evaluation/benchmark_v02/__init__.py` exports `BenchmarkMethodAdapter` only, not the legacy runner.

`methods/__init__.py` exports TTHA composition types. `runtime/__init__.py` exports only generic executor/candidate/trace/backend/cache contracts.

- [ ] **Step 5: Replace active H_ref tests with TTHA and private-fossil guards**

Delete tests that treat H_ref as a canonical method. The new TTHA integration test uses a replay backend and proves the canonical benchmark adapter maps identity to unchanged output and PROGRAM choice to canonical operators.

Move the old known fast-path fingerprint assertions into `tests/frozen_protocol/test_legacy_reference_equivalence.py`, importing only the benchmark-private fossil, so the code move cannot silently change frozen v0.2 numbers. `test_no_active_h_ref.py` parses the import graph and allows `_frozen_reference` imports only from its own package and benchmark-v0.2 modules.

- [ ] **Step 6: Update the README composition diagram and commands**

Document TTHA as the sole active method, H0 as domain-naive, the relay alias `gpt-5.5` as the live Agent default, `AGICTO_API_KEY` as the secret input, and the two-cycle offline/live commands. State explicitly that the alias is not a dated snapshot claim. Label `_frozen_reference` clearly as a benchmark-v0.2 reproduction fossil, not an Agent Harness input. Do not present H_ref as an available method or compatibility option.

- [ ] **Step 7: Run targeted retirement and frozen-protocol tests**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/architecture/test_no_active_h_ref.py \
  SelfEvolvingHarnessTS/tests/architecture/test_ttha_dependency_rules.py \
  SelfEvolvingHarnessTS/tests/integration/test_ttha_method.py \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_legacy_reference_equivalence.py \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_benchmark_v02_smoke.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_t14_retirement
```

Expected: all pass; frozen benchmark evidence digests are unchanged and no active import reaches H_ref.

- [ ] **Step 8: Run the complete M0 acceptance suite**

Run:

```bash
cd /mnt/c/Users/辉/Desktop/Agent/.worktrees/architecture-convergence
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/contracts \
  SelfEvolvingHarnessTS/tests/runtime \
  SelfEvolvingHarnessTS/tests/methods \
  SelfEvolvingHarnessTS/tests/minipipe \
  SelfEvolvingHarnessTS/tests/integration/test_ttha_method.py \
  SelfEvolvingHarnessTS/tests/integration/test_minipipe_two_cycles.py \
  SelfEvolvingHarnessTS/tests/architecture \
  SelfEvolvingHarnessTS/tests/frozen_protocol -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_m0_final
```

Expected: all tests pass, no network is accessed, and the pinned local Chronos smoke runs. Then run the offline two-cycle CLI a second time into a fresh directory and compare normalized behavior/verdict receipts with the Task 13 run.

- [ ] **Step 9: Map and inspect all twenty acceptance criteria**

Use the coverage table below. For each row, record the exact passing test node ID in `runs/minipipe/reference/private/acceptance_receipt.json`; reject completion if any row has no test or receipt. Also run:

```bash
rg -n "methods\.h_ref_v02|runtime\.fast_path" \
  contracts operators runtime methods evaluation/minipipe cli tests/architecture tests/integration
```

Expected: no matches. Matches inside `evaluation/benchmark_v02/_frozen_reference`, frozen artifacts, or this plan are outside the active scan and are permitted only by the private-fossil test.

- [ ] **Step 10: Commit Task 14**

```bash
git add -A methods runtime evaluation/benchmark_v02 tests README.md
git commit -m "refactor: retire H_ref from active method runtime"
```

---

## Acceptance-Criteria Coverage

| Design criterion | Owning task | Required test/receipt |
| --- | --- | --- |
| 1. TTHA active, no active H_ref | 14 | `test_no_active_h_ref_method_or_runtime_remains` |
| 2. No TTHA/generic-runtime H_ref import | 10, 14 | `test_ttha_dependency_rules.py`, `test_no_active_h_ref.py` import scans |
| 3. One Agent core serves both roles | 5 | `test_fast_and_slow_paths_share_the_same_agent_core` |
| 4. Stable, canonicalized, domain-naive H0 | 1, 2 | canonical LF/CRLF/JSON-order fixtures and `test_h0_is_stable_procedural_and_domain_naive` |
| 5. Stale lock/dependency fails | 2, 11 | H0 lock mismatch and stale edit tests |
| 6. Oracle/judge walls | 7, 8, 10 | sanitizer, public-reader, prompt-payload leak tests; only fixed `R_public` crosses |
| 7. Identity and explicit choice | 1, 3, 5 | Candidate, pool, and identity-abstention tests |
| 8. Effect-equivalence contract | 3 | `test_effect_equivalence_uses_shape_dtype_and_bytes` plus tolerance fixture |
| 9. Fault causes distinguished with seeded-skill fixtures | 9 | first-fault parameterized fixtures including both observable-gap branches |
| 10. Oracle-only cannot sign library edit | 8, 9 | witness proof-asymmetry tests |
| 11. Fault router authorizes kind/surface | 9, 11 | router matrix and bootstrap rejection tests |
| 12. Atomic ADD; derived index | 11 | `test_add_skill_is_one_source_surface_and_index_is_derived` |
| 13. Single owner and stale replay | 11 | overlap, diff ownership, and precondition fixtures |
| 14. Monotonic probes; dual receipts; diagnostic period | 7, 8 | fixed twelve-point `R_public`, private-only `R_private`, probe-spec, and period tests |
| 15. Correct period abstention, separate backlog | 9, 12 | period ledger fixture |
| 16. Relay-safe effective-request cache identity | 4, 5 | exact Chat messages, alias/origin/SDK identity, cache provenance/effective-view tests |
| 17. Out-of-scope view/cache/behavior equality | 12 | out-of-scope and top-k displacement tests |
| 18. Verdict coverage | 12 | `test_verdict_truth_table` and infrastructure retry fixture |
| 19. Promotion lineage completeness | 13 | two-cycle lineage assertions |
| 20. Fixed-seed reproducibility | 6, 7, 13 | corpus SHAs, Chronos forecast SHA, repeated cycle comparison |

## Final Plan Self-Review

Before starting implementation, the executing agent must run these plan-only checks from the package root:

```bash
PLAN_FORBIDDEN='\b(T''BD|TO''DO|FIX''ME)\b|implement la''ter|similar t''o|appropriate error hand''ling|write tests for the ab''ove'
rg -n "$PLAN_FORBIDDEN" \
  docs/superpowers/plans/2026-07-18-agent-centric-minipipe-m0.md

git diff --check -- docs/superpowers/plans/2026-07-18-agent-centric-minipipe-m0.md
```

Expected: both commands produce no findings. Then manually verify these type/ownership chains before Task 1 begins:

```text
HarnessSnapshot (semantic content, no path)
  → MaterializedSnapshot (snapshot + immutable root)
  → AppliedEditReceipt (parent/candidate materialized identities)
  → PairedReplayReport (both run contexts and orthogonal facts)
  → LineageEvent (manifest/report/final-regression hashes)

PrivateSyntheticCase
  → structural PublicCaseView
  → public feature extraction and rehashed PublicCaseView
  → always-computed fixed rolling-origin R_public panel
  → TTHAAgentCore request
  → runtime DecisionTrace
  → private outcome, R_private, and CaseFeedback
  → allowlist sanitizer
  → public FailurePatternCard
  → untrusted EditManifest
  → controller validation and isolated replay
```

The implementation is not complete merely because the CLI exits zero. Completion requires all twenty coverage rows, both fixed-seed runs, the frozen benchmark digest check, an offline no-network run, and evidence that the live backend sends model alias exactly `gpt-5.5` to the configured Chat Completions relay while making no dated-snapshot reproducibility claim.
