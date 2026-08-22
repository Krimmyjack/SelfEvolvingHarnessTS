"""T3 (#39): the task-conditioned proposal exam -- does task_spec alone split the proposals?

Gate: C13 (training-side task flip) confirmed in T1b v3.  This book proves the
**proposal layer** only: with an empty store, zero outcome leakage and no
task-to-action mapping anywhere in the input, can the Agent -- reading nothing
but the injected task_spec -- propose different, direction-appropriate
programs for the same data Pattern under two tasks?  Execution, adoption and
feedback are out of scope (T4/T5).

``evidence_grade = POSITIVE_CONTROL``, permanently: the substrate is the T1
injected training block, the answer keys are derived inside this runner from
the frozen T1 / T1b v3 artifacts, and no claim generalizes beyond the
injected-flip instrument.

Design (book #39 v2, pre-distribution revision merged):

* substrate: the T1 injected copy's training block [120, 900), twelve train
  series, presented through the *current ssi context construction*
  (``run_e2_skill_store_integration._base_input`` + the same observation
  extractor).  Two arms share every byte except ``task_spec``.
* store state: a freshly materialized empty store snapshot -- 0 Guidance /
  0 Experience / 0 learned Skill (h0's three always-on bootstrap procedures
  are standing operating procedure, not learned content; they are identical
  in both arms).  All six draws read the same snapshot.
* arms: F carries the T2-frozen forecasting task_spec verbatim; AD carries
  the frozen anomaly_detection task_spec (consumer_id ad_ridge_train_v3).
  One neutral note, identical in both arms, states that task_spec is the
  only authoritative task description; no task-to-action mapping anywhere.
* menu: {identity, outlier_iqr, outlier_mad, hampel_filter, winsorize} plus
  abstain, pinned by the book.  Each draw returns a shortlist with an
  explicit top-1, or abstains.
* sampling: 3 draws per task, order F, AD, F, AD, F, AD; backend pinned to
  gpt-5.6-sol; at most one validation retry per draw (6 + 6 = 12 calls cap).
  OFF_MENU output is archived verbatim, the draw is invalid, and it is *not*
  re-thrown; invalid draws leave the distance matrix and break the arm's 3/3.
* answer keys (derived in-runner, never hand-copied): aggregate layer --
  F = programs with T1 forecasting delayed aggregate gain >= +0.005,
  AD = {identity} plus programs with T1b v3 ad_train_gain >= -0.005;
  Risk layer -- inside the aggregate key, every per-series reading >= -0.005
  (identity is 0 by definition; abstain counts as appropriate at this layer).

Verdict ladder (pre-registered): invalid draws > 2/6 ->
EXAM_PROTOCOL_UNREADABLE; complete separation (min cross-task Jaccard
distance over the 9 cross pairs > max same-task distance over the 6 pairs)
plus 3/3 aggregate direction on both arms -> TASK_CONDITIONED_PROPOSALS_CONFIRMED;
plus 3/3 Risk on both arms -> TASK_CONDITIONED_RISK_AWARE_PROPOSALS_CONFIRMED;
separation without direction -> TASK_SEPARATED_DIRECTION_INAPPROPRIATE;
no separation -> SAMPLING_VARIANCE_DOMINATES (credible negative, full
distance distribution attached).

Budgets: LLM <= 12 calls, forecasting retrains 0, AD evaluations 0 (answer
keys are read off frozen artifacts; nothing is measured).  The T1 injected
copies, both ledgers and both answer-key artifacts are read-only here;
sha256 before/after is recorded.  No commit (Part 0 excepted), no spawned
agents, no file outside the two deliverables and _scratch/skill_store/
t3_task_exam_v1/.

Run:

    python evaluation/functional/run_e2_t3_task_conditioned_exam.py
    python evaluation/functional/run_e2_t3_task_conditioned_exam.py --smoke-only
    python evaluation/functional/run_e2_t3_task_conditioned_exam.py --rehearse-write

Writes ``artifacts/functional/e2/t3_task_exam_v1.json`` and ``.md``.
``--rehearse-write`` is a 0-LLM fake-backend write-path rehearsal; it
never touches the two deliverables.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_t1_flip_control as t1  # noqa: E402  -- frozen T1 book (paths)
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402
from run_e2_operational_pipeline import (  # noqa: E402
    FROZEN_SURFACE_V9,
    _freeze,
    _verify,
)

from evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    _default_backend_factory,
)
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
)
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    AgentRole,
    PublicAgentInput,
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentResponse,
    AgentTransportError,
    BudgetedAgentBackend,
)

PROTOCOL_VERSION = "t3_task_exam_v1"
EVIDENCE_GRADE = "POSITIVE_CONTROL"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t3_task_exam_v1.json"
OUT_MD = E2 / "t3_task_exam_v1.md"
T1_ARTIFACT = E2 / "t1_flip_control_v1.json"
T1B_V3_ARTIFACT = E2 / "t1b_training_flip_v3.json"
T1_DIR = t1.T1_DIR  # _scratch/phase_t/injected/t1, read-only this round
BLOCK = t1.BLOCK  # [120, 900)
H0_ROOT = ssi.H0_ROOT
STORE_ROOT = PROJECT_ROOT / "_scratch" / "skill_store" / "t3_task_exam_v1"

PART0_CHECKPOINT = {
    "commit": "bd5922d",
    "files": 6,
    "note": (
        "T1b v3 deliverables (consumer + runner --v3 entry + v3 artifacts) "
        "+ main-line ledger/roadmap revisions closing the T1b volume and "
        "unlocking T3"
    ),
}

# ------------------------------------------------------------ exam constants
EXAM_MODEL = "gpt-5.6-sol"  # book-pinned backend
LLM_BUDGET = 12  # 6 decisive draws + at most one validation retry each
DRAWS_PER_TASK = 3
VALIDATION_RETRIES = 1  # per draw, parse-level failures only
STAGE = "t3_task_conditioned_proposal"
SCHEMA_NAME = "t3_task_conditioned_proposal_v1"

# The book-pinned menu.  ssi's own menu is the 7-treatment recipe menu; the
# exam narrows it to the five entries the frozen answer keys cover, identity
# included.  abstain is a decision, not a menu entry.
MENU: tuple[str, ...] = (
    "identity",
    "outlier_iqr",
    "outlier_mad",
    "hampel_filter",
    "winsorize",
)
ABSTAIN_TOKEN = "__ABSTAIN__"
HARM_LINE = -0.005
FORECAST_AGGREGATE_LINE = 0.005

# The frozen AD task_spec (#39 pre-distribution revision, point 3, verbatim).
AD_TASK_SPEC: dict[str, str] = {
    "task_id": "anomaly_detection",
    "consumer_id": "ad_ridge_train_v3",
    "quality_semantics": (
        "good preparation preserves the event signal needed to train the AD "
        "Consumer and improves event F1 on an untouched query"
    ),
}
# The F task_spec is not restated by hand: it is whatever the T2-wired ssi
# default path emits for consumer_variant="pooled", asserted verbatim below.
EXPECTED_F_TASK_SPEC: dict[str, str] = {
    "task_id": "forecasting",
    "consumer_id": "pooled_ridge_a1",
    "quality_semantics": (
        "good preparation lowers the sMASE of the evaluation-series forecasts"
    ),
}

# One neutral note, identical bytes in both arms (revision point 5).  It names
# no program and maps no task to any action.
TASK_DESCRIPTION_AUTHORITY_NOTE = (
    "task_spec is the only authoritative description of the task and the "
    "Consumer for this call; every other field describes the data substrate "
    "and the pipeline wiring, which are shared across tasks."
)

EXAM_STAGE_NOTE = (
    "This is a proposal-only exercise. No Support evaluation is run here and "
    "no delayed window exists: you are naming what you would propose for "
    "this batch, and nothing is measured. `decision` is \"propose\" or "
    "\"abstain\". When you propose, `shortlist` lists the menu entries you "
    "would try, in the order you would try them, and `top1` names the single "
    "entry you would try first; `top1` must be a member of `shortlist`. "
    "`identity` is always available and means leaving the batch as it is. "
    "When the public evidence does not justify any proposal for this task, "
    "decide \"abstain\", leave `shortlist` empty and `top1` as \"\". "
    "`reason` is one or two sentences in public terms."
)

# The schema stays permissive on program names on purpose: an off-menu name
# must *parse* so the runner can classify it OFF_MENU and invalidate the draw
# without a re-throw (book rule).  Only envelope/schema failures retry (<=1).
EXAM_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "t3-task-conditioned-proposal/1",
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "shortlist", "top1", "reason"],
    "properties": {
        "decision": {"enum": ["propose", "abstain"]},
        "shortlist": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
            "maxItems": len(MENU),
        },
        "top1": {"type": "string"},
        "reason": {"type": "string"},
    },
}

# Expected keys, frozen in the main-line ledger (revision point 6).  These are
# expectations, never the scoring source: the keys below are derived from the
# frozen artifacts and any mismatch is reported as an ambiguity, not fixed.
EXPECTED_KEYS: dict[str, Any] = {
    "aggregate": {
        "forecasting": ["outlier_iqr", "outlier_mad", "hampel_filter", "winsorize"],
        "anomaly_detection": ["identity", "outlier_iqr", "outlier_mad", "hampel_filter"],
    },
    "risk": {
        "forecasting": ["outlier_iqr", "outlier_mad", "winsorize"],
        "anomaly_detection": ["identity"],
    },
    "risk_layer_also_credits": ["abstain"],
}

ARM_ORDER: tuple[str, ...] = (
    ("forecasting", "anomaly_detection") * DRAWS_PER_TASK
)

# The stand-in target/window describe the substrate, not a recipe-line cell.
# consumer_variant="pooled" is wiring: it selects the T2 default task_spec for
# the F arm and sits byte-identical in both arms, exactly the legacy field the
# smoke gate watches.
# Neutral names only: the prompt must not carry the substrate's
# positive-control identity, so nothing here says "injected".
EXAM_TARGET = {
    "target_id": "t3_exam_block",
    "cohort": "noaa_train_block",
    "consumer_variant": "pooled",
}
EXAM_WINDOW = {"window_id": "block_120_900"}


# ----------------------------------------------------------------- utilities
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    return wvc._plain(value)


def _json_text(doc: Mapping[str, Any]) -> str:
    """Serialize after a recursive de-freeze.  A shallow dict() leaves
    nested mappingproxy values (AgentResponse freezes provider_metadata)
    and that is what killed the first live write."""
    return json.dumps(_plain(doc), indent=2, ensure_ascii=False) + "\n"


# First live attempt: six draws spent, artifact not written.  Recovered
# from bash-joon3149/output.log (the command piped through tail -20).
FIRST_ATTEMPT = {
    "status": "SPENT_WRITE_FAILED",
    "task_id": "bash-joon3149",
    "when": "2026-08-22",
    "command": (
        "python.exe evaluation\\\\functional\\\\run_e2_t3_task_conditioned_exam.py "
        "2>&1 | tail -20"
    ),
    "failure": (
        "TypeError: Object of type mappingproxy is not JSON serializable "
        "at json.dumps(doc) after all six draws"
    ),
    "recovered_tail": [
        {
            "draw_index": 5,
            "arm": "forecasting",
            "classification": "VALID_PROPOSE",
            "top1": "outlier_mad",
            "validation_retry_count": 0,
        },
        {
            "draw_index": 6,
            "arm": "anomaly_detection",
            "classification": "VALID_PROPOSE",
            "top1": "identity",
            "validation_retry_count": 0,
        },
    ],
    "lost": "draws 1-4 (stdout truncated by tail -20; no backend cache)",
    "consequence": (
        "this delivered exam is a labeled second sample; the first six "
        "draws cannot be reconstructed and are not scored"
    ),
}


def _git(args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo_rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


# ------------------------------------------------------- the read-only inputs
def _read_only_inventory() -> dict[str, str]:
    """sha256 of everything this slice may read but never writes."""
    paths = [T1_ARTIFACT, T1B_V3_ARTIFACT, T1_DIR / "ledger.json", T1_DIR / "protocol.json"]
    paths.extend(sorted(T1_DIR.glob("*.npy")))
    return {_repo_rel(path): _sha256(path) for path in paths}


def _load_substrate() -> dict[str, Any]:
    """The T1 injected copy's training block, twelve series, read-only."""
    t1_doc = json.loads(T1_ARTIFACT.read_text(encoding="utf-8"))
    train_uids = [str(uid) for uid in t1_doc["roster"]["train"]]
    values: dict[str, np.ndarray] = {}
    for uid in train_uids:
        series = np.load(T1_DIR / ("%s.npy" % uid))
        values[uid] = np.asarray(series[BLOCK[0]:BLOCK[1]], dtype=np.float64)
    return {
        "train_uids": train_uids,
        "values": values,
        "block": [BLOCK[0], BLOCK[1]],
        "series_length": int(len(series)),
        "source": _repo_rel(T1_DIR),
    }


class _StandInSearch:
    """The ssi context construction's data interface over the exam substrate.

    ``support=[block length]`` makes the public-prefix cutoff the whole block:
    the proposal-only exam has no Support/delayed split, and the Agent observes
    features of the full block it would prepare.  ``values`` are the injected
    block bytes, never the pristine series.
    """

    def __init__(self, substrate: Mapping[str, Any]) -> None:
        block_len = int(substrate["block"][1]) - int(substrate["block"][0])
        self.support = [block_len]
        self.delayed: list[int] = []
        self.train_uids = list(substrate["train_uids"])
        self.eval_uids: list[str] = []
        self.values = dict(substrate["values"])
        self.exposure = "development_substrate__outcomes_not_read"


# ------------------------------------------------------------- the empty store
def _build_empty_store() -> dict[str, Any]:
    """Freshly materialize h0 with nothing learned: 0/0/0 store state."""
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    root = STORE_ROOT / "snapshots"
    store = SnapshotStore(root)
    snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
    materialized = store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    return {
        "store_root": _repo_rel(root),
        "h0_source": _repo_rel(H0_ROOT),
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "harness_content_sha": snapshot.harness_content_sha,
        "skill_ids": [skill.skill_id for skill in snapshot.skills],
        "materialized_root": _repo_rel(materialized.root),
        "_snapshot": snapshot,
    }


# -------------------------------------------------------------- the two arms
def _public_inputs(
    search: _StandInSearch,
) -> dict[str, Any]:
    """One body per arm through the live ssi construction, task_spec apart."""
    observation = wvc._observation_table(search)
    context = ssi._public_features(search)
    inputs: dict[str, Any] = {"_context": context}
    for arm, override in (
        ("forecasting", None),
        ("anomaly_detection", AD_TASK_SPEC),
    ):
        base = ssi._base_input(
            target=EXAM_TARGET,
            window=EXAM_WINDOW,
            search=search,
            observation=observation,
            task_spec_override=override,
        )
        # Documented structural deviations from the ssi body, both forced by
        # the book: the menu is the pinned five-entry exam menu (the answer
        # keys cover nothing else), and the evaluation_budget block is dropped
        # because this exercise runs no Support evaluation -- presenting its
        # economics would describe an instrument that never runs.
        base["program_menu"] = list(MENU)
        removed = base.pop("evaluation_budget")
        base["task_description_authority"] = TASK_DESCRIPTION_AUTHORITY_NOTE
        base["stage_note"] = EXAM_STAGE_NOTE
        inputs[arm] = base
        inputs["_removed_evaluation_budget_block"] = removed
    return inputs


def _render_prompts(
    view: Any, inputs: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """The exact prompt bytes per arm, rendered through the core's own path.

    case_id is call wiring and is never rendered, so one rendering per arm is
    byte-identical to what every draw of that arm sends.
    """
    core = TTHAAgentCore.__new__(TTHAAgentCore)  # rendering only; no backend
    prompts: dict[str, dict[str, str]] = {}
    for arm in ("forecasting", "anomaly_detection"):
        public_agent_input = PublicAgentInput.create(
            "T3EXAM_SMOKE", inputs[arm]
        )
        messages = TTHAAgentCore._messages(
            core,
            role=AgentRole.FAST,
            stage=STAGE,
            public_input=public_agent_input.public_data,
            harness_view=view,
            output_schema_name=SCHEMA_NAME,
            output_schema=EXAM_SCHEMA,
            tool_schemas=(),
        )
        prompts[arm] = {
            "system": str(messages[0]["content"]),
            "user": str(messages[1]["content"]),
        }
    return prompts


# ----------------------------------------------------------------- answer keys
def _derive_answer_keys() -> dict[str, Any]:
    """Derive both layers from the frozen artifacts.  Never hand-copied."""
    t1_doc = json.loads(T1_ARTIFACT.read_text(encoding="utf-8"))
    v3_doc = json.loads(T1B_V3_ARTIFACT.read_text(encoding="utf-8"))

    f_aggregate_rows = {
        str(row["program"]): float(row["forecasting_delayed_aggregate_gain"])
        for row in t1_doc["part_c"]["c1_flip_check_per_program"]
    }
    f_per_series = {
        program: {
            str(uid): float(gain)
            for uid, gain in t1_doc["part_b_arms"]["forecasting"][program][
                "delayed_gain"
            ]["per_eval_series_gain"].items()
        }
        for program in MENU
        if program != "identity"
    }
    ad_aggregate_rows = {
        str(row["program"]): float(row["ad_train_gain_pooled"])
        for row in v3_doc["part_d"]["flip_check_per_program"]
    }
    ad_per_series = {
        program: {
            str(uid): float(gain)
            for uid, gain in v3_doc["part_d"]["ad_gain_per_series"][program].items()
        }
        for program in MENU
        if program != "identity"
    }

    cross_checks: dict[str, Any] = {}
    # The v3 artifact carries the T1 forecasting readings; both copies of the
    # aggregate and of the per-series vectors must agree exactly.
    v3_f_aggregate = {
        str(row["program"]): float(row["forecasting_delayed_aggregate_gain"])
        for row in v3_doc["part_d"]["flip_check_per_program"]
    }
    cross_checks["forecasting_aggregate_t1_vs_v3"] = f_aggregate_rows == v3_f_aggregate
    carried = v3_doc["part_d"]["forecasting_delayed_per_series"]
    cross_checks["forecasting_per_series_t1_vs_v3"] = all(
        f_per_series[p] == {str(k): float(v) for k, v in carried[p].items()}
        for p in f_per_series
    )

    f_agg_key = sorted(
        p for p, gain in f_aggregate_rows.items() if gain >= FORECAST_AGGREGATE_LINE
    )
    ad_agg_key = ["identity"] + sorted(
        p for p, gain in ad_aggregate_rows.items() if gain >= HARM_LINE
    )
    f_risk_key = sorted(
        p
        for p in f_agg_key
        if all(gain >= HARM_LINE for gain in f_per_series[p].values())
    )
    ad_risk_key = ["identity"] + sorted(
        p
        for p in ad_agg_key
        if p != "identity"
        and all(gain >= HARM_LINE for gain in ad_per_series[p].values())
    )

    derived = {
        "aggregate": {"forecasting": f_agg_key, "anomaly_detection": ad_agg_key},
        "risk": {"forecasting": f_risk_key, "anomaly_detection": ad_risk_key},
    }
    mismatches = []
    for layer in ("aggregate", "risk"):
        for arm in ("forecasting", "anomaly_detection"):
            if sorted(derived[layer][arm]) != sorted(EXPECTED_KEYS[layer][arm]):
                mismatches.append({
                    "layer": layer,
                    "arm": arm,
                    "derived": derived[layer][arm],
                    "expected": EXPECTED_KEYS[layer][arm],
                })
    return {
        "derivation": {
            "rule": (
                "aggregate layer: F = programs with T1 forecasting delayed "
                "aggregate gain >= +0.005 (t1 part_c.c1_flip_check_per_program); "
                "AD = {identity} plus programs with T1b v3 ad_train_gain >= "
                "-0.005 (v3 part_d.flip_check_per_program).  Risk layer: inside "
                "the aggregate key, every per-series reading >= -0.005 (t1 "
                "part_b_arms.forecasting[*].delayed_gain.per_eval_series_gain; "
                "v3 part_d.ad_gain_per_series).  identity gain is 0 by "
                "definition; abstain is credited at the Risk layer only."
            ),
            "sources": [_repo_rel(T1_ARTIFACT), _repo_rel(T1B_V3_ARTIFACT)],
            "cross_checks": cross_checks,
        },
        "readings": {
            "forecasting_delayed_aggregate_gain": f_aggregate_rows,
            "forecasting_delayed_per_series": f_per_series,
            "ad_train_gain_aggregate": ad_aggregate_rows,
            "ad_gain_per_series": ad_per_series,
        },
        "derived": derived,
        "expected": EXPECTED_KEYS,
        "matches_expectation": not mismatches,
        "mismatches": mismatches,
    }


# --------------------------------------------------------------- the smoke gate
def _smoke_gate(
    *,
    inputs: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, str]],
    store: Mapping[str, Any],
    keys: Mapping[str, Any],
) -> dict[str, Any]:
    """A1: deterministic assertions, run before any backend call exists."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    f_spec = inputs["forecasting"]["task_spec"]
    ad_spec = inputs["anomaly_detection"]["task_spec"]
    check(
        "forecasting task_spec is the T2-wired ssi default, verbatim",
        f_spec == EXPECTED_F_TASK_SPEC,
        f_spec,
    )
    check(
        "anomaly_detection task_spec is the frozen revision string, verbatim",
        ad_spec == AD_TASK_SPEC,
        ad_spec,
    )
    stripped_f = {k: v for k, v in inputs["forecasting"].items() if k != "task_spec"}
    stripped_ad = {
        k: v for k, v in inputs["anomaly_detection"].items() if k != "task_spec"
    }
    check(
        "public inputs are byte-identical once task_spec is removed",
        canonical_sha256(_plain(stripped_f)) == canonical_sha256(_plain(stripped_ad)),
        {
            "sha_without_task_spec": canonical_sha256(_plain(stripped_f)),
        },
    )
    user_f = prompts["forecasting"]["user"]
    user_ad = prompts["anomaly_detection"]["user"]
    spec_f_bytes = canonical_json_bytes(f_spec).decode("utf-8")
    spec_ad_bytes = canonical_json_bytes(ad_spec).decode("utf-8")
    check(
        "the two prompts differ exactly at the task_spec bytes",
        user_f.count(spec_f_bytes) == 1
        and user_ad == user_f.replace(spec_f_bytes, spec_ad_bytes),
        {
            "task_spec_occurrences_in_forecasting_prompt": user_f.count(spec_f_bytes),
            "replace_test": user_ad == user_f.replace(spec_f_bytes, spec_ad_bytes),
        },
    )
    check(
        "the system prompt is one byte sequence across the arms",
        prompts["forecasting"]["system"] == prompts["anomaly_detection"]["system"],
        {"system_sha256": hashlib.sha256(
            prompts["forecasting"]["system"].encode("utf-8")
        ).hexdigest()},
    )
    ad_full = prompts["anomaly_detection"]["system"] + prompts["anomaly_detection"]["user"]
    f_full = prompts["forecasting"]["system"] + prompts["forecasting"]["user"]
    check(
        "the AD arm is nowhere redescribed as forecasting",
        "forecast" not in ad_full.lower() and "smase" not in ad_full.lower(),
        {
            "forecast_occurrences": ad_full.lower().count("forecast"),
            "smase_occurrences": ad_full.lower().count("smase"),
        },
    )
    check(
        "the F arm carries no anomaly-task wording",
        "anomaly" not in f_full.lower(),
        {"anomaly_occurrences": f_full.lower().count("anomaly")},
    )

    # Information wall: no T1/T1b gain reading and no flip conclusion may
    # appear in either prompt.  Full-precision repr matches are hard failures;
    # 4-decimal rounded forms are recorded as warnings only (a 4-decimal
    # coincidence is not exposure).
    numbers: set[float] = set()
    readings = keys["readings"]
    for mapping in (
        readings["forecasting_delayed_aggregate_gain"],
        readings["ad_train_gain_aggregate"],
    ):
        numbers.update(float(v) for v in mapping.values())
    for per_series in (
        *readings["forecasting_delayed_per_series"].values(),
        *readings["ad_gain_per_series"].values(),
    ):
        numbers.update(float(v) for v in per_series.values())
    hard_hits: list[str] = []
    soft_hits: list[str] = []
    for value in sorted(numbers):
        if abs(value) < 1e-12:
            continue  # 0.0 is uninformative and legitimately present
        exact = repr(value)
        for arm, text in (("forecasting", f_full), ("anomaly_detection", ad_full)):
            if exact in text:
                hard_hits.append("%s:%s" % (arm, exact))
            elif ("%.4f" % value) in text:
                soft_hits.append("%s:%.4f" % (arm, value))
    word_hits = [
        token
        for token in ("flip", "positive_control", "answer_key")
        if token in f_full.lower() or token in ad_full.lower()
    ]
    check(
        "information wall: no T1/T1b gain reading or flip conclusion in either prompt",
        not hard_hits and not word_hits,
        {
            "full_precision_numeric_hits": hard_hits,
            "word_hits": word_hits,
            "rounded_form_warnings_not_blocking": soft_hits,
        },
    )
    check(
        "store state is 0 Guidance / 0 Experience / 0 learned Skill",
        store["resolved_memory_ids"] == [] and store["learned_skill_count"] == 0,
        {
            "resolved_skill_ids": store["resolved_skill_ids"],
            "resolved_memory_ids": store["resolved_memory_ids"],
            "learned_skill_count": store["learned_skill_count"],
            "bootstrap_note": store["bootstrap_note"],
        },
    )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "rule": (
            "every check is deterministic and runs before the first backend "
            "call; a failed gate stops the exam with no LLM spent"
        ),
    }


# ------------------------------------------------------------ draw classification
def _classify(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic runner-side reading of one parsed draw.

    OFF_MENU and contradictory payloads are valid parses but invalid draws:
    archived verbatim, never re-thrown, excluded from the distance matrix and
    breaking the arm's 3/3.
    """
    decision = str(payload.get("decision", ""))
    shortlist = [str(item) for item in payload.get("shortlist", ())]
    top1 = str(payload.get("top1", ""))
    menu = set(MENU)
    if decision == "abstain":
        if shortlist or top1 != "":
            return {
                "classification": "INVALID_CONTRADICTION",
                "valid": False,
                "why": "decision is abstain but shortlist/top1 are not empty",
                "proposal_set": None,
                "top1": None,
            }
        return {
            "classification": "VALID_ABSTAIN",
            "valid": True,
            "why": None,
            "proposal_set": [ABSTAIN_TOKEN],
            "top1": None,
        }
    off_menu = sorted((set(shortlist) | {top1}) - menu - {""})
    if off_menu:
        return {
            "classification": "INVALID_OFF_MENU",
            "valid": False,
            "why": "names outside the pinned menu: %s" % off_menu,
            "off_menu": off_menu,
            "proposal_set": None,
            "top1": None,
        }
    if not shortlist:
        return {
            "classification": "INVALID_CONTRADICTION",
            "valid": False,
            "why": "decision is propose but the shortlist is empty",
            "proposal_set": None,
            "top1": None,
        }
    if top1 not in shortlist:
        return {
            "classification": "INVALID_CONTRADICTION",
            "valid": False,
            "why": "top1 is not a member of the shortlist",
            "proposal_set": None,
            "top1": None,
        }
    return {
        "classification": "VALID_PROPOSE",
        "valid": True,
        "why": None,
        "proposal_set": sorted(set(shortlist)),
        "top1": top1,
    }


def _jaccard_distance(a: Sequence[str], b: Sequence[str]) -> float:
    left, right = set(a), set(b)
    union = left | right
    if not union:
        raise ValueError("Jaccard distance is undefined for two empty sets")
    return 1.0 - len(left & right) / len(union)


# -------------------------------------------------------------------- the exam
def _run_draws(
    *,
    view: Any,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    backend = _default_backend_factory(LLM_BUDGET)
    gateway = wvc.NoToolGateway({"episode_id": "t3_task_exam_v1", "arm": "exam"})
    core = TTHAAgentCore(backend, gateway, model=EXAM_MODEL, base_url=NF_BASE_URL)
    draws: list[dict[str, Any]] = []
    stopped = None
    for index, arm in enumerate(ARM_ORDER):
        tag = "F" if arm == "forecasting" else "AD"
        case_id = "T3EXAM_%s%d" % (tag, index + 1)
        public_input = inputs[arm]
        models_before = set(backend.returned_models)
        record: dict[str, Any] = {
            "draw_index": index + 1,
            "arm": arm,
            "case_id": case_id,
            "public_input_sha256": canonical_sha256(_plain(public_input)),
        }
        try:
            result = core.run_stage(
                role=AgentRole.FAST,
                stage=STAGE,
                case_id=case_id,
                public_input=public_input,
                harness_view=view,
                output_schema_name=SCHEMA_NAME,
                output_schema=EXAM_SCHEMA,
                source_snapshot_sha=view.effective_harness_view_sha,
                validation_retries=VALIDATION_RETRIES,
                post_validator=None,
            )
        except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
            record.update({
                "classification": "INVALID_PARSE",
                "valid": False,
                "protocol_error": "%s: %s" % (type(exc).__name__, exc),
                "validation_retry_count": getattr(exc, "validation_retry_count", None),
                "validation_error_codes": list(
                    getattr(exc, "validation_error_codes", ())
                ),
                "last_assistant_text_head": str(
                    getattr(exc, "last_assistant_text", "")
                )[:500],
                "raw_assistant_text": None,
                "parsed_payload": None,
                "proposal_set": None,
                "top1": None,
            })
        except (AgentTransportError, AgentCallBudgetExceeded) as exc:
            record.update({
                "classification": "INFRASTRUCTURE_STOP",
                "valid": False,
                "infrastructure_error": "%s: %s" % (type(exc).__name__, exc),
            })
            draws.append(record)
            stopped = "%s: %s" % (type(exc).__name__, exc)
            break
        else:
            payload = dict(result.payload)
            reading = _classify(payload)
            metadata = (
                dict(result.response.provider_metadata)
                if isinstance(result.response.provider_metadata, Mapping)
                else {}
            )
            record.update({
                "classification": reading["classification"],
                "valid": reading["valid"],
                "why": reading.get("why"),
                "raw_assistant_text": str(result.response.assistant_text),
                "parsed_payload": _plain(payload),
                "proposal_set": reading["proposal_set"],
                "top1": reading["top1"],
                "validation_retry_count": int(result.validation_retry_count),
                "validation_error_codes": list(result.validation_error_codes),
                "first_pass_valid": bool(result.first_pass_valid),
                "request_hashes": list(result.request_hashes),
                "returned_model": metadata.get("returned_model", ""),
                # provider_metadata is frozen (mappingproxy) by AgentResponse;
                # _plain de-freezes recursively, a shallow dict() would not
                "usage": _plain(metadata.get("usage", {})),
            })
        record["returned_models_delta"] = sorted(
            set(backend.returned_models) - models_before
        )
        record["llm_calls_cumulative"] = int(backend.calls)
        draws.append(record)
        print(
            "T3EXAM draw %d %s -> %s top1=%s retries=%s"
            % (
                index + 1,
                arm,
                record["classification"],
                record.get("top1"),
                record.get("validation_retry_count"),
            ),
            flush=True,
        )
    return {
        "draws": draws,
        "stopped": stopped,
        "llm_calls": int(backend.calls),
        "prompt_tokens": int(backend.prompt_tokens),
        "completion_tokens": int(backend.completion_tokens),
        "returned_models": sorted(backend.returned_models),
    }


# ------------------------------------------------------------- matrix + scoring
def _distance_matrix(draws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = {
        arm: [row for row in draws if row["arm"] == arm and row["valid"]]
        for arm in ("forecasting", "anomaly_detection")
    }

    def pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "pair": "%s/%s" % (a["case_id"], b["case_id"]),
            "a": sorted(a["proposal_set"]),
            "b": sorted(b["proposal_set"]),
            "jaccard_distance": _jaccard_distance(a["proposal_set"], b["proposal_set"]),
            "top1_agree": a.get("top1") == b.get("top1"),
        }

    same: list[dict[str, Any]] = []
    for arm_rows in valid.values():
        for i in range(len(arm_rows)):
            for j in range(i + 1, len(arm_rows)):
                same.append(pair(arm_rows[i], arm_rows[j]))
    cross = [pair(a, b) for a in valid["forecasting"] for b in valid["anomaly_detection"]]
    computable = bool(same) and bool(cross)
    min_cross = min(row["jaccard_distance"] for row in cross) if cross else None
    max_same = max(row["jaccard_distance"] for row in same) if same else None
    separated = bool(computable and min_cross > max_same)
    return {
        "rule": (
            "sets are the proposed menu entries; an abstaining draw is the "
            "singleton {__ABSTAIN__}; invalid draws are excluded.  Complete "
            "separation = min over cross-task pairs > max over same-task pairs."
        ),
        "same_task_pairs": same,
        "cross_task_pairs": cross,
        "computable": computable,
        "min_cross_task": min_cross,
        "max_same_task": max_same,
        "complete_separation": separated,
        "top1_agreement_same_task": (
            sum(1 for row in same if row["top1_agree"]) / len(same) if same else None
        ),
    }


def _score(
    draws: Sequence[Mapping[str, Any]], keys: Mapping[str, Any]
) -> dict[str, Any]:
    derived = keys["derived"]
    layers: dict[str, Any] = {}
    for layer, key_map, abstain_ok in (
        ("aggregate", derived["aggregate"], False),
        ("risk", derived["risk"], True),
    ):
        per_arm: dict[str, Any] = {}
        for arm in ("forecasting", "anomaly_detection"):
            key = set(key_map[arm])
            rows = []
            for draw in (row for row in draws if row["arm"] == arm):
                if not draw["valid"]:
                    appropriate, basis = False, "invalid draw breaks the 3/3"
                elif draw["classification"] == "VALID_ABSTAIN":
                    appropriate = abstain_ok
                    basis = (
                        "abstain is credited at the Risk layer"
                        if abstain_ok
                        else "abstain carries no top-1 and is not in the aggregate key"
                    )
                else:
                    appropriate = draw["top1"] in key
                    basis = "top1 %r vs key %s" % (draw["top1"], sorted(key))
                rows.append({
                    "case_id": draw["case_id"],
                    "classification": draw["classification"],
                    "top1": draw.get("top1"),
                    "appropriate": bool(appropriate),
                    "basis": basis,
                })
            count = sum(1 for row in rows if row["appropriate"])
            per_arm[arm] = {
                "key": sorted(key),
                "abstain_credited": abstain_ok,
                "appropriate": count,
                "of": DRAWS_PER_TASK,
                "passed": count == DRAWS_PER_TASK,
                "draws": rows,
            }
        layers[layer] = per_arm
    return {
        "aggregate_direction_passed": (
            layers["aggregate"]["forecasting"]["passed"]
            and layers["aggregate"]["anomaly_detection"]["passed"]
        ),
        "risk_passed": (
            layers["risk"]["forecasting"]["passed"]
            and layers["risk"]["anomaly_detection"]["passed"]
        ),
        "layers": layers,
        "quantization_note": (
            "AD per-series granularity is 4 events per series, so one event "
            "changing hands moves a series reading by ~0.2 or more; the AD "
            "Risk layer reads approximately as 'did the proposal avoid "
            "touching the substrate at all'.  F Risk is read at the -0.005 "
            "harm line on four evaluation series."
        ),
    }


def _verdict(
    *,
    draws: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any],
    scoring: Mapping[str, Any],
) -> dict[str, Any]:
    invalid = [row for row in draws if not row["valid"]]
    separated = bool(matrix["complete_separation"])
    direction = bool(scoring["aggregate_direction_passed"])
    risk = bool(scoring["risk_passed"])
    if len(invalid) > 2:
        verdict = "EXAM_PROTOCOL_UNREADABLE"
        reason = "%d of 6 draws are invalid (OFF_MENU or parse failure)" % len(invalid)
    elif not separated:
        verdict = "SAMPLING_VARIANCE_DOMINATES"
        reason = (
            "no complete separation: min cross-task distance %s, max same-task "
            "distance %s -- the full distribution is attached"
            % (matrix["min_cross_task"], matrix["max_same_task"])
        )
    elif not direction:
        verdict = "TASK_SEPARATED_DIRECTION_INAPPROPRIATE"
        reason = (
            "complete separation holds but the aggregate-direction layer is "
            "not 3/3 on both arms"
        )
    elif risk:
        verdict = "TASK_CONDITIONED_RISK_AWARE_PROPOSALS_CONFIRMED"
        reason = (
            "complete separation, 3/3 aggregate direction and 3/3 Risk-layer "
            "appropriateness on both arms"
        )
    else:
        verdict = "TASK_CONDITIONED_PROPOSALS_CONFIRMED"
        reason = (
            "complete separation and 3/3 aggregate direction on both arms; "
            "the Risk layer is not 3/3 (expected to be reachable only with "
            "Experience -- the T4 entry evidence)"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "ladder_trace": {
            "invalid_draws": len(invalid),
            "invalid_draw_case_ids": [row["case_id"] for row in invalid],
            "complete_separation": separated,
            "aggregate_direction_passed": direction,
            "risk_passed": risk,
        },
        "interpretation_note": (
            "task semantics are visible at deployment by construction; this "
            "reading proves the proposals are conditioned on the task "
            "observation, not that the Agent discovered the task physics from "
            "the data, and it says nothing about execution or adoption"
        ),
    }


# -------------------------------------------------------------------- report
def _render_md(doc: Mapping[str, Any]) -> str:
    lines = [
        "# T3 (#39) task-conditioned proposal exam -- %s" % doc["verdict"],
        "",
        "- protocol: `%s` (evidence grade %s, permanent)" % (
            doc["protocol_version"], doc["evidence_grade"],
        ),
        "- Part 0 checkpoint: `%s` (%d files)" % (
            doc["part0_checkpoint"]["commit"], doc["part0_checkpoint"]["files"],
        ),
        "- backend: `%s` at `%s`; request carries model+messages only "
        "(provider default sampling); returned models: %s" % (
            doc["backend_declaration"]["requested_model"],
            doc["backend_declaration"]["base_url"],
            ", ".join(doc["backend_declaration"]["returned_models"]) or "none",
        ),
        "- cost: %d LLM calls (%d in, %d out tokens); 0 forecasting retrains; "
        "0 AD evaluations" % (
            doc["budgets"]["llm_calls"],
            doc["backend_declaration"]["prompt_tokens"],
            doc["backend_declaration"]["completion_tokens"],
        ),
        "- store: `%s` -- %s" % (
            doc["store_state"]["runtime_bundle_sha"][:12],
            doc["store_state"]["empty_store_statement"],
        ),
        "- sampling: `%s` (first attempt %s)" % (
            (doc.get("sampling_attempt") or {}).get("this_run", "unlabeled"),
            ((doc.get("sampling_attempt") or {}).get("first_attempt") or {}).get(
                "status", "unrecorded"
            ),
        ),
        "",
        "## Smoke gate (A1, before any LLM call): %s" % (
            "PASSED" if doc["smoke_gate"]["passed"] else "FAILED"
        ),
        "",
    ]
    for row in doc["smoke_gate"]["checks"]:
        lines.append("- [%s] %s" % ("x" if row["passed"] else " ", row["name"]))
    keys = doc["answer_keys"]
    lines += [
        "",
        "## Answer keys (derived in-runner from frozen artifacts)",
        "",
        "- aggregate F: %s" % ", ".join(keys["derived"]["aggregate"]["forecasting"]),
        "- aggregate AD: %s" % ", ".join(keys["derived"]["aggregate"]["anomaly_detection"]),
        "- risk F: %s" % ", ".join(keys["derived"]["risk"]["forecasting"]),
        "- risk AD: %s (+ abstain credited)" % ", ".join(keys["derived"]["risk"]["anomaly_detection"]),
        "- matches the frozen expectation: %s" % keys["matches_expectation"],
        "",
        "## Draws (order %s)" % ", ".join(doc["arm_order"]),
        "",
        "| # | arm | classification | top1 | shortlist | retries | returned model |",
        "|---|-----|----------------|------|-----------|---------|----------------|",
    ]
    for row in doc["draws"]:
        payload = row.get("parsed_payload") or {}
        lines.append(
            "| %d | %s | %s | %s | %s | %s | %s |"
            % (
                row["draw_index"],
                row["arm"],
                row["classification"],
                row.get("top1") or ("__ABSTAIN__" if row["classification"] == "VALID_ABSTAIN" else "-"),
                ", ".join(payload.get("shortlist", ()) or ()) or "-",
                row.get("validation_retry_count"),
                row.get("returned_model") or "-",
            )
        )
    matrix = doc["distance_matrix"]
    lines += [
        "",
        "## Distance matrix",
        "",
        "- same-task pairs: %s" % ", ".join(
            "%s=%.4f" % (row["pair"], row["jaccard_distance"])
            for row in matrix["same_task_pairs"]
        ),
        "- cross-task pairs: %s" % ", ".join(
            "%s=%.4f" % (row["pair"], row["jaccard_distance"])
            for row in matrix["cross_task_pairs"]
        ),
        "- min cross-task: %s; max same-task: %s; complete separation: %s" % (
            matrix["min_cross_task"], matrix["max_same_task"],
            matrix["complete_separation"],
        ),
        "",
        "## Verdict",
        "",
        "**%s** -- %s" % (doc["verdict"], doc["verdict_reason"]),
        "",
        "> %s" % doc["interpretation_note"],
        "",
    ]
    if doc["ambiguities_reported_not_self_adjudicated"]:
        lines += ["## Ambiguities (reported, not self-adjudicated)", ""]
        for item in doc["ambiguities_reported_not_self_adjudicated"]:
            lines.append("- %s" % item)
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------- main
def main() -> None:
    started = time.perf_counter()
    smoke_only = "--smoke-only" in sys.argv[1:]
    ambiguities: list[str] = []

    frozen_before = _freeze()
    git_status_start = _git(["status", "--short", "-uno"])
    read_only_before = _read_only_inventory()

    substrate = _load_substrate()
    search = _StandInSearch(substrate)
    store = _build_empty_store()
    snapshot = store["_snapshot"]
    context = ssi._public_features(search)
    view = resolve_harness_view(snapshot, dict(context["features"]), role="fast")
    learned = [
        skill.skill_id
        for skill in snapshot.skills
        if skill.skill_kind.value != "bootstrap_procedure"
    ]
    store_state = {
        key: value for key, value in store.items() if not key.startswith("_")
    }
    store_state.update({
        "resolved_skill_ids": list(view.skill_ids),
        "resolved_memory_ids": list(view.memory_ids),
        "effective_harness_view_sha": view.effective_harness_view_sha,
        "learned_skill_count": len(learned),
        "bootstrap_note": (
            "the three resolved skills are h0's always-on bootstrap "
            "procedures (standing operating procedure, identical in every "
            "experiment); the empty-store claim is 0 learned Guidance cards, "
            "0 Experience memories, 0 learned Skills"
        ),
        "empty_store_statement": (
            "0 Guidance / 0 Experience / 0 learned Skill; bootstrap "
            "procedures always on; one snapshot read by all six draws"
        ),
    })

    inputs = _public_inputs(search)
    prompts = _render_prompts(view, inputs)
    keys = _derive_answer_keys()
    smoke = _smoke_gate(inputs=inputs, prompts=prompts, store=store_state, keys=keys)

    doc: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_grade_note": (
            "permanent: an injected flip can be constructive, so this slice "
            "can never authorize Shared Capability execution rights"
        ),
        "role": "task-conditioned proposal exam (proposal layer only)",
        "part0_checkpoint": dict(PART0_CHECKPOINT),
        "sampling_attempt": {
            "this_run": "second_sample",
            "first_attempt": dict(FIRST_ATTEMPT),
        },
        "verdict": None,
        "stopped": None,
        "ambiguities_reported_not_self_adjudicated": ambiguities,
        "arm_order": list(ARM_ORDER),
        "menu": list(MENU),
        "task_specs": {
            "forecasting": inputs["forecasting"]["task_spec"],
            "anomaly_detection": inputs["anomaly_detection"]["task_spec"],
        },
        "substrate": {
            "source": substrate["source"],
            "block": substrate["block"],
            "series_length": substrate["series_length"],
            "train_uids": substrate["train_uids"],
            "observation_cutoff": int(search.support[0]),
            "construction": (
                "the ssi public context construction (_base_input + the "
                "shared observation extractor) over the T1 injected copy's "
                "training block; both arms share every byte except task_spec"
            ),
            "deviations_from_the_ssi_body": [
                "program_menu narrowed to the book-pinned five entries "
                "(identity + the four programs the frozen answer keys cover)",
                "evaluation_budget block dropped: the exam runs no Support "
                "evaluation, so its economics are not presented",
                "one neutral task-description-authority note added, "
                "byte-identical in both arms (revision point 5)",
            ],
        },
        "store_state": store_state,
        "answer_keys": keys,
        "smoke_gate": smoke,
        "prompts_verbatim": {
            "rule": (
                "one rendering per arm; case_id is call wiring and is never "
                "rendered, so every draw of an arm sends exactly these bytes"
            ),
            "system_both_arms": prompts["forecasting"]["system"],
            "user_forecasting": prompts["forecasting"]["user"],
            "user_anomaly_detection": prompts["anomaly_detection"]["user"],
        },
        "backend_declaration": {
            "requested_model": EXAM_MODEL,
            "base_url": NF_BASE_URL,
            "request_fields": (
                "model + messages only; no explicit sampling parameters are "
                "sent, so provider defaults apply (the standing default of "
                "this path)"
            ),
            "returned_models": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
        "budgets": {
            "llm_budget": LLM_BUDGET,
            "llm_calls": 0,
            "forecasting_retrains": 0,
            "forecasting_retrain_budget": 0,
            "ad_evaluations": 0,
            "ad_evaluation_budget": 0,
            "budgets_respected": True,
        },
        "frozen_surface": {
            "name": "FROZEN_SURFACE_V9",
            "raw_entries": len(list(FROZEN_SURFACE_V9)),
            "unique_files": len(set(FROZEN_SURFACE_V9)),
            "before_files": len(frozen_before),
            "after": None,
        },
        "git": {
            "status_uno_short_at_start": git_status_start,
            "diff_name_only_at_end": None,
        },
        "read_only_integrity": {
            "paths": sorted(read_only_before),
            "before": read_only_before,
            "after": None,
            "unchanged": None,
        },
        "wall_seconds": None,
    }

    ambiguities.append(
        "labeled second sample: first live attempt bash-joon3149 spent the "
        "six draws and died on mappingproxy json.dumps; draws 1-4 were lost "
        "to tail -20; recovered tail is F5=outlier_mad / AD6=identity; the "
        "first six draws are not scored"
    )
    if not keys["matches_expectation"]:
        ambiguities.append(
            "derived answer keys differ from the frozen expectation: %s "
            "(scoring used the derived keys, per the book the mismatch is "
            "reported, not self-adjudicated)" % keys["mismatches"]
        )
    if keys["derivation"]["cross_checks"] != {
        "forecasting_aggregate_t1_vs_v3": True,
        "forecasting_per_series_t1_vs_v3": True,
    }:
        ambiguities.append(
            "the v3 artifact's carried T1 forecasting readings do not match "
            "the T1 artifact exactly: %s" % keys["derivation"]["cross_checks"]
        )

    if not smoke["passed"]:
        doc["stopped"] = "SMOKE_GATE_FAILED"
        doc["wall_seconds"] = time.perf_counter() - started
        OUT_JSON.write_text(_json_text(doc), encoding="utf-8")
        print("SMOKE GATE FAILED -- no LLM call made; see %s" % OUT_JSON)
        return
    if smoke_only:
        print("smoke gate passed (--smoke-only; no LLM call, nothing written)")
        for row in smoke["checks"]:
            print("  [%s] %s" % ("x" if row["passed"] else " ", row["name"]))
        return

    outcome = _run_draws(view=view, inputs=inputs)
    draws = outcome["draws"]
    doc["stopped"] = outcome["stopped"]
    doc["budgets"]["llm_calls"] = outcome["llm_calls"]
    doc["budgets"]["budgets_respected"] = outcome["llm_calls"] <= LLM_BUDGET
    doc["backend_declaration"].update({
        "returned_models": outcome["returned_models"],
        "prompt_tokens": outcome["prompt_tokens"],
        "completion_tokens": outcome["completion_tokens"],
    })
    doc["draws"] = draws

    if outcome["stopped"] is not None:
        ambiguities.append(
            "the exam stopped on an infrastructure error (%s); no verdict is "
            "claimed and the partial record is archived" % outcome["stopped"]
        )
    else:
        matrix = _distance_matrix(draws)
        scoring = _score(draws, keys)
        verdict = _verdict(draws=draws, matrix=matrix, scoring=scoring)
        doc["distance_matrix"] = matrix
        doc["scoring"] = scoring
        doc["verdict"] = verdict["verdict"]
        doc["verdict_reason"] = verdict["reason"]
        doc["verdict_ladder_trace"] = verdict["ladder_trace"]
        doc["interpretation_note"] = verdict["interpretation_note"]
        if not matrix["computable"]:
            ambiguities.append(
                "the distance matrix is not fully computable on the valid "
                "draws; the separation reading is partial"
            )

    doc["frozen_surface"]["after"] = _verify(frozen_before)
    doc["git"]["diff_name_only_at_end"] = _git(["diff", "--name-only", "HEAD"])
    read_only_after = _read_only_inventory()
    doc["read_only_integrity"].update({
        "after": read_only_after,
        "unchanged": read_only_after == read_only_before,
    })
    doc["wall_seconds"] = time.perf_counter() - started

    OUT_JSON.write_text(_json_text(doc), encoding="utf-8")
    if doc["verdict"] is not None:
        OUT_MD.write_text(_render_md(doc), encoding="utf-8")
    print(
        "T3 exam: verdict=%s llm=%d stopped=%s"
        % (doc["verdict"], outcome["llm_calls"], outcome["stopped"])
    )


def _fake_backend_factory(maximum_calls: int):
    """0-LLM write-path rehearsal.  Never used by the live exam."""

    class _FakeTransport:
        def complete(self, request):
            user = ""
            for message in request.messages:
                if message.get("role") == "user":
                    user = str(message.get("content") or "")
                    break
            if (
                AD_TASK_SPEC["task_id"] in user
                and AD_TASK_SPEC["consumer_id"] in user
            ):
                payload = {
                    "decision": "propose",
                    "shortlist": ["identity"],
                    "top1": "identity",
                    "reason": "rehearsal anomaly-detection proposal",
                }
            else:
                payload = {
                    "decision": "propose",
                    "shortlist": ["outlier_mad", "winsorize"],
                    "top1": "outlier_mad",
                    "reason": "rehearsal forecasting proposal",
                }
            envelope = {
                "schema_version": "agent-envelope/1",
                "kind": "stage_result",
                "stage": STAGE,
                "payload": payload,
            }
            return AgentResponse.valid(
                envelope,
                raw_response={"choices": [{"message": {"content": "rehearsal"}}]},
                provider_metadata={
                    "returned_model": "rehearsal-fake",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

    return BudgetedAgentBackend(_FakeTransport(), maximum_calls=maximum_calls)


def _rehearse_write() -> None:
    """Full write path against a fake backend.  Does not touch the live pair."""
    self_mod = sys.modules[__name__]
    scratch = PROJECT_ROOT / "_scratch" / "skill_store" / "t3_task_exam_v1_rehearsal"
    if scratch.exists():
        shutil.rmtree(scratch)
    live_json = E2 / "t3_task_exam_v1.json"
    live_md = E2 / "t3_task_exam_v1.md"
    before_live = {
        "json_exists": live_json.exists(),
        "md_exists": live_md.exists(),
        "json_sha": _sha256(live_json) if live_json.exists() else None,
        "md_sha": _sha256(live_md) if live_md.exists() else None,
    }
    original_store = self_mod.STORE_ROOT
    original_json = self_mod.OUT_JSON
    original_md = self_mod.OUT_MD
    original_factory = self_mod._default_backend_factory
    self_mod.STORE_ROOT = scratch
    self_mod.OUT_JSON = scratch / "rehearsal.json"
    self_mod.OUT_MD = scratch / "rehearsal.md"
    self_mod._default_backend_factory = _fake_backend_factory
    try:
        self_mod.main()
        text = self_mod.OUT_JSON.read_text(encoding="utf-8")
        json.loads(text)
        if not self_mod.OUT_MD.is_file():
            raise SystemExit("rehearsal did not write markdown")
        after_live = {
            "json_exists": live_json.exists(),
            "md_exists": live_md.exists(),
            "json_sha": _sha256(live_json) if live_json.exists() else None,
            "md_sha": _sha256(live_md) if live_md.exists() else None,
        }
        if after_live != before_live:
            raise SystemExit("rehearsal leaked onto the live deliverables")
        print("rehearsal write path ok: %s" % self_mod.OUT_JSON)
    finally:
        self_mod.STORE_ROOT = original_store
        self_mod.OUT_JSON = original_json
        self_mod.OUT_MD = original_md
        self_mod._default_backend_factory = original_factory


if __name__ == "__main__":
    if "--rehearse-write" in sys.argv[1:]:
        _rehearse_write()
    else:
        main()
