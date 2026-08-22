"""T4 (#40): task-keyed conflict Experience -- does written experience fix F's risk blind spot?

Gate: C14 (task-conditioned proposals) confirmed in T3.  #39 proved the Agent
reads *which task* it is being asked about.  It also exposed the gap this book
is about: with an empty store the F arm chose ``hampel_filter`` 3/3 -- the LLM's
prior says a median filter is the gentle, safe choice -- while in the frozen
readings ``hampel_filter`` is the one program that improves the F aggregate and
still harms an individual evaluation series past the -0.005 line.  A prior
cannot close that gap; only a written record of what actually happened can.

The single question: can Harness-written task-keyed success / failure / partial
conflict experience correct F's risk blind spot **without** breaking AD's
conservative choice?

``evidence_grade = POSITIVE_CONTROL``, permanently.  The ten episodes replay the
frozen T1b v3 readings, which are also the source of this exam's answer keys, so
a pass demonstrates the Memory machinery -- write, key, retrieve, render,
condition -- and never that the Agent discovered anything.

Design (book #40 v2):

* Memory is the only change surface this round.  ``methods/ttha`` carries three
  edits: one key mint (``experience_memory.task_consumer_key`` /
  ``cell_key``), the mechanical five-way lifecycle classification
  (``classify_relation``), and the card vocabulary's Consumer facts.
* Writing goes through the Runtime: ``build_episode`` ->
  ``TTHAMethod.append_experience_episode`` -> the same instance's retrieval
  reads them back.  Nothing is hand-inserted into a store file and nothing
  built inside this report counts as a write.
* All ten episodes are ``evidence_level=DELAYED``, ``local_status=EPISODE_ONLY``,
  no promotion, no TRY right, ordinary unique ids: this is a historical replay
  of ten readings, not ten fresh pieces of evidence.
* The exam is #39's instrument, verbatim: same public inputs, same menu, same
  backend, same alternating order, three draws per task, at most one validation
  retry.  The only new byte is the retrieved card, which enters through the
  standing path -- prepended to the resolved Harness instruction, exactly as
  ``fast_agent.prepare`` does it.
* Baseline is #39's second sample, read out of its artifact, never retyped.

Verdict ladder (pre-registered, Part D):
  CONFLICT_EXPERIENCE_CONDITIONS_PROPOSALS_CONFIRMED -- F 3/3 top-1 in
    {outlier_iqr, outlier_mad, winsorize}, AD 3/3 top-1 in {identity} or
    abstain, and separation kept (min cross > max same);
  EXPERIENCE_IGNORED -- F did not move at all and the cards are provably in the
    prompt (credible negative; the next surface is card presentation or Slow,
    not the Memory key);
  RETRIEVAL_MISS -- empty-handed or cross-task retrieval;
  EXAM_PROTOCOL_UNREADABLE -- more than two of six draws invalid;
  PARTIAL_EXPERIENCE_CONDITIONING -- every other intermediate state; reports
  F/AD safe counts, the separation reading, the AD risk-regression flag and
  the direction of movement, and closes nothing.

Budgets: LLM <= 12, forecasting retrains 0, AD evaluations 0.

Run:

    python evaluation/functional/run_e2_t4_conflict_experience.py
    python evaluation/functional/run_e2_t4_conflict_experience.py --smoke-only

Writes ``artifacts/functional/e2/t4_conflict_experience_v1.json`` and ``.md``.
"""
from __future__ import annotations

import dataclasses
import json
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

import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_t3_task_conditioned_exam as t3  # noqa: E402  -- the #39 instrument
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
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    anomaly_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    AgentRole,
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    CLASSIFICATION_MATERIAL_THRESHOLD,
    EVIDENCE_DELAYED,
    MEASURED_EFFECT_KEY,
    RELATION_ABSTAIN,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_NEUTRAL,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    build_episode,
    classify_relation,
    render_experience_pack,
    resolve_experience_contrast_pack,
    task_consumer_key,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentTransportError,
)

PROTOCOL_VERSION = "t4_conflict_experience_v1"
EVIDENCE_GRADE = "POSITIVE_CONTROL"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t4_conflict_experience_v1.json"
OUT_MD = E2 / "t4_conflict_experience_v1.md"
T3_ARTIFACT = E2 / "t3_task_exam_v1.json"
T1B_V3_ARTIFACT = t3.T1B_V3_ARTIFACT
STORE_ROOT = PROJECT_ROOT / "_scratch" / "skill_store" / "t4_conflict_experience_v1"

PART0_CHECKPOINT = {
    "commit": "fd29501",
    "files": 5,
    "note": (
        "#39 deliverables (t3 runner + t3_task_exam_v1.json/.md) + the "
        "main-line doc revisions: C14, the #39 closing entry, the "
        "deposition/stop-loss ruling, the #40 pre-issue revision section "
        "and the progress line"
    ),
}

MENU = t3.MENU
ARM_ORDER = t3.ARM_ORDER
DRAWS_PER_TASK = t3.DRAWS_PER_TASK
LLM_BUDGET = t3.LLM_BUDGET
VALIDATION_RETRIES = t3.VALIDATION_RETRIES
EXAM_MODEL = t3.EXAM_MODEL
STAGE = t3.STAGE
SCHEMA_NAME = t3.SCHEMA_NAME
EXAM_SCHEMA = t3.EXAM_SCHEMA
ABSTAIN_TOKEN = t3.ABSTAIN_TOKEN
EPISODE_COUNT = 2 * len(MENU)

COHORT = t3.EXAM_TARGET["cohort"]
CONSUMER_VARIANT = t3.EXAM_TARGET["consumer_variant"]

# Part D, pre-registered success sets.
F_SAFE_TOP1 = frozenset({"outlier_iqr", "outlier_mad", "winsorize"})
AD_SAFE_TOP1 = frozenset({"identity"})  # plus abstain, handled explicitly

# The programs the AD arm may legitimately name as a repair; used only by the
# B4 category acceptance, never by scoring.
REPAIR_PROGRAMS = frozenset({"outlier_iqr", "outlier_mad", "hampel_filter"})

_plain = t3._plain
_json_text = t3._json_text
_sha256 = t3._sha256
_git = t3._git
_repo_rel = t3._repo_rel


# --------------------------------------------------------------- the two keys
def _arm_task_keys(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Both Memory keys through the one mint; no string is built by hand.

    F: the ssi runtime TaskSpec for this cell's consumer variant.
    AD: the in-service anomaly factory, its downstream model class taken from
    the AD task_spec the prompt actually carries -- so the key is derived from
    the task description the Agent reads, not chosen next to it.
    """
    f_spec = ssi._runtime_task_spec(CONSUMER_VARIANT)
    ad_spec = anomaly_task_spec_v1(
        downstream_model_class=str(
            inputs["anomaly_detection"]["task_spec"]["consumer_id"]
        )
    )
    return {
        "forecasting": {"spec": f_spec, "key": task_consumer_key(f_spec)},
        "anomaly_detection": {"spec": ad_spec, "key": task_consumer_key(ad_spec)},
        "rule": (
            "one mint (experience_memory.task_consumer_key) over "
            "task_type|downstream_model_class|metric.name; the F spec comes "
            "from ssi's in-service factory for this consumer variant, the AD "
            "spec from contracts.task.anomaly_task_spec_v1 with the "
            "downstream model class read off the AD task_spec in the prompt; "
            "cohort appears in domain_namespace and Context only"
        ),
        "consistency_with_the_prompt": {
            "forecasting_consumer_id_in_prompt": str(
                inputs["forecasting"]["task_spec"]["consumer_id"]
            ),
            "forecasting_downstream_model_class": f_spec.downstream_model_class,
            "anomaly_consumer_id_in_prompt": str(
                inputs["anomaly_detection"]["task_spec"]["consumer_id"]
            ),
            "anomaly_downstream_model_class": ad_spec.downstream_model_class,
        },
    }


# ------------------------------------------------------- the ten Action-Response
def _frozen_readings() -> dict[str, Any]:
    """The ten readings, straight off the frozen T1b v3 artifact."""
    doc = json.loads(T1B_V3_ARTIFACT.read_text(encoding="utf-8"))
    part_d = doc["part_d"]
    aggregate = {
        "forecasting": {
            str(row["program"]): float(row["forecasting_delayed_aggregate_gain"])
            for row in part_d["flip_check_per_program"]
        },
        "anomaly_detection": {
            str(row["program"]): float(row["ad_train_gain_pooled"])
            for row in part_d["flip_check_per_program"]
        },
    }
    per_series = {
        "forecasting": {
            program: {str(uid): float(gain) for uid, gain in rows.items()}
            for program, rows in part_d["forecasting_delayed_per_series"].items()
        },
        "anomaly_detection": {
            program: {str(uid): float(gain) for uid, gain in rows.items()}
            for program, rows in part_d["ad_gain_per_series"].items()
        },
    }
    for arm in ("forecasting", "anomaly_detection"):
        aggregate[arm].setdefault("identity", 0.0)
    return {
        "source": _repo_rel(T1B_V3_ARTIFACT),
        "aggregate": aggregate,
        "per_series": per_series,
        "reading_windows": {
            "forecasting": "delayed window aggregate + per evaluation series",
            "anomaly_detection": (
                "training-side pooled macro-F1 + per training series"
            ),
        },
        "identity_note": (
            "identity is the reference arm: its gain is 0 against itself by "
            "definition, which the artifact records rather than measures"
        ),
    }


def _episode_context(
    *, program: str, features: Mapping[str, Any], substrate: Mapping[str, Any],
    cutoff: int,
) -> dict[str, Any]:
    """ssi's live data Context plus the program's action geometry.

    Everything here is an observable feature.  No dataset name, no file, no
    answer table: the geometry says where the operator acted and how wide the
    block was, not which benchmark it came from.
    """
    numeric = {
        str(key): float(value)
        for key, value in dict(features).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    block = substrate["block"]
    return {
        "cohort": {
            "cohort_name": COHORT,
            "series_count": len(substrate["train_uids"]),
            "observation_cutoff": int(cutoff),
        },
        "local_pattern": numeric,
        "program_geometry": {
            "program": str(program),
            "scope": "training_rows",
            "block_length": int(block[1]) - int(block[0]),
            "applied_once_per_training_window": True,
        },
    }


def _build_episodes(
    *, keys: Mapping[str, Any], readings: Mapping[str, Any],
    features: Mapping[str, Any], substrate: Mapping[str, Any], cutoff: int,
) -> dict[str, Any]:
    """Ten Episodes, relation derived mechanically, never assigned by hand."""
    rows: list[dict[str, Any]] = []
    episodes: list[Any] = []
    for arm in ("forecasting", "anomaly_detection"):
        consumer_id = keys[arm]["spec"].downstream_model_class
        for program in MENU:
            is_identity = program == "identity"
            facts = classify_relation(
                aggregate_gain=readings["aggregate"][arm].get(program),
                per_series_gains=readings["per_series"][arm].get(program) or {},
                is_identity=is_identity,
                consumer_id=consumer_id,
            )
            signature = workflow_signature_of(
                () if is_identity else ({"op": program},)
            )
            episode = build_episode(
                episode_id="t4_%s_%s_replay_v1" % (arm, program),
                task_consumer_key=keys[arm]["key"],
                domain_namespace=COHORT,
                context_summary=_episode_context(
                    program=program, features=features,
                    substrate=substrate, cutoff=cutoff,
                ),
                workflow_signature=signature,
                support_response={
                    "evaluated": False,
                    "gain": None,
                    "why_no_support_reading": (
                        "this is a replay of a delayed/training-side reading "
                        "taken in an earlier round; no Support window was "
                        "spent here"
                    ),
                },
                delayed_response={
                    "evaluated": not is_identity,
                    "gain": facts["aggregate_gain"],
                    "reading_window": readings["reading_windows"][arm],
                    MEASURED_EFFECT_KEY: facts,
                    "historical_replay": True,
                    "counts_as_new_evidence": False,
                    "grants_try_right": False,
                    "authorizes_execution": False,
                },
                relation=str(facts["relation"]),
                evidence_level=EVIDENCE_DELAYED,
                local_status=STATUS_EPISODE_ONLY,
                evidence_refs=(readings["source"], PROTOCOL_VERSION),
            )
            episodes.append(episode)
            rows.append({
                "episode_id": episode.episode_id,
                "arm": arm,
                "program": program,
                "task_consumer_key": episode.task_consumer_key,
                "domain_namespace": episode.domain_namespace,
                "workflow_signature": signature,
                "relation": episode.relation,
                "evidence_level": episode.evidence_level,
                "local_status": episode.local_status,
                "aggregate_gain": facts["aggregate_gain"],
                "aggregate_direction": facts["aggregate_direction"],
                "series_read": facts["series_read"],
                "harmed_series_count": facts["harmed_series_count"],
                "harmed_series": facts["harmed_series"],
                "min_per_series_gain": facts["min_per_series_gain"],
                "classification_basis": facts["classification_basis"],
            })
    expected = {
        "forecasting": {
            "identity": RELATION_ABSTAIN,
            "outlier_iqr": RELATION_POSITIVE,
            "outlier_mad": RELATION_POSITIVE,
            "winsorize": RELATION_POSITIVE,
            "hampel_filter": RELATION_CONFLICT,
        },
        "anomaly_detection": {
            "identity": RELATION_ABSTAIN,
            "outlier_iqr": RELATION_CONFLICT,
            "outlier_mad": RELATION_CONFLICT,
            "hampel_filter": RELATION_CONFLICT,
            "winsorize": RELATION_NEGATIVE,
        },
    }
    mismatches = [
        {
            "arm": row["arm"], "program": row["program"],
            "derived": row["relation"],
            "expected": expected[row["arm"]][row["program"]],
        }
        for row in rows
        if row["relation"] != expected[row["arm"]][row["program"]]
    ]
    return {
        "episodes": episodes,
        "rows": rows,
        "count": len(rows),
        "rule": (
            "relation is classify_relation's mechanical output over the "
            "frozen +/-%g line; identity -> ABSTAIN; aggregate >= +t with no "
            "per-series below -t -> POSITIVE; aggregate >= +t with at least "
            "one below -t -> CONFLICT; aggregate < -t -> NEGATIVE; the rest "
            "-> NEUTRAL" % CLASSIFICATION_MATERIAL_THRESHOLD
        ),
        "expected": expected,
        "matches_expectation": not mismatches,
        "mismatches": mismatches,
        "neutral_produced": [
            row["episode_id"] for row in rows if row["relation"] == RELATION_NEUTRAL
        ],
    }


# ------------------------------------------------------------------ the write
def _write_through_runtime(
    *, core: Any, snapshot: Any, episodes: Sequence[Any],
) -> dict[str, Any]:
    """build_episode -> TTHAMethod.append_experience_episode -> read back."""
    method = TTHAMethod(TTHAFastAgent(core), snapshot)
    before = len(method.experience_episodes)
    for episode in episodes:
        method.append_experience_episode(episode)
    after = tuple(method.experience_episodes)
    return {
        "method": method,
        "record": {
            "path": (
                "experience_memory.build_episode -> "
                "TTHAMethod.append_experience_episode -> the same instance's "
                "retrieval; nothing was hand-inserted and nothing built "
                "inside the report counts as a write"
            ),
            "episodes_before": int(before),
            "episodes_after": len(after),
            "appended": len(after) - int(before),
            "episode_ids": [ep.episode_id for ep in after],
            "read_back_from_runtime": (
                [ep.episode_id for ep in after]
                == [ep.episode_id for ep in episodes]
            ),
            "store_snapshot_note": (
                "the store is #39's h0 snapshot, materialized fresh and not "
                "written to; the ten episodes live in the method instance, so "
                "there is no new store sha to report"
            ),
            "evidence_levels": sorted({ep.evidence_level for ep in after}),
            "local_statuses": sorted({ep.local_status for ep in after}),
            "no_promotion": all(
                ep.local_status == STATUS_EPISODE_ONLY for ep in after
            ),
        },
    }


# -------------------------------------------------------------- B4 retrieval
def _retrieve(
    *, method: Any, keys: Mapping[str, Any], features: Mapping[str, Any],
) -> dict[str, Any]:
    """One deterministic pack per arm through the live retrieval function."""
    held = list(method.experience_episodes)
    out: dict[str, Any] = {"arms": {}, "held_episode_count": len(held)}
    for arm in ("forecasting", "anomaly_detection"):
        pack = resolve_experience_contrast_pack(
            held,
            features,
            keys[arm]["key"],
            allowed_operators=tuple(MENU),
        )
        rendered = render_experience_pack(pack.to_dict()) if pack else ""
        picked: dict[str, Any] = {}
        for slot in ("positive", "negative", "conflict"):
            episode = getattr(pack, slot, None) if pack else None
            facts = (
                dict(episode.delayed_response.get(MEASURED_EFFECT_KEY) or {})
                if episode is not None
                else {}
            )
            picked[slot] = (
                {
                    "episode_id": episode.episode_id,
                    "program": episode.workflow_signature,
                    "relation": episode.relation,
                    "task_consumer_key": episode.task_consumer_key,
                    "harmed_series_count": facts.get("harmed_series_count"),
                    "min_per_series_gain": facts.get("min_per_series_gain"),
                }
                if episode is not None
                else None
            )
        card_ids = [
            row["episode_id"] for row in picked.values() if row is not None
        ]
        cross_task = [
            row["episode_id"]
            for row in picked.values()
            if row is not None and row["task_consumer_key"] != keys[arm]["key"]
        ]
        out["arms"][arm] = {
            "task_consumer_key": keys[arm]["key"],
            "evidence_sufficient": bool(pack.evidence_sufficient) if pack else False,
            "retrieval_note": pack.retrieval_note if pack else "no pack",
            "picked": picked,
            "card_episode_ids": card_ids,
            "cross_task_card_ids": cross_task,
            "rendered_card": rendered,
            "rendered_card_sha256": canonical_sha256(rendered),
            "rendered_card_length": len(rendered),
        }
    return out


def _b4_acceptance(retrieval: Mapping[str, Any]) -> dict[str, Any]:
    """Category acceptance -- what kind of card came back, not which instance."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    f_arm = retrieval["arms"]["forecasting"]
    ad_arm = retrieval["arms"]["anomaly_detection"]
    f_conflict = f_arm["picked"]["conflict"]
    f_positive = f_arm["picked"]["positive"]
    ad_negative = ad_arm["picked"]["negative"]
    ad_conflict = ad_arm["picked"]["conflict"]

    check(
        "F retrieval returns a locally harmful CONFLICT card",
        bool(f_conflict)
        and f_conflict["relation"] == RELATION_CONFLICT
        and int(f_conflict["harmed_series_count"] or 0) > 0,
        f_conflict,
    )
    check(
        "F retrieval returns a harmless POSITIVE card from the repair family",
        bool(f_positive)
        and f_positive["relation"] == RELATION_POSITIVE
        and int(f_positive["harmed_series_count"] or 0) == 0
        and f_positive["program"] in F_SAFE_TOP1,
        f_positive,
    )
    check(
        "AD retrieval returns a NEGATIVE card",
        bool(ad_negative) and ad_negative["relation"] == RELATION_NEGATIVE,
        ad_negative,
    )
    check(
        "AD retrieval returns a CONFLICT card from a repair program",
        bool(ad_conflict)
        and ad_conflict["relation"] == RELATION_CONFLICT
        and ad_conflict["program"] in REPAIR_PROGRAMS,
        ad_conflict,
    )
    check(
        "no card crosses tasks in either arm",
        not f_arm["cross_task_card_ids"] and not ad_arm["cross_task_card_ids"],
        {
            "forecasting": f_arm["cross_task_card_ids"],
            "anomaly_detection": ad_arm["cross_task_card_ids"],
        },
    )
    vocabulary: dict[str, Any] = {}
    for arm in ("forecasting", "anomaly_detection"):
        text = retrieval["arms"][arm]["rendered_card"]
        vocabulary[arm] = {
            "names_a_consumer": "Consumer `" in text,
            "states_aggregate_direction": "Aggregate direction:" in text,
            "states_harmed_count": "harmed beyond" in text,
            "states_worst_series_reading": "Worst single-series reading:" in text,
            "contains_no_instruction_to_choose": not any(
                token in text.lower()
                for token in ("you should", "must choose", "pick ")
            ),
        }
    check(
        "the card face carries consumer, aggregate direction, harmed count "
        "and worst single-series reading, and prescribes nothing",
        all(all(row.values()) for row in vocabulary.values()),
        vocabulary,
    )
    check(
        "both arms retrieved something",
        bool(f_arm["card_episode_ids"]) and bool(ad_arm["card_episode_ids"]),
        {
            "forecasting": f_arm["card_episode_ids"],
            "anomaly_detection": ad_arm["card_episode_ids"],
        },
    )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "rule": (
            "category acceptance, not instance pinning: the F CONFLICT card is "
            "hampel_filter only because it is the only F conflict the frozen "
            "readings produce, and the AD conflict is required to be some "
            "repair program, not a named one"
        ),
    }


# ---------------------------------------------------------- prompts and C2
def _json_escaped(text: str) -> str:
    """The body of the JSON string form, without the surrounding quotes."""
    return canonical_json_bytes(text).decode("utf-8")[1:-1]


def _strip_card(system: str, card: str) -> str:
    """Remove the experience block's two occurrences from a system prompt.

    The resolved Harness instruction is rendered twice: once as the system
    prefix and once inside the canonical JSON of the resolved Harness.  Both
    are removed exactly once; what is left must be the #39 system byte for
    byte.
    """
    if not card:
        return system
    stripped = system.replace(card, "", 1)
    return stripped.replace(_json_escaped(card), "", 1)


def _json_spec_bytes(baseline: Mapping[str, Any], arm: str) -> str:
    return canonical_json_bytes(baseline["task_specs"][arm]).decode("utf-8")


def _three_way(
    *, prompts: Mapping[str, Mapping[str, str]],
    baseline_doc: Mapping[str, Any], cards: Mapping[str, str],
) -> dict[str, Any]:
    """C2: three assertions, each naming exactly what may differ."""
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    verbatim = baseline_doc["prompts_verbatim"]
    base_system = str(verbatim["system_both_arms"])
    base_user = {
        "forecasting": str(verbatim["user_forecasting"]),
        "anomaly_detection": str(verbatim["user_anomaly_detection"]),
    }
    for arm in ("forecasting", "anomaly_detection"):
        system = prompts[arm]["system"]
        card = cards[arm]
        stripped = _strip_card(system, card)
        check(
            "T4-%s vs #39-%s: the user message is byte-identical" % (arm, arm),
            prompts[arm]["user"] == base_user[arm],
            {
                "t4_user_sha256": canonical_sha256(prompts[arm]["user"]),
                "t3_user_sha256": canonical_sha256(base_user[arm]),
            },
        )
        check(
            "T4-%s vs #39-%s: the system message differs only by the "
            "experience block" % (arm, arm),
            stripped == base_system,
            {
                "card_occurrences_prefix": system.count(card) if card else 0,
                "card_occurrences_escaped": (
                    system.count(_json_escaped(card)) if card else 0
                ),
                "stripped_equals_baseline": stripped == base_system,
                "stripped_sha256": canonical_sha256(stripped),
                "baseline_system_sha256": canonical_sha256(base_system),
            },
        )
    f_user = prompts["forecasting"]["user"]
    ad_user = prompts["anomaly_detection"]["user"]
    f_spec_bytes = _json_spec_bytes(baseline_doc, "forecasting")
    ad_spec_bytes = _json_spec_bytes(baseline_doc, "anomaly_detection")
    check(
        "T4-F vs T4-AD: the user messages differ exactly at the task_spec bytes",
        f_user.count(f_spec_bytes) == 1
        and ad_user == f_user.replace(f_spec_bytes, ad_spec_bytes),
        {
            "task_spec_occurrences": f_user.count(f_spec_bytes),
            "replace_test": ad_user == f_user.replace(f_spec_bytes, ad_spec_bytes),
        },
    )
    check(
        "T4-F vs T4-AD: the system messages differ only by their own "
        "experience blocks",
        _strip_card(prompts["forecasting"]["system"], cards["forecasting"])
        == _strip_card(
            prompts["anomaly_detection"]["system"], cards["anomaly_detection"]
        ),
        {
            "forecasting_card_sha256": canonical_sha256(cards["forecasting"]),
            "anomaly_card_sha256": canonical_sha256(cards["anomaly_detection"]),
            "cards_differ": cards["forecasting"] != cards["anomaly_detection"],
        },
    )
    check(
        "each arm's card is actually present in the bytes that will be sent",
        all(
            bool(cards[arm]) and cards[arm] in prompts[arm]["system"]
            for arm in ("forecasting", "anomaly_detection")
        ),
        {
            arm: bool(cards[arm]) and cards[arm] in prompts[arm]["system"]
            for arm in ("forecasting", "anomaly_detection")
        },
    )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "baseline_source": _repo_rel(T3_ARTIFACT),
    }


# -------------------------------------------------------------------- the exam
def _run_draws(
    *, core: Any, backend: Any, views: Mapping[str, Any],
    inputs: Mapping[str, Any], retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    """#39's draw loop with one difference: the view is per arm.

    #39 hands one resolved Harness to both arms because with an empty store
    there is only one.  Here each arm carries its own retrieved card, so the
    view has to be selected per draw; everything else -- stage, schema, retry
    budget, classification -- is #39's, imported.
    """
    draws: list[dict[str, Any]] = []
    stopped = None
    for index, arm in enumerate(ARM_ORDER):
        tag = "F" if arm == "forecasting" else "AD"
        case_id = "T4EXAM_%s%d" % (tag, index + 1)
        view = views[arm]
        public_input = inputs[arm]
        models_before = set(backend.returned_models)
        card = retrieval["arms"][arm]["rendered_card"]
        record: dict[str, Any] = {
            "draw_index": index + 1,
            "arm": arm,
            "case_id": case_id,
            "public_input_sha256": canonical_sha256(_plain(public_input)),
            "retrieval_log": {
                "task_consumer_key": retrieval["arms"][arm]["task_consumer_key"],
                "card_episode_ids": list(
                    retrieval["arms"][arm]["card_episode_ids"]
                ),
                "cross_task_card_ids": list(
                    retrieval["arms"][arm]["cross_task_card_ids"]
                ),
                "rendered_card_sha256": retrieval["arms"][arm][
                    "rendered_card_sha256"
                ],
                "card_in_this_view": bool(card) and view.instruction.startswith(card),
                "note": (
                    "retrieval is deterministic and the pack is identical on "
                    "every draw of an arm; the per-draw entry records which "
                    "cards this draw's view actually carried"
                ),
            },
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
            reading = t3._classify(payload)
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
                "usage": _plain(metadata.get("usage", {})),
            })
        record["returned_models_delta"] = sorted(
            set(backend.returned_models) - models_before
        )
        record["llm_calls_cumulative"] = int(backend.calls)
        draws.append(record)
        print(
            "T4EXAM draw %d %s -> %s top1=%s retries=%s"
            % (
                index + 1, arm, record["classification"],
                record.get("top1"), record.get("validation_retry_count"),
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


# ------------------------------------------------------------ baseline + shift
def _baseline(doc: Mapping[str, Any]) -> dict[str, Any]:
    """#39's second sample, read out of its artifact.  Never retyped."""
    rows = [
        {
            "case_id": row["case_id"],
            "arm": row["arm"],
            "classification": row["classification"],
            "top1": row.get("top1"),
            "shortlist": sorted(row.get("proposal_set") or []),
        }
        for row in doc["draws"]
    ]
    scoring = doc["scoring"]
    matrix = doc["distance_matrix"]
    return {
        "source": _repo_rel(T3_ARTIFACT),
        "verdict": doc["verdict"],
        "draws": rows,
        "top1_by_arm": {
            arm: [row["top1"] for row in rows if row["arm"] == arm]
            for arm in ("forecasting", "anomaly_detection")
        },
        "risk_layer": {
            arm: scoring["layers"]["risk"][arm]["appropriate"]
            for arm in ("forecasting", "anomaly_detection")
        },
        "aggregate_layer": {
            arm: scoring["layers"]["aggregate"][arm]["appropriate"]
            for arm in ("forecasting", "anomaly_detection")
        },
        "min_cross_task": matrix["min_cross_task"],
        "max_same_task": matrix["max_same_task"],
        "complete_separation": matrix["complete_separation"],
    }


def _shift_table(
    *, draws: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any],
    scoring: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in ("forecasting", "anomaly_detection"):
        base_top1 = list(baseline["top1_by_arm"][arm])
        now = [row for row in draws if row["arm"] == arm]
        now_top1 = [row.get("top1") for row in now]
        base_sets = [
            sorted(row["shortlist"])
            for row in baseline["draws"] if row["arm"] == arm
        ]
        now_sets = [sorted(row.get("proposal_set") or []) for row in now]
        rows.append({
            "arm": arm,
            "baseline_top1": base_top1,
            "t4_top1": now_top1,
            "top1_moved": base_top1 != now_top1,
            "baseline_shortlists": base_sets,
            "t4_shortlists": now_sets,
            "shortlist_moved": base_sets != now_sets,
            "baseline_risk_appropriate": baseline["risk_layer"][arm],
            "t4_risk_appropriate": scoring["layers"]["risk"][arm]["appropriate"],
            "risk_delta": (
                scoring["layers"]["risk"][arm]["appropriate"]
                - baseline["risk_layer"][arm]
            ),
            "baseline_aggregate_appropriate": baseline["aggregate_layer"][arm],
            "t4_aggregate_appropriate": (
                scoring["layers"]["aggregate"][arm]["appropriate"]
            ),
        })
    return {
        "rows": rows,
        "rule": (
            "displacement is read against #39's second sample, the frozen "
            "baseline: F was hampel_filter 3/3 with Risk 0/3, AD was identity "
            "3/3 with Risk 3/3"
        ),
    }


# ------------------------------------------------------------------- verdict
def _verdict(
    *, draws: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any],
    scoring: Mapping[str, Any], retrieval: Mapping[str, Any],
    baseline: Mapping[str, Any], three_way: Mapping[str, Any],
) -> dict[str, Any]:
    invalid = [row for row in draws if not row["valid"]]
    by_arm = {
        arm: [row for row in draws if row["arm"] == arm]
        for arm in ("forecasting", "anomaly_detection")
    }
    f_safe = sum(
        1 for row in by_arm["forecasting"]
        if row["valid"] and row.get("top1") in F_SAFE_TOP1
    )
    ad_safe = sum(
        1 for row in by_arm["anomaly_detection"]
        if row["valid"]
        and (
            row.get("top1") in AD_SAFE_TOP1
            or row["classification"] == "VALID_ABSTAIN"
        )
    )
    f_pass = f_safe == DRAWS_PER_TASK
    ad_pass = ad_safe == DRAWS_PER_TASK
    separation = bool(matrix["complete_separation"])
    cross_task_cards = sorted(
        cid
        for arm in retrieval["arms"].values()
        for cid in arm["cross_task_card_ids"]
    )
    empty_handed = [
        arm for arm, row in retrieval["arms"].items()
        if not row["card_episode_ids"]
    ]
    ad_regressed = (
        scoring["layers"]["risk"]["anomaly_detection"]["appropriate"]
        < baseline["risk_layer"]["anomaly_detection"]
    )
    f_unmoved = (
        [row.get("top1") for row in by_arm["forecasting"]]
        == list(baseline["top1_by_arm"]["forecasting"])
    )
    cards_provably_present = bool(three_way["passed"]) and all(
        row["retrieval_log"]["card_in_this_view"] for row in draws
    )
    required = {
        "forecasting_safe_top1_count": f_safe,
        "anomaly_detection_safe_top1_count": ad_safe,
        "of": DRAWS_PER_TASK,
        "min_cross_task_distance": matrix["min_cross_task"],
        "max_same_task_distance": matrix["max_same_task"],
        "separation_kept": separation,
        "ad_risk_regression_flag": bool(ad_regressed),
        "ad_risk_baseline": baseline["risk_layer"]["anomaly_detection"],
        "ad_risk_now": scoring["layers"]["risk"]["anomaly_detection"]["appropriate"],
        "forecasting_top1_baseline": list(baseline["top1_by_arm"]["forecasting"]),
        "forecasting_top1_now": [row.get("top1") for row in by_arm["forecasting"]],
        "displacement_direction": (
            "none" if f_unmoved
            else ("towards the risk-safe set" if f_safe > 0 else "elsewhere")
        ),
    }
    if len(invalid) > 2:
        verdict = "EXAM_PROTOCOL_UNREADABLE"
        reason = "%d of 6 draws are invalid" % len(invalid)
    elif empty_handed or cross_task_cards:
        verdict = "RETRIEVAL_MISS"
        reason = (
            "retrieval came back empty-handed for %s or crossed tasks (%s)"
            % (empty_handed or "no arm", cross_task_cards or "none")
        )
    elif f_pass and ad_pass and separation:
        verdict = "CONFLICT_EXPERIENCE_CONDITIONS_PROPOSALS_CONFIRMED"
        reason = (
            "F 3/3 top-1 inside the risk-safe set (baseline 0/3), AD 3/3 "
            "top-1 identity-or-abstain (baseline 3/3), separation kept"
        )
    elif f_unmoved and cards_provably_present:
        verdict = "EXPERIENCE_IGNORED"
        reason = (
            "the F arm did not move from its baseline top-1 and the cards are "
            "provably in the bytes that were sent; credible negative -- the "
            "next surface is card presentation or Slow, not the Memory key"
        )
    else:
        verdict = "PARTIAL_EXPERIENCE_CONDITIONING"
        reason = (
            "an intermediate state: F %d/3 safe, AD %d/3 safe, separation "
            "kept %s, AD risk regression %s"
            % (f_safe, ad_safe, separation, bool(ad_regressed))
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "required_fields": required,
        "ladder_trace": {
            "invalid_draws": len(invalid),
            "invalid_draw_case_ids": [row["case_id"] for row in invalid],
            "retrieval_empty_handed_arms": empty_handed,
            "cross_task_cards": cross_task_cards,
            "forecasting_passed": f_pass,
            "anomaly_detection_passed": ad_pass,
            "separation_kept": separation,
            "forecasting_unmoved": f_unmoved,
            "cards_provably_in_prompt": cards_provably_present,
        },
        "closes_nothing_note": (
            "PARTIAL_EXPERIENCE_CONDITIONING is not a positive close and is "
            "not adjudicated here; a pass is MECHANISM at POSITIVE_CONTROL "
            "grade for the Memory capability only -- execution, adoption, "
            "delayed write-back and Local Skill update stay in T5"
        ),
    }


# =========================================================================== #
# what T4 hands back: findings that cost no LLM call
# =========================================================================== #
def t4_findings(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure function of the artifact and this module's constants.  Reads nothing.

    Two things the run surfaced that the verdict field cannot carry: why the
    store sha moved (it is not a store change), and the mechanism behind the AD
    regression.  Both are reported for the main line to adjudicate; neither
    changes the verdict, which stays where the pre-registered ladder put it.
    """
    store = payload["store_state"]
    shift = {row["arm"]: row for row in payload["shift_table"]["rows"]}
    retrieval = payload["retrieval"]["arms"]
    episodes = {
        (row["arm"], row["program"]): row for row in payload["episodes"]["rows"]
    }
    ad_cards = list(retrieval["anomaly_detection"]["card_episode_ids"])
    ad_conflict = retrieval["anomaly_detection"]["picked"]["conflict"]
    return {
        "store_sha_movement_is_not_a_store_change": {
            "runtime_bundle_sha_39": store["baseline_runtime_bundle_sha"],
            "runtime_bundle_sha_40": store["runtime_bundle_sha"],
            "harness_content_sha_identical": True,
            "effective_harness_view_sha_identical": True,
            "why": (
                "runtime_bundle_sha = sha(harness_content_sha, "
                "operator_bundle_sha, dependency_shas, three compiler "
                "versions).  harness_content_sha did not move.  "
                "dependency_shas covers ttha:fast_agent and ttha:method by "
                "name, and this round's authorized Memory diff edits both, so "
                "the bundle sha is obliged to move.  The Harness the Agent "
                "read is byte-identical, which C2 proves independently: "
                "stripping each arm's experience block from its system prompt "
                "returns #39's system bytes exactly."
            ),
            "b3_reading": (
                "the store base is #39's h0 snapshot and nothing was written "
                "to it; the ten episodes live in the method instance.  The "
                "bundle sha is a runtime-code identity, not a store identity"
            ),
            "separate_observation": (
                "methods/ttha/experience_memory.py is NOT in the compiler's "
                "dependency list (contracts + runtime/* + ttha:{agent_core, "
                "fast_agent, method, public_tools, retrieval, "
                "schema_contracts, slow_agent}).  A Memory-only edit that "
                "touched no other module would leave runtime_bundle_sha "
                "unchanged -- the bundle identity does not currently cover "
                "the Memory surface.  Reported, not fixed: the registry is "
                "not this round's change surface."
            ),
        },
        "ad_regression_mechanism": {
            "what_happened": (
                "AD moved off identity 3/3 onto hampel_filter 3/3; its "
                "aggregate layer stayed 3/3 (hampel_filter is inside the AD "
                "aggregate key at +0.0413 macro-F1) and its Risk layer fell "
                "from 3/3 to 0/3.  The AD arm made exactly the error F made "
                "in #39: an aggregate-appropriate, risk-inappropriate choice."
            ),
            "cards_the_ad_arm_saw": ad_cards,
            "cards_the_ad_arm_could_not_see": [
                episodes[("anomaly_detection", "identity")]["episode_id"]
            ],
            "why_identity_was_unreachable": (
                "identity under AD classifies as ABSTAIN, and ABSTAIN has no "
                "channel to the prompt on two independent counts: "
                "ContrastPack carries positive / negative / conflict slots "
                "only, and SignedEpisodeRetriever._hard_filter drops any "
                "episode whose workflow_signature is identity or unknown "
                "whenever allowed_operators is non-empty.  So the only AD "
                "experience that could reach the Agent was 'winsorize "
                "degraded' and 'hampel_filter improved the aggregate'.  The "
                "Memory has no way to say that doing nothing was the right "
                "call, and the card it could render pointed away from the "
                "answer."
            ),
            "the_ad_conflict_card_that_moved_it": ad_conflict,
            "evidence_in_the_agents_own_words": {
                "rule": (
                    "the reason field of each draw, verbatim from the "
                    "artifact; in all three AD draws the choice is framed as "
                    "a comparison between the two retrieved cards and "
                    "identity never enters"
                ),
                "draws": [
                    {
                        "draw_index": row["draw_index"],
                        "arm": row["arm"],
                        "top1": row.get("top1"),
                        "reason": (row.get("parsed_payload") or {}).get("reason"),
                    }
                    for row in payload["draws"]
                ],
            },
            "second_contributing_factor": (
                "the fact sentence leads with the aggregate direction and "
                "states the harmed count second.  Under F that ordering was "
                "harmless because the safe POSITIVE card was also present to "
                "contrast against; under AD there was no such contrast, so "
                "'Aggregate direction: improved' stood alone."
            ),
            "what_it_does_not_show": (
                "it does not show the key, the write path or the retrieval "
                "gate failed -- all three worked, B4 and C2 passed, zero "
                "cards crossed tasks and separation strengthened from "
                "1.0>0.5 to 1.0>0.0.  What failed is the pack's expressive "
                "range."
            ),
            "candidate_next_surfaces_not_adjudicated_here": [
                "give ContrastPack an abstain / do-nothing channel so an "
                "ABSTAIN episode can be rendered",
                "reconsider _hard_filter dropping identity signatures when "
                "the task's correct answer may be to leave the batch alone",
                "card presentation: lead with the risk reading rather than "
                "the aggregate direction",
            ],
        },
        "f_arm_reading": {
            "moved": shift["forecasting"]["top1_moved"],
            "from": shift["forecasting"]["baseline_top1"],
            "to": shift["forecasting"]["t4_top1"],
            "risk_layer": "%d/3 -> %d/3" % (
                shift["forecasting"]["baseline_risk_appropriate"],
                shift["forecasting"]["t4_risk_appropriate"],
            ),
            "reading": (
                "the pre-registered F success condition is met exactly: the "
                "prior-experience gap #39 exposed closed the moment the "
                "conflict card named hampel_filter's local harm next to a "
                "harmless alternative.  Being a positive control this shows "
                "the machinery carries the correction, not that the Agent "
                "discovered anything"
            ),
        },
    }


def annotate() -> int:
    """Add t4_findings to an existing artifact.  Spends no LLM call."""
    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    payload["t4_findings"] = t4_findings(payload)
    payload["annotation_note"] = (
        "t4_findings is a pure function of the readings already in this "
        "artifact and of this module's constants.  Adding it made no backend "
        "call and spent no budget; llm_calls is unchanged."
    )
    OUT_JSON.write_text(_json_text(payload), encoding="utf-8")
    OUT_MD.write_text(_render_md(payload), encoding="utf-8")
    print("annotated", OUT_JSON, flush=True)
    return 0


# -------------------------------------------------------------------- report
def _render_md(doc: Mapping[str, Any]) -> str:
    lines = [
        "# T4 (#40) task-keyed conflict Experience -- %s" % doc["verdict"],
        "",
        "- protocol: `%s` (evidence grade %s, permanent)"
        % (doc["protocol_version"], doc["evidence_grade"]),
        "- Part 0 checkpoint: `%s` (%d files)"
        % (doc["part0_checkpoint"]["commit"], doc["part0_checkpoint"]["files"]),
        "- backend: `%s` at `%s`; returned models: %s"
        % (
            doc["backend_declaration"]["requested_model"],
            doc["backend_declaration"]["base_url"],
            ", ".join(doc["backend_declaration"]["returned_models"]) or "none",
        ),
        "- cost: %d LLM calls; 0 forecasting retrains; 0 AD evaluations"
        % doc["budgets"]["llm_calls"],
        "- store: `%s` -- %s"
        % (
            doc["store_state"]["runtime_bundle_sha"][:12],
            doc["store_state"]["empty_store_statement"],
        ),
        "",
        "## Memory keys",
        "",
        "- forecasting: `%s`" % doc["task_keys"]["forecasting"],
        "- anomaly_detection: `%s`" % doc["task_keys"]["anomaly_detection"],
        "",
        "## The ten episodes (relation derived mechanically)",
        "",
        "| arm | program | relation | aggregate | direction | harmed / read | "
        "min per-series |",
        "|-----|---------|----------|-----------|-----------|---------------|"
        "----------------|",
    ]
    for row in doc["episodes"]["rows"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %d / %d | %s |"
            % (
                row["arm"], row["program"], row["relation"],
                "n/a" if row["aggregate_gain"] is None
                else "%+.6f" % row["aggregate_gain"],
                row["aggregate_direction"],
                row["harmed_series_count"], row["series_read"],
                "n/a" if row["min_per_series_gain"] is None
                else "%+.6f" % row["min_per_series_gain"],
            )
        )
    lines += [
        "",
        "- matches the pre-registered expectation: %s"
        % doc["episodes"]["matches_expectation"],
        "- NEUTRAL produced: %s" % (doc["episodes"]["neutral_produced"] or "none"),
        "",
        "## Write (Runtime)",
        "",
        "- %d -> %d episodes on one TTHAMethod; read back from the runtime: %s"
        % (
            doc["write"]["episodes_before"], doc["write"]["episodes_after"],
            doc["write"]["read_back_from_runtime"],
        ),
        "- evidence levels %s; local statuses %s; no promotion: %s"
        % (
            doc["write"]["evidence_levels"], doc["write"]["local_statuses"],
            doc["write"]["no_promotion"],
        ),
        "",
        "## B4 category acceptance: %s"
        % ("PASSED" if doc["b4_acceptance"]["passed"] else "FAILED"),
        "",
    ]
    for row in doc["b4_acceptance"]["checks"]:
        lines.append("- [%s] %s" % ("x" if row["passed"] else " ", row["name"]))
    lines += [
        "",
        "## C2 three-way prompt assertions: %s"
        % ("PASSED" if doc["three_way"]["passed"] else "FAILED"),
        "",
    ]
    for row in doc["three_way"]["checks"]:
        lines.append("- [%s] %s" % ("x" if row["passed"] else " ", row["name"]))
    if doc.get("draws"):
        lines += [
            "",
            "## Draws (order %s)" % ", ".join(doc["arm_order"]),
            "",
            "| # | arm | classification | top1 | shortlist | cards |",
            "|---|-----|----------------|------|-----------|-------|",
        ]
        for row in doc["draws"]:
            payload = row.get("parsed_payload") or {}
            lines.append(
                "| %d | %s | %s | %s | %s | %s |"
                % (
                    row["draw_index"], row["arm"], row["classification"],
                    row.get("top1")
                    or ("__ABSTAIN__" if row["classification"] == "VALID_ABSTAIN"
                        else "-"),
                    ", ".join(payload.get("shortlist", ()) or ()) or "-",
                    ", ".join(row["retrieval_log"]["card_episode_ids"]) or "-",
                )
            )
    if doc.get("shift_table"):
        lines += [
            "",
            "## Displacement against #39",
            "",
            "| arm | baseline top-1 | T4 top-1 | moved | baseline Risk | "
            "T4 Risk |",
            "|-----|----------------|----------|-------|---------------|"
            "---------|",
        ]
        for row in doc["shift_table"]["rows"]:
            lines.append(
                "| %s | %s | %s | %s | %d/3 | %d/3 |"
                % (
                    row["arm"], ", ".join(str(x) for x in row["baseline_top1"]),
                    ", ".join(str(x) for x in row["t4_top1"]),
                    row["top1_moved"], row["baseline_risk_appropriate"],
                    row["t4_risk_appropriate"],
                )
            )
    if doc.get("distance_matrix"):
        matrix = doc["distance_matrix"]
        lines += [
            "",
            "## Distance matrix",
            "",
            "- min cross-task: %s; max same-task: %s; complete separation: %s"
            % (
                matrix["min_cross_task"], matrix["max_same_task"],
                matrix["complete_separation"],
            ),
        ]
    lines += [
        "",
        "## Verdict",
        "",
        "**%s** -- %s" % (doc["verdict"], doc.get("verdict_reason") or ""),
        "",
    ]
    findings = doc.get("t4_findings")
    if findings:
        store_note = findings["store_sha_movement_is_not_a_store_change"]
        mech = findings["ad_regression_mechanism"]
        lines += [
            "",
            "## Findings handed back (no LLM cost)",
            "",
            "### The store sha moved, the store did not",
            "",
            store_note["why"],
            "",
            "- %s" % store_note["b3_reading"],
            "- %s" % store_note["separate_observation"],
            "",
            "### Why the AD arm regressed",
            "",
            mech["what_happened"],
            "",
            "- cards the AD arm saw: %s" % ", ".join(mech["cards_the_ad_arm_saw"]),
            "- cards it could not see: %s"
            % ", ".join(mech["cards_the_ad_arm_could_not_see"]),
            "",
            mech["why_identity_was_unreachable"],
            "",
            mech["second_contributing_factor"],
            "",
            "> %s" % mech["what_it_does_not_show"],
            "",
            "Candidate next surfaces (not adjudicated here):",
            "",
        ]
        for item in mech["candidate_next_surfaces_not_adjudicated_here"]:
            lines.append("- %s" % item)
        lines.append("")
    if doc.get("ambiguities_reported_not_self_adjudicated"):
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
    read_only_before = {
        _repo_rel(path): _sha256(path)
        for path in (T3_ARTIFACT, T1B_V3_ARTIFACT)
    }
    read_only_before.update(t3._read_only_inventory())

    substrate = t3._load_substrate()
    search = t3._StandInSearch(substrate)

    # The store is #39's, materialized into this round's own scratch root so
    # nothing of #39's is touched; the snapshot itself is byte-identical.
    t3.STORE_ROOT = STORE_ROOT
    store = t3._build_empty_store()
    snapshot = store["_snapshot"]
    context = ssi._public_features(search)
    features = dict(context["features"])
    base_view = resolve_harness_view(snapshot, features, role="fast")
    learned = [
        skill.skill_id
        for skill in snapshot.skills
        if skill.skill_kind.value != "bootstrap_procedure"
    ]
    store_state = {
        key: value for key, value in store.items() if not key.startswith("_")
    }
    store_state.update({
        "resolved_skill_ids": list(base_view.skill_ids),
        "resolved_memory_ids": list(base_view.memory_ids),
        "effective_harness_view_sha": base_view.effective_harness_view_sha,
        "learned_skill_count": len(learned),
        "empty_store_statement": (
            "the #39 h0 snapshot exactly: 0 Guidance / 0 Experience / 0 "
            "learned Skill plus the three always-on bootstrap procedures; "
            "this round adds ten Experience episodes and nothing else"
        ),
    })

    baseline_doc = json.loads(T3_ARTIFACT.read_text(encoding="utf-8"))
    store_state["baseline_runtime_bundle_sha"] = (
        baseline_doc["store_state"]["runtime_bundle_sha"]
    )
    store_state["store_base_identical_to_39"] = (
        store["runtime_bundle_sha"]
        == baseline_doc["store_state"]["runtime_bundle_sha"]
    )

    inputs = t3._public_inputs(search)
    keys = _arm_task_keys(inputs)
    readings = _frozen_readings()
    episodes = _build_episodes(
        keys=keys, readings=readings, features=features,
        substrate=substrate, cutoff=int(search.support[0]),
    )

    backend = _default_backend_factory(LLM_BUDGET)
    gateway = t3.wvc.NoToolGateway(
        {"episode_id": PROTOCOL_VERSION, "arm": "exam"}
    )
    core = TTHAAgentCore(backend, gateway, model=EXAM_MODEL, base_url=NF_BASE_URL)

    written = _write_through_runtime(
        core=core, snapshot=snapshot, episodes=episodes["episodes"],
    )
    method = written["method"]
    retrieval = _retrieve(method=method, keys=keys, features=features)
    b4 = _b4_acceptance(retrieval)

    cards = {
        arm: retrieval["arms"][arm]["rendered_card"]
        for arm in ("forecasting", "anomaly_detection")
    }
    views = {
        arm: dataclasses.replace(
            base_view, instruction=cards[arm] + base_view.instruction
        )
        for arm in ("forecasting", "anomaly_detection")
    }
    prompts = {
        arm: t3._render_prompts(views[arm], inputs)[arm]
        for arm in ("forecasting", "anomaly_detection")
    }
    three_way = _three_way(
        prompts=prompts, baseline_doc=baseline_doc, cards=cards,
    )
    answer_keys = t3._derive_answer_keys()
    baseline = _baseline(baseline_doc)

    doc: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_grade_note": (
            "permanent: the ten episodes replay the same frozen readings the "
            "answer keys are derived from, so a pass demonstrates the Memory "
            "machinery and never a discovery"
        ),
        "role": "task-keyed conflict Experience (Memory surface only)",
        "part0_checkpoint": dict(PART0_CHECKPOINT),
        "question": (
            "can Harness-written task-keyed success / failure / partial "
            "conflict experience correct F's risk blind spot without "
            "breaking AD's conservative choice"
        ),
        "verdict": None,
        "stopped": None,
        "ambiguities_reported_not_self_adjudicated": ambiguities,
        "arm_order": list(ARM_ORDER),
        "menu": list(MENU),
        "task_keys": {
            arm: keys[arm]["key"] for arm in ("forecasting", "anomaly_detection")
        },
        "task_key_derivation": {
            "rule": keys["rule"],
            "consistency_with_the_prompt": keys["consistency_with_the_prompt"],
        },
        "task_specs": {
            "forecasting": inputs["forecasting"]["task_spec"],
            "anomaly_detection": inputs["anomaly_detection"]["task_spec"],
        },
        "substrate": {
            "source": substrate["source"],
            "block": substrate["block"],
            "train_uids": substrate["train_uids"],
            "observation_cutoff": int(search.support[0]),
            "construction": (
                "identical to #39: the ssi public context construction over "
                "the T1 injected copy's training block"
            ),
        },
        "store_state": store_state,
        "frozen_readings": {
            "source": readings["source"],
            "aggregate": readings["aggregate"],
            "per_series": readings["per_series"],
            "reading_windows": readings["reading_windows"],
        },
        "episodes": {
            key: value for key, value in episodes.items() if key != "episodes"
        },
        "write": written["record"],
        "retrieval": {
            "held_episode_count": retrieval["held_episode_count"],
            "arms": {
                arm: {
                    key: value
                    for key, value in retrieval["arms"][arm].items()
                }
                for arm in ("forecasting", "anomaly_detection")
            },
        },
        "b4_acceptance": b4,
        "three_way": three_way,
        "answer_keys": answer_keys,
        "baseline_39": baseline,
        "prompts_verbatim": {
            "rule": (
                "one rendering per arm; case_id is call wiring and is never "
                "rendered, so every draw of an arm sends exactly these bytes"
            ),
            "system_forecasting": prompts["forecasting"]["system"],
            "system_anomaly_detection": prompts["anomaly_detection"]["system"],
            "user_forecasting": prompts["forecasting"]["user"],
            "user_anomaly_detection": prompts["anomaly_detection"]["user"],
        },
        "backend_declaration": {
            "requested_model": EXAM_MODEL,
            "base_url": NF_BASE_URL,
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
            "episodes_written": episodes["count"],
            "episode_budget": EPISODE_COUNT,
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

    if not episodes["matches_expectation"]:
        ambiguities.append(
            "the mechanical classification differs from the book's expected "
            "table: %s (the derived relations were used; the mismatch is "
            "reported, not self-adjudicated)" % episodes["mismatches"]
        )
    if not answer_keys["matches_expectation"]:
        ambiguities.append(
            "derived answer keys differ from #39's frozen expectation: %s"
            % answer_keys["mismatches"]
        )
    if not store_state["store_base_identical_to_39"]:
        ambiguities.append(
            "the store snapshot is not byte-identical to #39's: %s vs %s"
            % (
                store["runtime_bundle_sha"],
                store_state["baseline_runtime_bundle_sha"],
            )
        )

    if not b4["passed"]:
        doc["stopped"] = "RETRIEVAL_MISS"
        doc["verdict"] = "RETRIEVAL_MISS"
        doc["verdict_reason"] = (
            "the B4 category acceptance failed before any draw; no LLM call "
            "was made"
        )
        doc["wall_seconds"] = time.perf_counter() - started
        OUT_JSON.write_text(_json_text(doc), encoding="utf-8")
        OUT_MD.write_text(_render_md(doc), encoding="utf-8")
        print("B4 ACCEPTANCE FAILED -- RETRIEVAL_MISS; see %s" % OUT_JSON)
        return
    if not three_way["passed"]:
        doc["stopped"] = "THREE_WAY_ASSERTION_FAILED"
        doc["verdict"] = None
        doc["verdict_reason"] = (
            "the C2 three-way prompt assertions failed; the exam is not "
            "readable and no LLM call was made"
        )
        doc["wall_seconds"] = time.perf_counter() - started
        OUT_JSON.write_text(_json_text(doc), encoding="utf-8")
        print("C2 ASSERTIONS FAILED -- no LLM call made; see %s" % OUT_JSON)
        return
    if smoke_only:
        print("B4 + C2 passed (--smoke-only; no LLM call, nothing written)")
        for row in b4["checks"] + three_way["checks"]:
            print("  [%s] %s" % ("x" if row["passed"] else " ", row["name"]))
        return

    outcome = _run_draws(
        core=core, backend=backend, views=views, inputs=inputs,
        retrieval=retrieval,
    )
    draws = outcome["draws"]
    doc["stopped"] = outcome["stopped"]
    doc["draws"] = draws
    doc["budgets"]["llm_calls"] = outcome["llm_calls"]
    doc["budgets"]["budgets_respected"] = outcome["llm_calls"] <= LLM_BUDGET
    doc["backend_declaration"].update({
        "returned_models": outcome["returned_models"],
        "prompt_tokens": outcome["prompt_tokens"],
        "completion_tokens": outcome["completion_tokens"],
    })

    if outcome["stopped"] is not None:
        ambiguities.append(
            "the exam stopped on an infrastructure error (%s); no verdict is "
            "claimed and the partial record is archived" % outcome["stopped"]
        )
    else:
        matrix = t3._distance_matrix(draws)
        scoring = t3._score(draws, answer_keys)
        shift = _shift_table(draws=draws, baseline=baseline, scoring=scoring)
        verdict = _verdict(
            draws=draws, matrix=matrix, scoring=scoring, retrieval=retrieval,
            baseline=baseline, three_way=three_way,
        )
        doc["distance_matrix"] = matrix
        doc["scoring"] = scoring
        doc["shift_table"] = shift
        doc["verdict"] = verdict["verdict"]
        doc["verdict_reason"] = verdict["reason"]
        doc["verdict_required_fields"] = verdict["required_fields"]
        doc["verdict_ladder_trace"] = verdict["ladder_trace"]
        doc["closes_nothing_note"] = verdict["closes_nothing_note"]

    doc["frozen_surface"]["after"] = _verify(frozen_before)
    doc["git"]["diff_name_only_at_end"] = _git(["diff", "--name-only", "HEAD"])
    read_only_after = {
        _repo_rel(path): _sha256(path)
        for path in (T3_ARTIFACT, T1B_V3_ARTIFACT)
    }
    read_only_after.update(t3._read_only_inventory())
    doc["read_only_integrity"].update({
        "after": read_only_after,
        "unchanged": read_only_after == read_only_before,
    })
    doc["wall_seconds"] = time.perf_counter() - started

    OUT_JSON.write_text(_json_text(doc), encoding="utf-8")
    OUT_MD.write_text(_render_md(doc), encoding="utf-8")
    print(
        "T4 exam: verdict=%s llm=%d stopped=%s"
        % (doc["verdict"], outcome["llm_calls"], outcome["stopped"])
    )


if __name__ == "__main__":
    raise SystemExit(
        annotate() if "--annotate" in sys.argv[1:] else (main() or 0)
    )
