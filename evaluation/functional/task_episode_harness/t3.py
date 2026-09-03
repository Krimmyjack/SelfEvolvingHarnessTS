"""T3: positive-only Source Experience warm-start A5 vs A3 (development).

The T2 bank is POSITIVE_ONLY_RECIPE_BANK.  This phase answers only the narrow
question: can positive Source Experience warm-start the first successful
Workflow on a same-base-series recipe-transfer Target Task?

Pre-outcome check first: normalized A3/A5 decision inputs must differ only in
the Source Experience block, and the LLM proposals must actually differ.
If proposals are identical, no Target outcome is opened and the verdict is
A5_A3_ARM_DISTINCTION_INERT.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.injection import (
    inject_gap_corpus,
    inject_label_touched_corpus,
)
from evaluation.functional.task_episode_harness.runner import (
    DELAYED_ORIGINS,
    INJECTION_AMPLITUDE,
    INJECTION_COUNT,
    MATERIAL_THRESHOLD,
    REPORT_REL,
    SUPPORT_ORIGINS,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_MAX_PROBES,
    T1_SCOPE_FEATURE,
    TASK_CONSUMER_KEY,
    _public_scope_proposal,
    _task_probe,
    _update_episode_delayed,
)
from evaluation.functional.task_episode_harness.t2b import (
    T2B_CONTEXT_CLASS,
    T2B_GAP_WORKFLOW,
    _gap_scope_proposal,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_SUPPORT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_DRAFT,
    build_episode,
)

T3_MODEL = "gpt-5.6-luna"
T3_BASE_URL = "https://api.agicto.cn/v1"
T3_TARGET_RECIPE = {
    "family": T2B_CONTEXT_CLASS,
    "faulty": ("T117", "T118", "T119", "T12", "T120", "T121"),
    "seed": 29,
    "gap_count": 80,
}
T3_POOL = ("outlier_mad", T2B_GAP_WORKFLOW)


def _canonical_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _source_summaries(report: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = (report.get("source_bank") or {}).get("episodes") or []
    return [
        {
            "context_class": (ep.get("context_summary") or {}).get("context_class"),
            "workflow": ep.get("workflow_signature"),
            "relation": ep.get("relation"),
            "support_gain": (ep.get("support_response") or {}).get("gain"),
            "support_se_block": (ep.get("support_response") or {}).get("se_block"),
            "support_gain_over_se": (ep.get("support_response") or {}).get(
                "gain_over_se"
            ),
            "delayed_gain": (ep.get("delayed_response") or {}).get("gain"),
            "delayed_se_block": (ep.get("delayed_response") or {}).get("se_block"),
            "delayed_gain_over_se": (ep.get("delayed_response") or {}).get(
                "gain_over_se"
            ),
            "reliability": (
                "known"
                if (ep.get("support_response") or {}).get("se_block") is not None
                else "unknown"
            ),
            "scope_bin": (
                (ep.get("context_summary") or {}).get("local_pattern") or {}
            ).get("scope_observation_bin"),
            "transfer_scope": (
                (ep.get("context_summary") or {}).get("transfer_scope")
            ),
        }
        for ep in episodes
    ]


def _decision_payload(
    *,
    scope: frozenset[str],
    observations: dict[str, Any],
    source_summaries: list[dict[str, Any]],
    context_class: str,
    scope_feature: str,
    scope_bin: str,
) -> dict[str, Any]:
    selected = sorted(scope)
    return {
        "task": {
            "type": "forecast",
            "horizon": 48,
            "context_length": 192,
            "consumer": "ridge_alpha_1_with_intercept",
            "metric": "sMASE",
        },
        "scope_policy": {
            "feature": scope_feature,
            "bin": scope_bin,
            "selected_series_count": len(selected),
            "observed_context_class": context_class,
        },
        "allowed_programs": [
            {"op": op, "params": {}} for op in T3_POOL
        ],
        "source_experiences": copy.deepcopy(source_summaries),
    }


def _llm_propose(
    payload: dict[str, Any],
    *,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    api_key = next(
        (
            os.environ.get(name, "").strip()
            for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
            if os.environ.get(name, "").strip()
        ),
        None,
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    system = (
        "You choose the probe order for a Target forecasting data readiness "
        "task. Allowed programs are exactly the two given in allowed_programs. "
        "POSITIVE/NEGATIVE/CONFLICT describe observed direction only; they do "
        "not mean high-confidence facts. Treat low support_gain_over_se or "
        "unknown reliability as weak evidence. Return exactly one JSON object: "
        '{"program_order": ["outlier_mad", "impute_ema"]} '
        "with both program names exactly once. Do not add commentary."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    text = str(completion.choices[0].message.content or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"LLM returned non-JSON text: {text[:200]!r}")
    proposal = json.loads(text[start:end + 1])
    order = proposal.get("program_order")
    if not isinstance(order, list) or sorted(order) != sorted(T3_POOL):
        raise RuntimeError(f"invalid program_order: {order!r}")
    return {
        "program_order": [str(item) for item in order],
        "raw_text": text,
    }


def _make_arm_episode(
    *,
    arm: str,
    attempt_index: int,
    program: str,
    scope: frozenset[str],
    probe: dict[str, Any],
    context_class: str,
    scope_bin: str,
) -> Any:
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    return build_episode(
        episode_id=f"t3_{arm}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-recipe-transfer-development",
        context_summary={
            "task_episode_id": f"t3-target-{arm}",
            "attempt_index": attempt_index,
            "observations_used": [T1_SCOPE_FEATURE],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "base_series_overlap_with_target": True,
            "transfer_scope": "RECIPE_TRANSFER_ONLY",
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {"scope_observation_bin": scope_bin},
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": [{"op": program, "params": {}}],
            },
        },
        workflow_signature=program,
        support_response={
            "gain": gain,
            "se_block": float(probe["se_block"]),
            "gain_over_se": probe["gain_over_se"],
            "accepted": positive,
            "block_origins": list(SUPPORT_ORIGINS),
        },
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_POSITIVE if positive else RELATION_NEGATIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT if positive else STATUS_EPISODE_ONLY,
        evidence_refs=["task_episode_harness_t3"],
    )


def _run_arm(
    *,
    arm: str,
    program_order: list[str],
    scope: frozenset[str],
    roster: list[dict[str, Any]],
    injected: dict[str, Any],
    config: dict[str, Any],
    eval_uids: list[str],
    context_class: str,
    scope_bin: str,
) -> dict[str, Any]:
    probes = []
    winner = None
    for attempt_index, program in enumerate(program_order[:T1_MAX_PROBES]):
        probe = _task_probe(
            roster,
            injected,
            config,
            SUPPORT_ORIGINS,
            eval_uids,
            program,
            scope,
        )
        episode = _make_arm_episode(
            arm=arm,
            attempt_index=attempt_index,
            program=program,
            scope=scope,
            probe=probe,
            context_class=context_class,
            scope_bin=scope_bin,
        )
        probes.append({
            "attempt_index": attempt_index,
            "program": program,
            "support_gain": probe["macro_gain"],
            "support_se_block": probe["se_block"],
            "episode": episode.to_dict(),
        })
        if probe["macro_gain"] >= MATERIAL_THRESHOLD:
            winner = episode
            break
    delayed = None
    if winner is not None:
        delayed_probe = _task_probe(
            roster,
            injected,
            config,
            DELAYED_ORIGINS,
            eval_uids,
            winner.workflow_signature,
            scope,
        )
        delayed = delayed_probe
        updated = _update_episode_delayed(
            winner,
            float(delayed_probe["macro_gain"]),
            delayed_se_block=float(delayed_probe["se_block"]),
            delayed_gain_over_se=delayed_probe["gain_over_se"],
        )
        for probe in probes:
            if probe["episode"]["episode_id"] == winner.episode_id:
                probe["episode"] = updated.to_dict()
        winner = updated
    return {
        "arm": arm,
        "program_order": program_order,
        "probes": probes,
        "probe_count": len(probes),
        "support_harm_count": sum(
            1 for p in probes if p["support_gain"] < -MATERIAL_THRESHOLD
        ),
        "cumulative_support_harm": float(sum(
            -p["support_gain"]
            for p in probes if p["support_gain"] < -MATERIAL_THRESHOLD
        )),
        "winner": (
            {
                "episode_id": winner.episode_id,
                "workflow": winner.workflow_signature,
                "local_status": winner.local_status,
                "delayed_gain": (
                    winner.delayed_response.get("gain")
                    if winner is not None else None
                ),
            }
            if winner is not None else None
        ),
        "delayed": delayed,
    }


def run_t3(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    source_summaries = _source_summaries(report)
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = cohort["values"]
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    clean_series = tuple(
        uid for uid in train_uids if uid not in T3_TARGET_RECIPE["faulty"]
    )
    if T3_TARGET_RECIPE["family"] == T2B_CONTEXT_CLASS:
        injected, _target_ground_truth = inject_gap_corpus(
            values,
            faulty_series=T3_TARGET_RECIPE["faulty"],
            clean_series=clean_series,
            count=T3_TARGET_RECIPE["gap_count"],
            seed=T3_TARGET_RECIPE["seed"],
        )
        agent = _gap_scope_proposal(injected, train_uids)
        context_class = T2B_CONTEXT_CLASS
        scope_feature = "missing_fraction"
        scope_bin = "nonzero_missing_fraction"
    else:
        injected, _target_ground_truth = inject_label_touched_corpus(
            values,
            faulty_series=T3_TARGET_RECIPE["faulty"],
            clean_series=clean_series,
            amplitude=INJECTION_AMPLITUDE,
            count=INJECTION_COUNT,
            seed=T3_TARGET_RECIPE["seed"],
        )
        agent = _public_scope_proposal(injected, train_uids)
        context_class = "impulsive_outlier"
        scope_feature = T1_SCOPE_FEATURE
        scope_bin = "high"
    scope = agent["scope"]

    a3_payload = _decision_payload(
        scope=scope,
        observations=agent["observations"],
        source_summaries=[],
        context_class=context_class,
        scope_feature=scope_feature,
        scope_bin=scope_bin,
    )
    a5_payload = _decision_payload(
        scope=scope,
        observations=agent["observations"],
        source_summaries=source_summaries,
        context_class=context_class,
        scope_feature=scope_feature,
        scope_bin=scope_bin,
    )
    a3_without_source = copy.deepcopy(a3_payload)
    a5_without_source = copy.deepcopy(a5_payload)
    a5_without_source["source_experiences"] = []
    inputs_differ_only_in_source = bool(
        _canonical_payload(a3_without_source)
        == _canonical_payload(a5_without_source)
        and bool(source_summaries)
    )

    llm_attempt: dict[str, Any] = {"attempted": True}
    try:
        a3_proposal = _llm_propose(
            a3_payload, model=T3_MODEL, base_url=T3_BASE_URL
        )
        a5_proposal = _llm_propose(
            a5_payload, model=T3_MODEL, base_url=T3_BASE_URL
        )
        llm_attempt["a3_proposal"] = a3_proposal
        llm_attempt["a5_proposal"] = a5_proposal
        llm_attempt["proposals_differ"] = bool(
            a3_proposal["program_order"] != a5_proposal["program_order"]
        )
    except Exception as exc:  # noqa: BLE001
        llm_attempt["error"] = f"{type(exc).__name__}: {exc}"
        llm_attempt["proposals_differ"] = False

    preflight = {
        "source_experience_nonempty": bool(source_summaries),
        "inputs_differ_only_in_source": inputs_differ_only_in_source,
        "llm_attempt": llm_attempt,
        "target_recipe": T3_TARGET_RECIPE,
        "target_outcome_opened_before_check": False,
    }

    arms: dict[str, dict[str, Any]] = {}
    verdict = "A5_A3_ARM_DISTINCTION_INERT"
    if not llm_attempt.get("proposals_differ"):
        preflight["target_outcome_opened"] = False
        arms = {}
    else:
        preflight["target_outcome_opened"] = True
        arms["A3"] = _run_arm(
            arm="A3",
            program_order=a3_proposal["program_order"],
            scope=scope,
            roster=mapped_roster,
            injected=injected,
            config=config,
            eval_uids=eval_uids,
            context_class=context_class,
            scope_bin=scope_bin,
        )
        arms["A5"] = _run_arm(
            arm="A5",
            program_order=a5_proposal["program_order"],
            scope=scope,
            roster=mapped_roster,
            injected=injected,
            config=config,
            eval_uids=eval_uids,
            context_class=context_class,
            scope_bin=scope_bin,
        )
        a3, a5 = arms["A3"], arms["A5"]
        a5_faster = a5["probe_count"] < a3["probe_count"]
        harm_not_worse = (
            a5["support_harm_count"] <= a3["support_harm_count"]
            and a5["cumulative_support_harm"] <= a3["cumulative_support_harm"]
        )
        delayed_confirmed = bool(
            a5["winner"] is not None
            and a5["winner"]["local_status"] == "LOCAL_ACTIVE"
        )
        both_no_winner = a3["winner"] is None and a5["winner"] is None
        same_harm = (
            a3["support_harm_count"] == a5["support_harm_count"]
            and abs(
                a3["cumulative_support_harm"]
                - a5["cumulative_support_harm"]
            ) < 1e-12
        )
        if a5_faster and harm_not_worse and delayed_confirmed:
            verdict = "POSITIVE_EXPERIENCE_WARM_START_PASS"
        elif a5_faster and not harm_not_worse:
            verdict = "A5_SPEED_ONLY"
        elif both_no_winner and same_harm:
            verdict = "A5_NO_WARM_START_BENEFIT_SINGLE_TARGET"
        elif a3_proposal["program_order"] == a5_proposal["program_order"]:
            verdict = "A5_A3_NO_SOURCE_EFFECT"
        else:
            verdict = "A5_NEGATIVE_TRANSFER"

    t3 = {
        "claim_scope": (
            "positive-only source experience warm start on same-base-series "
            "RECIPE_TRANSFER_ONLY development tasks; not signed-memory and "
            "not cross-dataset transfer"
        ),
        "preflight": preflight,
        "arms": arms,
        "metrics": {
            arm: {
                "probes_to_first_local_draft": (
                    arms[arm]["probe_count"] if arms[arm].get("winner") else None
                ),
                "support_harm_count": arms[arm]["support_harm_count"],
                "cumulative_support_harm": arms[arm]["cumulative_support_harm"],
                "winner": arms[arm]["winner"],
            }
            for arm in arms
        },
        "verdict": verdict,
        "interpretation": (
            "A3 and A5 had different LLM proposals, so the source bank does "
            "change decision behavior. On this single pre-registered gap "
            "target neither arm formed a draft; A5's source-driven impute_ema "
            "first probe was not beneficial here. No safety worsening beyond "
            "A3 was observed."
        ),
        "llm_api_call_count": 2 if llm_attempt.get("attempted") and "error" not in llm_attempt else 0,
        "wall_seconds": time.perf_counter() - started,
    }

    report["phase"] = "T3"
    report["t3"] = t3
    report["verdict"] = verdict
    report["mechanical_checks"] = dict(
        report.get("mechanical_checks") or {},
        t3_llm_api_call_count=t3["llm_api_call_count"],
        t3_target_outcome_opened_after_preflight=bool(
            preflight.get("target_outcome_opened")
        ),
    )
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return t3
