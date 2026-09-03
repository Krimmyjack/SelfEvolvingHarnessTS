"""Focused C0 protocol assertions for the task Context inlet binding.

No LLM call and no Support/Query outcome are opened by these tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_cohort

from evaluation.functional.task_episode_harness import natural_flow
from evaluation.functional.task_episode_harness.natural_flow import (
    NATURAL_EPISODES,
    _natural_initial_order,
)
from evaluation.functional.task_episode_harness.public_context import (
    C0_MATCHING_TASK_ID,
    C0_NON_MATCHING_TASK_ID,
    PUBLIC_CONTEXT_PROJECTION_FEATURE,
    build_task_public_context,
    run_context_census,
)
from SelfEvolvingHarnessTS.contracts.observables import observable_numeric_bin
from SelfEvolvingHarnessTS.methods.ttha.method import (
    _applicability_from_card,
    _applicability_is_wide,
    _applicability_reachable,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
    evaluate_applicability,
)


def _cohort() -> tuple[dict[str, np.ndarray], list[str]]:
    cohort = _load_cohort(PROJECT_ROOT)
    values = cohort["values"]
    train_uids = [
        row["series_uid"] for row in cohort["roster"] if row["role"] == "train"
    ]
    return values, train_uids


def _synthetic_values() -> dict[str, np.ndarray]:
    # Same public prefix, different future: the post-cutoff difference must be
    # invisible to the C0 projection.
    prefix = np.zeros(48, dtype=np.float64)
    prefix[20] = 25.0
    prefix[21] = -25.0
    return {
        "S1": np.concatenate([prefix, np.zeros(24)]),
        "S2": np.concatenate([prefix, np.full(24, 7.5)]),
    }


def _natural_contexts() -> dict[str, dict]:
    values, train_uids = _cohort()
    return {
        spec["task_episode_id"]: build_task_public_context(
            values,
            train_uids,
            observation_cutoff=int(spec["support_origins"][0]),
        )
        for spec in NATURAL_EPISODES
    }


def test_same_public_prefix_is_deterministic_and_outcome_blind() -> None:
    values = _synthetic_values()
    train_uids = ["S1", "S2"]
    context_a = build_task_public_context(values, train_uids, 40)
    context_b = build_task_public_context(values, train_uids, 40)

    assert context_a["task_fast_features"] == context_b["task_fast_features"]
    assert context_a["task_signature"] == context_b["task_signature"]
    assert context_a["representative_uid"] == context_b["representative_uid"]

    # Only the future differs; the public-prefix projection must not change.
    values["S1"][40:] = 999.0
    values["S2"][40:] = -999.0
    context_c = build_task_public_context(values, train_uids, 40)
    assert context_c["task_signature"] == context_a["task_signature"]
    assert context_c["task_fast_features"] == context_a["task_fast_features"]


def test_natural_cutoffs_produce_at_least_two_task_signatures() -> None:
    contexts = _natural_contexts()
    for spec in NATURAL_EPISODES:
        context = contexts[spec["task_episode_id"]]
        assert context["observation_cutoff"] == int(
            spec["support_origins"][0]
        )

    signatures = {
        task_id: context["task_signature"]
        for task_id, context in contexts.items()
    }
    distinct = []
    for signature in signatures.values():
        if signature not in distinct:
            distinct.append(signature)
    assert len(distinct) >= 2
    assert (
        signatures[C0_MATCHING_TASK_ID]
        != signatures[C0_NON_MATCHING_TASK_ID]
    )
    # The selected projection feature is the differentiating leaf.
    assert (
        contexts[C0_MATCHING_TASK_ID]["task_signature"][
            PUBLIC_CONTEXT_PROJECTION_FEATURE
        ]
        != contexts[C0_NON_MATCHING_TASK_ID]["task_signature"][
            PUBLIC_CONTEXT_PROJECTION_FEATURE
        ]
    )


def test_c0_census_freezes_matching_and_nonmatching_contexts() -> None:
    contexts = _natural_contexts()
    census = run_context_census(contexts)
    assert census["verdict"] == "TASK_CONTEXT_INLET_BINDING_PASS"
    assert census["matching_task_id"] == C0_MATCHING_TASK_ID
    assert census["non_matching_task_id"] == C0_NON_MATCHING_TASK_ID
    assert census["matching_usable"] is True
    assert (
        census["matching_signature"]
        != census["non_matching_signature"]
    )


def test_runtime_reachability_uses_the_same_task_projection() -> None:
    contexts = _natural_contexts()
    matching = contexts[C0_MATCHING_TASK_ID]
    non_matching = contexts[C0_NON_MATCHING_TASK_ID]
    card = {
        "pattern_id": "c0-test",
        "observable_signature": dict(matching["task_signature"]),
    }
    applicability = _applicability_from_card(card)
    assert _applicability_is_wide(applicability) is False

    for leaf_key in (
        "task_kind",
        PUBLIC_CONTEXT_PROJECTION_FEATURE,
    ):
        assert leaf_key in matching["task_fast_features"]
    assert (
        observable_numeric_bin(
            PUBLIC_CONTEXT_PROJECTION_FEATURE,
            float(
                matching["task_fast_features"][
                    PUBLIC_CONTEXT_PROJECTION_FEATURE
                ]
            ),
        )
        == matching["task_signature"][PUBLIC_CONTEXT_PROJECTION_FEATURE]
    )

    reachable, reason = _applicability_reachable(
        card, applicability, matching["task_fast_features"]
    )
    assert reachable is True, reason

    matched, _ = evaluate_applicability(
        applicability, matching["task_fast_features"]
    )
    assert matched is True
    non_matched, _ = evaluate_applicability(
        applicability, non_matching["task_fast_features"]
    )
    assert non_matched is False


def test_natural_forward_path_sends_real_task_context_without_llm(
    monkeypatch,
) -> None:
    contexts = _natural_contexts()
    public_context = contexts[C0_MATCHING_TASK_ID]
    captured: dict = {}

    def fake_call(messages: list[dict[str, str]]) -> dict:
        captured["messages"] = messages
        return {"program_order": ["outlier_mad"]}

    monkeypatch.setattr(natural_flow, "_nf_call", fake_call)
    result = _natural_initial_order(
        frozenset(public_context["scope_series_uids"]),
        public_context,
    )
    assert result["program_order"] == ["outlier_mad"]
    payload = __import__("json").loads(captured["messages"][1]["content"])
    assert payload["task_context"]["task_signature"] == public_context[
        "task_signature"
    ]
    assert payload["task_context"]["observation_cutoff"] == int(
        public_context["observation_cutoff"]
    )
    assert payload["scope_policy"]["selected_series_count"] == len(
        public_context["scope_series_uids"]
    )
