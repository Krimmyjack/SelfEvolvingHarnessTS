"""Bounded lexical compatibility: accept the spelling, never widen the evidence.

Two live failures, both of them the Agent doing the right thing and a strict
reader refusing it:

* it wrote one sentence of reasoning and then a syntactically valid
  tool_request envelope;
* it cited ``missing_fraction=0.0`` / ``period_evidence_status=OK`` in
  ``evidence_features`` instead of the bare feature name.

Neither is a wrong action.  Both ended a Task Episode in
AGENT_PROTOCOL_ERROR.  The repairs widen what spelling is accepted and nothing
else: no schema is relaxed, no execution permission changes, and a citation
whose value does not match what the Agent was actually shown is still refused.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.fast_path import (  # noqa: E402
    _normalize_evidence_citations,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.gateway import (  # noqa: E402
    CohortScopePublicToolGateway,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    rescue_prose_wrapped_envelope,
)

_ENVELOPE = (
    '{"schema_version":"agent-envelope/1","kind":"tool_request",'
    '"call_id":"call_3","tool_name":"localize_regions",'
    '"arguments":{"series_uid":"T234"}}'
)


def test_one_valid_envelope_wrapped_in_prose_is_recovered():
    """The exact shape the closing run died on."""
    text, recovery = rescue_prose_wrapped_envelope(
        "I have one strong public signal now: T234 shows a large level "
        "excursion, so I am localizing the region first." + _ENVELOPE
    )
    assert recovery == "RECOVERED_PROSE_WRAPPED_ENVELOPE"
    assert json.loads(text)["tool_name"] == "localize_regions"
    # Trailing prose too.
    text, recovery = rescue_prose_wrapped_envelope(_ENVELOPE + " — calling it now.")
    assert recovery == "RECOVERED_PROSE_WRAPPED_ENVELOPE"


def test_the_rescue_refuses_everything_ambiguous():
    for text in (
        # two documents: ambiguous which one is the action
        _ENVELOPE + " " + _ENVELOPE,
        # not a valid envelope
        'text {"schema_version":"agent-envelope/1","kind":"tool_request",'
        '"call_id":"c1"}',
        # nothing to recover
        "I cannot decide yet.",
        "",
    ):
        assert rescue_prose_wrapped_envelope(text)[1] == "NOT_RESCUED"


def _served() -> dict[str, set[str]]:
    values = {
        "S1": np.concatenate([np.linspace(0.0, 1.0, 60), [50.0, 0.5, 0.6, 0.7]]),
        "S2": np.linspace(1.0, 2.0, 64),
    }
    gateway = CohortScopePublicToolGateway(
        values, task_kind="forecast", observation_cutoff=64, maximum_calls=4
    )
    gateway.call("summarize_series", {"series_uid": "S1"})
    return gateway.observed_feature_values()


def test_key_equals_value_is_accepted_only_when_the_value_is_the_served_one():
    served = _served()
    key = "missing_fraction"
    assert key in served
    exact = next(iter(served[key]))

    payload = {
        "pattern_hypotheses": [
            {
                "hypothesis_id": "h1",
                "evidence_features": [f"{key}={exact}", key],
            }
        ]
    }
    normalized = _normalize_evidence_citations(payload, served)
    assert [row["normalized_to"] for row in normalized] == [key]
    # Both citations are now the canonical bare key.
    assert payload["pattern_hypotheses"][0]["evidence_features"] == [key, key]


def test_a_wrong_value_or_an_unserved_key_is_left_for_the_validator_to_reject():
    served = _served()
    key = "missing_fraction"
    payload = {
        "pattern_hypotheses": [
            {
                "hypothesis_id": "h1",
                "evidence_features": [
                    f"{key}=999.0",              # served key, value never seen
                    "never_served_feature=0.0",  # key never served at all
                ],
            }
        ]
    }
    assert _normalize_evidence_citations(payload, served) == []
    # Untouched, so the grounding rule still refuses them.
    assert payload["pattern_hypotheses"][0]["evidence_features"] == [
        f"{key}=999.0", "never_served_feature=0.0",
    ]


def test_normalization_cannot_invent_evidence_the_agent_never_fetched():
    """A gateway that served nothing normalizes nothing."""
    payload = {
        "pattern_hypotheses": [
            {"hypothesis_id": "h1", "evidence_features": ["missing_fraction=0.0"]}
        ]
    }
    assert _normalize_evidence_citations(payload, {}) == []


@pytest.mark.parametrize("bad", [None, {}, {"pattern_hypotheses": "nope"}])
def test_normalizer_is_inert_on_payloads_without_hypotheses(bad):
    assert _normalize_evidence_citations(bad or {}, _served()) == []
