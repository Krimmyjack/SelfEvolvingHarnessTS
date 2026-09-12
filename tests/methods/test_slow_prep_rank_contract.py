"""Contract tests for slow_prep_rank_v1 (registered candidate; NOT EditManifest)."""

from __future__ import annotations

import pytest

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (
    AgentProtocolError,
    TTHAAgentCore,
    validate_local_schema,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
    PRIMARY_FIXED_DIV_K5,
    h1_simple_bias_shortlist,
)


def _menu_k5() -> list[str]:
    return [
        "identity",
        "outlier_iqr(k=1.0)",
        "outlier_mad(k=2.5)",
        "winsorize(limits=0.05)",
        "fft_decompose",
        "outlier_iqr(k=0.8)>winsorize(limits=0.1)>fft_decompose",
        "outlier_mad(k=2.5)>winsorize>fft_decompose",
    ]


def _valid_payload(*, k: int = 5) -> dict:
    menu = _menu_k5()
    keys = menu[:k]
    return {
        "role": "slow",
        "stage": "prep_menu_rank",
        "schema": "slow_prep_rank_v1",
        "proposal": {
            "ordered_prep_keys": keys,
            "rationale": "form-only exploration under fit budget",
            "form_signals_used": ["pick", "family", "C_A", "C_B"],
            "abstain": False,
        },
        "allowlist_binding": {
            "m_large_id": "beijing_v2_ridge_m56",
            "m_large_sha256": canonical_sha256(list(menu)),
            "k": k,
        },
        "form_memory": {
            "pick": "outlier_iqr",
            "family": "robust_scale",
            "C_A": 0.12,
            "C_B": 0.34,
            "top_families": ["robust_scale", "spectral"],
        },
        "non_claims": ["not_edit_manifest", "not_propose_edit"],
    }


def test_schema_happy_path_ordered_prep_keys_length_k():
    schema = TTHAAgentCore.load_stage_schema("slow_prep_rank_v1")
    payload = _valid_payload(k=5)
    validate_local_schema(payload, schema)
    assert len(payload["proposal"]["ordered_prep_keys"]) == 5
    assert payload["stage"] == "prep_menu_rank"
    assert payload["schema"] == "slow_prep_rank_v1"


def test_schema_rejects_edit_manifest():
    schema = TTHAAgentCore.load_stage_schema("slow_prep_rank_v1")
    payload = _valid_payload()
    payload["edit_manifest"] = {"edit_id": "should-not-pass"}
    with pytest.raises(AgentProtocolError):
        validate_local_schema(payload, schema)


def test_schema_rejects_form_memory_E():
    schema = TTHAAgentCore.load_stage_schema("slow_prep_rank_v1")
    payload = _valid_payload()
    payload["form_memory"]["E"] = 0.99
    with pytest.raises(AgentProtocolError):
        validate_local_schema(payload, schema)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(schema="slow_edit_v1"),
        lambda row: row.update(stage="edit"),
        lambda row: row.update(role="fast"),
    ],
)
def test_schema_rejects_wrong_schema_or_stage(mutate):
    schema = TTHAAgentCore.load_stage_schema("slow_prep_rank_v1")
    payload = _valid_payload()
    mutate(payload)
    with pytest.raises(AgentProtocolError):
        validate_local_schema(payload, schema)


def test_h1_simple_bias_shortlist_reserves_simplex():
    slow = [
        "outlier_mad(k=2.5)>winsorize>fft_decompose",
        "outlier_iqr(k=0.8)>winsorize(limits=0.1)>fft_decompose",
        "fft_decompose",
        "winsorize(limits=0.05)",
        "identity",
    ]
    out, meta = h1_simple_bias_shortlist(slow, PRIMARY_FIXED_DIV_K5, k=5, m=2)
    assert len(out) == 5
    assert meta["forced_simplex"] == ["identity", "outlier_iqr(k=1.0)"]
    assert "identity" in out
    assert "outlier_iqr(k=1.0)" in out
    # pipe-head extension of Fixed-div simple should be demoted when room allows
    assert "outlier_mad(k=2.5)>winsorize>fft_decompose" in meta["skipped_div_simple_extensions"]


def test_stage_schema_registered_in_contracts():
    schema = TTHAAgentCore.load_stage_schema("slow_prep_rank_v1")
    assert schema["properties"]["schema"]["const"] == "slow_prep_rank_v1"
    assert schema["properties"]["stage"]["const"] == "prep_menu_rank"
    assert schema.get("additionalProperties") is False
