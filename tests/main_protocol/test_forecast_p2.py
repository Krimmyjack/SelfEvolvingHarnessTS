from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from evaluation.main_protocol_p2 import run_forecast_p2 as p2


@pytest.fixture(scope="module")
def pilot(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("forecast_p2")
    old_output = p2.OUT_JSON
    p1_before = p2.P1_REPORT.read_bytes()
    p2.OUT_JSON = root / "pilot.json"
    try:
        payload = p2.run()
        yield payload, p2.OUT_JSON, root, p1_before
    finally:
        p2.OUT_JSON = old_output


def _record(payload: Mapping, decision: int, arm: str) -> dict:
    return next(
        row
        for row in payload["runs"]
        if row["decision_index"] == decision and row["arm"] == arm
    )


def _walk(value: object):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk(nested)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            yield from _walk(nested)
    elif isinstance(value, str):
        yield value


def test_p1_release_is_the_only_upstream_launch_gate():
    gate = p2._assert_p1_release()
    assert all(gate["checks"].values())
    assert gate["released_scope"] == "Forecast P2 single-flow pilot only"


def test_course_uses_only_exposed_kdd_20_20_and_two_distinct_controls():
    base, record = p2.forecast_p1._load_exposed_cell()
    first = p2._make_cell(base, 7)
    second = p2._make_cell(base, 8)
    assert len(base.support_a) == len(base.support_b) == 20
    assert set(base.support_a).isdisjoint(base.support_b)
    assert record["data_role"] == "EXPOSED_DEVELOPMENT"
    assert np.array_equal(
        first.observation_block, second.observation_block, equal_nan=True
    )
    assert any(
        not np.array_equal(first.values[uid], second.values[uid], equal_nan=True)
        for uid in base.support_b
    )
    assert all(
        np.array_equal(first.values[uid], base.values[uid], equal_nan=True)
        for uid in base.support_a
    )


def test_real_lifecycle_produces_revocation_and_reencounter_difference(pilot):
    payload = pilot[0]
    u1_k0 = _record(payload, 1, "K0-fixed")
    u1_a5 = _record(payload, 1, "A5-online")
    u2_k0 = _record(payload, 2, "K0-fixed")
    u2_a5 = _record(payload, 2, "A5-online")

    assert u1_k0["state_before"] == u1_a5["state_before"]
    assert u1_a5["support_a"]["relation"] == "POSITIVE"
    assert u1_a5["support_a"]["gain"] == pytest.approx(1.7547956437077405)
    assert u1_a5["support_b"]["relation"] == "NEGATIVE"
    assert u1_a5["support_b"]["gain"] == pytest.approx(-0.15189708424945625)
    assert u1_a5["update"]["production_revocation"] is True
    assert u1_a5["state_after"]["controlled_card"]["present"] is False

    assert u2_a5["state_before"] == u1_a5["state_after"]
    assert u2_k0["state_before"] == u1_k0["state_before"]
    assert u2_k0["trace"]["controlled_supply_count"] == 1
    assert u2_k0["trace"]["controlled_probe_count"] == 1
    assert u2_k0["trace"]["abstained"] is False
    assert u2_a5["trace"]["controlled_supply_count"] == 0
    assert u2_a5["trace"]["controlled_probe_count"] == 0
    assert u2_a5["trace"]["abstained"] is True
    assert p2.derive_treatment_gate(payload["runs"])["treatment_nonempty"] is True


def test_treatment_validator_rejects_false_positive_shortcuts(pilot):
    runs = pilot[0]["runs"]

    no_update = copy.deepcopy(runs)
    broken = _record({"runs": no_update}, 1, "A5-online")
    broken["update"]["production_revocation"] = False
    broken["update"]["kind"] = "NONE"
    assert p2.derive_treatment_gate(no_update)["treatment_nonempty"] is False

    retrieval_only = copy.deepcopy(runs)
    later = _record({"runs": retrieval_only}, 2, "A5-online")
    later["trace"]["retrieved_controlled_card"] = True
    assert p2.derive_treatment_gate(retrieval_only)["treatment_nonempty"] is False

    wrong_pattern = copy.deepcopy(runs)
    later = _record({"runs": wrong_pattern}, 2, "A5-online")
    later["context"]["pattern"] = {"task_kind": "forecast", "missing_fraction": "high"}
    assert p2.derive_treatment_gate(wrong_pattern)["treatment_nonempty"] is False

    autonomous_difference = copy.deepcopy(runs)
    later = _record({"runs": autonomous_difference}, 2, "A5-online")
    later["trace"]["autonomous_nonidentity_candidate_count"] = 1
    assert p2.derive_treatment_gate(autonomous_difference)["treatment_nonempty"] is False


def test_p2_pass_is_mechanism_only_and_releases_only_p3_integration(pilot):
    payload = pilot[0]
    assert payload["verdict"] == (
        "P2_FORECAST_TREATMENT_WIRING_PASS__P3_INTEGRATION_RELEASED"
    )
    assert payload["p2_complete"] is True
    assert payload["release_p3"] is True
    assert payload["p3_complete"] is False
    assert payload["release_p4"] is False
    assert payload["live_outcome_release"] is False
    assert payload["treatment_gate"]["status"] == "TREATMENT_WIRING_NONEMPTY"
    assert payload["rq3_event_gate"]["online_evolution_positive_claim"] is False
    assert payload["controlled_witness"]["card_learned_from_source_evidence"] is False
    assert payload["claim_boundaries"]["performance_claim"] is False
    assert payload["claim_boundaries"]["natural_data_capability_claim"] is False


def test_four_arm_isolation_b4_and_closed_boundaries(pilot):
    payload = pilot[0]
    for decision in (1, 2):
        rows = [row for row in payload["runs"] if row["decision_index"] == decision]
        assert {row["arm"] for row in rows} == set(p2.ARMS)
    labels = [
        row["store_isolation_label"]
        for row in payload["runs"]
        if row["store_isolation_label"] is not None
    ]
    assert len(labels) == len(set(labels))
    assert p2._arm_failures(payload["runs"]) == []
    assert p2._budget_failures(payload["runs"]) == []
    assert p2._boundary_failures(payload["runs"]) == []
    assert payload["budget_gate"]["external_llm_calls"] == 0
    assert payload["budget_gate"]["tokens"] == 0
    assert all(row["usage"]["within_caps"] for row in payload["runs"])
    assert all(
        all(value == 0 for value in row["boundary_counts"].values())
        for row in payload["runs"]
    )
    assert not payload["blocking_failures"]


def test_single_report_preserves_p1_and_contains_no_new_integrity_tokens(pilot):
    payload, output, root, p1_before = pilot
    assert output.is_file()
    assert [path.name for path in root.iterdir()] == ["pilot.json"]
    assert p2.P1_REPORT.read_bytes() == p1_before

    blocked = {
        "sha",
        "sha256",
        "hash",
        "digest",
        "checksum",
        "manifest",
        "fingerprint",
    }
    source = p2.__file__ and open(p2.__file__, encoding="utf-8").read().lower()
    source_segments = {
        part
        for token in re.findall(r"[a-z0-9_]+", source)
        for part in token.split("_")
    }
    assert not (blocked & source_segments)
    artifact_words = {str(value).lower() for value in _walk(payload)}
    assert not (blocked & artifact_words)
    assert re.search(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", json.dumps(payload)) is None
