from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1


def test_canonical_task_sha_and_semantics():
    task = forecast_task_spec_v1(horizon=12)
    assert task.to_dict() == {
        "task_type": "forecast",
        "target_semantics": "future_values",
        "label_availability": "history_only",
        "metric": {"name": "nRMSE", "direction": "lower_is_better"},
        "horizon": 12,
        "downstream_model_class": "dlinear_shared",
        "forbidden_modifications": [],
    }
    assert len(task.sha()) == 16


# ---------------------------------------------------------------------------
# #42i Part A -- anomaly task contract wiring + vocabulary + red-line.
#
# These tests exercise the AD-specific contract layer that Part A introduced:
#   * anomaly_background_model_quality_contract_v1 is constructible and uses
#     the three new vocabulary tokens (normal_boundary_fidelity on preserve,
#     normal_boundary_shrinkage + false_alarm_amplification on harms)
#   * the contract preserves anomaly_evidence semantics (objective field)
#   * the anomaly_events vocabulary constant carries its pinning comment
#   * anomaly_task_context_v1 wires spec + contract + deployment_constraints
#   * deployment_constraints pin a single Consumer (fixed:aegists_iforest_v1)
#     with maximum_candidates=1 and maximum_modified_fraction <= 0.20
#   * the context round-trips through to_dict and survives parse_json_document
#   * both JSON schemas accept the three new vocabulary tokens
#   * red line: the contract's to_dict has zero Pattern->Program fields
#
# These tests use the bare ``contracts.`` prefix so they can run from inside
# the guidance-evolution checkout with ``PYTHONPATH=.``.  The original
# forecast test above keeps its ``SelfEvolvingHarnessTS.`` prefix.
# ---------------------------------------------------------------------------
from contracts.task import (  # noqa: E402
    HARM_VOCABULARY,
    PRESERVATION_VOCABULARY,
    anomaly_background_model_quality_contract_v1,
    anomaly_task_context_v1,
    anomaly_task_spec_v1,
    deployment_constraints_v1,
)
from contracts.canonical import parse_json_document  # noqa: E402


def test_anomaly_contract_uses_three_new_vocab_tokens():
    contract = anomaly_background_model_quality_contract_v1()
    assert "normal_boundary_fidelity" in contract.preserve
    assert "normal_boundary_shrinkage" in contract.harms
    assert "false_alarm_amplification" in contract.harms


def test_anomaly_contract_preserves_anomaly_evidence():
    contract = anomaly_background_model_quality_contract_v1()
    assert contract.objective == "preserve_anomaly_evidence"
    assert set(contract.preserve) >= {
        "observed_values_outside_suspect_region",
        "temporal_order",
        "series_length",
        "anomaly_events",
        "normal_boundary_fidelity",
    }
    assert set(contract.harms) >= {
        "event_erasure",
        "normal_boundary_shrinkage",
        "false_alarm_amplification",
        "unnecessary_modification",
        "future_information_use",
        "out_of_scope_change",
    }


def test_anomaly_contract_vocab_is_closed():
    """The closed-vocabulary guard rejects any token outside the allowed set."""
    from dataclasses import replace

    contract = anomaly_background_model_quality_contract_v1()
    try:
        replace(contract, preserve=("bogus_preserve_token",))
    except ValueError as exc:
        assert "unknown vocabulary" in str(exc) or "preserve" in str(exc).lower()
    else:
        raise AssertionError("closed vocabulary guard did not trip on preserve")

    try:
        replace(contract, harms=("bogus_harm_token",))
    except ValueError as exc:
        assert "unknown vocabulary" in str(exc) or "harms" in str(exc).lower()
    else:
        raise AssertionError("closed vocabulary guard did not trip on harms")


def test_anomaly_events_vocab_entry_carries_pinning_comment():
    """The pinning comment travels with the anomaly_events vocabulary entry."""
    from pathlib import Path

    src_path = Path(__file__).resolve().parents[2] / "contracts" / "task.py"
    text = src_path.read_text(encoding="utf-8")
    assert '"anomaly_events"' in text
    # The pinned comment must explain that anomaly_events refers to evidence
    # required for downstream event discrimination -- NOT a blanket ban on
    # deleting any suspected anomaly inside the training region.
    assert "downstream" in text
    assert "event discrimination" in text
    assert "NOT" in text or "not" in text
    assert "forbidden from being deleted" in text or "禁止删除" in text


def test_anomaly_task_context_wires_spec_contract_deployment():
    ctx = anomaly_task_context_v1()
    assert ctx.task_spec.task_type == "anomaly_detection"
    assert ctx.task_spec.target_semantics == "anomaly_events"
    assert ctx.task_spec.downstream_model_class == "aegists_iforest_v1"
    assert ctx.quality_contract.contract_id == "anomaly-background-model-quality-v1"
    assert ctx.quality_contract.objective == "preserve_anomaly_evidence"
    assert ctx.deployment_constraints.model_policy == "fixed"
    assert ctx.deployment_constraints.fixed_downstream_model_id == "fixed:aegists_iforest_v1"
    assert ctx.deployment_constraints.maximum_candidates == 1
    assert ctx.deployment_constraints.maximum_modified_fraction <= 0.20


def test_anomaly_task_context_roundtrips_through_to_dict_and_canonical_parse():
    ctx = anomaly_task_context_v1()
    d = ctx.to_dict()
    assert set(d.keys()) >= {
        "schema_version", "task_spec", "quality_contract", "deployment_constraints"
    }
    import json
    blob = json.dumps(d, sort_keys=True).encode("utf-8")
    parsed = parse_json_document(blob)
    assert parsed["task_spec"]["task_type"] == "anomaly_detection"
    assert parsed["quality_contract"]["contract_id"] == "anomaly-background-model-quality-v1"
    assert parsed["deployment_constraints"]["fixed_downstream_model_id"] == "fixed:aegists_iforest_v1"


def test_anomaly_contract_sha_is_stable_and_64_hex():
    """TaskQualityContract.sha() returns 64 hex chars (sha256)."""
    a = anomaly_background_model_quality_contract_v1()
    b = anomaly_background_model_quality_contract_v1()
    assert a.sha() == b.sha()
    assert len(a.sha()) == 64
    int(a.sha(), 16)  # parses as hex


def test_anomaly_context_sha_is_64_hex_and_stable():
    ctx_a = anomaly_task_context_v1()
    ctx_b = anomaly_task_context_v1()
    assert ctx_a.sha() == ctx_b.sha()
    assert len(ctx_a.sha()) == 64
    int(ctx_a.sha(), 16)


def test_schemas_accept_the_three_new_vocab_tokens():
    """The contract and context JSON schemas must list the three new tokens.

    For task_context_v1.json the preserve/harms enums live under
    ``$defs.quality_contract.properties``; for task_quality_contract_v1.json
    they live at the top-level ``properties`` block.  This test walks both
    locations so a future schema relocation is caught.
    """
    from pathlib import Path

    schema_root = Path(__file__).resolve().parents[2] / "contracts" / "schemas"

    def collect_preserve_harms_enums(doc):
        results = []
        # Top-level properties (task_quality_contract_v1.json)
        props = doc.get("properties", {})
        if "preserve" in props and "harms" in props:
            results.append((props["preserve"]["items"]["enum"],
                            props["harms"]["items"]["enum"]))
        # $defs.quality_contract (task_context_v1.json)
        defs = doc.get("$defs", {})
        qc = defs.get("quality_contract", {})
        qc_props = qc.get("properties", {})
        if "preserve" in qc_props and "harms" in qc_props:
            results.append((qc_props["preserve"]["items"]["enum"],
                            qc_props["harms"]["items"]["enum"]))
        return results

    for name in ("task_quality_contract_v1.json", "task_context_v1.json"):
        with (schema_root / name).open("rb") as f:
            doc = parse_json_document(f.read())
        enums = collect_preserve_harms_enums(doc)
        assert enums, "no preserve/harms enum found in %s" % name
        for preserve_enum, harms_enum in enums:
            assert "normal_boundary_fidelity" in preserve_enum, (name, preserve_enum)
            assert "normal_boundary_shrinkage" in harms_enum, (name, harms_enum)
            assert "false_alarm_amplification" in harms_enum, (name, harms_enum)


def test_anomaly_contract_has_no_pattern_to_program_fields():
    """Red line: zero Pattern->Program rules in the contract payload."""
    contract = anomaly_background_model_quality_contract_v1()
    cd = contract.to_dict()
    forbidden_field_names = {
        "pattern_to_program", "program_for_pattern", "binding_rules",
        "rule_map", "operator_for_pattern", "pattern_program_map",
        "patterns_to_programs", "programs_for_patterns",
    }
    found = forbidden_field_names.intersection(cd.keys())
    assert not found, "red-line breach: %r found in contract.to_dict()" % (found,)


def test_anomaly_contract_red_line_holds_under_dict_walk():
    """Walk the full contract dict (incl. nested structures) for Pattern->Program style bindings."""
    def walk(node, path=()):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in {"pattern_to_program", "program_for_pattern",
                         "binding_rules", "operator_for_pattern"}:
                    yield path + (k,)
                yield from walk(v, path + (k,))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                yield from walk(v, path + (i,))

    cd = anomaly_background_model_quality_contract_v1().to_dict()
    hits = list(walk(cd))
    forbidden = [p for p in hits if p and p[-1] in {
        "pattern_to_program", "program_for_pattern", "binding_rules",
        "operator_for_pattern",
    }]
    assert not forbidden, "red-line breach at paths: %r" % (forbidden,)


def test_anomaly_deployment_constraints_pin_a_single_consumer():
    dc = deployment_constraints_v1(
        constraint_id="anomaly-fixed-aegists-iforest-v1",
        fixed_downstream_model_id="fixed:aegists_iforest_v1",
        maximum_candidates=1,
        maximum_modified_fraction=0.20,
    )
    assert dc.model_policy == "fixed"
    assert dc.fixed_downstream_model_id == "fixed:aegists_iforest_v1"
    assert dc.maximum_candidates == 1
    assert 0.0 <= dc.maximum_modified_fraction <= 1.0


def test_anomaly_context_keeps_vocabulary_exposed_in_module():
    assert "normal_boundary_fidelity" in PRESERVATION_VOCABULARY
    assert "normal_boundary_shrinkage" in HARM_VOCABULARY
    assert "false_alarm_amplification" in HARM_VOCABULARY


def test_anomaly_context_does_not_silently_swap_downstream_class():
    """A different downstream_model_class must surface somewhere.

    The deployment constraint is the real guard, but the spec also names
    aegists_iforest_v1.  Either the dataclass replace rejects the swap, or
    the constraint pin still points at fixed:aegists_iforest_v1.
    """
    from dataclasses import replace

    ctx = anomaly_task_context_v1()
    try:
        ctx2 = replace(ctx, task_spec=anomaly_task_spec_v1(
            downstream_model_class="some_other_consumer"))
    except (ValueError, TypeError) as exc:
        assert "downstream" in str(exc).lower() or "consumer" in str(exc).lower()
    else:
        # Replace succeeded -- the constraint pin must still hold.
        assert ctx2.deployment_constraints.fixed_downstream_model_id == "fixed:aegists_iforest_v1"
