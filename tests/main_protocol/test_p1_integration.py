from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping, Sequence

import pytest

from evaluation.main_protocol_p1 import common
from evaluation.main_protocol_p1 import run_p1 as master


@pytest.fixture(scope="module")
def forecast_pair() -> tuple[dict, dict]:
    raw = json.loads(master.FORECAST_REPORT.read_text(encoding="utf-8"))
    normalized = common.normalize_component(raw)
    assert common.validate_component(normalized) == []
    return normalized, raw


def _retask(component: Mapping, task: str) -> dict:
    result = copy.deepcopy(dict(component))
    result["task"] = task
    result["reported_component_pass"] = True
    return result


def _install_components(
    monkeypatch: pytest.MonkeyPatch,
    forecast_pair: tuple[dict, dict],
    *,
    classification: Mapping | None = None,
    anomaly: Mapping | None = None,
) -> None:
    forecast, raw = forecast_pair
    classification = classification or _retask(forecast, "classification")
    anomaly = anomaly or _retask(forecast, "anomaly_detection")
    monkeypatch.setattr(
        master,
        "_load_forecast",
        lambda: (copy.deepcopy(forecast), copy.deepcopy(raw)),
    )
    monkeypatch.setattr(
        master,
        "_run_classification",
        lambda _mode: copy.deepcopy(dict(classification)),
    )
    monkeypatch.setattr(
        master,
        "_run_anomaly",
        lambda _mode: copy.deepcopy(dict(anomaly)),
    )


def _walk_keys(value: object):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            yield from _walk_keys(nested)


def test_existing_forecast_is_read_only_and_never_reexecuted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    before = master.FORECAST_REPORT.read_bytes()

    monkeypatch.setattr(master, "OUT_JSON", tmp_path / "master.json")
    monkeypatch.setattr(master, "OUT_MD", tmp_path / "master.md")
    component, _raw = master._load_forecast()

    assert common.validate_component(component) == []
    source = inspect.getsource(master._load_forecast)
    assert "run_forecast_p1" not in source
    assert ".run(" not in source
    assert master.FORECAST_REPORT.read_bytes() == before
    assert not master.OUT_JSON.exists()
    assert not master.OUT_MD.exists()


def test_all_green_master_has_exactly_three_by_thirteen_rows(
    monkeypatch: pytest.MonkeyPatch,
    forecast_pair: tuple[dict, dict],
) -> None:
    forecast = forecast_pair[0]
    anomaly = _retask(forecast, "anomaly_detection")
    identity = next(
        row for row in anomaly["methods"] if row["method"] == "Identity"
    )
    identity["usage"]["raw_consumer_fits"] = {
        "support_a": 17,
        "support_b": 19,
        "total": 36,
    }
    anomaly = common.normalize_component(anomaly)
    assert common.validate_component(anomaly) == []
    _install_components(
        monkeypatch,
        forecast_pair,
        anomaly=anomaly,
    )

    payload = master.build_report()

    assert payload["overall_p1_complete"] is True
    assert payload["release_p2"] is True
    assert payload["live_outcome_release"] is False
    assert payload["natural_final_outcome_reads"] == 0
    assert payload["development_query_evaluations"] == 0
    assert set(payload["components"]) == set(common.TASKS)
    assert sum(
        len(component["methods"])
        for component in payload["components"].values()
    ) == 39
    for task, component in payload["components"].items():
        assert component["task"] == task
        assert [row["method"] for row in component["methods"]] == list(
            common.MANDATORY_METHODS
        )
        for row in component["methods"]:
            assert set(row["surfaces"]) == set(common.SURFACES)
            assert row["usage"]["within_caps"] is True

    aegis_keys = [
        key for key in _walk_keys(payload)
        if key in {"aegis_adapter", "aegists_adapter"}
    ]
    assert aegis_keys == ["aegis_adapter"]
    assert payload["aegis_adapter"]["blocking"] is False
    assert payload["ad_method_gate"]["status"] == "NOT_RELEASED_BY_P1"
    assert payload["ad_method_gate"]["ad_evolution_release"] is False
    assert "Forecast" in payload["release_scope"]
    assert master.OUT_JSON.name == "p1_core_baseline_smoke_20260830.json"
    assert master.OUT_MD.name == "p1_core_baseline_smoke_20260830.md"

    forbidden_fields = {
        "sha", "sha256", "checksum", "digest", "manifest_hash",
        "inventory_digest", "execution_signature",
    }
    assert not (set(_walk_keys(payload)) & forbidden_fields)


def test_p0_failure_skips_new_component_execution(
    monkeypatch: pytest.MonkeyPatch,
    forecast_pair: tuple[dict, dict],
) -> None:
    forecast, raw = forecast_pair
    monkeypatch.setattr(
        master,
        "_read_object",
        lambda _path: {
            "verdict": {
                "audit": "P0B_INCOMPLETE",
                "execution": "P0B_BLOCKED",
                "p1_release": False,
            }
        },
    )
    monkeypatch.setattr(
        master,
        "_load_forecast",
        lambda: (copy.deepcopy(forecast), copy.deepcopy(raw)),
    )

    def forbidden_component(_mode):
        raise AssertionError("a P1 component ran without P0b release")

    monkeypatch.setattr(master, "_run_classification", forbidden_component)
    monkeypatch.setattr(master, "_run_anomaly", forbidden_component)

    payload = master.build_report()

    assert payload["overall_p1_complete"] is False
    assert payload["release_p2"] is False
    assert payload["component_pass"]["classification"] is False
    assert payload["component_pass"]["anomaly_detection"] is False


def _missing_method(component: dict) -> None:
    component["methods"].pop()


def _duplicate_method(component: dict) -> None:
    component["methods"].append(copy.deepcopy(component["methods"][0]))


def _over_budget(component: dict) -> None:
    component["methods"][0]["usage"]["full_support_evaluations"] = {
        "support_a": 4,
        "support_b": 1,
    }


def _protocol_error(component: dict) -> None:
    component["protocol_errors"]["cross_task_skill_leakage"] = 1


def _final_read(component: dict) -> None:
    component["protocol_errors"]["natural_final_outcome_reads"] = 1


def _query_read(component: dict) -> None:
    component["protocol_errors"]["development_query_evaluations"] = 1


def _lifecycle_not_exercised(component: dict) -> None:
    component["backend"]["production_lifecycle_exercised"] = False


def _ttha_method_not_exercised(component: dict) -> None:
    component["backend"]["production_ttha_method_exercised"] = False


def _online_round_not_exercised(component: dict) -> None:
    component["backend"]["production_run_online_round_exercised"] = False


def _common_dsl_identity_missing(component: dict) -> None:
    component["common_dsl_contract"]["identity_available"] = False


def _best_fixed_charged_to_target(component: dict) -> None:
    row = next(
        row for row in component["methods"]
        if row["method"] == "Best Fixed Per-task"
    )
    row["details"]["cost_by_phase"]["evolution_selection"][
        "charged_to_target_b4"
    ] = True


def _a5_writeback_disabled(component: dict) -> None:
    row = next(
        row for row in component["methods"] if row["method"] == "A5-online"
    )
    row["details"]["writeback_channel"] = False


def _reported_pass_missing(component: dict) -> None:
    component["reported_component_pass"] = None


@pytest.mark.parametrize(
    "mutate",
    (
        _missing_method,
        _duplicate_method,
        _over_budget,
        _protocol_error,
        _final_read,
        _query_read,
        _lifecycle_not_exercised,
        _ttha_method_not_exercised,
        _online_round_not_exercised,
        _common_dsl_identity_missing,
        _best_fixed_charged_to_target,
        _a5_writeback_disabled,
        _reported_pass_missing,
    ),
    ids=(
        "missing-method",
        "duplicate-method",
        "budget",
        "protocol-error",
        "natural-final",
        "development-query",
        "production-lifecycle",
        "production-ttha-method",
        "production-online-round",
        "common-dsl",
        "best-fixed-selection-cost",
        "a5-writeback",
        "reported-pass",
    ),
)
def test_any_component_contract_fault_blocks_p1_and_p2(
    monkeypatch: pytest.MonkeyPatch,
    forecast_pair: tuple[dict, dict],
    mutate,
) -> None:
    classification = _retask(forecast_pair[0], "classification")
    mutate(classification)
    classification = common.normalize_component(classification)
    _install_components(
        monkeypatch,
        forecast_pair,
        classification=classification,
    )

    payload = master.build_report()

    assert payload["component_pass"]["classification"] is False
    assert payload["overall_p1_complete"] is False
    assert payload["release_p2"] is False
    assert payload["live_outcome_release"] is False
    assert payload["blocking_failures"]


def test_blocking_aegis_is_rejected_even_when_components_pass(
    monkeypatch: pytest.MonkeyPatch,
    forecast_pair: tuple[dict, dict],
) -> None:
    forecast, raw = forecast_pair
    raw = copy.deepcopy(raw)
    raw["aegists_adapter"]["blocking"] = True
    monkeypatch.setattr(
        master,
        "_load_forecast",
        lambda: (copy.deepcopy(forecast), copy.deepcopy(raw)),
    )
    monkeypatch.setattr(
        master,
        "_run_classification",
        lambda _mode: _retask(forecast, "classification"),
    )
    monkeypatch.setattr(
        master,
        "_run_anomaly",
        lambda _mode: _retask(forecast, "anomaly_detection"),
    )

    payload = master.build_report()

    assert payload["overall_p1_complete"] is False
    assert payload["release_p2"] is False
    assert payload["aegis_adapter"]["blocking"] is True
