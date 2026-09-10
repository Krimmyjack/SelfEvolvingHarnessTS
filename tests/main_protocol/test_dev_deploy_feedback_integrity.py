"""Directional tests for DEV-DEPLOY prepare-status and formation evidence.

0 project LLM.  No Consumer fits.  Does not rewrite historical JSON.
Covers A1 (FAILED/trace vs identity vs UNKNOWN), A2 (per-series menu
pairs and historical empty-selection marking), and a read-only A3
combination-space diagnosis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SelfEvolvingHarnessTS.contracts.candidate import Candidate  # noqa: E402
from SelfEvolvingHarnessTS.contracts.method import (  # noqa: E402
    ExecutionReceipt,
    PreparedSeries,
    PreparationResult,
    PreparationStatus,
)
from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: E402
from SelfEvolvingHarnessTS.runtime.candidate_pool import CandidatePool  # noqa: E402
from SelfEvolvingHarnessTS.runtime.candidate_verification import (  # noqa: E402
    verify_candidate,
)

from evaluation.main_protocol_p4 import run_dev_deploy1 as deploy1  # noqa: E402
from evaluation.main_protocol_p4.run_dev_deploy2 import (  # noqa: E402
    _render_case,
    _render_menu,
)


def _trace(**kwargs):
    defaults = dict(
        chosen_candidate_id="",
        candidate_program_steps={},
        compilation_status="ok",
        execution_status="ok",
        candidate_ids=("identity",),
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _failed(error="AgentProtocolError: schema rejected"):
    return PreparationResult(
        PreparationStatus.FAILED, None, None,
        ExecutionReceipt(ok=False, error=error),
    )


def _abstained_identity():
    return PreparationResult(
        PreparationStatus.ABSTAINED,
        PreparedSeries("series-1", np.asarray([1.0, 2.0, 3.0]), (),
                       "original_units"),
        None,
        ExecutionReceipt(ok=True),
    )


def _prepared(ops=("outlier_iqr",)):
    return PreparationResult(
        PreparationStatus.PREPARED,
        PreparedSeries("series-1", np.asarray([1.0, 2.0, 3.0]), ops,
                       "original_units"),
        None,
        ExecutionReceipt(ok=True),
    )


class _Verify:
    def __init__(self, passed: bool) -> None:
        self.passed = passed


class _Gain:
    def __init__(self, gain) -> None:
        self.gain = gain


class _Scorer:
    def __init__(self, gain: float = 0.5) -> None:
        self.gain = gain
        self.calls: list[tuple] = []

    def sequence_reading(self, position, uid, steps, origin):
        self.calls.append((position, uid, tuple(steps), origin))
        return _Gain(self.gain)


def _historical_empty_row(**extra):
    row = {
        "series_uid": "T14",
        "origin": 1656,
        "public_features": {"missing_fraction": 0.1},
        "candidate_programs": {},
        "chosen_candidate_id": None,
        "deploy_status": "EMPTY_SELECTION_FALLBACK_TO_IDENTITY",
        "deployed_program": "identity",
        "deployed_program_steps_reconstructed": "UNKNOWN",
        "deployment_gain_at_origin": 0.0,
        "fixed_program_gain_at_plus48": 0.0,
        "action_facts": "UNKNOWN",
        "faults": [],
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# A1.1 internal FAILED with leftover trace; external throw
# ---------------------------------------------------------------------------

def test_failed_prepare_with_nonempty_identity_trace_is_not_identity():
    trace = _trace(chosen_candidate_id="identity",
                   compilation_status="failed",
                   execution_status="not_started")
    interpreted = deploy1.interpret_prepare_outcome(
        result=_failed(), trace=trace)
    assert interpreted["deploy_status"] == "PREPARE_FAILED"
    assert interpreted["valid_mechanism_decision"] is False
    scored = deploy1.apply_legality_and_score(interpreted)
    assert scored["deployment_gain_at_origin"] == "UNKNOWN"
    assert scored["fixed_program_gain_at_plus48"] == "UNKNOWN"
    assert scored["deployed_program"] == "UNKNOWN"
    record = deploy1.finalize_fastonly_decision(
        {"faults": []}, result=_failed(), trace=trace, thrown=None,
        scorer=None, ctx=None, uid="T14", origin=1656,
        delayed_origin=1704, position=9)
    assert record["deploy_status"] == "PREPARE_FAILED"
    assert record["valid_mechanism_decision"] is False
    assert record["prepare_error"]["trace_present"] is True
    assert record["prepare_error"]["chosen_candidate_id_on_trace"] == "identity"
    assert record["active_model_choice"] is False


def test_external_exception_is_not_identity_zero():
    thrown = RuntimeError("backend exploded")
    interpreted = deploy1.interpret_prepare_outcome(
        result=None, trace=None, thrown=thrown)
    assert interpreted["deploy_status"] == "FAULT_NO_DECISION"
    scored = deploy1.apply_legality_and_score(interpreted)
    assert scored["deployment_gain_at_origin"] == "UNKNOWN"
    assert scored["deployed_program"] == "UNKNOWN"
    record = deploy1.finalize_fastonly_decision(
        {"faults": [{"kind": "RuntimeError", "why": "backend exploded"}]},
        result=None, trace=_trace(chosen_candidate_id="identity"),
        thrown=thrown, scorer=None, ctx=None, uid="T14", origin=1656,
        delayed_origin=1704, position=9)
    # leftover trace after a throw without a PreparationResult is still
    # not a decision -- interpret sees thrown+result None first.
    assert record["deploy_status"] == "FAULT_NO_DECISION"
    assert record["deployment_gain_at_origin"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# A1.2 explicit identity zero; legal program path; account stop
# ---------------------------------------------------------------------------

def test_explicit_identity_keeps_legal_zero():
    interpreted = deploy1.interpret_prepare_outcome(
        result=_abstained_identity(),
        trace=_trace(chosen_candidate_id="identity"))
    assert interpreted["deploy_status"] == "IDENTITY_SELECTED"
    scored = deploy1.apply_legality_and_score(interpreted)
    assert scored["deployment_gain_at_origin"] == 0.0
    assert scored["fixed_program_gain_at_plus48"] == 0.0
    assert scored["deployed_program"] == "identity"
    assert scored["valid_mechanism_decision"] is True


def test_legal_program_path_still_scores_the_chosen_steps():
    steps = (("outlier_iqr", {"k": 1.5}),)
    trace = _trace(
        chosen_candidate_id="cand_a",
        candidate_program_steps={"cand_a": steps},
        candidate_ids=("identity", "cand_a"),
    )
    interpreted = deploy1.interpret_prepare_outcome(
        result=_prepared(), trace=trace)
    assert interpreted["mechanism_gain_policy"] == "score_if_legal_else_raw_fallback"
    scorer = _Scorer(0.5)
    scored = deploy1.apply_legality_and_score(
        interpreted, verify=lambda _s: _Verify(True), scorer=scorer,
        position=9, uid="T14", origin=1656, delayed_origin=1704)
    assert scored["deploy_status"] == "DEPLOYED"
    assert scored["deployment_gain_at_origin"] == 0.5
    assert scored["valid_mechanism_decision"] is True
    assert scorer.calls[0][2] == steps
    raw = deploy1.apply_legality_and_score(
        interpreted, verify=lambda _s: _Verify(False), scorer=_Scorer(0.5),
        position=9, uid="T14", origin=1656, delayed_origin=1704)
    assert raw["deploy_status"] == "LEGALITY_FALLBACK_RAW"
    assert raw["deployment_gain_at_origin"] == 0.0
    assert raw["active_model_choice"] is False


def test_account_or_permission_error_still_stops_not_abstains():
    with pytest.raises(deploy1.AccountFault):
        deploy1.raise_if_account_fault(
            "HTTPError: Error code: 402 Insufficient Balance")
    with pytest.raises(deploy1.AccountFault):
        deploy1.raise_if_account_fault("insufficient_user_quota 用户额度不足")
    with pytest.raises(deploy1.AccountFault):
        deploy1.interpret_prepare_outcome(
            result=_failed("Error code: 402 payment required"),
            trace=_trace(chosen_candidate_id="identity",
                         compilation_status="failed"),
        )


def test_empty_and_unknown_candidate_are_not_identity_fallback():
    empty = deploy1.interpret_prepare_outcome(
        result=_prepared(), trace=_trace(chosen_candidate_id=""))
    assert empty["deploy_status"] == "EMPTY_SELECTION_NO_VALID_DECISION"
    assert deploy1.apply_legality_and_score(empty)[
        "deployment_gain_at_origin"] == "UNKNOWN"
    unknown = deploy1.interpret_prepare_outcome(
        result=_prepared(),
        trace=_trace(chosen_candidate_id="cand_missing",
                     candidate_program_steps={"cand_a": (("impute_linear", {}),)}))
    assert unknown["deploy_status"] == "UNKNOWN_CANDIDATE_NO_VALID_DECISION"
    assert deploy1.apply_legality_and_score(unknown)[
        "deployment_gain_at_origin"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# A1.3 one missing decision => population mean UNKNOWN
# ---------------------------------------------------------------------------

def test_one_missing_decision_makes_population_mean_unknown():
    rows = [
        {"series_uid": "T1", "deploy_status": "IDENTITY_SELECTED",
         "deployment_gain_at_origin": 0.0,
         "fixed_program_gain_at_plus48": 0.0},
        {"series_uid": "T2", "deploy_status": "PREPARE_FAILED",
         "deployment_gain_at_origin": "UNKNOWN",
         "fixed_program_gain_at_plus48": "UNKNOWN"},
    ]
    summary = deploy1.summarize_fastonly_unit(
        group="g1", position=9, unit={"origin": 1656},
        population=["T1", "T2"], rows=rows)
    assert summary["deployment_gain_at_origin_mean"] is None
    assert summary["n_total"] == 2
    assert summary["n_scored"] == 1
    assert summary["n_missing_valid_decision"] == 1
    assert summary[
        "deployment_gain_at_origin_mean_of_scored_subset_diagnostic_only"] == 0.0
    assert len(summary["sequences"]) == 2


def test_missing_population_member_stays_in_denominator_as_unknown():
    rows = [{"series_uid": "T1", "deploy_status": "IDENTITY_SELECTED",
             "deployment_gain_at_origin": 0.0,
             "fixed_program_gain_at_plus48": 0.0}]
    summary = deploy1.summarize_fastonly_unit(
        group="g1", position=9, unit={},
        population=["T1", "T2"], rows=rows)
    assert [r["series_uid"] for r in summary["sequences"]] == ["T1", "T2"]
    assert summary["sequences"][1]["deployment_gain_at_origin"] == "UNKNOWN"
    assert summary["deployment_gain_at_origin_mean"] is None
    assert summary["n_total"] == 2


def test_two_legal_identity_zeros_still_mean_zero():
    rows = [
        {"series_uid": "T1", "deploy_status": "IDENTITY_SELECTED",
         "deployment_gain_at_origin": 0.0,
         "fixed_program_gain_at_plus48": 0.0},
        {"series_uid": "T2", "deploy_status": "IDENTITY_SELECTED",
         "deployment_gain_at_origin": 0.0,
         "fixed_program_gain_at_plus48": 0.0},
    ]
    summary = deploy1.summarize_fastonly_unit(
        group="g1", position=9, unit={},
        population=["T1", "T2"], rows=rows)
    assert summary["deployment_gain_at_origin_mean"] == 0.0
    assert summary["n_missing_valid_decision"] == 0


# ---------------------------------------------------------------------------
# A1.4 historical empty-selection read does not rewrite the file
# ---------------------------------------------------------------------------

def test_historical_empty_selection_view_does_not_rewrite_source(tmp_path):
    payload = {"sequences": [_historical_empty_row()]}
    path = tmp_path / "historical_empty.json"
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    path.write_text(text, encoding="utf-8")
    before = path.read_bytes()
    view = deploy1.read_sequences_file_view(path)
    assert path.read_bytes() == before
    assert view[0]["valid_mechanism_decision"] is False
    assert view[0]["deployment_gain_at_origin"] == "UNKNOWN"
    assert view[0]["original_deployment_gain_at_origin"] == 0.0
    assert view[0]["original_deploy_status"] == (
        "EMPTY_SELECTION_FALLBACK_TO_IDENTITY")
    loaded = deploy1.load_checkpoint_rows(json.loads(path.read_text()))
    assert path.read_bytes() == before
    assert loaded["T14"]["deployment_gain_at_origin"] == "UNKNOWN"
    stats = deploy1.mechanism_unit_stats(payload["sequences"], population_n=1)
    assert stats["origin_mean"] is None
    assert stats["n_missing_valid_decision"] == 1
    case = _render_case(_historical_empty_row(), include_effects=True)
    assert case["deployment_gain_at_origin"] == "UNKNOWN"
    assert case["original_recorded_gain_is_not_valid_utility_evidence"] is True
    assert case["original_recorded_gain_at_origin"] == 0.0
    assert case["valid_mechanism_decision"] is False


# ---------------------------------------------------------------------------
# A2 / A3.5 per-series menu pairs; true zero vs unmeasured
# ---------------------------------------------------------------------------

def _pq_menu():
    return {"per_position": {
        9: {
            "P": {
                "status": "READ", "aggregate_gain": 0.0,
                "harmed_fraction": 0.5, "max_single_series_harm": 1.0,
                "per_series_gain": {"T1": 1.0, "T2": -1.0},
            },
            "Q": {
                "status": "READ", "aggregate_gain": 0.0,
                "harmed_fraction": 0.5, "max_single_series_harm": 1.0,
                "per_series_gain": {"T1": -1.0, "T2": 1.0},
            },
            "identity": {
                "status": "READ", "aggregate_gain": 0.0,
                "harmed_fraction": 0.0, "max_single_series_harm": 0.0,
                "per_series_gain": {"T1": 0.0, "T2": 0.0},
            },
            "unmeasured": {"status": "UNAVAILABLE"},
        }
    }}


def test_menu_render_keeps_per_series_sign_flips_and_true_zeros():
    rendered = _render_menu(_pq_menu(), include_effects=True)
    p_map = rendered["9"]["P"]["per_series_gain"]
    q_map = rendered["9"]["Q"]["per_series_gain"]
    assert rendered["9"]["P"]["aggregate_gain"] == rendered["9"]["Q"][
        "aggregate_gain"] == 0.0
    assert p_map["T1"] == 1.0 and q_map["T1"] == -1.0
    assert p_map["T2"] == -1.0 and q_map["T2"] == 1.0
    assert rendered["9"]["identity"]["per_series_gain"]["T1"] == 0.0
    assert "per_series_gain" not in rendered["9"]["unmeasured"]
    assert rendered["9"]["unmeasured"] == {"status": "UNAVAILABLE"}


def test_cminus_menu_and_case_have_no_effect_or_winner_leak():
    menu = _pq_menu()
    r_menu = _render_menu(menu, include_effects=True)
    c_menu = _render_menu(menu, include_effects=True)
    cminus_menu = _render_menu(menu, include_effects=False)
    assert r_menu == c_menu
    blob = json.dumps(cminus_menu)
    assert "per_series_gain" not in blob
    assert "aggregate_gain" not in blob
    assert "winner" not in blob.lower()
    assert cminus_menu["9"]["P"] == {"status": "READ"}
    valid = {
        "series_uid": "T1", "origin": 1656, "public_features": {},
        "candidate_programs": {"cand_a": "outlier_iqr(k=1.5)"},
        "chosen_candidate_id": "cand_a",
        "deploy_status": "DEPLOYED",
        "deployed_program": "outlier_iqr(k=1.5)",
        "deployed_program_steps_reconstructed": [
            {"op": "outlier_iqr", "params": {"k": 1.5}}],
        "deployment_gain_at_origin": 0.4,
        "fixed_program_gain_at_plus48": 0.1,
        "action_facts": "UNKNOWN",
    }
    cminus_case = _render_case(valid, include_effects=False)
    assert "deployment_gain_at_origin" not in cminus_case
    assert "fixed_program_gain_at_plus48" not in cminus_case
    assert "original_recorded_gain_at_origin" not in cminus_case
    r_case = _render_case(valid, include_effects=True)
    c_case = _render_case(valid, include_effects=True)
    assert r_case == c_case
    assert r_case["deployment_gain_at_origin"] == 0.4
    historical_cminus = _render_case(_historical_empty_row(),
                                     include_effects=False)
    assert "deployment_gain_at_origin" not in historical_cminus
    assert "original_recorded_gain_at_origin" not in historical_cminus
    assert historical_cminus["valid_mechanism_decision"] is False


def test_compact_prepare_error_redacts_credential_text():
    result = _failed("Authorization: Bearer super-secret-api_key-value")
    blob = deploy1.compact_prepare_error(
        result=result, trace=_trace(), thrown=None,
        deploy_status="PREPARE_FAILED")
    assert blob is not None
    lowered = blob["error"].lower()
    assert "super-secret" not in lowered
    assert "api_key" not in lowered
    assert "bearer" not in lowered
    assert "redacted" in lowered


# ---------------------------------------------------------------------------
# A3 read-only combination / pool diagnosis (no Consumer, no LLM)
# ---------------------------------------------------------------------------

def test_propose_schema_allows_one_to_four_steps_and_three_candidates():
    schema = json.loads(
        (ROOT / "methods" / "ttha" / "schemas" / "fast_propose_v1.json")
        .read_text(encoding="utf-8"))
    candidates = schema["properties"]["candidates"]
    steps = candidates["items"]["properties"]["steps"]
    assert candidates["maxItems"] == 3
    assert steps["minItems"] == 1
    assert steps["maxItems"] == 4
    compiled = Program.from_steps(
        [("impute_linear", {}), ("outlier_iqr", {"k": 1.5})],
        source="combo_probe")
    assert len(compiled.steps) == 2
    assert [step.op for step in compiled.steps] == [
        "impute_linear", "outlier_iqr"]


def test_candidate_pool_truncates_to_total_k_including_identity():
    policy = json.loads(
        (ROOT / "methods" / "ttha" / "harness" / "h0"
         / "candidate_policy.json").read_text(encoding="utf-8"))
    assert policy["total_k"] == 4
    assert policy["agent_program_slots"] == 3
    programs = [
        Candidate.program_candidate(
            name,
            Program.from_steps([(op, params)], source=name),
            source="agent")
        for name, op, params in (
            ("a", "impute_linear", {}),
            ("b", "outlier_iqr", {"k": 1.5}),
            ("c", "outlier_mad", {"k": 3.5}),
            ("d", "winsorize", {"limits": 0.05}),
        )
    ]
    pool = CandidatePool.build(programs, total_k=int(policy["total_k"]))
    ids = [c.candidate_id for c in pool.candidates]
    assert ids[0] == "identity"
    assert len(ids) == 4
    assert "d" not in ids


def test_default_single_step_actionability_vs_two_step_compile():
    """Read-only: compile is not the same as 'the model will propose this'.

    Probes whether a default single-step candidate is rejected by the 0.35
    verifier while a two-step program that includes that operator would
    pass.  Does not change the operator menu.
    """
    from SelfEvolvingHarnessTS.methods.ttha.exploration_policy import (
        ExplorationPolicy,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _default_params_from_contract,
    )

    assert ExplorationPolicy().agent_proposals_kept == 1

    raw = np.concatenate([np.zeros(60, dtype=np.float64),
                          np.full(40, 1000.0, dtype=np.float64)])
    gappy = raw.copy()
    gappy[5:15] = np.nan
    fixtures = {
        "all_finite_spike40": raw,
        "gappy_spike40": gappy,
    }
    operators = (
        "denoise_stl", "winsorize", "outlier_iqr", "impute_linear",
        "period_median_complete",
    )
    single = {}
    for name in operators:
        params = _default_params_from_contract(name)
        for fixture_name, series in fixtures.items():
            candidate = Candidate.program_candidate(
                "probe_%s" % name,
                Program.from_steps([(name, params)], source="actionability_probe"),
                source="actionability_probe")
            artifact = verify_candidate(
                candidate, series, allowed_operators=(name,),
                maximum_modified_fraction=0.35)
            single[(name, fixture_name)] = {
                "selectable": bool(artifact.selectable),
                "rejection_code": artifact.receipt.rejection_code,
                "modified_fraction": artifact.receipt.modified_fraction,
                "params": params,
            }

    combos = [
        (("period_median_complete", "outlier_iqr"), "gappy_spike40"),
        (("impute_linear", "outlier_iqr"), "gappy_spike40"),
        (("impute_linear", "denoise_stl"), "gappy_spike40"),
        (("period_median_complete", "winsorize"), "gappy_spike40"),
    ]
    combo_rows = []
    misprune = None
    for (op_a, op_b), fixture_name in combos:
        series = fixtures[fixture_name]
        params_a = _default_params_from_contract(op_a)
        params_b = _default_params_from_contract(op_b)
        program = Program.from_steps(
            [(op_a, params_a), (op_b, params_b)], source="combo_probe")
        candidate = Candidate.program_candidate(
            "combo", program, source="combo_probe")
        artifact = verify_candidate(
            candidate, series, allowed_operators=(op_a, op_b),
            maximum_modified_fraction=0.35)
        row = {
            "ops": (op_a, op_b),
            "fixture": fixture_name,
            "selectable": bool(artifact.selectable),
            "rejection_code": artifact.receipt.rejection_code,
            "modified_fraction": artifact.receipt.modified_fraction,
        }
        combo_rows.append(row)
        a_fail = not single[(op_a, fixture_name)]["selectable"]
        b_fail = not single[(op_b, fixture_name)]["selectable"]
        if row["selectable"] and (a_fail or b_fail):
            misprune = {
                "combo": row,
                "single_a": single[(op_a, fixture_name)],
                "single_b": single[(op_b, fixture_name)],
            }

    # Combination is compilable on this fixture set.  Whether default
    # single-step actionability drops a combo-legal operator is a finding
    # for the report, not a code change in this task.
    assert any(len(Program.from_steps(
        [(a, _default_params_from_contract(a)),
         (b, _default_params_from_contract(b))],
        source="combo_probe").steps) == 2 for a, b in (("impute_linear",
                                                        "outlier_iqr"),))
    diagnosis = {
        "single_step_default": {("%s@%s" % k): v for k, v in single.items()},
        "two_step": combo_rows,
        "misprune_example": misprune,
        "note": (
            "compilable two-step != model-proposed two-step; default "
            "agent_proposals_kept=1 and candidate_policy.total_k=4 further "
            "truncate the pool after propose maxItems=3"
        ),
    }
    # Keep the diagnosis inspectable in the pytest traceback if needed.
    assert "single_step_default" in diagnosis
    if misprune is not None:
        assert misprune["combo"]["selectable"] is True
        assert (
            not misprune["single_a"]["selectable"]
            or not misprune["single_b"]["selectable"]
        )


# Root review: exercise integration/resume/aggregation, not only helpers.

def _valid_row(uid="T1", gain=0.0, delayed=0.0):
    return {"series_uid": uid, "deploy_status": "IDENTITY_SELECTED",
            "deployed_program": "identity", "faults": [],
            "deployment_gain_at_origin": gain,
            "fixed_program_gain_at_plus48": delayed,
            "llm_requests_sent": 0}


def test_unsuccessful_receipt_and_invalid_status_are_not_valid_choices():
    for result in (
        SimpleNamespace(status=PreparationStatus.PREPARED,
                        receipt=ExecutionReceipt(False, "execution failed")),
        SimpleNamespace(status="bad_status", receipt=ExecutionReceipt(True)),
    ):
        outcome = deploy1.interpret_prepare_outcome(
            result=result, trace=_trace(chosen_candidate_id="identity"))
        assert outcome["deploy_status"] == "INVALID_PREPARATION_RESULT"
        assert deploy1.apply_legality_and_score(outcome)[
            "deployment_gain_at_origin"] == "UNKNOWN"


def test_internal_failure_is_counted_and_sensitive_prefix_not_retained():
    record = deploy1.finalize_fastonly_decision(
        {"faults": []}, result=_failed("api_key=do-not-retain: failed"),
        trace=_trace(), scorer=None, ctx=None, uid="T14", origin=1,
        delayed_origin=2, position=9)
    assert record["faults"][0]["kind"] == "PREPARE_FAILED"
    assert "do-not-retain" not in json.dumps(record)


def test_historical_pooled_alias_retains_original_zero_only_as_annotation():
    row = _historical_empty_row(
        deployment_gain_at_origin="UNKNOWN",
        fixed_program_gain_at_plus48="UNKNOWN",
        original_recorded_gain_at_origin=0.0,
        original_recorded_gain_at_plus48=0.0)
    case = _render_case(row, include_effects=True)
    assert case["deployment_gain_at_origin"] == "UNKNOWN"
    assert case["original_recorded_gain_at_origin"] == 0.0


def test_partial_legacy_resume_does_not_overwrite_or_requery_old_row(
        tmp_path, monkeypatch):
    source = tmp_path / "u009.json"
    source.write_text(json.dumps({"sequences": [_historical_empty_row()]}),
                      encoding="utf-8")
    before = source.read_bytes()
    calls = []

    def fake_decision(**kwargs):
        calls.append(kwargs["uid"])
        return _valid_row(kwargs["uid"])

    monkeypatch.setattr(deploy1, "run_fastonly_decision", fake_decision)
    monkeypatch.setattr(deploy1.seq3, "SequenceGuard", lambda **kwargs: None)
    scorer = SimpleNamespace(ctx={9: SimpleNamespace(unit={})},
                             population=lambda p: ["T14", "T2"], physical_fits=0)
    budget = SimpleNamespace(
        ledgers=[], llm_by_arm={}, require=lambda n: None,
        spend_llm=lambda *a: None, spend_fits=lambda *a: None,
        note_evaluation=lambda *a: None)
    args = dict(group="g1", position=9, scorer=scorer, snapshot=None,
                machinery={}, budget=budget, checkpoint_path=source,
                concurrency=1)
    first = deploy1.run_fastonly_unit(**args)
    assert calls == ["T2"]
    assert first["deployment_gain_at_origin_mean"] is None
    assert source.read_bytes() == before
    assert (tmp_path / "u009.feedback_integrity.json").is_file()
    second = deploy1.run_fastonly_unit(**args)
    assert calls == ["T2"]
    assert second["n_missing_valid_decision"] == 1
    assert source.read_bytes() == before


def test_account_failure_stops_following_units(monkeypatch):
    calls = []

    def stopped_unit(**kwargs):
        calls.append(kwargs["position"])
        return {"stopped_at": {"why": "AccountFault"}}

    monkeypatch.setattr(deploy1, "run_fastonly_unit", stopped_unit)
    result = deploy1.run_fastonly_group(
        group="g1", scorer=None, snapshot=None, machinery={}, budget=None,
        positions=(9, 10))
    assert calls == [9]
    assert result["status"] == "BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT"


def test_part1_cli_returns_failure_for_incomplete_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy1, "run_part_b", lambda **kw: {
        "status": "RAN", "groups": {"g1": {
            "status": "COMPLETE", "units": {9: {
                "deployment_gain_at_origin_mean": None}}}}})
    code = deploy1.main(["--part", "fastonly", "--group", "g1",
                         "--position", "9", "--out", str(tmp_path / "out.json")])
    assert code == 2


def test_part2_aggregation_rechecks_legacy_rows_not_old_mean(tmp_path, monkeypatch):
    from evaluation.main_protocol_p4 import run_dev_deploy2 as deploy2
    monkeypatch.setattr(deploy2, "ART", tmp_path)
    original_bytes = {}
    for position in deploy2.REVIEW_POSITIONS:
        path = tmp_path / ("dev_deploy2_fastonly__g1_F_u%s.json" % position)
        payload = {"decision_population": ["T14", "T2"],
                   "deployment_gain_at_origin_mean": 0.5,
                   "sequences": [_historical_empty_row(), _valid_row("T2", 1.0)]}
        path.write_text(json.dumps(payload), encoding="utf-8")
        original_bytes[path] = path.read_bytes()
    result = deploy2._arm_review_mean("g1", "F", "fixture")
    assert result["complete"] is False
    assert all(path.read_bytes() == content for path, content in original_bytes.items())


def test_part2_partial_delayed_does_not_use_readable_subset(tmp_path, monkeypatch):
    from evaluation.main_protocol_p4 import run_dev_deploy2 as deploy2
    monkeypatch.setattr(deploy2, "ART", tmp_path)
    for position in deploy2.REVIEW_POSITIONS:
        path = tmp_path / ("dev_deploy2_fastonly__g1_F_u%s.json" % position)
        payload = {"decision_population": ["T1", "T2"],
                   "deployment_gain_at_origin_mean": 0.0,
                   "sequences": [_valid_row("T1"),
                                 _valid_row("T2", delayed="UNKNOWN")]}
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = deploy2._arm_review_mean("g1", "F", "fixture")
    assert result["complete"] is True  # Origin is complete, +48 is not.
    assert result["equal_weighted_origin_mean"] == 0.0
    assert result["equal_weighted_delayed_mean"] is None
    assert deploy2._paired_diff(result, result, "equal_weighted_delayed_mean") == "UNKNOWN"


def test_menu_typed_parameters_are_neutral_in_all_arms(monkeypatch):
    from evaluation.main_protocol_p4 import run_dev_deploy2 as deploy2
    monkeypatch.setattr(deploy2, "FORMATION_POSITIONS", (9,))
    monkeypatch.setattr(deploy1, "legal_forecast_menu", lambda: ["impute_linear"])
    scorer = SimpleNamespace(
        ctx={9: SimpleNamespace(origin=100, config={"period": 12})},
        population=lambda p: ["T1"],
        population_reading=lambda *args: {
            "aggregate_gain": 0.2, "harmed_fraction": 0.0,
            "max_single_series_harm": 0.0, "per_series_gain": {"T1": 0.2}})
    menu = deploy2.menu_readings_at_formation(scorer)
    full = _render_menu(menu, include_effects=True)
    neutral = _render_menu(menu, include_effects=False)
    assert full["9"]["impute_linear"]["program_steps"] == neutral["9"][
        "impute_linear"]["program_steps"]
    assert neutral["9"]["identity"]["program_steps"] == []
    assert "per_series_gain" not in json.dumps(neutral)
    assert "aggregate_gain" not in json.dumps(neutral)
