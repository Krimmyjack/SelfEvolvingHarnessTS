"""S2a Part P: forecast four-arm live course + G2 silence guard.

Reduced course (sol pre-authorised). 0 new data sources. Oracle files are
not opened during the arm phase. Classification supply card v0 is preloaded
on A5/K0 only; K0 itself is empty of forecast knowledge.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
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

import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_ps0c_ps1 as ps0c  # noqa: E402
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402
import run_e2_s2a_electricity_sweep as elec  # noqa: E402
import run_e2_s2a_forecast_oracle as traffic  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402
import run_v1_kdd2018_natural_slow_update as kdd  # noqa: E402

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    deployment_constraints_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    MEASURED_EFFECT_KEY,
    classify_relation,
    task_consumer_key,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor,
)
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_NAMES,
)
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    source_skill_of_candidate,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    skill_revision as sr,
    source_skill as ss,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
FREEZE_JSON = E2 / "s2a_course_frozen.json"
GATES_JSON = E2 / "sa1_minimal_gates.json"
OUT_JSON = E2 / "s2a_g1_run1.json"
OUT_MD = E2 / "s2a_g1_run1.md"
CHECKPOINT = E2 / "s2a_g1_run1.checkpoint.json"

PROTOCOL = "s2a_forecast_curriculum_v1"
CLS_SKILL_ID = "sa1_supply_scope_v2"
FORECAST_SKILL_ID = "s2a_forecast_supply_v0"
ARM_STATIC = s1.ARM_STATIC
ARM_A3 = s1.ARM_A3
ARM_K0 = s1.ARM_K0
ARM_A5 = s1.ARM_A5
ARMS = (ARM_STATIC, ARM_A3, ARM_K0, ARM_A5)
ADAPTIVE = (ARM_A3, ARM_K0, ARM_A5)
# Origin tokens must be prefix lengths of request.values: run_online_round
# rebinds the tool gateway from series0[:origin] (online_loop.py:373-380),
# so a small token rebuilds the gateway on a 1-point slice and prepare's
# verify_context refuses. The S1 shape (block.size / block.size+1) is kept.
SUPPORT_TRIAL_BUDGET = 2
LLM_CAP = 120
FIT_CAP = 200
WALL_SECONDS = int(5 * 60 * 60)
FIT_PER_UNIT = 16
HARM_BAR = 0.005
EXPERIMENT_PROGRAMS = (
    "outlier_iqr", "outlier_mad", "hampel_filter", "winsorize",
)

PRE_REGISTERED = [
    {"id": "P1", "claim": "v1.1 adapter: classification 149 green; no Skill/Memory semantic change"},
    {"id": "P2", "claim": "G2: classification card retrieval/scope_match/supply are all zero on every forecast unit"},
    {"id": "P3", "claim": "if the producer earns a dual-gate POSITIVE, a strong beneficiary converts the supplied forecast card"},
    {"id": "P4", "claim": "conflict-field R2: N/A on the reduced course"},
    {"id": "P5", "claim": "harm events are zero in every arm"},
    {"id": "P6", "claim": "K0-fixed ≡ A3-reset (empty forecast K0; classification card silent). Material divergence is a variance alarm"},
    {"id": "P7", "claim": "N/A (reduced course: no natural conflict field, R2 untested)"},
]


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


def _config() -> dict[str, object]:
    cfg = dict(kdd._config())
    cfg["period"] = traffic.PERIOD
    cfg["dataset_id"] = "s2a_forecast_curriculum"
    return cfg


def _task_spec() -> Any:
    forbidden = tuple(sorted(n for n in OPERATOR_NAMES
                             if n not in EXPERIMENT_PROGRAMS))
    return forecast_task_spec_v1(
        horizon=48,
        downstream_model_class=traffic.CONSUMER_ID,
        metric=MetricSpec(traffic.METRIC, "lower_is_better"),
        forbidden_modifications=forbidden,
    )


def _task_context(spec: Any) -> Any:
    return forecast_task_context_v1(
        task_spec=spec,
        deployment_constraints=deployment_constraints_v1(
            constraint_id="s2a-forecast-fixed-consumer-v1",
            fixed_downstream_model_id="fixed:pooled-ridge-a1",
            maximum_candidates=1 + SUPPORT_TRIAL_BUDGET,
            maximum_modified_fraction=0.35),
    )


def _live_agent(block: Any, backend: Any) -> Any:
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        LocalPublicToolGateway,
    )
    from evaluation.functional.task_episode_harness.agentic.runner import (
        live_transport,
    )

    target = live_transport(default_model=cls.SLOW_MODEL)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(block, task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    return TTHAFastAgent(core)


def _card_builder(_episode: object) -> Mapping[str, object]:
    return {"pattern_id": "s2a-forecast-block",
            "observable_signature": {"task_kind": "forecast"}}


class ForecastEval:
    """bch._evaluate_assignment wrapped as ScopeExecutor.evaluate_fn."""

    def __init__(self, budget: cls.FitBudget, ledger: dict[str, int]) -> None:
        self.budget = budget
        self.ledger = ledger
        self._memo: dict[tuple[Any, ...], dict[str, Any]] = {}

    def __call__(self, roster, values, compiled, config, *, origin):
        train = tuple(str(r["series_uid"]) for r in roster
                      if str(r["role"]) == "train")
        sig = (train, int(origin), id(compiled) if compiled is not None else 0)
        if compiled is None and sig in self._memo:
            return dict(self._memo[sig])
        assignment = {uid: compiled for uid in train} if compiled is not None else {}
        self.budget.spend(1)
        self.ledger["fit"] = int(self.ledger.get("fit") or 0) + 1
        raw = bch._evaluate_assignment(
            roster, values, assignment, config, origin=origin)
        out = {
            "mean_smase": float(raw["mean_smase"]),
            "per_view_smase": [float(x) for x in raw["per_view_smase"]],
            "behavior_point_count": int(raw.get("behavior_point_count") or 0),
        }
        if compiled is None:
            self._memo[sig] = dict(out)
        return out


class FaceExecutor:
    """Dispatch Support/delayed tokens onto two real-origin executors."""

    def __init__(self, faces: Mapping[int, ScopeExecutor],
                 real_origin: int) -> None:
        self._faces = dict(faces)
        self._real = int(real_origin)

    def evaluate(self, steps, origin):
        token = int(origin)
        if token not in self._faces:
            raise KeyError("unknown face token %s" % token)
        return self._faces[token].evaluate(steps, self._real)


def _roster(train: Sequence[str], eval_uids: Sequence[str]
            ) -> list[dict[str, str]]:
    return ([{"series_uid": uid, "role": "train"} for uid in train]
            + [{"series_uid": uid, "role": "eval"} for uid in eval_uids])


def _h0():
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    return compile_snapshot(PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
                            verify_lock=False)


def _cls_card() -> dict[str, Any]:
    gates = json.loads(GATES_JSON.read_text(encoding="utf-8"))
    card = dict(gates["card_v0"])
    return {key: card[key] for key in s1._SKILL_ENTRY_FIELDS if key in card}


def _install(base: Any, card: Mapping[str, Any], *, store_root: Path,
             tag: str) -> Any:
    snapshot, _applied = s1._apply_entries(base, [card],
                                           store_root=store_root, tag=tag)
    return snapshot


class CellBank:
    def __init__(self) -> None:
        self._e_names: list[str] | None = None
        self._e_pool: dict[str, np.ndarray] | None = None
        self._t_names: list[str] | None = None
        self._t_pool: dict[str, np.ndarray] | None = None
        self._cells: dict[str, dict[str, Any]] = {}

    def _elec(self) -> tuple[list[str], dict[str, np.ndarray]]:
        if self._e_names is None:
            names, pool, _meta = elec._load_pool()
            self._e_names, self._e_pool = names, pool
        return self._e_names, self._e_pool

    def _traf(self) -> tuple[list[str], dict[str, np.ndarray]]:
        if self._t_names is None:
            names, pool = traffic._load_pool()
            self._t_names, self._t_pool = names, pool
        return self._t_names, self._t_pool

    def get(self, unit_id: str, freeze: Mapping[str, Any]) -> dict[str, Any]:
        if unit_id in self._cells:
            return self._cells[unit_id]
        if unit_id.startswith("electricity_impulsive_outlier_"):
            names, pool = self._elec()
            recut, _left = elec._recut(names)
            spec = next(c for c in recut if c["unit_id"] == unit_id)
            values = traffic._inject(spec, pool)
        elif unit_id == "traffic_gap_00":
            names, pool = self._traf()
            spec = next(c for c in traffic._recut(names)
                        if c["unit_id"] == unit_id)
            values = traffic._inject(spec, pool)
        elif unit_id == "traffic_clean_identity_00":
            spec = dict(freeze["clean_cell"])
            names, pool = self._traf()
            values = {uid: np.asarray(pool[uid], dtype=np.float64)
                      for uid in list(spec["train"]) + list(spec["heldout"])}
        else:
            raise Stop("CELL_CONSTRUCTION_FAILED", "unknown unit " + unit_id)
        first = np.asarray(values[str(spec["train"][0])], dtype=np.float64)
        cell = {
            **spec,
            "values": values,
            "observation_block": first[:traffic.ORIGIN_HELDIN].copy(),
            "condition": traffic.CONDITION,
            "source": spec.get("source") or spec.get("family"),
        }
        self._cells[unit_id] = cell
        return cell


def _features(block: np.ndarray) -> dict[str, Any]:
    return dict(extract_public_features(block, task_kind="forecast"))


def _pattern_view(features: Mapping[str, Any]) -> dict[str, Any]:
    binned = s1._binned_contract_leaves(features)
    keep = set(s1a.PATTERN_KEYS) | {"task_kind"}
    return {key: binned[key] for key in binned if key in keep}


def _candidate_sources(record: Mapping[str, Any], *, skill_id: str | None,
                       card_ops: Sequence[str]) -> dict[str, Any]:
    from evaluation.functional.run_e2_capstone_epilepsy2 import dedup_swallowed

    proposals = list(record.get("proposals") or ())
    supplied = [row for row in proposals
                if str(row.get("candidate_id", "")).startswith("cand_skill_")]
    self_proposed = [
        row for row in proposals
        if not str(row.get("candidate_id", "")).startswith("cand_skill_")
        and str(row.get("candidate_id")) != "identity"]
    swallowed = {"dedup_swallowed": False}
    if skill_id:
        swallowed = dedup_swallowed(record, skill_id=skill_id,
                                    card_operators=card_ops)
    return {
        "supplied": len(supplied),
        "self_proposed": len(self_proposed),
        "dedup_swallowed": bool(swallowed.get("dedup_swallowed")),
        "supplied_ids": [str(row.get("candidate_id")) for row in supplied],
        "self_proposed_ids": [str(row.get("candidate_id"))
                              for row in self_proposed],
        "dedup_detail": swallowed,
    }


def _g2_faces(record: Mapping[str, Any]) -> dict[str, int]:
    retrieved = [str(x) for x in (record.get("retrieved_skill_ids") or [])]
    scope = dict(record.get("scope_match_by_skill_id") or {})
    supplied = 0
    for episode in record.get("episodes") or []:
        if str(episode.get("source_skill_id") or "") == CLS_SKILL_ID:
            supplied += 1
    for cand in record.get("pool") or []:
        if source_skill_of_candidate(cand) == CLS_SKILL_ID:
            supplied += 1
            break
    return {
        "retrieval": 1 if CLS_SKILL_ID in retrieved else 0,
        "scope_match": 1 if bool(scope.get(CLS_SKILL_ID)) else 0,
        "supply": 1 if supplied else 0,
    }


def _run_round(*, state: Mapping[str, Any], cell: Mapping[str, Any],
               unit_id: str, arm: str, fit_budget: cls.FitBudget,
               ledger: Any, fit_ledger: dict[str, int]) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
        activate_approved, open_delayed, run_online_round,
    )

    values = cell["values"]
    cfg = _config()
    spec = _task_spec()
    ctx = _task_context(spec)
    ev = ForecastEval(fit_budget, fit_ledger)
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    support_origin = int(block.size)
    delayed_origin = support_origin + 1
    support_ex = ScopeExecutor(
        _roster(cell["delayed"], cell["support"]), values, cfg,
        evaluate_fn=ev, max_modified_fraction=0.35)
    delayed_ex = ScopeExecutor(
        _roster(cell["support"], cell["delayed"]), values, cfg,
        evaluate_fn=ev, max_modified_fraction=0.35)
    executor = FaceExecutor(
        {support_origin: support_ex, delayed_origin: delayed_ex},
        traffic.ORIGIN_HELDIN)
    observed = dict(resolver.window_context(values, traffic.ORIGIN_HELDIN,
                                            traffic.PERIOD))
    observed["bound_period"] = float(traffic.PERIOD)
    request = PreparationRequest(
        "s2a-%s" % unit_id, block, spec, dict(observed), task_context=ctx)
    features = _features(block)
    method = state["method"]
    method.bind_round_data(block, task_kind="forecast")
    started = time.time()
    llm_before = int(getattr(ledger, "calls", 0) or 0)
    entries_before = {str(s.skill_id): s
                      for s in method._active_snapshot().skills}
    skills_before = set(entries_before)
    result = run_online_round(
        method, executor, request, values,
        origin=support_origin, slow_agent=None,
        controller=state["controller"], store=state["store"],
        card_builder=_card_builder,
        round_name="%s_%s_r1" % (arm.lower(), unit_id),
        budget=SUPPORT_TRIAL_BUDGET, allow_slow=False,
        domain=unit_id,
        period=traffic.PERIOD, fast_features=features,
        allow_fast_skill=True, runtime_prior_slot=False)
    open_delayed(result, executor, delayed_origin=delayed_origin,
                 store=state["store"])
    activated = False
    if result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
        if activated:
            state["approved_skill_ids"].append(str(result.approved_skill_id))
    incumbent_before = state.get("incumbent")
    state["incumbent"] = cls._incumbent_after_delayed(result, incumbent_before)
    skills_after = {str(s.skill_id) for s in method._active_snapshot().skills}
    minted = sorted(skills_after - skills_before)
    trace = method.last_trace
    steps_map = dict(getattr(trace, "candidate_program_steps", None) or {})
    candidate_ids = [str(item) for item
                     in (getattr(trace, "candidate_ids", ()) or ())]
    chosen_id = getattr(trace, "chosen_candidate_id", None)
    probed_by_id = {}
    for probe in result.actual_probed_programs:
        probed_by_id.setdefault(str(probe.get("candidate_id")), probe)
    proposals = []
    for candidate_id in candidate_ids:
        steps = steps_map.get(candidate_id)
        operators = s1._steps_operators(steps)
        probe = probed_by_id.get(candidate_id)
        if probe is not None:
            outcome = str(probe.get("kind"))
        elif candidate_id not in steps_map:
            outcome = "dropped_before_probe_no_compiled_steps"
        elif candidate_id == "identity":
            outcome = "identity_baseline"
        else:
            outcome = "not_reached_support_budget_exhausted"
        proposals.append({
            "candidate_id": candidate_id,
            "operators": operators,
            "family": s1._family_of(operators),
            "compiled": candidate_id in steps_map,
            "chosen_by_select": candidate_id == chosen_id,
            "outcome": outcome,
            "gain": (probe or {}).get("gain"),
            "source_skill_id": source_skill_of_candidate(candidate_id),
        })
    probes = []
    for probe in result.actual_probed_programs:
        cid = str(probe.get("candidate_id"))
        operators = s1._steps_operators(steps_map.get(cid))
        probes.append({
            "candidate_id": probe.get("candidate_id"),
            "kind": probe.get("kind"),
            "gain": probe.get("gain"),
            "passed": probe.get("passed"),
            "operators": operators,
            "family": s1._family_of(operators),
        })
    retrieved = [str(item) for item in
                 (getattr(trace, "retrieved_skill_ids", ()) or ())]
    entry_shas = {sid: ss.skill_content_sha(entry)
                  for sid, entry in entries_before.items()}
    fresh_ids = set(result.episode_ids)
    fresh = [e for e in method.experience_episodes if e.episode_id in fresh_ids]
    record = {
        "unit_id": unit_id,
        "round": "r1",
        "arm": arm,
        "dataset": cell.get("dataset"),
        "task_consumer_key": task_consumer_key(spec),
        "pool": candidate_ids,
        "chosen": chosen_id,
        "retrieved_skill_ids": retrieved,
        "fast_features_binned": s1._binned_contract_leaves(features),
        "pattern_view": _pattern_view(features),
        "scope_match_by_skill_id": s1._scope_match_by_skill_id(
            entries_before, features),
        "guidance_conditioned_by_skill_id": s1._guidance_conditioned_by_skill_id(
            entries_before, retrieved),
        "installed_skill_content_sha": dict(sorted(entry_shas.items())),
        "proposals": proposals,
        "probes": probes,
        "winner_program": s1._plain(result.winner_program),
        "abstained": bool(result.abstained),
        "harm_count": int(result.harm_count),
        "delayed_utility": result.delayed_utility,
        "approved_skill_id": result.approved_skill_id,
        "activated": activated,
        "incumbent_before_round": s1._plain(incumbent_before),
        "incumbent_after_round": s1._plain(state["incumbent"]),
        "minted_skill_ids": minted,
        "episodes": [{
            "episode_id": e.episode_id,
            **s1._episode_attribution(e, entries_before, entry_shas),
            "workflow_signature": e.workflow_signature,
            "relation": e.relation,
            "evidence_level": e.evidence_level,
            "local_status": e.local_status,
            "support_gain": (e.support_response or {}).get("gain"),
            "delayed_gain": (e.delayed_response or {}).get("gain"),
            "support_effect": s1._plain(
                (e.support_response or {}).get(MEASURED_EFFECT_KEY)),
            "delayed_effect": s1._plain(
                (e.delayed_response or {}).get(MEASURED_EFFECT_KEY)),
        } for e in fresh],
        "consumer_fits_after": fit_budget.used,
        "llm_calls_this_round": (
            int(getattr(ledger, "calls", 0) or 0) - llm_before),
        "seconds": round(time.time() - started, 2),
    }
    record["g2"] = _g2_faces(record)
    return record


def _score_heldout(*, cell: Mapping[str, Any], applied: Sequence[Mapping[str, Any]],
                   fit_budget: cls.FitBudget, fit_ledger: dict[str, int]
                   ) -> dict[str, Any]:
    cfg = _config()
    roster = _roster(cell["train"], cell["heldout"])
    ev = ForecastEval(fit_budget, fit_ledger)
    identity = ev(roster, cell["values"], None, cfg,
                  origin=traffic.ORIGIN_HELDOUT)
    compiled = None
    if applied:
        compiled = cls._FrozenCompiled(applied)
    scored = ev(roster, cell["values"], compiled, cfg,
                origin=traffic.ORIGIN_HELDOUT)
    id_smase = float(identity["mean_smase"])
    sc_smase = float(scored["mean_smase"])
    gain = id_smase - sc_smase
    per = [float(a) - float(b) for a, b in zip(
        identity["per_view_smase"], scored["per_view_smase"])]
    worst = min(per, default=0.0)
    relation = classify_relation(
        aggregate_gain=gain, is_identity=not applied,
        consumer_id=traffic.CONSUMER_ID, material_threshold=HARM_BAR)
    return {
        "heldout_identity_smase": id_smase,
        "heldout_smase": sc_smase,
        "heldout_gain": gain,
        "heldout_per_series_gain": per,
        "worst_series_gain": worst,
        "harm_event": bool(worst < -HARM_BAR),
        "heldout_relation": relation.get("relation"),
    }


def _deploy(*, state: Mapping[str, Any], cell: Mapping[str, Any],
            arm: str, fit_budget: cls.FitBudget,
            fit_ledger: dict[str, int]) -> dict[str, Any]:
    features = _features(np.asarray(cell["observation_block"], dtype=np.float64))
    if arm == ARM_STATIC:
        decision = {
            "applied_steps": [],
            "source": "identity",
            "active_skill_id": None,
            "recall_hit": False,
            "view_skill_ids": [],
            "why": "Static: identity frozen on every unit",
        }
    else:
        decision = cls._frozen_recall(state, features)
        if not decision.get("applied_steps") and state.get("incumbent"):
            decision = {
                "applied_steps": s1._plain(state["incumbent"]),
                "source": "incumbent_after_delayed",
                "active_skill_id": None,
                "recall_hit": False,
                "view_skill_ids": decision.get("view_skill_ids") or [],
                "why": "delayed-approved incumbent; delayed refuse keeps identity",
            }
    applied = list(decision.get("applied_steps") or [])
    scored = _score_heldout(cell=cell, applied=applied,
                            fit_budget=fit_budget, fit_ledger=fit_ledger)
    return {
        "arm": arm,
        "deploy_source": decision.get("source"),
        "deploy_why": decision.get("why"),
        "active_skill_id": decision.get("active_skill_id"),
        "applied_program": applied,
        "view_skill_ids": decision.get("view_skill_ids"),
        **scored,
    }


def run_unit(*, unit: Mapping[str, Any], cell: Mapping[str, Any], arm: str,
             base_snapshot: Any, backend: Any, store_root: Path,
             fit_ledger: dict[str, int]) -> dict[str, Any]:
    unit_id = str(unit["unit_id"])
    s1._set_phase(s1.PHASE_ARM, unit=unit_id, arm=arm)
    fit_budget = cls.FitBudget(FIT_PER_UNIT)
    llm_before = int(getattr(backend, "calls", 0) or 0)
    started = time.time()
    state = s1._new_state(
        snapshot=base_snapshot,
        agent=_live_agent(cell["observation_block"],
                          backend.new_arm_backend()),
        store_root=store_root, tag="%s_%s" % (arm.replace("-", "_"), unit_id))
    records: list[dict[str, Any]] = []
    if arm != ARM_STATIC:
        records.append(_run_round(
            state=state, cell=cell, unit_id=unit_id, arm=arm,
            fit_budget=fit_budget, ledger=backend, fit_ledger=fit_ledger))
    deployment = _deploy(state=state, cell=cell, arm=arm,
                         fit_budget=fit_budget, fit_ledger=fit_ledger)
    end_snapshot = state["method"]._active_snapshot()
    return {
        "unit_id": unit_id,
        "role": unit["role"],
        "source": unit.get("source"),
        "arm": arm,
        "base_runtime_bundle_sha": state["base_sha"],
        "end_runtime_bundle_sha": end_snapshot.runtime_bundle_sha,
        "end_skill_ids": sorted(str(s.skill_id) for s in end_snapshot.skills),
        "rounds": records,
        "deployment": deployment,
        "approved_skill_ids": list(state["approved_skill_ids"]),
        "llm_calls": int(getattr(backend, "calls", 0) or 0) - llm_before,
        "consumer_fits": fit_budget.used,
        "seconds": round(time.time() - started, 2),
        "g2": (records[0].get("g2") if records else
               {"retrieval": 0, "scope_match": 0, "supply": 0}),
        "_end_snapshot": end_snapshot,
        "_state": state,
    }


def _live_units(freeze: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    pos = 0
    for row in freeze["course"]:
        if row["role"] == "boundary_compile":
            continue
        pos += 1
        item = dict(row)
        item["position"] = pos
        item["counts_toward_cumulative_regret"] = True
        out.append(item)
    return out


def _supply_row(scored: Mapping[str, Any]) -> dict[str, Any] | None:
    for record in scored.get("rounds") or []:
        pattern = dict(record.get("pattern_view") or
                       record.get("fast_features_binned") or {})
        key = str(record.get("task_consumer_key") or "")
        parts = key.split("|")
        task_kind = parts[0] if len(parts) == 3 else "forecast"
        consumer = parts[1] if len(parts) == 3 else traffic.CONSUMER_ID
        metric = parts[2] if len(parts) == 3 else traffic.METRIC
        for episode in record.get("episodes") or []:
            if str(episode.get("relation")) != "POSITIVE":
                continue
            if str(episode.get("local_status")) != "LOCAL_ACTIVE":
                continue
            sig = str(episode.get("workflow_signature") or "")
            if not sig or sig in ("identity", "unknown"):
                continue
            return {
                "task_episode_id": str(scored["unit_id"]),
                "unit_id": str(scored["unit_id"]),
                "run_id": "%s@r1" % scored["unit_id"],
                "program": sig,
                "relation": "POSITIVE",
                "conditioned_snapshot": False,
                "task_kind": task_kind,
                "consumer_id": consumer,
                "metric": metric,
                "pattern": pattern,
                "support_gain": float(episode.get("support_gain") or 0.0),
                "delayed_gain": float(episode.get("delayed_gain") or 0.0),
            }
    return None


def _compile_forecast_card(row: Mapping[str, Any]) -> dict[str, Any]:
    compiled = ss.compile_supply_tier(
        [row], skill_id=FORECAST_SKILL_ID,
        legal_features=ss._edit_schema_features(PROJECT_ROOT),
        pattern_family=None,
        pattern_axis_provenance=(
            "n=1 source intersection of the producer Pattern view; "
            "forecast extractor + observable_numeric_bin + "
            "s1a.PATTERN_KEYS. No S1a forecast cluster exists; "
            "inventing one is forbidden."))
    return compiled


def _revise_forecast(snapshot: Any, scored: Mapping[str, Any],
                     unit: Mapping[str, Any], *, store_root: Path,
                     card_sha: str) -> dict[str, Any]:
    entry = next((s for s in snapshot.skills
                  if str(s.skill_id) == FORECAST_SKILL_ID), None)
    if entry is None:
        return {"applied": [], "snapshot": snapshot, "card_sha": card_sha}
    rounds = scored.get("rounds") or []
    scope_ok = any(bool((r.get("scope_match_by_skill_id") or {})
                        .get(FORECAST_SKILL_ID)) for r in rounds)
    supplied = [e for r in rounds for e in (r.get("episodes") or [])
                if str(e.get("source_skill_id") or "") == FORECAST_SKILL_ID]
    converted = [e for e in supplied
                 if str(e.get("relation")) == "POSITIVE"
                 and str(e.get("local_status")) == "LOCAL_ACTIVE"]
    refused = [e for e in supplied if e not in converted]
    guards = dict(entry.risk_guards or {})
    ast = json.loads(json.dumps(s1._plain(entry.observable_applicability)))
    applied: list[str] = []
    narrowed = None
    for episode in converted:
        guards = sr.append_evidence_row(guards, {
            "rule": "R1_positive",
            "unit_id": str(unit["unit_id"]),
            "support_gain": float(episode.get("support_gain") or 0.0),
            "delayed_gain": float(episode.get("delayed_gain") or 0.0),
            "heldout_gain": float((scored.get("deployment") or {})
                                  .get("heldout_gain") or 0.0),
            "card_content_sha_when_earned": card_sha,
            "counts_toward_authorization": False,
        })
        applied.append("R1")
    harmed = bool((scored.get("deployment") or {}).get("harm_event"))
    if (refused or harmed) and scope_ok:
        source_views = [dict(row.get("pattern_view") or {}) for row in
                        (guards.get("evidence") or {}).get("sources") or []]
        refusing_view = dict((rounds[0] if rounds else {}).get("pattern_view")
                             or {})
        exclusion = sr.compile_exclusion(
            refusing_view=refusing_view, source_views=source_views,
            axes=sr.contracted_axes(ss._edit_schema_features(PROJECT_ROOT)))
        rule = "R3_negative_or_harm" if (
            harmed or any(str(e.get("relation")) == "NEGATIVE"
                          for e in refused)) else "R2_conflict"
        if exclusion["leaves"]:
            narrowed = sr.narrow_applicability(ast, exclusion["leaves"])
            applied.append("R2" if rule == "R2_conflict" else "R3")
        if rule == "R3_negative_or_harm":
            guards = sr.append_demotion(guards, {
                "rule": rule, "trigger_unit": str(unit["unit_id"]),
                "harm_event": harmed})
            if "R3" not in applied:
                applied.append("R3")
    if not applied and narrowed is None:
        return {"applied": [], "snapshot": snapshot, "card_sha": card_sha}
    patched = sr.patch_card(
        snapshot, skill_id=FORECAST_SKILL_ID, store_root=store_root,
        tag="revise_%s" % unit["unit_id"],
        risk_guards=guards, observable_applicability=narrowed,
        predicted_data_effect=tuple(applied) or ("skill_revised",))
    return {"applied": applied, "snapshot": patched["snapshot"],
            "card_sha": patched["card_sha"],
            "receipts": patched.get("receipts")}


def _score_unit(unit: Mapping[str, Any], result: Mapping[str, Any],
                *, forecast_skill: str | None) -> dict[str, Any]:
    public = s1._public_unit_result(result)
    dep = public.get("deployment") or {}
    rounds = public.get("rounds") or []
    ops = []
    for step in (dep.get("applied_program") or []):
        if isinstance(step, Mapping):
            ops.append(str(step.get("op")))
        elif isinstance(step, (list, tuple)) and step:
            ops.append(str(step[0]))
    card_ops = []
    if forecast_skill and rounds:
        geo = None
        for sid, sha in (rounds[0].get("installed_skill_content_sha")
                         or {}).items():
            del sha
            if sid == forecast_skill:
                geo = [forecast_skill]
        card_ops = [op for op in ops if op]
    sources = _candidate_sources(rounds[0], skill_id=forecast_skill,
                                 card_ops=card_ops) if rounds else {
        "supplied": 0, "self_proposed": 0, "dedup_swallowed": False}
    scope = {}
    if rounds:
        scope = dict(rounds[0].get("scope_match_by_skill_id") or {})
    return {
        "position": unit["position"],
        "unit_id": unit["unit_id"],
        "role": unit["role"],
        "source": unit.get("source"),
        "arm": result["arm"],
        "deploy_source": dep.get("deploy_source"),
        "applied_ops": ops,
        "heldout_gain": float(dep.get("heldout_gain") or 0.0),
        "heldout_smase": dep.get("heldout_smase"),
        "harm_event": bool(dep.get("harm_event")),
        "worst_series_gain": dep.get("worst_series_gain"),
        "llm_calls": int(result.get("llm_calls") or 0),
        "consumer_fits": int(result.get("consumer_fits") or 0),
        "seconds": float(result.get("seconds") or 0.0),
        "g2": result.get("g2"),
        "scope_match": {
            "classification": bool(scope.get(CLS_SKILL_ID)),
            "forecast": bool(scope.get(FORECAST_SKILL_ID)),
        },
        "candidate_sources": sources,
        "rounds": rounds,
        "deployment": dep,
        "end_skill_ids": result.get("end_skill_ids"),
    }


def _verdict(rows: Sequence[Mapping[str, Any]], chain: Sequence[Mapping[str, Any]],
             g2: Mapping[str, Any], *, stopped: str | None) -> dict[str, Any]:
    if stopped:
        return {"label": stopped, "why": "run stopped before judging"}
    a5 = [r for r in rows if r["arm"] == ARM_A5]
    forecast_installed = any(c.get("skill_id") == FORECAST_SKILL_ID
                             for c in chain)
    converted = 0
    for row in a5:
        if row["role"].startswith("strong") and row.get("candidate_sources", {}).get("supplied"):
            if str(row.get("deploy_source")) != "identity":
                converted += 1
        if row["role"].startswith("strong"):
            for rec in row.get("rounds") or []:
                for ep in rec.get("episodes") or []:
                    if (str(ep.get("source_skill_id")) == FORECAST_SKILL_ID
                            and str(ep.get("relation")) == "POSITIVE"
                            and str(ep.get("local_status")) == "LOCAL_ACTIVE"):
                        converted += 1
    harm = sum(1 for r in rows if r.get("harm_event"))
    g2_zero = bool(g2.get("all_zero"))
    if not forecast_installed:
        return {"label": "TREATMENT_EMPTY",
                "why": "producer did not yield a dual-gate POSITIVE; no forecast card"}
    if g2_zero and converted and harm == 0:
        return {"label": "S2A_PORTABLE_REDUCED",
                "why": "forecast card compiled; at least one strong beneficiary converted; G2 silent; harm 0"}
    return {"label": "S2A_PARTIAL",
            "why": "card or isolation landed but conversion/harm/G2 incomplete",
            "converted_strong": converted, "harm": harm, "g2_zero": g2_zero}


def _predictions(rows, chain, g2, verdict) -> list[dict[str, Any]]:
    a3 = [r for r in rows if r["arm"] == ARM_A3]
    k0 = [r for r in rows if r["arm"] == ARM_K0]
    a5 = [r for r in rows if r["arm"] == ARM_A5]
    diverged = []
    for u3, u0 in zip(a3, k0):
        if u3["unit_id"] != u0["unit_id"]:
            continue
        d = abs(float(u3["heldout_gain"]) - float(u0["heldout_gain"]))
        if d > 0.05:
            diverged.append({"unit": u3["unit_id"], "abs_delta": d})
    return [
        {"id": "P1", "predicted": "hold", "observed": "already landed (149 green)"},
        {"id": "P2", "predicted": "G2 three faces all zero",
         "observed": ("untested (no live unit)" if not g2.get("tested")
                      else "all_zero=%s faces=%s"
                      % (g2.get("all_zero"), g2.get("faces")))},
        {"id": "P3", "predicted": "producer hit => strong beneficiary converts",
         "observed": "forecast_card=%s a5_strong_gains=%s" % (
             any(c.get("skill_id") == FORECAST_SKILL_ID for c in chain),
             [(r["unit_id"], r["heldout_gain"], r.get("candidate_sources"))
              for r in a5 if str(r["role"]).startswith("strong")])},
        {"id": "P4", "predicted": "N/A", "observed": "N/A"},
        {"id": "P5", "predicted": "harm 0",
         "observed": "harm_events=%d" % sum(1 for r in rows if r.get("harm_event"))},
        {"id": "P6", "predicted": "K0 ≡ A3",
         "observed": "material_divergences=%s" % (diverged or "none")},
        {"id": "P7", "predicted": "N/A", "observed": "N/A (reduced course)"},
        {"id": "verdict", "predicted": "S2A_PORTABLE_REDUCED | S2A_PARTIAL | TREATMENT_EMPTY",
         "observed": verdict.get("label")},
    ]


def _markdown(payload: Mapping[str, Any]) -> str:
    v = payload.get("verdict") or {}
    g2 = payload.get("g2") or {}
    rows = payload.get("rows") or []
    lines = [
        "# S2a G1/G2 live (reduced course)",
        "",
        "**S2a 判词:%s;自产卡:%s;守卫三面:%s;核心数字 LLM %s / fit %s**"
        % (v.get("label"),
           ("是" if any(c.get("skill_id") == FORECAST_SKILL_ID
                        for c in (payload.get("version_chain") or []))
            else "否"),
           ("全零" if g2.get("all_zero") else "非零")
           if g2.get("tested") else "未考",
           (payload.get("ledger") or {}).get("llm"),
           (payload.get("ledger") or {}).get("fit")),
        "",
        v.get("why", ""),
        "",
    ]
    transport = payload.get("transport") or {}
    if transport:
        lines += [
            "## Transport (not part of verdict)",
            "",
            "- source: %s" % transport.get("source"),
            "- request: %s @ %s" % (
                transport.get("request_model"), transport.get("request_base_url")),
            "- first_returned_model: %s" % transport.get("first_returned_model"),
            "- r1_inspect: %s" % transport.get("r1_inspect"),
            "- r2_inspect: %s" % transport.get("r2_inspect"),
            "- note: %s" % transport.get("note"),
            "",
        ]
    lines += [
        "## Predictions",
        "",
    ]
    for row in payload.get("predictions") or []:
        lines.append("- **%s**: predicted %s; observed %s"
                     % (row["id"], row.get("predicted"), row.get("observed")))
    lines += ["", "## Card version chain", ""]
    for row in payload.get("version_chain") or []:
        lines.append("- %s sha=%s trigger=%s" % (
            row.get("version"), str(row.get("card_content_sha") or "")[:12],
            row.get("trigger_unit") or row.get("produced_by")))
    lines += ["", "## Units", "",
              "| pos | unit | role | Static gain | A3 | K0 | A5 | A5 G2 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    by = {}
    for row in rows:
        by.setdefault(row["unit_id"], {})[row["arm"]] = row
    freeze_course = (payload.get("course") or [])
    for unit in freeze_course:
        uid = unit["unit_id"]
        pack = by.get(uid) or {}
        g2u = (pack.get(ARM_A5) or {}).get("g2") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (unit.get("position"), uid, unit["role"],
               _fmt((pack.get(ARM_STATIC) or {}).get("heldout_gain")),
               _fmt((pack.get(ARM_A3) or {}).get("heldout_gain")),
               _fmt((pack.get(ARM_K0) or {}).get("heldout_gain")),
               _fmt((pack.get(ARM_A5) or {}).get("heldout_gain")),
               g2u))
    if payload.get("first_fault"):
        lines += ["", "## First fault", "", str(payload["first_fault"])]
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return "%+.4f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def run_course(seed: str = "r1", *, resume: bool = False) -> int:
    freeze = json.loads(FREEZE_JSON.read_text(encoding="utf-8"))
    course = _live_units(freeze)
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    s1.bind_curriculum_identity(
        task_kind="forecast", consumer_id=traffic.CONSUMER_ID,
        metric=traffic.METRIC)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "seed": seed,
        "branch": "reduced",
        "R2_note": freeze.get("R2_note"),
        "course": course,
        "delta_material": freeze.get("delta_material"),
        "verdict_vocabulary": freeze.get("verdict_vocabulary"),
        "pre_registered_predictions": PRE_REGISTERED,
        "git_head": s1._git("rev-parse", "HEAD"),
    }
    rows: list[dict[str, Any]] = []
    revisions: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    ledger = {"llm": 0, "fit": 0}
    done: set[tuple[int, str]] = set()
    out_json = E2 / ("s2a_g1_run1.json" if seed == "r1"
                     else "s2a_g1_run1_%s.json" % seed)
    out_md = E2 / ("s2a_g1_run1.md" if seed == "r1"
                   else "s2a_g1_run1_%s.md" % seed)
    checkpoint = E2 / ("s2a_g1_run1.checkpoint.json" if seed == "r1"
                       else "s2a_g1_run1_%s.checkpoint.json" % seed)
    if resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows = list(saved.get("rows") or [])
        revisions = list(saved.get("revisions") or [])
        chain = list(saved.get("version_chain") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        started = time.time() - float(saved.get("wall_seconds") or 0.0)
        done = {(int(row["position"]), str(row["arm"])) for row in rows}
        payload["resumed_from_checkpoint"] = sorted(
            "%s/%s" % item for item in done)

    def _save() -> None:
        checkpoint.write_text(json.dumps(ps0c.redact(s1._plain({
            "rows": rows, "revisions": revisions, "version_chain": chain,
            "ledger": ledger,
            "wall_seconds": round(time.time() - started, 1),
        })), indent=1, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")

    stopped: str | None = None
    first_fault: str | None = None
    bank = CellBank()
    try:
        probe = None
        for wave in range(1, 6):
            probe = ps0c.probe_new_backend()
            print("PROBE wave=%d ok=%s model=%s" % (
                wave, probe.get("ok"), probe.get("returned_model")),
                  flush=True)
            if probe.get("ok"):
                break
            reason = str(probe.get("reason") or "")
            transient = any(token in reason for token in (
                "530", "1033", "tunnel_error", "retryable"))
            if not transient or wave == 5:
                break
            print("PROBE transient; sleep 120s before wave %d"
                  % (wave + 1), flush=True)
            time.sleep(120)
        payload["backend_probe"] = ps0c.redact({
            k: (probe or {}).get(k) for k in
            ("ok", "returned_model", "reason", "expected_model")})
        if not (probe or {}).get("ok"):
            raise Stop("BACKEND_UNAVAILABLE",
                       str((probe or {}).get("reason")))

        store_root = Path(tempfile.gettempdir()) / ("s2a_g1_%s" % seed)
        if not resume and store_root.exists():
            shutil.rmtree(store_root)
        h0 = _h0()
        cls_card = _cls_card()
        k0_fixed = _install(h0, cls_card, store_root=store_root / "seed",
                            tag="k0_fixed")
        a5 = _install(h0, cls_card, store_root=store_root / "seed",
                      tag="a5_v0")
        payload["k0"] = {
            "empty_forecast_k0": True,
            "h0_sha": h0.runtime_bundle_sha,
            "k0_fixed_sha": k0_fixed.runtime_bundle_sha,
            "a5_v0_sha": a5.runtime_bundle_sha,
            "classification_card": CLS_SKILL_ID,
            "k0_skill_ids": sorted(str(s.skill_id) for s in k0_fixed.skills),
        }
        backend = cls._live_backend(LLM_CAP)
        from evaluation.functional.task_episode_harness.agentic.runner import (
            live_transport,
        )
        target = live_transport(default_model=cls.SLOW_MODEL)
        payload["transport"] = {
            "request_base_url": target["base_url"],
            "request_model": target["model"],
            "source": target["source"],
            "r1_inspect": "api.agicto.cn + gpt-5.6-sol (agicto direct)",
            "r2_inspect": "CPA relay when M0_AGENT_* set (host+model mapped together)",
            "note": "transport-layer difference only; not part of the verdict",
        }
        forecast_sha = None
        g2_faces: list[dict[str, Any]] = []

        for unit in course:
            position = int(unit["position"])
            uid = str(unit["unit_id"])
            if time.time() - started > WALL_SECONDS:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "wall before " + uid)
            if ledger["llm"] >= LLM_CAP or ledger["fit"] >= FIT_CAP:
                raise Stop("COMPUTE_BUDGET_EXCEEDED", "budget before " + uid)
            print("UNIT %d %s (%s)" % (position, uid, unit["role"]), flush=True)
            cell = bank.get(uid, freeze)

            for arm in ARMS:
                if (position, arm) in done:
                    continue
                base = {ARM_STATIC: h0, ARM_A3: h0,
                        ARM_K0: k0_fixed, ARM_A5: a5}[arm]
                result = run_unit(
                    unit=unit, cell=cell, arm=arm, base_snapshot=base,
                    backend=backend, store_root=store_root,
                    fit_ledger=ledger)
                ledger["llm"] = int(backend.calls)
                scored = _score_unit(unit, result,
                                     forecast_skill=FORECAST_SKILL_ID
                                     if forecast_sha else None)
                rows.append(scored)
                if arm in (ARM_K0, ARM_A5):
                    g2_faces.append({"position": position, "unit_id": uid,
                                     "arm": arm, **(scored.get("g2") or {})})
                    faces = scored.get("g2") or {}
                    if any(int(faces.get(k) or 0) for k in
                           ("retrieval", "scope_match", "supply")):
                        first_fault = (
                            "G2_FIRST_FAULT %s/%s retrieval=%s "
                            "scope_match=%s supply=%s"
                            % (uid, arm, faces.get("retrieval"),
                               faces.get("scope_match"), faces.get("supply")))
                        raise Stop("S2A_G2_LEAK", first_fault)
                _save()
                print("  %-10s deploy=%s gain=%+.4f g2=%s src=%s"
                      % (arm, scored.get("deploy_source"),
                         scored.get("heldout_gain") or 0.0,
                         scored.get("g2"), scored.get("candidate_sources")),
                      flush=True)

                if arm == ARM_A5:
                    if unit["role"] == "producer" and not forecast_sha:
                        seed_row = _supply_row(scored)
                        compiled = (_compile_forecast_card(seed_row)
                                    if seed_row else
                                    {"card": None,
                                     "withheld_because": "no_positive_local_active"})
                        payload["boundary_compile"] = {
                            "unit_id": uid,
                            "withheld_because": compiled.get("withheld_because"),
                            "has_card": compiled.get("card") is not None,
                            "scope": compiled.get("scope"),
                        }
                        if compiled.get("card") is None:
                            print("  BOUNDARY withheld: %s"
                                  % compiled.get("withheld_because"),
                                  flush=True)
                        else:
                            a5 = _install(a5, compiled["card"],
                                          store_root=store_root / "seed",
                                          tag="a5_forecast_v0")
                            forecast_sha = ss.skill_content_sha(
                                next(s for s in a5.skills
                                     if str(s.skill_id) == FORECAST_SKILL_ID))
                            chain.append({
                                "version": "v0",
                                "skill_id": FORECAST_SKILL_ID,
                                "card_content_sha": forecast_sha,
                                "installed_after": uid,
                                "produced_by": "ladder_v2_compile_supply_tier",
                                "runtime_bundle_sha": a5.runtime_bundle_sha,
                            })
                            print("  CARD v0 %s" % forecast_sha[:12],
                                  flush=True)
                    elif forecast_sha:
                        revision = _revise_forecast(
                            a5, scored, unit,
                            store_root=store_root / "revise",
                            card_sha=forecast_sha)
                        a5 = revision["snapshot"]
                        if revision["applied"]:
                            chain.append({
                                "version": "v%d" % len(chain),
                                "skill_id": FORECAST_SKILL_ID,
                                "card_content_sha": revision["card_sha"],
                                "produced_by": "+".join(revision["applied"]),
                                "trigger_unit": uid,
                                "runtime_bundle_sha": a5.runtime_bundle_sha,
                            })
                            print("    REVISION %s"
                                  % "+".join(revision["applied"]),
                                  flush=True)
                        forecast_sha = revision["card_sha"]
                        revisions.append({
                            "position": position, "unit_id": uid,
                            "applied": revision["applied"],
                            "card_content_sha_after": revision["card_sha"]})
                    _save()
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
        if first_fault:
            payload["first_fault"] = first_fault
    except Exception as exc:  # noqa: BLE001
        import traceback
        text = "%s: %s" % (type(exc).__name__, exc)
        transport = (
            type(exc).__name__ in ("AgentTransportError", "InternalServerError")
            or "AgentTransportError" in text
            or "InternalServerError" in text
            or "530" in text
            or "tunnel_error" in text
        )
        stopped = "BACKEND_UNAVAILABLE" if transport else "INSTRUMENT_UNREADABLE"
        payload["stop"] = {
            "verdict": stopped,
            "reason": ps0c.redact(text),
            "traceback": ps0c.redact(traceback.format_exc()),
        }
        if transport:
            first_fault = "BACKEND_UNAVAILABLE — 授权中继隧道不可达"
            payload["first_fault"] = first_fault
    finally:
        s1.bind_curriculum_identity()

    faces = g2_faces if "g2_faces" in locals() else []
    g2 = {
        "faces": faces,
        "tested": bool(faces),
        "all_zero": (
            all(not int(row.get(k) or 0)
                for row in faces
                for k in ("retrieval", "scope_match", "supply"))
            if faces else None),
    }
    verdict = _verdict(rows, chain, g2, stopped=stopped)
    transport = payload.setdefault("transport", {})
    if "backend" in locals():
        transport["first_returned_model"] = getattr(
            backend, "first_returned_model", None)
    payload.update({
        "rows": rows,
        "revisions": revisions,
        "version_chain": chain,
        "ledger": {**ledger, "llm_cap": LLM_CAP, "fit_cap": FIT_CAP,
                   "wall_seconds": round(time.time() - started, 1)},
        "g2": g2,
        "verdict": verdict,
        "predictions": _predictions(rows, chain, g2, verdict),
    })
    out_json.write_text(
        json.dumps(ps0c.redact(s1._plain(payload)), indent=1,
                   ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    out_md.write_text(_markdown(payload), encoding="utf-8")
    print("VERDICT %s; g2_zero=%s; llm=%s fit=%s"
          % (verdict.get("label"), g2.get("all_zero"),
             ledger["llm"], ledger["fit"]), flush=True)
    return 0 if not stopped else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", default="r1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.run or args.resume:
        code = run_course(args.seed, resume=args.resume)
        if args.seed == "r1" and not args.resume:
            artifact = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            label = str((artifact.get("verdict") or {}).get("label") or "")
            if label == "TREATMENT_EMPTY":
                print("R1 TREATMENT_EMPTY; sampling r2 once", flush=True)
                return run_course("r2", resume=False)
        return code
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
