"""Fake-transport contract tests for DEV-TRAIN-1.  Zero project API, zero KDD."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse
from evaluation.main_protocol_p4 import dev_train1_evaluator as numeric
from evaluation.main_protocol_p4 import dev_train1_runner as R


def _series(seed: int, n: int, spikes: tuple[tuple[int, float], ...] = ()) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    out = 8.0 + np.sin(2.0 * np.pi * t / 24.0) + 0.05 * rng.randn(n)
    for index, value in spikes:
        out[int(index)] = float(value)
    return out


def _bundle(n_train: int = 2, n_eval: int = 2, length: int = 480) -> R.DataBundle:
    values = {}
    train = []
    eval_ids = []
    for i in range(n_train):
        uid = "t%d" % i
        train.append(uid)
        spikes = ((70, 800.0), (90, -800.0)) if i == 0 else ()
        values[uid] = _series(10 + i, length, spikes)
    for i in range(n_eval):
        uid = "e%d" % i
        eval_ids.append(uid)
        values[uid] = _series(50 + i, length)
    return R.DataBundle(values=values, train_uids=train, eval_uids=eval_ids, period=24)


def _spec(bundle: R.DataBundle, **kwargs: object) -> R.RunSpec:
    fields = dict(
        train_uids=list(bundle.train_uids),
        eval_uids=list(bundle.eval_uids),
        anchors=(240,),
        period=24,
        formation_units=((9, 312),),
        eval_units=((12, 360),),
        training_prefix_end=288,
        concurrency=4,
        max_llm=200,
        max_fits=10,
        max_wall_seconds=3600,
        onestep_ops=("identity",),
        live_api=False,
    )
    fields.update(kwargs)
    return R.RunSpec(**fields)


def _stage(stage: str, payload: dict) -> AgentResponse:
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "stage_result",
            "stage": stage,
            "payload": payload,
        },
        raw_response={"id": "fake-%s" % stage, "model": "cpa-grok-4.6"},
        provider_metadata={"returned_model": "cpa-grok-4.6"},
    )


class ScriptedBackend:
    """Deterministic Fast/Slow envelopes.  Not a simplified agent."""

    def __init__(
        self,
        *,
        mode: str = "identity",
        delay_by_uid: dict[str, float] | None = None,
        account_uid: str | None = None,
        fail_uid: str | None = None,
    ) -> None:
        self.mode = mode
        self.delay_by_uid = dict(delay_by_uid or {})
        self.account_uid = account_uid
        self.fail_uid = fail_uid
        self.calls = 0
        self.returned_models: set[str] = set()
        self.requests: list[object] = []

    def complete(self, request: object) -> AgentResponse:
        self.requests.append(request)
        case_id = str(getattr(request, "case_id", "") or "")
        delay = float(self.delay_by_uid.get(case_id, 0.0) or 0.0)
        if delay:
            time.sleep(delay)
        if self.account_uid and case_id == self.account_uid:
            raise RuntimeError("Error code: 402 Insufficient Balance")
        if self.fail_uid and case_id == self.fail_uid:
            raise RuntimeError("injected per-sequence failure")
        self.calls += 1
        self.returned_models.add("cpa-grok-4.6")
        stage = str(getattr(request, "stage", "") or "")
        if stage == "inspect":
            return _stage("inspect", {
                "inspected_region_fractions": [[0.0, 1.0]],
                "requested_public_tools": [],
                "uncertainty": "high",
            })
        if stage == "propose":
            if self.mode == "winsor":
                return _stage("propose", {
                    "candidates": [{
                        "candidate_id": "winsor_a",
                        "steps": [{"op": "winsorize", "params": {}}],
                    }],
                })
            if self.mode == "absolute_region":
                return _stage("propose", {
                    "candidates": [{
                        "candidate_id": "abs_region",
                        "steps": [{
                            "op": "winsorize",
                            "params": {"start_index": 700},
                        }],
                    }],
                })
            return _stage("propose", {"candidates": []})
        if stage == "select":
            chosen = {
                "identity": "identity",
                "winsor": "winsor_a",
                "absolute_region": "abs_region",
            }.get(self.mode, "identity")
            return _stage("select", {
                "chosen_candidate_id": chosen,
                "verification_actions": ["public_evidence_insufficient"],
            })
        if stage == "edit":
            return AgentResponse.valid(
                {
                    "schema_version": "agent-envelope/1",
                    "kind": "no_proposal",
                    "stage": "edit",
                    "reason_code": "insufficient_public_evidence",
                },
                raw_response={"id": "fake-edit", "model": "cpa-grok-4.6"},
                provider_metadata={"returned_model": "cpa-grok-4.6"},
            )
        raise RuntimeError("unexpected stage %s" % stage)


def _factory(mode: str = "identity", **kwargs: object):
    def factory() -> ScriptedBackend:
        return ScriptedBackend(mode=mode, **kwargs)
    return factory


def _public_input(request: object) -> dict:
    messages = list(getattr(request, "messages", ()) or ())
    if len(messages) < 2:
        return {}
    payload = json.loads(messages[1]["content"])
    public = payload.get("public_input")
    return dict(public) if isinstance(public, dict) else {}


def test_h0_has_no_frozen_program(machinery, snapshot) -> None:
    del machinery
    for skill in snapshot.skills:
        body = str(getattr(skill, "body", "") or "")
        if R.FROZEN_PROGRAM_MARKER in body:
            raise AssertionError("h0 skill %s carries frozen program steps" % skill.skill_id)


def test_concurrency_keys_ignore_completion_order(machinery, snapshot) -> None:
    bundle = _bundle(n_eval=4, n_train=2)
    spec = _spec(bundle, concurrency=4)
    budget = R.Budget(max_llm=200, max_fits=10, path=None)
    task_ctx = R.task_context_for(spec)
    run_dir = R.RUNS_ROOT / "contract_concurrency"
    delays = {"e0": 0.12, "e1": 0.08, "e2": 0.04, "e3": 0.0}
    rows = R.run_batch_fast(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=_factory("identity", delay_by_uid=delays),
        uids=spec.eval_uids, role="predict_input", arm="probe",
        position=12, origin=360, task_ctx=task_ctx, budget=budget,
        checkpoint_path=run_dir / "eval_v.json",
    )
    keys = [row["key"] for row in rows]
    expected = [R.decision_key("predict_input", "probe", 12, uid) for uid in spec.eval_uids]
    if keys != expected:
        raise AssertionError("keys followed completion order: %s" % keys)
    for row, uid in zip(rows, spec.eval_uids):
        if row["series_uid"] != uid:
            raise AssertionError("row/uid mismatch")
        if row["bound_series_uid"] != uid:
            raise AssertionError("request bound to the wrong uid")


def test_budget_persist_and_resume(machinery, snapshot) -> None:
    del machinery, snapshot
    path = R.RUNS_ROOT / "contract_budget" / "budget.json"
    budget = R.Budget(max_llm=200, max_fits=10, path=path)
    budget.path = path
    budget.spend_llm("old_skill", 3)
    budget.spend_fits("static", 1)
    loaded = R.Budget.load(path, max_llm=200, max_fits=10)
    if loaded.llm() != 3 or loaded.fits() != 1:
        raise AssertionError("budget did not resume %s %s" % (loaded.llm(), loaded.fits()))


def test_slow_not_reasked(machinery, snapshot) -> None:
    bundle = _bundle()
    spec = _spec(bundle, concurrency=2)
    edits = {"n": 0}

    class Counting(ScriptedBackend):
        def complete(self, request: object) -> AgentResponse:
            if str(getattr(request, "stage", "")) == "edit":
                edits["n"] += 1
            return super().complete(request)

    def factory() -> Counting:
        return Counting(mode="identity")

    first = R.run_package(
        spec=spec, bundle=bundle, run_id="contract_slow_resume",
        backend_factory=factory, machinery=machinery, snapshot=snapshot,
        skip_slow=False,
    )
    if first.get("exit_code") not in (0, None) and first.get("status") not in (
        "COMPLETED", "RUNNING",
    ):
        if first.get("status") != "COMPLETED":
            raise AssertionError("first run failed: %s %s" % (first.get("status"), first.get("detail")))
    after_first = edits["n"]
    if after_first < 1:
        raise AssertionError("Slow edit stage was not reached")
    second = R.run_package(
        spec=spec, bundle=bundle, run_id="contract_slow_resume",
        backend_factory=factory, machinery=machinery, snapshot=snapshot,
        skip_slow=False,
    )
    del second
    if edits["n"] != after_first:
        raise AssertionError("Slow was asked again on resume: %s -> %s" % (after_first, edits["n"]))


def test_account_fault_nonzero(machinery, snapshot) -> None:
    bundle = _bundle(n_train=1, n_eval=1)
    spec = _spec(bundle, concurrency=1, onestep_ops=("identity",))
    out = R.run_package(
        spec=spec, bundle=bundle, run_id="contract_account",
        backend_factory=_factory("identity", account_uid=spec.train_uids[0]),
        machinery=machinery, snapshot=snapshot, skip_slow=True,
    )
    if out.get("exit_code") != 2 or out.get("status") != "ACCOUNT_OR_PERMISSION_FAULT":
        raise AssertionError("account fault was not a non-zero stop: %s" % out)


def test_incomplete_population_does_not_shrink(machinery, snapshot) -> None:
    del machinery, snapshot
    scored = {
        "eval_uids": ["e0", "e1"],
        "per_view_smase": [0.4, None],
        "mean_smase": None,
        "complete": False,
    }
    pop = R.population_from_scores(["e0", "e1"], scored)
    if pop["mean_smase"] != "UNKNOWN":
        raise AssertionError("incomplete population produced a mean")
    if pop["n_eval"] != 2 or pop["n_unknown"] != 1:
        raise AssertionError("denominator changed: %s" % pop)
    if pop["denominator_was_not_shrunk"] is not True:
        raise AssertionError("shrink flag missing")


def test_prompt_has_role_consumer_not_outcome(machinery, snapshot) -> None:
    bundle = _bundle(n_train=1, n_eval=1)
    spec = _spec(bundle, concurrency=1)
    captured: list[ScriptedBackend] = []

    def factory() -> ScriptedBackend:
        backend = ScriptedBackend(mode="identity")
        captured.append(backend)
        return backend

    task_ctx = R.task_context_for(spec)
    budget = R.Budget(max_llm=50, max_fits=10, path=None)
    R.run_batch_fast(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=factory, uids=spec.train_uids, role="train_workflow",
        arm="prompt", position=0, origin=288, task_ctx=task_ctx, budget=budget,
        checkpoint_path=R.RUNS_ROOT / "contract_prompt" / "train_w.json",
    )
    if not captured or not captured[0].requests:
        raise AssertionError("Fast did not send a request")
    public = _public_input(captured[0].requests[0])
    if "consumer_structure" not in public:
        raise AssertionError("consumer_structure missing from Fast public_input")
    if public.get("decision_role") != "train_workflow":
        raise AssertionError("decision_role missing or wrong: %s" % public.get("decision_role"))
    family = (public.get("consumer_structure") or {}).get("family")
    if family != "linear_ridge":
        raise AssertionError("consumer family not rendered: %s" % family)
    leaks = R.forbidden_outcome_paths(public)
    if leaks:
        raise AssertionError("outcome fields in Fast public_input: %s" % leaks)


def test_identity_eval_not_forced_zero(machinery, snapshot) -> None:
    del machinery, snapshot
    bundle = _bundle(n_train=2, n_eval=2)
    spec = _spec(bundle)
    budget = R.Budget(max_llm=10, max_fits=10, path=None)
    winsor = (("winsorize", {}),)
    train = {bundle.train_uids[0]: winsor, bundle.train_uids[1]: None}
    eval_asg = {uid: None for uid in bundle.eval_uids}
    static_model = R.fit_arm(
        spec=spec, bundle=bundle, train_assignment={uid: None for uid in bundle.train_uids},
        budget=budget, arm="static", origin=288,
    )
    het_model = R.fit_arm(
        spec=spec, bundle=bundle, train_assignment=train,
        budget=budget, arm="het", origin=288,
    )
    static_pred = R.predict_arm(
        spec=spec, bundle=bundle, model=static_model, eval_assignment=eval_asg, origin=360,
    )
    het_pred = R.predict_arm(
        spec=spec, bundle=bundle, model=het_model, eval_assignment=eval_asg, origin=360,
    )
    if int(het_pred["consumer_fits"]) != 0:
        raise AssertionError("predict billed a fit")
    static_score = R.score_arm(spec=spec, bundle=bundle, predictions=static_pred, origin=360)
    het_score = R.score_arm(spec=spec, bundle=bundle, predictions=het_pred, origin=360)
    restored = numeric.FrozenRidge.from_dict(het_model.to_dict())
    again = R.predict_arm(
        spec=spec, bundle=bundle, model=restored, eval_assignment=eval_asg, origin=360,
    )
    if not np.allclose(again["predictions"], het_pred["predictions"]):
        raise AssertionError("FrozenRidge round-trip changed predictions")
    if static_score["mean_smase"] == "UNKNOWN" or het_score["mean_smase"] == "UNKNOWN":
        raise AssertionError("readable identity-eval scores were UNKNOWN")
    if float(het_score["mean_smase"]) == 0.0 and float(static_score["mean_smase"]) != 0.0:
        raise AssertionError("identity eval was forced to a global zero")


def test_train_fault_not_silent_identity() -> None:
    records = {
        "t0": {
            "valid_mechanism_decision": False,
            "assignment_policy": "unknown",
            "typed_steps": None,
            "deploy_status": "PREPARE_FAILED",
        },
        "t1": {
            "valid_mechanism_decision": True,
            "assignment_policy": "explicit_identity",
            "typed_steps": None,
        },
    }
    assignment, missing = R.complete_assignment(["t0", "t1"], records)
    if assignment is not None:
        raise AssertionError("faulted train uid was filled: %s" % assignment)
    if missing != ["t0"]:
        raise AssertionError("missing set wrong: %s" % missing)


def test_region_absolute_not_remapped(machinery, snapshot) -> None:
    bundle = _bundle(n_train=1, n_eval=1)
    spec = _spec(bundle, concurrency=1)
    task_ctx = R.task_context_for(spec)
    budget = R.Budget(max_llm=50, max_fits=10, path=None)
    rows = R.run_batch_fast(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=_factory("absolute_region"),
        uids=spec.train_uids, role="train_workflow", arm="region",
        position=0, origin=288, task_ctx=task_ctx, budget=budget,
        checkpoint_path=R.RUNS_ROOT / "contract_region" / "train_w.json",
    )
    row = rows[0]
    if row.get("deploy_status") != "REGION_GEOMETRY_UNMAPPED":
        raise AssertionError("absolute region was not blocked: %s" % row.get("deploy_status"))
    if row.get("valid_mechanism_decision"):
        raise AssertionError("unmapped region was treated as a valid decision")
    if R.assignment_value(row) != "MISSING":
        raise AssertionError("unmapped region became an assignment")
    geo = row.get("region_geometry") or {}
    if geo.get("did_not_replace_with_whole_window") is not True:
        raise AssertionError("whole_window substitution was not refused")
    steps = row.get("typed_steps") or []
    if not steps or "start_index" not in (steps[0].get("params") or {}):
        raise AssertionError("typed params were stripped")


def test_full_program_persisted(machinery, snapshot) -> None:
    bundle = _bundle(n_train=1, n_eval=1)
    spec = _spec(bundle, concurrency=1)
    task_ctx = R.task_context_for(spec)
    budget = R.Budget(max_llm=50, max_fits=10, path=None)
    rows = R.run_batch_fast(
        spec=spec, bundle=bundle, snapshot=snapshot, machinery=machinery,
        backend_factory=_factory("winsor"),
        uids=spec.train_uids, role="train_workflow", arm="persist",
        position=0, origin=288, task_ctx=task_ctx, budget=budget,
        checkpoint_path=R.RUNS_ROOT / "contract_persist" / "train_w.json",
    )
    row = rows[0]
    if not row.get("valid_mechanism_decision"):
        raise AssertionError("winsor decision invalid: %s" % row)
    steps = row.get("typed_steps") or []
    if not steps or steps[0].get("op") != "winsorize":
        raise AssertionError("typed steps not kept: %s" % steps)
    if "params" not in steps[0]:
        raise AssertionError("params dropped")
    value = R.assignment_value(row)
    if value is None or value[0][0] != "winsorize":
        raise AssertionError("assignment lost order/params: %s" % (value,))


def main() -> int:
    machinery = R.load_machinery()
    snapshot = R.load_h0_snapshot(machinery)
    tests = [
        ("h0_has_no_frozen_program", lambda: test_h0_has_no_frozen_program(machinery, snapshot)),
        ("concurrency_keys", lambda: test_concurrency_keys_ignore_completion_order(machinery, snapshot)),
        ("budget_resume", lambda: test_budget_persist_and_resume(machinery, snapshot)),
        ("slow_not_reasked", lambda: test_slow_not_reasked(machinery, snapshot)),
        ("account_fault_nonzero", lambda: test_account_fault_nonzero(machinery, snapshot)),
        ("incomplete_population", lambda: test_incomplete_population_does_not_shrink(machinery, snapshot)),
        ("prompt_role_consumer", lambda: test_prompt_has_role_consumer_not_outcome(machinery, snapshot)),
        ("identity_eval_not_zero", lambda: test_identity_eval_not_forced_zero(machinery, snapshot)),
        ("train_fault_not_identity", test_train_fault_not_silent_identity),
        ("region_not_remapped", lambda: test_region_absolute_not_remapped(machinery, snapshot)),
        ("program_persisted", lambda: test_full_program_persisted(machinery, snapshot)),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            sys.stdout.write("PASS %s\n" % name)
        except Exception as exc:
            sys.stdout.write("FAIL %s: %s\n" % (name, exc))
            traceback.print_exc()
            failed.append(name)
    sys.stdout.write("%d passed, %d failed\n" % (len(tests) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
