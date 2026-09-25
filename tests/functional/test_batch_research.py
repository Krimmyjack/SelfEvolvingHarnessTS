"""Contract tests with synthetic tensors/scripted calls: zero real fits or LLM."""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
# Explicit root binding works in Windows and WSL, without depending on the
# historical package alias/symlink pointing to a different checkout.
spec = importlib.util.spec_from_file_location(
    "SelfEvolvingHarnessTS", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules["SelfEvolvingHarnessTS"] = package
spec.loader.exec_module(package)

from SelfEvolvingHarnessTS.methods.ttha.batch_research import (
    Candidate, Guidance, Knowledge, Limits, TOOLS, ToolInputError, apply_update, run_job, scope_state,
)

SEEDS = (11, 12, 13)
FIELDS = frozenset({"lag24_corr", "last168_std_over_full_T_std"})


def candidate(name, loss=None, job="job", seeds=SEEDS):
    losses = tuple(tuple(tuple(loss + r * 0.1 for _ in range(32)) for _ in range(2))
                   for r in range(len(seeds))) if loss is not None else ()
    return Candidate(name, {"default": name}, job,
                     seeds if loss is not None else (),
                     tuple(f"{job}/{name}/seed-{s}" for s in seeds) if loss is not None else (),
                     losses)


def action(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


class FakeAdapter:
    def __init__(self):
        self.fits = []
        self.built = 0

    def overview(self):
        return {"entities": [{"entity_index": e, "lag24_corr": e / 32} for e in range(32)],
                "batch_features": {}, "consumer": {"name": "scripted contract fixture"}}

    def inspect_data(self, arguments, **kwargs):
        return {"batch_features": {"lag24_corr": 0.8}, "read_region": "T"}

    def build_material(self, arguments, **kwargs):
        self.built += 1
        return candidate(f"q{self.built}")

    def inspect_material(self, plan_id, arguments, **kwargs):
        return {"plan_id": plan_id, "read_region": "T", "derived_views": 1}

    def evaluate(self, plan, seeds, *, feedback, remaining_seconds):
        self.fits.append((plan.plan_id, seeds, feedback))
        trained = candidate(plan.plan_id, 0.5, seeds=seeds)
        return trained if feedback else dataclasses.replace(trained, ca_losses=())


class Script:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return {"actions": next(self.batches)}


class BatchResearchTests(unittest.TestCase):
    def run_script(self, script, **kwargs):
        adapter = kwargs.pop("adapter", FakeAdapter())
        result = run_job(job_id="job", knowledge=kwargs.pop("knowledge", Knowledge()),
                         adapter=adapter, client=script, seeds=SEEDS,
                         baselines=kwargs.pop("baselines", (candidate("none", 2), candidate("fixed", 1))),
                         allowed_features=FIELDS,
                         tool_contracts={name: {"description": "scripted fixture"} for name in TOOLS}, **kwargs)
        return result, adapter

    def test_whole_batch_loop_commit_is_not_an_argmin(self):
        script = Script([
            [action("inspect_data", fields=["lag24_corr"])],
            [action("build_material", default={"steps": []})],
            [action("inspect_material", plan_id="q1"), action("evaluate", plan_id="q1"),
             action("compare", a="fixed", b="q1")],
            [action("commit", plan_id="fixed", reason="Keep the incumbent for this fixture")],
        ])
        result, adapter = self.run_script(script)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(result.committed_plan_id, "fixed")  # q1 has a lower mock loss
        self.assertEqual(result.delivery_model_ref, "job/fixed/seed-11")
        self.assertEqual(adapter.fits, [("q1", SEEDS, True)])
        self.assertEqual(len(script.requests[0]["overview"]["entities"]), 32)
        comparison = next(x["output"] for x in result.trace if x["event"] == "tool_completed" and x["tool"] == "compare")
        self.assertAlmostEqual(comparison["mean_delta"], 0.5)
        self.assertEqual(len(comparison["delta_by_seed_entity"][0]), 32)

    def test_evidence_roundtrip_defers_decisions_without_charging_or_selecting(self):
        script = Script([
            [action("inspect_data"), action("build_material", default={"steps": []})],
            [action("build_material", default={"steps": []}), action("inspect_material", plan_id="q1"),
             action("evaluate", plan_id="q1")],
            [action("evaluate", plan_id="q1"), action("compare", a="fixed", b="q1"),
             action("commit", plan_id="q1", reason="not yet informed")],
            [action("commit", plan_id="fixed", reason="Explicit choice after reading comparison")],
        ])
        result, adapter = self.run_script(script, evidence_roundtrip=True)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(result.committed_plan_id, "fixed")
        self.assertEqual(adapter.built, 1)
        self.assertEqual(len(adapter.fits), 1)
        self.assertEqual(result.tool_calls, 6)
        self.assertEqual(result.calls, 4)
        deferred = [r for r in result.trace if r['event'] == 'action_batch_deferred']
        self.assertEqual([r['actions'][0]['tool'] for r in deferred], ['build_material', 'evaluate', 'commit'])
        self.assertTrue(script.requests[0]['contract']['evidence_roundtrip'])
        self.assertTrue(any(r.get('tool') == 'compare' and r['event'] == 'tool_completed'
                            for r in script.requests[-1]['current_trace']))

    def test_real_policy_error_roundtrip_is_explicit_and_default_stays_strict(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluation.main_protocol_p4 import batch_research_runtime as rt
        from methods.ttha.batch_base import policy as live_policy
        import copy
        bad = live_policy.fixed_mixup_policy()
        bad["rules"] = [{"when": {"feature": "last168_std_over_full", "op": "<", "value": .5}, "steps": []}]
        good = copy.deepcopy(bad)
        good["rules"][0]["when"]["feature"] = "last168_std_over_full_T_std"
        for enabled in (False, True):
            adapter = rt.Adapter.__new__(rt.Adapter)
            adapter.refs, adapter.specs, adapter.fitted = {}, {}, {}
            adapter.tool_error_feedback = enabled
            adapter.ctx = SimpleNamespace(job=SimpleNamespace(job_id="job"), job_dir=Path("unused"),
                overview=lambda: {"entities": [{"last168_std_over_full_T_std": .8} for _ in range(32)], "summary": {}})
            script = Script([
                [action("build_material", plan_id="proposal", policy=bad),
                 action("evaluate", plan_id="proposal"), action("commit", plan_id="fixed", reason="must not execute suffix")],
                [action("build_material", plan_id="proposal", policy=good),
                 action("commit", plan_id="fixed", reason="explicit decision after correction")],
            ])
            with patch.object(rt, "br", sys.modules[Candidate.__module__]), \
                 patch.object(rt.materials, "build", return_value=SimpleNamespace(material_id="proposal")) as build, \
                 patch.object(rt.context, "write_json") as write, patch.object(rt, "fit") as fit:
                result, _ = self.run_script(script, adapter=adapter, max_tool_corrections=2 if enabled else 0)
            fit.assert_not_called()
            self.assertEqual(bad["rules"][0]["when"]["feature"], "last168_std_over_full")
            if not enabled:
                self.assertEqual((result.status, result.reason, result.calls), ("INCOMPLETE", "PolicyError", 1))
                self.assertEqual(adapter.refs, {})
                build.assert_not_called(); write.assert_not_called()
            else:
                self.assertEqual((result.status, result.committed_plan_id, result.calls, result.tool_calls), ("COMPLETE", "fixed", 2, 3))
                self.assertEqual(build.call_count, 1)
                self.assertEqual(write.call_count, 1)
                self.assertEqual(set(adapter.refs), {"proposal"})
                rejected = next(r for r in script.requests[1]["current_trace"] if r["event"] == "tool_rejected")
                self.assertIn("last168_std_over_full", rejected["error"]["message"])
                self.assertIn("last168_std_over_full_T_std", rejected["error"]["valid_observation_fields"])
                self.assertEqual([a["tool"] for a in rejected["unexecuted_actions"]], ["evaluate", "commit"])
                self.assertEqual(script.requests[1]["remaining"]["tool_corrections"], 1)
                self.assertTrue(script.requests[0]["contract"]["tool_error_feedback"])

    def test_tool_corrections_preserve_completed_actions_and_roundtrip(self):
        script = Script([
            [action("build_material"), action("commit", plan_id="unknown", reason="bad id")],
            [action("evaluate", plan_id="q1"), action("commit", plan_id="q1", reason="deferred")],
            [action("commit", plan_id="fixed", reason="explicit keep after feedback")],
        ])
        result, adapter = self.run_script(script, max_tool_corrections=2, evidence_roundtrip=True)
        self.assertEqual((result.status, result.committed_plan_id, result.calls, result.tool_calls), ("COMPLETE", "fixed", 3, 4))
        self.assertEqual(adapter.built, 1)
        self.assertEqual(adapter.fits, [("q1", SEEDS, True)])
        self.assertIn("q1", [c["plan_id"] for c in script.requests[1]["candidates"]])

    def test_repeated_tool_errors_cannot_extend_any_budget_or_fallback(self):
        for limits, correction_cap, expected_calls, expected_tools in (
            (Limits(), 1, 2, 2),
            (Limits(max_calls=1), 2, 1, 1),
            (Limits(max_tools=1), 2, 2, 1),
        ):
            script = Script([[action("evaluate", plan_id="unknown")]] * 4)
            result, adapter = self.run_script(script, max_tool_corrections=correction_cap, limits=limits)
            self.assertEqual((result.status, result.failure_kind), ("INCOMPLETE", "BUDGET_EXHAUSTED"))
            self.assertEqual((result.calls, result.tool_calls), (expected_calls, expected_tools))
            self.assertIsNone(result.committed_plan_id)
            self.assertEqual(adapter.fits, [])

    def test_tool_correction_does_not_swallow_permissions_runtime_or_bad_envelope(self):
        for fault in (PermissionError("outside T"), RuntimeError("worker failed"), ValueError("internal invariant")):
            adapter = FakeAdapter()
            def fail(*args, **kwargs):
                raise fault
            adapter.inspect_data = fail
            script = Script([[action("inspect_data")], [action("commit", plan_id="fixed", reason="not reached")]])
            result, _ = self.run_script(script, adapter=adapter, max_tool_corrections=2)
            self.assertEqual((result.status, result.calls, result.failure_kind), ("INCOMPLETE", 1, "RUNTIME_FAILED"))
            self.assertFalse(any(r["event"] == "tool_rejected" for r in result.trace))
        bad = Script([[action("evaluate", plan_id="fixed"), {"tool": "commit"}]])
        result, adapter = self.run_script(bad, max_tool_corrections=2)
        self.assertEqual((result.calls, result.tool_calls, result.failure_kind), (1, 0, "PARSE_OR_VALIDATION_FAILED"))
        self.assertEqual(adapter.fits, [])

    def test_live_adapter_material_failure_is_not_an_input_correction(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from evaluation.main_protocol_p4 import batch_research_runtime as rt
        adapter = rt.Adapter.__new__(rt.Adapter)
        adapter.refs, adapter.specs, adapter.fitted = {}, {}, {}
        adapter.tool_error_feedback = True
        adapter.ctx = SimpleNamespace(job=SimpleNamespace(job_id="job"), job_dir=Path("unused"),
            overview=lambda: {"entities": [{} for _ in range(32)], "summary": {}})
        script = Script([[action("build_material", plan_id="proposal", policy=rt.policy.identity_policy())],
                         [action("commit", plan_id="fixed", reason="not reached")]])
        with patch.object(rt, "br", sys.modules[Candidate.__module__]), \
             patch.object(rt.materials, "build", side_effect=RuntimeError("disk failure")), \
             patch.object(rt.context, "write_json") as write:
            result, _ = self.run_script(script, adapter=adapter, max_tool_corrections=2)
        self.assertEqual((result.status, result.calls, result.reason), ("INCOMPLETE", 1, "RuntimeError"))
        self.assertEqual(adapter.refs, {})
        write.assert_not_called()

    def test_unknown_scope_can_be_resolved_by_inspection(self):
        h = Knowledge(1, (Guidance("observation_guidance", "Inspect the current pattern",
                                  {"feature": "lag24_corr", "op": ">", "value": 0.5}),))
        script = Script([[action("inspect_data", fields=["lag24_corr"])],
                         [action("commit", plan_id="fixed", reason="Explicit keep")]])
        result, _ = self.run_script(script, knowledge=h)
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(script.requests[0]["guidance"]["loaded"], [])
        self.assertEqual(script.requests[0]["guidance"]["applicability"][0]["state"], "UNKNOWN")
        self.assertIn("build_material", script.requests[0]["tools"])
        self.assertEqual(script.requests[1]["guidance"]["loaded"][0]["body"], "Inspect the current pattern")
        self.assertEqual(h.version, 1)

    def test_nonmatch_and_incomplete_do_not_disable_fast(self):
        for scope, expected in (({"const": False}, "NO_MATCH"), (None, "INCOMPLETE_NO_APPLICABILITY")):
            h = Knowledge(2, (Guidance("construction_guidance", "Should not be exposed", scope),))
            script = Script([[action("commit", plan_id="fixed", reason="Explicit keep")]])
            result, _ = self.run_script(script, knowledge=h)
            self.assertEqual(result.status, "COMPLETE")
            self.assertEqual(result.calls, 1)
            self.assertEqual(script.requests[0]["guidance"]["loaded"], [])
            self.assertEqual(script.requests[0]["guidance"]["applicability"][0]["state"], expected)

    def test_private_adapter_output_is_rejected_before_another_call(self):
        adapter = FakeAdapter()
        adapter.inspect_data = lambda *a, **kw: {"C_B": {"score": 0.001}}
        script = Script([[action("inspect_data")], [action("commit", plan_id="fixed", reason="x")]])
        result, _ = self.run_script(script, adapter=adapter)
        self.assertEqual(result.status, "INCOMPLETE")
        self.assertEqual(result.calls, 1)
        self.assertIsNone(result.committed_plan_id)
        self.assertFalse(any(x.get("output", {}).get("C_B") for x in result.trace))

    def test_call_and_tool_exhaustion_never_fabricate_a_commit(self):
        cases = [
            (Script([[action("overview")], [action("overview")]]), Limits(max_calls=2)),
            (Script([[action("overview"), action("overview"), action("overview")]]), Limits(max_tools=2)),
        ]
        for script, limits in cases:
            result, _ = self.run_script(script, limits=limits)
            self.assertEqual(result.failure_kind, "BUDGET_EXHAUSTED")
            self.assertIsNone(result.committed_plan_id)
            self.assertIsNone(result.delivery_model_ref)

    def test_new_candidate_budget_is_reserved_before_fitting(self):
        script = Script([[action("build_material")], [action("evaluate", plan_id="q1")],
                         [action("build_material")], [action("evaluate", plan_id="q2")]])
        result, adapter = self.run_script(script, limits=Limits(max_new_evaluations=1))
        self.assertEqual(result.new_evaluations, 1)
        self.assertEqual(len(adapter.fits), 1)
        self.assertEqual(result.failure_kind, "BUDGET_EXHAUSTED")

    def test_commit_does_not_train_an_unevaluated_heldin_plan(self):
        script = Script([[action("build_material")], [action("commit", plan_id="q1", reason="x")]])
        result, adapter = self.run_script(script)
        self.assertEqual(result.status, "INCOMPLETE")
        self.assertEqual(adapter.fits, [])

    def test_deployment_can_train_and_commit_without_feedback(self):
        script = Script([[action("build_material")], [action("commit", plan_id="q1", reason="Deploy")]])
        result, adapter = self.run_script(script, mode="deploy", baselines=())
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(adapter.fits, [("q1", SEEDS, False)])
        self.assertNotIn("evaluate", script.requests[0]["tools"])
        self.assertNotIn("compare", script.requests[0]["tools"])
        bad = Script([[action("evaluate", plan_id="q1")]])
        rejected, adapter = self.run_script(bad, mode="deploy", baselines=())
        self.assertEqual(rejected.failure_kind, "PARSE_OR_VALIDATION_FAILED")
        self.assertEqual(adapter.fits, [])

    def test_first_entity_is_not_a_substitute_for_the_batch(self):
        adapter = FakeAdapter()
        adapter.overview = lambda: {"entities": [{"entity_index": 0}], "batch_features": {}}
        script = Script([[action("commit", plan_id="fixed", reason="x")]])
        result, _ = self.run_script(script, adapter=adapter)
        self.assertEqual(result.status, "INCOMPLETE")
        self.assertEqual(result.calls, 0)

    def test_foreign_job_or_incomplete_seed_cannot_enter_feedback(self):
        for baseline in (candidate("fixed", 1, job="other"), candidate("fixed", 1, seeds=(11, 12))):
            script = Script([[action("commit", plan_id="fixed", reason="x")]])
            result, _ = self.run_script(script, baselines=(baseline,))
            self.assertEqual(result.status, "INCOMPLETE")
            self.assertEqual(result.calls, 0)

    def test_transport_failure_is_not_an_identity_result(self):
        def failing_client(request):
            raise RuntimeError("secret transport detail must not appear in the public trace")
        result, _ = self.run_script(failing_client)
        self.assertEqual(result.failure_kind, "AGENT_CALL_FAILED")
        self.assertIsNone(result.committed_plan_id)
        self.assertNotIn("secret transport", repr(result.trace))

    def test_no_actions_after_commit(self):
        script = Script([[action("commit", plan_id="fixed", reason="x"), action("overview")]])
        result, adapter = self.run_script(script)
        self.assertEqual(result.failure_kind, "PARSE_OR_VALIDATION_FAILED")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(adapter.fits, [])

    def test_scope_and_one_hook_update_reuse_existing_evaluator(self):
        self.assertEqual(scope_state({"const": True}, {}, FIELDS)[0], "MATCH")
        self.assertEqual(scope_state({"feature": "lag24_corr", "op": ">", "value": 0.5},
                                    {"lag24_corr": float("nan")}, FIELDS)[0], "UNKNOWN")
        parent = Knowledge()
        proposal = {"decision": "PROPOSE", "hook": "decision_guidance", "body": "Report uncertainty before commit",
                    "observable_applicability": {"const": True}, "evidence_refs": ["job:1"], "rationale": "Observed"}
        child = apply_update(parent, proposal, allowed_features=FIELDS, legal_evidence_refs=frozenset({"job:1"}))
        self.assertEqual(parent.entries, ())
        self.assertEqual(child.version, 1)
        self.assertEqual(child.entries[0].hook, "decision_guidance")
        self.assertEqual(apply_update(child, {"decision": "KEEP", "rationale": "No supported edit"},
                                      allowed_features=FIELDS, legal_evidence_refs=frozenset()), child)
        for change in ({"observable_applicability": None}, {"evidence_refs": ["unseen"]},
                       {"observable_applicability": {"feature": "entity_id", "op": "==", "value": 5}}):
            with self.assertRaises(ValueError):
                apply_update(parent, proposal | change, allowed_features=FIELDS, legal_evidence_refs=frozenset({"job:1"}))



class RealAdapterContracts(unittest.TestCase):
    """Real T/material arithmetic; mocked fitting only where explicitly stated."""
    def test_delayed_guards_precede_any_loader(self):
        import tempfile
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT))
        from methods.ttha.batch_base import commit
        with tempfile.TemporaryDirectory() as d, patch.object(commit.context, 'open_job', side_effect=AssertionError('loaded before guard')) as loader:
            for name in ('open_c_b','freeze_e','score_e'):
                with self.assertRaises(PermissionError): getattr(commit,name)(d,'electricity','a55')
            loader.assert_not_called()

    def test_global_e_guard_precedes_loader_and_requires_every_peer(self):
        import tempfile
        import json
        from unittest.mock import patch
        sys.path.insert(0, str(ROOT))
        from methods.ttha.batch_base import commit
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            branches = [root / 'parent', root / 'child']
            def write(path, obj):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(obj), encoding='utf-8')
            write(root / 'experiment_started.json', {})
            write(root / 'execution_finished.json', {'branches': [[str(b), 'a55'] for b in branches]})
            for b in branches:
                write(b / 'a55/c_b_scores.json', {})
                (b / 'a55/pred.npz').touch()
                write(b / 'a55/e_frozen.json', {'models': {'cell': {'path': str(b / 'a55/pred.npz')}}})
            with patch.object(commit.context, 'open_job', side_effect=RuntimeError('valid boundary reached')) as loader:
                with self.assertRaises(PermissionError):
                    commit.score_e(branches[0], 'electricity', 'a55')
                write(root / 'all_e_predictions_frozen.json', {'branches': [str(branches[0])]})
                with self.assertRaises(PermissionError):
                    commit.score_e(branches[0], 'electricity', 'a55')
                write(root / 'all_e_predictions_frozen.json', {'branches': [str(b) for b in branches]})
                (branches[1] / 'a55/pred.npz').unlink()
                with self.assertRaises(PermissionError):
                    commit.score_e(branches[0], 'electricity', 'a55')
                loader.assert_not_called()
                (branches[1] / 'a55/pred.npz').touch()
                with self.assertRaisesRegex(RuntimeError, 'valid boundary reached'):
                    commit.score_e(branches[0], 'electricity', 'a55')
                self.assertEqual(loader.call_count, 1)

    def test_real_material_and_full_batch_adapter(self):
        import tempfile
        import numpy as np
        sys.path.insert(0, str(ROOT))
        from evaluation.main_protocol_p4 import batch_research_runtime as rt
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            led=rt.RuntimeLedger(d/'budget.json',max_fit_attempts=1,max_wall_s=300)
            a=rt.Adapter(d/'run','a55',led,ROOT)
            ov=a.overview()
            self.assertEqual(len(ov['entities']),32)
            self.assertEqual(a.ctx.slice.row_end,a.ctx.job.t)
            self.assertEqual(ov['batch_features']['batch_median_lag24_corr'],float(np.median([r['lag24_corr'] for r in ov['entities']])))
            c=a.build_material({'plan_id':'R','policy':rt.policy.fixed_mixup_policy()},remaining_seconds=300)
            ch=np.load(ROOT/'_scratch/ts_aug_donor_pattern_probe/materials/a55__children.npz')
            z=np.load(a.refs['R'].path)
            self.assertTrue(np.array_equal(z['Xc'],ch['RX']))
            self.assertTrue(np.array_equal(z['Yc'],ch['RY']))
            z.close();ch.close()
            with self.assertRaises(PermissionError): a.inspect_data({'entity_indices':[0],'kind':'segment','sub_range':[a.ctx.job.t,a.ctx.job.t+48]},remaining_seconds=300)
            with self.assertRaises(ValueError): a.inspect_data({'entity_indices':[-1]},remaining_seconds=300)
            a.build_material({'plan_id':'same','policy':rt.policy.fixed_mixup_policy()},remaining_seconds=300)
            self.assertEqual(a.refs['same'].alias_of,'R')
            self.assertEqual(led.s['fit_attempts'],0)

    def test_full_feedback_tensor_matches_frozen_metric(self):
        import numpy as np
        sys.path.insert(0, str(ROOT))
        from methods.ttha.batch_base import data
        target=np.arange(32*2*48,dtype=float).reshape(32,2,48)/100
        pred=target+np.arange(1,33)[:,None,None]
        sc=data.score_predictions(pred,target,np.zeros(32),np.ones(32),np.ones(32))
        tensor=np.asarray(sc['per_origin_entity_normalized_mse'])
        self.assertEqual(tensor.shape,(2,32))
        self.assertEqual(float(tensor.mean()),sc['normalized_mse_macro'])

    def test_no_feedback_worker_opens_only_material(self):
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT))
        from methods.ttha.batch_base import train
        with patch.object(train.context,'open_job',side_effect=RuntimeError('stop before training')) as opened:
            with self.assertRaises(RuntimeError): train.fit_one('electricity','a55','unused','N',1,feedback=False)
            self.assertEqual(opened.call_args.kwargs['stage'],'material')


class RuntimeCostContracts(unittest.TestCase):
    def test_fit_is_reserved_before_worker_and_timeout_is_bounded(self):
        import tempfile
        import subprocess
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT))
        from evaluation.main_protocol_p4 import batch_research_runtime as rt
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); (d/'a55').mkdir()
            led=rt.RuntimeLedger(d/'budget.json',max_fit_attempts=1,max_wall_s=10)
            ctx=NS(job=NS(job_id='a55',dataset='electricity'),job_dir=d/'a55',run_dir=d)
            ref=NS(material_id='N',alias_of=None)
            def interrupted(cmd,**kwargs):
                self.assertEqual(led.s['fit_attempts'],1)
                self.assertLessEqual(kwargs['timeout'],10)
                raise subprocess.TimeoutExpired(cmd,kwargs['timeout'])
            with patch.object(rt.subprocess,'run',side_effect=interrupted):
                with self.assertRaises(RuntimeError):rt.fit(ctx,ref,(1,),led,ROOT)
            self.assertEqual(led.s['fits_failed'],1)
            with self.assertRaises(rt.budget.BudgetExhausted):led.reserve_fit('second')

    def test_model_mismatch_is_saved_and_stops_client(self):
        import tempfile
        from types import SimpleNamespace as NS
        sys.path.insert(0,str(ROOT))
        from evaluation.main_protocol_p4 import batch_research_runtime as rt
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            led=rt.RuntimeLedger(d/'budget.json',max_fit_attempts=0,max_llm_requests=2,max_llm_tokens=100000,max_wall_s=100)
            reply=NS(model='wrong-model',usage=NS(prompt_tokens=12,completion_tokens=3),choices=[],model_dump=lambda **kw:{'model':'wrong-model'})
            calls=[]
            def create(**kw):
                calls.append(kw)
                self.assertEqual(led.s['llm_requests'],1)
                self.assertEqual(led.s['llm_http_attempts'],1)
                return reply
            c=rt.MeteredClient.__new__(rt.MeteredClient)
            c.api=NS(chat=NS(completions=NS(create=create)));c.ledger=led;c.out=d;c.fatal=False;c.last_text=None
            with self.assertRaises(rt.llm.AccountFault): c.call('fast','test',{},'synthetic test',max_tokens=10)
            self.assertTrue((d/'001_fast_response.json').exists())
            with self.assertRaises(rt.llm.AccountFault): c.call('fast','test',{},'synthetic test',max_tokens=10)
            self.assertEqual(len(calls),1)
            self.assertEqual(led.s['llm_tokens_in'],12)

if __name__ == "__main__":
    unittest.main()
