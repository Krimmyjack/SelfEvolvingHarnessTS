"""Focused run-permission checks: zero API/raw-data, one synthetic Ridge fit."""
import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.main_protocol_p4 import dev_train1_runner as R
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _verification_limits, _actionable_operators
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view


def main():
    loader = importlib.util.spec_from_file_location("train_fixture", Path(__file__).with_name("test_runner_contract.py"))
    fixtures = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(fixtures)
    bundle = fixtures._bundle()
    legacy = fixtures._spec(bundle)
    modern = replace(legacy, maximum_modified_fraction=1.0)
    machinery = R.load_machinery()
    before = {str(p.relative_to(R.H0_ROOT)): p.read_bytes() for p in R.H0_ROOT.rglob("*") if p.is_file()}
    parent = R.load_h0_snapshot(machinery)
    candidate = R.configure_snapshot_for_spec(modern, parent, machinery)
    assert R.configure_snapshot_for_spec(legacy, parent, machinery) is parent
    assert candidate.runtime_bundle_sha != parent.runtime_bundle_sha
    assert candidate.instruction == parent.instruction and candidate.skills == parent.skills
    expected = dict(parent.verification)
    expected["max_modified_fraction"] = 1.0
    assert dict(candidate.verification) == expected
    assert before == {str(p.relative_to(R.H0_ROOT)): p.read_bytes() for p in R.H0_ROOT.rglob("*") if p.is_file()}
    observations = bundle.values[bundle.eval_uids[0]][:192]
    for spec, snapshot, expected_cap in ((legacy,parent,.35),(modern,candidate,1.0)):
        request = R.make_request(uid=bundle.eval_uids[0], observed=observations,
                                 spec=spec, task_ctx=R.task_context_for(spec))
        view = resolve_harness_view(snapshot, R.extract_public_features(observations,task_kind="forecast"),role="fast")
        cap, preserve = _verification_limits(request,view)
        assert cap == expected_cap and preserve
        actionable = _actionable_operators(request, observations, view, ("fft_decompose","smooth_ma","smooth_ema"))
        assert ("fft_decompose" in actionable) == (expected_cap == 1.0), actionable
    steps = (("fft_decompose",{}),)
    assignment = {uid:steps for uid in bundle.train_uids}
    try:
        R.assert_train_windows_legal(legacy,bundle,assignment,training_design="whole_window")
    except R.numeric.PreparationFailed:
        pass
    else:
        raise AssertionError("legacy train cap did not reject smoothing")
    R.assert_train_windows_legal(modern,bundle,assignment,training_design="whole_window")
    budget = R.Budget(max_fits=4)
    with tempfile.TemporaryDirectory(prefix="train-cap-test-") as directory:
        path = Path(directory)
        model = R.freeze_model(spec=modern,bundle=bundle,assignment=assignment,budget=budget,arm="modern",run_dir=path)
        assert budget.fits() == 1
        ev = {uid:steps for uid in bundle.eval_uids}
        R.predict_arm(spec=modern,bundle=bundle,model=model,eval_assignment=ev,origin=360)
        try:
            R.predict_arm(spec=legacy,bundle=bundle,model=model,eval_assignment=ev,origin=360)
        except R.numeric.PreparationFailed:
            pass
        else:
            raise AssertionError("legacy predict cap did not reject smoothing")
        saved_model = (path/"modern/frozen_ridge.json").read_bytes()
        try:
            R.freeze_model(spec=legacy,bundle=bundle,assignment=assignment,budget=budget,arm="modern",run_dir=path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("cross-cap model reuse passed")
        assert saved_model == (path/"modern/frozen_ridge.json").read_bytes() and budget.fits()==1
        R.atomic_write_json(path/"preflight.json",{"legacy":True})
        prior = (path/"preflight.json").read_bytes()
        R.validate_run_cap(legacy,path)
        try:
            R.validate_run_cap(modern,path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("cross-cap legacy preflight passed")
        assert prior == (path/"preflight.json").read_bytes()
    payload = {"status":"PASS","project_llm_calls":0,"synthetic_fits":budget.fits(),
               "checks":["real compiled snapshot, canonical unchanged","real Fast limit and FFT actionability",
                         "train and predict cap agreement","cross-cap model restore rejects",
                         "legacy preflight missing cap remains035","other verification fields preserved"]}
    R.atomic_write_json(Path(__file__).with_name("uncapped_contract_test.json"),payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
