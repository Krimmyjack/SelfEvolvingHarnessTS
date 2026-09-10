"""Only TRAIN-1B's new input paths; fake transport, zero API and zero fits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import tempfile
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4 import dev_train1b_observations as O
from test_runner_contract import ScriptedBackend, _public_input


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from walk(child)


def plain(value):
    return json.dumps(value, sort_keys=True, default=R._json_default)


def main():
    assert {key for key in O.RUNTIME_CAPABILITIES if key.endswith("_probe_direction")} == {
        "imputation_probe_direction", "clipping_probe_direction", "denoising_probe_direction", "level_probe_direction"}
    spec = R.RunSpec(train_uids=["train"], eval_uids=["eval"], concurrency=4)
    raw = 8 + np.sin(np.arange(1200) * 2 * np.pi / 24)
    raw[315:325] = np.nan
    train = O.execution_observations(raw[:900], "train_workflow", spec)
    lengths = [node["n_points"] for node in walk(train) if "n_points" in node]
    assert lengths.count(240) == len(spec.anchors), lengths
    assert lengths.count(192) == len(spec.anchors), lengths
    assert lengths.count(48) == len(spec.anchors), lengths
    for size in (48, 192, 240):
        assert any(node.get("n_points") == size for node in walk(train))
    assert "NaN" not in plain(train), "public JSON contains nonfinite values"
    try:
        O.execution_observations(raw[:191], "predict_input", spec)
    except ValueError:
        pass
    else:
        raise AssertionError("prediction accepted the wrong observation length")
    print("PASS exact role windows and lengths")

    orig_fast, orig_extra, orig_slow = R.run_fast_decision, R.public_extra, R.build_slow_card
    barrier = threading.Barrier(4)
    values = {}
    for i in range(4):
        item = raw.copy()
        item[808:808+i*10] = np.nan
        values[str(i)] = item
    bundle = R.DataBundle(values=values, train_uids=[], eval_uids=list(values))
    def stub_fast(**kw):
        barrier.wait(timeout=10)
        extra = R.public_extra(kw["role"], observed_length=192, spec=kw["spec"])
        return {"series_uid": kw["uid"], "public_seen": extra}
    R.run_fast_decision = stub_fast
    with tempfile.TemporaryDirectory(prefix="train1b-input-") as tmp:
        try:
            with O.install_observations(Path(tmp)):
                def one(uid):
                    return R.run_fast_decision(spec=spec,bundle=bundle,snapshot=None,machinery=None,
                        backend_factory=None,uid=uid,role="predict_input",arm="test",position=1,
                        origin=1000,task_ctx=None)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    rows = list(pool.map(one, bundle.eval_uids))
                for row in rows:
                    expected = O.execution_observations(values[row["series_uid"]][808:1000], "predict_input", spec)
                    assert plain(row["execution_observations"]) == plain(expected)
                    assert plain(row["public_seen"]["execution_observations"]) == plain(expected)
                assert "execution_observations" not in R.public_extra("slow", observed_length=0, spec=spec)
            assert R.run_fast_decision is stub_fast
        finally:
            R.run_fast_decision = orig_fast
    assert R.public_extra is orig_extra and R.build_slow_card is orig_slow
    print("PASS four-worker isolation and restored hooks")

    machinery = R.load_machinery()
    snapshot = R.load_h0_snapshot(machinery)
    bundle = R.DataBundle(values={"train":raw, "eval":raw},train_uids=["train"],eval_uids=["eval"])
    task_ctx = R.task_context_for(spec)
    captures = []
    def factory():
        backend = ScriptedBackend(mode="identity")
        captures.append(backend)
        return backend
    with tempfile.TemporaryDirectory(prefix="train1b-flow-") as tmp:
        with O.install_observations(Path(tmp)):
            train_row = R.run_fast_decision(spec=spec,bundle=bundle,snapshot=snapshot,machinery=machinery,
                backend_factory=factory,uid="train",role="train_workflow",arm="test",position=0,
                origin=None,task_ctx=task_ctx)
            eval_row = R.run_fast_decision(spec=spec,bundle=bundle,snapshot=snapshot,machinery=machinery,
                backend_factory=factory,uid="eval",role="predict_input",arm="test",position=9,
                origin=1000,task_ctx=task_ctx)
            for capture in captures:
                assert len(capture.requests) >= 2
                for request in capture.requests:
                    value = _public_input(request)
                    assert "runtime_capabilities" in value and "execution_observations" in value
            card = R.build_slow_card(spec=spec,train_records=[train_row],
                eval_records_by_origin={1000:[eval_row]},formation_scores={1000:{"per_uid_utility":{"eval":-0.1}}},
                xy_decomposition={"status":"not_measured_in_fake_test"})
            assert plain(card["training_batch"]["members"][0]["execution_observations"]) == plain(train_row["execution_observations"])
            assert plain(card["failing_eval_decisions"][0]["execution_observations"]) == plain(eval_row["execution_observations"])
            assert card["training_batch"]["members"][0]["loo_credit"] == "UNKNOWN"
            assert (Path(tmp)/"slow_input.json").is_file()
    assert R.run_fast_decision is orig_fast and R.public_extra is orig_extra and R.build_slow_card is orig_slow
    print("PASS real Fast public_input and Slow evidence delivery; API=0 fits=0")


if __name__ == "__main__":
    main()
