"""Approved TRAIN-2: four arms, four total Fast slots, serial CPU fits."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import os
from pathlib import Path
import sys
import time

# No thread-heavy BLAS contention alongside the API workers.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4 import dev_train2_workflow as W

RUN_ID = "dev_train2_uncapped_workflow_10x10_20260910"
RUN_DIR = R.RUNS_ROOT / RUN_ID
RECEIPT = ROOT / ".aris/runs" / RUN_ID / "launch.json"


class ModelCheckedBackend:
    def __init__(self):
        self.inner = R.build_live_backend(maximum_calls=R.MAX_LLM)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, request):
        response = self.inner.complete(request)
        returned = (getattr(response, "provider_metadata", None) or {}).get("returned_model")
        if returned != R.REQUIRED_RETURNED_MODEL:
            raise R.AccountFault("MODEL_IDENTITY_MISMATCH: frozen requested/returned model contract")
        return response


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    roster = R.development_roster_from_metadata()
    train, evaluate = roster["train_uids"][:10], roster["eval_uids"][:10]
    if len(set(train)) != 10 or len(set(evaluate)) != 10 or set(train) & set(evaluate):
        raise RuntimeError("incomplete or overlapping 10/10 roster")
    spec = W.make_spec(train, evaluate, live_api=True, concurrency=4,
                       maximum_modified_fraction=1.0)
    contract = W.build_run_contract(spec)
    R.validate_run_cap(spec, RUN_DIR)
    if args.preflight:
        # Reads only metadata; no dataset, API, fit, or run-directory writes.
        if (RUN_DIR / "run_contract.json").is_file():
            if R.read_json(RUN_DIR / "run_contract.json") != contract:
                raise RuntimeError("existing run contract differs")
        print("PREFLIGHT_OK 10/10; seven workflows; cap=1; Fast=4 total; fits=1 writer", flush=True)
        return 0
    if (RUN_DIR / "result.json").is_file():
        old = R.read_json(RUN_DIR / "result.json")
        if old.get("status") == "COMPLETED" or not args.resume:
            raise RuntimeError("existing result; no automatic rerun of a completed package")
    elif (RUN_DIR / "budget.json").is_file() and not args.resume:
        raise RuntimeError("existing partial package; inspect it, then use --resume")
    if not os.environ.get("CPA_API_KEY"):
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
            os.environ["CPA_API_KEY"] = winreg.QueryValueEx(env, "CPA_API_KEY")[0]
    if not os.environ.get("CPA_API_KEY", "").strip():
        raise RuntimeError("CPA_API_KEY unavailable")
    W.write_or_check_contract(RUN_DIR, contract)
    receipt = {
        "status": "STARTING", "backend": "local Windows project Python; CPU Ridge plus API",
        "experiment_pid": os.getpid(), "started_at_utc": utc_now(), "cwd": str(ROOT),
        "command": "D:\\Anaconda_envs\\envs\\project\\python.exe _scratch/dev_train1/launch_train2.py" + (" --resume" if args.resume else ""),
        "run_dir": str(RUN_DIR), "log_path": str(RUN_DIR / "live.log"),
        "expected_output": str(RUN_DIR / "result.json"), "gpu": None,
        "estimated_duration": "endpoint dependent; bounded by six-hour package ceiling",
        "estimated_cost": "<=2000 API attempts, <=500 CPU fits; no new compute service",
        "resource_limits": contract,
    }
    if RECEIPT.is_file():
        receipt["previous_launch"] = R.read_json(RECEIPT)
    R.atomic_write_json(RECEIPT, receipt)
    R.PACKAGE = W.PACKAGE
    R.PRIOR_PACKAGE = W.THIS_PACKAGE_PRIOR
    code = 2
    with (RUN_DIR / "live.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print("START TRAIN-2; pid=%s; cap=1; global Fast=4; fits serial" % os.getpid(), flush=True)
            try:
                full = R.load_kdd_missing_development_bundle()
                bundle = R.DataBundle(values={uid: full.values[uid] for uid in train + evaluate},
                                      train_uids=train, eval_uids=evaluate, period=full.period)
                del full
                print("DATA_READY train=10 eval=10", flush=True)
                receipt["status"] = "RUNNING"
                R.atomic_write_json(RECEIPT, receipt)
                result = W.run(spec=spec, bundle=bundle, run_id=RUN_ID,
                               backend_factory=ModelCheckedBackend)
                code = int(result.get("exit_code", 2))
                receipt["result_status"] = result.get("status")
                receipt["incomplete_readouts"] = result.get("incomplete_readouts")
                print("FINISH", result.get("status"), "exit", code, flush=True)
            except Exception as exc:
                receipt["error"] = "%s: %s" % (type(exc).__name__, R._sanitize(exc))
                print(receipt["error"], flush=True)
            finally:
                receipt.update(status="PROCESS_EXITED", exit_code=code, ended_at_utc=utc_now())
                if (RUN_DIR / "budget.json").is_file():
                    receipt["final_budget"] = R.read_json(RUN_DIR / "budget.json")
                R.atomic_write_json(RECEIPT, receipt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
