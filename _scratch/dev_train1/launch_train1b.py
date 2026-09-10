"""Bounded 10/10 TRAIN-1B launch, reusing the existing numerical runner."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import os
from pathlib import Path
import shutil
import sys
import time
import winreg

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4.dev_train1b_observations import install_observations

RUN_ID = "dev_train1b_role_observations_10x10_20260910"
RUN_DIR = R.RUNS_ROOT / RUN_ID
RECEIPT_PATH = ROOT / ".aris/runs/dev-train1b-10x10-20260910/launch.json"


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ModelCheckedBackend:
    def __init__(self):
        self.inner = R.build_live_backend(maximum_calls=R.MAX_LLM)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, request):
        response = self.inner.complete(request)
        identity = (getattr(response, "provider_metadata", None) or {}).get("returned_model")
        if identity != R.REQUIRED_RETURNED_MODEL:
            raise R.AccountFault("MODEL_IDENTITY_MISMATCH: requested alias did not return the frozen model")
        return response


def main():
    if not os.environ.get("CPA_API_KEY"):
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
            os.environ["CPA_API_KEY"] = winreg.QueryValueEx(env, "CPA_API_KEY")[0]
    if not os.environ.get("CPA_API_KEY", "").strip():
        raise RuntimeError("CPA_API_KEY is unavailable")

    roster = R.development_roster_from_metadata()
    train_uids, eval_uids = roster["train_uids"][:10], roster["eval_uids"][:10]
    if len(set(train_uids)) != 10 or len(set(eval_uids)) != 10 or set(train_uids) & set(eval_uids):
        raise RuntimeError("10/10 role roster is incomplete or overlapping")
    spec = R.RunSpec(train_uids=train_uids, eval_uids=eval_uids, concurrency=4, live_api=True)
    protected = {}
    resuming = "--resume-failed" in sys.argv
    if (RUN_DIR / "result.json").exists() and not resuming:
        raise RuntimeError("existing result: explicit --resume-failed is required")
    if resuming:
        check = R.read_json(RUN_DIR / "recovery_transport_check.json")
        if check.get("status") != "OK" or check.get("returned_model") != R.REQUIRED_RETURNED_MODEL:
            raise RuntimeError("recovery transport check did not pass")
        if time.time() - float(check["time"]) > 900:
            raise RuntimeError("recovery transport check is older than 15 minutes")
        pending = []
        for arm in ("old_skill_formation", "old_skill", "new_skill"):
            files = sorted((RUN_DIR / arm).glob("eval_v_*.json")) + [RUN_DIR / arm / "train_w.json"]
            for path in files:
                data = R.read_json(path)
                missing = [r for r in data["sequences"] if not R.checkpoint_is_final(r)]
                if missing:
                    if arm != "new_skill" or not path.name.startswith("eval_v_"):
                        raise RuntimeError("recovery would re-ask an unapproved decision")
                    pending.extend(r["key"] for r in missing)
                else:
                    protected[str(path.relative_to(RUN_DIR))] = data
        if not 0 < len(pending) <= 20:
            raise RuntimeError("expected at most the twenty failed prediction decisions")
        for path in (RUN_DIR / "boundary.json", RUN_DIR / "slow_input.json", *RUN_DIR.glob("*/frozen_ridge.json")):
            protected[str(path.relative_to(RUN_DIR))] = R.read_json(path)
        archive = RUN_DIR / "before_transport_recovery"
        if archive.exists():
            raise RuntimeError("prior recovery archive exists; inspect it before another recovery attempt")
        for rel in ("result.json", "budget.json", "new_skill/eval_v_u012.json", "new_skill/eval_v_u013.json",
                    "new_skill/predictions_u012.json", "new_skill/predictions_u013.json"):
            target = archive / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUN_DIR / rel, target)
        R.atomic_write_json(archive / "recovery_scope.json", {"pending_keys": pending,
            "protected_paths": list(protected), "allowed": "failed new_skill prediction decisions only"})
    contract = {
        "package": "DEV-TRAIN-1B", "data_role": "EXPOSED_DEVELOPMENT_VARIANT__KDD2018_WITH_MISSING",
        "train_uids": train_uids, "eval_uids": eval_uids,
        "roster_rule": "first ten in each existing span [40,80] role roster; not outcome-selected",
        "formation_units": [list(x) for x in spec.formation_units],
        "eval_units": [list(x) for x in spec.eval_units],
        "model_requested": spec.requested_model, "model_returned_required": R.REQUIRED_RETURNED_MODEL,
        "base_url": spec.base_url, "fast_concurrency": 4, "fit_writers": 1,
        "max_llm": spec.max_llm, "max_fits": spec.max_fits, "max_wall_seconds": spec.max_wall_seconds,
        "observation_version": "window_role_summary_v1", "same_interface_for_all_agent_arms": True,
        "direct_comparison_with_old_20x20_not_allowed": True,
    }
    contract_path = RUN_DIR / "run_contract.json"
    if contract_path.exists():
        if R.read_json(contract_path) != contract:
            raise RuntimeError("existing run contract differs; refusing to reuse checkpoints")
    else:
        R.atomic_write_json(contract_path, contract)
    receipt = {
        "status": "STARTING", "backend": "local Windows project Python; CPU Ridge plus API",
        "experiment_pid": os.getpid(), "started_at_utc": utc_now(), "cwd": str(ROOT),
        "command": "D:\\Anaconda_envs\\envs\\project\\python.exe _scratch/dev_train1/launch_train1b.py" + (" --resume-failed" if resuming else ""),
        "run_dir": str(RUN_DIR), "log_path": str(RUN_DIR / "live.log"),
        "expected_output": str(RUN_DIR / "result.json"), "gpu": None,
        "estimated_duration": "about 1-2 active hours; endpoint dependent", "resource_limits": contract,
        "estimated_cost": "API attempts <=2000, CPU fits <=500; no new paid compute service",
        "resuming_failed_predictions_only": resuming,
    }
    if RECEIPT_PATH.exists():
        receipt["previous_launch"] = R.read_json(RECEIPT_PATH)
    R.atomic_write_json(RECEIPT_PATH, receipt)

    R.PACKAGE = "DEV-TRAIN-1B"
    R.PRIOR_PACKAGE = {"llm": 0, "fits": 0, "source": "new package; previous package costs remain in original receipts"}
    original_preflight = R.preflight_record

    def preflight(s, snapshot):
        value = original_preflight(s, snapshot)
        value["roster"] = {"n_train": len(s.train_uids), "n_eval": len(s.eval_uids),
                           "train_uids": list(s.train_uids), "eval_uids": list(s.eval_uids),
                           "rule": contract["roster_rule"], "span": [40, 80]}
        value["observation_version"] = contract["observation_version"]
        return value

    R.preflight_record = preflight
    code = 2
    with (RUN_DIR / "live.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print("START DEV-TRAIN-1B 10/10, four Fast workers, pid=%s" % os.getpid(), flush=True)
            try:
                full = R.load_kdd_missing_development_bundle()
                bundle = R.DataBundle(values={uid: full.values[uid] for uid in train_uids + eval_uids},
                                      train_uids=train_uids, eval_uids=eval_uids, period=full.period)
                del full
                machinery = R.load_machinery()
                snapshot = R.load_h0_snapshot(machinery)
                if resuming and snapshot.runtime_bundle_sha != R.read_json(RUN_DIR / "preflight.json")["h0_runtime_bundle_sha"]:
                    raise RuntimeError("h0 changed since the original run; recovery stopped")
                receipt["status"] = "RUNNING"
                R.atomic_write_json(RECEIPT_PATH, receipt)
                with install_observations(RUN_DIR):
                    result = R.run_package(spec=spec, bundle=bundle, run_id=RUN_ID,
                                           backend_factory=ModelCheckedBackend, machinery=machinery, snapshot=snapshot)
                code = int(result.get("exit_code", 2))
                receipt["result_status"] = result.get("status")
                incomplete = []
                for name, arm in result.get("arms", {}).items():
                    if name == "new_skill" and arm.get("treatment") != "UPDATE_TREATMENT":
                        continue
                    for _pos, origin in spec.eval_units:
                        score = (arm.get("scores") or {}).get(str(origin), {})
                        if score.get("n_readable") != len(spec.eval_uids) or score.get("mean_smase") in (None, "UNKNOWN"):
                            incomplete.append([name, origin])
                receipt["incomplete_readouts"] = incomplete
                receipt["readout_status"] = "INCOMPLETE" if incomplete else "COMPLETE"
                if incomplete and code == 0:
                    code = 2
                if resuming:
                    changed = [rel for rel, before in protected.items() if R.read_json(RUN_DIR / rel) != before]
                    receipt["recovery_protected_state"] = {"checked_files": len(protected), "changed": changed}
                    if changed:
                        code = 2
            except Exception as exc:
                print(type(exc).__name__, R._sanitize(exc, 500), flush=True)
            finally:
                R.preflight_record = original_preflight
                receipt.update(status="PROCESS_EXITED", exit_code=code, ended_at_utc=utc_now())
                if (RUN_DIR / "budget.json").exists():
                    receipt["final_budget"] = R.read_json(RUN_DIR / "budget.json")
                R.atomic_write_json(RECEIPT_PATH, receipt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
