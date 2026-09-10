"""Local DEV-TRAIN-1 launch; secrets stay in the process environment."""
import json
import os
from pathlib import Path
import sys
import time
import winreg

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
if not os.environ.get("CPA_API_KEY"):
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
        os.environ["CPA_API_KEY"] = winreg.QueryValueEx(env, "CPA_API_KEY")[0]

from evaluation.main_protocol_p4 import dev_train1_runner as R
from evaluation.main_protocol_p4.run_dev_train1 import main

run_id = "dev_train1_live_20260910"
run_dir = R.RUNS_ROOT / run_id
run_dir.mkdir(parents=True, exist_ok=True)
receipt_path = ROOT / ".aris/runs/dev-train1-20260910/launch.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if "--resume-after-reboot" in sys.argv:
    stop_upper = float(sys.argv[sys.argv.index("--resume-after-reboot") + 1])
    budget_path = run_dir / "budget.json"
    before = R.read_json(budget_path)
    recovery_path = run_dir / "recovery_after_reboot.json"
    if recovery_path.exists():
        raise RuntimeError("this reboot recovery already recorded; do not deduct downtime twice")
    resumed_at = time.time()
    recovery = {
        "reason": "user requested recovery after Windows planned update restart",
        "prior_process_absent": True,
        "stop_upper_epoch": stop_upper,
        "stop_evidence": "Windows LastBootUpTime=1788982430.5; conservative charging through boot after System 1074/6006",
        "resumed_at_epoch": resumed_at,
        "excluded_downtime_seconds": max(0.0, resumed_at - stop_upper),
        "budget_before": before,
        "formation_and_slow_frozen": True,
        "old_train_before": R.read_json(run_dir / "old_skill/train_w.json"),
    }
    R.atomic_write_json(recovery_path, recovery)
    before["paused_seconds"] = float(before.get("paused_seconds") or 0.0) + recovery["excluded_downtime_seconds"]
    R.atomic_write_json(budget_path, before)
    receipt["previous_launch"] = {key: receipt.get(key) for key in ("experiment_pid", "started_at_utc", "status")}
    receipt["recovery"] = str(recovery_path)
receipt.update({
    "status": "LAUNCHING", "experiment_pid": os.getpid(),
    "command": "python -m evaluation.main_protocol_p4.run_dev_train1 --phase run --run-id " + run_id + " --live --concurrency 4",
    "cwd": str(ROOT), "run_dir": str(run_dir),
    "log_path": str(run_dir / "live.log"),
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "course_started": (run_dir / "budget.json").exists(), "synthetic_fits_before_course": 43,
    "concurrency": {"requested_fast": 4, "fit_writers": 1},
})
R.atomic_write_json(receipt_path, receipt)
with (run_dir / "live.log").open("a", encoding="utf-8", buffering=1) as log:
    sys.stdout = sys.stderr = log
    print("START DEV-TRAIN-1 pid=%s concurrency=4" % os.getpid(), flush=True)
    try:
        code = main(["--phase", "run", "--run-id", run_id, "--live", "--concurrency", "4"])
    except Exception as exc:
        print(type(exc).__name__, R._sanitize(exc, 500), flush=True)
        code = 2
    receipt.update({"status": "PROCESS_EXITED", "exit_code": code,
                    "ended_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if (run_dir / "budget.json").exists():
        end_budget = R.read_json(run_dir / "budget.json")
        receipt["real_data_fits"] = end_budget.get("fits_total")
        receipt["course_llm_attempts"] = end_budget.get("llm_total")
    R.atomic_write_json(receipt_path, receipt)
raise SystemExit(code)
