"""One-off parallel worker for E1-v2 preflight public Context computation.

Uses the exact frozen ``build_task_public_context`` helper; no outcome is read.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (str(ROOT), str(ROOT / "evaluation" / "functional"), str(ROOT / "methods" / "ttha")):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.functional.task_episode_harness.e1 import (
    _frozen_task_roster,
    _load_kdd_roster,
    build_task_public_context,
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: e1v2_context_worker.py TASK_INDEX OUTPUT_REL")
    task_index = int(sys.argv[1])
    output_rel = sys.argv[2]
    spec = _frozen_task_roster()[task_index]
    started = time.perf_counter()
    roster, values, _selected = _load_kdd_roster(
        ROOT, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
    )
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]
    context = build_task_public_context(
        values,
        train_uids,
        observation_cutoff=int(spec["support_origins"][0]),
    )
    output_path = ROOT / output_rel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_episode_id": spec["task_episode_id"],
        "task_index": task_index,
        "observation_cutoff": int(spec["support_origins"][0]),
        "public_context": context,
        "worker_seconds": round(time.perf_counter() - started, 2),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"WROTE {spec['task_episode_id']} "
        f"cutoff={spec['support_origins'][0]} "
        f"sig={dict(context['task_signature'])} "
        f"seconds={round(time.perf_counter() - started, 2)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
