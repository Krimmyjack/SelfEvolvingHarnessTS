"""T1: A3 single-arm run-to-run variance calibration (post-c166b63).

Thin driver around ``run_g1_pipeline``.  Does not modify the runner.  Writes
only to ``.calib_a3_state/runXX`` and ``artifacts/functional/e2/calib_a3_runXX.json``.
Never touches the mainline ``.g1_pipeline_state`` or
``g1_agentic_pipeline_report.json``.

Both paired arms use ``warm_arm_snapshot=None`` (identical A3 configuration).
Each run therefore yields two A3-config trajectories.  Relay routing is
recorded via ``BudgetedAgentBackend.returned_models``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    _default_backend_factory,
    run_g1_pipeline,
)


TASK_COUNT = 9
COHORT = "electricity"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "functional" / "e2"


class _CollectingFactory:
    """Wrap the stock backend factory and keep every live backend."""

    def __init__(self) -> None:
        self.backends: list[Any] = []

    def __call__(self, maximum_calls: int) -> Any:
        backend = _default_backend_factory(maximum_calls)
        self.backends.append(backend)
        return backend

    def returned_models(self) -> list[str]:
        models: set[str] = set()
        for backend in self.backends:
            models.update(getattr(backend, "returned_models", ()) or ())
        return sorted(models)


def _models_from_report(report: dict[str, Any]) -> list[str]:
    models: set[str] = set()
    for row in report.get("rows") or []:
        for arm in ("A3", "A5"):
            arm_row = row.get(arm) or {}
            llm = ((arm_row.get("cost") or {}).get("llm") or {})
            for name in llm.get("returned_models") or ():
                if isinstance(name, str) and name:
                    models.add(name)
    return sorted(models)


def _matched_flags(report: dict[str, Any]) -> list[bool]:
    flags: list[bool] = []
    for row in report.get("rows") or []:
        retrieval = row.get("source_prior_retrieval") or {}
        flags.append(bool(retrieval.get("matched")))
    return flags


def run_one(index: int) -> dict[str, Any]:
    if index < 1:
        raise ValueError("run index is 1-based")
    label = f"run{index:02d}"
    state_rel = f".calib_a3_state/{label}"
    report_path = ARTIFACT_DIR / f"calib_a3_{label}.json"
    factory = _CollectingFactory()
    started = time.perf_counter()
    print(
        f"CALIB_A3_START {label} cohort={COHORT} tasks={TASK_COUNT} "
        f"state_rel={state_rel} report={report_path}",
        flush=True,
    )
    result = run_g1_pipeline(
        cohort_name=COHORT,
        task_count=TASK_COUNT,
        paired_arms=True,
        run_slow=False,
        warm_arm_snapshot=None,
        state_rel=state_rel,
        report_path=report_path,
        write_report=True,
        backend_factory=factory,
    )
    models = sorted(set(factory.returned_models()) | set(_models_from_report(result)))
    matched = _matched_flags(result)
    n_matched = sum(1 for flag in matched if flag)
    print(
        "CALIB_A3_DONE %s wall=%.1fs returned_models=%s matched=%d/%d "
        "verdict=%s"
        % (
            label,
            time.perf_counter() - started,
            models,
            n_matched,
            len(matched),
            result.get("verdict"),
        ),
        flush=True,
    )
    if n_matched:
        print(
            "CALIB_A3_WARN %s %d/%d source_prior_retrieval.matched=true; "
            "those WARM (A5-label) rows are not A3-config and must be dropped "
            "from the variance sample. Use COLD (A3) only for those tasks."
            % (label, n_matched, len(matched)),
            flush=True,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        help="run a single 1-based index (smoke: --only 1)",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="inclusive count from --start when --only is omitted (default 3)",
    )
    args = parser.parse_args(argv)
    if args.only is not None:
        indices = [args.only]
    else:
        indices = list(range(args.start, args.start + args.n))
    for index in indices:
        run_one(index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
