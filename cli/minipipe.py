from __future__ import annotations

import argparse
import os
from pathlib import Path

from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import run_cycles
from SelfEvolvingHarnessTS.evaluation.minipipe.fixtures.contract_policy import (
    DeterministicContractValuator,
    load_replay_backend,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    AgictoChatCompletionsBackend,
)


def _parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Agent-centric TTHA M0 mini-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one or more complete M0 cycles")
    run.add_argument("--cycles", type=int, default=2)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--backend", choices=("agicto", "replay"), default="agicto")
    run.add_argument("--replay-file", type=Path)
    run.add_argument("--model")
    run.add_argument("--base-url")
    run.add_argument(
        "--h0-root",
        type=Path,
        default=package_root / "methods" / "ttha" / "harness" / "h0",
    )
    run.add_argument(
        "--rules",
        type=Path,
        default=package_root / "evaluation" / "minipipe" / "config" / "m0_rules.json",
    )
    run.add_argument(
        "--valuator-manifest",
        type=Path,
        default=package_root / "evaluation" / "minipipe" / "valuation" / "model_manifest.json",
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--overwrite-empty-run-dir", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> int:
    model = args.model or os.environ.get("M0_AGENT_MODEL", DEFAULT_AGENT_MODEL)
    base_url = args.base_url or os.environ.get(
        "M0_AGENT_BASE_URL", DEFAULT_AGENT_BASE_URL
    )
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise SystemExit("run directory is non-empty; refusing to overwrite scientific artifacts")
    if args.backend == "replay":
        if args.replay_file is None:
            raise SystemExit("--backend replay requires --replay-file")
        backend = load_replay_backend(args.replay_file)
        valuator = DeterministicContractValuator()
    else:
        if args.replay_file is not None:
            raise SystemExit("--replay-file is only valid with --backend replay")
        api_key = os.environ.get("AGICTO_API_KEY")
        if not api_key:
            raise SystemExit("AGICTO_API_KEY is required for --backend agicto")
        backend = AgictoChatCompletionsBackend(
            api_key=api_key,
            base_url=base_url,
        )
        valuator = FrozenChronosValuator(manifest_path=args.valuator_manifest)
    result = run_cycles(
        cycles=args.cycles,
        run_root=run_dir,
        backend=backend,
        valuator=valuator,
        h0_root=args.h0_root,
        rules_path=args.rules,
        model=model,
        base_url=base_url,
        resume=args.resume,
    )
    for cycle in result.cycles:
        promoted = ",".join(cycle.promoted_edit_ids) or "none"
        print(
            f"{cycle.cycle_id}: start={cycle.starting_snapshot_sha} "
            f"end={cycle.ending_snapshot_sha} promoted={promoted}"
        )
    print(f"active_snapshot_sha={result.active_snapshot_sha}")
    print(f"run_context_sha={result.run_context.run_context_sha}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
