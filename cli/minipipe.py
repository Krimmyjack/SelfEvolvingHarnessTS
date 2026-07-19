from __future__ import annotations

import argparse
import os
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import parse_json_document
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
    AgentTransportError,
    AgictoChatCompletionsBackend,
    BudgetedAgentBackend,
)


_MAX_SAFE_TRANSPORT_RESUMES = 2


def _resolve_api_key(api_key_file: Path | None) -> str:
    """Resolve a relay credential without placing it in argv or artifacts."""

    if api_key_file is not None:
        try:
            value = api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"cannot read --api-key-file: {exc}") from exc
    else:
        value = os.environ.get("AGICTO_API_KEY", "").strip()
    if not value:
        raise SystemExit("AGICTO_API_KEY or --api-key-file is required")
    return value


def _safe_genesis_only_resume(run_dir: Path) -> bool:
    """Return true only before a cycle has emitted any scientific event."""

    lineage_path = run_dir / "harness_lineage.jsonl"
    context_path = run_dir / "private" / "run_context.json"
    if not lineage_path.is_file() or not context_path.is_file():
        return False
    cycle_root = run_dir / "cycles"
    if cycle_root.is_dir() and any(cycle_root.glob("cycle-*")):
        return False
    rows = [line for line in lineage_path.read_bytes().splitlines() if line]
    if len(rows) != 1:
        return False
    try:
        event = parse_json_document(rows[0])
    except (TypeError, ValueError, UnicodeError):
        return False
    return isinstance(event, dict) and event.get("event_kind") == "GENESIS"


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
        "--api-key-file",
        type=Path,
        help=(
            "Optional short-lived credential file for cross-runtime execution; "
            "its contents are never written to run artifacts."
        ),
    )
    run.add_argument(
        "--max-api-calls",
        type=int,
        help="Hard limit on uncached live relay calls for this invocation.",
    )
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
        if args.api_key_file is not None:
            raise SystemExit("--api-key-file is only valid with --backend agicto")
        backend = load_replay_backend(args.replay_file)
        valuator = DeterministicContractValuator()
    else:
        if args.replay_file is not None:
            raise SystemExit("--replay-file is only valid with --backend replay")
        api_key = _resolve_api_key(args.api_key_file)
        backend = AgictoChatCompletionsBackend(
            api_key=api_key,
            base_url=base_url,
        )
        if args.max_api_calls is not None:
            backend = BudgetedAgentBackend(
                backend,
                maximum_calls=args.max_api_calls,
            )
        valuator = FrozenChronosValuator(manifest_path=args.valuator_manifest)
    transport_resumes = 0
    resume = args.resume
    while True:
        try:
            result = run_cycles(
                cycles=args.cycles,
                run_root=run_dir,
                backend=backend,
                valuator=valuator,
                h0_root=args.h0_root,
                rules_path=args.rules,
                model=model,
                base_url=base_url,
                resume=resume,
            )
            break
        except AgentTransportError:
            if (
                transport_resumes >= _MAX_SAFE_TRANSPORT_RESUMES
                or not _safe_genesis_only_resume(run_dir)
            ):
                raise
            transport_resumes += 1
            resume = True
            print(
                "transient_transport_resume="
                f"{transport_resumes}/{_MAX_SAFE_TRANSPORT_RESUMES}"
            )
    for cycle in result.cycles:
        promoted = ",".join(cycle.promoted_edit_ids) or "none"
        print(
            f"{cycle.cycle_id}: start={cycle.starting_snapshot_sha} "
            f"end={cycle.ending_snapshot_sha} promoted={promoted}"
        )
    print(f"active_snapshot_sha={result.active_snapshot_sha}")
    print(f"run_context_sha={result.run_context.run_context_sha}")
    print(f"valuation_source={result.run_context.valuation_source}")
    print(f"ingestion_policy_id={result.run_context.ingestion_policy_id}")
    if isinstance(backend, BudgetedAgentBackend):
        print(f"paid_agent_calls={backend.calls}")
        print(f"prompt_tokens={backend.prompt_tokens}")
        print(f"completion_tokens={backend.completion_tokens}")
    print(f"transport_resumes={transport_resumes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
