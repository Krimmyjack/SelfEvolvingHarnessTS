"""Command-line entry point for the benchmark-v0 phase state machine."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .benchmark.runner import default_cli_handlers


def _paths(parser: argparse.ArgumentParser, *, out: bool = True) -> None:
    parser.add_argument("--root", required=True)
    if out:
        parser.add_argument("--out", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m SelfEvolvingHarnessTS.run_benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire")
    _paths(acquire, out=False)
    acquisition_mode = acquire.add_mutually_exclusive_group()
    acquisition_mode.add_argument("--automatic", action="store_true")
    acquisition_mode.add_argument("--manual-status", action="store_true")

    probe = subparsers.add_parser("probe")
    _paths(probe)

    freeze = subparsers.add_parser("freeze")
    _paths(freeze)

    dry_run = subparsers.add_parser("dry-run")
    _paths(dry_run)
    dry_run.add_argument("--method", required=True)

    confirm = subparsers.add_parser("confirm")
    _paths(confirm)
    confirm.add_argument("--method", required=True)

    dev = subparsers.add_parser("dev-eval")
    _paths(dev)
    dev.add_argument("--split", default="dev_query", choices=("dev_query",))
    dev.add_argument("--baselines", required=True)

    campaign = subparsers.add_parser("campaign-freeze")
    _paths(campaign)
    campaign.add_argument("--campaign-id", required=True)

    final = subparsers.add_parser("final-eval")
    _paths(final)
    final.add_argument("--campaign-id", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    handlers: Mapping[str, Callable[[Any], int]] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    selected = default_cli_handlers() if handlers is None else dict(handlers)
    try:
        handler = selected[args.command]
    except KeyError as exc:
        raise RuntimeError(f"no handler configured for CLI phase {args.command!r}") from exc
    result = handler(args)
    return int(result or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
