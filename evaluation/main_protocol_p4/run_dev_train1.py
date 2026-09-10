"""DEV-TRAIN-1 unique CLI entry.  One logical package."""
from __future__ import annotations

import argparse
import sys

from evaluation.main_protocol_p4 import dev_train1_runner as runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_dev_train1")
    # Only the two phases this package actually implements are offered.
    # "formation" / "boundary" / "review" / "score" were advertised but never
    # existed as separate entry points: the course runs as one bounded
    # package.  Naming them here and then ignoring --phase under --live is how
    # `--phase preflight --live` could launch the whole course.
    parser.add_argument(
        "--phase", choices=("preflight", "run"), default="run",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--concurrency", type=int, default=runner.CONCURRENCY)
    parser.add_argument("--maximum-modified-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.phase == "preflight":
        # Checked before --live and before any data or transport is touched.
        if args.live:
            sys.stderr.write(
                "--phase preflight never opens data or transport; drop --live "
                "or use --phase run.\n"
            )
            return 2
        machinery = runner.load_machinery()
        snapshot = runner.load_h0_snapshot(machinery)
        spec = runner.RunSpec(
            train_uids=[], eval_uids=[], concurrency=int(args.concurrency),
            maximum_modified_fraction=args.maximum_modified_fraction,
        )
        run_dir = runner.RUNS_ROOT / args.run_id
        runner.validate_run_cap(spec, run_dir)
        snapshot = runner.configure_snapshot_for_spec(spec, snapshot, machinery)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "preflight.json"
        runner.atomic_write_json(path, runner.preflight_record(spec, snapshot))
        sys.stdout.write("%s\n" % path)
        return 0

    if args.live:
        bundle = runner.load_kdd_missing_development_bundle()
        spec = runner.RunSpec(
            train_uids=list(bundle.train_uids),
            eval_uids=list(bundle.eval_uids),
            concurrency=int(args.concurrency),
            live_api=True,
            requested_model=runner.REQUESTED_MODEL,
            base_url=runner.LOOPBACK_URL,
            maximum_modified_fraction=args.maximum_modified_fraction,
        )
        def factory() -> object:
            return runner.build_live_backend(
                maximum_calls=runner.PER_SEQUENCE_LLM_CAP,
            )
        out = runner.run_package(
            spec=spec, bundle=bundle, run_id=str(args.run_id),
            backend_factory=factory,
        )
        sys.stdout.write("%s\n" % (out.get("status"),))
        if out.get("detail"):
            sys.stderr.write("%s\n" % str(out.get("detail"))[:400])
        return int(out.get("exit_code") or 0)

    sys.stderr.write(
        "Non-live Fast/Slow execution is the fake-contract test:\n"
        "  python _scratch/dev_train1/test_runner_contract.py\n"
        "Live: python -m evaluation.main_protocol_p4.run_dev_train1 "
        "--phase run --run-id <id> --live\n"
        "Requested model %s, required returned %s, base %s. "
        "Course span [40, 80]. New run directory: %s\n"
        % (runner.REQUESTED_MODEL, runner.REQUIRED_RETURNED_MODEL,
           runner.LOOPBACK_URL, runner.RUNS_ROOT / args.run_id)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
