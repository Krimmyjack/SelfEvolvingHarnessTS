from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import run_cycles

from .contract_policy import (
    ContractPolicyBackend,
    DeterministicContractValuator,
    RecordingAgentBackend,
    load_replay_backend,
)


def build_offline_replay(path: Path) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m0-offline-fixture-") as temporary_name:
        temporary = Path(temporary_name)
        author = ContractPolicyBackend()
        recorder = RecordingAgentBackend(author)
        authored = run_cycles(
            cycles=2,
            run_root=temporary / "authored",
            backend=recorder,
            valuator=DeterministicContractValuator(),
        )
        provisional = temporary / "m0_offline_replay_v1.jsonl"
        recorder.write(provisional)
        replay = load_replay_backend(provisional)
        reproduced = run_cycles(
            cycles=2,
            run_root=temporary / "replayed",
            backend=replay,
            valuator=DeterministicContractValuator(),
        )
        if authored.normalized_behavior_shas != reproduced.normalized_behavior_shas:
            raise AssertionError("offline replay changed normalized Agent behavior")
        if authored.scientific_verdicts != reproduced.scientific_verdicts:
            raise AssertionError("offline replay changed scientific edit verdicts")
        if replay.call_count != len(recorder.rows):
            raise AssertionError("pure replay did not consume every immutable tape row")
        payload = provisional.read_bytes()
        handle, temporary_output_name = tempfile.mkstemp(
            prefix=f".{destination.name}-", dir=destination.parent
        )
        temporary_output = Path(temporary_output_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_output, destination)
        finally:
            if temporary_output.exists():
                temporary_output.unlink()
    return destination


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and self-verify the deterministic M0 offline response tape."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = build_offline_replay(args.out)
    print(f"offline_replay={path}")
    print(
        "tape_sha="
        + canonical_sha256(
            {"bytes_sha": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["build_offline_replay"]
