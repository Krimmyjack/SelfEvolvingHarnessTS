from __future__ import annotations

import argparse
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.cycle import run_cycles
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    AgentResponse,
    ReplayAgentBackend,
)
from SelfEvolvingHarnessTS.runtime.llm_cache import _response_from_payload


def load_cached_responses(cache_root: Path) -> dict[str, AgentResponse]:
    """Load an immutable live cache as semantic-hash replay evidence.

    This intentionally reuses only requests whose complete effective request is
    unchanged. A newly routed slow-path request therefore becomes a typed tape
    miss instead of silently borrowing an unrelated response.
    """

    cache_root = Path(cache_root).resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root does not exist: {cache_root}")
    responses: dict[str, AgentResponse] = {}
    response_hashes: dict[str, str] = {}
    for path in sorted(cache_root.glob("*.json"), key=lambda item: item.name):
        record = parse_json_document(path.read_bytes())
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != "agent-cache-record/1"
        ):
            raise ValueError(f"invalid Agent cache record: {path.name}")
        key = record.get("key")
        payload = record.get("response")
        if not isinstance(key, dict) or not isinstance(payload, dict):
            raise ValueError(f"incomplete Agent cache record: {path.name}")
        semantic_hash = key.get("semantic_request_hash")
        expected_response_hash = record.get("response_hash")
        if not isinstance(semantic_hash, str) or not isinstance(
            expected_response_hash, str
        ):
            raise ValueError(f"invalid cache identity: {path.name}")
        if canonical_sha256(payload) != expected_response_hash:
            raise ValueError(f"cache response hash mismatch: {path.name}")
        if (
            semantic_hash in response_hashes
            and response_hashes[semantic_hash] != expected_response_hash
        ):
            raise ValueError("one semantic request maps to conflicting responses")
        responses[semantic_hash] = _response_from_payload(payload)
        response_hashes[semantic_hash] = expected_response_hash
    if not responses:
        raise ValueError("source cache contains no responses")
    return responses


def _parser() -> argparse.ArgumentParser:
    package_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Zero-network attribution audit using a prior live Agent cache"
    )
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_AGENT_BASE_URL)
    parser.add_argument(
        "--h0-root",
        type=Path,
        default=package_root / "methods" / "ttha" / "harness" / "h0",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=package_root / "evaluation" / "minipipe" / "config" / "m0_rules.json",
    )
    parser.add_argument(
        "--valuator-manifest",
        type=Path,
        default=(
            package_root
            / "evaluation"
            / "minipipe"
            / "valuation"
            / "model_manifest.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backend = ReplayAgentBackend(load_cached_responses(args.source_cache))
    result = run_cycles(
        cycles=1,
        run_root=args.run_dir,
        backend=backend,
        valuator=FrozenChronosValuator(manifest_path=args.valuator_manifest),
        h0_root=args.h0_root,
        rules_path=args.rules,
        model=args.model,
        base_url=args.base_url,
    )
    cycle = result.cycles[0]
    print(f"cycle={cycle.cycle_id}")
    print(f"active_snapshot_sha={result.active_snapshot_sha}")
    print(f"source_cache_response_count={len(load_cached_responses(args.source_cache))}")
    print(f"replayed_call_count={backend.call_count}")
    print(f"verdicts={','.join(result.scientific_verdicts) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_cached_responses", "main"]
