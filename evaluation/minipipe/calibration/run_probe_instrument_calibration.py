from __future__ import annotations

import argparse
import json
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.probes.panel import (
    M0_PROBE_SPECS,
    PROBE_INSTRUMENT_EPOCH,
    _apply_probe,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RULES = _ROOT / "evaluation/minipipe/config/m0_rules.json"


def calibrate_probe_instrument(rules_path: Path = _DEFAULT_RULES) -> dict[str, object]:
    rules = load_m0_rules(rules_path)
    corpus = build_core_corpus(rules)
    valuator = FrozenChronosValuator()
    required = {
        "missing": "imputation",
        "level_shift": "level_correction",
    }
    witnesses: list[dict[str, object]] = []
    threshold = float(rules["probe_gain_min"])
    for family, probe_name in required.items():
        witness: dict[str, object] | None = None
        for case in corpus.targets:
            if case.private_family != family:
                continue
            baseline = valuator.evaluate(
                case.corrupt_context,
                case.clean_future,
                scale_context=case.clean_context,
            )
            transformed, modified_fraction = _apply_probe(
                case.corrupt_context,
                probe_name,
                M0_PROBE_SPECS[probe_name].betas[-1],
            )
            arm = valuator.evaluate(
                transformed,
                case.clean_future,
                scale_context=case.clean_context,
            )
            response = arm.utility_u - baseline.utility_u
            if (
                arm.filled_context_sha != baseline.filled_context_sha
                and abs(response) >= threshold
            ):
                witness = {
                    "family": family,
                    "probe_id": probe_name,
                    "case_id": case.case_id,
                    "beta": M0_PROBE_SPECS[probe_name].betas[-1],
                    "modified_fraction": modified_fraction,
                    "response_r_private": response,
                    "identity_model_input_sha": baseline.filled_context_sha,
                    "probe_model_input_sha": arm.filled_context_sha,
                    "spec_sha": M0_PROBE_SPECS[probe_name].implementation_sha,
                }
                break
        if witness is None:
            raise AssertionError(
                f"probe-instrument/2 has no sensitive frozen-Chronos witness for {family}"
            )
        witnesses.append(witness)
    payload: dict[str, object] = {
        "schema_version": "probe-instrument-calibration/1",
        "instrument_epoch": PROBE_INSTRUMENT_EPOCH,
        "probe_specs_sha": canonical_sha256(
            {name: spec.implementation_sha for name, spec in M0_PROBE_SPECS.items()}
        ),
        "valuator_manifest_sha": valuator.model_manifest_sha,
        "ingestion_policy_id": valuator.ingestion_policy_id,
        "gain_threshold": threshold,
        "witnesses": witnesses,
        "status": "PASS",
    }
    payload["receipt_sha"] = canonical_sha256(payload)
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate probe-instrument/2 against pinned local Chronos."
    )
    parser.add_argument("--rules", type=Path, default=_DEFAULT_RULES)
    args = parser.parse_args()
    print(json.dumps(calibrate_probe_instrument(args.rules), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["calibrate_probe_instrument"]
