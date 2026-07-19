from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.contracts import CasePurpose
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
from SelfEvolvingHarnessTS.runtime.public_features import (
    PublicFeatureExtraction,
    extract_public_features,
)


_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_PATH = Path(__file__).with_name("fixed_program_baseline_v1.json")
_RULES_PATH = _ROOT / "config" / "m0_rules.json"


def _load_object(path: Path) -> dict[str, object]:
    value = parse_json_document(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_program(
    program: Sequence[Sequence[object]],
    *,
    features: PublicFeatureExtraction,
) -> list[tuple[str, dict[str, object]]]:
    steps: list[tuple[str, dict[str, object]]] = []
    for entry in program:
        if (
            len(entry) != 2
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], Mapping)
        ):
            raise ValueError("invalid fixed baseline program entry")
        params = dict(entry[1])
        if params.pop("period_from", None) == "pre_period":
            params["period"] = int(features.pre_period)
        for source_key in tuple(params):
            if not source_key.endswith("_from"):
                continue
            target_key = source_key[: -len("_from")]
            feature_name = params.pop(source_key)
            if not isinstance(feature_name, str) or feature_name not in features.mapping:
                raise ValueError("unknown public feature parameter binding")
            params[target_key] = features.mapping[feature_name]
        steps.append((entry[0], params))
    return steps


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def run_fixed_program_baseline(
    valuator: FrozenChronosValuator,
    *,
    rules_path: Path = _RULES_PATH,
    baseline_path: Path = _BASELINE_PATH,
) -> dict[str, object]:
    rules = load_m0_rules(rules_path)
    corpus = build_core_corpus(rules)
    baseline = _load_object(baseline_path)
    catalog = baseline.get("observable_witnesses")
    if not isinstance(catalog, dict):
        raise ValueError("fixed baseline has no observable witness catalog")
    case_rows: list[dict[str, object]] = []
    family_damage: dict[str, list[float]] = defaultdict(list)
    family_best_gain: dict[str, list[float]] = defaultdict(list)

    for case in corpus.all_cases:
        corrupt_receipt = valuator.evaluate(
            case.corrupt_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        clean_receipt = valuator.evaluate(
            case.clean_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        features = extract_public_features(case.corrupt_context)
        raw_programs = catalog.get(case.private_family, [])
        if not isinstance(raw_programs, list):
            raise ValueError("fixed baseline family catalog must be a list")
        programs: list[dict[str, object]] = []
        for raw_program in raw_programs:
            if not isinstance(raw_program, list):
                raise ValueError("fixed baseline program must be a list")
            steps = _resolve_program(raw_program, features=features)
            execution = run_pipeline(
                steps,
                case.corrupt_context,
                source="fixed_program_baseline_private",
            )
            if not execution.ok or execution.artifact is None:
                programs.append(
                    {
                        "program_sha": canonical_sha256(raw_program),
                        "steps": raw_program,
                        "status": "EXECUTION_FAILED",
                        "effect_distinct": False,
                        "utility_u": None,
                        "gain_g": None,
                    }
                )
                continue
            receipt = valuator.evaluate(
                execution.artifact,
                case.clean_future,
                scale_context=case.clean_context,
            )
            distinct = not np.array_equal(
                np.asarray(execution.artifact, dtype=np.float64),
                np.asarray(case.corrupt_context, dtype=np.float64),
                equal_nan=True,
            )
            programs.append(
                {
                    "program_sha": canonical_sha256(raw_program),
                    "steps": raw_program,
                    "status": "OK",
                    "effect_distinct": distinct,
                    "utility_u": receipt.utility_u,
                    "gain_g": receipt.utility_u - corrupt_receipt.utility_u,
                }
            )
        eligible = [
            row
            for row in programs
            if row["status"] == "OK"
            and row["effect_distinct"] is True
            and isinstance(row["gain_g"], (int, float))
        ]
        best = max(
            eligible,
            key=lambda row: (float(row["gain_g"]), str(row["program_sha"])),
            default=None,
        )
        damage = clean_receipt.utility_u - corrupt_receipt.utility_u
        best_gain = float(best["gain_g"]) if best is not None else 0.0
        if case.purpose is CasePurpose.TARGET:
            family_damage[case.private_family].append(damage)
            family_best_gain[case.private_family].append(best_gain)
        case_rows.append(
            {
                "case_id": case.case_id,
                "purpose": case.purpose.value,
                "private_family": case.private_family,
                "private_severity": case.private_severity,
                "identity_utility_u": corrupt_receipt.utility_u,
                "clean_utility_u": clean_receipt.utility_u,
                "damage_d": damage,
                "best_program_sha": None if best is None else best["program_sha"],
                "best_gain_g": best_gain,
                "recoverable_at_m0_thresholds": (
                    case.purpose is CasePurpose.TARGET
                    and damage >= float(rules["critic_damage_min"])
                    and best_gain >= float(rules["candidate_gain_min"])
                ),
                "programs": programs,
            }
        )

    family_summary = {
        family: {
            "target_count": len(family_damage[family]),
            "positive_damage_count": sum(
                value >= float(rules["critic_damage_min"])
                for value in family_damage[family]
            ),
            "recoverable_count": sum(
                damage >= float(rules["critic_damage_min"])
                and gain >= float(rules["candidate_gain_min"])
                for damage, gain in zip(
                    family_damage[family], family_best_gain[family]
                )
            ),
            "median_damage_d": _median(family_damage[family]),
            "median_best_gain_g": _median(family_best_gain[family]),
            "total_damage_d": float(sum(family_damage[family])),
            "total_best_gain_g": float(sum(family_best_gain[family])),
        }
        for family in sorted(family_damage)
    }
    payload = {
        "schema_version": "fixed-program-baseline-report/1",
        "visibility": "grader_private_only",
        "interpretation": (
            "Private family-routed expressibility ceiling; not a deployment selector."
        ),
        "rules_sha": rules.rules_sha,
        "corpus_version": rules["corpus"]["corpus_version"],
        "baseline_spec_sha": canonical_sha256(baseline),
        "valuator_manifest_sha": valuator.model_manifest_sha,
        "valuation_source": valuator.valuation_source,
        "ingestion_policy_id": valuator.ingestion_policy_id,
        "family_summary": family_summary,
        "cases": case_rows,
    }
    return {**payload, "report_sha": canonical_sha256(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the zero-LLM private fixed-program M0 baseline."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_fixed_program_baseline(FrozenChronosValuator())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(report["report_sha"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_fixed_program_baseline"]
