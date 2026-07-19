from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_json_bytes,
    canonical_sha256,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline


_ROOT = Path(__file__).resolve().parents[3]


def run_calibration(
    *,
    output_path: Path,
    rules_path: Path,
    valuator_manifest: Path,
    doses: tuple[float, ...],
    global_z_min: float,
) -> dict[str, object]:
    rules = load_m0_rules(rules_path)
    corpus = build_core_corpus(rules)
    cases = tuple(
        case for case in corpus.all_cases if case.private_family == "impulsive_outlier"
    )
    valuator = FrozenChronosValuator(manifest_path=valuator_manifest)
    rows: list[dict[str, object]] = []
    for case in cases:
        baseline = valuator.evaluate(
            case.corrupt_context,
            case.clean_future,
            scale_context=case.clean_context,
        )
        for dose in doses:
            execution = run_pipeline(
                [
                    (
                        "hampel_filter",
                        {
                            "window": 7,
                            "n_sigmas": dose,
                            "global_z_min": global_z_min,
                        },
                    )
                ],
                case.corrupt_context,
                source="outlier-dose-calibration",
            )
            if not execution.ok or execution.artifact is None:
                raise RuntimeError(f"Hampel calibration failed for {case.case_id}")
            prepared = np.asarray(execution.artifact, dtype=np.float64)
            changed = ~np.isclose(
                prepared,
                case.corrupt_context,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            receipt = valuator.evaluate(
                prepared,
                case.clean_future,
                scale_context=case.clean_context,
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "case_purpose": case.purpose.value,
                    "n_sigmas": dose,
                    "modified_fraction": float(np.mean(changed)),
                    "repair_gain_g": receipt.utility_u - baseline.utility_u,
                }
            )
    summaries: list[dict[str, object]] = []
    for dose in doses:
        target = [
            row
            for row in rows
            if row["n_sigmas"] == dose and row["case_purpose"] == "target"
        ]
        risks = [
            row
            for row in rows
            if row["n_sigmas"] == dose and row["case_purpose"] != "target"
        ]
        summaries.append(
            {
                "n_sigmas": dose,
                "target_positive_gain_count": sum(
                    float(row["repair_gain_g"]) > 0.0 for row in target
                ),
                "target_mean_gain_g": float(
                    np.mean([float(row["repair_gain_g"]) for row in target])
                ),
                "target_max_modified_fraction": max(
                    float(row["modified_fraction"]) for row in target
                ),
                "risk_mean_gain_if_erroneously_applied": float(
                    np.mean([float(row["repair_gain_g"]) for row in risks])
                ),
                "risk_max_modified_fraction_if_erroneously_applied": max(
                    float(row["modified_fraction"]) for row in risks
                ),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "outlier-hampel-dose-calibration/2",
        "scientific_role": "offline_operator_dose_calibration_not_harness_growth",
        "rules_sha": rules.rules_sha,
        "valuation_source": valuator.valuation_source,
        "ingestion_policy_id": valuator.ingestion_policy_id,
        "doses": list(doses),
        "global_z_min": global_z_min,
        "summaries": summaries,
        "case_rows": rows,
    }
    payload["calibration_sha"] = canonical_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate bounded Hampel doses with the frozen Chronos judge."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rules",
        type=Path,
        default=_ROOT / "evaluation/minipipe/config/m0_rules.json",
    )
    parser.add_argument(
        "--valuator-manifest",
        type=Path,
        default=_ROOT / "evaluation/minipipe/valuation/model_manifest.json",
    )
    parser.add_argument(
        "--doses",
        type=float,
        nargs="+",
        default=(5.0, 8.0, 10.0, 12.0),
    )
    parser.add_argument("--global-z-min", type=float, default=4.0)
    args = parser.parse_args()
    payload = run_calibration(
        output_path=args.output,
        rules_path=args.rules,
        valuator_manifest=args.valuator_manifest,
        doses=tuple(args.doses),
        global_z_min=float(args.global_z_min),
    )
    for row in payload["summaries"]:
        print(row)
    print(f"calibration_sha={payload['calibration_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_calibration"]
