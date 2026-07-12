"""run_p5_safety.py — P5-C safety 收口（prereg §5.0 冻结：confirmatory slice 一次性评估）。

评估对象 = P4 晋升产物 bundle_v0.e1（全部规则/阈值 dev 冻结）vs v0，confirmatory slice
（seeds 40–59，从未参与任何调参）上 true 判官四联：coverage / gain / harm rate /
worst-group LCB。anomaly 面必须零扰动（规则任务作用域核验）。S2 记录线的 sealed holdout
访问**显式延期**为独立注册访问（非静默跳过）。

命令：D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p5_safety
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .policy.action_spec import action_menu_v1
from .policy.edits import AddRiskRule, PolicyBundle, apply_edits, bundle_v0
from .policy.risk_policy import RiskRule
from .run_p5_identity_gate import CONFIRMATORY_SEEDS, _confirmatory_rows
from .slow_path.bundle_store import BundleStore
from .slow_path.true_judge_validator import SubstrateRouterPolicy, evaluate_bundle

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P5Safety"
_P4_BUNDLES = Path(__file__).resolve().parent / "results" / "Stage2" / "P4Promotion" / "bundles"

# P4 晋升规则的冻结副本（重建路径；若 P4 production artifact 在，则以其为准并核验一致）
_P4_RULE = RiskRule(
    rule_id="mined_ban_f0_median_w25_snr_low",
    when={"cell": {"snr": "low"}, "base_action_in": ["f0_median_w25"]},
    then={"op": "ban", "action": "f0_median_w9"},
    scope="region:cell_snr=low",
    provenance={"source": "proposer:enum", "grid_rule": "mined_ban_f0_median_w25_snr_low"},
)


def _promoted_bundle() -> Tuple[PolicyBundle, str]:
    if (_P4_BUNDLES / "chain.json").exists():
        head = BundleStore(_P4_BUNDLES).head()
        if any(r.rule_id == _P4_RULE.rule_id for r in head.risk.rules):
            return head, "loaded_from_P4_production_artifact"
    rebuilt, log = apply_edits(bundle_v0(), [AddRiskRule(_P4_RULE)])
    assert log[0]["applied"]
    return rebuilt, "rebuilt_from_frozen_rule_copy"


def _grouped_lcb(values_by_group: Dict[int, List[float]], rng: np.random.Generator, b: int) -> float:
    groups = sorted(values_by_group)
    means = np.empty(b)
    for i in range(b):
        picked = rng.choice(groups, size=len(groups), replace=True)
        pooled = [v for g in picked for v in values_by_group[g]]
        means[i] = float(np.mean(pooled))
    return float(np.percentile(means, 5.0))


def run_p5_safety(seeds: Sequence[int] = CONFIRMATORY_SEEDS, n_per_seed: int = 3,
                  out_dir: Path | str | None = None, bootstrap_b: int = 2000) -> Dict[str, Any]:
    rows = _confirmatory_rows(seeds, n_per_seed)
    menu = action_menu_v1()
    router = SubstrateRouterPolicy()
    v0 = bundle_v0()
    v1, v1_source = _promoted_bundle()
    rng = np.random.default_rng(20260710 + 7)

    report: Dict[str, Any] = {
        "phase": "P5C_safety_closeout",
        "protocol": "thresholds/rules frozen on dev; one-shot confirmatory evaluation (seeds 40-59)",
        "one_shot": True,
        "n_rows": len(rows),
        "promoted_bundle": {"version": v1.version, "sha": v1.sha(), "source": v1_source},
        "s2_sealed_holdout": "deferred_separate_registered_access",
    }
    for task in ("forecast", "anomaly_detection"):
        base = evaluate_bundle(v0, rows, task, menu, router)
        cand = evaluate_bundle(v1, rows, task, menu, router)
        gains = np.asarray([c["true_delta"] - b["true_delta"] for b, c in zip(base, cand)])
        fired = [bool(c["fired_rules"]) for c in cand]
        identical = all(b["action_id"] == c["action_id"] and b["artifact_sha"] == c["artifact_sha"]
                        for b, c in zip(base, cand))
        if task == "forecast":
            cell_groups: Dict[str, Dict[int, List[float]]] = {}
            for row, g in zip(rows, gains):
                cell_groups.setdefault(str(row["cell"]), {}).setdefault(
                    int(row["group_seed"]), []).append(float(g))
            worst_cell_lcb = {cell: _grouped_lcb(groups, rng, bootstrap_b)
                              for cell, groups in sorted(cell_groups.items())}
            report["forecast"] = {
                "coverage_fired": float(np.mean(fired)),
                "gain_vs_v0_mean": float(gains.mean()),
                "harm_rate_vs_v0": float(np.mean(gains < -1e-9)),
                "worst_cell_lcb": worst_cell_lcb,
                "min_worst_cell_lcb": float(min(worst_cell_lcb.values())),
            }
        else:
            report["anomaly_detection"] = {
                "rows_identical_to_v0": bool(identical),
                "coverage_fired": float(np.mean(fired)),
            }

    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "generated_by": "run_p5_safety",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §5.0（P5-C 冻结）",
        "seeds": [int(s) for s in seeds], "n_per_seed": int(n_per_seed),
        "judges": "true judges only", "bootstrap_b": int(bootstrap_b),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()
    report = run_p5_safety(out_dir=args.out_dir)
    print(json.dumps({"forecast": report["forecast"],
                      "anomaly_identical": report["anomaly_detection"]["rows_identical_to_v0"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
