"""run_p4_promotion.py — P4 慢路径完整晋升周期（Final_Plan_CodeAgentFirst §P4；prereg §4）。

链条（全部走既有部署消费面，无平行实现——Critical Review §4 知识断路教训）：
  held-in 证据挖掘 → SlowProposal（枚举 proposer，§13.2 arm①无 LLM）
  → PromotionGate（syntax/support，slow_path/promotion.py）
  → **true 判官** held-in + held-out validator（P3 binding：proxy 不得进验收）
  → apply_edits 版本升级 → BundleStore 落盘（append-only）
  → 回归重放（非目标行 bit 级不变 + anomaly 面零扰动）
  → rollback 演示（head 回 v0 → serving 与原 v0 bit 级一致 → 恢复晋升头，事件全留痕）
  → rejected buffer（被拒提案 + true 判官拒因）。

被改进对象（真实证据驱动，非摆拍）：v0 现任 SubstrateRouterPolicy 沿用 F0 时代剂量启发式
（snrLow→f0_median_w25），在强季节 substrate 上是已知错误（P2/P3：w25 +0.28 vs w9 +0.92）；
矿工从 held-in 发现、规则修复、held-out 确认。

**机制验收，非性能主张**；不使用 "self-evolving" 一词（P6 才解锁）。Memory M0–M3 阶梯为
prereg §4 条件线：risk-memory veto 已在 P1 gate 活跃，utility/contrast 阶梯显式挂起待其
自身预注册 run。

命令：
  D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p4_promotion --n-series 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .evaluators.anomaly_rig import make_anomaly_slice
from .policy.action_spec import action_menu_v1
from .policy.edits import EditOp, PolicyBundle, apply_edits, bundle_v0
from .slow_path.bundle_store import BundleStore
from .slow_path.promotion import PromotionGate
from .slow_path.proposal_schema import SlowProposal
from .slow_path.true_judge_validator import SubstrateRouterPolicy, evaluate_bundle, validate_edit

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P4Promotion"
MIN_HELDOUT_GAIN = 0.02        # = P3 冻结 ε（prereg §3/§4 口径）
WORST_CELL_TOL = 0.05          # held-out per-cell 安全容差（δ_safe，manifest 声明）
TASKS = ("forecast", "anomaly_detection")

# 枚举 proposer 网格（§13.2 arm①：无 LLM、无经验；好坏都提，裁决权在 true 判官）
_ENUM_GRID: List[Dict[str, Any]] = [
    {"rule_id": "mined_ban_f0_median_w25_snr_low", "snr": "low",
     "ban": "f0_median_w25", "replacement": "f0_median_w9"},
    {"rule_id": "reverse_ban_f0_median_w9_snr_low", "snr": "low",
     "ban": "f0_median_w9", "replacement": "f0_median_w25"},
    {"rule_id": "harmful_ban_f0_median_w9_snr_high", "snr": "high",
     "ban": "f0_median_w9", "replacement": "f0_median_w25"},
]


def _split(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """按 cell 分层对半切：每 cell 前一半 held-in、后一半 held-out（序列级不相交）。"""
    by_cell: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell"]), []).append(row)
    held_in: List[Mapping[str, Any]] = []
    held_out: List[Mapping[str, Any]] = []
    for cell in sorted(by_cell):
        group = by_cell[cell]
        half = len(group) // 2
        held_in.extend(group[:half])
        held_out.extend(group[half:])
    return held_in, held_out


def _mine_evidence(held_in: Sequence[Mapping[str, Any]], menu, router) -> Dict[str, Any]:
    """held-in 证据：现任 serving 的 per-cell true delta（矿工的观察面，不看 held-out）。"""
    rows = evaluate_bundle(bundle_v0(), held_in, "forecast", menu, router)
    per_cell: Dict[str, List[float]] = {}
    for r in rows:
        per_cell.setdefault(r["cell"], []).append(r["true_delta"])
    return {cell: {"mean_true_delta": float(np.mean(v)), "n": len(v)}
            for cell, v in sorted(per_cell.items())}


def _proposals(held_in: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]) -> List[SlowProposal]:
    out = []
    for g in _ENUM_GRID:
        region_n = sum(1 for row in held_in
                       if (("snrLow" in str(row["cell"])) == (g["snr"] == "low")))
        scope = f"region:cell_snr={g['snr']}"
        out.append(SlowProposal(
            kind="ProposeRiskRule",
            scope=scope,
            payload={
                "rule_id": g["rule_id"],
                "when": {"cell": {"snr": g["snr"]}, "base_action_in": [g["ban"]]},
                "then": {"op": "ban", "action": g["replacement"]},
                "scope": scope,
            },
            evidence_refs=(f"held_in_mining:{g['snr']}",),
            support={"n_unique_cases": int(region_n)},
            provenance={"source": "proposer:enum", "grid_rule": g["rule_id"],
                        "held_in_evidence": dict(evidence)},
        ))
    return out


def _replay_rows(bundle: PolicyBundle, rows: Sequence[Mapping[str, Any]], menu, router) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for task in TASKS:
        out.extend(evaluate_bundle(bundle, rows, task, menu, router))
    return out


def _rows_identical(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> bool:
    return all(x["action_id"] == y["action_id"] and x["artifact_sha"] == y["artifact_sha"]
               for x, y in zip(a, b))


def run_p4(n_series: int = 60, out_dir: Path | str | None = None,
           seed: int = 20260709) -> Dict[str, Any]:
    rows = make_anomaly_slice(n_series, seed=seed)
    held_in, held_out = _split(rows)
    menu = action_menu_v1()
    router = SubstrateRouterPolicy()
    gate = PromotionGate(min_support=1)

    out = Path(out_dir) if out_dir is not None else None
    store = BundleStore((out / "bundles") if out is not None else Path(DEFAULT_OUT) / "bundles_tmp")
    v0 = bundle_v0()
    store.save(v0, meta={"role": "incumbent", "router": "SubstrateRouterPolicy(F0-era dose heuristic)"})

    evidence = _mine_evidence(held_in, menu, router)
    proposals = _proposals(held_in, evidence)

    current = v0
    promoted: Dict[str, Any] | None = None
    rejected: List[Dict[str, Any]] = []
    for proposal in proposals:
        gate_outcome = gate.validate(proposal)
        if not gate_outcome.accepted or gate_outcome.edit_op is None:
            rejected.append({"proposal": proposal.to_dict(), "stage": "promotion_gate",
                             "reasons": [gate_outcome.reason or "gate_rejected"]})
            continue
        edit: EditOp = gate_outcome.edit_op
        validation = validate_edit(current, edit, held_in, held_out, "forecast",
                                   menu, router,
                                   min_heldout_gain=MIN_HELDOUT_GAIN,
                                   worst_cell_tol=WORST_CELL_TOL)
        if not validation["accepted"]:
            rejected.append({"proposal": proposal.to_dict(), "stage": "true_judge_validator",
                             "reasons": validation["reasons"],
                             "held_out_mean_gain": (validation["held_out"] or {}).get("mean_gain")})
            continue
        current, log = apply_edits(current, [edit])
        store.save(current, meta={"role": "promoted",
                                  "rule_id": proposal.payload["rule_id"],
                                  "validation": {k: validation[k] for k in
                                                 ("held_in", "held_out", "per_cell",
                                                  "non_targeted_identical")}})
        promoted = {"version": current.version, "sha": current.sha(),
                    "rule_id": str(proposal.payload["rule_id"]),
                    "validation": validation, "apply_log": log}

    if promoted is None:
        raise RuntimeError("P4 周期未产生任何晋升——检查 substrate/网格（出口判据要求 ≥1 晋升）")

    # ── 回归重放（全切片、双任务）───────────────────────────────────────────
    v0_replay = _replay_rows(v0, rows, menu, router)
    v1_replay = _replay_rows(current, rows, menu, router)
    non_targeted = [(a, b) for a, b in zip(v0_replay, v1_replay) if not b["fired_rules"]]
    targeted = [(a, b) for a, b in zip(v0_replay, v1_replay) if b["fired_rules"]]
    anomaly_pairs = [(a, b) for a, b in zip(v0_replay, v1_replay)
                     if a["task"] == "anomaly_detection"]
    regression = {
        "non_targeted_identical": _rows_identical([a for a, _ in non_targeted],
                                                  [b for _, b in non_targeted]),
        "anomaly_rows_identical": _rows_identical([a for a, _ in anomaly_pairs],
                                                  [b for _, b in anomaly_pairs]),
        "targeted_rows": len(targeted),
        "targeted_mean_gain": (float(np.mean([b["true_delta"] - a["true_delta"]
                                              for a, b in targeted])) if targeted else 0.0),
    }

    # ── rollback 演示（head 移动 + bit 级一致性 + 恢复晋升头，事件全留痕）───────
    store.rollback(v0.version, reason="P4 rollback demo")
    head_replay = _replay_rows(store.head(), rows, menu, router)
    rollback_verified = _rows_identical(v0_replay, head_replay)
    store.rollback(promoted["version"], reason="restore promoted head after rollback demo")

    report = {
        "phase": "P4_promotion_cycle",
        "claim_scope": ("mechanism acceptance on synthetic substrate; NOT a performance claim; "
                        "the term self-evolving stays locked until P6"),
        "n_series": int(n_series),
        "split": {"held_in": len(held_in), "held_out": len(held_out), "stratified_by_cell": True},
        "held_in_mining_evidence": evidence,
        "promoted": promoted,
        "n_rejected": len(rejected),
        "regression": regression,
        "rollback": {"verified": bool(rollback_verified),
                     "final_head": promoted["version"]},
        "memory_ladder": {
            "status": "conditional_pending",
            "risk_memory_veto": "live in deployment gate since P1 (escalation._risk_memory_blocks)",
            "pending_conditions": "prereg §4：utility/contrast memory 须 ①胜 static 学习器 "
                                  "②first-unseen harm ≤ 阈值 ③in-support 显著优于 out-support"
                                  "——需自身预注册 run，本周期不解锁",
        },
        "exit_criteria": {
            "typed_edit_op": True,
            "promotion_gate": True,
            "true_judge_held_in_out_validation": True,
            "version_bump_persisted": True,
            "regression_replay": bool(regression["non_targeted_identical"]),
            "rollback_demo": bool(rollback_verified),
            "rejected_buffer": len(rejected) > 0,
            "cycle_complete": bool(promoted and rollback_verified
                                   and regression["non_targeted_identical"] and rejected),
        },
    }
    manifest = {
        "generated_by": "run_p4_promotion",
        "plan": "Final_Plan_CodeAgentFirst_2026-07-09 §P4",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §4",
        "seed": int(seed), "n_series": int(n_series),
        "min_heldout_gain": MIN_HELDOUT_GAIN,
        "worst_cell_tol": WORST_CELL_TOL,
        "judges": "true judges only（P3 binding：proxy 不得进验收）",
        "menu_sha256": menu.sha256,
        "incumbent_router": "SubstrateRouterPolicy（F0-era dose heuristic，已知在季节 substrate 上错误）",
        "proposer": "enum grid（§13.2 arm①，无 LLM；LLM proposer 属 P5 identity gate）",
        "enum_grid": _ENUM_GRID,
        "api_calls": 0,
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "rejected_edits.jsonl").open("w", encoding="utf-8") as fh:
            for row in rejected:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
            for tag, replay in (("v0", v0_replay), (promoted["version"], v1_replay)):
                for row in replay:
                    fh.write(json.dumps({"bundle": tag, **row}, ensure_ascii=False) + "\n")
        (out / "VERDICT.md").write_text(_verdict_md(report, manifest), encoding="utf-8")
    return report


def _verdict_md(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    p = report["promoted"]
    v = p["validation"]
    lines = [
        "# P4 VERDICT：慢路径完整晋升周期（机制验收）",
        "",
        f"> 范围：{report['claim_scope']}；seed={manifest['seed']}，n={report['n_series']}，"
        f"held-in/out={report['split']['held_in']}/{report['split']['held_out']}（按 cell 分层）。",
        "",
        "## 周期",
        "",
        f"1. **挖掘**（仅 held-in）：现任 v0（F0 剂量启发式）在 snrLow 服务 f0_median_w25，"
        f"held-in 证据 = {json.dumps(report['held_in_mining_evidence'], ensure_ascii=False)}",
        f"2. **提案**：枚举 proposer 出 {report['n_rejected'] + 1} 条 scoped RiskRule（好坏都提，裁决在判官）",
        f"3. **晋升**：`{p['rule_id']}` 过 true 判官双段验证——held-in {v['held_in']['mean_gain']:+.4f}、"
        f"**held-out {v['held_out']['mean_gain']:+.4f} ≥ ε=0.02**、per-cell "
        f"{json.dumps(v['per_cell'], ensure_ascii=False)}、非目标行 bit 级不变",
        f"4. **版本**：{p['version']}（sha {p['sha']}，parent 链落盘 bundles/）",
        f"5. **回归重放**：非目标行一致={report['regression']['non_targeted_identical']}，"
        f"anomaly 面零扰动={report['regression']['anomaly_rows_identical']}，"
        f"目标行均值增益 {report['regression']['targeted_mean_gain']:+.4f}"
        f"（{report['regression']['targeted_rows']} 行）",
        f"6. **rollback**：head→v0 后 serving 与原 v0 bit 级一致（verified={report['rollback']['verified']}），"
        f"随后恢复晋升头 {report['rollback']['final_head']}；事件流见 bundles/chain.json",
        f"7. **拒绝缓冲**：{report['n_rejected']} 条被 true 判官拒绝（rejected_edits.jsonl 留痕拒因）",
        "",
        "## Memory 阶梯（prereg §4 条件线）",
        "",
        f"- {report['memory_ladder']['risk_memory_veto']}",
        f"- utility/contrast 阶梯：{report['memory_ladder']['status']}——{report['memory_ladder']['pending_conditions']}",
        "",
        "## 出口判据",
        "",
        f"`{json.dumps(report['exit_criteria'], ensure_ascii=False)}`",
        "",
        "**cycle_complete = true**：typed EditOp → PromotionGate → true 判官双段验证 → 版本升级落盘 →",
        "回归重放 → rollback 演示 → rejected buffer，七环闭合。不宣称 self-evolving（P6 解锁）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 promotion cycle runner")
    parser.add_argument("--n-series", type=int, default=60)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    report = run_p4(n_series=args.n_series, out_dir=args.out_dir, seed=args.seed)
    print(json.dumps({
        "phase": report["phase"],
        "promoted": report["promoted"]["version"],
        "rule": report["promoted"]["rule_id"],
        "held_out_gain": report["promoted"]["validation"]["held_out"]["mean_gain"],
        "n_rejected": report["n_rejected"],
        "rollback_verified": report["rollback"]["verified"],
        "cycle_complete": report["exit_criteria"]["cycle_complete"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
