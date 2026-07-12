"""slow_path/true_judge_validator.py — true 判官 held-in/held-out validator（P4，Final_Plan §P4）。

P3 保真度判决的 binding 后果（prereg §3/§4）：promotion 的 accept/reject **一律以冻结
true 判官计分**（forecast=seasonal-naive nRMSE vs 干净未来；anomaly=residual z-score F1
vs 注入标签），gym-proxy 只可作 proposer 侧搜索信号。

serving 走部署消费面：candidate EditOp → apply_edits → compile_bundle →
**RiskAwareRouterPolicy**（§13.4 焊点）→ menu 动作 → sandbox 执行——学到的知识若不被
该路径消费，就不算晋升（Critical Review §4 知识断路教训）。

判据（caller 在 manifest 里声明数值）：
  ① held-in  mean paired gain > 0（发现段方向一致）
  ② held-out mean paired gain ≥ min_heldout_gain（默认取 P3 冻结 ε=0.02）
  ③ held-out 每 cell mean gain ≥ −worst_cell_tol（worst-group 安全，F0 season 教训）
  ④ 规则未触发的行 serving **bit 级不变**（作用域纪律 = 无非目标扰动）
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..evaluators.anomaly_rig import anomaly_readiness_eval
from ..policy.action_spec import ActionMenu
from ..policy.edits import EditOp, PolicyBundle, apply_edits, compile_bundle
from ..policy.router_policy import RouterPolicy, RoutingDecision
from ..run_p2_motivation import seasonal_naive_nrmse
from ..sandbox.executor import run_pipeline


class SubstrateRouterPolicy(RouterPolicy):
    """gym substrate 的 v0 现任 router（P4 晋升实验的被改进对象）。

    forecast 沿用 F0 时代剂量启发式：snrLow → f0_median_w25、snrHigh → f0_median_w9
    ——在强季节 substrate 上 w25 是**已知错误**（P2/P3 实测 w25 均值 +0.28 vs w9 +0.92），
    这正是慢路径要用证据修复的对象。anomaly → v_none（插补基线，registry 契约内唯一合法面）。
    """

    def predict(self, conditioning_key: Dict[str, Any], action_menu: ActionMenu,
                model_menu: Optional[List[str]] = None) -> RoutingDecision:
        task = str((conditioning_key.get("task") or {}).get("type") or "forecast")
        cell = str(conditioning_key.get("cell_id") or "")
        if task != "forecast":
            action = "v_none"
        else:
            action = "f0_median_w25" if "snrLow" in cell else "f0_median_w9"
        return RoutingDecision(action_id=action, abstained=False,
                               fallback_action="v_none",
                               provenance={"source": "substrate_incumbent_v0"})


def _conditioning_key(row: Mapping[str, Any], task: str) -> Dict[str, Any]:
    return {"pattern": {"struct_feats": {}}, "task": {"type": task},
            "cell_id": str(row["cell"])}


def _true_delta(task: str, artifact: np.ndarray, row: Mapping[str, Any], x: np.ndarray) -> float:
    if task == "forecast":
        raw_s = seasonal_naive_nrmse(x, row["future_clean"], int(row["period"]))
        art_s = seasonal_naive_nrmse(artifact, row["future_clean"], int(row["period"]))
        return float(raw_s - art_s)
    raw_f1 = anomaly_readiness_eval(x, row["labels"], raw_reference=x)["F1"]
    art_f1 = anomaly_readiness_eval(artifact, row["labels"], raw_reference=x)["F1"]
    return float(art_f1 - raw_f1)


def evaluate_bundle(bundle: PolicyBundle, rows: Sequence[Mapping[str, Any]], task: str,
                    menu: ActionMenu, base_router: RouterPolicy) -> List[Dict[str, Any]]:
    """bundle → RiskAwareRouterPolicy → 逐行 serve + true 判官打分。"""
    router = compile_bundle(bundle, base_router, base_menu=menu)
    out: List[Dict[str, Any]] = []
    for row in rows:
        x = np.asarray(row["x"], dtype=float)
        decision = router.predict(_conditioning_key(row, task), menu)
        spec = menu.actions[decision.action_id]
        result = run_pipeline([(s.op, dict(s.params)) for s in spec.steps], x)
        ok = bool(result.ok and result.artifact is not None and result.artifact.shape == x.shape)
        artifact = np.asarray(result.artifact, dtype=float) if ok else x.copy()
        fired = list((decision.provenance.get("risk_policy") or {}).get("fired", []))
        out.append({
            "uid": str(row["uid"]), "cell": str(row["cell"]), "task": task,
            "action_id": decision.action_id, "fired_rules": fired,
            "executed_ok": ok,
            "true_delta": _true_delta(task, artifact, row, x) if ok else 0.0,
            "artifact_sha": hashlib.sha256(artifact.tobytes()).hexdigest()[:16],  # bit 级一致性核验
        })
    return out


def validate_edit(bundle: PolicyBundle, edit: EditOp,
                  held_in: Sequence[Mapping[str, Any]], held_out: Sequence[Mapping[str, Any]],
                  task: str, menu: ActionMenu, base_router: RouterPolicy,
                  *, min_heldout_gain: float, worst_cell_tol: float) -> Dict[str, Any]:
    """candidate EditOp 的 true 判官双段验证（四判据，见模块 docstring）。"""
    candidate, log = apply_edits(bundle, [edit])
    if not log or not log[0]["applied"]:
        return {"accepted": False,
                "reasons": [f"edit_op_invalid: {log[0]['reason'] if log else 'no-op'}"],
                "held_in": None, "held_out": None, "per_cell": None,
                "non_targeted_identical": None, "candidate_version": None}

    def _paired(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
        base_rows = evaluate_bundle(bundle, rows, task, menu, base_router)
        cand_rows = evaluate_bundle(candidate, rows, task, menu, base_router)
        gains = np.asarray([c["true_delta"] - b["true_delta"]
                            for b, c in zip(base_rows, cand_rows)], dtype=float)
        return base_rows, cand_rows, gains

    base_in, cand_in, gains_in = _paired(held_in)
    base_out, cand_out, gains_out = _paired(held_out)

    non_targeted_identical = all(
        (c["artifact_sha"] == b["artifact_sha"] and c["action_id"] == b["action_id"])
        for b, c in zip(base_in + base_out, cand_in + cand_out) if not c["fired_rules"])

    cells = sorted({c["cell"] for c in cand_out})
    per_cell = {cell: float(np.mean([c["true_delta"] - b["true_delta"]
                                     for b, c in zip(base_out, cand_out) if c["cell"] == cell]))
                for cell in cells}

    reasons: List[str] = []
    mean_in, mean_out = float(gains_in.mean()), float(gains_out.mean())
    if mean_in <= 0.0:
        reasons.append(f"held_in mean gain {mean_in:+.4f} ≤ 0（发现段方向不成立）")
    if mean_out < float(min_heldout_gain):
        reasons.append(f"held_out mean gain {mean_out:+.4f} < min_heldout_gain {min_heldout_gain}"
                       "（true 判官，P3 冻结 ε 口径）")
    bad_cells = {cell: g for cell, g in per_cell.items() if g < -float(worst_cell_tol)}
    if bad_cells:
        reasons.append(f"worst-group 违规（held-out per-cell gain < −{worst_cell_tol}）: {bad_cells}")
    if not non_targeted_identical:
        reasons.append("非目标行 serving 发生变化（作用域纪律违规）")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "held_in": {"mean_gain": mean_in, "n": int(gains_in.size),
                    "fired_rows": int(sum(1 for c in cand_in if c["fired_rules"]))},
        "held_out": {"mean_gain": mean_out, "n": int(gains_out.size),
                     "fired_rows": int(sum(1 for c in cand_out if c["fired_rules"]))},
        "per_cell": per_cell,
        "non_targeted_identical": bool(non_targeted_identical),
        "candidate_version": candidate.version,
        "candidate_sha": candidate.sha(),
    }
