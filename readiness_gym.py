"""readiness_gym.py — TS-Readiness Replay Gym（P3，Final_Plan_CodeAgentFirst §P3）。

0-API 离线环境：proposer（deterministic / code agent）在预算内对 ProgramSpec v1 候选做
**真实执行 + label-safe proxy 评估**，拿结构化 trace 反馈（R1：连续数值 + 失败原因），
最终 finalize 的 true 下游 delta 只落 episode result（实验者/validator 可见），
**永不进 agent 可见的 observation**（泄漏纪律，tests 守）。

label-safe proxy（test-time 合法反馈面，prereg R5）：
  forecast  seasonal-naive 回测：用 artifact 前缀的最后一个周期预测，对 **raw 观测尾窗**
            打分（目标固定为 raw 观测 → 程序改不了考卷，防自评膨胀）
  anomaly   告警保存率：raw 上冻结标定的检测器在 artifact 上保留了多少 raw 告警
            （±tol 膨胀匹配）；无标签、可部署

true 判官（result-only）：P2 冻结判官——forecast=seasonal-naive nRMSE vs 干净未来；
anomaly=residual z-score F1 vs 注入标签（raw 上标定）。

设计注记（P4 待办）：check_execution_invariants 的修改率对平滑程序天然 ≈1.0（每点微调
皆计数）——gym 把 modified_fraction/violations 作为 **trace 信息**上报，不据此拒绝；
拒绝语义属 SafetyGate 部署路径，β 预算需在 P4 重校准（计数空间 vs 幅度空间）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .evaluators.anomaly_rig import DETECTOR_SPEC, _dilate, anomaly_readiness_eval, residual_zscore_scores
from .harness.layers import minimal_l2
from .policy.evidence_packet import default_allowed_grammar
from .policy.program_edit import (
    ProgramSpecV1,
    check_execution_invariants,
    guard_matches,
    spec_v1_from_dict,
    validate_v1,
)
from .policy.task_spec import TaskSpec, anomaly_task_spec_v1, forecast_task_spec_v1
from .run_p2_motivation import seasonal_naive_nrmse
from .sandbox.executor import run_pipeline

GYM_SPEC = {
    "name": "ts_readiness_replay_gym",
    "version": "v1",
    "api_calls": 0,
    "proxies": {"forecast": "seasonal_naive_backtest_vs_raw_tail",
                "anomaly_detection": "alarm_preservation_frozen_scale"},
    "true_judges": {"forecast": "seasonal_naive_nrmse_vs_clean_future",
                    "anomaly_detection": "residual_zscore_F1_vs_injected_labels"},
    "invariants_policy": "reported_in_trace_not_rejecting (SafetyGate owns rejection, P4)",
}


# ── label-safe proxies ──────────────────────────────────────────────────────

def forecast_backtest_proxy(v: np.ndarray, raw_x: np.ndarray, period: int,
                            *, max_origins: int = 6) -> float:
    """rolling-origin seasonal-naive 回测 nRMSE（K 折均值，label-safe：只用历史观测）。

    单折版的教训（P3 首轮实测）：一个 24 点噪声考卷使 within-series 排序保真只有 ~0.35；
    多折把考卷噪声压 K 倍。每折 k：pred = v 的第 k+1 末周期，target = raw 的第 k 末周期
    （目标恒为 raw 观测 → 程序改不了考卷，防自评膨胀）。"""
    v = np.asarray(v, dtype=float).ravel()
    raw_x = np.asarray(raw_x, dtype=float).ravel()
    k_max = min(int(max_origins), v.size // period - 1)
    scores: List[float] = []
    for k in range(1, k_max + 1):
        hi = -(k - 1) * period if k > 1 else None
        target = raw_x[-k * period: hi]
        mask = np.isfinite(target)
        if int(mask.sum()) < 3:
            continue
        pred = v[-(k + 1) * period: -k * period].copy()
        if np.any(~np.isfinite(pred)):
            prefix = v[:-k * period]
            fill = float(np.nanmean(prefix)) if np.any(np.isfinite(prefix)) else 0.0
            pred = np.where(np.isfinite(pred), pred, fill)
        rmse = float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))
        scores.append(rmse / (float(np.std(target[mask])) + 1e-9))
    return float(np.mean(scores)) if scores else float("nan")


def forecast_proxy_delta(artifact: np.ndarray, raw_x: np.ndarray, period: int) -> float:
    """正 = 比 raw 前缀预测得更好（label-safe：只用历史观测）。"""
    return forecast_backtest_proxy(raw_x, raw_x, period) - forecast_backtest_proxy(artifact, raw_x, period)


def anomaly_alarm_preservation(artifact: np.ndarray, raw_x: np.ndarray) -> float:
    """raw 告警在 artifact 上的保存率（检测器尺度在 raw 上冻结；±tol 膨胀匹配）。无告警→1.0。"""
    w, t, tol = DETECTOR_SPEC["window"], DETECTOR_SPEC["threshold"], DETECTOR_SPEC["tol"]
    raw_flags = residual_zscore_scores(raw_x, window=w) > t
    n_raw = int(raw_flags.sum())
    if n_raw == 0:
        return 1.0
    art_flags = residual_zscore_scores(artifact, window=w, scale_reference=raw_x) > t
    preserved = int((raw_flags & _dilate(art_flags, tol)).sum())
    return preserved / n_raw


def anomaly_proxy_delta(artifact: np.ndarray, raw_x: np.ndarray) -> float:
    """≤0：丢告警即受罚（raw 自身=0）。"""
    return anomaly_alarm_preservation(artifact, raw_x) - 1.0


# ── Gym ─────────────────────────────────────────────────────────────────────

class ReadinessGym:
    """episodes = evaluators.anomaly_rig.make_anomaly_slice 行；task ∈ {forecast, anomaly_detection}。"""

    def __init__(self, rows: List[Mapping[str, Any]], *, task: str = "forecast",
                 budget: int = 8, defaults: Optional[Mapping[str, Mapping]] = None):
        if task not in ("forecast", "anomaly_detection"):
            raise ValueError(f"task 须 ∈ {{forecast, anomaly_detection}}，得到 {task!r}")
        self.rows = list(rows)
        self.task = task
        self.budget = int(budget)
        self.defaults = dict(defaults) if defaults is not None else minimal_l2().operator_defaults
        self.task_spec: TaskSpec = (forecast_task_spec_v1(horizon=24,
                                                          downstream_model_class="seasonal_naive_h24")
                                    if task == "forecast" else
                                    anomaly_task_spec_v1(
                                        downstream_model_class=f"residual_zscore_w{DETECTOR_SPEC['window']}"
                                                               f"_t{DETECTOR_SPEC['threshold']}"))
        self._state: Dict[int, Dict[str, Any]] = {}
        self._results: Dict[int, Dict[str, Any]] = {}

    @property
    def n_episodes(self) -> int:
        return len(self.rows)

    # ── observation（泄漏纪律：labels/future/true/raw 序列永不出现）────────────
    def _fingerprint(self, i: int) -> Dict[str, Any]:
        """P0 指纹（P5-A.2 前置①）：全部从**观测序列本身**计算（含估计 SNR，非生成真值），
        使 allowed_grammar 宣传的每个 guard 特征都有观测面背书——修复 P5-A 实测的契约不一致
        （宣传 guard 特征而不提供 → 179/180 guarded 程序被拒）。每 episode 计算一次并缓存。"""
        st = self._state[i]
        if "fingerprint" not in st:
            from .conditioning.key import struct_feats as _struct_feats
            from .e32_policy import P_FEATS
            f = _struct_feats(np.asarray(self.rows[i]["x"], dtype=float))
            st["fingerprint"] = {
                "snr": float(f.get("SNR", 0.0)),
                "missing_rate": float(f.get("missing_rate", 0.0)),
                "struct_feats": {k: float(f.get(k, 0.0)) for k in P_FEATS},
            }
        return st["fingerprint"]

    def _observation(self, i: int) -> Dict[str, Any]:
        st = self._state[i]
        row = self.rows[i]
        x = np.asarray(row["x"], dtype=float)
        fp = self._fingerprint(i)
        obs = {
            "episode_uid": str(row["uid"]),
            "task": self.task_spec.to_packet_dict(),
            "pattern": {
                "cell": str(row["cell"]),
                "period": int(row["period"]),
                "n": int(x.size),
                "snr": fp["snr"],
                "missing_rate": fp["missing_rate"],
                "struct_feats": dict(fp["struct_feats"]),
                "iqr_observed": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
            },
            "allowed_grammar": default_allowed_grammar(),
            "budget_remaining": int(st["budget"]),
            "evals": list(st["evals"]),
        }
        json.dumps(obs, allow_nan=False)   # 构造性保证可序列化（NaN 不得进 observation）
        return obs

    def reset(self, i: int) -> Dict[str, Any]:
        self._state[i] = {"budget": self.budget, "evals": [], "done": False, "used": 0}
        self._results.pop(i, None)
        return self._observation(i)

    # ── 执行与评估 ───────────────────────────────────────────────────────────
    def _execute(self, spec: ProgramSpecV1, x: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        resolved = [(op, {**self.defaults.get(op, {}), **dict(params)}) for op, params in spec.steps]
        result = run_pipeline(resolved, x)
        if not result.ok or result.artifact is None or result.artifact.shape != x.shape:
            return None, f"execution_failed:{result.error or 'shape'}"
        return np.asarray(result.artifact, dtype=float), ""

    def _proxy_delta(self, artifact: np.ndarray, row: Mapping[str, Any]) -> float:
        x = np.asarray(row["x"], dtype=float)
        if self.task == "forecast":
            return float(forecast_proxy_delta(artifact, x, int(row["period"])))
        return float(anomaly_proxy_delta(artifact, x))

    def _true_delta(self, artifact: np.ndarray, row: Mapping[str, Any]) -> float:
        x = np.asarray(row["x"], dtype=float)
        if self.task == "forecast":
            raw_s = seasonal_naive_nrmse(x, row["future_clean"], int(row["period"]))
            art_s = seasonal_naive_nrmse(artifact, row["future_clean"], int(row["period"]))
            return float(raw_s - art_s)
        raw_f1 = anomaly_readiness_eval(x, row["labels"], raw_reference=x)["F1"]
        art_f1 = anomaly_readiness_eval(artifact, row["labels"], raw_reference=x)["F1"]
        return float(art_f1 - raw_f1)

    def _parse_program(self, payload: Any) -> Tuple[Optional[ProgramSpecV1], str]:
        try:
            spec = spec_v1_from_dict(payload)
        except ValueError as exc:
            return None, f"malformed_program_spec:{exc}"
        ok, why = validate_v1(spec)
        if not ok:
            return None, why
        if spec.task_type != self.task:
            return None, f"task_mismatch: program task_type={spec.task_type!r} ≠ gym task {self.task!r}"
        return spec, ""

    def _eval_program(self, i: int, payload: Any) -> Dict[str, Any]:
        row = self.rows[i]
        x = np.asarray(row["x"], dtype=float)
        spec, reason = self._parse_program(payload)
        if spec is None:
            return {"ok": False, "reason": reason, "program_sha": None,
                    "proxy_delta": None, "modified_fraction": None, "violations": []}
        obs_pattern = self._fingerprint(i)   # P5-A.2：guard 用与 observation 同源的 P0 指纹评估
        if spec.pattern_guard:
            try:
                if not guard_matches(spec, obs_pattern):
                    return {"ok": False, "reason": "pattern_guard_unsatisfied",
                            "program_sha": spec.sha(), "proxy_delta": None,
                            "modified_fraction": None, "violations": []}
            except KeyError:
                return {"ok": False, "reason": "pattern_guard_feature_missing",
                        "program_sha": spec.sha(), "proxy_delta": None,
                        "modified_fraction": None, "violations": []}
        artifact, err = self._execute(spec, x)
        if artifact is None:
            return {"ok": False, "reason": err, "program_sha": spec.sha(),
                    "proxy_delta": None, "modified_fraction": None, "violations": []}
        inv_ok, detail = check_execution_invariants(spec, x, artifact)
        proxy = self._proxy_delta(artifact, row)
        return {"ok": True, "reason": "",
                "program_sha": spec.sha(),
                "proxy_delta": None if not np.isfinite(proxy) else float(proxy),
                "modified_fraction": detail.get("modified_fraction"),
                "violations": list(detail.get("violations") or []),
                "invariants_ok": bool(inv_ok)}

    def step(self, action: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        # 单 episode 语义：最近 reset 的 episode（多 episode 并行由调用方管理索引）
        i = self._current_episode()
        st = self._state[i]
        if st["done"]:
            raise RuntimeError("episode 已结束：先 reset")
        op = str(action.get("op"))

        if op == "proxy_eval":
            if st["budget"] <= 0:
                st["evals"].append({"ok": False, "reason": "budget_exhausted",
                                    "program_sha": None, "proxy_delta": None,
                                    "modified_fraction": None, "violations": []})
                return self._observation(i), False
            st["budget"] -= 1
            st["used"] += 1
            st["evals"].append(self._eval_program(i, action.get("program_spec")))
            return self._observation(i), False

        if op in ("finalize", "abstain"):
            st["done"] = True
            row = self.rows[i]
            x = np.asarray(row["x"], dtype=float)
            payload = action.get("program_spec") if op == "finalize" else None
            if payload is None:
                self._results[i] = self._result_row(i, "abstain", None, 0.0, 0.0)
                return self._observation(i), True
            spec, reason = self._parse_program(payload)
            if spec is None:
                self._results[i] = self._result_row(i, "invalid_program_fallback_raw", None, 0.0, 0.0,
                                                    reason=reason)
                return self._observation(i), True
            artifact, err = self._execute(spec, x)
            if artifact is None:
                self._results[i] = self._result_row(i, "execution_failed_fallback_raw", spec, 0.0, 0.0,
                                                    reason=err)
                return self._observation(i), True
            proxy = self._proxy_delta(artifact, row)
            true = self._true_delta(artifact, row)
            self._results[i] = self._result_row(i, "program", spec, proxy, true)
            return self._observation(i), True

        raise ValueError(f"未知 gym action op={op!r}（∈ proxy_eval/finalize/abstain）")

    def _result_row(self, i: int, kind: str, spec: Optional[ProgramSpecV1],
                    proxy: float, true: float, reason: str = "") -> Dict[str, Any]:
        st = self._state[i]
        return {
            "uid": str(self.rows[i]["uid"]),
            "task": self.task,
            "final_kind": kind,
            "final_program_sha": spec.sha() if spec is not None else None,
            "proxy_delta_final": float(proxy) if np.isfinite(proxy) else None,
            "true_delta": float(true),
            "proxy_evals_used": int(st["used"]),
            "budget_remaining": int(st["budget"]),
            "reason": reason,
        }

    def _current_episode(self) -> int:
        active = [i for i, st in self._state.items() if not st["done"]]
        if not active:
            done_eps = sorted(self._state)
            if done_eps:
                raise RuntimeError("所有已 reset 的 episode 均已结束：先 reset 新 episode")
            raise RuntimeError("未 reset 任何 episode")
        return active[-1]

    def result(self, i: int) -> Dict[str, Any]:
        if i not in self._results:
            raise KeyError(f"episode {i} 尚无 result（未 finalize/abstain）")
        return dict(self._results[i])
