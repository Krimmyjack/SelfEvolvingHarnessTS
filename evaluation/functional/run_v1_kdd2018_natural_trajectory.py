"""KDD_CUP_2018_NATURAL_FAST_TRAJECTORY（P1，用户顺序实验 2026-08-11）。

P0 premise-only 已 PASS（outlier family 4 个合法替代）。P1：自然 Fast
轨迹寻找 material failure——零 outcome 冻结 cohort（长度/公开 Context/
outlier family 静态合法）+ 3 轮在线 Fast（正常入口 + Runtime 预算 2 实测
——material failure 检测，零 LLM）。

冻结规则（零 outcome——不读 gain、不因 outcome 换 series）：
  - 长度 ≥ 984（hourly；同 uci 窗口 Source 600/648、R1 792/840、R2 888/936）
  - 公开 Context 有 outlier 信号（level_excursion 或 region 特征）
  - outlier family（winsorize/outlier_iqr/outlier_mad/hampel）静态 verifier
    合法 ≥2（ScopeExecutor.verify——零 gain）
  - Source/Target 互斥预留（270 条分片——P4 用）

每轮：Fast prepare（force_pool=outlier family，探测 ≤2）→ evaluate →
Episode 写回 → material failure（gain < −M）检测。

Verdict：NATURAL_FAILURE_DETECTED / NO_NATURAL_FAILURE / PROTOCOL_FAILURE
（gain=None 仪器失败——按 P3.2 修复语义）。

用法：
  python evaluation/functional/run_v1_kdd2018_natural_trajectory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

PERIOD = 24  # hourly
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGINS = (600, 792, 888)
MIN_LEN = 984
OUTLIER_FAMILY = ("winsorize", "outlier_iqr", "outlier_mad",
                  "hampel_filter")
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
FROZEN_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_frozen_cohort.jsonl"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_natural_trajectory_report.json"


def _config() -> dict[str, object]:
    return {
        "dataset_id": "kdd_cup_2018",
        "sampling": "hourly_regular",
        "period": PERIOD,
        "anchors": list(range(312, 853, 60)),
        "support_origin": ORIGINS[0],
        "selection_origin": ORIGINS[0],
    }


def _evaluate_kdd(roster, values, compiled, config, *, origin):
    mapped = [dict(row, role="eval") if str(row["role"]) != "train"
              else dict(row) for row in roster]
    return v6._evaluate(mapped, values, compiled, config, origin=origin)


def _request(series0: np.ndarray, values, origin: int) -> PreparationRequest:
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    return PreparationRequest(
        "kdd2018-trajectory",
        series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )


def _family_steps(values: np.ndarray, origin: int,
                  op: str) -> tuple[tuple[str, dict], ...]:
    s0 = np.asarray(values[:origin], dtype=np.float64)
    fe = dict(extract_public_features(s0, task_kind="forecast"))
    bindings = OPERATOR_METADATA[op].get("public_parameter_bindings") or {}
    if bindings:
        params = {p: float(fe[f]) for p, f in bindings.items() if f in fe}
        if len(params) != len(bindings):
            return ()
    else:
        params = dict(wiring.contract_params(op, PERIOD))
    return ((op, params),)


def main() -> int:
    cache = np.load(CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    lengths = cache["lengths"]

    # ---- 冻结 cohort（零 outcome：长度 + 公开 Context + outlier family
    # 静态合法——ScopeExecutor.verify 零 gain）----
    role_seq = ["train"] * 12 + ["support"] * 4 + ["query"] * 4
    frozen: list[dict[str, object]] = []
    for i, n in enumerate(names):
        if len(frozen) >= 20:
            break
        s = np.asarray(values[i][:ORIGINS[0]], dtype=np.float64)
        fe = dict(extract_public_features(s, task_kind="forecast"))
        signal = bool(float(fe.get("level_excursion_score", 0.0)) > 1.0
                      or "estimated_region_start_fraction" in fe)
        if not signal:
            continue
        # family 静态合法 ≥2（cohort of 1——预筛；正式 cohort 验证在
        # 冻结后）
        ok = 0
        for op in OUTLIER_FAMILY:
            steps = _family_steps(values[i], ORIGINS[0], op)
            if not steps:
                continue
            ex = ScopeExecutor([{"series_uid": n, "role": "train"}],
                               {n: np.asarray(values[i], dtype=np.float64)},
                               _config(), evaluate_fn=_evaluate_kdd)
            if ex.verify(steps, ORIGINS[0]).passed:
                ok += 1
        if ok < 2:
            continue
        frozen.append({"cohort": "K0", "role": role_seq[len(frozen)],
                       "series_name": n, "type": "kdd2018"})
    if len(frozen) < 20:
        print(json.dumps({"verdict": "PREMISE_UNAVAILABLE",
                          "n_frozen": len(frozen)}, indent=1))
        return 0
    FROZEN_REL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen),
        encoding="utf-8")

    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in frozen]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in frozen}
    series0 = vals[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, vals, _config(),
                             evaluate_fn=_evaluate_kdd)
    h0 = compile_snapshot(PROJECT_ROOT / "methods/ttha/harness/h0",
                          verify_lock=False)

    rounds: list[dict[str, Any]] = []
    failure_detected = False
    failure_detail = None
    for r_i, origin in enumerate(ORIGINS):
        backend = sealed.SealedProbeBackend(
            explore=True, operators=OUTLIER_FAMILY,
            max_propose_candidates=2, force_pool=True)
        method = sealed.TTHAMethod(
            sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                backend,
                LocalPublicToolGateway(series0[:origin],
                                       task_kind="forecast"))),
            h0, ())
        method.bind_round_data(series0[:origin], task_kind="forecast")
        result = method.prepare(_request(series0, vals, origin))
        trace = method.last_trace
        steps_map = dict(trace.candidate_program_steps or {})
        pool_ops = [c[len("cand_"):] for c in trace.candidate_ids
                    if c.startswith("cand_") and c in steps_map]
        rd: dict[str, Any] = {"origin": origin, "pool": list(trace.candidate_ids),
                              "probes": [], "protocol_failure": False}
        for i, op in enumerate(pool_ops[:2]):
            steps = steps_map[f"cand_{op}"]
            rr = executor.evaluate(steps, origin)
            gain = (float(rr.gain) if rr.gain is not None else None)
            passed = bool(rr.verification.passed)
            entry: dict[str, Any] = {"probe": i + 1, "op": op, "gain": gain,
                                     "passed": passed}
            if passed and gain is None:
                rd["protocol_failure"] = True
                rd["protocol_reason"] = f"outcome_unavailable ({op})"
                rd["probes"].append(entry)
                break
            if passed:
                ep = tll.write_target_episode(
                    domain="kdd_cup_2018", op=op,
                    episode_id_suffix=f"_kdd_r{r_i + 1}_p{i + 1}",
                    program_steps=[{"op": o, "params": dict(p)}
                                   for o, p in steps],
                    support_gain=gain if gain is not None else 0.0,
                    delayed_gain=None,
                    support_context=dict(resolver.window_context(
                        vals, origin, PERIOD)))
                entry["episode_id"] = ep.episode_id
                entry["relation"] = ep.relation
                if gain is not None and gain < -M:
                    failure_detected = True
                    failure_detail = {"round": r_i + 1, "origin": origin,
                                      "op": op, "gain": gain}
            rd["probes"].append(entry)
            if gain is not None and gain >= M:
                break
        rounds.append(rd)
        print(f"== R{r_i + 1} @{origin}: "
              f"{[(p['op'], p['gain']) for p in rd['probes']]} "
              f"proto_fail={rd.get('protocol_failure')}")
        if rd.get("protocol_failure"):
            break
        if failure_detected:
            break

    if any(r.get("protocol_failure") for r in rounds):
        verdict = "PROTOCOL_FAILURE"
    elif failure_detected:
        verdict = "NATURAL_FAILURE_DETECTED"
    else:
        verdict = "NO_NATURAL_FAILURE"
    print(f"== verdict: {verdict}  failure={failure_detail}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-natural-trajectory",
        "note": "P1 自然 Fast 轨迹（零 LLM；KDD 2018；material failure "
                "检测——零 outcome 冻结）",
        "cohort": [r["series_name"] for r in frozen],
        "rounds": rounds,
        "failure_detected": failure_detected,
        "failure_detail": failure_detail,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    from typing import Any
    raise SystemExit(main())
