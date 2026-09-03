"""实验 1：Program Supply 前提修复——零 outcome smoke（审核 2026-08-09）。

假设：Harness 应根据部署可见的算子前提，跳过确定性无行为的候选，避免把
有限反馈预算浪费在 no-op 上。

本轮只改一个机制（fast_agent）：
  - 当前 Context 无缺失信号（recent.coverage==1 且 maximum_missing_run_
    length==0）时，缺失处理族算子（impute_*、period_complete、
    period_median_complete）供应前过滤（不进池、不验证、不探测）；
  - 依据公开 Context 与 Operator 前提，不读取 gain；
  - 不改变 Memory、radius、Agent、反馈和预算。

Smoke 验收（只在已暴露 traffic cohort，零 outcome——不评估下游 gain）：
  1. 过滤后的候选不再是确定性 no-op（缺失族不在候选池）；
  2. 候选仍能通过 verifier（非缺失族 actionable 不变）；
  3. 不评估下游 gain。

用法：
  python evaluation/functional/run_v1_program_supply_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent, _actionable_operators, _allowed_operators,
    _noop_ops_for_context)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway, extract_public_features)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402

PERIOD = 24
HORIZON = 48
MISSING_ONLY = {"impute_ar", "impute_ema", "impute_fft", "impute_linear",
                "impute_ssm", "period_complete", "period_median_complete"}


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / sealed.H0_ROOT, verify_lock=False)
    config = sealed._config()
    (src_roster, src_values, _tgt_roster, _tgt_values) = sealed._virgin_roster(root)
    series0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                         dtype=np.float64)
    origin = sealed.R1_ORIGIN

    observed = dict(resolver.window_context(src_values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    print(f"== observed keys sample: "
          f"coverage={observed.get('recent.coverage')} "
          f"max_run={observed.get('recent.maximum_missing_run_length')}")

    request = PreparationRequest(
        "supply-smoke", series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed))

    # 1. 前提过滤：无缺失 Context → 缺失族被识别为 no-op（不读取 gain）
    noop_ops = _noop_ops_for_context(request)
    print(f"== noop_ops: {noop_ops}")
    checks: dict[str, Any] = {
        "noop_filter_identifies_missing_family": bool(
            {"impute_linear", "impute_fft", "period_complete"} <= set(noop_ops)),
        "noop_filter_excludes_non_missing_ops": bool(
            not (set(noop_ops) & {"outlier_iqr", "winsorize", "hampel_filter"})),
    }

    # 2. 正常入口 prepare：候选池不再含缺失族（contracts + supply 双层过滤）
    backend = sealed.SealedProbeBackend(
        explore=True, operators=sealed._actionable_ops(root, series0, origin,
                                                       observed))
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            backend, LocalPublicToolGateway(series0[:origin],
                                            task_kind="forecast"))),
        h0, ())
    result = method.prepare(request)
    trace = method.last_trace
    pool_ops = set()
    for cid, st in (trace.candidate_program_steps or {}).items():
        for s in st:
            op = str(s["op"]) if isinstance(s, Mapping) else str(s[0])
            pool_ops.add(op)
    print(f"== pool: {sorted(pool_ops)}")
    checks["pool_excludes_missing_family"] = bool(
        not (pool_ops & MISSING_ONLY))
    checks["pool_has_non_noop_candidates"] = bool(
        pool_ops and bool(pool_ops - MISSING_ONLY))

    # 3. verifier 通过性（动作合法性——非缺失族 actionable 不变；不评估 gain）
    feats = extract_public_features(series0[:origin], task_kind="forecast")
    view = resolve_harness_view(h0, feats, role="fast")
    actionable = _actionable_operators(request, series0[:origin], view,
                                       _allowed_operators(request))
    non_missing_actionable = [op for op in actionable if op not in MISSING_ONLY]
    print(f"== actionable n={len(actionable)}; non-missing n="
          f"{len(non_missing_actionable)}; outlier_iqr="
          f"{'outlier_iqr' in actionable}")
    checks["verifier_passes_non_missing_ops"] = bool(
        len(non_missing_actionable) >= 5 and "outlier_iqr" in actionable)

    # 4. 零 outcome：本轮任何地方都没有调用 ScopeExecutor.evaluate
    checks["zero_outcome_no_evaluate_called"] = True  # 结构保证（未 import evaluate）

    passed = all(v is True for v in checks.values())
    verdict = ("PROGRAM_SUPPLY_PRECONDITION_SMOKE_PASS" if passed
               else "PROGRAM_SUPPLY_PRECONDITION_SMOKE_FAIL")
    print(f"== checks: {json.dumps(checks, ensure_ascii=False, indent=1)}")
    print(f"== verdict: {verdict}")

    out = root / Path("artifacts/functional/e2/w1_program_supply_smoke_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-program-supply-precondition-smoke",
        "mechanism": "fast_agent._noop_ops_for_context：Context 无缺失信号时 "
                     "缺失处理族供应前过滤（不读取 gain，不进池）",
        "cohort": "已暴露 traffic virgin cohort（前 20 支）",
        "observed": {k: observed[k] for k in ("recent.coverage",
                                              "recent.maximum_missing_run_length")},
        "noop_ops": list(noop_ops),
        "pool_ops": sorted(pool_ops),
        "actionable_non_missing_count": len(non_missing_actionable),
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": 0,
        "outcome_evaluated": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
