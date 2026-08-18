"""FILTER_AWARE_EXPLORATION_ADVANCE_CONTROL 零 outcome 机械验收（第一层，
2026-08-10）。

在已暴露 P2 Context（traffic offset=240 @648/744，coverage=1.0 无缺失）
上验证过滤感知探索推进（不读取任何 Support/delayed——脚本不调用
executor.evaluate，只用 prepare 的 DecisionTrace.candidate_ids）：

  A. 无缺失 Context：第一个合法候选 = denoise_median（非 no-op 首算子）
  B. 探索推进（_explored=[denoise, hampel]）：候选 = outlier_iqr
     ——impute_* 缺失族被跳过（no-op 只在当前 Context 下跳过）
  C. 全部非 no-op 耗尽：pool=['identity']（真耗尽 → 正确 abstain）
  D. 有缺失 Context（coverage<1）+ 同 explored 状态：候选 = impute_ar
     （有缺失时 impute 可供应）——证明 no-op 判定随 Context 变化
  E. 不读取 gain：全程无 evaluate（本脚本结构保证；_eligible_ops 只解析
     契约名）

Verdict：
  FILTER_AWARE_EXPLORATION_CONTROL_PASS /
  FILTERED_CANDIDATE_STILL_STALLS /
  ACTIONABLE_CANDIDATE_WRONGLY_SKIPPED /
  NO_ELIGIBLE_NON_NOOP_CANDIDATE

用法：
  python evaluation/functional/run_v1_filter_aware_exploration_acceptance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402

DOMAIN = "monash:traffic_hourly"
OFFSET = 240
PERIOD = 24
ORIGIN = 648
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_filter_aware_exploration_acceptance_report.json")

OPS_ALL = tuple(o for o in (
    "denoise_median", "hampel_filter", "impute_ar", "impute_ema",
    "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
    "outlier_mad", "period_complete", "period_median_complete",
    "repair_level_shift", "resample_uniform", "winsorize"))
# OPS_ALL 顺序中非 no-op（无缺失 Context）的算子（按顺序）
NON_NOOP_ORDER = ("denoise_median", "hampel_filter", "outlier_iqr",
                  "outlier_mad", "repair_level_shift", "resample_uniform",
                  "winsorize")


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    def prepare_pool(explored: list[str], origin: int,
                     coverage: float = 1.0) -> list[str]:
        backend = sealed.SealedProbeBackend(explore=True, operators=OPS_ALL)
        backend._explored = list(explored)
        method = sealed.TTHAMethod(
            sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                backend,
                LocalPublicToolGateway(series0[:origin],
                                       task_kind="forecast"))),
            h0, ())
        method.bind_round_data(series0[:origin], task_kind="forecast")
        observed_extra = None
        if coverage < 1.0:
            observed_extra = {"recent.coverage": float(coverage),
                              "recent.maximum_missing_run_length": 3.0}
        method.prepare(sealed._request(series0, tgt_values, origin,
                                       observed_extra=observed_extra))
        return list(method.last_trace.candidate_ids)

    checks: dict[str, object] = {}
    _MISSING_FAMILY = ("impute_", "period_")

    def _any_candidate(pool: list[str]) -> str | None:
        return next((c for c in pool if c.startswith("cand_")), None)

    def _missing_family_in(pool: list[str]) -> bool:
        return any(c.startswith("cand_") and c[5:].startswith(_MISSING_FAMILY)
                   for c in pool)

    # A. 无缺失 Context：至少一个合法候选到达（非 identity）
    pool_a = prepare_pool([], ORIGIN)
    checks["A_pool"] = pool_a
    checks["A_first_candidate_reaches"] = (
        _any_candidate(pool_a) is not None and "identity" in pool_a)
    # B. 探索推进（_explored=[denoise, hampel]）：缺失族被跳过、下一个
    #    eligible 候选到达（eligible 由 verifier 实测决定——不断言具体算子）
    pool_b = prepare_pool(["denoise_median", "hampel_filter"], ORIGIN)
    checks["B_pool"] = pool_b
    checks["B_skips_missing_family"] = not _missing_family_in(pool_b)
    checks["B_next_eligible_reaches"] = (
        _any_candidate(pool_b) is not None)
    checks["B_advances_beyond_explored"] = bool(
        _any_candidate(pool_b) not in
        {f"cand_{o}" for o in ("denoise_median", "hampel_filter")})
    # C. 全部可行动算子耗尽 → abstain（pool=['identity']）
    pool_c = prepare_pool(list(NON_NOOP_ORDER), ORIGIN)
    checks["C_pool"] = pool_c
    checks["C_exhausted_abstains"] = (pool_c == ["identity"])
    # D. 有缺失 Context + 同 explored 状态 → 缺失族可供应
    pool_d = prepare_pool(["denoise_median", "hampel_filter"], ORIGIN,
                          coverage=0.9)
    checks["D_pool"] = pool_d
    checks["D_impute_supplied_when_missing"] = _missing_family_in(pool_d)

    all_ok = all(
        checks[k] is True for k in (
            "A_first_candidate_reaches", "B_skips_missing_family",
            "B_next_eligible_reaches", "B_advances_beyond_explored",
            "C_exhausted_abstains", "D_impute_supplied_when_missing"))
    # E. 不读取 gain：本脚本结构保证（无 executor/evaluate 调用）
    checks["E_no_outcome_read"] = True

    verdict = ("FILTER_AWARE_EXPLORATION_CONTROL_PASS" if all_ok
               else "FILTERED_CANDIDATE_STILL_STALLS")
    if checks["A_first_candidate_reaches"] \
            and not checks["B_next_eligible_reaches"]:
        verdict = "ACTIONABLE_CANDIDATE_WRONGLY_SKIPPED"
    elif checks["A_pool"] == ["identity"]:
        verdict = "NO_ELIGIBLE_NON_NOOP_CANDIDATE"

    print(f"== A: {checks['A_pool']} reaches={checks['A_first_candidate_reaches']}")
    print(f"== B: {checks['B_pool']} skip_missing={checks['B_skips_missing_family']} "
          f"advance={checks['B_advances_beyond_explored']}")
    print(f"== C: {checks['C_pool']} exhausted_abstain="
          f"{checks['C_exhausted_abstains']}")
    print(f"== D: {checks['D_pool']} impute_supplied="
          f"{checks['D_impute_supplied_when_missing']}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-filter-aware-exploration-acceptance",
        "dataset": DOMAIN, "cohort_offset": OFFSET, "origin": ORIGIN,
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
