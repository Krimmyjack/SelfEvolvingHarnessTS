"""KDD_CUP_2018_HEADROOM_DIAGNOSIS（用户裁决 2026-08-11：复用已暴露 KDD
cohort 做确定性归因——零 LLM、development、不消费 virgin）。

诊断同一自然失败 Context（winsorize @600 失败）下 outlier family 全部
合法替代的真实 headroom：outlier_iqr / outlier_mad / hampel_filter
（outlier_mad 已在 P2 实测 −0.0014——补测其余）。

每候选记录：verifier、Support gain（@600）、delayed gain（@648）、
是否真正改变输入（behavior_point_count > 0）。

结果分支（用户裁决）：
  1. 所有替代都无稳定 headroom → KDD_OUTLIER_REPLACEMENT_NO_HEADROOM
     （关闭该 family，转 P4）
  2. 存在有效替代但 LLM 没选到 → HEADROOM_EXISTS_SELECTION_MISSED
     （first fault = Candidate Selection——双候选 Runtime 实测）
  3. Support 正 delayed 负 → CONTEXT_SCOPE_UNSTABLE（不形成 Skill）

用法：
  python evaluation/functional/run_v1_kdd2018_headroom_diagnosis.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _load_cohort,
)

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGIN = 600
DELAYED = ORIGIN + HORIZON
FAMILY = ("winsorize", "outlier_iqr", "outlier_mad", "hampel_filter")
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_headroom_diagnosis_report.json"


def _steps(values: np.ndarray, origin: int,
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
    root = PROJECT_ROOT
    cohort = _load_cohort(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)

    results: list[dict[str, object]] = []
    for op in FAMILY:
        steps = _steps(series0, ORIGIN, op)
        entry: dict[str, object] = {"op": op}
        if not steps:
            entry["verifier"] = False
            results.append(entry)
            continue
        v = executor.verify(steps, ORIGIN)
        entry["verifier"] = bool(v.passed)
        entry["checked_windows"] = v.checked_windows
        if not v.passed:
            results.append(entry)
            continue
        rr = executor.evaluate(steps, ORIGIN)
        sg = (float(rr.gain) if rr.gain is not None else None)
        entry["support_gain"] = sg
        entry["support_passed"] = bool(rr.verification.passed)
        entry["behavior_point_count"] = int(
            getattr(rr, "behavior_point_count", 0))
        rd = executor.evaluate(steps, DELAYED)
        dg = (float(rd.gain) if rd.gain is not None else None)
        entry["delayed_gain"] = dg
        entry["delayed_ok"] = bool(dg is not None and dg >= -M)
        print(f"== {op}: verifier={entry['verifier']} "
              f"support={sg} delayed={dg} "
              f"behavior={entry['behavior_point_count']}")
        results.append(entry)

    # ---- 结果分支（用户裁决）----
    valid = [r for r in results if r.get("support_gain") is not None]
    alternatives = [r for r in valid if str(r["op"]) != "winsorize"]
    has_headroom = [r for r in alternatives
                    if r["support_gain"] is not None
                    and float(r["support_gain"]) >= M
                    and r.get("delayed_ok")]
    stable_neg = [r for r in alternatives
                  if r["support_gain"] is not None
                  and float(r["support_gain"]) < M]
    ctx_unstable = [r for r in alternatives
                    if r["support_gain"] is not None
                    and float(r["support_gain"]) >= M
                    and not r.get("delayed_ok")]
    if has_headroom:
        verdict = "HEADROOM_EXISTS_SELECTION_MISSED"
    elif ctx_unstable:
        verdict = "CONTEXT_SCOPE_UNSTABLE"
    else:
        verdict = "KDD_OUTLIER_REPLACEMENT_NO_HEADROOM"
    print(f"== verdict: {verdict}")
    print(f"== has_headroom={[r['op'] for r in has_headroom]} "
          f"stable_neg={[r['op'] for r in stable_neg]} "
          f"unstable={[r['op'] for r in ctx_unstable]}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-headroom-diagnosis",
        "note": "development 归因（零 LLM；已暴露 cohort；不产生新 Claim/"
                "不消费 virgin）",
        "origin": ORIGIN, "delayed": DELAYED,
        "results": results,
        "has_headroom": [r["op"] for r in has_headroom],
        "stable_negative": [r["op"] for r in stable_neg],
        "context_unstable": [r["op"] for r in ctx_unstable],
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
