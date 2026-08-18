"""V1 actionable headroom 扫描（零 LLM，2026-08-08）。

审查裁决第一步：在不放宽 H0 安全约束的前提下，找到至少一个
  modified_fraction ≤ 0.35（通过当前 verifier）且真实 Support gain ≥ M
  （headroom）的合法 Workflow——合法正控通过前不调 LLM、不宣称完整 Pipeline。

方法：
  1. actionable 算子集 = fast_agent._actionable_operators（构造默认候选实测
     verifier——与候选供给同源）；
  2. 对每个 actionable 算子 × gefcom 链全部 origin（736/784/832/880/928/976）
     用 v1.gain_at（确定性、零 LLM）评估 Support gain；
  3. 找 gain ≥ M 的 (op, origin) 对——合法的正控候选。

诚实口径：此扫描是"headroom 筛选"（找候选），不是"闭环验证"——合法候选的
最终确认需在真实入口用 verifier 返回的 PreparedSeries 做 Support（后续步骤）。

用法：
  python evaluation/functional/run_v1_actionable_headroom_scan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _actionable_operators,
    _allowed_operators,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402

REPORT_OUT_REL = Path("artifacts/functional/e2/w1_actionable_headroom_scan_report.json")
TARGET_DOMAIN = "gefcom"
CHAIN_ORIGINS = [736, 784, 832, 880, 928, 976]
PERIOD = 24


def main() -> int:
    root = PROJECT_ROOT
    m = core.MATERIAL_THRESHOLD
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    series0 = np.asarray(values[list(values.keys())[0]])
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)

    # actionable 集合（与真实入口候选供给同源——verifier 实测）
    request = PreparationRequest(
        "headroom-scan",
        series0,
        forecast_task_spec_v1(horizon=48, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series0, task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    allowed = _allowed_operators(request)
    actionable = _actionable_operators(request, series0, view, allowed)
    print(f"== actionable: {len(actionable)}/{len(allowed)} "
          f"{sorted(actionable)}")

    from run_w2_operator_scan import _default_params

    hits: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for op in sorted(actionable):
        compiled = v1.make_compiled(op, _default_params(op, period))
        for origin in CHAIN_ORIGINS:
            g = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            rows.append({"operator": op, "origin": origin, "support_gain": g})
            if g is not None and g >= m:
                hits.append({"operator": op, "origin": origin,
                             "support_gain": round(float(g), 6)})
        print(f"  {op:24s} gains={[round(float(r['support_gain']), 4) if r['support_gain'] is not None else None for r in rows if r['operator'] == op]}")

    verdict = ("ACTIONABLE_HEADROOM_FOUND" if hits else "NO_ACTIONABLE_HEADROOM")
    print(f"\n== hits ({len(hits)}): {hits}")
    print(f"== verdict: {verdict}")
    if not hits:
        print("== 诚实结论：H0 合法动作空间（14 个局部修复算子）在 gefcom 链上"
              "无 Support gain ≥ M 的 headroom——不靠放宽约束，换数据/链或立项"
              "全局变换 profile。")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-actionable-headroom-scan",
            "domain": TARGET_DOMAIN,
            "chain_origins": CHAIN_ORIGINS,
            "actionable_operators": sorted(actionable),
            "excluded_operators": sorted(set(allowed) - set(actionable)),
            "rows": rows,
            "hits": hits,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
