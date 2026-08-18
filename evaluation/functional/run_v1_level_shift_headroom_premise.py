"""LEVEL_SHIFT_PROGRAM_HEADROOM_PREMISE（用户裁决 2026-08-10，零 LLM）。

在已暴露 UCI offset=40 上用相同执行器验证 repair_level_shift 是否真的有
headroom（不调用 LLM、不修改 Harness）：

  identity（基准 0）
  denoise_median（当前候选池里的算子）
  repair_level_shift（level shift 对应算子）

要求：
  - Operator contract 默认合法参数（wiring.contract_params）
  - 经过相同 H0 verifier（ScopeExecutor.evaluate 含 verifier 先行）
  - 测真实 Support @origin；冻结 Program 后打开 delayed @origin+HORIZON

决策点：648/744/840（3 个；任一稳定 headroom 即判定 PRESENT 的判定基准
在冻结设计里：全部 3 点都无 headroom 才是 NO_LEVEL_SHIFT_HEADROOM；
任一 Support 正但 delayed 翻负 → SUPPORT_ONLY_UNSTABLE；repair 在 H0 下
不可执行 → VERIFIER_REJECTED）。

Verdict（预注册）：
  LEVEL_SHIFT_HEADROOM_PRESENT / NO_LEVEL_SHIFT_HEADROOM /
  SUPPORT_ONLY_UNSTABLE / VERIFIER_REJECTED

用法：
  python evaluation/functional/run_v1_level_shift_headroom_premise.py
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

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

DOMAIN = "uci_electricity_load_diagrams"
OFFSET = 40
PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGINS = (648, 744, 840)
PROGRAMS = ("identity", "denoise_median", "repair_level_shift")
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_level_shift_headroom_premise_report.json")


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)

    rows: list[dict[str, object]] = []
    for origin in ORIGINS:
        row: dict[str, object] = {"origin": origin}
        for op in PROGRAMS:
            if op == "identity":
                row[op] = {"support_gain": 0.0, "passed": True,
                           "delayed_gain": 0.0}  # identity 基准约定 0
                continue
            steps = ((op, dict(wiring.contract_params(op, PERIOD))),)
            rs = executor.evaluate(steps, origin)
            gs = (float(rs.gain) if rs.gain is not None else None)
            passed = bool(rs.verification.passed)
            rd = executor.evaluate(steps, origin + HORIZON)
            gd = (float(rd.gain) if rd.gain is not None else None)
            row[op] = {"support_gain": gs, "passed": passed,
                       "delayed_gain": gd}
        rows.append(row)
        print(f"== @{origin}: " + ", ".join(
            f"{op}={row[op]['support_gain'] and round(row[op]['support_gain'], 5)}"
            f"(d={row[op]['delayed_gain'] and round(row[op]['delayed_gain'], 5)})"
            for op in PROGRAMS))

    # ---- verdict（预注册判定）----
    repair_rows = [r["repair_level_shift"] for r in rows]
    verifier_rejected = any(not r["passed"] for r in repair_rows)
    support_positive = [
        r["support_gain"] for r in repair_rows
        if r["support_gain"] is not None and r["support_gain"] >= M]
    delayed_flip = [
        r["delayed_gain"] for r in repair_rows
        if r["delayed_gain"] is not None and r["delayed_gain"] < -M]
    if verifier_rejected:
        verdict = "VERIFIER_REJECTED"
    elif support_positive and not delayed_flip:
        verdict = "LEVEL_SHIFT_HEADROOM_PRESENT"
    elif support_positive and delayed_flip:
        verdict = "SUPPORT_ONLY_UNSTABLE"
    else:
        verdict = "NO_LEVEL_SHIFT_HEADROOM"

    print(f"== verdict: {verdict} "
          f"(repair positive points={len(support_positive)}, "
          f"delayed flips={len(delayed_flip)})")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-level-shift-headroom-premise",
        "dataset": DOMAIN, "cohort_offset": OFFSET,
        "origins": list(ORIGINS),
        "programs": list(PROGRAMS),
        "material_threshold": M,
        "results": rows,
        "repair_support_positive_count": len(support_positive),
        "repair_delayed_flip_count": len(delayed_flip),
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
