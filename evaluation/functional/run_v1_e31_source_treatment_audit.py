"""E31_SOURCE_TREATMENT_AUDIT（用户裁决 2026-08-12：零新 outcome——
development——明确记录 Source treatment 是否产生可行动 prior）。

问题（用户裁决承重 1）：A5 输入 = 1 条真实 Monash winsorize 正例 +
同一 Episode 机械 signswap 负例——同一算子同一 Context 正负证据可能
聚合为 CONFLICT——Runtime prior 只接受 POSITIVE_PRIOR——R600 可能
没有 cand_prior_winsorize——"Memory rendered" ≠ "Source prior
actionable"。

audit（零 LLM——确定性重放 E31 装置 @600）：
  1. resolve_order 对 Source 组合（正 + signswap 负）的 per_op verdict
     （winsorize——POSITIVE_PRIOR / CONFLICT / UNKNOWN?）；
  2. Runtime prior 生成条件（POSITIVE_PRIOR）→ 是否生成 cand_prior_*；
  3. 区分 rendered（指令注入）与 actionable（prior 进池）。

输出：
  SOURCE_PRIOR_ACTIONABLE : winsorize POSITIVE_PRIOR 且 cand_prior_*
    进池
  SOURCE_PRIOR_NOT_ACTIONABLE : verdict 非 POSITIVE_PRIOR（如
    CONFLICT——正负中和）→ prior 未生成（记录确切 verdict）
  SOURCE_PRIOR_UNKNOWN : 其他

用法：
  python evaluation/functional/run_v1_e31_source_treatment_audit.py
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
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_memory_gate import (  # noqa: E402
    _monash_source_episodes,
    _signswap,
)
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _request,
)

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    resolve_order,
)

ORIGIN = 600
PERIOD = 24
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_e31_source_treatment_audit_report.json"


def main() -> int:
    root = PROJECT_ROOT
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_e31.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    series0 = np.asarray(values[names.index(str(rows[0]["series_name"]))],
                         dtype=np.float64)
    req = _request(series0, {str(rows[0]["series_name"]): series0}, ORIGIN)
    observed = dict(getattr(req, "observed_pattern_spec", {}) or {})
    qc = {k: float(v) for k, v in observed.items()
          if str(k).startswith(("recent.", "change."))
          and isinstance(v, (int, float))}

    source_pos = _monash_source_episodes(root)
    source_neg = _signswap(source_pos)
    source_combo = (*source_pos, *source_neg)
    fe = dict(extract_public_features(series0[:ORIGIN],
                                      task_kind="forecast"))

    # 1. signed verdict（resolve_order——确定性）
    order, signed = resolve_order(
        query_context=qc, episodes=source_combo,
        operators=("winsorize", "outlier_mad", "hampel_filter"),
        material_threshold=resolver.MATERIAL_THRESHOLD,
        task_consumer_key="forecast|ridge|sMASE",
        allowed_operators=("winsorize", "outlier_mad", "hampel_filter"))
    per_op = signed.get("per_op", {})
    winsorize_verdict = (per_op.get("winsorize") or {}).get("verdict")
    summary = signed.get("summary", {}).get("verdict_counts")

    # 2. prior 生成条件（POSITIVE_PRIOR）——与 fast_agent 同逻辑
    prior_op = next(
        (op for op, st in per_op.items()
         if (st or {}).get("verdict") == "POSITIVE_PRIOR"), None)
    prior_actionable = bool(prior_op is not None)

    # 3. rendered vs actionable：渲染路径（signed 存在 → rendered）
    rendered = bool(per_op)

    if prior_actionable:
        verdict = "SOURCE_PRIOR_ACTIONABLE"
        reason = (f"winsorize verdict={winsorize_verdict} "
                  f"prior_op={prior_op}")
    elif winsorize_verdict is not None \
            and winsorize_verdict != "POSITIVE_PRIOR":
        verdict = "SOURCE_PRIOR_NOT_ACTIONABLE"
        reason = (f"winsorize verdict={winsorize_verdict} "
                  f"(CONFLICT 等——正负中和) — prior 未生成")
    else:
        verdict = "SOURCE_PRIOR_UNKNOWN"
        reason = f"verdict_counts={summary}"
    print(f"== source combo: {[getattr(e, 'episode_id', '?') for e in source_combo]}")
    print(f"== winsorize verdict: {winsorize_verdict}")
    print(f"== verdict_counts: {json.dumps(summary)}")
    print(f"== rendered: {rendered}  prior_actionable: {prior_actionable}")
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-e31-source-treatment-audit",
        "note": "E31 Source treatment audit（零 LLM/零新 outcome——"
                "development——用户裁决 2026-08-12 承重 1）",
        "origin": ORIGIN,
        "source_episodes": [getattr(e, "episode_id", "?")
                            for e in source_combo],
        "winsorize_verdict": winsorize_verdict,
        "verdict_counts": summary,
        "prior_generation": {
            "prior_op": prior_op,
            "actionable": prior_actionable,
            "cand_prior_expected": bool(prior_actionable)},
        "rendered_vs_actionable": {
            "rendered": rendered,
            "actionable": prior_actionable,
            "distinction": "rendered=指令注入；actionable=prior 进池"},
        "verdict": verdict,
        "reason": reason,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
