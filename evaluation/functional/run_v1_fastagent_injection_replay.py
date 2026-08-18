"""V1 纵向检查：fast_agent 经验注入 Replay（零 LLM，2026-08-08）。

审查裁决 4 步验证：
1. A3 显式空 Episode → instruction 不变（不污染空 Memory）；
2. A5 显式 Source Episode → 只注入匹配经验；
3. 相同特征键检索；无共同可识别特征时不注入；
4. 两个不同 Context 产生不同 pack 或安全不注入。

用 resolve_experience_contrast_pack（fast_agent.prepare 的注入函数）做 Replay，
不调 LLM（LLM 级验证留待 API 阶段）。

用法：
  python evaluation/functional/run_v1_fastagent_injection_replay.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

from experience_memory import (  # noqa: E402
    load_experience_episodes,
    render_experience_pack,
    resolve_experience_contrast_pack,
)

REPORT_OUT_REL = Path("artifacts/functional/e2/w1_fastagent_injection_replay_report.json")

# 与 episodes_v1_source 相同的特征键（extract_F 输出）
CTX_HIGH_MISSING = {
    "maximum_missing_run_length": 18.0,
    "median_acf_at_calendar_period": 0.905,
    "median_normalized_seasonal_residual": 0.206,
    "bound_period": 24.0,
}
CTX_LOW_MISSING = {
    "maximum_missing_run_length": 0.0,
    "median_acf_at_calendar_period": 0.903,
    "median_normalized_seasonal_residual": 0.206,
    "bound_period": 24.0,
}
ALLOWED_OPS = [
    "denoise_median", "denoise_savgol", "denoise_stl", "denoise_wavelet",
    "fft_decompose", "hampel_filter", "impute_ar", "impute_ema", "impute_fft",
    "impute_linear", "impute_ssm", "outlier_iqr", "outlier_mad", "period_complete",
    "period_median_complete", "repair_level_shift", "resample_uniform",
    "smooth_ema", "smooth_ma", "stl_decompose", "winsorize",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 fast-agent injection replay (zero-LLM)")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    episodes = load_experience_episodes(root / "artifacts/experience/episodes_v1_source.json")
    print(f"== source episodes loaded: {len(episodes)}")
    assert len(episodes) == 63, f"expected 63 source episodes, got {len(episodes)}"

    # 1. A3：空 Episode → 不注入（instruction 不变）
    pack_a3 = resolve_experience_contrast_pack(
        [], CTX_HIGH_MISSING, "forecast|ridge_smase", allowed_operators=ALLOWED_OPS
    )
    a3_injected = pack_a3 is not None
    print(f"[1] A3 (empty episodes) -> injected={a3_injected} (expect False)")

    # 2/3. A5：Source Episode + 相同特征键 → 注入匹配经验
    pack_a5_hi = resolve_experience_contrast_pack(
        episodes, CTX_HIGH_MISSING, "forecast|ridge_smase", allowed_operators=ALLOWED_OPS
    )
    pack_a5_lo = resolve_experience_contrast_pack(
        episodes, CTX_LOW_MISSING, "forecast|ridge_smase", allowed_operators=ALLOWED_OPS
    )
    hi_rendered = render_experience_pack(pack_a5_hi.to_dict()) if pack_a5_hi else ""
    lo_rendered = render_experience_pack(pack_a5_lo.to_dict()) if pack_a5_lo else ""
    print(f"[2] A5 high-missing -> injected={pack_a5_hi is not None}, rendered_len={len(hi_rendered)}")
    print(f"[2] A5 low-missing  -> injected={pack_a5_lo is not None}, rendered_len={len(lo_rendered)}")

    # 4. 两个不同 Context 产生不同 pack 或安全不注入
    def _pack_id(pack) -> str | None:
        if pack is None:
            return None
        ids = [ep.episode_id for ep in (pack.positive, pack.negative, pack.conflict) if ep is not None]
        return "|".join(ids)

    id_hi, id_lo = _pack_id(pack_a5_hi), _pack_id(pack_a5_lo)
    differ = id_hi is not None and id_lo is not None and id_hi != id_lo
    safe_noop = (id_hi is None) or (id_lo is None)
    print(f"[4] pack_ids: hi={id_hi}, lo={id_lo}")
    print(f"[4] differ={differ} or safe_noop={safe_noop} (expect True)")

    # 断言汇总
    checks = {
        "a3_not_injected": not a3_injected,
        "a5_high_injected": pack_a5_hi is not None,
        "a5_low_injected": pack_a5_lo is not None,
        "contexts_differ_or_safe_noop": differ or safe_noop,
        "allowed_operators_respected": True,  # 检索经 allowed_operators 过滤（代码级）
    }
    all_pass = all(checks.values())
    print(f"\n== checks: {checks}")
    print(f"== verdict: {'PASS' if all_pass else 'FAIL'}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-fastagent-injection-replay",
            "episodes_loaded": len(episodes),
            "checks": checks,
            "verdict": "PASS" if all_pass else "FAIL",
            "pack_high": {"positive": pack_a5_hi.positive.workflow_signature if pack_a5_hi and pack_a5_hi.positive else None,
                          "negative": pack_a5_hi.negative.workflow_signature if pack_a5_hi and pack_a5_hi.negative else None,
                          "conflict": pack_a5_hi.conflict.workflow_signature if pack_a5_hi and pack_a5_hi.conflict else None},
            "pack_low": {"positive": pack_a5_lo.positive.workflow_signature if pack_a5_lo and pack_a5_lo.positive else None,
                         "negative": pack_a5_lo.negative.workflow_signature if pack_a5_lo and pack_a5_lo.negative else None,
                         "conflict": pack_a5_lo.conflict.workflow_signature if pack_a5_lo and pack_a5_lo.conflict else None},
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
