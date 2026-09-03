"""工作包 2 边界检查：零 LLM 确定性 C 段重放（deepseek 副本，2026-08-06）。

用途（用户决定第 3 项）：把已出现的候选算子（impute_ar/impute_fft/period_median_complete）
在修正后的 C 切片（B 段之后、不重叠）上确定性重放，不调用 LLM：
- 若至少一个候选的 delayed gain（C 段 selection_gain）为正 → 批准恢复三域全量 Campaign；
- 若全部为负 → 停止增加 LLM 样本，处理 Program headroom 或 Support→Delayed 可识别性。

B/C 不重叠规则：C_origin = B_support + HORIZON；约束 B_support + 2*HORIZON <= max_len。
GEFCom：max_len=1024 → shift=16 → B=928、C=976（976+48=1024 刚好）。

用法：
  python evaluation/functional/run_w2_c_replay.py [--domain gefcom] [--candidates impute_ar,impute_fft,period_median_complete]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402

HORIZON = 48
DEFAULT_CANDIDATES = ("impute_ar", "impute_fft", "period_median_complete")
PARAMS = {
    "period_median_complete": {"period": 24, "cycles": 3, "min_donors": 2},
    "impute_ar": {},
    "impute_fft": {},
}


def _max_len_for(root: Path, dataset_id: str) -> int:
    rows = [
        json.loads(line)
        for line in (root / "artifacts/frozen/benchmark_v02/series_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    lens = [int(r["length"]) for r in rows if r.get("dataset_id") == dataset_id]
    if not lens:
        raise ValueError(f"no registry rows for {dataset_id}")
    return max(lens)


def _make_fixed_proposer(op: str, params: Mapping[str, object]):
    """确定性候选注入 proposer（非 LLM；固定返回指定算子候选）。"""
    def proposer(payload: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "decision": "PROPOSE",
            "steps": [{"op": op, "params": dict(params), "bindings": {}}],
            "requested_observations": [],
            "fallback": "IDENTITY",
        }
    return proposer


def _abstain_proposer(payload: Mapping[str, object]) -> Mapping[str, object]:
    return {"decision": "ABSTAIN"}


def main() -> int:
    parser = argparse.ArgumentParser(description="W2 zero-LLM C-segment deterministic replay")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom")
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    args = parser.parse_args()
    root = args.root.resolve()

    domain = args.domain
    if domain not in v6.DATASET_CONFIGS:
        raise SystemExit(f"unknown domain: {domain}")
    config = dict(v6.DATASET_CONFIGS[domain])
    dataset_id = str(config["dataset_id"])
    max_len = _max_len_for(root, dataset_id)
    period = int(config.get("period", 1))
    support_origin = int(config["support_origin"])

    # B/C 不重叠：shift = min(period, max_len - support - 2*HORIZON)；C = B + HORIZON
    shift = min(period, max_len - support_origin - 2 * HORIZON)
    if shift <= 0:
        raise SystemExit(f"domain {domain}: no room for non-overlapping C segment "
                         f"(max_len={max_len}, support={support_origin})")
    b_support = support_origin + shift
    c_origin = b_support + HORIZON
    print(f"== {domain}: max_len={max_len}, B_support={b_support}, C_origin={c_origin}, "
          f"B=[{b_support},{b_support + HORIZON}) C=[{c_origin},{c_origin + HORIZON}) (non-overlapping)")

    # 覆盖 config：一次 run 同时用 B 段 support origin 与 C 段 selection origin
    cfg = dict(config)
    cfg["support_origin"] = b_support
    cfg["selection_origin"] = c_origin
    v6.DATASET_CONFIGS[domain] = cfg  # v6.run 内部从此处读 origins

    candidates = tuple(c for c in args.candidates.split(",") if c.strip())
    results: dict[str, Any] = {}
    for op in candidates:
        proposer = _make_fixed_proposer(op, PARAMS.get(op, {}))
        try:
            report = v6.run(
                root, initial_proposer=proposer, revision_proposer=proposer,
                dataset_key=domain, write_report=False,
            )
        finally:
            v6.DATASET_CONFIGS[domain] = dict(config)  # 恢复
        # 提取 B 段 support gains 与 C 段 selection
        gains = []
        proposals = report.get("generation_proposals")
        if isinstance(proposals, list):
            for row in proposals:
                if isinstance(row, dict):
                    sr = row.get("support_response")
                    if isinstance(sr, dict) and isinstance(sr.get("support_gain"), (int, float)):
                        gains.append(float(sr["support_gain"]))
        selection = report.get("selection")
        delayed_gain = None
        delayed_harm = None
        if isinstance(selection, dict) and isinstance(selection.get("selection_gain"), (int, float)):
            delayed_gain = float(selection["selection_gain"])
            delayed_harm = delayed_gain < 0
        results[op] = {
            "support_gains": gains,
            "support_first_positive": next((i + 1 for i, g in enumerate(gains) if g > 0), None),
            "delayed_gain": delayed_gain,
            "delayed_harm": delayed_harm,
            "final_status": report.get("final_status"),
        }
        print(f"  {op:24s} support={[round(g, 4) for g in gains]} "
              f"delayed={delayed_gain if delayed_gain is not None else 'n/a'} "
              f"({report.get('final_status')})")

    # abstain 对照（无候选路径）
    try:
        a_report = v6.run(
            root, initial_proposer=_abstain_proposer, revision_proposer=_abstain_proposer,
            dataset_key=domain, write_report=False,
        )
    finally:
        v6.DATASET_CONFIGS[domain] = dict(config)
    results["__abstain__"] = {"final_status": a_report.get("final_status"), "delayed_gain": None}

    any_positive = any(
        r.get("delayed_gain") is not None and r["delayed_gain"] > 0
        for op, r in results.items() if not op.startswith("__")
    )
    # 用户规则：至少一个 delayed 正 → 批准；否则（全负或 B 段全失败无 delayed）→ 停止
    verdict = "PROCEED_TO_FULL_CAMPAIGN" if any_positive else "STOP_INCREASING_LLM_SAMPLES"
    print(f"\n== verdict: {verdict}")
    print("   (PROCEED: >=1 candidate delayed gain positive on non-overlapping C segment; "
          "STOP: no candidate has positive delayed gain)")

    out = root / "artifacts/functional/e2/w2_c_replay_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "w2-c-replay-zero-llm",
            "domain": domain,
            "b_support": b_support,
            "c_origin": c_origin,
            "candidates": results,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
