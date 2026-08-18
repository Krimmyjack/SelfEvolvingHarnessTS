"""工作包 2 边界检查：26 算子 × B/C 双切片确定性扫描（零 LLM，deepseek 副本）。

回答两个问题（用户决定方向）：
1. GEFCom 候选空间里是否存在"B 段正且 C 段正"的算子（可识别 headroom）？
2. Support→Delayed 符号翻转是普遍现象还是个别算子？

B/C 不重叠：C_origin = B_support + HORIZON（GEFCom：B=928、C=976）。
每个算子：固定候选注入 proposer（非 LLM）→ v6.run → B 段 support gain + C 段 delayed gain。

用法：
  python evaluation/functional/run_w2_operator_scan.py [--domain gefcom]
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
import SelfEvolvingHarnessTS.operators.registry as reg  # noqa: E402

HORIZON = 48


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


def _default_params(op: str, domain_period: int) -> dict[str, object]:
    """从 public_parameter_schema 构造最小合法参数；schema=null 用 {}。"""
    md = reg.OPERATOR_METADATA.get(op, {})
    schema = md.get("public_parameter_schema")
    if not schema:
        return {}
    props = schema.get("properties") or {}
    params: dict[str, object] = {}
    for name, spec in props.items():
        if "default" in spec:
            params[name] = spec["default"]
        elif spec.get("type") == "integer":
            params[name] = spec.get("minimum", 1)
        elif spec.get("type") == "number":
            params[name] = spec.get("minimum", 1.0) or 1.0
    for req in schema.get("required") or []:
        if req not in params:
            spec = props.get(req, {})
            if spec.get("type") == "integer":
                params[req] = spec.get("minimum", 1)
            else:
                params[req] = 1
    if "period" in params:
        params["period"] = domain_period  # 域周期覆盖
    return params


def _make_fixed_proposer(op: str, params: Mapping[str, object]):
    def proposer(payload: Mapping[str, object]) -> Mapping[str, object]:
        return {
            "decision": "PROPOSE",
            "steps": [{"op": op, "params": dict(params), "bindings": {}}],
            "requested_observations": [],
            "fallback": "IDENTITY",
        }
    return proposer


def main() -> int:
    parser = argparse.ArgumentParser(description="W2 zero-LLM operator scan (B/C dual segment)")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom")
    args = parser.parse_args()
    root = args.root.resolve()

    domain = args.domain
    config = dict(v6.DATASET_CONFIGS[domain])
    max_len = _max_len_for(root, str(config["dataset_id"]))
    period = int(config.get("period", 1))
    support_origin = int(config["support_origin"])
    shift = min(period, max_len - support_origin - 2 * HORIZON)
    if shift <= 0:
        raise SystemExit(f"{domain}: no room for non-overlapping C segment")
    b_support = support_origin + shift
    c_origin = b_support + HORIZON
    cfg = dict(config)
    cfg["support_origin"] = b_support
    cfg["selection_origin"] = c_origin
    v6.DATASET_CONFIGS[domain] = cfg
    print(f"== {domain}: B=[{b_support},{b_support + HORIZON}) C=[{c_origin},{c_origin + HORIZON})")

    # forecast 允许的算子
    forecast_ops = [
        name for name in reg.OPERATOR_NAMES
        if "forecast" in (reg.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
    ]
    print(f"== forecast-allowed operators: {len(forecast_ops)}")

    results: dict[str, Any] = {}
    for op in sorted(forecast_ops):
        params = _default_params(op, period)
        proposer = _make_fixed_proposer(op, params)
        try:
            report = v6.run(
                root, initial_proposer=proposer, revision_proposer=proposer,
                dataset_key=domain, write_report=False,
            )
        except Exception as exc:  # compile/execution 失败不阻塞扫描
            results[op] = {"status": "SKIP", "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            print(f"  {op:28s} SKIP ({type(exc).__name__})")
            continue
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
        if isinstance(selection, dict) and isinstance(selection.get("selection_gain"), (int, float)):
            delayed_gain = float(selection["selection_gain"])
        b_positive = any(g > 0 for g in gains)
        c_positive = delayed_gain is not None and delayed_gain > 0
        executed = bool(gains)  # 空 support_gains 不算有效执行（用户裁决）
        results[op] = {
            "status": "EXECUTED" if executed else "NOT_EXECUTED",
            "support_gains": gains,
            "delayed_gain": delayed_gain,
            "b_positive": b_positive,
            "c_positive": c_positive,
            "final_status": report.get("final_status"),
        }
        cls = ("B+C+" if b_positive and c_positive
               else "B+C-" if b_positive and delayed_gain is not None and not c_positive
               else "B-C+" if not b_positive and c_positive
               else "B-C-" if delayed_gain is not None
               else "B-fail" if not b_positive else "B+C?")
        print(f"  {op:28s} B={[round(g, 4) for g in gains]} C={delayed_gain if delayed_gain is not None else 'n/a':>9} -> {cls}")

    # 结论
    b_pos_c_pos = [op for op, r in results.items() if r.get("b_positive") and r.get("c_positive")]
    flipped = [
        op for op, r in results.items()
        if r.get("b_positive") and r.get("delayed_gain") is not None and not r.get("c_positive")
    ]
    attempted = len(forecast_ops)
    executed = sum(1 for r in results.values() if r.get("status") == "EXECUTED")
    not_executed = [op for op, r in results.items() if r.get("status") == "NOT_EXECUTED"]
    reached_c = sum(1 for r in results.values() if r.get("delayed_gain") is not None)
    print(f"\n== scan summary: attempted={attempted}, executed={executed}, "
          f"NOT_EXECUTED={not_executed}, reached C={reached_c}")
    print(f"== B+C+ (identifiable headroom): {len(b_pos_c_pos)} -> {b_pos_c_pos}")
    print(f"== B+C- (support-delayed flip): {len(flipped)} -> {flipped}")
    verdict = (
        "HEADROOM_EXISTS_PROCEED" if b_pos_c_pos
        else "NO_IDENTIFIABLE_HEADROOM_STOP"
    )
    print(f"== verdict: {verdict}")

    out = root / f"artifacts/functional/e2/w2_operator_scan_report_{domain}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "w2-operator-scan-zero-llm",
            "domain": domain,
            "b_support": b_support,
            "c_origin": c_origin,
            "operator_results": results,
            "b_positive_c_positive": b_pos_c_pos,
            "support_delayed_flipped": flipped,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
