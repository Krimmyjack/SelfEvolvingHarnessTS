"""V1 跨域 Memory 覆盖扫描（零 Target outcome，2026-08-08）。

审查裁决：有边界的现有 Memory 覆盖扫描——
  GEFCom → NOAA / NN5
  NOAA → GEFCom / NN5
  NN5 → GEFCom / NOAA
只用已有 Episode（确定性重放链记忆）与部署可见 Context（window_context），
不打开新 Target outcome、不调 δ。目的不是测试矩阵，而是找出是否存在
至少一条真实跨域匹配路径。

结果分两种（裁决）：
  - 找到 Context match + Target headroom：立即只跑这一对 A5/A3；
  - 所有方向都无匹配：可信定位为当前 14 维指纹的跨域覆盖/尺度问题
    （CROSS_DOMAIN_COVERAGE_GAP），再批准一次特征层修改——不能继续靠
    换数据集碰运气。

headroom 依据（历史开发材料，非新 outcome）：
  gefcom/nn5: w1_target_local_loop_3rounds_report_{domain}.json 的探测 gains；
  noaa: w1_a5_vs_a3_report_noaa.json 的 A3 探测 gains（denoise_savgol +0.0243）。

用法：
  python evaluation/functional/run_v1_memory_coverage_scan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_target_local_loop as loop  # noqa: E402
import signed_radius as resolver  # noqa: E402

HORIZON = 48
MAX_TARGET_PROBES = 2
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_memory_coverage_scan_report.json")

# (source_support, source_delayed), 链切片, period, 目标候选 origin（部署可见决策点）
DOMAINS: dict[str, dict[str, Any]] = {
    "gefcom": {
        "source": (640, 688),
        "chain": [(736, 784), (832, 880), (928, 976)],
        "period": 24,
        "target_origins": [736, 784, 832, 880, 928, 976],
    },
    "nn5": {
        "source": (536, 584),
        "chain": [(632, 680), (728, None)],
        "period": 7,
        "target_origins": [632, 680, 728],
    },
    "noaa": {
        "source": (832, 880),
        "chain": [],
        "period": 24,
        "target_origins": [832, 880, 928, 976],
    },
}

HISTORY_REPORTS: dict[str, list[Path]] = {
    "gefcom": [Path("artifacts/functional/e2/w1_target_local_loop_3rounds_report_gefcom.json")],
    "nn5": [Path("artifacts/functional/e2/w1_target_local_loop_3rounds_report_nn5.json")],
    "noaa": [Path("artifacts/functional/e2/w1_a5_vs_a3_report_noaa.json")],
}


def build_source_memory(domain: str, config: Mapping[str, object], roster: Any,
                        values: Mapping[str, Any]) -> list[Any]:
    period = int(config.get("period", 1))
    src = DOMAINS[domain]["source"]
    baseline_cache: dict[int, float] = {}
    episodes, _ = v1.build_source_memory(
        domain=domain, roster=roster, values=values, config=config,
        operators=sorted(n for n in v6.OPERATOR_NAMES
                         if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                         and n not in core.CTS_EXCLUDED),
        source_support_origin=src[0], source_delayed_origin=src[1],
        baseline_cache=baseline_cache,
        context_fn=lambda o: resolver.window_context(values, o, period),
    )
    return episodes


def build_chain_memory(domain: str, config: Mapping[str, object], roster: Any,
                       values: Mapping[str, Any], source: Sequence[Any]) -> list[Any]:
    """链探测重放（确定性，结果与冻结链报告一致）——gefcom/nn5 的本地 Episode。"""
    from run_w2_operator_scan import _default_params
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    operators = sorted(n for n in v6.OPERATOR_NAMES
                       if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                       and n not in core.CTS_EXCLUDED)

    def probe_at(order: Sequence[str], origin: int) -> dict[str, Any]:
        gains: list[float] = []
        probed: list[str] = []
        for op in order:
            if len(probed) >= MAX_TARGET_PROBES:
                break
            compiled = v1.make_compiled(op, _default_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            if g is None:
                continue
            probed.append(op)
            gains.append(g)
            if g >= core.MATERIAL_THRESHOLD:
                break
        return {"probe_order": probed, "support_gains": gains}

    local: list[Any] = []
    for ts, td in DOMAINS[domain]["chain"]:
        f_support = resolver.window_context(values, ts, period)
        order, _signed = resolver.resolve_order(
            query_context=f_support, episodes=list(source) + local,
            operators=operators, material_threshold=core.MATERIAL_THRESHOLD)
        r = probe_at(order, ts)
        start = len(local)
        for op, g in zip(r["probe_order"], r["support_gains"]):
            local.append(loop.write_target_episode(
                domain=domain, op=op,
                program_steps=[{"op": op, "params": dict(_default_params(op, period))}],
                support_gain=g, delayed_gain=None, support_context=f_support))
        if not r["probe_order"]:
            local.append(loop.write_abstain_episode(domain=domain, reason="no_valid_plan"))
        if td is not None:
            f_delayed = resolver.window_context(values, td, period)
            new_local = []
            for i, ep in enumerate(local):
                if i < start or ep.workflow_signature == "identity":
                    new_local.append(ep)
                    continue
                compiled = loop.compiled_from_episode(ep, period)
                dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
                new_local.append(loop.update_delayed_status(ep, dg, delayed_context=f_delayed)
                                 if dg is not None else ep)
            local[:] = new_local
    return local


def target_headroom(domain: str) -> list[dict[str, Any]]:
    """目标域历史开发材料中的合法正收益候选（非新 outcome）。"""
    evidence: list[dict[str, Any]] = []
    for rel in HISTORY_REPORTS[domain]:
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        hist = json.loads(p.read_text(encoding="utf-8"))
        gains: list[tuple[str, float]] = []
        for arm_key in ("a3", "a5"):
            arm = hist.get(arm_key)
            if isinstance(arm, Mapping):
                for op, g in zip(arm.get("probe_order") or [], arm.get("support_gains") or []):
                    if isinstance(g, (int, float)) and g >= core.MATERIAL_THRESHOLD:
                        gains.append((op, float(g)))
        for rd in hist.get("rounds") or []:
            for arm_key in ("a3", "a5"):
                arm = rd.get(arm_key)
                if isinstance(arm, Mapping):
                    for op, g in zip(arm.get("probe_order") or [], arm.get("support_gains") or []):
                        if isinstance(g, (int, float)) and g >= core.MATERIAL_THRESHOLD:
                            gains.append((op, float(g)))
        seen: set[str] = set()
        for op, g in gains:
            if op not in seen:
                seen.add(op)
                evidence.append({"operator": op, "support_gain": g,
                                 "source": str(rel)})
    return evidence


def main() -> int:
    root = PROJECT_ROOT
    # 三域数据 + 记忆（确定性重放，无新 Target outcome）
    memories: dict[str, list[Any]] = {}
    datas: dict[str, Any] = {}
    for domain in DOMAINS:
        config = dict(v6.DATASET_CONFIGS[domain])
        roster, values = v6._fixed_roster(root, config)
        datas[domain] = (config, roster, values)
        src = build_source_memory(domain, config, roster, values)
        local = build_chain_memory(domain, config, roster, values, src)
        memories[domain] = list(src) + list(local)
        print(f"== {domain}: memory={len(memories[domain])} "
              f"(source={len(src)} local={len(local)})")

    # 6 方向扫描（零 Target outcome：只用部署可见 Context + 半径判定）
    results: list[dict[str, Any]] = []
    matched_pairs: list[dict[str, Any]] = []
    for src_domain, memory in memories.items():
        for tgt_domain in DOMAINS:
            if src_domain == tgt_domain:
                continue
            tcfg, _trost, tvalues = datas[tgt_domain]
            tperiod = int(tcfg.get("period", 1))
            matched_origins: list[int] = []
            per_origin: list[dict[str, Any]] = []
            for origin in DOMAINS[tgt_domain]["target_origins"]:
                q = resolver.window_context(tvalues, origin, tperiod)
                _order, signed = resolver.resolve_order(
                    query_context=q, episodes=memory,
                    operators=sorted(n for n in v6.OPERATOR_NAMES
                                     if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                                     and n not in core.CTS_EXCLUDED),
                    material_threshold=core.MATERIAL_THRESHOLD)
                counts = signed["summary"]["verdict_counts"]
                matched = counts[resolver.POSITIVE_PRIOR] + counts[resolver.CONFLICT] \
                    + counts[resolver.RISK_PRIOR]
                per_origin.append({"origin": origin, "matched": matched,
                                   "counts": counts})
                if matched > 0:
                    matched_origins.append(origin)
            headroom = target_headroom(tgt_domain)
            direction = {
                "source": src_domain, "target": tgt_domain,
                "memory_count": len(memory),
                "per_origin": per_origin,
                "matched_origins": matched_origins,
                "context_match": len(matched_origins) > 0,
                "target_headroom": headroom,
            }
            results.append(direction)
            if direction["context_match"] and headroom:
                matched_pairs.append(direction)
            print(f"  {src_domain} -> {tgt_domain}: match={matched_origins} "
                  f"headroom={[e['operator'] for e in headroom]}")

    if matched_pairs:
        verdict = "CROSS_DOMAIN_PAIR_FOUND"
        pairs_str = ", ".join(f"{p['source']}->{p['target']}" for p in matched_pairs)
        print(f"\n== matched pair(s): {pairs_str}")
    else:
        verdict = "CROSS_DOMAIN_COVERAGE_GAP"
        print("\n== all 6 directions: no context match (or no headroom)")

    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-memory-coverage-scan",
            "directions": results,
            "matched_pairs": [{"source": p["source"], "target": p["target"]}
                              for p in matched_pairs],
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
