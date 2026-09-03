"""BATCH_CENSUS_DEV（Wave 3，2026-08-13：全量 Exposed Evidence Census +
预声明 development block 打开——development exposure——零新 Claim）。

Part A —— EXPOSED 部分（零新评估——读已有报告）：
  - T117 四轮探测（w1_group_auto_trigger_dev_report.json）
  - E31 fresh run 探测（w1_e31_fresh_a5_two_slot_report.json）
  - E31 winsorize 仪器失效（w1_group_evidence_chain_gate1_report.json）
  - 跨独立 series 重复 failure family 检查（Task/Consumer 同——
    forecast|ridge|sMASE——按完整 workflow 指纹 × sign）

Part B —— DEVELOPMENT 部分（预声明块打开——development exposure）：
  - Series（预注册——cache 顺序，排除已冻结 p41/e31 后前 4）：
    T1, T10, T100, T101
  - Origins（预注册——与 sealed 装置同构）：600, 792, 888, 984
  - Eval 集（预注册——已冻结 p41 评估集）：T128, T129, T13, T130,
    T131, T132, T133, T134
  - 装置：H0 + SealedProbeBackend（force_pool——winsorize 先探——
    T117 同构），budget 2，allow_slow=False（只收集 factual
    Action–Response，不触发单条 Slow）
  - 每轮合法 Action-Response 写 Episode（完整 workflow 指纹 + per_view
    + origin）→ 跨 series 分组（group_first_faults, min_group=2）

Family 排序（确定性）：独立 series 数 ↓ × |harm| ↓ × 稳定性（多 origin
全负）——top ≤3 family 做共同 replacement headroom（per-episode 解析
executor——origin 跨 series 会碰撞）。

verdict（预注册）：
  CENSUS_EXPOSED_NO_CROSS_SERIES_FAMILY : 已暴露证据中无跨独立 series
    重复 family（如实——触发 development block 打开）
  DEVELOPMENT_FAMILY_FOUND             : 开发块中 ≥1 跨 series family
    （报告排序 + headroom）
  NO_INDEPENDENT_FAILURE_FAMILY        : 开发块中也无
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_batch_census_dev.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)

from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
BUDGET = 2
# 预注册 development block（运行前确定——不按 outcome 挑选）
DEV_SERIES = ("T1", "T10", "T100", "T101")
DEV_ORIGINS = (600, 792, 888, 984)
EVAL_SERIES = ("T128", "T129", "T13", "T130",
               "T131", "T132", "T133", "T134")
OPS = ("winsorize", "outlier_mad", "hampel_filter")
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_batch_census_dev_report.json"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _census_exposed(root: Path) -> dict[str, Any]:
    """Part A：EXPOSED census（读已有报告——零新评估）。family 键 =
    f"{op}|{sign}"（JSON 安全——tuple key 不可序列化）。"""
    out: dict[str, Any] = {"families": {}, "notes": []}

    def _add(op: str, series: str, origin: int, gain: float) -> None:
        sign = "NEGATIVE" if gain < -M else "POSITIVE"
        key = f"{op}|{sign}"
        fam = out["families"].setdefault(
            key, {"workflow": op, "sign": sign,
                  "series": set(), "episodes": []})
        fam["series"].add(series)
        fam["episodes"].append(
            {"series": series, "origin": origin, "gain": gain})

    # T117 四轮探测（auto-trigger dev 报告）
    gat = json.loads((root / "artifacts/functional/e2"
                      / "w1_group_auto_trigger_dev_report.json")
                     .read_text(encoding="utf-8"))
    for r in gat["rounds"]:
        for cid, gain in r.get("probes") or []:
            if gain is None:
                continue
            _add(cid.replace("cand_", ""), "T117", r["origin"], gain)
    # E31 fresh run 探测（A5 臂为主——已暴露 outcome）
    e31 = json.loads((root / "artifacts/functional/e2"
                      / "w1_e31_fresh_a5_two_slot_report.json")
                     .read_text(encoding="utf-8"))
    for arm_key in ("A5", "A3"):
        arm = e31.get("arms", {}).get(arm_key, {})
        for r in arm.get("rounds", []):
            for _op, gain in r.get("probes") or []:
                if gain is None:
                    continue
                op = _op.replace("_local", "").replace("_repair", "")
                _add(op, "T153", r["origin"], gain)
    # E31 winsorize 仪器失效（gate1 报告——census 定性记录）
    gate1 = json.loads((root / "artifacts/functional/e2"
                        / "w1_group_evidence_chain_gate1_report.json")
                       .read_text(encoding="utf-8"))
    e31w = gate1.get("checks", {}).get("e31_t153_winsorize_792", {})
    out["e31_winsorize_instrument"] = e31w
    out["notes"].append(
        "E31 winsorize instrument-invalid（scale floor）——不构成 material "
        "失败 Episode——census 按 VALIDITY 纪律排除")
    # 跨独立 series 检查
    cross = {k: v for k, v in out["families"].items()
             if len(v["series"]) >= 2}
    out["cross_series_families"] = [{"workflow": v["workflow"],
                                     "sign": v["sign"],
                                     "series": sorted(v["series"]),
                                     "episodes": v["episodes"]}
                                    for k, v in sorted(cross.items())]
    out["families"] = {k: {"workflow": v["workflow"], "sign": v["sign"],
                           "series": sorted(v["series"]),
                           "episodes": v["episodes"]}
                       for k, v in out["families"].items()}
    return out


def _dev_executor(root: Path, series: str) -> tuple[ScopeExecutor, dict]:
    roster = ([{"series_uid": series, "role": "train"}]
              + [{"series_uid": s, "role": "eval"} for s in EVAL_SERIES])
    values = {s: _load_series(root, s) for s in (series,) + EVAL_SERIES}
    return ScopeExecutor(roster, values, _config(),
                         evaluate_fn=_evaluate_kdd), values


def _family_headroom(root: Path, family: Mapping[str, Any],
                     executors: Mapping[str, ScopeExecutor]) -> dict[str, Any]:
    """top family 的共同 replacement headroom（per-episode 解析 executor
    ——origin 跨 series 碰撞——手动循环）。"""
    out: dict[str, Any] = {}
    eps = family["episodes"]
    for alt in OPS:
        per_ep = []
        all_pos = True
        for e in eps:
            sid, origin = e["series"], e["origin"]
            steps = ((alt, dict(wiring.contract_params(alt, PERIOD))),)
            rr = executors[sid].evaluate(tuple(steps), origin)
            g = (float(rr.gain) if rr.gain is not None else None)
            per_ep.append({"series": sid, "origin": origin, "gain": g})
            if g is None or g < M:
                all_pos = False
        out[alt] = {"per_episode_gains": per_ep,
                    "common_positive": all_pos}
    return out


def main() -> int:
    root = PROJECT_ROOT
    report: dict[str, Any] = {
        "experiment_id": "v1-batch-census-dev",
        "note": "Wave 3：全量 Exposed Evidence Census + 预声明 development "
                "block（development exposure——零新 Claim——不形成 "
                "跨域/迁移结论）",
        "pre_registered": {"dev_series": list(DEV_SERIES),
                           "origins": list(DEV_ORIGINS),
                           "eval_series": list(EVAL_SERIES)},
    }

    # ---- Part A：EXPOSED ----
    exposed = _census_exposed(root)
    report["exposed"] = exposed
    if exposed["cross_series_families"]:
        exposed_verdict = "EXPOSED_CROSS_SERIES_FAMILY_PRESENT"
    else:
        exposed_verdict = "CENSUS_EXPOSED_NO_CROSS_SERIES_FAMILY"
    report["exposed_verdict"] = exposed_verdict
    print(f"== exposed verdict: {exposed_verdict}")

    # ---- Part B：DEVELOPMENT block（预注册打开）----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    executors: dict[str, ScopeExecutor] = {}
    vals_by_series: dict[str, dict] = {}
    methods: dict[str, TTHAMethod] = {}
    skipped: list[dict[str, Any]] = []
    for sid in DEV_SERIES:
        series_arr = _load_series(root, sid)
        if len(series_arr) < DEV_ORIGINS[-1] + HORIZON:
            skipped.append({"series": sid, "reason": "too_short",
                            "length": int(len(series_arr))})
            continue
        ex, vals = _dev_executor(root, sid)
        executors[sid] = ex
        vals_by_series[sid] = vals
        core = sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=OPS,
                                      max_propose_candidates=3,
                                      force_pool=True),
            LocalPublicToolGateway(series_arr[:DEV_ORIGINS[0]],
                                   task_kind="forecast"))
        methods[sid] = TTHAMethod(sealed.TTHAFastAgent(core), h0, ())
    report["development_skipped"] = skipped
    all_episodes = []
    rounds_log: dict[str, list[dict[str, Any]]] = {}
    for sid in DEV_SERIES:
        if sid not in methods:
            continue
        rounds_log[sid] = []
        for origin in DEV_ORIGINS:
            series_arr = _load_series(root, sid)
            core = methods[sid].fast_agent.core
            core.backend = sealed.SealedProbeBackend(
                explore=True, operators=OPS, max_propose_candidates=3,
                force_pool=True)
            r = run_online_round(
                methods[sid], executors[sid],
                _request(series_arr, vals_by_series[sid], origin),
                vals_by_series[sid],
                origin=origin, slow_agent=None, controller=None, store=None,
                card_builder=lambda e: {"pattern_id": "x",
                                        "observable_signature":
                                            {"task_kind": "forecast"}},
                round_name=f"census_{sid}_{origin}", budget=BUDGET,
                allow_slow=False, domain=f"kdd2018_dev_{sid}",
                period=PERIOD,
                fast_features=dict(extract_public_features(
                    series_arr[:origin], task_kind="forecast")),
                allow_fast_skill=False, runtime_prior_slot=False,
                allow_group_slow=False)
            rounds_log[sid].append({
                "origin": origin,
                "probes": [(p["candidate_id"], p.get("gain"))
                           for p in r.actual_probed_programs],
                "episodes_written": list(r.episode_ids)})
            print(f"== dev {sid}@{origin}: probes="
                  f"{[(p['candidate_id'], p.get('gain')) for p in r.actual_probed_programs]}")
        eps = list(methods[sid]._experience_episodes)
        all_episodes.extend(eps)
        print(f"== dev {sid}: episodes={[e.episode_id for e in eps]}")
    report["development_rounds"] = rounds_log

    # 跨 series 分组（完整 workflow 指纹 × sign）
    groups = group_first_faults(all_episodes, min_group=2)
    families = []
    for g in groups:
        eps = g["episodes"]
        series = sorted({e.episode_id.split("_dev_")[1].split("_")[0]
                         for e in eps}) if eps else []
        gains = [(float((e.support_response or {}).get("gain") or 0.0),
                  e) for e in eps]
        fam = {
            "workflow": g["workflow"],
            "sign": g["sign"],
            "n_episodes": len(eps),
            "independent_series": series,
            "n_series": len(series),
            "origins": sorted({int((e.context_summary or {})
                                  .get("support_origin") or 0) for e in eps}),
            "episodes": [{"series": e.episode_id.split("_dev_")[1].split("_")[0],
                          "origin": int((e.context_summary or {})
                                        .get("support_origin") or 0),
                          "gain": float((e.support_response or {})
                                        .get("gain") or 0.0)}
                         for _, e in sorted(gains)],
            "min_gain": min(g[0] for g in gains),
            "max_gain": max(g[0] for g in gains),
        }
        families.append(fam)
    # 排序：独立 series 数 ↓ × |min harm| ↓ × origin 数 ↓
    families.sort(key=lambda f: (-f["n_series"], f["min_gain"],
                                 -len(f["origins"])))
    for f in families[:3]:
        f["replacement_headroom"] = _family_headroom(root, f, executors)
    report["development_families"] = families

    if families and any(f["n_series"] >= 2 for f in families):
        verdict = "DEVELOPMENT_FAMILY_FOUND"
    else:
        verdict = "NO_INDEPENDENT_FAILURE_FAMILY"
    report["verdict"] = verdict
    print(f"== development families: "
          + json.dumps(families, ensure_ascii=False, default=str))
    print(f"== verdict: {verdict}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
