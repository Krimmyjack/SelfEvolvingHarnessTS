"""run_p5a3_final.py — P5-A.3 architecture-complete confirmatory（prereg_p5a3_final.md 冻结协议）。

三轮谱系的终审：P5-A（破损接口）→ P5-A.2（接口修复受控归因）→ **P5-A.3 =
完整信息面（packet v2 连续证据，R1 兑现）+ ReadinessPlan→deterministic compiler**。
真实 Monash × seeds 80–99 一次性；true 判官验收；四指标分解（semantic/compliance/
selection/harness-benefit）；selection regret = 决策后离线评估全部候选（只入报告不回流）。

开跑硬前置：`--preflight` 通过（新 key 注入 + n_api/n_hit 分离核验）后才允许 `--backend llm`
正式消耗 seeds。stub 模式仅供机械回归（不消耗 confirmatory 语义）。

命令：
  preflight:  python -m SelfEvolvingHarnessTS.run_p5a3_final --preflight
  正式:       python -m SelfEvolvingHarnessTS.run_p5a3_final --backend llm
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data.load_real import FORECAST_PRESETS, _forecast_from_signal, load_signals
from .e32_policy import P_FEATS
from .harness.layers import minimal_l2
from .policy.action_spec import action_menu_v1
from .policy.code_agent_composer import CodeAgentComposer
from .policy.evidence_packet import build_evidence_packet_v2
from .policy.program_edit import is_novel_v1, spec_v1_from_dict
from .policy.readiness_plan import PlanComposer
from .policy.seed_programs import seed_skill_cards
from .policy.task_spec import forecast_task_spec_v1
from .readiness_gym import ReadinessGym
from .run_p2_motivation import seasonal_naive_nrmse
from .run_p5_identity_gate import _DET_LADDER, _DEV_MEMORY, _CountingClient, sample_random_program
from .sandbox.executor import run_pipeline

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P5A3Final"
_QUADRANT_RECORDS = Path(__file__).resolve().parent / "results" / "Stage2" / "P5Quadrant" / "records.jsonl"
ARMS = ("frozen", "random_valid", "det_search", "pv2_direct",
        "pv2_plan_compiler", "plan_compiler_no_ce", "plan_compiler_no_skills")
PRIMARY_ARM, BASELINE_ARM = "pv2_plan_compiler", "det_search"
EPSILON, DELTA_SAFE, K_NOVEL_DISTINCT, BUDGET = 0.02, 0.05, 3, 3
FINAL_SEEDS = tuple(range(80, 100))


def _digest(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _episodes(seeds: Sequence[int], n_per_seed: int) -> List[Dict[str, Any]]:
    signals = load_signals()
    rows: List[Dict[str, Any]] = []
    for s in seeds:
        for j in range(int(n_per_seed)):
            idx = (int(s) - int(seeds[0])) * n_per_seed + j
            sig = signals[idx % len(signals)]
            preset = FORECAST_PRESETS[(idx // len(signals)) % len(FORECAST_PRESETS)]
            raw = _forecast_from_signal(sig, preset, seed=_digest(s, j, sig.config, sig.item_id, preset))
            rows.append({
                "uid": f"a3_{s}_{j}",
                "cell": f"{sig.config}|{preset}",
                "series_family": sig.config,                 # 外评二审命名修正
                "pattern_preset": preset,
                "series_uid": raw.series_uid,
                "group_seed": int(s),
                "x": np.asarray(raw.history, dtype=float),
                "future_clean": np.asarray(raw.future, dtype=float),   # 真实观测未来
                "period": int(raw.period),
                "labels": np.zeros(0, dtype=bool),           # forecast-only：不使用
            })
    return rows


# ── continuous evidence（P5-B 真实 records 聚合；同 preset、排除同 series_uid=LODO 防泄漏）──
def _load_quadrant_rows() -> List[Dict[str, Any]]:
    if not _QUADRANT_RECORDS.exists():
        return []
    return [json.loads(l) for l in _QUADRANT_RECORDS.read_text(encoding="utf-8").strip().splitlines()]


def _ce_for(quadrant_rows: List[Dict[str, Any]], preset: str, exclude_series_uid: str
            ) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    src = [r for r in quadrant_rows
           if r.get("pattern") == preset and r.get("series_uid") != exclude_series_uid]
    if not src:
        return out
    # P5Quadrant records 行含 regret/recipe；per-action deltas 在原始 episode 构建时记录于
    # oracle_delta/oracle_action + regret 字段——此处用可得的 per-quadrant recipe 统计的替代面：
    # 直接聚合 oracle_delta（各动作上界）与 same-pattern recipe regret 的连续统计。
    oracle = [float(r["oracle_delta"]) for r in src]
    out["oracle_upper_bound"] = {"mean": float(np.mean(oracle)),
                                 "q25": float(np.percentile(oracle, 25)),
                                 "q75": float(np.percentile(oracle, 75)),
                                 "n": len(oracle)}
    for quadrant in ("same_domain_same_pattern", "diff_domain_same_pattern"):
        vals = [float(r["regret"][quadrant]) for r in src
                if r.get("regret", {}).get(quadrant) is not None]
        if vals:
            out[f"regret_{quadrant}"] = {"mean": float(np.mean(vals)),
                                         "q25": float(np.percentile(vals, 25)),
                                         "q75": float(np.percentile(vals, 75)),
                                         "n": len(vals)}
    oracle_actions: Dict[str, int] = {}
    for r in src:
        oracle_actions[str(r["oracle_action"])] = oracle_actions.get(str(r["oracle_action"]), 0) + 1
    for act, n in sorted(oracle_actions.items()):
        out[f"oracle_share_{act}"] = {"share": n / len(src), "n": n}
    return out


def _packet(row: Mapping[str, Any], fingerprint: Mapping[str, Any], menu,
            quadrant_rows: List[Dict[str, Any]], *, skills: bool, ce: bool) -> Dict[str, Any]:
    record = {"uid": row["uid"], "cell": row["cell"],
              "snr": float(fingerprint["snr"]), "miss_rate": float(fingerprint["missing_rate"]),
              "X_p": [float(fingerprint["struct_feats"].get(k, 0.0)) for k in P_FEATS]}
    return build_evidence_packet_v2(
        record,
        skills=([c for c in seed_skill_cards() if c["task_scope"] == "forecast"] if skills else []),
        memory_rows=_DEV_MEMORY,
        action_menu_meta=menu.to_dict(),
        task_spec=forecast_task_spec_v1(horizon=int(row["period"]),
                                        downstream_model_class="seasonal_naive_real_future"),
        continuous_evidence=(_ce_for(quadrant_rows, str(row["pattern_preset"]),
                                     str(row["series_uid"])) if ce else None),
        trace_summaries=[],
    )


def _offline_true_delta(prog: Mapping[str, Any], row: Mapping[str, Any],
                        defaults: Mapping[str, Mapping]) -> Optional[float]:
    """selection-regret 用：离线评估候选 true delta（只入报告，不回流选择器）。"""
    try:
        spec = spec_v1_from_dict(prog)
    except ValueError:
        return None
    from .policy.program_edit import _resolved_steps_v1
    x = np.asarray(row["x"], dtype=float)
    result = run_pipeline([(op, dict(p)) for op, p in _resolved_steps_v1(spec, defaults)], x)
    if not result.ok or result.artifact is None or result.artifact.shape != x.shape:
        return None
    raw_s = seasonal_naive_nrmse(x, row["future_clean"], int(row["period"]))
    art_s = seasonal_naive_nrmse(np.asarray(result.artifact, float), row["future_clean"], int(row["period"]))
    return float(raw_s - art_s)


def _arm_candidates(arm: str, row, fingerprint, menu, quadrant_rows, rng,
                    client, backend: str) -> Tuple[List[Tuple[Optional[dict], str, Dict[str, Any]]], Optional[dict]]:
    """→ ([(prog|None, invalid_reason, meta)], packet|None)。meta 含 repair_used/compile_info。"""
    if arm == "frozen":
        return [], None
    if arm == "random_valid":
        return [(sample_random_program(rng), "", {}) for _ in range(BUDGET)], None
    if arm == "det_search":
        return [(dict(c), "", {}) for c in _DET_LADDER], None
    skills = arm != "plan_compiler_no_skills"
    ce = arm != "plan_compiler_no_ce"
    packet = _packet(row, fingerprint, menu, quadrant_rows, skills=skills, ce=ce)
    out: List[Tuple[Optional[dict], str, Dict[str, Any]]] = []
    for nonce in range(BUDGET):
        if arm == "pv2_direct":
            o = CodeAgentComposer(backend=backend, llm=client, nonce=nonce,
                                  repair_retries=1).compose(packet)
            meta = {"repair_used": o.api_calls > 1, "api_calls": o.api_calls}
            out.append((dict(o.candidate.program_spec) if o.candidate else None,
                        o.invalid_reason, meta))
        else:                                                 # plan 系臂
            o = PlanComposer(backend=backend, llm=client, nonce=nonce,
                             repair_retries=1).compose(packet)
            meta = {"repair_used": o.api_calls > 1, "api_calls": o.api_calls,
                    "compile_info": o.compile_info}
            out.append((dict(o.candidate.program_spec) if o.candidate else None,
                        o.invalid_reason, meta))
    return out, packet


def run_p5a3(seeds: Sequence[int] = FINAL_SEEDS, n_per_seed: int = 3,
             out_dir: Path | str | None = None, backend: str = "stub",
             bootstrap_b: int = 2000, max_api_calls: int = 1500) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"
    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").strip().splitlines():
            row = json.loads(line)
            done[(row["arm"], row["uid"])] = row

    client = None
    if backend == "llm":
        from .llm.client import LLMClient
        client = _CountingClient(LLMClient(model="flash", temperature=0.7,
                                           cache_name="p5a3_final", timeout=120,
                                           max_api_calls=max_api_calls))
    menu = action_menu_v1()
    defaults = minimal_l2().operator_defaults
    quadrant_rows = _load_quadrant_rows()
    rows = _episodes(seeds, n_per_seed)
    resumed = 0

    with records_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            gym = ReadinessGym([row], task="forecast", budget=BUDGET)
            for arm in ARMS:
                key = (arm, row["uid"])
                if key in done:
                    resumed += 1
                    continue
                obs = gym.reset(0)
                fingerprint = gym._fingerprint(0)
                rng = np.random.default_rng((int(row["group_seed"]),
                                             _digest(row["uid"]) & 0xFFFF, ARMS.index(arm)))
                candidates, _packet_used = _arm_candidates(arm, row, fingerprint, menu,
                                                           quadrant_rows, rng, client, backend)
                evals: List[Dict[str, Any]] = []
                for prog, reason, meta in candidates:
                    if prog is None:
                        evals.append({"ok": False, "reason": f"itt_noop:{reason}",
                                      "proxy_delta": None, "true_delta_offline": None,
                                      "program": None, **meta})
                        continue
                    obs, _ = gym.step({"op": "proxy_eval", "program_spec": prog})
                    entry = obs["evals"][-1]
                    evals.append({"ok": entry.get("ok"), "reason": entry.get("reason", ""),
                                  "proxy_delta": entry.get("proxy_delta"),
                                  "true_delta_offline": _offline_true_delta(prog, row, defaults),
                                  "program": prog, **meta})
                valid = [e for e in evals if e["ok"] and e["proxy_delta"] is not None]
                if not valid:
                    gym.step({"op": "abstain"})
                else:
                    best = max(valid, key=lambda e: e["proxy_delta"])
                    gym.step({"op": "finalize", "program_spec": best["program"]})
                result = gym.result(0)
                offline = [e["true_delta_offline"] for e in valid
                           if e["true_delta_offline"] is not None]
                regret = (float(max(offline) - result["true_delta"])
                          if offline and result["final_kind"] == "program" else None)
                novel = False
                if result["final_kind"] == "program" and valid:
                    final = max(valid, key=lambda e: e["proxy_delta"])["program"]
                    try:
                        novel = is_novel_v1(spec_v1_from_dict(final))
                    except ValueError:
                        novel = False
                rec = {"arm": arm, "uid": row["uid"], "group_seed": int(row["group_seed"]),
                       "series_family": row["series_family"], "pattern_preset": row["pattern_preset"],
                       "final_kind": result["final_kind"], "true_delta": result["true_delta"],
                       "final_program_sha": result["final_program_sha"],
                       "novel_vs_menu": bool(novel), "selection_regret": regret,
                       "candidates": [{k: v for k, v in e.items() if k != "program"}
                                      for e in evals]}
                done[key] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    # ── 汇总（四指标）────────────────────────────────────────────────────────
    stats_rng = np.random.default_rng(20260710 + 3)

    def _arm_rows(arm):
        return [r for (a, _), r in done.items() if a == arm]

    def _grouped_ci(diffs_by_group):
        groups = sorted(diffs_by_group)
        if not groups:
            return (0.0, 0.0)
        means = np.empty(bootstrap_b)
        for i in range(bootstrap_b):
            picked = stats_rng.choice(groups, size=len(groups), replace=True)
            means[i] = float(np.mean([v for g in picked for v in diffs_by_group[g]]))
        lo, hi = np.percentile(means, (5.0, 95.0))
        return float(lo), float(hi)

    arms_report = {}
    for arm in ARMS:
        rs = _arm_rows(arm)
        regrets = [r["selection_regret"] for r in rs if r["selection_regret"] is not None]
        cands = [c for r in rs for c in r["candidates"]]
        arms_report[arm] = {
            "n_episodes": len(rs),
            "harness_benefit_mean_true_delta": float(np.mean([r["true_delta"] for r in rs])) if rs else 0.0,
            "compliance_valid_rate": (float(np.mean([bool(c["ok"]) for c in cands])) if cands else None),
            "compliance_repair_used": int(sum(1 for c in cands if c.get("repair_used"))),
            "selection_mean_regret": (float(np.mean(regrets)) if regrets else None),
            "abstain_rate": float(np.mean([r["final_kind"] != "program" for r in rs])) if rs else None,
        }
    # semantic（plan 臂）：编译台账聚合
    for arm in ("pv2_plan_compiler", "plan_compiler_no_ce", "plan_compiler_no_skills"):
        cands = [c for r in _arm_rows(arm) for c in r["candidates"] if c.get("compile_info")]
        if cands:
            arms_report[arm]["semantic_guard_drop_rate"] = float(np.mean(
                [1.0 if c["compile_info"]["dropped_guards"] else 0.0 for c in cands]))
            arms_report[arm]["semantic_step_drop_rate"] = float(np.mean(
                [1.0 if c["compile_info"]["dropped_steps"] else 0.0 for c in cands]))

    diffs_by_group: Dict[int, List[float]] = {}
    for (a, uid), r in done.items():
        if a != PRIMARY_ARM:
            continue
        other = done.get((BASELINE_ARM, uid))
        if other is not None:
            diffs_by_group.setdefault(int(r["group_seed"]), []).append(
                float(r["true_delta"]) - float(other["true_delta"]))
    all_diffs = [v for vs in diffs_by_group.values() for v in vs]
    diff_mean = float(np.mean(all_diffs)) if all_diffs else 0.0
    ci = _grouped_ci(diffs_by_group)

    cell_lcb: Dict[str, float] = {}
    for preset in FORECAST_PRESETS:
        groups: Dict[int, List[float]] = {}
        for (a, uid), r in done.items():
            if a != PRIMARY_ARM or r["pattern_preset"] != preset:
                continue
            other = done.get((BASELINE_ARM, uid))
            if other is not None:
                groups.setdefault(int(r["group_seed"]), []).append(
                    float(r["true_delta"]) - float(other["true_delta"]))
        if groups:
            cell_lcb[preset] = _grouped_ci(groups)[0]

    novel_shas = {r["final_program_sha"] for r in _arm_rows(PRIMARY_ARM)
                  if r["novel_vs_menu"] and float(r["true_delta"]) >= EPSILON}
    criteria = {
        "utility_vs_det": bool(diff_mean >= EPSILON and ci[0] > 0.0),
        "worst_group": bool(cell_lcb and all(v >= -DELTA_SAFE for v in cell_lcb.values())),
        "novel_effective_distinct": bool(len(novel_shas) >= K_NOVEL_DISTINCT),
        "cost_disclosed": True,
    }
    claim = ("llm_driven_harness_evolution" if all(criteria.values())
             else "self_updating_deterministic_with_llm_novelty_supplier")

    report = {
        "phase": "P5A3_architecture_complete_confirmatory",
        "protocol": "prereg_p5a3_final.md frozen; real Monash; true judges; packet v2 as contract source",
        "n_episodes": len(rows), "seeds": [int(s) for s in seeds], "resumed_episodes": resumed,
        "arms": arms_report,
        "primary_comparison": {"primary_arm": PRIMARY_ARM, "baseline_arm": BASELINE_ARM,
                               "diff_mean": diff_mean, "epsilon": EPSILON,
                               "ci90": [ci[0], ci[1]], "grouped_by": "seed"},
        "per_preset_diff_lcb": cell_lcb,
        "novel_effective_distinct_sha": len(novel_shas),
        "headline_criteria": criteria,
        "claim_branch": claim,
        "cost_ledger": {"backend": backend,
                        "composer_calls": int(client.calls) if client else 0,
                        "llm_wall_seconds": round(float(client.wall), 1) if client else 0.0,
                        "client_stats": (client._client.stats() if client is not None
                                         and hasattr(client._client, "stats") else None)},
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "generated_by": "run_p5a3_final",
        "prereg": "results/Stage2/prereg_p5a3_final.md",
        "epsilon": EPSILON, "delta_safe": DELTA_SAFE,
        "k_novel_distinct": K_NOVEL_DISTINCT, "budget": BUDGET,
        "seeds": [int(s) for s in seeds], "n_per_seed": int(n_per_seed),
        "backend": backend, "bootstrap_b": int(bootstrap_b),
        "data": "real Monash via data/load_real (12 signals x FORECAST_PRESETS)",
        "continuous_evidence": "P5Quadrant records, same-preset, exclude same series_uid (LODO)",
        "naming": "series_family/pattern_preset（外评二审命名修正；不再用 anomaly|* cell）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def preflight() -> bool:
    """开跑硬前置：新 key 注入 + n_api/n_hit 分离 + 计数不混淆（prereg 最后一节）。"""
    import os
    import uuid
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(json.dumps({"preflight": "FAIL", "reason": "DEEPSEEK_API_KEY 未设置"
                          "（旧 key 须平台撤销后注入新 key）"}, ensure_ascii=False))
        return False
    from .llm.client import LLMClient
    c = LLMClient(model="flash", temperature=0.0, cache_name=f"p5a3_preflight_{uuid.uuid4().hex[:6]}",
                  timeout=60, max_retries=1, max_api_calls=2)
    out1 = c("You are a probe.", "Reply with exactly: READY")
    out2 = c("You are a probe.", "Reply with exactly: READY")
    ok = ("READY" in out1.upper() and out2 == out1 and c.n_api == 1 and c.n_hit == 1)
    print(json.dumps({"preflight": "PASS" if ok else "FAIL",
                      "n_api": c.n_api, "n_hit": c.n_hit, "stats": c.stats()}, ensure_ascii=False))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="P5-A.3 final trial runner")
    parser.add_argument("--backend", type=str, default="stub", choices=["stub", "llm"])
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--n-per-seed", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        raise SystemExit(0 if preflight() else 1)
    if args.backend == "llm" and not preflight():
        raise SystemExit("preflight 未过：不消耗 seeds 80–99（prereg 硬前置）")
    report = run_p5a3(n_per_seed=args.n_per_seed, out_dir=args.out_dir, backend=args.backend)
    print(json.dumps({"claim_branch": report["claim_branch"],
                      "diff_mean": report["primary_comparison"]["diff_mean"],
                      "ci90": report["primary_comparison"]["ci90"],
                      "criteria": report["headline_criteria"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
