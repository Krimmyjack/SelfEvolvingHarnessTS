"""run_p5_identity_gate.py — P5-A code-agent identity gate（正式判决；prereg §5/§5.0 冻结协议）。

六臂同预算（B=3 候选/episode，gym proxy 只作搜索信号，**验收只认 true 判官**）：
  frozen         abstain（true=0）
  random_valid   3 个 grammar 均匀采样，按 proxy 选优
  det_search     dev 冻结优先梯 [winsor+savgol, winsor+median9, median9]（P3 dev top-3，
                 含剂量机器 = R2 反稻草人），按 proxy 选优
  ca_plain       cached-DeepSeek，gym observation 为唯一输入面（3 nonce 采样按 proxy 选优）
  ca_skills      + seed skill bank v1 卡片（**先验主臂**，prereg §5.0）
  ca_skills_memory + dev 冻结 utility/risk memory 摘要
ITT：invalid/empty LLM 输出消耗候选预算、计入台账。checkpoint/resume（A-36 教训）。

headline 判据（prereg §5，全满足才 LLM headline）：
  ① ca_skills − det_search ≥ ε=0.02 且 grouped bootstrap CI90 不跨 0（group=生成 seed）
  ② worst-group（per-cell 配对差）LCB ≥ −δ_safe=0.05
  ③ 有效新颖编辑 ≥ K=3（is_novel_v1 ∧ episode true_delta ≥ ε）
  ④ API/成本全披露
任一不满足 → claim 分支 = self-updating deterministic with LLM-optional（合法结局）。

命令（正式，一次性）：
  D:\\Anaconda_envs\\envs\\project\\python.exe -m SelfEvolvingHarnessTS.run_p5_identity_gate --backend llm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .evaluators.anomaly_rig import make_anomaly_slice
from .policy.code_agent_composer import CodeAgentComposer
from .policy.program_edit import is_novel_v1, spec_v1_from_dict
from .policy.seed_programs import seed_skill_cards
from .readiness_gym import ReadinessGym

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "P5IdentityGate"
ARMS = ("frozen", "random_valid", "det_search", "ca_plain", "ca_skills", "ca_skills_memory")
EPSILON = 0.02          # prereg §3 冻结
DELTA_SAFE = 0.05       # prereg §5.0 冻结
K_NOVEL = 3             # prereg §5.0 冻结
BUDGET = 3              # 候选预算/臂/episode（prereg §5.0）
CONFIRMATORY_SEEDS = tuple(range(40, 60))

# dev 冻结优先梯（P3 dev slice top-3 均值：winsor+savgol +0.970 / winsor+median9 +0.919 / median9 +0.918）
_DET_LADDER: List[Dict[str, Any]] = [
    {"grammar": "v1", "steps": [["impute_linear", {}], ["winsorize", {}], ["denoise_savgol", {}]],
     "scope": ["*"], "task_type": "forecast", "pattern_guard": [],
     "risk_budget_beta": 0.3, "fallback": "v_impute_linear"},
    {"grammar": "v1", "steps": [["impute_linear", {}], ["winsorize", {}], ["denoise_median", {"window": 9}]],
     "scope": ["*"], "task_type": "forecast", "pattern_guard": [],
     "risk_budget_beta": 0.3, "fallback": "v_impute_linear"},
    {"grammar": "v1", "steps": [["impute_linear", {}], ["denoise_median", {"window": 9}]],
     "scope": ["*"], "task_type": "forecast", "pattern_guard": [],
     "risk_budget_beta": 0.3, "fallback": "v_impute_linear"},
]

# dev 冻结 memory 摘要（P2/P3/P4 dev 证据；连续数值口径 R1）
_DEV_MEMORY = {
    "utility_memory": [
        {"pattern_region": "snr_low", "program": {"steps": [["impute_linear", {}], ["denoise_median", {"window": 9}]]},
         "utility_delta_vs_raw": 0.918, "support_n": 60, "source": "dev_slice_P3"},
        {"pattern_region": "any", "program": {"steps": [["impute_linear", {}], ["winsorize", {}], ["denoise_savgol", {}]]},
         "utility_delta_vs_raw": 0.970, "support_n": 60, "source": "dev_slice_P3"},
    ],
    "risk_memory": [
        {"action_id": "f0_median_w25", "harm_delta_vs_raw": 0.64, "role": "warn",
         "pattern_region": "seasonal", "failure_signature": "window≈period 抹平季节（F0/P4 证据）",
         "support_n": 60, "source": "dev_slice_P3_P4"},
    ],
}


def _confirmatory_rows(seeds: Sequence[int], n_per_seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for row in make_anomaly_slice(n_per_seed, seed=int(seed)):
            r = dict(row)
            r["uid"] = f"c{seed}_{row['uid']}"
            r["group_seed"] = int(seed)
            rows.append(r)
    return rows


def sample_random_program(rng: np.random.Generator) -> Dict[str, Any]:
    from .policy.program_edit import DENOISERS, IMPUTERS, OUTLIERS, WINDOWED, WINDOW_GRID
    steps: List[List[Any]] = [[str(rng.choice(list(IMPUTERS))), {}]]
    if rng.random() < 0.85:
        if rng.random() < 0.5:
            steps.append([str(rng.choice(list(OUTLIERS))), {}])
        op = str(rng.choice(list(DENOISERS)))
        params = {"window": int(rng.choice(list(WINDOW_GRID)))} if op in WINDOWED else {}
        if len(steps) < 3:
            steps.append([op, params])
    return {"grammar": "v1", "steps": steps, "scope": ["*"], "task_type": "forecast",
            "pattern_guard": [], "risk_budget_beta": 0.3, "fallback": "v_impute_linear"}


class _CountingClient:
    def __init__(self, client):
        self._client = client
        self.calls = 0
        self.wall = 0.0

    def __call__(self, system: str, user: str, nonce: int = 0) -> str:
        t0 = time.time()
        try:
            return self._client(system, user, nonce=nonce)
        finally:
            self.calls += 1
            self.wall += time.time() - t0


def _ca_candidates(obs: Mapping[str, Any], arm: str, client, backend: str,
                   repair_retries: int = 0) -> List[Tuple[Optional[dict], str]]:
    """CA 臂的 3 个候选（ITT：invalid → (None, reason) 仍占预算）。"""
    packet = dict(obs)
    if arm in ("ca_skills", "ca_skills_memory"):
        packet["skills"] = [c for c in seed_skill_cards() if c["task_scope"] == "forecast"]
    if arm == "ca_skills_memory":
        packet["memory"] = _DEV_MEMORY
    out: List[Tuple[Optional[dict], str]] = []
    for nonce in range(BUDGET):
        composer = CodeAgentComposer(backend=backend, llm=client, nonce=nonce,
                                     repair_retries=repair_retries)
        outcome = composer.compose(packet)
        if outcome.candidate is None:
            out.append((None, outcome.invalid_reason or "invalid"))
        else:
            out.append((dict(outcome.candidate.program_spec), ""))
    return out


def _arm_candidates(arm: str, obs: Mapping[str, Any], rng: np.random.Generator,
                    client, backend: str, repair_retries: int = 0) -> List[Tuple[Optional[dict], str]]:
    if arm == "frozen":
        return []
    if arm == "random_valid":
        return [(sample_random_program(rng), "") for _ in range(BUDGET)]
    if arm == "det_search":
        return [(dict(c), "") for c in _DET_LADDER]
    return _ca_candidates(obs, arm, client, backend, repair_retries=repair_retries)


def _play_episode(gym: ReadinessGym, i: int, arm: str, rng: np.random.Generator,
                  client, backend: str, repair_retries: int = 0) -> Dict[str, Any]:
    obs = gym.reset(i)
    candidates = _arm_candidates(arm, obs, rng, client, backend, repair_retries=repair_retries)
    evals: List[Dict[str, Any]] = []
    for prog, invalid_reason in candidates:
        if prog is None:
            evals.append({"proxy_delta": None, "ok": False, "reason": f"itt_noop:{invalid_reason}",
                          "program": None})
            continue
        obs, _ = gym.step({"op": "proxy_eval", "program_spec": prog})
        entry = obs["evals"][-1]
        evals.append({"proxy_delta": entry.get("proxy_delta"), "ok": entry.get("ok"),
                      "reason": entry.get("reason", ""), "program": prog})
    valid = [e for e in evals if e["ok"] and e["proxy_delta"] is not None]
    if not valid:
        gym.step({"op": "abstain"})
    else:
        best = max(valid, key=lambda e: e["proxy_delta"])
        gym.step({"op": "finalize", "program_spec": best["program"]})
    result = gym.result(i)
    novel = False
    if result["final_kind"] == "program" and valid:
        final = max(valid, key=lambda e: e["proxy_delta"])["program"]
        try:
            novel = is_novel_v1(spec_v1_from_dict(final))
        except ValueError:
            novel = False
    return {"final_kind": result["final_kind"], "true_delta": result["true_delta"],
            "final_program_sha": result["final_program_sha"], "novel_vs_menu": bool(novel),
            "candidates": [{"ok": e["ok"], "proxy_delta": e["proxy_delta"], "reason": e["reason"]}
                           for e in evals]}


def _grouped_bootstrap_ci(values_by_group: Dict[int, List[float]], rng: np.random.Generator,
                          b: int) -> Tuple[float, float]:
    groups = sorted(values_by_group)
    means = np.empty(b)
    for i in range(b):
        picked = rng.choice(groups, size=len(groups), replace=True)
        pooled = [v for g in picked for v in values_by_group[g]]
        means[i] = float(np.mean(pooled))
    lo, hi = np.percentile(means, (5.0, 95.0))
    return float(lo), float(hi)


RETRIAL_SEEDS = tuple(range(60, 80))    # P5-A.2（prereg_p5a2_retrial.md）：新种子一次性


def run_p5_identity(seeds: Sequence[int] = CONFIRMATORY_SEEDS, n_per_seed: int = 3,
                    out_dir: Path | str | None = None, ca_backend: str = "stub",
                    bootstrap_b: int = 2000, max_api_calls: int = 800,
                    repair_retries: int = 0, cache_name: str = "p5_identity") -> Dict[str, Any]:
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"

    done: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").strip().splitlines():
            if line:
                row = json.loads(line)
                done[(row["arm"], row["uid"])] = row

    client = None
    if ca_backend == "llm":
        from .llm.client import LLMClient
        client = _CountingClient(LLMClient(model="flash", temperature=0.7,
                                           cache_name=cache_name, timeout=120,
                                           max_api_calls=max_api_calls))
    rows = _confirmatory_rows(seeds, n_per_seed)
    resumed = 0
    with records_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            gym = ReadinessGym([row], task="forecast", budget=BUDGET)
            for arm in ARMS:
                key = (arm, row["uid"])
                if key in done:
                    resumed += 1
                    continue
                uid_digest = int(hashlib.sha256(row["uid"].encode()).hexdigest()[:8], 16)  # 进程盐无关
                rng = np.random.default_rng((int(row["group_seed"]), uid_digest, ARMS.index(arm)))
                episode = _play_episode(gym, 0, arm, rng, client, ca_backend,
                                        repair_retries=repair_retries)
                rec = {"arm": arm, "uid": row["uid"], "group_seed": int(row["group_seed"]),
                       "cell": row["cell"], **episode}
                done[key] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    # ── 汇总 ────────────────────────────────────────────────────────────────
    stats_rng = np.random.default_rng(20260710)
    arms_report: Dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [r for (a, _), r in done.items() if a == arm]
        deltas = [float(r["true_delta"]) for r in arm_rows]
        arms_report[arm] = {
            "n_episodes": len(arm_rows),
            "mean_true_delta": float(np.mean(deltas)) if deltas else 0.0,
            "abstain_rate": float(np.mean([r["final_kind"] != "program" for r in arm_rows])) if arm_rows else None,
            "itt_noop_candidates": int(sum(1 for r in arm_rows for c in r["candidates"]
                                           if str(c.get("reason", "")).startswith("itt_noop"))),
        }

    def _diff_by_group(arm_a: str, arm_b: str) -> Dict[int, List[float]]:
        by_group: Dict[int, List[float]] = {}
        for (a, uid), r in done.items():
            if a != arm_a:
                continue
            other = done.get((arm_b, uid))
            if other is None:
                continue
            by_group.setdefault(int(r["group_seed"]), []).append(
                float(r["true_delta"]) - float(other["true_delta"]))
        return by_group

    diff_groups = _diff_by_group("ca_skills", "det_search")
    all_diffs = [v for vs in diff_groups.values() for v in vs]
    diff_mean = float(np.mean(all_diffs)) if all_diffs else 0.0
    ci = _grouped_bootstrap_ci(diff_groups, stats_rng, bootstrap_b) if diff_groups else (0.0, 0.0)

    per_cell_lcb: Dict[str, float] = {}
    cells = sorted({r["cell"] for (a, _), r in done.items() if a == "ca_skills"})
    for cell in cells:
        cell_groups: Dict[int, List[float]] = {}
        for (a, uid), r in done.items():
            if a != "ca_skills" or r["cell"] != cell:
                continue
            other = done.get(("det_search", uid))
            if other is not None:
                cell_groups.setdefault(int(r["group_seed"]), []).append(
                    float(r["true_delta"]) - float(other["true_delta"]))
        if cell_groups:
            lo, _hi = _grouped_bootstrap_ci(cell_groups, stats_rng, bootstrap_b)
            per_cell_lcb[cell] = lo

    novel_effective = sum(1 for (a, _), r in done.items()
                          if a == "ca_skills" and r["novel_vs_menu"]
                          and float(r["true_delta"]) >= EPSILON)

    criteria = {
        "utility_vs_det": bool(diff_mean >= EPSILON and ci[0] > 0.0),
        "worst_group": bool(all(lcb >= -DELTA_SAFE for lcb in per_cell_lcb.values())) if per_cell_lcb else False,
        "novel_effective_edits": bool(novel_effective >= K_NOVEL),
        "cost_disclosed": True,
    }
    claim = ("llm_driven_harness_evolution" if all(criteria.values())
             else "self_updating_deterministic_with_llm_optional")

    report = {
        "phase": "P5A_identity_gate",
        "protocol": "prereg §5/§5.0 frozen; confirmatory seeds one-shot; true judges only",
        "n_episodes": len(rows),
        "seeds": [int(s) for s in seeds],
        "resumed_episodes": resumed,
        "arms": arms_report,
        "primary_comparison": {
            "primary_ca_arm": "ca_skills", "baseline_arm": "det_search",
            "diff_mean": diff_mean, "epsilon": EPSILON,
            "ci90": [ci[0], ci[1]], "grouped_by": "generator_seed",
        },
        "per_cell_diff_lcb": per_cell_lcb,
        "novel_effective_edits": int(novel_effective),
        "headline_criteria": criteria,
        "claim_branch": claim,
        "cost_ledger": {
            "backend": ca_backend,
            # 勘误口径（外评②）：calls = composer 调用次数（**含磁盘缓存命中**）；
            # 真实网络/缓存分离见 client_stats（n_api/n_hit，P5-A/P5-A.2 归档 run 无此字段）
            "api_calls": int(client.calls) if client is not None else 0,
            "llm_wall_seconds": round(float(client.wall), 1) if client is not None else 0.0,
            "client_stats": (client._client.stats() if client is not None
                             and hasattr(client._client, "stats") else None),
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "generated_by": "run_p5_identity_gate",
        "plan": "Final_Plan_CodeAgentFirst_2026-07-09 §P5",
        "prereg": "results/Stage2/prereg_codeagent_first_P1_P5.md §5/§5.0",
        "epsilon": EPSILON, "delta_safe": DELTA_SAFE, "k_novel": K_NOVEL, "budget": BUDGET,
        "det_ladder": _DET_LADDER, "dev_memory": _DEV_MEMORY,
        "seeds": [int(s) for s in seeds], "n_per_seed": int(n_per_seed),
        "ca_backend": ca_backend, "bootstrap_b": int(bootstrap_b),
        "repair_retries": int(repair_retries),
        # 代码级接口版本（P5-A.2 两前置落地后全局生效；P5-A 原 run 已归档，其 manifest 无此字段）
        "interface_version": "gym_fingerprint_v2 + prompt_v2_exemplar",
        "task": "forecast_only（anomaly 按 prereg §5.0 排除）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P5-A identity gate (one-shot confirmatory)")
    parser.add_argument("--backend", type=str, default="stub", choices=["stub", "llm"])
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--n-per-seed", type=int, default=3)
    parser.add_argument("--retrial", action="store_true",
                        help="P5-A.2（prereg_p5a2_retrial.md）：seeds 60–79、repair_retries=1、新缓存名")
    args = parser.parse_args()
    if args.retrial:
        out = (Path(__file__).resolve().parent / "results" / "Stage2" / "P5A2Retrial"
               if args.out_dir == str(DEFAULT_OUT) else Path(args.out_dir))
        report = run_p5_identity(seeds=RETRIAL_SEEDS, n_per_seed=args.n_per_seed,
                                 out_dir=out, ca_backend=args.backend,
                                 repair_retries=1, cache_name="p5a2_identity")
    else:
        report = run_p5_identity(n_per_seed=args.n_per_seed, out_dir=args.out_dir,
                                 ca_backend=args.backend)
    print(json.dumps({
        "claim_branch": report["claim_branch"],
        "diff_mean": report["primary_comparison"]["diff_mean"],
        "ci90": report["primary_comparison"]["ci90"],
        "criteria": report["headline_criteria"],
        "api_calls": report["cost_ledger"]["api_calls"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
