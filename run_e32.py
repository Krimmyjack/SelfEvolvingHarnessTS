"""run_e32.py — E-3.2 六臂+诊断臂 正式 runner（A-37 协议 + A-39 修复包，评审第十五轮）。

评估器 = `e32_nested.run_policy_folds`（A-39① P0 修复：action-loss 生成嵌入 policy outer fold；
旧"全语料 OOF → 事后切折"的 `policy_data_from_corpus` 已废除——A-31 同类失格）。

两口径（A-39③ 预注册）：
  primary_no_Sar  排除 S_ar（其实测 SNR 支持与其余结构不交、snrHigh 不可达 → 条件化比较=外推）。
                  **D-3.2e 六判据在本口径判**。
  all_data        全语料稳健性附录（安全 worst-group 两口径都报，不解释为完全去除 SNR 混杂）。
诊断臂：oracle_struct（结构上界）+ true_d_gbdt（生成器真实 noise/miss——D+P 只胜 measured-D
不胜 true-D ⇒ 部分 Pattern 收益=修补 SNR 估计误差）。两者不进主比较。

统计（A-39②⑤）：判据(vi)=SNR 分层置换（n_perm=99，per-perm 独立种子+checkpoint）；主比较
CI=paired-uid bootstrap（B=2000；caveat：条件于已拟合头/router，未含重拟合方差）；(iv)=trend
保留率（候选臂=dp_abstain）。产物：freeze.json（先落盘）/ per-uid records / report.json。

门禁：无 `--dev` 拒跑；无 A-31e manifest 须 `--allow-imbalanced`（结果不进文档）；
永不触碰 confirmatory seeds 20–39。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
  -m SelfEvolvingHarnessTS.run_e32 --dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .augment_corpus import RESULTS_A31E, build_augmented_corpus
from .e32_nested import aggregate_records, perm_stat_fn, run_policy_folds, stratified_folds
from .e32_policy import (ABLATION_MA, D_FEATS, EPS, FALLBACK_ACTION, KAPPA, N_ENSEMBLE, P_FEATS,
                         PRUNED_POOL_CORE, GBDTArm, make_arms, paired_bootstrap_ci,
                         residualized_perm_test, snr_strata, trend_retention, verdict_d32e)
from .evaluators.frozen_probe import FrozenProbe
from .family0_actions import f0_variants
from .fast_path.perceive import perceive
from .harness import HarnessState
from .run_main_table import fixed_harness_variants
from .run_variance_decomp import DEG_GRID, assign_cells, build_cell_cache, build_corpus

RESULTS = Path(__file__).resolve().parent / "results" / "E3_2"
DEFAULT_SEED = 20260704
OUTER_K, INNER_K = 5, 4
N_PERM, BOOT_B = 99, 2000
GBDT_PARAMS = dict(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.7)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _variant_map(actions: List[str], task: str = "forecast"):
    all_v = {**fixed_harness_variants(task), **f0_variants(task)}
    missing = [a for a in actions if a not in all_v]
    assert not missing, f"动作未注册：{missing}"
    return {a: all_v[a] for a in actions}


def _true_d_of(uid: str, manifest_by_uid: Dict[str, dict]):
    if ":A31e:" in uid or ":A38C:" in uid:      # 补样 uid（dev A-38 / confirmatory A-40④）→ manifest 载真值
        e = manifest_by_uid[uid]
        return float(e["noise"]), float(e["miss"])
    struct, dname, j = uid.split(":")
    dp = DEG_GRID[dname]
    return float(dp["noise"]), float(dp["miss"])


def collect_cells_data(corpus, actions: List[str], verbose: bool = True,
                       manifest_by_uid: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """每 cell：动作缓存（交集对齐 uid）+ struct_feats + origin + 生成器真实 (noise, miss)。
    manifest_by_uid=None → 默认读 A31e manifest（dev 行为不变）；confirmatory 传入合并 manifest。"""
    h = HarnessState.from_minimal()
    fp = FrozenProbe()
    variants = _variant_map(actions)
    cells, _ = assign_cells(corpus)
    if manifest_by_uid is None:
        man = RESULTS_A31E / "manifest.json"
        manifest_by_uid = ({e["uid"]: e for e in json.loads(man.read_text("utf-8"))["entries"]}
                           if man.exists() else {})
    out: Dict[str, dict] = {}
    for cid in sorted(cells):
        series = cells[cid]
        t0 = time.time()
        action_caches, common_uids, origin_of = build_cell_cache(fp, series, variants)
        if not common_uids:
            continue
        cu = set(common_uids)
        feats_of = {rs.series_uid: perceive(rs.history, "forecast", h)["pattern"]["struct_feats"]
                    for rs in series if rs.series_uid in cu}
        true_d = {u: _true_d_of(u, manifest_by_uid) for u in common_uids}
        out[cid] = dict(action_caches=action_caches, uids=list(common_uids),
                        origin_of={u: origin_of[u] for u in common_uids},
                        feats_of=feats_of, true_d=true_d)
        if verbose:
            print(f"  [cache] {cid:26s} n={len(common_uids):3d} [{time.time()-t0:.0f}s]", flush=True)
    return out


def make_all_arms(seed: int) -> Dict[str, Callable[[], object]]:
    arms = make_arms(seed)
    arms["true_d_gbdt"] = lambda: GBDTArm(("t",), seed=seed)      # 诊断臂（A-39③）
    return arms


def _scope_filter(cells_data: Dict[str, dict], scope: str) -> Dict[str, dict]:
    if scope == "all_data":
        return cells_data
    assert scope == "primary_no_Sar"
    out = {}
    for cid, cd in cells_data.items():
        keep = [u for u in cd["uids"] if cd["origin_of"][u] != "S_ar"]
        if keep:
            out[cid] = dict(action_caches={a: {u: c[u] for u in keep}
                                           for a, c in cd["action_caches"].items()},
                            uids=keep,
                            origin_of={u: cd["origin_of"][u] for u in keep},
                            feats_of={u: cd["feats_of"][u] for u in keep},
                            true_d={u: cd["true_d"][u] for u in keep})
    return out


def run_scope(cells_data_all: Dict[str, dict], actions: List[str], scope: str, out_dir: Path,
              seed: int, n_perm: int, with_perm: bool) -> dict:
    cells_data = _scope_filter(cells_data_all, scope)
    all_uids = [u for cd in cells_data.values() for u in cd["uids"]]
    strat_of = {u: f"{cid}|{cd['origin_of'][u]}"
                for cid, cd in cells_data.items() for u in cd["uids"]}
    arms = make_all_arms(seed)
    print(f"\n== scope={scope}  n={len(all_uids)} ==", flush=True)

    # —— arm-in：分层 5 折（嵌入式标签，A-39①）——
    fold_of = stratified_folds(all_uids, strat_of, OUTER_K, seed)
    folds = [(f"armin{f}", [u for u in all_uids if fold_of[u] != f],
              [u for u in all_uids if fold_of[u] == f]) for f in range(OUTER_K)]
    res = run_policy_folds(cells_data, actions, arms, folds, inner_k=INNER_K, seed=seed,
                           ckpt_dir=out_dir / f"ckpt_{scope}")
    agg = aggregate_records(res["records"], actions, list(arms))
    (out_dir / f"records_{scope}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in res["records"]), "utf-8")

    # —— LODO（留一结构 = level-1 压力测试，A-37⑤）——
    origins = sorted({o for cd in cells_data.values() for o in cd["origin_of"].values()})
    lodo_arms = {k: v for k, v in arms.items() if k != "oracle_struct"}
    lodo_folds = [(f"lodo_{g}", [u for u in all_uids if strat_of[u].split("|")[-1] != g],
                   [u for u in all_uids if strat_of[u].split("|")[-1] == g]) for g in origins]
    lres = run_policy_folds(cells_data, actions, lodo_arms, lodo_folds, inner_k=INNER_K,
                            seed=seed, ckpt_dir=out_dir / f"ckpt_{scope}_lodo")
    lodo_out: Dict[str, dict] = {}
    for g in origins:
        recs_g = [r for r in lres["records"] if r["fold"] == f"lodo_{g}"]
        agg_g = aggregate_records(recs_g, actions, list(lodo_arms))
        lodo_out[g] = {n: dict(mean_loss=agg_g[n]["mean_loss"], mean_regret=agg_g[n]["mean_regret"],
                               worst_group_mean=agg_g[n]["worst_group_mean"],
                               abstain_rate=agg_g[n]["abstain_rate"],
                               mean_delta_vs_incumbent=float(np.mean(agg_g[n]["per_uid_delta_vs_incumbent"])))
                       for n in lodo_arms}

    # —— 统计产物：paired bootstrap CI / trend 保留率 / abstain per subgroup ——
    def _ci(a: str, b: str) -> dict:
        return paired_bootstrap_ci(np.array(agg[a]["per_uid_regret"]),
                                   np.array(agg[b]["per_uid_regret"]), n_boot=BOOT_B, seed=seed)
    cis = {"dp_vs_global": _ci("dp_gbdt", "global"), "dp_vs_dlookup": _ci("dp_gbdt", "d_lookup"),
           "dp_vs_dgbdt": _ci("dp_gbdt", "d_gbdt"), "abstain_vs_dp": _ci("dp_abstain", "dp_gbdt"),
           "dp_vs_true_d": _ci("dp_gbdt", "true_d_gbdt")}
    data_eval = agg["_data_eval"]
    idx_of = {n: np.array([actions.index(r["arms"][n]["pick"]) for r in res["records"]])
              for n in ("dp_abstain", "dp_gbdt", "d_lookup")}
    Lrows = data_eval.L[np.arange(data_eval.n), :]
    lp = {n: Lrows[np.arange(data_eval.n), idx_of[n]] for n in idx_of}
    ret_abstain = trend_retention(data_eval, lp["dp_abstain"], lp["d_lookup"])
    ret_dp = trend_retention(data_eval, lp["dp_gbdt"], lp["d_lookup"])
    abst_sub: Dict[str, float] = {}
    for key in sorted({f"{r['cell']}|{r['origin']}" for r in res["records"]}):
        sub = [r for r in res["records"] if f"{r['cell']}|{r['origin']}" == key]
        abst_sub[key] = float(np.mean([r["arms"]["dp_abstain"]["abstain"] for r in sub]))

    # —— 判据 (vi)：SNR 分层置换（仅 primary 判；checkpoint 续跑）——
    perm = None
    if with_perm:
        L_test_of = {r["uid"]: r["L_test"] for r in res["records"]}
        stat, uid_sorted = perm_stat_fn(cells_data, actions, res["fold_details"], L_test_of,
                                        lambda: GBDTArm(("d", "p"), seed=seed),
                                        agg["d_gbdt"]["mean_regret"])
        pos = {u: i for i, u in enumerate(agg["_uid_order"])}
        X_p_sorted = np.array([res["records"][pos[u]]["X_p"] for u in uid_sorted])
        snr_sorted = np.array([res["records"][pos[u]]["snr"] for u in uid_sorted])
        cell_sorted = np.array([res["records"][pos[u]]["cell"] for u in uid_sorted])
        strata = snr_strata(snr_sorted, cell_sorted, n_bins=3)
        pfile = out_dir / f"perm_nulls_{scope}.json"
        done = json.loads(pfile.read_text("utf-8")) if pfile.exists() else []
        print(f"  [perm] n_perm={n_perm}（已有 {len(done)}）", flush=True)
        while len(done) < n_perm:                              # 每 10 个 null 落盘一次（可续）
            upto = min(len(done) + 10, n_perm)
            r = residualized_perm_test(stat, X_p_sorted, strata, n_perm=upto, seed=seed,
                                       done_nulls=done, progress=0)
            done = r["nulls"]
            tmp = pfile.with_suffix(".tmp")
            tmp.write_text(json.dumps(done), "utf-8")
            tmp.replace(pfile)
            print(f"  [perm] {len(done)}/{n_perm}", flush=True)
        perm = residualized_perm_test(stat, X_p_sorted, strata, n_perm=n_perm, seed=seed,
                                      done_nulls=done)
        perm.pop("nulls")

    verdict = verdict_d32e({k: v for k, v in agg.items() if not k.startswith("_")},
                           trend_retention_val=ret_abstain["retention"],
                           perm_p=perm["p"] if perm else None) if scope == "primary_no_Sar" else None
    arm_summary = {n: {k: v for k, v in agg[n].items()
                       if k not in ("per_uid_regret", "per_uid_loss", "per_uid_delta_vs_incumbent")}
                   for n in arms}
    return dict(n=len(all_uids), arm_in=arm_summary, lodo_level1_structure=lodo_out,
                paired_bootstrap_cis=cis,
                trend_retention=dict(dp_abstain=ret_abstain, dp_gbdt=ret_dp),
                abstain_rate_by_subgroup=abst_sub, perm_test=perm, verdict_d32e=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--allow-imbalanced", action="store_true")
    ap.add_argument("--ma-ablation", action="store_true")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    if not args.dev:
        raise SystemExit("门禁（A-37/A-39）：正式 E-3.2 须协议冻结后运行；确认后加 --dev。")
    man = RESULTS_A31E / "manifest.json"
    if man.exists():
        corpus, corpus_tag = build_augmented_corpus(20), "dev(20)+A31e"
    elif args.allow_imbalanced:
        corpus, corpus_tag = build_corpus(20), "dev(20) IMBALANCED（结果不得进文档）"
    else:
        raise SystemExit("A-31e manifest 缺失（硬前置）；机制调试可 --allow-imbalanced。")

    actions = PRUNED_POOL_CORE + (ABLATION_MA if args.ma_ablation else [])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    freeze = dict(  # A-39⑤：冻结块先落盘
        date="2026-07-04", amendment="A-37+A-39", actions=actions,
        p_feats=list(P_FEATS), d_feats=list(D_FEATS), gbdt_params=GBDT_PARAMS,
        kappa=KAPPA, n_ensemble=N_ENSEMBLE, fallback=FALLBACK_ACTION, eps=EPS,
        outer_k=OUTER_K, inner_k=INNER_K, seed=args.seed, n_perm=args.n_perm, boot_b=BOOT_B,
        scopes=dict(primary="primary_no_Sar（排除 S_ar，判据口径）", robustness="all_data"),
        corpus=corpus_tag,
        manifest_sha=_sha(json.loads(man.read_text("utf-8"))) if man.exists() else None,
        config_sha=None)
    freeze["config_sha"] = _sha({k: v for k, v in freeze.items() if k != "config_sha"})
    (out_dir / "freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=1), "utf-8")
    print(f"freeze.json 落盘（config_sha={freeze['config_sha']}）", flush=True)

    print(f"E-3.2：语料={corpus_tag} n={len(corpus)} 动作池={len(actions)}", flush=True)
    t0 = time.time()
    cells_data = collect_cells_data(corpus, actions)
    print(f"缓存完成 [{time.time()-t0:.0f}s]", flush=True)

    report = dict(freeze=freeze, scopes={})
    report["scopes"]["primary_no_Sar"] = run_scope(cells_data, actions, "primary_no_Sar",
                                                   out_dir, args.seed, args.n_perm, with_perm=True)
    report["scopes"]["all_data"] = run_scope(cells_data, actions, "all_data",
                                             out_dir, args.seed, args.n_perm, with_perm=False)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print("\n== D-3.2e 判决（primary_no_Sar）==")
    print(json.dumps(report["scopes"]["primary_no_Sar"]["verdict_d32e"], ensure_ascii=False, indent=1))
    print(f"→ {out_dir / 'report.json'}  总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
