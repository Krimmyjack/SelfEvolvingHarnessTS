"""run_confirmatory.py — 一次性 confirmatory 执行器（A-40/A-41）：locked-transfer 主 + replication 次。

主 estimand（A-40③）：dev 训练的 **frozen router**（frozen_arms.joblib，SHA 核验，**永不 fit**）
直接应用 confirmatory uid；测量标签 = confirmatory 内 cell×origin 分层 grouped OOF（router 冻结
→ 无选择泄漏）。次 estimand：replication = run_e32 nested 机器在 confirmatory 语料重新 cross-fit
（独立目录，A-41⑥守卫⑧）。

主判决 CI = grouped full-refit bootstrap（A-33c 机器：每 replicate 组重采样 uid + 身份分折 +
sample_weight 重拟合测量头 + 变折 seed；**router picks 冻结 → 只有测量头随 replicate 重拟合**）；
per-replicate 独立种子 + 原子 checkpoint/resume（A-36）。原始分布与 cell-equal 聚合分开报
（A-41⑥守卫⑨）。布尔判据表见 confirmatory_freeze.json["criteria"]。

stages（每 stage 可独立后台跑，caches 每次重建 ~几分钟）：
  lt       语料+缓存+OOF 标签+冻结臂评估+点统计 → locked_transfer/records_locked_{scope}.jsonl
  ci       full-refit bootstrap（--scope，checkpoint 续跑）→ locked_transfer/fullrefit_{scope}.json
  repl     replication（次要 estimand）→ replication/
  verdict  汇总（含 reporter_panel_{scope}.json 若已跑）→ confirmatory_report.json + 布尔表

门禁：freeze 存在 + router SHA 核验 + A38C manifest 存在；--smoke-dev = dev 语料全链路冒烟
（in-memory 训练臂、写 smoke_dev/、不触碰 holdout、结果不进文档）。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_confirmatory --stage lt --open-holdout
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .confirmatory_freeze import (FREEZE_PATH, FROZEN_ARMS, FULL_REFIT_B, GATE_COMPARATORS,
                                  OOF_K, PAIRED_BOOT_B, REPORT_COMPARATORS, RESULTS_CONF,
                                  load_frozen_arms, train_final_arms)
from .e32_nested import _policy_data, aggregate_records, stratified_folds
from .e32_policy import (DELTA_SAFE, EPS, FALLBACK_ACTION, PRUNED_POOL_CORE,
                         paired_bootstrap_ci, trend_retention)
from .nested_supply import _eval_uids, _fit_head, _fit_head_w, make_folds
from .run_e32 import _scope_filter, collect_cells_data, make_all_arms

LT_DIR_NAME, REPL_DIR_NAME = "locked_transfer", "replication"   # 守卫⑧：目录分离
DEFAULT_CONF_SEED = 20260705            # confirmatory 折/bootstrap 种子（≠ dev 20260704，避免任何折叠对齐幻觉）
CKPT_EVERY = 10


# ══════════════════════════════════════════════════════════════════════════
# freeze 核验 + 语料装配
# ══════════════════════════════════════════════════════════════════════════
def verify_freeze():
    if not FREEZE_PATH.exists():
        raise SystemExit("A-41 门禁：confirmatory_freeze.json 未落盘。")
    freeze = json.loads(FREEZE_PATH.read_text("utf-8"))
    arms_blob = load_frozen_arms(FROZEN_ARMS, verify_sha=freeze["router"]["sha256"])   # 守卫①
    return freeze, arms_blob


def load_corpus_and_manifest(smoke_dev: bool):
    if smoke_dev:
        from .augment_corpus import RESULTS_A31E, build_augmented_corpus
        man = json.loads((RESULTS_A31E / "manifest.json").read_text("utf-8"))["entries"]
        return build_augmented_corpus(20), {e["uid"]: e for e in man}
    from .confirmatory_corpus import build_confirmatory_corpus, manifest_by_uid_a38c
    corpus = build_confirmatory_corpus()
    return corpus, manifest_by_uid_a38c()


# ══════════════════════════════════════════════════════════════════════════
# 测量标签：confirmatory 内 grouped OOF（router 冻结 → 头只承担测量，无选择泄漏）
# ══════════════════════════════════════════════════════════════════════════
def oof_labels(cells_data: Dict[str, dict], actions: Sequence[str], k: int, seed: int,
               verbose: bool = True) -> Dict[str, Dict[str, float]]:
    all_uids = [u for cd in cells_data.values() for u in cd["uids"]]
    strat_of = {u: f"{cid}|{cd['origin_of'][u]}" for cid, cd in cells_data.items()
                for u in cd["uids"]}
    fold_of = stratified_folds(all_uids, strat_of, k, seed)
    L_of: Dict[str, Dict[str, float]] = {}
    for cid in sorted(cells_data):
        cd = cells_data[cid]
        t0 = time.time()
        for a in actions:
            caches = cd["action_caches"][a]
            for f in range(k):
                tr = [u for u in cd["uids"] if fold_of[u] != f]
                te = [u for u in cd["uids"] if fold_of[u] == f]
                if not tr or not te:
                    continue
                head = _fit_head(caches, tr)
                for u, v in _eval_uids(head, caches, te).items():
                    L_of.setdefault(u, {})[a] = v
        if verbose:
            print(f"  [labels] {cid:26s} [{time.time()-t0:.0f}s]", flush=True)
    return L_of


# ══════════════════════════════════════════════════════════════════════════
# 冻结臂评估（picks 只读特征；L 只用于事后测量——守卫②/污染测试的被测对象）
# ══════════════════════════════════════════════════════════════════════════
def frozen_eval(cells_data: Dict[str, dict], actions: List[str], arms: Dict[str, object],
                L_of: Dict[str, Dict[str, float]]) -> List[dict]:
    order = sorted(u for cd in cells_data.values() for u in cd["uids"])
    data = _policy_data(cells_data, actions, order, L_of)
    idx = np.arange(data.n)
    picks_by, abst_by = {}, {}
    for name, arm in arms.items():
        p, a = arm.picks(data, idx)                            # 永不 fit（A-41②）
        picks_by[name], abst_by[name] = p, a
    records = []
    for i, u in enumerate(order):
        records.append(dict(
            uid=u, fold="locked",
            cell=str(data.cell[i]), origin=str(data.origin[i]),
            snr=float(data.X_d[i, 0]), miss_rate=float(data.X_d[i, 1]),
            X_p=[float(x) for x in data.X_p[i]], X_t=[float(x) for x in data.X_t[i]],
            L_test={a: float(L_of[u][a]) for a in actions},
            arms={n: dict(pick=actions[int(picks_by[n][i])], abstain=bool(abst_by[n][i]))
                  for n in arms}))
    return records


def cell_equal_stats(records: List[dict], actions: List[str], arm_names: Sequence[str]) -> dict:
    """cell-equal 聚合（守卫⑨：与原始分布分开报）：per-cell 均值 regret 再等权平均。"""
    out = {}
    cells = sorted({r["cell"] for r in records})
    for n in arm_names:
        per_cell = {}
        for c in cells:
            rs = [r for r in records if r["cell"] == c]
            reg = [r["L_test"][r["arms"][n]["pick"]] - min(r["L_test"].values()) for r in rs]
            per_cell[c] = float(np.mean(reg))
        out[n] = dict(per_cell=per_cell, cell_equal_mean_regret=float(np.mean(list(per_cell.values()))))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 主判决 CI：grouped full-refit bootstrap（A-33c 机器移植到 locked transfer）
# picks 冻结不随 replicate 变 → 每 replicate 只重拟合测量头 + 重算 oracle/regret/子群
# ══════════════════════════════════════════════════════════════════════════
def full_refit_bootstrap(cells_data: Dict[str, dict], actions: List[str],
                         picks_of: Dict[str, Dict[str, str]], comparisons: List[str],
                         n_boot: int, seed: int, k: int = OOF_K,
                         ckpt_path: Optional[Path] = None, progress: int = 10) -> dict:
    uid_cell = {u: cid for cid, cd in cells_data.items() for u in cd["uids"]}
    uid_sub = {u: f"{cid}|{cells_data[cid]['origin_of'][u]}" for cid, cd in cells_data.items()
               for u in cd["uids"]}
    uids = sorted(uid_cell)
    N = len(uids)
    sub_keys = sorted(set(uid_sub.values()))
    arm_names = sorted(picks_of)
    state = dict(done=0, comp={c: [] for c in comparisons},
                 arm_regret={n: [] for n in arm_names},
                 comp_cell_eq={c: [] for c in comparisons},
                 sub_dp_delta={s: [] for s in sub_keys})
    if ckpt_path and ckpt_path.exists():
        st = json.loads(ckpt_path.read_text("utf-8"))
        if st.get("seed") == seed and st.get("n_uid") == N:
            state = st["state"]
            print(f"  [resume] full-refit 从 {state['done']}/{n_boot} 续跑", flush=True)

    def _save(done):
        if not ckpt_path:
            return
        state["done"] = done
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(seed=seed, n_uid=N, state=state)), "utf-8")
        tmp.replace(ckpt_path)

    t0 = time.time()
    for b in range(state["done"], n_boot):
        rng = np.random.default_rng(seed + 4242 + 7907 * (b + 1))     # per-replicate 独立（A-36）
        samp = [uids[i] for i in rng.integers(0, N, N)]
        mult = Counter(samp)
        distinct = sorted(mult)
        dset = set(distinct)
        bseed = seed + 100003 * (b + 1)                               # 每 replicate 变折 seed
        fold_of = make_folds(distinct, k, bseed)                      # 按身份分折（副本同折防泄漏）
        L_b: Dict[str, Dict[str, float]] = {u: {} for u in distinct}
        for cid, cd in cells_data.items():
            uids_c = [u for u in cd["uids"] if u in dset]
            for f in range(k):
                tr = [u for u in uids_c if fold_of[u] != f]
                te = [u for u in uids_c if fold_of[u] == f]
                if not tr or not te:
                    continue
                for a in actions:
                    head = _fit_head_w(cd["action_caches"][a], tr, mult)
                    for u, v in _eval_uids(head, cd["action_caches"][a], te).items():
                        L_b[u][a] = v
        ok = [u for u in distinct if len(L_b[u]) == len(actions)]
        w = np.array([mult[u] for u in ok], float)
        oracle = np.array([min(L_b[u].values()) for u in ok])
        reg = {n: np.array([L_b[u][picks_of[n][u]] for u in ok]) - oracle for n in arm_names}
        wm = {n: float(np.sum(w * reg[n]) / np.sum(w)) for n in arm_names}
        for n in arm_names:
            state["arm_regret"][n].append(wm[n])
        for c in comparisons:                                          # "dp_abstain_vs_<base>"
            base = c.split("_vs_")[1]
            state["comp"][c].append(wm["dp_abstain"] - wm[base])
            cells_ = sorted({uid_cell[u] for u in ok})                 # cell-equal 口径（守卫⑨）
            ce = []
            for cc in cells_:
                m = np.array([uid_cell[u] == cc for u in ok])
                ce.append(float(np.sum(w[m] * (reg["dp_abstain"][m] - reg[base][m])) / np.sum(w[m])))
            state["comp_cell_eq"][c].append(float(np.mean(ce)))
        inc = np.array([L_b[u][FALLBACK_ACTION] for u in ok])          # Δ vs incumbent（子群安全）
        dp_loss = np.array([L_b[u][picks_of["dp_abstain"][u]] for u in ok])
        delta_inc = inc - dp_loss
        for s in sub_keys:
            m = np.array([uid_sub[u] == s for u in ok])
            if m.any():
                state["sub_dp_delta"][s].append(float(np.sum(w[m] * delta_inc[m]) / np.sum(w[m])))
        if (b + 1) % CKPT_EVERY == 0:
            _save(b + 1)
        if progress and (b + 1) % progress == 0:
            print(f"  [full-refit] {b+1}/{n_boot} [{time.time()-t0:.0f}s]", flush=True)
    _save(n_boot)

    def _ci(vals):
        a = np.array(vals)
        return dict(boot_mean=float(a.mean()), ci_lo=float(np.percentile(a, 2.5)),
                    ci_hi=float(np.percentile(a, 97.5)), n_boot=int(len(a)))
    return dict(
        comparisons={c: _ci(v) for c, v in state["comp"].items()},
        comparisons_cell_equal={c: _ci(v) for c, v in state["comp_cell_eq"].items()},
        arm_regret_boot={n: _ci(v) for n, v in state["arm_regret"].items()},
        subgroup_dp_delta_q05={s: (float(np.percentile(np.array(v), 5)) if v else None)
                               for s, v in state["sub_dp_delta"].items()},
        n_boot=int(n_boot), seed=int(seed), k=int(k),
        method="grouped_full_refit_bootstrap(locked-transfer: 头重拟合、picks 冻结)")


# ══════════════════════════════════════════════════════════════════════════
# stages
# ══════════════════════════════════════════════════════════════════════════
def _out_root(smoke_dev: bool) -> Path:
    return RESULTS_CONF / ("smoke_dev" if smoke_dev else "")


def _get_arms(freeze, smoke_dev: bool) -> Dict[str, object]:
    if freeze is not None:
        return load_frozen_arms(FROZEN_ARMS, verify_sha=freeze["router"]["sha256"])["arms"]  # 守卫①
    if smoke_dev:
        if FROZEN_ARMS.exists():
            return load_frozen_arms(FROZEN_ARMS)["arms"]
        print("[smoke] frozen_arms 不存在 → in-memory 训练（不写正式产物）", flush=True)
        fitted, _ = train_final_arms()
        return fitted
    raise SystemExit("A-41 门禁：非 smoke 必须有 freeze。")


def stage_lt(args, freeze):
    out = _out_root(args.smoke_dev) / LT_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    actions = list(PRUNED_POOL_CORE)
    arms = _get_arms(freeze, args.smoke_dev)
    corpus, man = load_corpus_and_manifest(args.smoke_dev)
    print(f"locked-transfer：语料 n={len(corpus)}（smoke_dev={args.smoke_dev}）", flush=True)
    t0 = time.time()
    cells_all = collect_cells_data(corpus, actions, manifest_by_uid=man)
    print(f"缓存完成 [{time.time()-t0:.0f}s]", flush=True)
    for scope in ("primary_no_Sar", "all_data"):
        cells = _scope_filter(cells_all, scope)
        n = sum(len(cd["uids"]) for cd in cells.values())
        print(f"\n== [lt] scope={scope} n={n} ==", flush=True)
        L_of = oof_labels(cells, actions, OOF_K, args.seed)
        records = frozen_eval(cells, actions, arms, L_of)
        (out / f"records_locked_{scope}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records), "utf-8")
        agg = aggregate_records(records, actions, sorted(arms))
        data_eval = agg["_data_eval"]
        idx_of = {n_: np.array([actions.index(r["arms"][n_]["pick"]) for r in records])
                  for n_ in ("dp_abstain", "d_lookup")}
        Lrows = data_eval.L
        lp = {n_: Lrows[np.arange(data_eval.n), idx_of[n_]] for n_ in idx_of}
        ret = trend_retention(data_eval, lp["dp_abstain"], lp["d_lookup"])
        cis = {}
        for base in list(GATE_COMPARATORS) + list(REPORT_COMPARATORS):
            cis[f"dp_abstain_vs_{base}"] = paired_bootstrap_ci(
                np.array(agg["dp_abstain"]["per_uid_regret"]),
                np.array(agg[base]["per_uid_regret"]), n_boot=PAIRED_BOOT_B, seed=args.seed)
        abst_sub = {}
        for key in sorted({f"{r['cell']}|{r['origin']}" for r in records}):
            sub = [r for r in records if f"{r['cell']}|{r['origin']}" == key]
            abst_sub[key] = float(np.mean([r["arms"]["dp_abstain"]["abstain"] for r in sub]))
        point = dict(
            n=n, scope=scope,
            arm_summary={n_: {k: v for k, v in agg[n_].items()
                              if k not in ("per_uid_regret", "per_uid_loss",
                                           "per_uid_delta_vs_incumbent")}
                         for n_ in sorted(arms)},
            paired_uid_cis_secondary=cis,
            trend_retention=ret,
            abstain_rate_by_subgroup=abst_sub,
            cell_equal=cell_equal_stats(records, actions, sorted(arms)))
        (out / f"lt_point_{scope}.json").write_text(
            json.dumps(point, ensure_ascii=False, indent=1), "utf-8")
        print(f"  regret: dp_abstain={agg['dp_abstain']['mean_regret']:.4f} "
              f"global={agg['global']['mean_regret']:.4f} d_lookup={agg['d_lookup']['mean_regret']:.4f} "
              f"d_gbdt={agg['d_gbdt']['mean_regret']:.4f}  retention={ret['retention']}", flush=True)


def stage_ci(args, freeze):
    out = _out_root(args.smoke_dev) / LT_DIR_NAME
    actions = list(PRUNED_POOL_CORE)
    corpus, man = load_corpus_and_manifest(args.smoke_dev)
    cells_all = collect_cells_data(corpus, actions, manifest_by_uid=man, verbose=False)
    cells = _scope_filter(cells_all, args.scope)
    rec_path = out / f"records_locked_{args.scope}.jsonl"
    records = [json.loads(l) for l in rec_path.read_text("utf-8").splitlines() if l.strip()]
    arm_names = ["dp_abstain"] + list(GATE_COMPARATORS) + list(REPORT_COMPARATORS)
    picks_of = {n: {r["uid"]: r["arms"][n]["pick"] for r in records} for n in arm_names}
    comparisons = [f"dp_abstain_vs_{b}" for b in list(GATE_COMPARATORS) + list(REPORT_COMPARATORS)]
    res = full_refit_bootstrap(cells, actions, picks_of, comparisons,
                               n_boot=args.n_boot, seed=args.seed,
                               ckpt_path=out / f"ckpt_fullrefit_{args.scope}.json",
                               progress=max(1, args.n_boot // 50))
    (out / f"fullrefit_{args.scope}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    for c in comparisons:
        r = res["comparisons"][c]
        print(f"  {c:28s} boot_mean={r['boot_mean']:+.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]",
              flush=True)


def stage_repl(args, freeze):
    from .run_e32 import run_scope
    out = _out_root(args.smoke_dev) / REPL_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    actions = list(PRUNED_POOL_CORE)
    corpus, man = load_corpus_and_manifest(args.smoke_dev)
    cells_all = collect_cells_data(corpus, actions, manifest_by_uid=man)
    report = dict(estimand="replication（次要，A-40③）", scopes={})
    report["scopes"]["primary_no_Sar"] = run_scope(cells_all, actions, "primary_no_Sar",
                                                   out, args.seed, 0, with_perm=False)
    report["scopes"]["all_data"] = run_scope(cells_all, actions, "all_data",
                                             out, args.seed, 0, with_perm=False)
    (out / "replication_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"→ {out / 'replication_report.json'}", flush=True)


def stage_verdict(args, freeze):
    root = _out_root(args.smoke_dev)
    lt = root / LT_DIR_NAME
    point = json.loads((lt / "lt_point_primary_no_Sar.json").read_text("utf-8"))
    fr = json.loads((lt / "fullrefit_primary_no_Sar.json").read_text("utf-8"))
    arm = point["arm_summary"]
    reg = {n: arm[n]["mean_regret"] for n in arm}
    verdict = {}
    for i, base in enumerate(GATE_COMPARATORS, 1):
        d_point = reg["dp_abstain"] - reg[base]
        ci_hi = fr["comparisons"][f"dp_abstain_vs_{base}"]["ci_hi"]
        verdict[f"C{i}_vs_{base}"] = dict(point=float(d_point), fullrefit_ci_hi=float(ci_hi),
                                          passed=bool(d_point < -EPS and ci_hi < 0))
    season_q05 = {s: q for s, q in fr["subgroup_dp_delta_q05"].items()
                  if "S_season" in s and q is not None}
    worst_season = min(season_q05.values()) if season_q05 else float("nan")
    verdict["C4_season_worst_lcb"] = dict(fullrefit_q05_by_subgroup=season_q05,
                                          worst=float(worst_season),
                                          passed=bool(worst_season > -DELTA_SAFE))
    ret = point["trend_retention"]["retention"]
    verdict["C5_trend_retention"] = dict(retention=ret,
                                         passed=(bool(ret >= 0.5) if ret is not None else None))
    rp_path = root / LT_DIR_NAME / "reporter_panel_primary_no_Sar.json"
    if rp_path.exists():
        rp = json.loads(rp_path.read_text("utf-8"))
        verdict["C6_reporter"] = rp["gate"]
    else:
        verdict["C6_reporter"] = dict(passed=None, note="reporter panel 未跑")
    all_q05 = {s: q for s, q in fr["subgroup_dp_delta_q05"].items() if q is not None}
    worst_all = min(all_q05, key=all_q05.get) if all_q05 else None
    report = dict(
        freeze_config_sha=freeze["config_sha"] if freeze else None,
        smoke_dev=args.smoke_dev,
        verdict=verdict,
        gates_passed=all(v.get("passed") for v in verdict.values()
                         if isinstance(v, dict) and v.get("passed") is not None),
        all_gates_decided=all(isinstance(v, dict) and v.get("passed") is not None
                              for v in verdict.values()),
        report_only=dict(
            overall_worst_group=dict(subgroup=worst_all,
                                     fullrefit_q05=all_q05.get(worst_all),
                                     note="不作门；不得写'全局安全'（A-40⑦）"),
            cell_equal=point["cell_equal"],
            comparisons_cell_equal=fr["comparisons_cell_equal"],
            report_comparators={f"dp_abstain_vs_{b}": fr["comparisons"].get(f"dp_abstain_vs_{b}")
                                for b in REPORT_COMPARATORS}))
    (root / "confirmatory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    print(f"→ {root / 'confirmatory_report.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["lt", "ci", "repl", "verdict"])
    ap.add_argument("--scope", default="primary_no_Sar", choices=["primary_no_Sar", "all_data"])
    ap.add_argument("--n-boot", type=int, default=FULL_REFIT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_CONF_SEED)
    ap.add_argument("--smoke-dev", action="store_true",
                    help="dev 语料全链路冒烟（不触碰 holdout、写 smoke_dev/、结果不进文档）")
    ap.add_argument("--open-holdout", action="store_true",
                    help="确认一次性打开 seeds 20–39（非 smoke 必须显式给出）")
    args = ap.parse_args()
    if not args.smoke_dev and not args.open_holdout:
        raise SystemExit("门禁：非 smoke 运行读取 holdout，必须显式 --open-holdout（A-41⑦ 一次性动作）。")
    freeze = None
    if not args.smoke_dev:
        freeze, _ = verify_freeze()
        print(f"freeze 核验通过 config_sha={freeze['config_sha']} router_sha={freeze['router']['sha256'][:16]}…",
              flush=True)
    elif FREEZE_PATH.exists():
        freeze = json.loads(FREEZE_PATH.read_text("utf-8"))
    {"lt": stage_lt, "ci": stage_ci, "repl": stage_repl, "verdict": stage_verdict}[args.stage](args, freeze)


if __name__ == "__main__":
    main()
