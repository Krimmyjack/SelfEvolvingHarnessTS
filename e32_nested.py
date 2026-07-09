"""e32_nested.py — E-3.2 正式评估器：action-loss 生成嵌入 policy outer fold（A-39①，评审第十五轮 P0）。

旧流程（`policy_data_from_corpus`，已废）先在全语料算每动作 OOF loss、再事后切 policy 折——
与 A-31 裁定失格的模式同类：给 policy outer-train 标签的 Ridge 头见过 outer-test uid 的
history 窗 → train 标签含 test 分布信息、评估头的训练边界 ≠ policy 边界。

本模块的正确程序（每个 policy outer fold）：
  1) router 训练标签 = **outer-train 内部** inner grouped OOF（inner_k=4，头逐 inner fold 重拟合）；
  2) 评估标签 = 每动作 Ridge 头 **仅在 outer-train∩cell 重拟合** → 对 outer-test 算全动作 loss；
  3) 臂只见 (outer-train 特征+标签) 与 (outer-test 特征)；
  4) policy loss / oracle regret / 安全子群全部用 outer-test 标签。
folds 间相互独立（fold 结果只依赖该 fold 的 uid 划分与 seed）→ per-fold checkpoint/resume
bit 级一致（A-36 精神）。头拟合/评估复用 nested_supply（判官口径与 E-1.1/F0 一致）。

守卫（tests/test_e32_nested.py，全过才允许正式 E-3.2）：污染 outer-test future 不改 picks /
NaN 污染 outer-test 训练窗后 train 标签 bit 级不变（test uid 从未进任何头训练）/ origin 置换
不改 learned 臂 picks / resume ≡ 一次跑。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .e32_policy import D_FEATS, P_FEATS, PolicyData, _summarize
from .nested_supply import _eval_uids, _fit_head, make_folds


# ══════════════════════════════════════════════════════════════════════════
# 折构造（确定性）
# ══════════════════════════════════════════════════════════════════════════
def stratified_folds(uids: Sequence[str], strat_of: Dict[str, str], k: int,
                     seed: int) -> Dict[str, int]:
    """cell×origin 分层的确定性 K 折：层内乱序 round-robin，层间偏移防余数堆积。"""
    rng = np.random.default_rng(seed)
    by: Dict[str, List[str]] = defaultdict(list)
    for u in sorted(uids):
        by[strat_of[u]].append(u)
    fold_of: Dict[str, int] = {}
    offset = 0
    for s in sorted(by):
        us = by[s]
        us = [us[i] for i in rng.permutation(len(us))]
        for i, u in enumerate(us):
            fold_of[u] = (i + offset) % k
        offset += len(us)
    return fold_of


# ══════════════════════════════════════════════════════════════════════════
# 每 fold 的标签生成（嵌入式，P0 修复本体）
# ══════════════════════════════════════════════════════════════════════════
def labels_for_fold(cells_data: Dict[str, dict], actions: Sequence[str], tr_set: set,
                    te_set: set, inner_k: int, seed: int
                    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """→ (L_train{uid:{a:loss}} inner-OOF only-on-outer-train, L_test{uid:{a:loss}} outer-train头)。"""
    L_train: Dict[str, Dict[str, float]] = {}
    L_test: Dict[str, Dict[str, float]] = {}
    for cid in sorted(cells_data):
        cd = cells_data[cid]
        tr_c = [u for u in cd["uids"] if u in tr_set]
        te_c = [u for u in cd["uids"] if u in te_set]
        if not tr_c:
            continue
        inner_fold = make_folds(tr_c, inner_k, seed)          # 只依赖 outer-train
        for a in actions:
            caches = cd["action_caches"][a]
            for f in range(inner_k):                          # ① router 标签：outer-train 内 OOF
                itr = [u for u in tr_c if inner_fold[u] != f]
                ite = [u for u in tr_c if inner_fold[u] == f]
                if not itr or not ite:
                    continue
                head = _fit_head(caches, itr)
                for u, v in _eval_uids(head, caches, ite).items():
                    L_train.setdefault(u, {})[a] = v
            if te_c:                                          # ② 评估标签：头仅在 outer-train 重拟合
                head = _fit_head(caches, tr_c)
                for u, v in _eval_uids(head, caches, te_c).items():
                    L_test.setdefault(u, {})[a] = v
    return L_train, L_test


def _policy_data(cells_data: Dict[str, dict], actions: List[str], order: List[str],
                 L_of: Dict[str, Dict[str, float]], X_p_override: Optional[np.ndarray] = None
                 ) -> PolicyData:
    """按 order 拼 PolicyData；L 缺失处 NaN（臂的 fit 只读 train 行、picks 只读特征）。"""
    uid_cell = {u: cid for cid, cd in cells_data.items() for u in cd["uids"]}
    L = np.full((len(order), len(actions)), np.nan)
    Xd, Xp, Xt, cells, origins = [], [], [], [], []
    for i, u in enumerate(order):
        cd = cells_data[uid_cell[u]]
        f = cd["feats_of"][u]
        for j, a in enumerate(actions):
            if u in L_of and a in L_of[u]:
                L[i, j] = L_of[u][a]
        Xd.append([float(f.get(k, 0.0)) for k in D_FEATS])
        Xp.append([float(f.get(k, 0.0)) for k in P_FEATS])
        Xt.append(list(cd.get("true_d", {}).get(u, (0.0, 0.0))))
        cells.append(uid_cell[u])
        origins.append(cd["origin_of"][u])
    return PolicyData(uids=list(order), actions=actions, L=L,
                      X_d=np.array(Xd), X_p=np.array(Xp) if X_p_override is None else X_p_override,
                      cell=np.array(cells), origin=np.array(origins), X_t=np.array(Xt))


# ══════════════════════════════════════════════════════════════════════════
# 主流程：折列表 → 每折（标签 → 臂拟合 → picks）→ per-uid 记录
# ══════════════════════════════════════════════════════════════════════════
def run_policy_folds(cells_data: Dict[str, dict], actions: List[str],
                     arms_factory: Dict[str, Callable[[], object]],
                     folds: List[Tuple[str, List[str], List[str]]], inner_k: int = 4,
                     seed: int = 0, ckpt_dir: Optional[Path] = None,
                     _stop_after: Optional[int] = None, verbose: bool = True) -> dict:
    """folds = [(name, tr_uids, te_uids)]。返回 {records, fold_details}；per-fold checkpoint。
    fold 计算只依赖 (该折 uid 划分, seed) → resume 与一次跑 bit 级一致。"""
    records: List[dict] = []
    fold_details: List[dict] = []
    for fi, (fname, tr, te) in enumerate(folds):
        if _stop_after is not None and fi >= _stop_after:
            break
        ck = (ckpt_dir / f"fold_{fname}.json") if ckpt_dir else None
        if ck and ck.exists():
            doc = json.loads(ck.read_text("utf-8"))
            records.extend(doc["records"])
            fold_details.append(doc["detail"])
            if verbose:
                print(f"  [fold {fname}] resume（{len(doc['records'])} uid）", flush=True)
            continue
        tr_set, te_set = set(tr), set(te)
        assert not (tr_set & te_set), "outer train/test 必须不交"
        fseed = seed + 7919 * (fi + 1)
        L_train, L_test = labels_for_fold(cells_data, actions, tr_set, te_set, inner_k, fseed)
        order = sorted(tr) + sorted(te)
        # fit 用数据：train 行=inner-OOF 标签，test 行=NaN（结构上不可能读 test loss）
        L_fit = {u: L_train[u] for u in L_train}
        data_fit = _policy_data(cells_data, actions, order, L_fit)
        tr_idx = np.arange(len(tr))
        te_idx = np.arange(len(tr), len(order))
        picks_by_arm: Dict[str, np.ndarray] = {}
        abst_by_arm: Dict[str, np.ndarray] = {}
        for name, mk in arms_factory.items():
            arm = mk().fit(data_fit, tr_idx)
            p, a = arm.picks(data_fit, te_idx)
            picks_by_arm[name], abst_by_arm[name] = p, a
        te_order = order[len(tr):]
        frecs = []
        for i, u in enumerate(te_order):
            cd_row = int(te_idx[i])
            frecs.append(dict(
                uid=u, fold=fname,
                cell=str(data_fit.cell[cd_row]), origin=str(data_fit.origin[cd_row]),
                snr=float(data_fit.X_d[cd_row, 0]), miss_rate=float(data_fit.X_d[cd_row, 1]),
                X_p=[float(x) for x in data_fit.X_p[cd_row]],
                X_t=[float(x) for x in data_fit.X_t[cd_row]],
                L_test={a: float(L_test[u][a]) for a in actions},
                arms={n: dict(pick=actions[int(picks_by_arm[n][i])],
                              abstain=bool(abst_by_arm[n][i])) for n in arms_factory}))
        detail = dict(name=fname, n_train=len(tr), n_test=len(te),
                      train_uids=sorted(tr), test_uids=sorted(te),
                      L_train={u: {a: float(v) for a, v in d.items()} for u, d in L_train.items()})
        records.extend(frecs)
        fold_details.append(detail)
        if ck:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(records=frecs, detail=detail)), "utf-8")
            tmp.replace(ck)
        if verbose:
            print(f"  [fold {fname}] n_tr={len(tr)} n_te={len(te)} 完成", flush=True)
    return dict(records=records, fold_details=fold_details)


def aggregate_records(records: List[dict], actions: List[str],
                      arm_names: Sequence[str]) -> Dict[str, dict]:
    """per-uid 记录 → 各臂 {mean_loss, mean_regret, worst_group, abstain_rate, subgroups,
    per_uid_regret, per_uid_delta}（评估全用 outer-test 标签）。"""
    from .e32_policy import FALLBACK_ACTION
    order = [r["uid"] for r in records]
    L = np.array([[r["L_test"][a] for a in actions] for r in records])
    cell = np.array([r["cell"] for r in records])
    origin = np.array([r["origin"] for r in records])
    data_eval = PolicyData(uids=order, actions=list(actions), L=L,
                           X_d=np.array([[r["snr"], r["miss_rate"]] for r in records]),
                           X_p=np.array([r["X_p"] for r in records]),
                           cell=cell, origin=origin)
    oracle = L.min(axis=1)
    out: Dict[str, dict] = {}
    for n in arm_names:
        pick_idx = np.array([actions.index(r["arms"][n]["pick"]) for r in records])
        abst = np.array([r["arms"][n]["abstain"] for r in records], bool)
        loss_pick = L[np.arange(len(records)), pick_idx]
        s = _summarize(data_eval, loss_pick, abst)
        s["per_uid_regret"] = (loss_pick - oracle).tolist()
        s["per_uid_loss"] = loss_pick.tolist()
        s["per_uid_delta_vs_incumbent"] = (L[:, actions.index(FALLBACK_ACTION)] - loss_pick).tolist()
        out[n] = s
    out["_data_eval"] = data_eval
    out["_uid_order"] = order
    return out


def perm_stat_fn(cells_data: Dict[str, dict], actions: List[str],
                 fold_details: List[dict], L_test_of: Dict[str, Dict[str, float]],
                 dp_factory: Callable[[], object],
                 d_regret_mean: float) -> Tuple[Callable[[np.ndarray], float], List[str]]:
    """判据 (vi) 的统计量闭包：T(X_p) = regret(d_gbdt) − regret(dp_gbdt with X_p)（正值=P 有增量）。
    标签/折全部冻结（fold_details 的 L_train + records 汇出的 L_test_of），只重拟合 dp 臂。
    返回 (stat_fn, uid_sorted)——X_p 行序必须与 uid_sorted 对齐（置换也在该顺序上做）。"""
    uid_sorted = sorted({u for fd in fold_details for u in fd["train_uids"] + fd["test_uids"]})
    pos_of = {u: i for i, u in enumerate(uid_sorted)}

    def stat(X_p: np.ndarray) -> float:
        regrets: List[float] = []
        for fd in fold_details:
            tr, te = fd["train_uids"], fd["test_uids"]
            order = sorted(tr) + sorted(te)
            Xp_rows = X_p[[pos_of[u] for u in order]]
            data_fit = _policy_data(cells_data, actions, order,
                                    {u: fd["L_train"][u] for u in fd["L_train"]},
                                    X_p_override=Xp_rows)
            arm = dp_factory().fit(data_fit, np.arange(len(tr)))
            p, _ = arm.picks(data_fit, np.arange(len(tr), len(order)))
            for i, u in enumerate(order[len(tr):]):
                row = np.array([L_test_of[u][a] for a in actions])
                regrets.append(float(row[p[i]] - row.min()))
        return d_regret_mean - float(np.mean(regrets))

    return stat, uid_sorted
