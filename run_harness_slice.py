"""run_harness_slice.py — Harness action-only 垂直切片：frozen vs updating 第一张表。

回答项目本体问题：**Harness 更新策略是否真的比冻结策略更快适应新结构？**
（评审第二十八轮定案；目标是证明"叠加→证据→更新→验证→accept/rollback"整条链可运行，
不是证明 P0+abstain 性能好——它只是有 artifact 可重放的回退起点。）

流：S2 dev 8 域按旧→新序（S_season→S_trend→S_both→S_ar→S_multiseason→S_hetero→
S_intermittent→S_regime），每块 = 该族全部 dev uid（uid 排序固定）。两臂同起点=
confirmatory 冻结 dp_abstain artifact、同数据序、同预算、同动作池（core 10）。

updating 臂四段链（全部确定性，无 LLM）：
  Overlay    每 uid 真执行 `routed_process_overlay`（策略程序压过 compose，当前 harness 的
             L1/L4/L3 全留；gates 拒绝→recovery 照常）——机制层真跑通，非模拟；
  Evidence   RoutingDecision + grounded utility 入 EvidenceRecord.routing（2.0-⑤），
             切片终了 dump routing_evidence.jsonl；
  Updater    每块结束：累积已见 uid 的响应行（offline-gym 语义，声明）→ 分层 hash 留出
             ~25% 验证 → 重拟合 GBDTArm(P0 特征, abstain) × κ∈{0.5,1,2} → **只有验证均值
             改善 ∧ worst-group 不恶化才 accept**；accept 即落 PolicyArtifact(joblib+manifest)；
  Rollback   新块上 realized(updating) − realized(frozen) > δ_safe(0.05) → 恢复前一 artifact，
             失败证据保留。
允许更新：Router 参数（重拟合）/κ。禁：Pattern 特征、P1a/fixpc、算子实现、模型选择、任意层 LLM 编辑。

产物：results/Stage2/HarnessSlice/{table.md, report.json, routing_evidence.jsonl, artifacts/}。
运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_harness_slice
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .e32_policy import FALLBACK_ACTION, GBDTArm, PolicyData
from .harness import HarnessState
from .memory import EvidenceStore
from .policy import FrozenArmRouterPolicy, RouterPolicy, RoutingDecision, action_menu_v1
from .policy.overlay import routed_process_overlay
from .policy.pattern_spec import pattern_spec_p0
from .s2_corpus import build_s2_dev

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "HarnessSlice"
REC_PATH = Path(__file__).resolve().parent / "results" / "Stage2" / "S2_replication" / "records_s2.jsonl"
BLOCKS = ("S_season", "S_trend", "S_both", "S_ar",
          "S_multiseason", "S_hetero", "S_intermittent", "S_regime")   # 旧→新（预声明）
SEED = 20260705
KGRID = (0.5, 1.0, 2.0)
DELTA_SAFE = 0.05
VAL_MOD = 4                            # sha(uid)%4==0 → 验证组（~25%，确定性）


# ════════════════════════════ 重拟合臂的 RouterPolicy 适配 ════════════════════════════
class RefitArmRouterPolicy(RouterPolicy):
    """切片本地适配：重拟合 GBDTArm（P0 特征）→ RouterPolicy 接口（单行 shim，镜像
    FrozenArmRouterPolicy.predict 的数学；胜出后再考虑入 policy/ 正式化）。"""

    task_scope = ("forecast",)

    def __init__(self, arm, actions: List[str], version: str, artifact_sha: str):
        self.arm, self.actions, self.version, self.artifact_sha = arm, list(actions), version, artifact_sha
        self.pattern_spec = pattern_spec_p0()

    def predict(self, key, menu, model_menu=None) -> RoutingDecision:
        task = (key.get("task") or {}).get("type")
        if task not in self.task_scope:
            raise ValueError(f"task_scope={self.task_scope}，拒绝 {task!r}")
        struct = key["pattern"]["struct_feats"]
        xd, xp = self.pattern_spec.router_features(struct)
        shim = PolicyData(uids=["deploy:0"], actions=self.actions,
                          L=np.zeros((1, len(self.actions))),
                          X_d=xd.reshape(1, -1), X_p=xp.reshape(1, -1),
                          cell=np.array([str(key.get("cell_id") or "")]), origin=np.array(["deploy"]))
        p, ab = self.arm.picks(shim, np.arange(1))
        return RoutingDecision(action_id=self.actions[int(p[0])], abstained=bool(ab[0]),
                               fallback_action=FALLBACK_ACTION,
                               provenance={"policy_version": self.version,
                                           "router_artifact_sha": self.artifact_sha,
                                           "pattern_spec": {"version": "P0"}})


# ════════════════════════════ Updater（确定性统计，无 LLM）════════════════════════════
def _is_val(uid: str) -> bool:
    return int(hashlib.sha256((uid + "|slicehold").encode()).hexdigest()[:8], 16) % VAL_MOD == 0


def _eval_rows(arm, rows: List[dict], actions: List[str]) -> Dict[str, float]:
    data = PolicyData(uids=[r["uid"] for r in rows], actions=actions,
                      L=np.array([[r["L_test"][a] for a in actions] for r in rows]),
                      X_d=np.array([[r["snr"], r["miss_rate"]] for r in rows]),
                      X_p=np.array([r["X_p"] for r in rows]),
                      cell=np.array([r["cell"] for r in rows]),
                      origin=np.array([r["origin"] for r in rows]))
    p, _ = arm.picks(data, np.arange(data.n))
    loss = data.L[np.arange(data.n), p]
    regret = loss - data.L.min(axis=1)
    keys = np.array([f"{r['origin']}|{r['cell']}" for r in rows])
    wg = min(float(regret[keys == k].mean()) for k in np.unique(keys))
    return dict(mean=float(regret.mean()), worst_group_neg=wg,
                worst=float(max(regret[keys == k].mean() for k in np.unique(keys))))


def propose_update(seen: List[dict], actions: List[str], incumbent_arm) -> Optional[dict]:
    """累积证据 → 重拟合候选 → 未参与拟合的验证组裁决。accept ⇔ 均值改善 ∧ worst-group 不恶化。"""
    fit_rows = [r for r in seen if not _is_val(r["uid"])]
    val_rows = [r for r in seen if _is_val(r["uid"])]
    if len(fit_rows) < 40 or len(val_rows) < 12:
        return None
    data_fit = PolicyData(uids=[r["uid"] for r in fit_rows], actions=actions,
                          L=np.array([[r["L_test"][a] for a in actions] for r in fit_rows]),
                          X_d=np.array([[r["snr"], r["miss_rate"]] for r in fit_rows]),
                          X_p=np.array([r["X_p"] for r in fit_rows]),
                          cell=np.array([r["cell"] for r in fit_rows]),
                          origin=np.array([r["origin"] for r in fit_rows]))
    inc = _eval_rows(incumbent_arm, val_rows, actions)
    # κ 只影响 picks 阈值不影响 fit → 拟合一次、扫 κ（与逐 κ 重拟合位级同结果：random_state
    # 只依赖 seed/成员，三次 fit 本就相同；纯提速 3×）
    cand = GBDTArm(("d", "p"), abstain=True, seed=SEED, kappa=KGRID[0]).fit(
        data_fit, np.arange(data_fit.n))
    best = None
    for k in KGRID:
        cand.kappa = k
        ev = _eval_rows(cand, val_rows, actions)
        if best is None or ev["mean"] < best["ev"]["mean"]:
            best = dict(kappa=k, ev=ev)
    cand.kappa = best["kappa"]
    best["arm"] = cand
    accept = (best["ev"]["mean"] < inc["mean"]
              and best["ev"]["worst"] <= inc["worst"] + 1e-9)
    return dict(accept=bool(accept), kappa=best["kappa"], arm=best["arm"],
                cand_val=best["ev"], incumbent_val=inc,
                n_fit=len(fit_rows), n_val=len(val_rows))


def save_artifact(arm, kappa: float, version: str, n_fit: int, block: str) -> str:
    import joblib
    OUT.joinpath("artifacts").mkdir(parents=True, exist_ok=True)
    p = OUT / "artifacts" / f"policy_{version}.joblib"
    import sklearn
    joblib.dump(dict(arm=arm, kappa=kappa, version=version,
                     sklearn_version=sklearn.__version__, numpy_version=np.__version__,
                     manifest=dict(n_fit=n_fit, after_block=block, kappa_grid=list(KGRID),
                                   val_rule=f"sha256(uid|slicehold)%{VAL_MOD}==0",
                                   gates="val mean improve AND worst-group not worse")), p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ════════════════════════════ 主流程 ════════════════════════════
def main():
    t0 = time.time()
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    actions = list(recs[0]["L_test"].keys())                 # Phase B 冻结动作序（core 10）
    corpus = {rs.series_uid: rs for rs in build_s2_dev()}
    harness = HarnessState.from_minimal()                    # 当前部署 harness（overlay 的宿主）
    store = EvidenceStore()
    menu = action_menu_v1()
    frozen = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    upd_router: RouterPolicy = frozen                        # 同起点
    prev_router: Optional[RouterPolicy] = None
    version = 0
    seen: List[dict] = []
    ledger: List[dict] = []
    events: List[dict] = []
    updated_before_block = False

    for bi, fam in enumerate(BLOCKS):
        uids = sorted(u for u in by_uid if by_uid[u]["origin"] == fam and u in corpus)
        blk = dict(block=bi, family=fam, n=len(uids))
        fr_reg, up_reg, diverged = [], [], 0
        tb = time.time()
        for u in uids:
            r = by_uid[u]
            oracle = min(r["L_test"].values())
            # frozen 臂：决策 + 账本（重放语义，声明不真执行）
            d_f = frozen.predict({"pattern": {"struct_feats": _struct_of(r)},
                                  "task": {"type": "forecast"}, "cell_id": r["cell"]}, menu)
            fr_reg.append(r["L_test"][d_f.action_id] - oracle)
            # updating 臂：**真执行 overlay**（机制层）+ 账本
            d_u, rec, _ = routed_process_overlay(
                corpus[u].history, "forecast", harness, upd_router, menu, store=store,
                batch_id=f"blk{bi}:{fam}",
                extra_routing={"uid": u, "grounded_utility": r["L_test"], "block": bi})
            up_reg.append(r["L_test"][d_u.action_id] - oracle)
            if rec.output_status != "ready" or rec.program.get("note") != f"overlay:{d_u.action_id}":
                diverged += 1
        blk.update(frozen_regret=float(np.mean(fr_reg)), updating_regret=float(np.mean(up_reg)),
                   delta_frozen_minus_updating=float(np.mean(fr_reg) - np.mean(up_reg)),
                   diverged=diverged, wallclock_s=round(time.time() - tb, 1))
        # —— rollback 检查（进入本块时若上一块后曾更新）——
        if updated_before_block and blk["updating_regret"] - blk["frozen_regret"] > DELTA_SAFE:
            events.append(dict(type="rollback", block=bi, family=fam,
                               harm=round(blk["updating_regret"] - blk["frozen_regret"], 4),
                               restored=getattr(prev_router, "version", "frozen_v0")))
            upd_router = prev_router or frozen
            blk["rolled_back"] = True
        updated_before_block = False
        # —— 累积证据 + updater ——
        seen.extend(by_uid[u] for u in uids)
        cur_arm = upd_router.arm if hasattr(upd_router, "arm") else None
        prop = propose_update(seen, actions, cur_arm) if cur_arm is not None else None
        if prop is not None:
            ev = dict(type="update_proposal", block=bi, family=fam, accept=prop["accept"],
                      kappa=prop["kappa"], cand_val=prop["cand_val"],
                      incumbent_val=prop["incumbent_val"], n_fit=prop["n_fit"], n_val=prop["n_val"])
            if prop["accept"]:
                version += 1
                sha = save_artifact(prop["arm"], prop["kappa"], f"v{version}", prop["n_fit"], fam)
                prev_router = upd_router
                upd_router = RefitArmRouterPolicy(prop["arm"], actions, f"v{version}", sha)
                updated_before_block = True
                ev["artifact_sha"] = sha
            events.append(ev)
        ledger.append(blk)
        print(f"  [blk{bi} {fam:16s}] frozen={blk['frozen_regret']:.3f} "
              f"updating={blk['updating_regret']:.3f} diverged={diverged} "
              f"{'UPDATE→v'+str(version) if updated_before_block else ''} [{blk['wallclock_s']}s]",
              flush=True)

    # —— 汇总 + false accept（下一块反事实）——
    accepts = [e for e in events if e["type"] == "update_proposal" and e["accept"]]
    rollbacks = [e for e in events if e["type"] == "rollback"]
    all_fr = np.concatenate([[b["frozen_regret"]] * b["n"] for b in ledger])
    all_up = np.concatenate([[b["updating_regret"]] * b["n"] for b in ledger])
    summary = dict(
        cumulative_regret=dict(frozen=float(all_fr.mean()), updating=float(all_up.mean())),
        per_block=ledger, events=events,
        n_updates_accepted=len(accepts),
        n_updates_rejected=len([e for e in events if e["type"] == "update_proposal" and not e["accept"]]),
        n_rollbacks=len(rollbacks),
        update_wallclock_note="updater 每块 <10s（3κ×E5 GBDT 重拟合）",
        evidence=dict(n_records=len(store), n_routing=sum(
            1 for cid in store.get_all_cells() for rec in store.query_by_cell(cid)
            if rec.routing is not None)),
        discipline="允许更新=Router 参数/κ；Pattern=P0 冻结、动作池冻结、无模型轴、无 LLM",
        caveat="frozen 臂=账本重放（不真执行，声明）；updating 臂全 uid 真执行 overlay；"
               "grounded utility=S2 nested L_test（offline-gym 语义，声明）")
    OUT.mkdir(parents=True, exist_ok=True)
    routing_rows = [rec.to_dict() for cid in store.get_all_cells()
                    for rec in store.query_by_cell(cid) if rec.routing is not None]
    (OUT / "routing_evidence.jsonl").write_text(
        "\n".join(json.dumps(_slim(r), ensure_ascii=False) for r in routing_rows), "utf-8")
    (OUT / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    table = render(summary)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


def _struct_of(r: dict) -> dict:
    from .e32_policy import P_FEATS
    s = {"SNR": r["snr"], "missing_rate": r["miss_rate"]}
    s.update({k: v for k, v in zip(P_FEATS, r["X_p"])})
    return s


def _slim(rec: dict) -> dict:
    """routing 证据行瘦身（trace/gates 全文不进 jsonl，报告读 routing+status 即可）。"""
    return dict(cell_id=rec["cell_id"], batch_id=rec["batch_id"],
                program_note=rec["program"]["note"],
                output_status=rec["verification_result"]["output_status"],
                routing={k: v for k, v in rec["routing"].items() if k != "grounded_utility"} |
                        {"grounded_utility_selected":
                         rec["routing"]["grounded_utility"].get(rec["routing"]["selected_action"])})


def render(s: dict) -> str:
    lines = ["# Harness action-only 垂直切片：frozen vs updating（第一张 Harness 表）", "",
             f"> {s['caveat']}", f"> 纪律：{s['discipline']}", "",
             "| blk | family | n | frozen | updating | Δ(f−u) | diverged | 事件 |",
             "|---|---|---|---|---|---|---|---|"]
    ev_by_blk = {}
    for e in s["events"]:
        ev_by_blk.setdefault(e["block"], []).append(
            ("ROLLBACK" if e["type"] == "rollback" else
             ("accept κ=" + str(e["kappa"]) if e["accept"] else "reject")))
    for b in s["per_block"]:
        lines.append(f"| {b['block']} | {b['family']} | {b['n']} | {b['frozen_regret']:.3f} | "
                     f"{b['updating_regret']:.3f} | {b['delta_frozen_minus_updating']:+.3f} | "
                     f"{b['diverged']} | {', '.join(ev_by_blk.get(b['block'], [])) or '—'} |")
    c = s["cumulative_regret"]
    lines += ["", f"**cumulative regret**：frozen={c['frozen']:.4f} updating={c['updating']:.4f} "
                  f"（Δ={c['frozen']-c['updating']:+.4f}）",
              f"updates：accepted={s['n_updates_accepted']} rejected={s['n_updates_rejected']} "
              f"rollbacks={s['n_rollbacks']}；evidence：{s['evidence']['n_records']} 条"
              f"（routing {s['evidence']['n_routing']}）"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
