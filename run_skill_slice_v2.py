"""run_skill_slice_v2.py — 切片 v2：信息面/推理/粒度三方消歧（六臂；
prereg_skill_slice_v2.md，第七张设计决策表）。

A_frozen（incumbent）/ C_llm_v2（DataView v2 全程）/ D_llm_v2（升级式）/
C_llm_verify（强制求证对照：v1 面+标注+强制两段）/ Bplus_v2（featurized DataView v2
per-uid GBDT，Phase-B 冻结折 OOF——P1b 候选）/ D_bplus_v2（升级式作用域 B+）。

G-main：触发块上 D_llm_v2 vs D_bplus_v2 paired——"LLM 推理独立价值"主判据。
注册预测：B+ 吸收大部分收益。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_skill_slice_v2
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .e32_policy import GBDTArm, PolicyData
from .llm import get_client
from .policy import FrozenArmRouterPolicy
from .policy.dataview import _rep_uids, build_block_views_v2, featurize_uid_v2
from .policy.skill_composer import compile_block_policy, decide_block_v2
from .policy.skills import skills_sha
from .run_skill_slice import _hist_of, _regret, grouped_boot
from .run_updater2 import OUT as OUT2, REC_PATH, Served, half_of, locked_permutations

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "SkillSliceV2"
GATE_REPORT = Path(__file__).resolve().parent / "results" / "Stage2" / "BlockGate" / "report.json"
ARMIN_DIR = Path(__file__).resolve().parent / "results" / "Stage2" / "S2_replication" / "ckpt_armin"
DELTA_SAFE = 0.05
BOOT_B = 2000
SEED_FIT = 20260705                                  # Phase-C 臂拟合口径
SEED_BOOT = 20260706
ARMS = ("A_frozen", "C_llm_v2", "D_llm_v2", "C_llm_verify", "Bplus_v2", "D_bplus_v2")


def bplus_features(by_uid: Dict[str, dict], verbose=True) -> Dict[str, dict]:
    path = OUT / "bplus_features.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    feats: Dict[str, dict] = {}
    t0 = time.time()
    for i, u in enumerate(sorted(by_uid)):
        x_d, x_p = featurize_uid_v2(by_uid[u], _hist_of(u))
        feats[u] = dict(d=[float(v) for v in x_d], p=[float(v) for v in x_p])
        if verbose and (i + 1) % 150 == 0:
            print(f"  [feat] {i+1}/{len(by_uid)} [{time.time()-t0:.0f}s]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feats, ensure_ascii=False), "utf-8")
    print(f"  [feat] 完成 {len(feats)} uid [{time.time()-t0:.0f}s]", flush=True)
    return feats


def bplus_oof_picks(by_uid: Dict[str, dict], actions: List[str],
                    feats: Dict[str, dict]) -> Dict[str, str]:
    """Phase-B 冻结折逐字复用：fit on L_train / OOF picks（P1a/fixpc/sq 同协议）。"""
    picks: Dict[str, str] = {}
    for f in range(5):
        det = json.loads((ARMIN_DIR / f"fold_armin{f}.json").read_text("utf-8"))["detail"]
        tr, te = sorted(det["train_uids"]), sorted(det["test_uids"])
        order = tr + te
        L = np.full((len(order), len(actions)), np.nan)
        for i, u in enumerate(order):
            for j, a in enumerate(actions):
                if u in det["L_train"] and a in det["L_train"][u]:
                    L[i, j] = det["L_train"][u][a]
        data = PolicyData(uids=order, actions=actions, L=L,
                          X_d=np.array([feats[u]["d"] for u in order]),
                          X_p=np.array([feats[u]["p"] for u in order]),
                          cell=np.array([by_uid[u]["cell"] for u in order]),
                          origin=np.array([by_uid[u]["origin"] for u in order]))
        arm = GBDTArm(("d", "p"), seed=SEED_FIT).fit(data, np.arange(len(tr)))
        p, _ = arm.picks(data, np.arange(len(tr), len(order)))
        for i, u in enumerate(te):
            picks[u] = actions[int(p[i])]
        print(f"  [B+] fold {f} OOF picks 完成（n={len(te)}）", flush=True)
    return picks


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    actions = list(recs[0]["L_test"].keys())
    blocks_uids: Dict[Tuple[str, int], List[str]] = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for k in blocks_uids:
        blocks_uids[k] = sorted(blocks_uids[k])
    frozen_pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    frozen_srv = Served("frozen", frozen_pol.arm, frozen_pol.actions)
    perms = locked_permutations()
    gate_flag = {(r["perm"], r["block"]): r["flagged"]
                 for r in json.loads(GATE_REPORT.read_text("utf-8"))["rows"]}
    fp_of = {key: frozen_srv.picks([by_uid[u] for u in blocks_uids[key]])
             for key in sorted(blocks_uids)}
    for pi, order in enumerate(perms):                       # 守卫：A ≡ 账本
        ref = json.loads((OUT2 / "ckpt" / f"perm{pi}_frozen.json").read_text("utf-8"))["ledger"]
        for bi, (fam, hh) in enumerate(order):
            rows = [by_uid[u] for u in blocks_uids[(fam, hh)]]
            idx = {u: i for i, u in enumerate(blocks_uids[(fam, hh)])}
            got = float(_regret(rows, lambda r: fp_of[(fam, hh)][idx[r["uid"]]]).mean())
            assert abs(got - ref[bi]["regret"]) < 1e-9
    print("守卫过：A ≡ frozen 账本", flush=True)

    # —— B+ ——
    feats = bplus_features(by_uid)
    bp_path = OUT / "bplus_picks.json"
    bp = (json.loads(bp_path.read_text("utf-8")) if bp_path.exists()
          else bplus_oof_picks(by_uid, actions, feats))
    bp_path.write_text(json.dumps(bp, ensure_ascii=False), "utf-8")

    # —— LLM 决策（v2 / verify 各 16）——
    llm = get_client("flash", temperature=0.0, cache_name="skill_slice")
    decs: Dict[str, Dict[str, dict]] = {}
    for mode in ("v2", "verify"):
        path = OUT / f"decisions_{mode}.json"
        decs[mode] = json.loads(path.read_text("utf-8")) if path.exists() else {}
        for key in sorted(blocks_uids):
            tag = f"{key[0]}:h{key[1]}"
            if tag in decs[mode]:
                continue
            rows = [by_uid[u] for u in blocks_uids[key]]
            hist_of = {u: _hist_of(u) for u in _rep_uids(rows)}
            views = build_block_views_v2(rows, hist_of, fp_of[key], abstains=[])
            d = decide_block_v2(views, llm, tag, mode)
            decs[mode][tag] = d
            path.write_text(json.dumps(decs[mode], ensure_ascii=False, indent=1), "utf-8")
            ok = d["decision"] is not None
            print(f"  [decide:{mode}] {tag:20s} calls={d['n_calls']} "
                  f"viol={d.get('violation')} "
                  f"{'OK: ' + json.dumps(d['decision']['default']) if ok else 'FAIL→frozen'} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    pol: Dict[str, Dict[Tuple[str, int], Optional[Dict[str, str]]]] = {"v2": {}, "verify": {}}
    for mode in ("v2", "verify"):
        for key in sorted(blocks_uids):
            rows = [by_uid[u] for u in blocks_uids[key]]
            pol[mode][key] = compile_block_policy(decs[mode][f"{key[0]}:h{key[1]}"]["decision"], rows)

    # —— 六臂流重放 ——
    ledgers: Dict[str, list] = {a: [] for a in ARMS}
    uid_reg: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {a: {} for a in ARMS}
    for pi, order in enumerate(perms):
        led = {a: [] for a in ARMS}
        first_seen: set = set()
        for bi, (fam, hh) in enumerate(order):
            key = (fam, hh)
            rows = [by_uid[u] for u in blocks_uids[key]]
            idx = {u: i for i, u in enumerate(blocks_uids[key])}
            fr = _regret(rows, lambda r: fp_of[key][idx[r["uid"]]])
            lr = {m: (_regret(rows, lambda r: pol[m][key][r["uid"]])
                      if pol[m][key] is not None else fr) for m in ("v2", "verify")}
            bpr = _regret(rows, lambda r: bp[r["uid"]])
            trig = bool(gate_flag.get((pi, bi), False)) or bi == 0
            per_arm = {
                "A_frozen": (fr, "frozen"),
                "C_llm_v2": (lr["v2"], "llm"),
                "D_llm_v2": (lr["v2"], "llm") if trig else (fr, "frozen"),
                "C_llm_verify": (lr["verify"], "llm"),
                "Bplus_v2": (bpr, "bplus"),
                "D_bplus_v2": (bpr, "bplus") if trig else (fr, "frozen"),
            }
            for arm, (reg, served) in per_arm.items():
                led[arm].append(dict(block=bi, family=fam, half=hh,
                                     first=fam not in first_seen, triggered=trig,
                                     regret=float(reg.mean()), frozen_regret=float(fr.mean()),
                                     served=served))
                uid_reg[arm][(pi, bi)] = reg
            first_seen.add(fam)
        for arm in ARMS:
            ledgers[arm].append(led[arm])

    # —— 汇总 ——
    res: dict = {"arms": {}, "gates": {}, "diagnostics": {}}
    for arm in ARMS:
        cums = [float(np.mean([b["regret"] for b in led])) for led in ledgers[arm]]
        nb = [b for led in ledgers[arm] for b in led if b["served"] != "frozen"]
        diffs = [b["regret"] - b["frozen_regret"] for b in nb]
        res["arms"][arm] = dict(
            cum_mean=float(np.mean(cums)), cum_range=[float(min(cums)), float(max(cums))],
            upgrade_blocks=len(nb),
            up_mean_delta=float(np.mean(diffs)) if diffs else 0.0,
            up_max_harm=float(np.max(diffs)) if diffs else 0.0,
            g1=dict(mean_ok=bool(not diffs or np.mean(diffs) <= 0),
                    max_ok=bool(not diffs or np.max(diffs) < DELTA_SAFE)))
    a = res["arms"]["A_frozen"]
    for arm in ("D_llm_v2", "Bplus_v2", "D_bplus_v2"):
        groups = [uid_reg[arm][k] - uid_reg["A_frozen"][k] for k in sorted(uid_reg["A_frozen"])]
        pt, lo, hi = grouped_boot(groups, BOOT_B, SEED_BOOT)
        res["gates"][f"G2_{arm}_vs_A"] = dict(
            point=pt, ci=[lo, hi],
            passed=bool(res["arms"][arm]["cum_mean"] < a["cum_mean"] and hi < 0))
    trig_keys = [k for k in sorted(uid_reg["A_frozen"])
                 if any(b["block"] == k[1] and b["triggered"]
                        for b in ledgers["D_llm_v2"][k[0]])]
    gm_groups = [uid_reg["D_llm_v2"][k] - uid_reg["D_bplus_v2"][k] for k in trig_keys]
    gm_pt, gm_lo, gm_hi = grouped_boot(gm_groups, BOOT_B, SEED_BOOT)
    res["gates"]["G_main_llm_vs_bplus_on_triggered"] = dict(
        point=gm_pt, ci=[gm_lo, gm_hi], n_groups=len(gm_groups),
        llm_independent_value=bool(gm_hi < 0),
        note="负=LLM 优；注册预测=不成立（B+ 吸收）")
    sboth = []
    for pi in range(len(perms)):
        for b in ledgers["A_frozen"][pi]:
            if b["family"] == "S_both" and b["first"]:
                row = dict(perm=pi, frozen=round(b["frozen_regret"], 4))
                for arm in ("C_llm_v2", "C_llm_verify", "Bplus_v2"):
                    row[arm] = round(ledgers[arm][pi][b["block"]]["regret"], 4)
                sboth.append(row)
    res["gates"]["G3_sboth_first"] = sboth
    res["diagnostics"] = dict(
        llm_failures={m: sum(1 for v in pol[m].values() if v is None) for m in ("v2", "verify")},
        verify_violations=sum(1 for d in decs["verify"].values() if d.get("violation")),
        v2_window_requests=sum(1 for d in decs["v2"].values() if "window" in d["views_used"]),
        per_family_delta={arm: {fam: float(np.mean(
            [b["regret"] - b["frozen_regret"] for led in ledgers[arm] for b in led
             if b["family"] == fam])) for fam in sorted({k[0] for k in blocks_uids})}
            for arm in ("C_llm_v2", "C_llm_verify", "Bplus_v2")},
        trigger_instance_share="40/80=0.50（G4 教训：实例级）")
    res["config"] = dict(seed_fit=SEED_FIT, seed_boot=SEED_BOOT, skills_sha=skills_sha(),
                         prereg="results/Stage2/prereg_skill_slice_v2.md",
                         prediction="B+ 吸收大部分收益；LLM 独立价值不成立")
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")

    lines = ["# 切片 v2：信息面/推理/粒度三方消歧（六臂；prereg_skill_slice_v2.md）", "",
             "| arm | cum (5 排列均值) | 升级块数 | 升级块 Δ vs frozen (mean/max) | G1 |",
             "|---|---|---|---|---|"]
    for arm in ARMS:
        s = res["arms"][arm]
        lines.append(f"| {arm} | {s['cum_mean']:.4f} | {s['upgrade_blocks']} | "
                     f"{s['up_mean_delta']:+.4f} / {s['up_max_harm']:+.4f} | "
                     f"{'✅' if s['g1']['mean_ok'] and s['g1']['max_ok'] else '❌'} |")
    gm = res["gates"]["G_main_llm_vs_bplus_on_triggered"]
    lines += ["", "G2 vs A: " + " | ".join(
        f"{arm}: {res['gates'][f'G2_{arm}_vs_A']['point']:+.4f} "
        f"[{res['gates'][f'G2_{arm}_vs_A']['ci'][0]:+.4f},{res['gates'][f'G2_{arm}_vs_A']['ci'][1]:+.4f}]"
        f"→{res['gates'][f'G2_{arm}_vs_A']['passed']}"
        for arm in ("D_llm_v2", "Bplus_v2", "D_bplus_v2")),
        f"**G-main（触发块 LLM−B+）**: {gm['point']:+.4f} [{gm['ci'][0]:+.4f},{gm['ci'][1]:+.4f}] "
        f"→ LLM 独立价值 {'成立' if gm['llm_independent_value'] else '不成立（=注册预测）'}",
        f"G3 S_both 首遇: {json.dumps(sboth, ensure_ascii=False)}",
        f"诊断: {json.dumps({k: v for k, v in res['diagnostics'].items() if k != 'per_family_delta'}, ensure_ascii=False)}"]
    table = "\n".join(lines) + "\n"
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
