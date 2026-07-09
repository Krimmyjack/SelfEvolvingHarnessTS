"""run_skill_slice.py — LLM-Skill 垂直切片（四臂 A/B/C/D；prereg_skill_slice.md，
第六张设计决策表——LLM 首次上场）。

A=frozen dp_abstain incumbent（守卫：重算 ≡ Updater2 frozen ckpt）；
B=wrapper 守卫臂（skill 层往返编译，设计上 ≡ A bit 级，不占统计位）；
C=LLM composer 全 16 半块；D=升级式（BlockGate flag ∪ 各流 block0 → LLM，否则 ≡ A）。
块决策唯一化（16 个，(family,half) 键）：DataView 块局部 → 决策与排列位置无关（声明）。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_skill_slice [--smoke]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm import get_client
from .policy import FrozenArmRouterPolicy
from .policy.dataview import _rep_uids, build_block_views
from .policy.skill_composer import compile_block_policy, decide_block
from .policy.skills import action_to_skill, compile_skill, skills_sha
from .run_updater2 import OUT as OUT2, REC_PATH, Served, half_of, locked_permutations
from .s2_corpus import make_series

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "SkillSlice"
GATE_REPORT = Path(__file__).resolve().parent / "results" / "Stage2" / "BlockGate" / "report.json"
DELTA_SAFE = 0.05
BOOT_B = 2000
SEED = 20260706
ARMS = ("A_frozen", "B_wrapper", "C_llm_all", "D_escalation")


def _hist_of(uid: str) -> np.ndarray:
    _, fam, dname, j = uid.split(":")
    return make_series(fam, dname, int(j)).history


def _regret(rows: List[dict], pick_of) -> np.ndarray:
    oracle = np.array([min(r["L_test"].values()) for r in rows])
    return np.array([r["L_test"][pick_of(r)] for r in rows]) - oracle


def grouped_boot(diff_groups: List[np.ndarray], b: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(b)
    for k in range(b):
        idx = rng.integers(0, len(diff_groups), len(diff_groups))
        means[k] = float(np.concatenate([diff_groups[i] for i in idx]).mean())
    point = float(np.concatenate(diff_groups).mean())
    return point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只做前 2 个块决策并打印，不出表")
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    blocks_uids: Dict[Tuple[str, int], List[str]] = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for k in blocks_uids:
        blocks_uids[k] = sorted(blocks_uids[k])
    frozen_pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    frozen_srv = Served("frozen", frozen_pol.arm, frozen_pol.actions)
    perms = locked_permutations()
    gate_rows = json.loads(GATE_REPORT.read_text("utf-8"))["rows"]
    gate_flag = {(r["perm"], r["block"]): r["flagged"] for r in gate_rows}

    # —— frozen picks（块级缓存）+ 守卫 A/B ——
    fp_of: Dict[Tuple[str, int], List[str]] = {}
    for key in sorted(blocks_uids):
        rows = [by_uid[u] for u in blocks_uids[key]]
        fp_of[key] = frozen_srv.picks(rows)
        for a in fp_of[key]:                                   # G5 wrapper：skill 层往返恒等
            assert compile_skill(*action_to_skill(a)) == a, f"wrapper 往返失败: {a}"
    for pi, order in enumerate(perms):                         # 守卫：A ≡ Updater2 frozen 账本
        ref = json.loads((OUT2 / "ckpt" / f"perm{pi}_frozen.json").read_text("utf-8"))["ledger"]
        for bi, (fam, hh) in enumerate(order):
            rows = [by_uid[u] for u in blocks_uids[(fam, hh)]]
            got = float(_regret(rows, lambda r, k=(fam, hh): fp_of[k][blocks_uids[k].index(r["uid"])]).mean())
            assert abs(got - ref[bi]["regret"]) < 1e-9, f"A 守卫失败 perm{pi} blk{bi}"
    print(f"守卫过：A ≡ frozen 账本（80 块）；B wrapper 往返恒等（skills_sha={skills_sha()}）",
          flush=True)

    # —— 16 个唯一块决策（LLM；磁盘双缓存）——
    dec_path = OUT / "decisions.json"
    decisions: Dict[str, dict] = (json.loads(dec_path.read_text("utf-8"))
                                  if dec_path.exists() else {})
    llm = get_client("flash", temperature=0.0, cache_name="skill_slice")
    keys = sorted(blocks_uids)
    if args.smoke:
        keys = keys[:2]
    for key in keys:
        tag = f"{key[0]}:h{key[1]}"
        if tag in decisions:
            continue
        rows = [by_uid[u] for u in blocks_uids[key]]
        hist_of = {u: _hist_of(u) for u in _rep_uids(rows)}
        views = build_block_views(rows, hist_of, fp_of[key], abstains=[])
        d = decide_block(views, llm, tag)
        decisions[tag] = d
        dec_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=1), "utf-8")
        ok = d["decision"] is not None
        print(f"  [decide] {tag:20s} calls={d['n_calls']} views={d['views_used']} "
              f"{'OK: ' + json.dumps(d['decision']['default']) if ok else 'FAIL→frozen'} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    if args.smoke:
        print("--smoke 完成（不出表）", flush=True)
        return

    # —— 块级 LLM 政策编译 ——
    pol_of: Dict[Tuple[str, int], Optional[Dict[str, str]]] = {}
    for key in sorted(blocks_uids):
        rows = [by_uid[u] for u in blocks_uids[key]]
        pol_of[key] = compile_block_policy(decisions[f"{key[0]}:h{key[1]}"]["decision"], rows)

    # —— 四臂流重放 ——
    ledgers: Dict[str, list] = {a: [] for a in ARMS}
    uid_reg: Dict[str, Dict[Tuple[int, int], np.ndarray]] = {a: {} for a in ARMS}
    llm_fail = 0
    for pi, order in enumerate(perms):
        led = {a: [] for a in ARMS}
        first_seen: set = set()
        for bi, (fam, hh) in enumerate(order):
            key = (fam, hh)
            rows = [by_uid[u] for u in blocks_uids[key]]
            idx_of = {u: i for i, u in enumerate(blocks_uids[key])}
            fr = _regret(rows, lambda r: fp_of[key][idx_of[r["uid"]]])
            pol = pol_of[key]
            lr = _regret(rows, lambda r: pol[r["uid"]]) if pol is not None else fr
            if pol is None:
                llm_fail += 1
            trig = bool(gate_flag.get((pi, bi), False)) or bi == 0
            per_arm = {"A_frozen": (fr, "frozen"), "B_wrapper": (fr, "frozen"),
                       "C_llm_all": (lr, "llm" if pol is not None else "frozen_fallback"),
                       "D_escalation": ((lr, "llm" if pol is not None else "frozen_fallback")
                                        if trig else (fr, "frozen"))}
            for arm, (reg, served) in per_arm.items():
                led[arm].append(dict(block=bi, family=fam, half=hh,
                                     first=fam not in first_seen, triggered=trig,
                                     regret=float(reg.mean()), frozen_regret=float(fr.mean()),
                                     served=served))
                uid_reg[arm][(pi, bi)] = reg
            first_seen.add(fam)
        for arm in ARMS:
            ledgers[arm].append(led[arm])

    # —— 汇总与判据 ——
    res: dict = {"arms": {}, "gates": {}, "diagnostics": {}}
    for arm in ARMS:
        cums = [float(np.mean([b["regret"] for b in led])) for led in ledgers[arm]]
        lb = [b for led in ledgers[arm] for b in led if b["served"] == "llm"]
        diffs = [b["regret"] - b["frozen_regret"] for b in lb]
        res["arms"][arm] = dict(
            cum_mean=float(np.mean(cums)), cum_range=[float(min(cums)), float(max(cums))],
            llm_blocks=len(lb),
            llm_mean_delta_vs_frozen=float(np.mean(diffs)) if diffs else 0.0,
            llm_max_block_harm=float(np.max(diffs)) if diffs else 0.0)
    a, c, d = res["arms"]["A_frozen"], res["arms"]["C_llm_all"], res["arms"]["D_escalation"]
    g2_groups = [uid_reg["D_escalation"][k] - uid_reg["A_frozen"][k]
                 for k in sorted(uid_reg["A_frozen"])]
    g2_pt, g2_lo, g2_hi = grouped_boot(g2_groups, BOOT_B, SEED)
    res["gates"] = dict(
        G1_safety_C=dict(mean_ok=bool(c["llm_mean_delta_vs_frozen"] <= 0),
                         max_ok=bool(c["llm_max_block_harm"] < DELTA_SAFE)),
        G1_safety_D=dict(mean_ok=bool(d["llm_mean_delta_vs_frozen"] <= 0),
                         max_ok=bool(d["llm_max_block_harm"] < DELTA_SAFE)),
        G2_value_D_vs_A=dict(point=g2_pt, ci=[g2_lo, g2_hi],
                             passed=bool(d["cum_mean"] < a["cum_mean"] and g2_hi < 0)),
        G4_cost=dict(d_triggered_unique=len({(b["family"], b["half"])
                                             for led in ledgers["D_escalation"] for b in led
                                             if b["served"] != "frozen"}),
                     c_unique=16, passed=None),               # 填于下
        G5_wrapper="A≡B bit 级（守卫已 assert）")
    res["gates"]["G4_cost"]["passed"] = bool(
        res["gates"]["G4_cost"]["d_triggered_unique"] <= 0.6 * 16)
    # G3 + 诊断
    sboth_first, sboth_missed_c = [], []
    for pi in range(len(perms)):
        for b in ledgers["C_llm_all"][pi]:
            if b["family"] == "S_both" and b["first"]:
                sboth_first.append(dict(perm=pi, block=b["block"],
                                        llm=round(b["regret"], 4),
                                        frozen=round(b["frozen_regret"], 4),
                                        d_triggered=bool(b["triggered"])))
                if not b["triggered"]:
                    sboth_missed_c.append(b["regret"] - b["frozen_regret"])
    res["gates"]["G3_compositional"] = dict(
        blocks=sboth_first,
        llm_beats_frozen=bool(np.mean([x["llm"] for x in sboth_first])
                              < np.mean([x["frozen"] for x in sboth_first])))
    nt = [b["regret"] - b["frozen_regret"] for led in ledgers["C_llm_all"] for b in led
          if not b["triggered"]]
    res["diagnostics"] = dict(
        llm_failures=llm_fail,
        c_minus_frozen_on_nontriggered=float(np.mean(nt)) if nt else 0.0,
        c_on_gate_missed_sboth=([float(np.mean(sboth_missed_c))] if sboth_missed_c else "无漏网 S_both 首遇块"),
        views_requested={t: dec["views_used"][4:] for t, dec in decisions.items()},
        per_family_c_delta={fam: float(np.mean([b["regret"] - b["frozen_regret"]
                                                for led in ledgers["C_llm_all"] for b in led
                                                if b["family"] == fam]))
                            for fam in sorted({k[0] for k in blocks_uids})})
    g1c, g1d = res["gates"]["G1_safety_C"], res["gates"]["G1_safety_D"]
    passed_core = (g1d["mean_ok"] and g1d["max_ok"] and res["gates"]["G2_value_D_vs_A"]["passed"])
    res["verdict"] = ("PASS：升级式架构转正候选 → 下一轮 B+ 臂 + 慢路径 proposer 预注册"
                      if passed_core else
                      "G1 过 G2 败：LLM 无害无增值 → view log 收割进 P1b" if
                      (g1d["mean_ok"] and g1d["max_ok"]) else
                      "G1 败：deployment LLM composer 在此信息面被拒；view log 仍收割")
    res["config"] = dict(seed=SEED, skills_sha=skills_sha(), boot_b=BOOT_B,
                         delta_safe=DELTA_SAFE, llm="flash t=0 cache=skill_slice",
                         prereg="results/Stage2/prereg_skill_slice.md",
                         trigger="BlockGate flag ∪ block0（prereg §2 锁）")
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")

    lines = ["# LLM-Skill 四臂切片（16 半块 × 5 预锁排列；prereg_skill_slice.md）", "",
             "| arm | cum (5 排列均值 [min,max]) | LLM 块数 | LLM 块 Δ vs frozen (mean/max) |",
             "|---|---|---|---|"]
    for arm in ARMS:
        s = res["arms"][arm]
        lines.append(f"| {arm} | {s['cum_mean']:.4f} [{s['cum_range'][0]:.3f},{s['cum_range'][1]:.3f}] | "
                     f"{s['llm_blocks']} | {s['llm_mean_delta_vs_frozen']:+.4f} / "
                     f"{s['llm_max_block_harm']:+.4f} |")
    g2 = res["gates"]["G2_value_D_vs_A"]
    lines += ["", f"G1(C) mean≤0:{g1c['mean_ok']} max<δ:{g1c['max_ok']} | "
                  f"G1(D) mean≤0:{g1d['mean_ok']} max<δ:{g1d['max_ok']}",
              f"G2 D−A: {g2['point']:+.4f} [{g2['ci'][0]:+.4f},{g2['ci'][1]:+.4f}] → {g2['passed']}",
              f"G3 组合靶 S_both 首遇: {json.dumps(res['gates']['G3_compositional']['blocks'], ensure_ascii=False)}",
              f"G4 D 触发 {res['gates']['G4_cost']['d_triggered_unique']}/16 ≤60%: "
              f"{res['gates']['G4_cost']['passed']} | llm_failures={llm_fail}",
              f"**判决**：{res['verdict']}"]
    table = "\n".join(lines) + "\n"
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
