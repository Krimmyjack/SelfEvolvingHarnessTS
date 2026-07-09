"""run_c1_lite.py — Track C1-lite：P1b **表示** vs episodic **记忆机制** 的因果隔离（压缩计划 P0）。

用户压缩计划（额度受限）：**不跑完整 M0–M3**，只复用缓存记录、新增三臂做便宜因果隔离，回答
"C1 的收益究竟来自 P1b 表示，还是 episodic memory 机制本身"。离线、零新数据、无外部 API、
无一次性资源、无 torch 拟合——探索性 pilot（可重跑），**非 confirmatory**。

四行（frozen 作 P0 参照，直接复用；三臂新增）：
  frozen            现任 dp_abstain（P0 参照，缓存复用，不重跑）
  P1b-static        B+ GBDT 的 per-uid 固定 pick（P1b 表示→动作，**无 memory**；bplus_picks.json 复用）
                    ★注：B+ 为全局 in-sample 拟合 = **leaky 强上界**；作 static 基线时其泄漏优势
                    **偏向 memory 无价值的零假设**（memory 要赢得过它才算稳健）。
  P1b-memory        router_fusion（P1b 键情景记忆，in-support 才覆盖；MPilot_P1b 唯一胜 frozen 者）
  P1b-random-memory router_fusion **同门控/同覆盖率**，但记忆标签被打乱（几何邻居不变，feature→action
                    关联被破坏）→ **leakage-immune** 的"记忆内容是否承重"对照

判据（用户压缩计划 §1）：
  ① 正确 memory 同时胜 P1b-static 和 random-memory（cum regret 更低）；
  ② 收益集中在 recurrence 块 / in-support 序列；
  ③ escalation 后首遇 harm < δ_safe。
  三者全过 → Memory 线存活，进 B1b-mini 后的完整方法；否则关闭 Memory claim（不再跑 M0–M3/双底座/
  cross-domain memory）。random-memory 对照 leakage-immune，是"内容承重"的主检验；B+ 对照回答"相对
  静态估计器的独立价值"（泄漏偏向零假设，memory 胜出即稳健）。

复用 run_mpilot 的块/排列/支持域/情景记忆基建；固定 **k=5，不搜 k**（压缩计划）。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_c1_lite
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .policy import FrozenArmRouterPolicy
from .run_mpilot import (RESULTS, Memory, _bucket_stats, _metrics_from_ledgers,
                         build_support_g, in_support_g, make_featfn)
from .run_updater2 import (DELTA_SAFE, REC_PATH, SEED, Served, half_of,
                           locked_permutations)

K = 5                                              # 固定，不搜（压缩计划）
C1_ARMS = ("frozen", "P1b-static", "P1b-memory", "P1b-random-memory")


def episodic_pick_scrambled(mem: Memory, r: dict, actions: List[str], k: int,
                            perm: np.ndarray) -> str:
    """几何邻居与真实记忆完全一致，但读取的 L_test 来自 perm 重指派的行 → feature→action 关联被破坏，
    覆盖率/动作边际分布保持不变（leakage-immune 的"内容承重"对照）。"""
    z = (mem.featfn(r) - mem._mu) / mem._sd
    d = np.sqrt(((mem._Z - z) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    src = perm[idx]                                # 打乱标签来源
    M = np.array([[mem.rows[i]["L_test"][a] for a in actions] for i in src], float)
    return actions[int(M.mean(axis=0).argmin())]


def run_stream_c1lite(featfn, bplus: Dict[str, str], order, by_uid, blocks_uids,
                      actions, frozen_srv, perm_seed: int) -> Tuple[dict, int]:
    """单排列：一次流式扫描同时算四臂（共享块/排列/in-support 标记，保证可比）。"""
    mem = Memory(featfn)
    rng = np.random.default_rng(perm_seed)
    ledgers = {a: [] for a in C1_ARMS}
    buckets = {a: {"in": [], "out": []} for a in C1_ARMS}
    first_seen: set = set()
    n_missing_bplus = 0
    for bi, (fam, hh) in enumerate(order):
        rows = [by_uid[u] for u in blocks_uids[(fam, hh)]]
        oracle = np.array([min(r["L_test"].values()) for r in rows])
        fr_picks = frozen_srv.picks(rows)
        sup = build_support_g(mem.rows, featfn)
        mem.snapshot()
        ready = mem.ready(K)
        scramble = rng.permutation(mem._Z.shape[0]) if (ready and mem._Z is not None) else None
        arm_picks = {a: [] for a in C1_ARMS}
        is_first = fam not in first_seen
        for r, fp in zip(rows, fr_picks):
            in_sup = in_support_g(sup, r, featfn)
            ep = mem.episodic_pick(r, actions, K) if ready else fp
            ep_rand = (episodic_pick_scrambled(mem, r, actions, K, scramble)
                       if (ready and scramble is not None) else fp)
            bp = bplus.get(r["uid"])
            if bp is None or bp not in r["L_test"]:
                bp = fp
                n_missing_bplus += 1
            picks = {"frozen": fp, "P1b-static": bp,
                     "P1b-memory": ep if (in_sup and ready) else fp,
                     "P1b-random-memory": ep_rand if (in_sup and ready) else fp}
            for a in C1_ARMS:
                arm_picks[a].append(picks[a])
                buckets[a]["in" if in_sup else "out"].append(
                    (float(r["L_test"][fp] - r["L_test"][picks[a]]), is_first))
        for a in C1_ARMS:
            reg = np.array([r["L_test"][p] for r, p in zip(rows, arm_picks[a])]) - oracle
            fr_reg = np.array([r["L_test"][p] for r, p in zip(rows, fr_picks)]) - oracle
            ledgers[a].append(dict(block=bi, family=fam, half=hh, first=is_first,
                                   regret=float(reg.mean()), frozen_regret=float(fr_reg.mean()),
                                   n_override=int(sum(p != f for p, f in zip(arm_picks[a], fr_picks))),
                                   n=len(rows)))
        first_seen.add(fam)
        mem.add(rows)
    runs = {a: dict(ledger=ledgers[a], buckets=buckets[a]) for a in C1_ARMS}
    return runs, n_missing_bplus


def summarize(all_runs: Dict[str, List[dict]]) -> dict:
    out: dict = {"k": K, "arms": {}}
    for a in C1_ARMS:
        m = _metrics_from_ledgers(all_runs[a])
        m["buckets"] = _bucket_stats(all_runs[a])
        out["arms"][a] = m
    mem_c = out["arms"]["P1b-memory"]["cumulative_regret_mean"]
    stat_c = out["arms"]["P1b-static"]["cumulative_regret_mean"]
    rand_c = out["arms"]["P1b-random-memory"]["cumulative_regret_mean"]
    frz_c = out["arms"]["frozen"]["cumulative_regret_mean"]
    mb = out["arms"]["P1b-memory"]["buckets"]
    verdict = dict(
        memory_beats_static=bool(mem_c < stat_c),
        memory_beats_random=bool(mem_c < rand_c),
        memory_beats_frozen=bool(mem_c < frz_c),
        gain_on_recurrence=bool(out["arms"]["P1b-memory"]["recurrence_gain_mean"] > 0.0
                                and mb["in"]["mean_adv"] > mb["out"]["mean_adv"]),
        first_unseen_harm_ok=bool(out["arms"]["P1b-memory"]["first_unseen_harm_max_mean"] < DELTA_SAFE),
        deltas=dict(memory_vs_static=float(stat_c - mem_c), memory_vs_random=float(rand_c - mem_c),
                    memory_vs_frozen=float(frz_c - mem_c)))
    verdict["memory_line_survives"] = bool(
        verdict["memory_beats_static"] and verdict["memory_beats_random"]
        and verdict["gain_on_recurrence"])
    verdict["needs_escalation_gate"] = bool(not verdict["first_unseen_harm_ok"])
    out["verdict"] = verdict
    return out


def render(res: dict) -> str:
    L = ["# C1-lite：P1b 表示 vs episodic 记忆机制 因果隔离（16 半块×5 排列；k=5，非 confirmatory）", "",
         "| arm | cum regret [min,max] | 复发增益 | 首遇 harm(max) | override% | in-sup adv | out-sup adv |",
         "|---|---|---|---|---|---|---|"]
    for a in C1_ARMS:
        s = res["arms"][a]
        b = s["buckets"]
        L.append(f"| {a} | {s['cumulative_regret_mean']:.4f} "
                 f"[{s['cumulative_regret_range'][0]:.3f},{s['cumulative_regret_range'][1]:.3f}] | "
                 f"{s['recurrence_gain_mean']:+.4f} | {s['first_unseen_harm_max_mean']:+.3f} | "
                 f"{s['override_frac_mean']:.2f} | {b['in']['mean_adv']:+.4f}(n{b['in']['n']}) | "
                 f"{b['out']['mean_adv']:+.4f}(n{b['out']['n']}) |")
    v = res["verdict"]
    d = v["deltas"]
    L += ["", "**因果判读（点估计，非门控）**：",
          f"- ① memory 胜 static(B+)：**{v['memory_beats_static']}** "
          f"（Δ={d['memory_vs_static']:+.4f}；B+ 为 leaky 强上界，偏向零假设）",
          f"- ① memory 胜 random-memory：**{v['memory_beats_random']}** "
          f"（Δ={d['memory_vs_random']:+.4f}；leakage-immune 内容承重主检验）",
          f"- （旁证）memory 胜 frozen：**{v['memory_beats_frozen']}**（Δ={d['memory_vs_frozen']:+.4f}）",
          f"- ② 收益集中 recurrence/in-support：**{v['gain_on_recurrence']}**",
          f"- ③ 首遇 harm < δ_safe({DELTA_SAFE})：**{v['first_unseen_harm_ok']}**",
          "",
          f"**结论**：Memory 线存活（①∧②）= **{v['memory_line_survives']}**"
          + ("；需 escalation gate（首遇 harm 超 δ_safe）" if v["needs_escalation_gate"] else "") + "。",
          "",
          "> 存活 → 进 B1b-mini 后完整方法；否则关闭 Memory claim（不跑 M0–M3/双底座/cross-domain）。"
          " random-memory 对照 leakage-immune；B+ 对照泄漏偏向零假设。**pilot，非 confirmatory**。"]
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    OUT = RESULTS / "C1lite"
    OUT.mkdir(parents=True, exist_ok=True)
    featfn, key_name = make_featfn("p1b")
    bplus = json.loads((RESULTS / "SkillSliceV2" / "bplus_picks.json").read_text("utf-8"))
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    actions = list(recs[0]["L_test"].keys())
    blocks_uids: Dict[Tuple[str, int], List[str]] = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for kk in blocks_uids:
        blocks_uids[kk] = sorted(blocks_uids[kk])
    frozen_pol = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    frozen_srv = Served("frozen", frozen_pol.arm, frozen_pol.actions)
    perms = locked_permutations()

    all_runs = {a: [] for a in C1_ARMS}
    total_missing = 0
    for pi, order in enumerate(perms):
        runs, miss = run_stream_c1lite(featfn, bplus, order, by_uid, blocks_uids,
                                       actions, frozen_srv, perm_seed=SEED + pi * 1000)
        total_missing += miss
        for a in C1_ARMS:
            all_runs[a].append(runs[a])
    print(f"  完成 {len(perms)} 排列 [{time.time()-t0:.0f}s]；bplus 缺失/回退行数={total_missing}", flush=True)

    res = summarize(all_runs)
    res["config"] = dict(seed=SEED, delta_safe=DELTA_SAFE, k=K, retrieval_key=key_name,
                         bplus_missing_rows=total_missing, prereg="压缩计划 §1（C1-lite）",
                         note="探索性 pilot，非 confirmatory；frozen=P0 参照复用；B+=leaky 强上界（in-sample 全局拟合）；"
                         "random-memory=同门控同覆盖率打乱标签（leakage-immune）；seen=严格过去块；current L_test 仅事后计 regret")
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    table = render(res)
    (OUT / "table.md").write_text(table, "utf-8")
    print("\n" + table, flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
