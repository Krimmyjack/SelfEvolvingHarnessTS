"""run_proposer.py — Track B1 proposer 竞技场 **scaffold**（prereg_proposer.md）。

**这不是正式跑**——本模块建成后停在单 batch dry-run，正式外呼待用户审 prereg 后启动
（用户第三十七轮）。核心命题（§13.2）：LLM proposer 读 **per-family 分解** mining 报告，能否
在**同一可执行编辑空间、同候选预算**下，提出枚举/搜索找不到的、可迁移的 harness 结构编辑。

Readiness bar（用户第三十七轮硬要求）：四臂在**同一编辑空间**（下方 SPACE=scoped risk 规则的
有限网格）、**同 budget**（每臂恰 N 个候选）、**同评估**（gym 统一 select-on-selection →
eval-on-heldout）竞争。否则只回答"谁更会写 scoped 规则"，答不了项目身份。

四臂：
  random        空间内均匀采样（**负对照**，防 validator 当选择器的 winner's curse）
  det_search    读 mining 报告的确定性搜索（report_estimate 排序取 top-N）
  llm_nomem     LLM 读 per-family 报告
  llm_mem       LLM 读报告 + 检索到的过去成功编辑

模型：便宜=gpt-5.4-mini(agicto)；天花板=Claude Opus 4.8（正式跑用 subagent-cache，见 prereg §模型）。
评估=RiskAwareRouterPolicy replay（overlay 决策口径，查 L_test，无 torch）。

运行：
  --smoke        gpt-5.4-mini API 连通烟测（一次调用）
  --dry-run      单 held-out 族 + 小 budget + 四臂，产 scored 表（停，不进正式 LOFO）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm.client import LLMClient, extract_json
from .policy import FrozenArmRouterPolicy
from .policy.action_spec import action_menu_v1
from .policy.edits import AddRiskRule, bundle_v0, compile_bundle
from .policy.risk_policy import RiskPolicy, RiskRule, POOL_ACTIONS
from .run_harness_slice import _struct_of
from .run_updater2 import REC_PATH
from .s2_corpus import S2_FAMILIES

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "Proposer"
AGICTO_URL = "https://api.agicto.cn/v1/chat/completions"

# ════════════════════════════ 共享编辑空间（有限网格）════════════════════════════
FEATURES = ["seasonal_strength", "acf1", "trend_strength", "outlier_density", "lumpiness"]
OPS = [">=", "<="]
THRESH = [0.2, 0.4, 0.6]
GROUPS = {"heavy_median": ["f0_median_w9", "f0_median_w15", "f0_median_w25"],
          "stl": ["v_stl"],
          "smoothers": ["v_median", "v_savgol", "v_wavelet"]}
REPLACEMENTS = ["v_none", "v_median", "f0_median_w25"]
MAX_RULES = 3                                   # 候选编辑集规则数上限


def enumerate_space() -> List[RiskRule]:
    rules = []
    for feat in FEATURES:
        for op in OPS:
            for thr in THRESH:
                for gname, gacts in GROUPS.items():
                    for repl in REPLACEMENTS:
                        if repl in gacts:
                            continue
                        rules.append(RiskRule(
                            rule_id=f"{feat}|{op}|{thr}|{gname}|{repl}",
                            when={"feats": [{"name": feat, "op": op, "value": thr}],
                                  "base_action_in": list(gacts)},
                            then={"op": "ban", "action": repl},
                            scope=f"region:{feat}{op}{thr}",
                            provenance={"source": "space", "group": gname}))
    return rules


SPACE = enumerate_space()
_RULE_BY_ID = {r.rule_id: r for r in SPACE}


def _rule_from_fields(feat, op, value, group, replacement) -> Optional[RiskRule]:
    """把 (feat,op,value,group,repl) snap 进空间；越界 → None。det_search/LLM 共用（保证同空间）。"""
    if feat not in FEATURES or op not in OPS or group not in GROUPS or replacement not in REPLACEMENTS:
        return None
    if replacement in GROUPS[group]:
        return None
    thr = min(THRESH, key=lambda t: abs(t - float(value)))     # snap 到网格
    return _RULE_BY_ID.get(f"{feat}|{op}|{thr}|{group}|{replacement}")


# ════════════════════════════ mining 报告（per-family，禁聚合）════════════════════════════
def build_mining_report(rows: List[dict], actions: List[str]) -> dict:
    """per-family 分解：每族 × 每 region 谓词 × 每 group，frozen 若在该 region 选该 group 的
    平均 oracle-gap（>0=该 group 在该 region 被过度选用/有害的信号）。**不跨族聚合**（§13.2 纪律）。"""
    by_fam: Dict[str, List[dict]] = {}
    for r in rows:
        by_fam.setdefault(r["origin"], []).append(r)
    report = {"families": {}}
    for fam, frows in by_fam.items():
        sig = []
        for feat in FEATURES:
            for op in OPS:
                for thr in THRESH:
                    for gname, gacts in GROUPS.items():
                        sel = [r for r in frows
                               if _pred(_struct_of(r).get(feat), op, thr)]
                        if len(sel) < 3:
                            continue
                        gaps = [min(r["L_test"].values())  # oracle
                                for r in sel]
                        # frozen 在该 region 内选 group 动作的平均超额损失
                        harm = []
                        for r in sel:
                            best_g = min((r["L_test"][a] for a in gacts if a in r["L_test"]),
                                         default=None)
                            if best_g is None:
                                continue
                            harm.append(best_g - min(r["L_test"].values()))
                        if harm:
                            sig.append(dict(feat=feat, op=op, value=thr, group=gname,
                                            n=len(sel), mean_group_gap=float(np.mean(harm))))
        sig.sort(key=lambda s: -s["mean_group_gap"])
        report["families"][fam] = sig[:8]              # 每族 top-8 信号
    return report


def _pred(v, op, thr) -> bool:
    if v is None:
        return False
    return (v >= thr) if op == ">=" else (v <= thr)


def report_text(report: dict) -> str:
    L = ["Per-family mining signals (frozen's mean oracle-gap when picking an action-group "
         "in a feature-region; higher = that group is a worse choice there):"]
    for fam, sig in report["families"].items():
        L.append(f"\n[family {fam}]")
        for s in sig:
            L.append(f"  region {s['feat']}{s['op']}{s['value']} × group {s['group']} "
                     f"(n={s['n']}): group_gap={s['mean_group_gap']:.3f}")
    return "\n".join(L)


def report_estimate(report: dict, rule: RiskRule) -> float:
    """det_search 用：报告对一条 ban 规则收益的估计 = 被禁 group 在该 region 的平均 gap（越大越该禁）。"""
    w = rule.when["feats"][0]
    gname = rule.provenance.get("group")
    tot, n = 0.0, 0
    for sig in report["families"].values():
        for s in sig:
            if s["feat"] == w["name"] and s["op"] == w["op"] and s["value"] == w["value"] \
                    and s["group"] == gname:
                tot += s["mean_group_gap"]
                n += 1
    return tot / n if n else 0.0


# ════════════════════════════ 四臂 proposer（同 budget，同空间）════════════════════════════
class CandidateProposer:
    name = "base"

    def propose(self, report: dict, budget: int, memory: Optional[List] = None
                ) -> List[List[AddRiskRule]]:
        raise NotImplementedError


class RandomProposer(CandidateProposer):
    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def propose(self, report, budget, memory=None):
        cands = []
        for _ in range(budget):
            k = int(self.rng.integers(1, MAX_RULES + 1))
            idx = self.rng.choice(len(SPACE), size=k, replace=False)
            cands.append([AddRiskRule(SPACE[int(i)]) for i in idx])
        return cands


class DetSearchProposer(CandidateProposer):
    name = "det_search"

    def propose(self, report, budget, memory=None):
        ranked = sorted(SPACE, key=lambda r: -report_estimate(report, r))
        cands = [[AddRiskRule(r)] for r in ranked[:budget]]         # top-N 单规则候选
        # 再加一个 greedy 多规则候选（top-MAX_RULES 合并），替掉最弱的单规则候选
        if len(ranked) >= MAX_RULES and budget >= 1:
            cands[-1] = [AddRiskRule(r) for r in ranked[:MAX_RULES]]
        return cands


_LLM_INSTR = (
    "You edit a frozen time-series preprocessing router by proposing SCOPED OVERRIDE RULES.\n"
    "Each rule bans an action-GROUP in an observable feature-REGION and replaces it.\n"
    "VOCABULARY (you MUST stay inside it):\n"
    f"  feat ∈ {FEATURES}\n  op ∈ {OPS}\n  value ∈ {THRESH}\n"
    f"  group ∈ {list(GROUPS)}\n  replacement ∈ {REPLACEMENTS} (not in the banned group)\n"
    "Use the per-family signals: ban a group in a region where its group_gap is high.\n"
    "Keep rules SCOPED per-family/worst-group (do NOT over-generalize a family-specific harm).\n"
    'Return JSON ONLY: {"candidates": [ {"rules": [ '
    '{"feat":..,"op":..,"value":..,"group":..,"replacement":..} ]} ]}\n'
    "Give EXACTLY N distinct candidates; each candidate has 1..3 rules.")


class LLMProposer(CandidateProposer):
    def __init__(self, llm: LLMClient, with_memory: bool):
        self.llm = llm
        self.with_memory = with_memory
        self.name = "llm_mem" if with_memory else "llm_nomem"

    def propose(self, report, budget, memory=None):
        mem_txt = ""
        if self.with_memory and memory:
            mem_txt = "\n\nPast successful edits (retrieved experience):\n" + "\n".join(
                f"  {m}" for m in memory[:6])
        user = (f"{report_text(report)}{mem_txt}\n\nN = {budget}.\n{_LLM_INSTR}")
        raw = self.llm("You are a careful data-readiness policy editor. Output JSON only.",
                       user, nonce=0)
        spec = extract_json(raw)
        cands: List[List[AddRiskRule]] = []
        for c in (spec or {}).get("candidates", [])[:budget]:
            ops = []
            for rl in (c.get("rules") or [])[:MAX_RULES]:
                rule = _rule_from_fields(rl.get("feat"), rl.get("op"), rl.get("value"),
                                         rl.get("group"), rl.get("replacement"))
                if rule is not None:
                    ops.append(AddRiskRule(rule))
            if ops:
                cands.append(ops)
        return cands, raw            # raw 供审计


# ════════════════════════════ 评估（RiskAwareRouterPolicy replay）════════════════════════════
def _key(r: dict) -> dict:
    return {"pattern": {"struct_feats": _struct_of(r)}, "task": {"type": "forecast"},
            "cell_id": r["cell"]}


def precompute_frozen(rows: List[dict], frozen, menu) -> Dict[str, dict]:
    """冻结 pick 与 edit 无关 → 一次算好缓存（避免每候选每行重算 support-check）。"""
    fc: Dict[str, dict] = {}
    for r in rows:
        fc[r["uid"]] = dict(pick=frozen.predict(_key(r), menu).action_id,
                            struct=_struct_of(r), cell=r["cell"], origin=r["origin"], L=r["L_test"])
    return fc


def utility(edit_ops: List[AddRiskRule], uids: List[str], fc: Dict[str, dict]) -> dict:
    """→ {util, worst_family, n_fire}。util = mean(frozen_regret − edited_regret)（>0=改善）。
    用冻结 pick 缓存 + RiskPolicy.apply 直算——**等价 RiskAwareRouterPolicy**（weld 测试+烟测已证：
    覆盖 = base pick + risk.apply）；部署仍走 RiskAwareRouterPolicy（§13.4 焊接），此处只是同结果快算。"""
    from .policy.edits import apply_edits
    bundle, _ = apply_edits(bundle_v0(), edit_ops)
    risk = bundle.risk
    per_fam: Dict[str, List[float]] = {}
    n_fire = 0
    for u in uids:
        c = fc[u]
        fa = c["pick"]
        ea, _, _, _ = risk.apply(fa, False, c["struct"], c["cell"])
        if ea not in c["L"]:                          # 覆盖目标须可评估（∈池⊆L_test）；否则回退
            ea = fa
        if ea != fa:
            n_fire += 1
        per_fam.setdefault(c["origin"], []).append(c["L"][fa] - c["L"][ea])   # oracle 抵消
    util = float(np.mean([a for v in per_fam.values() for a in v]))
    worst = min(float(np.mean(v)) for v in per_fam.values())
    return dict(util=util, worst_family=worst, n_fire=n_fire)


def run_arm(proposer: CandidateProposer, report, budget, sel_uids, ho_uids, fc, memory=None) -> dict:
    t0 = time.time()
    out = proposer.propose(report, budget, memory)
    cands = out[0] if isinstance(out, tuple) else out
    raw = out[1] if isinstance(out, tuple) else None
    if not cands:
        return dict(arm=proposer.name, n_candidates=0, selected=None, heldout_util=0.0,
                    heldout_worst=0.0, note="无有效候选（LLM 解析空/越界）", seconds=round(time.time()-t0, 1))
    scored = [(utility(c, sel_uids, fc)["util"], i) for i, c in enumerate(cands)]
    best_i = max(scored, key=lambda t: t[0])[1]                # select-on-selection
    best = cands[best_i]
    ho = utility(best, ho_uids, fc)                            # eval-on-heldout
    n_calls = getattr(proposer, "llm", None).n_api if hasattr(proposer, "llm") else 0
    return dict(arm=proposer.name, n_candidates=len(cands),
                selected=[a.rule.rule_id for a in best],
                selection_util=round(scored[best_i][0], 4),
                heldout_util=round(ho["util"], 4), heldout_worst=round(ho["worst_family"], 4),
                heldout_n_fire=ho["n_fire"], llm_calls=n_calls,
                raw_len=(len(raw) if raw else 0), seconds=round(time.time() - t0, 1))


# ════════════════════════════ main ════════════════════════════
def _agicto_key() -> str:
    k = os.environ.get("AGICTO_API_KEY")
    if k:
        return k
    txt = (Path(__file__).resolve().parent.parent / "previous" / "check10_haiku_react.py").read_text("utf-8")
    m = re.search(r'API_KEY\s*=\s*"([^"]+)"', txt)
    if not m:
        raise RuntimeError("找不到 agicto key（设 AGICTO_API_KEY 或查 previous/check10）")
    return m.group(1)


def make_mini_client() -> LLMClient:
    return LLMClient(model="gpt-5.4-mini", temperature=0.0, cache_name="proposer_mini",
                     url=AGICTO_URL, key=_agicto_key())


def smoke():
    llm = make_mini_client()
    t0 = time.time()
    reply = llm("You are a terse assistant.", "Reply with exactly: PROPOSER_SMOKE_OK", nonce=0)
    print(f"gpt-5.4-mini reply: {reply!r}  [{time.time()-t0:.1f}s]  stats={llm.stats()}")


def dry_run(held_out: str = "S_both", budget: int = 8):
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    actions = list(recs[0]["L_test"].keys())
    heldout_rows = [r for r in recs if r["origin"] == held_out]
    selection_rows = [r for r in recs if r["origin"] != held_out]
    report = build_mining_report(selection_rows, actions)      # 报告不含 held-out 族
    frozen = FrozenArmRouterPolicy.load_frozen("dp_abstain")
    menu = action_menu_v1()
    mini = make_mini_client()
    fc = precompute_frozen(selection_rows + heldout_rows, frozen, menu)      # 冻结 pick 一次算好
    sel_uids = [r["uid"] for r in selection_rows]
    ho_uids = [r["uid"] for r in heldout_rows]
    arms: List[CandidateProposer] = [
        RandomProposer(seed=20260707), DetSearchProposer(),
        LLMProposer(mini, with_memory=False), LLMProposer(mini, with_memory=True)]
    print(f"dry-run: held_out={held_out} | space={len(SPACE)} rules | budget={budget} | "
          f"selection n={len(selection_rows)} heldout n={len(heldout_rows)}", flush=True)
    results = []
    for a in arms:
        r = run_arm(a, report, budget, sel_uids, ho_uids, fc,
                    memory=["ban stl in low-seasonal regions (S0.7)"] if a.name == "llm_mem" else None)
        results.append(r)
        print(f"  [{r['arm']:10s}] cands={r['n_candidates']} sel_util={r.get('selection_util')} "
              f"heldout_util={r['heldout_util']} worst={r.get('heldout_worst')} "
              f"llm_calls={r.get('llm_calls',0)} [{r['seconds']}s]", flush=True)
    payload = dict(mode="dry_run", held_out=held_out, budget=budget, space_size=len(SPACE),
                   results=results, families=list(S2_FAMILIES),
                   note="scaffold dry-run；正式 LOFO×full-budget×天花板模型待 prereg 审后开",
                   prereg="results/Stage2/prereg_proposer.md（待建/待锁）")
    (OUT / "dry_run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), "utf-8")
    print(f"产物：{OUT / 'dry_run.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--held-out", default="S_both")
    ap.add_argument("--budget", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        smoke()
    if args.dry_run:
        dry_run(args.held_out, args.budget)
    if not (args.smoke or args.dry_run):
        print("用 --smoke 或 --dry-run")


if __name__ == "__main__":
    main()
