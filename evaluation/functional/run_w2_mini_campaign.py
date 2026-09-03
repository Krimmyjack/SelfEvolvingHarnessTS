"""工作包 2：3-Domain Mini-Campaign（deepseek 副本，2026-08-06）。

按预注册文档（docs/WORK_PACK_2_3_MINI_CAMPAIGN_PREREGISTRATION.md，含 Amendment-001/002）：
- 3 Domain（nn5/gefcom/noaa）× A/B/C 三批（chronological，冻结评估纪律）
- 四 Arm：A3（Memory off）/ L+（只正）/ L±（全）/ MISMATCH（错配混淆控制）
- 预算：B∈{0,1,2}、每轮 2 候选 + 1 revision、2 次 Support eval（冻结）
- 指标：Adaptation AUC / first-positive probe / harm / abstention / coverage / citation
- 通过条件 §6 + 停止条件 §7 判定
- 可视化：matplotlib PNG（Arm AUC 曲线 + 汇总柱状图）→ artifacts/visualization/

用法：
  python evaluation/functional/run_w2_mini_campaign.py --dry-run      # mock proposer，零 LLM 调用
  python evaluation/functional/run_w2_mini_campaign.py                # 真实 LLM（需 OPENAI_API_KEY / AGICTO_API_KEY）
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
from experience_memory import (  # noqa: E402
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    SignedEpisodeRetriever,
    build_episode,
    load_episodes_from_v6_reports,
    workflow_signature_of,
)

REPORT_REL = Path("artifacts/functional/e2/w2_mini_campaign_report.json")
VIZ_DIR_REL = Path("artifacts/visualization")


def _report_path(provider: str, tag: str = "") -> Path:
    """报告路径带 provider 后缀（deepseek/gpt 结果不互相覆盖）；--tag 用于配对复验重复。"""
    suffix = f"_{tag}" if tag else ""
    return REPORT_REL.with_name(f"w2_mini_campaign_report_{provider}{suffix}.json")


def _viz_dir(provider: str) -> Path:
    return VIZ_DIR_REL / f"w2_{provider}"

# 3 Domain（正向迹象/负向迹象/冲突迹象；Dataset 选择由预注册 Amendment-001 确定的数据族）
DOMAIN_KEYS = ("nn5", "gefcom", "noaa")
DOMAIN_ROLE = {"nn5": "positive-signal", "gefcom": "negative-signal", "noaa": "conflict-signal"}

# 预算冻结：probe 预算、每轮候选数、revision 数、Support eval 次数
PROBE_BUDGET = 2
MAX_CANDIDATES = 2
MAX_REVISIONS = 1
MAX_SUPPORT_EVALS = 2


# ---------------------------------------------------------------------------
# mock proposer（dry-run：零 LLM 调用；检查对照包注入是否生效）
# ---------------------------------------------------------------------------

class _MockProposer:
    """dry-run proposer：pack 存在（Memory-on）时返回可编译候选，否则 ABSTAIN。

    这样 dry-run 真实演练"对照包注入 → Plan 改变 → support gain 产生 → AUC 计算"，
    而不仅是流程空转。候选用 v6 已验证的 period_median_complete（NN5 正向算子）。
    """

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.saw_contrast_pack = False
        self.saw_positive = False
        self.saw_negative = False
        self.saw_conflict = False
        self.calls = 0

    def __call__(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        self.calls += 1
        pack = payload.get("experience_contrast_pack")
        if isinstance(pack, Mapping):
            self.saw_contrast_pack = True
            self.saw_positive = pack.get("positive") is not None
            self.saw_negative = pack.get("negative") is not None
            self.saw_conflict = pack.get("conflict") is not None
        if self.saw_positive and self.arm in ("L+", "L±", "MISMATCH"):
            return {
                "decision": "PROPOSE",
                "steps": [
                    {
                        "op": "period_median_complete",
                        "params": {"cycles": 3, "min_donors": 2, "period": 7},
                        "bindings": {},
                    }
                ],
                "requested_observations": [],
                "fallback": "IDENTITY",
            }
        return {"decision": "ABSTAIN"}


# Provider 配置：agicto（默认）与 deepseek 均为 OpenAI 兼容接口
PROVIDERS = {
    "agicto": {
        "env_key": ("OPENAI_API_KEY", "AGICTO_API_KEY"),
        "base_url": "https://api.agicto.cn/v1",
        "default_model": "gpt-5.6-luna",
    },
    "deepseek": {
        "env_key": ("DEEPSEEK_API_KEY",),
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
    },
}


def _make_arm_proposer(arm: str, dry_run: bool, provider: str = "agicto", model: str | None = None) -> Any:
    if dry_run:
        return _MockProposer(arm)
    cfg = PROVIDERS[provider]
    api_key = next(
        (os.environ.get(k, "").strip() for k in cfg["env_key"] if os.environ.get(k, "").strip()),
        "",
    )
    if not api_key:
        raise SystemExit(f"{cfg['env_key']} required for live mode (provider={provider})")
    return v6.LiveJSONProposer(
        api_key=api_key,
        model=model or cfg["default_model"],
        base_url=cfg["base_url"],
    )


# ---------------------------------------------------------------------------
# Adaptation AUC（梯形法：按 probe 顺序的累积 support gain 曲线）
# ---------------------------------------------------------------------------

def adaptation_auc(gains: Sequence[float]) -> float:
    """按 probe 顺序累积 gain 的曲线下面积（probe 数归一化）。

    gains[i] = 第 i 次 probe 的 support gain（相对 incumbent）。
    AUC = Σ 累积收益 / 最大可能（全正且每次 +1 的累积面积），取值可负。
    """
    if not gains:
        return 0.0
    cum = 0.0
    area = 0.0
    for i, g in enumerate(gains):
        cum += float(g)
        area += cum
    # 归一化：最多 PROBE_BUDGET 次 probe，理想全正 +1 → 面积 = B(B+1)/2
    n = min(len(gains), PROBE_BUDGET)
    max_area = n * (n + 1) / 2.0
    return area / max_area if max_area else 0.0


def _support_gain_series(report: Mapping[str, object]) -> list[float]:
    """从 v6 generation report 的 proposals 提取按顺序的 support gains。

    真实字段名是 support_gain（兼容旧名 gain）。proposals 行结构：
    {"stage": ..., "candidate_id": ..., "support_response": {"accepted": bool,
     "support_gain": float, "per_view_gain": [...]}, ...}
    """
    gains: list[float] = []
    proposals = report.get("generation_proposals")
    if isinstance(proposals, Sequence):
        for row in proposals:
            if not isinstance(row, Mapping):
                continue
            sr = row.get("support_response")
            if isinstance(sr, Mapping):
                val = sr.get("support_gain")
                if not isinstance(val, (int, float)):
                    val = sr.get("gain")
                if isinstance(val, (int, float)):
                    gains.append(float(val))
    return gains


# ---------------------------------------------------------------------------
# Episode 构造（A 段 adapt 后写入）
# ---------------------------------------------------------------------------

def _episode_from_report(
    *,
    label: str,
    domain: str,
    report: Mapping[str, object],
) -> Any:
    """从 v6 generation report 构造 Episode——relation/status 由真实 response 自动决定。

    生命周期语义（用户裁决）：Support 正只能是 LOCAL_DRAFT；held-in delayed 正才是
    LOCAL_ACTIVE；Support 正而 delayed 负必须写 CONFLICT；B≤0 写 NEGATIVE；
    无有效 response 不写 Episode（不冒充负证据）。

    返回 None 表示该报告无有效 Action–Response，不应写入经验。
    """
    proposals = report.get("generation_proposals")
    steps: list[Mapping[str, object]] = []
    if isinstance(proposals, Sequence):
        accepted_row = next(
            (
                row
                for row in proposals
                if isinstance(row, Mapping)
                and isinstance(row.get("support_response"), Mapping)
                and row["support_response"].get("accepted") is True
            ),
            None,
        )
        if accepted_row is not None:
            compiled = accepted_row.get("compiled_program_steps")
            if isinstance(compiled, Sequence):
                steps = [s for s in compiled if isinstance(s, Mapping)]
    sig = workflow_signature_of(steps) if steps else "identity"
    gains = _support_gain_series(report)
    best_gain = max(gains) if gains else None
    accepted = bool(report.get("candidate_skill_draft"))
    selection = report.get("selection")
    delayed_gain = (
        float(selection["selection_gain"])
        if isinstance(selection, Mapping) and isinstance(selection.get("selection_gain"), (int, float))
        else None
    )

    # 无有效 response：无 proposals 或无可接受的候选 → 不写 Episode
    if not gains or best_gain is None or not accepted:
        return None

    # relation/status 自动决定
    if best_gain > 0 and delayed_gain is not None and delayed_gain > 0:
        relation, status = RELATION_POSITIVE, "LOCAL_ACTIVE"
    elif best_gain > 0 and delayed_gain is not None and delayed_gain <= 0:
        relation, status = "CONFLICT", "EPISODE_ONLY"  # Support 正、delayed 负 → 冲突，不获执行权
    elif best_gain > 0:
        relation, status = RELATION_POSITIVE, "LOCAL_DRAFT"  # delayed 未评估 → 仅草稿
    else:
        relation, status = "NEGATIVE", "EPISODE_ONLY"

    return build_episode(
        episode_id=f"{domain}_{label}",
        task_consumer_key="forecast|ridge_smase",
        domain_namespace=domain,
        context_summary={
            "cohort": {"series_count": 32, "evaluation_series_count": 8},
            "local_pattern": {"support_gain": best_gain},
            "program_geometry": {"scope": "training_rows"},
        },
        workflow_signature=sig,
        support_response={"gain": best_gain, "accepted": accepted},
        delayed_response={"evaluated": delayed_gain is not None, "gain": delayed_gain},
        relation=relation,
        evidence_level="DELAYED" if delayed_gain is not None else "SUPPORT",
        local_status=status,
        evidence_refs=[f"artifacts/functional/e2/{domain}_w2_report.json"],
    )


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _render_visualizations(
    results: Mapping[str, Any],
    viz_dir: Path,
) -> list[str]:
    """每 Domain 的 Arm AUC 曲线对比 + 汇总柱状图（matplotlib PNG）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # 可视化失败不阻塞报告
        return [f"visualization skipped: {exc}"]

    viz_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    arms = ("A3", "L+", "L±", "MISMATCH")

    # 1) 每 Domain：Arm × probe 顺序的累积 gain 曲线
    for domain, dres in results.get("domains", {}).items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in arms:
            series = dres.get("arms", {}).get(arm, {}).get("gain_series", [])
            cum = []
            acc = 0.0
            for g in series:
                acc += float(g)
                cum.append(acc)
            xs = list(range(1, len(cum) + 1))
            ax.plot(xs, cum, marker="o", label=f"{arm} (AUC={adaptation_auc(series):.3f})")
        ax.set_title(f"Domain {domain} [{DOMAIN_ROLE.get(domain, '')}]: cumulative support gain by probe")
        ax.set_xlabel("probe #")
        ax.set_ylabel("cumulative support gain")
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend()
        fig.tight_layout()
        out = viz_dir / f"w2_domain_{domain}_auc.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        written.append(str(out.relative_to(PROJECT_ROOT)))

    # 2) 汇总：每 Domain 的 Arm Adaptation AUC 柱状图
    fig, ax = plt.subplots(figsize=(8, 4.5))
    domain_list = list(results.get("domains", {}).keys())
    width = 0.2
    for i, arm in enumerate(arms):
        aucs = [
            results["domains"].get(d, {}).get("arms", {}).get(arm, {}).get("auc", 0.0)
            for d in domain_list
        ]
        xs = [j + i * width for j in range(len(domain_list))]
        ax.bar(xs, aucs, width=width, label=arm)
    ax.set_xticks([j + 1.5 * width for j in range(len(domain_list))])
    ax.set_xticklabels(domain_list)
    ax.set_ylabel("Adaptation AUC")
    ax.set_title("Mini-Campaign: Adaptation AUC by Arm and Domain")
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    fig.tight_layout()
    out = viz_dir / "w2_summary_auc_by_arm.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    written.append(str(out.relative_to(PROJECT_ROOT)))
    return written


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Work Package 2: 3-Domain Mini-Campaign")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="mock proposer, zero LLM calls")
    parser.add_argument("--domains", nargs="*", default=list(DOMAIN_KEYS), choices=list(DOMAIN_KEYS))
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="agicto",
                        help="LLM provider (agicto | deepseek)")
    parser.add_argument("--model", default=None, help="override provider default model")
    parser.add_argument("--tag", default="", help="report suffix for repeated runs (paired recheck)")
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("OPENAI_API_KEY", "")  # 防止 v6 内部 env 误判

    domains: dict[str, Any] = {}
    global_checks: dict[str, bool] = {}
    arm_auc: dict[str, list[float]] = {arm: [] for arm in ("A3", "L+", "L±", "MISMATCH")}
    injection_evidence: dict[str, Any] = {}

    # 评审修复：加载 v6 已暴露的 negative/conflict episodes（跨域对照经验，工作包 1 已构造）
    prior_episodes = load_episodes_from_v6_reports(root / "artifacts/functional/e2")
    prior_by_domain = {ep.episode_id: ep for ep in prior_episodes}
    print(f"== prior episodes loaded: {[ep.episode_id for ep in prior_episodes]}")

    for domain in args.domains:
        print(f"\n=== Domain {domain} [{DOMAIN_ROLE.get(domain)}] ===")
        dres: dict[str, Any] = {"role": DOMAIN_ROLE.get(domain), "arms": {}}

        # ---- A 段：adapt（原 origins，generation → Episode 写入）----
        a_proposer = _make_arm_proposer("A", args.dry_run, provider=args.provider, model=args.model)
        a_report = v6.run(
            root, initial_proposer=a_proposer, revision_proposer=a_proposer,
            dataset_key=domain, write_report=False,
        )
        ep_positive = _episode_from_report(label="adapt_A", domain=domain, report=a_report)
        # 检索池 = 同域 A 段经验（relation/status 由真实 response 决定）+ 已暴露的负/冲突经验
        pool = ([ep_positive] if ep_positive is not None else []) + list(prior_episodes)
        if ep_positive is None:
            print(f"  A: no valid Action-Response (gains={_support_gain_series(a_report)}, "
                  f"accepted={bool(a_report.get('candidate_skill_draft'))}) -> no episode written")
        else:
            print(f"  A: episode written ({ep_positive.episode_id}, sig={ep_positive.workflow_signature}, "
                  f"relation={ep_positive.relation}, status={ep_positive.local_status}, "
                  f"support gains {_support_gain_series(a_report)})")

        # ---- B 段：冻结评估（chronological：origins 后移一个 period；四 Arm 无 slow_path）----
        retriever = SignedEpisodeRetriever(pool, task_consumer_key="forecast|ridge_smase")
        query_context = dict(ep_positive.context_summary) if ep_positive is not None else {
            "cohort": {"series_count": 32, "evaluation_series_count": 8},
            "local_pattern": {"support_gain": 0.0},
            "program_geometry": {"scope": "training_rows"},
        }
        pack = retriever.retrieve(query_context, domain)

        # B 段 support_origin 后移（A→B 不同切片，chronological）；
        # selection_origin 保持不变（_fixed_roster 的 required_length 依赖它，
        # 后移会让序列长度不足）
        orig_cfg = dict(v6.DATASET_CONFIGS[domain])
        shift = int(orig_cfg.get("period", 1))
        v6.DATASET_CONFIGS[domain] = dict(orig_cfg)
        v6.DATASET_CONFIGS[domain]["support_origin"] = int(orig_cfg["support_origin"]) + shift

        try:
            for arm in ("A3", "L+", "L±", "MISMATCH"):
                proposer = _make_arm_proposer(arm, args.dry_run, provider=args.provider, model=args.model)

                def _proposer_with_pack(payload: Mapping[str, object], *, arm=arm, proposer=proposer, pack=pack) -> Mapping[str, object]:
                    pl = copy.deepcopy(dict(payload))
                    if arm == "A3":
                        pass  # 无 Memory
                    elif arm == "L+":
                        # 评审修复：L+ 只注入 positive（不注入负/冲突）
                        pl["experience_contrast_pack"] = (
                            {
                                "positive": pack.positive.to_dict() if pack.positive else None,
                                "negative": None,
                                "conflict": None,
                                "evidence_sufficient": pack.evidence_sufficient,
                                "retrieval_note": "positive-only",
                            }
                        )
                    elif arm == "L±":
                        # 评审修复：L± 注入完整 Signed Pack（正+负+冲突）
                        pl["experience_contrast_pack"] = pack.to_dict()
                    elif arm == "MISMATCH":
                        # 错配对照：同数量 episode，但 context 全错配
                        import experience_memory as em
                        mismatched = em.ContrastPack(
                            positive=em.build_episode(
                                episode_id="mismatch_ctx", task_consumer_key="forecast|ridge_smase",
                                domain_namespace="other_domain",
                                context_summary={"cohort": {"series_count": 999}, "local_pattern": {"support_gain": -9.0}, "program_geometry": {"scope": "wrong"}},
                                workflow_signature="period_median_complete", support_response={}, delayed_response={},
                                relation=RELATION_POSITIVE, evidence_level="SUPPORT", local_status="EPISODE_ONLY",
                            ) if pack.positive else None,
                            negative=None, conflict=None, evidence_sufficient=False, retrieval_note="mismatched context control",
                        )
                        pl["experience_contrast_pack"] = mismatched.to_dict()
                    return proposer(pl)

                arm_report = v6.run(
                    root, initial_proposer=_proposer_with_pack, revision_proposer=_proposer_with_pack,
                    dataset_key=domain, write_report=False,
                )
                gains = _support_gain_series(arm_report)
                auc = adaptation_auc(gains)
                arm_auc[arm].append(auc)
                # 评审修复：候选签名 + retrieved episode IDs + delayed（selection）
                proposals = arm_report.get("generation_proposals")
                cand_sigs = []
                if isinstance(proposals, Sequence):
                    for row in proposals:
                        if isinstance(row, Mapping):
                            steps = row.get("compiled_program_steps") or row.get("workflow_steps")
                            if isinstance(steps, Sequence):
                                cand_sigs.append(workflow_signature_of([s for s in steps if isinstance(s, Mapping)]))
                selection = arm_report.get("selection")
                delayed_gain = None
                if isinstance(selection, Mapping) and isinstance(selection.get("selection_gain"), (int, float)):
                    delayed_gain = float(selection["selection_gain"])
                # 实际注入的 Episode IDs（评审修复：记录实际注入，非检索器完整 pack）
                if arm == "L+":
                    injected_ids = [pack.positive.episode_id] if pack.positive else []
                elif arm == "L±":
                    injected_ids = [
                        ep.episode_id for ep in (pack.positive, pack.negative, pack.conflict) if ep is not None
                    ]
                else:
                    injected_ids = []
                dres["arms"][arm] = {
                    "auc": auc,
                    "gain_series": gains,
                    "first_positive_probe": next((i + 1 for i, g in enumerate(gains) if g > 0), None),
                    "harm_probe_count": sum(1 for g in gains if g < 0),
                    "abstained": (len(gains) == 0),
                    "candidate_signatures": cand_sigs,
                    "retrieved_episode_ids": injected_ids,
                    "delayed_gain": delayed_gain,
                    "delayed_harm": bool(delayed_gain is not None and delayed_gain < 0),
                    "support_origin": v6.DATASET_CONFIGS[domain]["support_origin"],
                }
                if isinstance(proposer, _MockProposer):
                    injection_evidence[f"{domain}/{arm}"] = {
                        "saw_contrast_pack": proposer.saw_contrast_pack,
                        "saw_positive": proposer.saw_positive,
                        "saw_negative": proposer.saw_negative,
                        "saw_conflict": proposer.saw_conflict,
                        "proposer_calls": proposer.calls,
                    }
                print(f"  B/{arm}: AUC={auc:.4f} gains={[round(g, 4) for g in gains]} "
                      f"delayed={delayed_gain if delayed_gain is not None else 'n/a'}")
        finally:
            v6.DATASET_CONFIGS[domain] = orig_cfg  # 恢复原 origins

        # ---- C 段：delayed 报告（B 段每 Arm 的 selection_origin 评估已在上方记录）----
        best_arm = max(dres["arms"], key=lambda a: dres["arms"][a]["auc"])
        dres["best_arm_by_auc"] = best_arm
        print(f"  C: delayed gains recorded per arm; best arm by AUC = {best_arm}")
        domains[domain] = dres

    # ---- 通过条件判定（预注册 §6；评审修复：六门 PASS/FAIL/NOT_EVALUATED）----
    def _macro(arm: str, key: str = "auc") -> float:
        vals = [domains[d]["arms"][arm][key] for d in args.domains]
        return sum(vals) / len(vals) if vals else 0.0

    lpm_auc = [domains[d]["arms"]["L±"]["auc"] for d in args.domains]
    a3_auc = [domains[d]["arms"]["A3"]["auc"] for d in args.domains]
    mismatch_auc = [domains[d]["arms"]["MISMATCH"]["auc"] for d in args.domains]
    global_checks["gate1_Lpm_macro_gt_A3"] = _macro("L±") > _macro("A3")
    global_checks["gate2_2of3_Lpm_not_worse"] = sum(1 for a, b in zip(lpm_auc, a3_auc) if a >= b) >= 2
    global_checks["gate2_1_domain_clearly_better"] = any(a - b > 0.05 for a, b in zip(lpm_auc, a3_auc))
    # gate3：L± 的 delayed harm 不高于 A3（评审补充；无 delayed 数据时 NOT_EVALUATED）
    lpm_dharm = sum(1 for d in args.domains if domains[d]["arms"]["L±"].get("delayed_harm"))
    a3_dharm = sum(1 for d in args.domains if domains[d]["arms"]["A3"].get("delayed_harm"))
    if all(domains[d]["arms"]["A3"].get("delayed_gain") is not None for d in args.domains):
        global_checks["gate3_Lpm_delayed_harm_not_gt_A3"] = lpm_dharm <= a3_dharm
    else:
        global_checks["gate3_Lpm_delayed_harm_not_gt_A3"] = "NOT_EVALUATED"
    # gate4：L± 非全 abstain（评审补充）
    global_checks["gate4_Lpm_not_all_abstain"] = any(
        not domains[d]["arms"]["L±"]["abstained"] for d in args.domains
    )
    # gate5：MISMATCH 不能取得相同效果（评审：提示性证据，不等 token 限制在报告标注）
    global_checks["gate5_mismatch_not_same"] = sum(mismatch_auc) / len(mismatch_auc) < sum(lpm_auc) / len(lpm_auc)
    # gate6：Episode 实际引用（评审：真实模式无法机械确认 LLM 引用；记录 retrieved IDs 供人工核）
    global_checks["gate6_episode_citation"] = "NOT_EVALUATED"

    # ---- 报告 + 可视化 ----
    report = {
        "experiment_id": "w2-mini-campaign",
        "tag": args.tag,
        "mode": "dry-run" if args.dry_run else "live",
        "scientific_role": "preregistered_mini_campaign",
        "claim_limit": (
            "Dry-run: process+metrics+visualization verification only, no natural evidence. "
            "Live mode claims only per preregistered §6 gates; no fresh Transfer claim."
        ),
        "domains": domains,
        "global_checks": global_checks,
        "injection_evidence": injection_evidence,
        "budget": {"probe_budget": PROBE_BUDGET, "max_candidates": MAX_CANDIDATES,
                   "max_revisions": MAX_REVISIONS, "max_support_evals": MAX_SUPPORT_EVALS},
        "viz": _render_visualizations({"domains": domains}, root / _viz_dir(args.provider + (f"_{args.tag}" if args.tag else ""))),
    }
    out = root / _report_path(args.provider, args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n== report -> {_report_path(args.provider, args.tag)}")
    print(f"== visualizations -> {report['viz']}")
    print("== global gates:", {k: v for k, v in global_checks.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
