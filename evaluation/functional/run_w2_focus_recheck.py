"""工作包 2 聚焦复验：NN5 新 series cohort（deepseek 副本，2026-08-06）。

用户裁决（方案 A）：冻结一组未使用的 20 条 NN5 序列作为新 cohort，保持其余协议不变。

设计：
- 经验源 = NN5 扫描结果自动写成普通经验（不手选赢家）：
  B+C+ → POSITIVE/LOCAL_ACTIVE；B+C- → CONFLICT/EPISODE_ONLY；
  B≤0（有负 gain）→ NEGATIVE/EPISODE_ONLY；无有效响应/SKIP → 不写。
- 新 cohort：排除已用 20 条序列后按同一规则选 20 条（monkey-patch v6._fixed_roster）。
- 四 Arm（A3/L+/L±/MISMATCH）× generation（LLM）→ Support 评估 → sealed C（selection）delayed。
- sealed C 结果不写入 Memory（C outcome 只更新未来 Memory，不追溯修改本次计划）。
- 判定（用户裁决）：L± 或 L+ 在 sealed C 上 delayed 正或安全 abstain，且优于 A3/MISMATCH → PROCEED_TO_THREE_DOMAIN；否则 STOP。

用法：
  python evaluation/functional/run_w2_focus_recheck.py [--provider agicto] [--dry-run]
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
    ContrastPack,
    SignedEpisodeRetriever,
    build_episode,
    workflow_signature_of,
)

HORIZON = 48
TRAIN_SERIES_COUNT = 12
EVAL_SERIES_COUNT = 8
SCAN_REPORT_REL = Path("artifacts/functional/e2/w2_operator_scan_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w2_focus_recheck_report.json")


# ---------------------------------------------------------------------------
# 1. 扫描结果 → 普通经验（自动，不手选赢家）
# ---------------------------------------------------------------------------

def episodes_from_scan(scan: Mapping[str, object], domain: str) -> list[Any]:
    """NN5 扫描结果自动写成普通经验。relation/status 由 B/C 响应决定。"""
    episodes: list[Any] = []
    results = scan.get("operator_results")
    if not isinstance(results, Mapping):
        raise ValueError("scan report lacks operator_results")
    for op, r in results.items():
        if not isinstance(r, Mapping):
            continue
        status = r.get("status")
        if status in ("SKIP", "NOT_EXECUTED"):
            continue  # 无有效响应不写（不冒充负证据）
        gains = [float(g) for g in (r.get("support_gains") or []) if isinstance(g, (int, float))]
        if not gains:
            continue
        best_gain = max(gains)
        delayed = r.get("delayed_gain")
        delayed_gain = float(delayed) if isinstance(delayed, (int, float)) else None
        if best_gain > 0 and delayed_gain is not None and delayed_gain > 0:
            relation, status_l = RELATION_POSITIVE, "LOCAL_ACTIVE"
        elif best_gain > 0 and delayed_gain is not None and delayed_gain <= 0:
            relation, status_l = "CONFLICT", "EPISODE_ONLY"
        elif best_gain > 0:
            relation, status_l = RELATION_POSITIVE, "LOCAL_DRAFT"
        else:
            relation, status_l = RELATION_NEGATIVE, "EPISODE_ONLY"
        episodes.append(
            build_episode(
                episode_id=f"{domain}_scan_{op}",
                task_consumer_key="forecast|ridge_smase",
                domain_namespace=domain,
                context_summary={
                    "cohort": {"series_count": 32, "evaluation_series_count": 8},
                    "local_pattern": {"support_gain": best_gain},
                    "program_geometry": {"scope": "training_rows"},
                },
                workflow_signature=op,
                support_response={"gain": best_gain, "accepted": best_gain > 0},
                delayed_response={"evaluated": delayed_gain is not None, "gain": delayed_gain},
                relation=relation,
                evidence_level="DELAYED" if delayed_gain is not None else "SUPPORT",
                local_status=status_l,
                evidence_refs=[str(SCAN_REPORT_REL)],
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# 2. 新 cohort roster（排除已用 20 条；monkey-patch v6._fixed_roster）
# ---------------------------------------------------------------------------

_ORIG_FIXED_ROSTER = v6._fixed_roster
_USED_UIDS: set[str] | None = None


def _roster_new_cohort(root: Path, config: Mapping[str, object]):
    """与 _fixed_roster 相同选择规则，但排除已用序列（新 cohort）。"""
    global _USED_UIDS
    if _USED_UIDS is None:
        used_roster, _ = _ORIG_FIXED_ROSTER(root, config)
        _USED_UIDS = {str(row["series_uid"]) for row in used_roster}
    import numpy as np

    registry_path = root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_length = int(config["selection_origin"]) + HORIZON
    eligible = sorted(
        (
            row
            for row in rows
            if row.get("dataset_id") == config["dataset_id"]
            and int(row.get("length", 0)) >= required_length
            and str(row["series_uid"]) not in _USED_UIDS
        ),
        key=lambda row: str(row["series_uid"]),
    )
    selected = eligible[: TRAIN_SERIES_COUNT + EVAL_SERIES_COUNT]
    if len(selected) != TRAIN_SERIES_COUNT + EVAL_SERIES_COUNT:
        raise ValueError(f"new cohort lacks {TRAIN_SERIES_COUNT + EVAL_SERIES_COUNT} unused series "
                         f"(eligible={len(eligible)})")
    record_dirs: dict[str, Path] = {}
    wanted = {str(row["series_uid"]) for row in selected}
    for record_path in (root / "data/benchmark_v0_2/clean_base").glob("*/record.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        uid = str(record.get("series_uid", ""))
        if uid in wanted:
            record_dirs[uid] = record_path.parent
    if set(record_dirs) != wanted:
        raise ValueError("new cohort missing clean_base records")
    values = {
        uid: np.asarray(np.load(directory / "values.npy", allow_pickle=False), dtype=np.float64)
        for uid, directory in record_dirs.items()
    }
    roster = [
        {"series_uid": str(row["series_uid"]),
         "role": "train" if index < TRAIN_SERIES_COUNT else "eval"}
        for index, row in enumerate(selected)
    ]
    return roster, values


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="W2 focus recheck on unused NN5 cohort")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="nn5", choices=("nn5", "gefcom", "noaa"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--provider", choices=("agicto", "deepseek"), default="agicto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cohort", choices=("exposed", "new"), default="new",
                        help="exposed=original 20 series (mechanism smoke); new=unused series")
    parser.add_argument("--exclude-cache", type=Path, default=None,
                        help="JSON list of already-used series uids to exclude (second unused cohort)")
    parser.add_argument("--arms", default="A3,L+,L±,MISMATCH", help="arms to run (comma-separated)")
    parser.add_argument("--scan-report", type=Path, default=None,
                        help="override scan report path (default per-domain)")
    parser.add_argument("--save-cohort", type=Path, default=None,
                        help="save the selected new-cohort uids to this JSON file")
    parser.add_argument("--tag", default="", help="report suffix for paired repeats")
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain

    # 经验源：NN5 扫描结果自动转 episodes
    scan_path = args.scan_report or (root / f"artifacts/functional/e2/w2_operator_scan_report_{domain}.json")
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    episodes = episodes_from_scan(scan, domain)
    print(f"== episodes from scan: {len(episodes)}")
    for ep in episodes:
        print(f"   {ep.episode_id:32s} relation={ep.relation:9s} status={ep.local_status} "
              f"sig={ep.workflow_signature}")

    # cohort roster（monkey-patch）
    config = dict(v6.DATASET_CONFIGS[domain])
    if args.cohort == "exposed":
        v6._fixed_roster = _ORIG_FIXED_ROSTER  # 已暴露 20 条（机制 smoke）
        roster, values = _ORIG_FIXED_ROSTER(root, config)
        used = {str(r["series_uid"]) for r in roster}
    else:
        v6._fixed_roster = _roster_new_cohort
        if args.exclude_cache is not None and args.exclude_cache.is_file():
            cached = json.loads(args.exclude_cache.read_text(encoding="utf-8"))
            _USED_UIDS = set(str(u) for u in cached)  # 覆盖为"已用全集"（含前一组新 cohort）
        else:
            _USED_UIDS = None  # 首次：只排除原 20
        roster, values = _roster_new_cohort(root, config)
        used = _USED_UIDS or set()
        if args.save_cohort is not None:
            args.save_cohort.parent.mkdir(parents=True, exist_ok=True)
            args.save_cohort.write_text(
                json.dumps(sorted(str(r["series_uid"]) for r in roster), ensure_ascii=False) + chr(10),
                encoding="utf-8",
            )
            print(f"== cohort saved -> {args.save_cohort}")
    new_uids = {str(row["series_uid"]) for row in roster}
    print(f"== new cohort: {len(new_uids)} unused series (disjoint from used: "
          f"{used.isdisjoint(new_uids)})")

    # 检索器（NN5 扫描经验）
    retriever = SignedEpisodeRetriever(episodes, task_consumer_key="forecast|ridge_smase")
    query_context = {
        "cohort": {"series_count": 32, "evaluation_series_count": 8},
        "local_pattern": {"support_gain": 0.0},
        "program_geometry": {"scope": "training_rows"},
    }
    pack = retriever.retrieve(query_context, domain)

    # 四 Arm generation（LLM；dry-run 用固定候选）
    def make_proposer(arm: str):
        if args.dry_run:
            class _Mock:
                def __init__(self):
                    self.saw_positive = False
                    self.calls = 0
                def __call__(self, payload):
                    self.calls += 1
                    p = payload.get("experience_contrast_pack")
                    if isinstance(p, Mapping) and p.get("positive"):
                        self.saw_positive = True
                        return {"decision": "PROPOSE",
                                "steps": [{"op": "period_median_complete",
                                           "params": {"period": 7, "cycles": 3, "min_donors": 2},
                                           "bindings": {}}],
                                "requested_observations": [], "fallback": "IDENTITY"}
                    return {"decision": "ABSTAIN"}
            return _Mock()
        cfg = {
            "agicto": ("OPENAI_API_KEY", "AGICTO_API_KEY", "https://api.agicto.cn/v1", "gpt-5.6-luna"),
            "deepseek": ("DEEPSEEK_API_KEY", "", "https://api.deepseek.com", "deepseek-v4-flash"),
        }[args.provider]
        api_key = next((os.environ.get(k, "").strip() for k in cfg[:2] if os.environ.get(k, "").strip()), "")
        if not api_key:
            raise SystemExit(f"{cfg[0]} required for provider={args.provider}")
        return v6.LiveJSONProposer(api_key=api_key, model=args.model or cfg[3], base_url=cfg[2])

    results: dict[str, Any] = {}
    arm_list = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    for arm in arm_list:
        proposer = make_proposer(arm)

        def _proposer_with_pack(payload: Mapping[str, object], *, arm=arm, proposer=proposer, pack=pack) -> Mapping[str, object]:
            pl = copy.deepcopy(dict(payload))
            if arm == "A3":
                pass
            elif arm == "L+":
                pl["experience_contrast_pack"] = (
                    {"positive": pack.positive.to_dict() if pack.positive else None,
                     "negative": None, "conflict": None,
                     "evidence_sufficient": pack.evidence_sufficient, "retrieval_note": "positive-only"}
                )
            elif arm == "L±":
                pl["experience_contrast_pack"] = pack.to_dict()
            elif arm == "MISMATCH":
                pl["experience_contrast_pack"] = ContrastPack(
                    positive=build_episode(
                        episode_id="mismatch_ctx", task_consumer_key="forecast|ridge_smase",
                        domain_namespace="other_domain",
                        context_summary={"cohort": {"series_count": 999}, "local_pattern": {"support_gain": -9.0},
                                         "program_geometry": {"scope": "wrong"}},
                        workflow_signature="period_median_complete", support_response={}, delayed_response={},
                        relation=RELATION_POSITIVE, evidence_level="SUPPORT", local_status="EPISODE_ONLY",
                    ) if pack.positive else None,
                    negative=None, conflict=None, evidence_sufficient=False,
                    retrieval_note="mismatched context control",
                ).to_dict()
            return proposer(pl)

        report = v6.run(root, initial_proposer=_proposer_with_pack, revision_proposer=_proposer_with_pack,
                        dataset_key=domain, write_report=False)
        gains = []
        proposals = report.get("generation_proposals")
        if isinstance(proposals, list):
            for row in proposals:
                if isinstance(row, dict):
                    sr = row.get("support_response")
                    if isinstance(sr, dict) and isinstance(sr.get("support_gain"), (int, float)):
                        gains.append(float(sr["support_gain"]))
        selection = report.get("selection")
        delayed_gain = None
        if isinstance(selection, dict) and isinstance(selection.get("selection_gain"), (int, float)):
            delayed_gain = float(selection["selection_gain"])
        injected = []
        if arm == "L+":
            injected = [pack.positive.episode_id] if pack.positive else []
        elif arm == "L±":
            injected = [ep.episode_id for ep in (pack.positive, pack.negative, pack.conflict) if ep is not None]
        cand_sigs = []
        exp_use: list[str] = []
        if isinstance(proposals, list):
            for row in proposals:
                if isinstance(row, dict):
                    steps = row.get("compiled_program_steps") or row.get("workflow_steps")
                    if isinstance(steps, list):
                        cand_sigs.append(workflow_signature_of([s for s in steps if isinstance(s, dict)]))
                    eu = row.get("experience_use")
                    if isinstance(eu, list):
                        exp_use.extend(str(x) for x in eu if isinstance(x, str))
        invalid_use = [e for e in exp_use if e not in injected]
        results[arm] = {
            "support_gains": gains,
            "first_positive_probe": next((i + 1 for i, g in enumerate(gains) if g > 0), None),
            "delayed_gain": delayed_gain,
            "delayed_harm": bool(delayed_gain is not None and delayed_gain < 0),
            "abstained": (len(gains) == 0),
            "injected_episode_ids": injected,
            "candidate_signatures": cand_sigs,
            "experience_use": exp_use,
            "experience_use_invalid": invalid_use,
            "final_status": report.get("final_status"),
        }
        print(f"  {arm:8s} support={[round(g, 4) for g in gains]} "
              f"delayed={delayed_gain if delayed_gain is not None else 'n/a':>9} "
              f"injected={injected}")

    # 判定（用户裁决；Arm 集合动态——smoke 可能只跑 A3,L+）
    l_arms = [a for a in ("L+", "L±") if a in results and results[a]["delayed_gain"] is not None]
    l_ok = any(results[a]["delayed_gain"] > 0 or results[a]["abstained"] for a in l_arms) if l_arms else False
    a3 = results.get("A3", {})
    mm = results.get("MISMATCH", {})
    a3_delayed = a3.get("delayed_gain")
    mm_delayed = mm.get("delayed_gain")
    better_than_baselines = all(
        (l_d is None) or (b_d is None) or (l_d >= b_d)
        for l_d in [results[a]["delayed_gain"] for a in l_arms]
        for b_d in [a3_delayed, mm_delayed]
    )
    # 非平凡计划差异门（评审裁决）：Memory Arm 相对 A3/MISMATCH 计划完全相同不得通过
    a3_sigs = a3.get("candidate_signatures") or []
    mm_sigs = mm.get("candidate_signatures") or []
    l_non_trivial = any(
        (results[a].get("candidate_signatures") or []) not in (a3_sigs, mm_sigs)
        for a in ("L+", "L±") if a in results
    )
    all_use_valid = all(
        bool(results[a].get("experience_use"))
        and not results[a].get("experience_use_invalid")
        for a in ("L+", "L±") if a in results
    )
    verdict = (
        "PROCEED_TO_THREE_DOMAIN"
        if (l_ok and better_than_baselines and l_non_trivial and all_use_valid)
        else ("STOP_IDENTICAL_PLAN" if not l_non_trivial else "STOP")
    )
    print(f"\n== verdict: {verdict} (L+ or L± delayed positive/abstain on sealed C, "
          f"not worse than A3/MISMATCH, non-trivial plan difference, valid experience_use)")

    out = root / REPORT_OUT_REL.with_name(
        f"w2_focus_recheck_report{('_' + args.tag) if args.tag else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "w2-focus-recheck-unused-nn5-cohort",
            "domain": domain,
            "new_cohort_series_count": len(new_uids),
            "cohort_disjoint_from_used": used.isdisjoint(new_uids),
            "episodes": [ep.to_dict() for ep in episodes],
            "arms": results,
            "verdict": verdict,
            "mode": "dry-run" if args.dry_run else "live",
            "experiment_status": "INCONCLUSIVE_MECHANICAL_EXPRESSION_GAP" if not args.dry_run else "mechanism-smoke",
            "llm_api_call_count": 0 if args.dry_run else "see arms",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
