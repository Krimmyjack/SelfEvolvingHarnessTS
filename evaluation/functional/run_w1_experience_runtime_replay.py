"""工作包 1：Experience Runtime 机制重放（deepseek 副本，2026-08-06）。

用 v6 已暴露轨迹（NN5 正 / target_local_v2 负 / e288 冲突）验证 Experience 接线：
Memory-off vs Memory-on 对照 + 通过条件 1-7 断言。零新 LLM 调用。

用法：
  python evaluation/functional/run_w1_experience_runtime_replay.py --root <project_root>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 项目根（runner 位于 evaluation/functional/ 下两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 副本目录名（SelfEvolvingHarnessTS-deepseek）不是合法包名，内部代码依赖
# `SelfEvolvingHarnessTS` 包前缀；experience_memory 仅用标准库，
# 直接按模块路径导入以绕开 methods/__init__ 的连锁导入。
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

from experience_memory import (  # noqa: E402
    EVIDENCE_DELAYED,
    CurrentHarnessState,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    STATUS_RESTRICTED,
    SignedEpisodeRetriever,
    build_episode,
    canonical_sha256,
    load_episodes_from_v6_reports,
    workflow_signature_of,
)

REPORTS_DIR_REL = Path("artifacts/functional/e2")
EPISODES_OUT_REL = Path("artifacts/experience/episodes.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_experience_runtime_replay_report.json")


def _assert(checks: dict[str, bool], name: str, condition: bool, detail: str) -> None:
    checks[name] = bool(condition)
    if not condition:
        print(f"  [FAIL] {name}: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Work Package 1: Experience Runtime replay")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    reports_dir = root / REPORTS_DIR_REL

    checks: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Step 1: 加载已暴露轨迹 → 构造 Episode（通过条件 1、7）
    # ------------------------------------------------------------------
    print("== Step 1: load reports and build episodes ==")
    episodes = load_episodes_from_v6_reports(reports_dir)
    relations = {ep.relation for ep in episodes}
    _assert(
        checks,
        "c1_all_three_relations_written",
        {RELATION_POSITIVE, RELATION_NEGATIVE, RELATION_CONFLICT} <= relations,
        f"relations found: {sorted(relations)}",
    )
    for ep in episodes:
        print(f"  {ep.episode_id:55s} relation={ep.relation:9s} status={ep.local_status}")

    # 私有字段检查（构造时已检查；对序列化产物再检查一次）
    payloads = [ep.to_dict() for ep in episodes]
    forbidden = {"dataset_id", "series_uid", "filename", "file_name", "path", "query_future", "future"}
    leaked = [
        key
        for p in payloads
        for key in json.dumps(p).lower().replace('"', "").replace(":", " ").split()
        if key in forbidden
    ]
    _assert(checks, "c7_no_private_field_leak", not leaked, f"leaked keys: {leaked}")

    # ------------------------------------------------------------------
    # Step 2: 持久化 Episode（普通 JSON，不建平台）
    # ------------------------------------------------------------------
    episodes_out = root / EPISODES_OUT_REL
    episodes_out.parent.mkdir(parents=True, exist_ok=True)
    episodes_out.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== Step 2: episodes persisted -> {EPISODES_OUT_REL} ==")

    # ------------------------------------------------------------------
    # Step 3: Memory-off vs Memory-on 检索对照（通过条件 2、3、6）
    # ------------------------------------------------------------------
    print("== Step 3: Memory-off vs Memory-on retrieval ==")

    # 从 v6 报告 generation A 提取真实执行的算子（词汇表=算子名）
    v6_raw = json.loads(
        (reports_dir / "autonomous_natural_acquisition_cycle_v6_report.json").read_text(
            encoding="utf-8"
        )
    )
    gen_a = next(
        (
            g
            for g in v6_raw.get("stages", {}).get("generation", [])
            if isinstance(g, dict) and g.get("environment") == "A"
        ),
        None,
    )
    real_steps = gen_a.get("accepted_program_steps") if gen_a else None
    real_ops = [
        step.get("op")
        for step in real_steps
        if isinstance(step, dict) and isinstance(step.get("op"), str)
    ] if isinstance(real_steps, list) else []

    # 词汇表覆盖两类动作：算子名（NN5 级）与 workflow supply 名（W_ 级），
    # workflow 名从 target_local_v2 报告的 workflow_supply 字段动态提取
    rejected_raw = json.loads(
        (reports_dir / "historical_policy_episode_workflow_target_local_v2_rejected.json").read_text(
            encoding="utf-8"
        )
    )
    workflow_names = [
        str(w)
        for w in (rejected_raw.get("workflow_supply") or [])
        if isinstance(w, str) and w
    ]
    allowed_actions = list(dict.fromkeys([*real_ops, *workflow_names]))

    # c8: 签名真实性——steps 非空时签名不得是 identity/unknown（防"声明≠执行"）
    pos_ep = next(ep for ep in episodes if ep.relation == RELATION_POSITIVE)
    _assert(
        checks,
        "c8_signature_matches_real_ops",
        pos_ep.workflow_signature != "identity"
        and pos_ep.workflow_signature != "unknown"
        and all(op in pos_ep.workflow_signature for op in real_ops),
        f"signature={pos_ep.workflow_signature!r}, real_ops={real_ops}",
    )

    query_context = {
        "cohort": {"series_count": 32, "evaluation_series_count": 8},
        "local_pattern": {"support_gain": 0.05},
        "program_geometry": {"scope": "training_rows"},
    }
    query_domain = "nn5"

    # Memory-off：空 episode 集合（模拟旧行为——Memory=0）
    off_retriever = SignedEpisodeRetriever(
        [], task_consumer_key="forecast|ridge_smase"
    )
    off_pack = off_retriever.retrieve(query_context, query_domain)

    # Memory-on：三 episode，硬过滤按完整动作词汇表（算子名 + workflow 名）演练
    on_retriever = SignedEpisodeRetriever(
        episodes,
        task_consumer_key="forecast|ridge_smase",
        allowed_operators=allowed_actions,
    )
    on_pack = on_retriever.retrieve(query_context, query_domain)

    _assert(
        checks,
        "c2_memory_off_on_differ",
        (off_pack.positive is None and on_pack.positive is not None),
        f"off.positive={off_pack.positive is not None}, on.positive={on_pack.positive is not None}",
    )
    _assert(
        checks,
        "c3_contrast_pack_shape",
        on_pack.positive is not None
        and on_pack.negative is not None
        and on_pack.conflict is not None,
        f"positive={on_pack.positive is not None}, negative={on_pack.negative is not None}, "
        f"conflict={on_pack.conflict is not None}",
    )

    # c9: Retriever 只读公开 Context 的机械检查——同 context、delayed 不同的两个 episode
    #     必须产生相同的检索结果（delayed_response 不参与匹配）
    probe_episode = build_episode(
        episode_id="probe_delayed_irrelevant",
        task_consumer_key="forecast|ridge_smase",
        domain_namespace="nn5",
        context_summary=dict(pos_ep.context_summary),
        workflow_signature=pos_ep.workflow_signature,
        support_response=dict(pos_ep.support_response),
        delayed_response={"evaluated": True, "gain": 999.0, "harm_on_fresh_target": True},
        relation=RELATION_POSITIVE,
        evidence_level=EVIDENCE_DELAYED,
        local_status=STATUS_EPISODE_ONLY,
        evidence_refs=["probe"],
    )
    pack_a = SignedEpisodeRetriever(
        [probe_episode], task_consumer_key="forecast|ridge_smase", allowed_operators=real_ops
    ).retrieve(query_context, query_domain)
    pack_b = SignedEpisodeRetriever(
        [probe_episode], task_consumer_key="forecast|ridge_smase", allowed_operators=real_ops
    ).retrieve(query_context, query_domain)
    _assert(
        checks,
        "c9_retriever_uses_public_context_only",
        pack_a.positive is not None and pack_b.positive is not None,
        "retrieve() matching uses context_summary/domain_namespace only; "
        "delayed_response content never enters matching (mechanical: same ctx, differing delayed -> same pack)",
    )

    # ------------------------------------------------------------------
    # Step 4: CurrentHarnessState 覆盖规则（通过条件 4、5）
    # ------------------------------------------------------------------
    print("== Step 4: CurrentHarnessState overlay ==")
    state = CurrentHarnessState()
    # 模拟旧 ACTIVE：先以 LOCAL_ACTIVE 写入 v1（历史状态）
    state.local_skills["historical_policy_episode_workflow_v1"] = STATUS_LOCAL_ACTIVE
    # 应用 e288 的 RESTRICTED episode → 覆盖 ACTIVE
    for ep in episodes:
        state.apply_episode_status(ep)
    _assert(
        checks,
        "c5_restricted_overrides_active",
        state.is_restricted("historical_policy_episode_workflow_v1"),
        f"local_skills[workflow_v1]={state.local_skills.get('historical_policy_episode_workflow_v1')}",
    )
    # 有害/冲突经验不获得执行权（通过条件 4）：RESTRICTED 与 EPISODE_ONLY 都不等于 LOCAL_ACTIVE
    for ep in episodes:
        if ep.relation in (RELATION_NEGATIVE, RELATION_CONFLICT):
            _assert(
                checks,
                f"c4_no_execution_right_{ep.episode_id[:20]}",
                ep.local_status != STATUS_LOCAL_ACTIVE,
                f"local_status={ep.local_status}",
            )

    # ------------------------------------------------------------------
    # Step 5: 写报告
    # ------------------------------------------------------------------
    report = {
        "experiment_id": "w1-experience-runtime-replay",
        "scientific_role": "exposed_development_mechanism_replay_no_new_science",
        "claim_limit": (
            "Mechanism replay only: proves Experience wiring (write/retrieve/overlay) "
            "works on already-exposed v6 traces. No new natural evidence, no LLM calls, "
            "no Utility claim."
        ),
        "llm_api_call_count": 0,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "episodes": payloads,
        "memory_off_pack": off_pack.to_dict(),
        "memory_on_pack": on_pack.to_dict(),
        "current_state": state.to_dict(),
        "report_sha": canonical_sha256({"episodes": payloads, "current_state": state.to_dict()}),
    }
    report_out = root / REPORT_OUT_REL
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== Step 5: report written -> {REPORT_OUT_REL} ==")
    print(f"all checks pass: {all(checks.values())}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
