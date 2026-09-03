"""EVIDENCE_COMPILER_DEV（P0 降级设计验证，2026-08-13：确定性 Evidence
Compiler——Runtime 依据 Batch Evidence 筛 robust candidate/abstain，
LLM 只编译 Typed Patch——development exposure——零新 Claim）。

背景（冻结结论）：P0 v1/v2 均 FAIL——LLM_BATCH_EVIDENCE_INTEGRATION_
NOT_ESTABLISHED（机械 first_fault=BATCH_ALIGNMENT_NOT_USED；细胞层
细化=SEMANTIC_OPERATOR_PRIOR_DOMINATES——patch 案例弃权 reason 全部
no_authorized_minimal_edit，swap 臂 5/6 跟随证据）。降级设计：决策层
确定性化，LLM 角色收窄为 manifest 编译。

验证链（两个自然结构——同一 M 同一门）：
  A. 自然 patch 结构（T117 winsorize 组 [888,984]——Wave2 witness 同构）：
     headroom（gate1 报告）→ unique_common_positive → hampel →
     evidence_compiler=True + runtime_selected_patch_id → 白名单收敛 →
     ReplaySlowAgent 编译（只编译——不选择）→ 组内 replay 全 ≥M →
     holdout @600 ≥ −M → pending。
  B. 自然 abstain 结构（DEV family 6 窗口——Wave4a 同构）：
     headroom（census 报告）→ unique_common_positive → None →
     evidence_compiler=True, runtime_selected_patch_id=None →
     stage=evidence_abstain + **零 LLM 调用**（wrapper 断言 propose_edit
     未被调用）。

verdict（预注册）：
  EVIDENCE_COMPILER_DEGRADED_CHAIN_PASS : A 链到 pending 且 B 确定性
    弃权零调用
  EVIDENCE_COMPILER_SELECTION_WRONG     : 确定性选择与门不符
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_evidence_compiler_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))
from SelfEvolvingHarnessTS.methods.ttha.program_supply import controlled_add_only_group_catalog, controlled_add_only_group_decision

import numpy as np  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
)
from run_v1_operational_self_evolution_loop import (  # noqa: E402
    ReplaySlowAgent,
    _skill_manifest,
)
from run_v1_batch_census_dev import _dev_executor, _load_series  # noqa: E402

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    deployment_constraints_v1,
    forecast_task_context_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    group_first_faults,
    unique_common_positive,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
M = resolver.MATERIAL_THRESHOLD
E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_evidence_compiler_dev_report.json"


class GuardedSlow:
    """包装 ReplaySlowAgent——断言 propose_edit 不被调用（确定性
    abstain 路径零 LLM 证据）。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.called = False

    def propose_edit(self, *args, **kw):
        self.called = True
        return self._inner.propose_edit(*args, **kw)


def _executor_t117(root: Path) -> ScopeExecutor:
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    values = {str(r["series_name"]): _load_series(root, str(r["series_name"]))
              for r in rows}
    return ScopeExecutor(roster, values, _config(),
                         evaluate_fn=_evaluate_kdd)


def _failure_episode(domain: str, series: str, origin: int,
                     gain: float, per_view: list[float],
                     steps: list[dict]) -> Any:
    return build_episode(
        episode_id=f"{domain}_target_winsorize_ec_{series}_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=domain,
        context_summary={
            "local_pattern": {"support_gain": gain},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": steps},
            "per_view_gain": list(per_view),
            "support_origin": origin,
        },
        workflow_signature=workflow_signature_of(steps),
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["evidence_compiler_dev"])


def main() -> int:
    root = PROJECT_ROOT
    gate1 = json.loads((E2 / "w1_group_evidence_chain_gate1_report.json")
                       .read_text(encoding="utf-8"))
    census = json.loads((E2 / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))

    # ---- A：自然 patch 结构（T117 组——确定性选择 hampel → LLM 只编译）
    ex117 = _executor_t117(root)
    t117_headroom = gate1["headroom"]
    alts = ("outlier_mad", "hampel_filter")
    chosen = unique_common_positive(t117_headroom, alts)
    checks: dict[str, Any] = {"t117_runtime_choice": chosen}
    if chosen != "hampel_filter":
        print(json.dumps({"verdict": "EVIDENCE_COMPILER_SELECTION_WRONG",
                          "checks": checks}, indent=1))
        return 0
    # 已暴露 T117 winsorize 失败值（w1_group_auto_trigger_dev_report.json
    # probes——零新评估）
    t117_fail_gains = [-0.1426334267351992, -0.08411687539427182]
    eps = [_failure_episode("kdd_cup_2018", "T117", o,
                            t117_fail_gains[i],
                            [], [{"op": "winsorize", "params": {}}])
           for i, o in enumerate((888, 984))]
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no group"}, indent=1))
        return 0
    group = groups[0]

    def _t117_card(g: Mapping[str, Any], cap: Mapping[str, Any]):
        return {
            "pattern_id": "group-winsorize-neg",
            "failure_family": "workflow_component_negative",
            "observable_signature": {"task_kind": "forecast"},
            "context": {},
            "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
            "typed_patch_options": [
                {"patch_id": f"patch-replace-winsorize-with-{alt}",
                 "program_steps": [{"op": alt,
                                    "params": dict(wiring.contract_params(
                                        alt, PERIOD))}]}
                for alt in alts],
            "facts": {"contrast_capsule": dict(cap)},
        }

    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    store = SnapshotStore(root / ".ec_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    manifest = _skill_manifest(
        skill_id="group_winsorize_replacement", op="hampel_filter",
        params=dict(wiring.contract_params("hampel_filter", PERIOD)),
        patch_id="patch-replace-winsorize-with-hampel_filter",
        base_sha=h0.harness_content_sha)
    slow = ReplaySlowAgent(manifest)
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            slow, LocalPublicToolGateway(
                _load_series(root, "T117")[:888], task_kind="forecast"))),
        h0, tuple(eps))
    ev_a = method.handle_group_feedback(group, {'workflow': 'winsorize', 'sign': 'NEGATIVE', 'n_episodes': 2}, slow_agent=slow, controller=controller, store=store, card_builder=_t117_card, evaluator_group=lambda s, e: ex117.evaluate(tuple(s), int((getattr(e, 'context_summary', {}) or {}).get('support_origin') or 0)), holdout_evaluator=lambda s, _m: ex117.evaluate(tuple(s), 600), fast_features=dict(extract_public_features(_load_series(root, 'T117')[:888], task_kind='forecast')), evidence_compiler=True, runtime_selected_patch_id='patch-replace-winsorize-with-hampel_filter', surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_evidence_compiler_dev-206'))
    checks["t117_chain"] = ev_a

    # ---- B：自然 abstain 结构（DEV family——确定性弃权零 LLM）
    fam0 = census["development_families"][0]
    dev_headroom = fam0["replacement_headroom"]
    chosen_b = unique_common_positive(dev_headroom,
                                      ("winsorize", "outlier_mad",
                                       "hampel_filter"))
    checks["dev_runtime_choice"] = chosen_b
    evals_b: dict[str, Any] = {}
    if chosen_b is None:
        ex_t100, _ = _dev_executor(root, "T100")
        eps_b = [_failure_episode(
            "kdd2018_dev_" + w["series"], w["series"], w["origin"],
            w["gain"], [], [{"op": "winsorize", "params": {}}])
            for w in fam0["episodes"]]
        groups_b = group_first_faults(eps_b, min_group=2)
        if not groups_b:
            print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                              "reason": "no dev group"}, indent=1))
            return 0
        group_b = groups_b[0]
        guarded = GuardedSlow(ReplaySlowAgent(manifest))
        method_b = TTHAMethod(
            TTHAFastAgent(TTHAAgentCore(
                ReplaySlowAgent(manifest),
                LocalPublicToolGateway(_load_series(root, "T100")[:600],
                                       task_kind="forecast"))),
            h0, tuple(eps_b))
        ev_b = method_b.handle_group_feedback(group_b, {'workflow': 'winsorize', 'sign': 'NEGATIVE', 'n_episodes': 6}, slow_agent=guarded, controller=controller, store=store, card_builder=_t117_card, evaluator_group=lambda s, e: ex_t100.evaluate(tuple(s), int((getattr(e, 'context_summary', {}) or {}).get('support_origin') or 0)), holdout_evaluator=None, fast_features=dict(extract_public_features(_load_series(root, 'T100')[:600], task_kind='forecast')), evidence_compiler=True, runtime_selected_patch_id=None, surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_evidence_compiler_dev-248'))
        evals_b = {"event": ev_b, "slow_called": guarded.called,
                   "zero_llm": not guarded.called}

    ok = bool(
        chosen == "hampel_filter"
        and ev_a.get("stage") == "pending"
        and ev_a.get("patch_id") == "patch-replace-winsorize-with-hampel_filter"
        and chosen_b is None
        and evals_b.get("event", {}).get("stage") == "evidence_abstain"
        and evals_b.get("zero_llm") is True)
    verdict = ("EVIDENCE_COMPILER_DEGRADED_CHAIN_PASS" if ok
               else "PROTOCOL_FAILURE")
    report = {
        "experiment_id": "v1-evidence-compiler-dev",
        "note": "P0 降级设计验证：确定性 Evidence Compiler（Runtime 决策"
                "——LLM 只编译）——development exposure——零新 Claim。"
                "背景：P0 v1/v2 FAIL（LLM_BATCH_EVIDENCE_INTEGRATION_NOT_"
                "ESTABLISHED——语义先验主导弃权 no_authorized_minimal_edit）",
        "t117": {"runtime_choice": chosen, "chain": ev_a},
        "dev": evals_b,
        "verdict": verdict,
    }
    print("== t117 runtime choice:", chosen)
    print("== t117 chain:", json.dumps(ev_a, ensure_ascii=False))
    print("== dev runtime choice:", chosen_b)
    print("== dev chain:", json.dumps(evals_b, ensure_ascii=False,
                                      default=str))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
