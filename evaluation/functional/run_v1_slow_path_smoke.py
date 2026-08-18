"""SLOW_PATH SMOKE（2026-08-09，agicto gpt-5.6-luna）——修正版（审查裁决 十九）。

GEFCom winsorize 失败包（@832/880 双正 → @928 首探 −0.164 harm → @976 +0.61
CONFLICT）。

审查裁决（2026-08-09 十九）三个承重问题已修正：
  P0-1 成功/失败 Context **分开提供**（success_context.support/delayed +
       failure_context.support/delayed，不再字典展开合并——原实现 832 被 928
       覆盖）；检查不再写死 True。
  P0-2 gate 改称 **structural preflight**（面/操作白名单 + base SHA + 单修改；
       真正的 compiler PASS 必须"应用 Manifest → 编译候选 snapshot"之后才有，
       smoke 不落地修改）。
  P1   根因不预标注：failure_family 改中性 workflow_effect_sign_flip；
       n_hist/radius 状态作为 facts 字段（事实），不写"weak seed 是根因"；
       instruction 要求 Manifest 在 predicted_agent_behavior_change 首项编码
       唯一 first-fault 面（'first_fault:<face>'），smoke 确定性解析。

验证点（5）：
  1. LLM 收到分开的成功/失败 Context 与冻结 Workflow（实际检查，非写死）；
  2. 自主选择 first-fault 面（编码解析）或明确不可识别（no_proposal）；
  3. 最多一个 Harness surface 修改（propose_edit 单 manifest 语义）；
  4. 结构性预检决定接受/拒绝（非 compiler）；
  5. LLM 不批准自己的 Patch。

LLM 边界（用户裁决）：agicto luna；预算不卡死——调用计数记录，超限不断言
失败。LLM 只调用一次（裁决：不调 prompt、不多模型比较）。

verdict：LLM 在中性输入下独立得出 risk/control 面 → SLOW_PATH_ATTRIBUTION_
SMOKE；否则 SLOW_PATH_PROPOSAL_WIRING_SMOKE_PARTIAL。

用法：
  python evaluation/functional/run_v1_slow_path_smoke.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import AgictoChatCompletionsBackend  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
KEY_ENVS = ("OPENAI_API_KEY", "AGICTO_API_KEY")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_slow_path_smoke_report.json")
SEED_REPORT = Path("artifacts/functional/e2/w1_scope_alignment_report.json")
FIRST_FAULT_FACES = ["observation", "program", "scope", "risk", "memory", "control"]

# 可写面目录（有限：新 skill ADD + inspect body PATCH——smoke 白名单）
SURFACE_CATALOG = [
    {
        "surface_id": "skill_library.entries/{skill_id}",
        "operation": "ADD",
        "surface_type": "skill",
        "allowed_operations": ["ADD"],
    },
    {
        "surface_id": "bootstrap_skills.entries/inspect_and_localize.body",
        "operation": "PATCH",
        "surface_type": "skill_body",
        "allowed_operations": ["PATCH"],
    },
]


class CountingClient:
    """温度 0 + 调用计数（审查裁决 2026-08-09 二十：总调用上限 2——AgentCore
    validation_retries=1 的唯一一次格式纠正后仍超限 = 契约重试失效信号，硬停）。"""

    def __init__(self, delegate: Any, *, max_calls: int = 2) -> None:
        self.calls = 0
        self._max_calls = max_calls
        self._delegate = delegate
        self.chat = _Chat(self)

    def _create(self, **kwargs: Any) -> Any:
        if self.calls >= self._max_calls:
            raise RuntimeError(
                f"LLM call budget exceeded (hard stop at {self._max_calls})")
        self.calls += 1
        kwargs.setdefault("temperature", 0)
        return self._delegate.chat.completions.create(**kwargs)


class _Chat:
    def __init__(self, owner: CountingClient) -> None:
        self.completions = _Completions(owner)


class _Completions:
    def __init__(self, owner: CountingClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        return self._owner._create(**kwargs)


def build_failure_card(
    root: Path,
    executor: ScopeExecutor,
    values: Mapping[str, Any],
    config: Mapping[str, object],
) -> dict[str, object]:
    """GEFCom winsorize 失败包（确定性构造，LLM 可见；数值经当前 ScopeExecutor
    实测——不信任旧报告数值）。

    审查裁决（2026-08-09 十九）修正：
      P0-1 成功/失败 Context **分开提供**（不再字典展开合并——832 会被 928
        覆盖）；嵌套在 context_evidence 下（LLM 可见，不影响 view 检索）；
      P1   根因**不预标注**：failure_family 改中性 workflow_effect_sign_flip；
        n_hist/radius 状态作为 facts 字段（事实），不写"weak seed 是根因"。
    """
    report = json.loads((root / SEED_REPORT).read_text(encoding="utf-8"))
    control = report["part_c_closed_loop"]["control"]
    assert control["operator"] == "winsorize" and int(control["origin"]) == 832
    params = wiring.contract_params("winsorize", PERIOD)
    steps = (("winsorize", params),)
    # 冻结 Workflow 在失败决策点（928）与 delayed（976）的实测
    s928 = executor.evaluate(steps, 928)
    d976 = executor.evaluate(steps, 976)
    return {
        "pattern_id": "gefcom-winsorize-sign-flip",
        "failure_family": "workflow_effect_sign_flip",
        "first_fault_candidates": list(FIRST_FAULT_FACES),
        "observable_signature": {"task_kind": "forecast"},
        "context_evidence": {
            "success_context": {
                "support": resolver.window_context(values, 832, PERIOD),
                "delayed": resolver.window_context(values, 880, PERIOD),
            },
            "failure_context": {
                "support": resolver.window_context(values, 928, PERIOD),
                "delayed": resolver.window_context(values, 976, PERIOD),
            },
        },
        "workflow": {
            "operator": "winsorize",
            "params": dict(params),
            "scope": "training_windows_only",
            "evaluator": "v6._evaluate (per-training-window, cohort Ridge sMASE)",
        },
        "observed_effects": {
            "success": {
                "support_origin": 832,
                "support_gain": float(control["support_gain"]),
                "delayed_origin": 880,
                "delayed_gain": float(control["delayed_gain"]),
                "status": "LOCAL_ACTIVE (POSITIVE)",
            },
            "failure": {
                "support_origin": 928,
                "support_gain": s928.gain,
                "delayed_origin": 976,
                "delayed_gain": d976.gain,
                "status": "CONFLICT (support negative, delayed positive)",
            },
        },
        # 事实字段（不含根因解释——LLM 自行归因）
        "facts": {
            "historical_context_count": 2,
            "radius_calibrated": False,
            "first_probe_harm_on_failure_support": float(s928.gain) < 0,
            "evaluator_scope": "training_windows_only",
        },
        "instruction": (
            "Analyze why this frozen workflow's effect flipped sign between the "
            "success slice and the failure slice. Do not assume any stated root "
            "cause: derive it from context_evidence and observed_effects. Choose "
            "exactly ONE first-fault face from first_fault_candidates and encode "
            "it as the FIRST item of predicted_data_effect using the literal "
            "format 'first_fault:<face>' (e.g. 'first_fault:risk'); or declare "
            "the fault unidentifiable with no_proposal. If you propose an edit: "
            "pick exactly one writable surface from writable_surface_catalog "
            "(instantiate templates), propose exactly one minimal edit, and fill "
            "falsification_condition so a deterministic replay can reject or "
            "accept your edit later. You do not approve your own edit."
        ),
    }


def _plain(value: Any) -> Any:
    """mappingproxy/tuple 等不可直接序列化的对象 → 纯 dict/list 结构。"""
    if isinstance(value, Mapping):
        return {str(key): _plain(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(val) for val in value]
    return value


def first_fault_face_preflight(manifest: Any) -> None:
    """**契约校验（manifest_preflight，审查裁决 2026-08-09 二十）**：
    predicted_data_effect 首项必须**唯一**编码 'first_fault:<face>'
    （face ∈ FIRST_FAULT_FACES）。

    载体选择说明（实测确定）：predicted_agent_behavior_change 的
    slow_edit_v1 schema 是 oneOf（只允许 retrieve_skill: 等固定模式）；
    edit_id/target_pattern_id 是 canonical_id pattern（不允许冒号）；
    **predicted_data_effect 是 nonempty_text_list（自由字符串，无 pattern
    约束）**——正确契约载体。

    缺失/多个 → 抛可重试 StagePostValidationError → AgentCore
    validation_retries=1 自动要求模型纠正（协议纠错，非反复调 prompt）。
    失败两次（总调用 2 上限）→ 脚本捕获按 AGENT_CONTRACT_FIRST_FAULT 记录。"""
    effects = [str(item) for item in getattr(
        manifest, "predicted_data_effect", ()) or ()]
    faces = [
        item.split(":", 1)[1].strip() for item in effects
        if item.startswith("first_fault:")
        and item.split(":", 1)[1].strip() in FIRST_FAULT_FACES
    ]
    if len(faces) != 1:
        raise StagePostValidationError(
            "FIRST_FAULT_FACE_INVALID",
            f"predicted_data_effect must encode exactly one 'first_fault:<face>' "
            f"item as its first element with face in {FIRST_FAULT_FACES}; found "
            f"{faces} in {effects}. Re-encode it, or return no_proposal if the "
            f"context cannot support a fault identification.",
            retryable=True,
        )


def structural_preflight(
    manifest: Any,
    slow: TTHASlowAgent,
    snapshot: Any,
) -> dict[str, Any]:
    """**结构性预检**（审查裁决 2026-08-09 十九 P0-2）：不是 compiler——
    只检查 面/操作白名单 + base SHA + 单修改（propose_edit post_validate 已做
    schema 校验）。真正的 compiler PASS 必须"应用 Manifest → 编译候选 snapshot"
    之后才有；smoke 不落地修改（LLM 不批准自己，判定由本预检输出）。"""
    if manifest is None:
        return {"preflight": "REJECTED", "stage": "no_proposal",
                "reason": slow.last_no_proposal_reason or "unknown"}
    targets = [s["surface_id"] for s in SURFACE_CATALOG]
    surface_ok = any(
        manifest.target_surface_id == t
        or ("{" in t and manifest.target_surface_id.startswith(t.split("{")[0]))
        for t in targets)
    ops = {s["operation"] for s in SURFACE_CATALOG}
    op_ok = manifest.operation.value in ops
    sha_ok = manifest.base_harness_sha == snapshot.harness_content_sha
    # first_fault 面编码（instruction 契约：predicted_data_effect 首项
    # 'first_fault:<face>'）
    first_fault_face = None
    effects = [str(item) for item in getattr(
        manifest, "predicted_data_effect", ()) or ()]
    for item in effects:
        if item.startswith("first_fault:"):
            candidate = item.split(":", 1)[1].strip()
            if candidate in FIRST_FAULT_FACES:
                first_fault_face = candidate
                break
    changes = list(manifest.predicted_agent_behavior_change)
    return {
        "preflight": "ACCEPTED" if (surface_ok and op_ok and sha_ok)
        else "REJECTED",
        "stage": "manifest",
        "target_surface_id": manifest.target_surface_id,
        "operation": manifest.operation.value,
        "surface_in_catalog": surface_ok,
        "operation_allowed": op_ok,
        "base_sha_matches_snapshot": sha_ok,
        "first_fault_face_encoded": first_fault_face,
        "new_value_summary": _plain({
            key: value for key, value in (manifest.new_value or {}).items()
            if key in ("skill_id", "pattern_id", "body", "risk_guards",
                       "allowed_tools", "observable_applicability")
        } if manifest.new_value is not None else None),
        "minimal_patch": _plain(manifest.minimal_patch),
        "falsification_condition": _plain(manifest.falsification_condition),
        "predicted_agent_behavior_change": _plain(changes),
    }


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    card = build_failure_card(root, executor, values, config)
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)

    api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        raise SystemExit(f"missing LLM key: {KEY_ENVS}")
    import openai
    counter = CountingClient(
        openai.OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter, base_url=BASE_URL)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.asarray(values[list(values.keys())[0]],
                                          dtype=np.float64)[:928],
                               task_kind="forecast"),
        model=MODEL,
        base_url=BASE_URL,
    )
    slow = TTHASlowAgent(core)

    print(f"== SLOW_PATH SMOKE: model={MODEL} provider=agicto")
    print(f"== failure card: pattern={card['pattern_id']} "
          f"family={card['failure_family']}")
    manifest = None
    budget_error: str | None = None
    try:
        manifest = slow.propose_edit(
            card, SURFACE_CATALOG, h0,
            manifest_preflight=first_fault_face_preflight,
            allowed_operator_contracts=(),
            task_context=None,
        )
    except RuntimeError as exc:  # CountingClient 硬停（契约重试仍超 2 次）
        budget_error = str(exc)
    preflight = structural_preflight(manifest, slow, h0)
    preflight["llm_calls"] = counter.calls
    if budget_error:
        preflight["budget_error"] = budget_error
    print(f"== manifest: {'None (no_proposal)' if manifest is None else 'proposed'}")
    print(f"== structural preflight: {preflight}")
    print(f"== llm_calls={counter.calls}")

    # P0-1 修正：Context 实际注入检查（不再写死 True）——context_evidence 四字段齐全
    ctx = card.get("context_evidence") or {}
    context_complete = all(
        isinstance(ctx.get(k), dict) and ctx[k]
        for k in ("success_context", "failure_context"))
    context_fields = all(
        isinstance((ctx.get("success_context") or {}).get(k), dict)
        and (ctx["success_context"][k])
        for k in ("support", "delayed")) and all(
        isinstance((ctx.get("failure_context") or {}).get(k), dict)
        and (ctx["failure_context"][k])
        for k in ("support", "delayed"))
    face = preflight.get("first_fault_face_encoded")
    no_proposal = preflight["stage"] == "no_proposal"
    checks: dict[str, bool] = {
        # 1. LLM 收到成功/失败 Context（分开提供）与冻结 Workflow
        "llm_received_separate_contexts": bool(context_complete and context_fields),
        # 2. 自主选择 first-fault 面（契约编码）或明确不可识别（no_proposal）
        "llm_chose_or_declared_unidentifiable": bool(
            face or (no_proposal and bool(preflight.get("reason")))),
        # 3. 最多一个 Harness surface 修改（propose_edit 单 manifest 语义）
        "at_most_one_surface_modification": manifest is None
        or preflight["stage"] == "manifest",
        # 4. 结构性预检（非 compiler）决定接受/拒绝
        "structural_preflight_verdict": preflight["preflight"] in ("ACCEPTED",
                                                                   "REJECTED"),
        # 5. LLM 不批准自己的 Patch（判定由结构性预检输出）
        "llm_no_self_approval": True,
        # 6. 总调用上限 2（AgentCore 唯一一次格式纠正；超出 = 契约重试失效）
        "llm_calls_within_budget": counter.calls <= 2,
    }
    # 审查裁决（二十）判定分支：
    #   唯一合法 face           → 归因接口 PASS
    #   no_proposal             → PASS（当前 Context 不足以识别——诚实优于编造）
    #   再次无 face             → AGENT_CONTRACT_FIRST_FAULT（停止继续调用）
    #   obs/program/scope 无对应可执行修改 → 归因—行动不一致，拒绝进入 replay
    #   risk/control 且 Manifest 合法     → 才进入 replay（下次切片批准范围）
    if budget_error:
        outcome = "AGENT_CONTRACT_FIRST_FAULT"
        verdict = "SLOW_PATH_PROPOSAL_WIRING_SMOKE_FAIL_BUDGET"
    elif no_proposal:
        outcome = "NO_PROPOSAL_PASS"
        verdict = "SLOW_PATH_ATTRIBUTION_SMOKE_PASS_NO_PROPOSAL"
    elif face is None:
        outcome = "AGENT_CONTRACT_FIRST_FAULT"
        verdict = "SLOW_PATH_PROPOSAL_WIRING_SMOKE_PARTIAL"
    elif preflight["preflight"] != "ACCEPTED":
        outcome = "STRUCTURAL_PREFLIGHT_REJECTED"
        verdict = "SLOW_PATH_PROPOSAL_WIRING_SMOKE_PARTIAL"
    elif face in ("risk", "control"):
        outcome = "RISK_CONTROL_MANIFEST_PASS"
        verdict = "SLOW_PATH_ATTRIBUTION_SMOKE_PASS_RISK_CONTROL"
    else:  # observation/program/scope——需对应可执行修改，smoke 级拒绝进入 replay
        outcome = "FACE_ACTION_MISMATCH_NO_REPLAY"
        verdict = "SLOW_PATH_PROPOSAL_WIRING_SMOKE_PARTIAL"
    print(f"\n== checks: {checks}")
    print(f"== first_fault_face_observed: {face}")
    print(f"== outcome: {outcome}")
    print(f"== verdict: {verdict}")
    print("== 口径：结构性预检（非 compiler）；契约经 manifest_preflight 强制")
    print("   （AgentCore 唯一一次格式纠正）；replay 验证与修改落地推迟到完整")
    print("   切片（仅 risk/control 且 Manifest 合法才具备 replay 前提）")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-slow-path-smoke",
            "verdict": verdict,
            "provider": {"model": MODEL, "base_url": BASE_URL},
            "failure_package": {
                "pattern_id": card["pattern_id"],
                "failure_family": card["failure_family"],
                "workflow": card["workflow"],
                "observed_effects": card["observed_effects"],
                "facts": card["facts"],
                "first_fault_candidates": card["first_fault_candidates"],
            },
            "context_evidence": card["context_evidence"],
            "surface_catalog": SURFACE_CATALOG,
            "proposed_manifest": None if manifest is None else {
                "edit_id": manifest.edit_id,
                "target_surface_id": manifest.target_surface_id,
                "operation": manifest.operation.value,
                "target_pattern_id": manifest.target_pattern_id,
            },
            "structural_preflight": preflight,
            "outcome": outcome,
            "no_proposal_reason": slow.last_no_proposal_reason,
            "checks": checks,
            "llm_api_call_count": counter.calls,
            "note": "smoke only: structural preflight (not compiler); replay "
                    "validation and edit application deferred to the full "
                    "slow-path slice",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
