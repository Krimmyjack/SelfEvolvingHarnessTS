"""GUIDANCE_EVOLUTION 聚焦测试（2026-08-14，rev6/P3 版）——方法层
handle_group_guidance 的 Runtime-owned binding 链（无 LLM——mock slow agent）：

  - Slow 提出的 manifest 携带错误/占位 surface_precondition.sha →
    Runtime（方法层）机械覆盖为真实 body SHA；
  - P3 clause 级编辑：minimal_patch.value = REPLACE_CLAUSE
    （target: propose.rule.<id> + new_clause）——Runtime 只替换该 clause，
    FIXED_CONTRACT / inspect_pattern_guidance / select_guidance 逐字不变；
  - surface/operation 白名单强制（非 guidance surface / 非 PATCH → 拒绝）；
  - malformed clause payload → clause_payload_invalid；unknown target →
    clause_target_unknown；
  - pending → activate_pending_guidance 需证据链齐全才写 active snapshot。

装置：compile_snapshot(h0, verify_lock=False)（h0 lock 与 dirty 作者根
不一致是既有状态——沿用全部 runner 的惯例）；SnapshotStore 用临时目录。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import (
    TTHAMethod,
    _apply_clause_replacement,
    _guidance_body,
    _parse_clause_payload,
)

SURFACE = "bootstrap_skills.entries/build_contrastive_candidates.body"
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

# REPLACE_CLAUSE 载荷（替换 propose.rule.no_legal_binding——文本与当前
# 不同，避免 NoOpEditError）
_NEW_CLAUSE = (
    "When no legal binding exists for a hypothesis, abstain by returning "
    "an empty candidate set rather than an unverifiable program; legality "
    "is decided by the operator contract and the runtime verifier, not by "
    "region width."
)
_CLAUSE_VALUE = (
    "REPLACE_CLAUSE\n"
    "target: propose.rule.no_legal_binding\n"
    "new_clause: " + _NEW_CLAUSE
)


class _FakeSlow:
    """真实 Slow 的确定性替身：返回一个 sha 错误的 PATCH manifest。"""

    def __init__(self, manifest: EditManifest | None) -> None:
        self.manifest = manifest
        self.last_no_proposal_reason = None
        self.last_stage_result = None

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        if self.manifest is None:
            self.last_no_proposal_reason = "insufficient_public_evidence"
            return None
        return self.manifest


def _manifest(target_surface: str, op: EditOperation, body: str,
              base_sha: str) -> EditManifest:
    return EditManifest(
        edit_id="test_guidance_clause",
        base_harness_sha=base_sha,
        target_pattern_id="test-group",
        target_surface_id=target_surface,
        operation=op,
        surface_precondition={"kind": "SHA",
                              "sha": "0" * 64},  # 故意错误——Runtime 覆盖
        dependency_precondition_shas={},
        minimal_patch={"value": body},
        new_value=None,
        observable_applicability=None,
        predicted_agent_behavior_change=("supply_effect_distinct",),
        predicted_data_effect=("evidence: test",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_candidate_supply",),
        patch_id=None,
    )


def _h0():
    return compile_snapshot(H0_ROOT, verify_lock=False)


def _card() -> Mapping[str, Any]:
    return {"pattern_id": "test-group-guidance",
            "observable_signature": {"task_kind": "forecast"}}


def test_clause_payload_parses_and_binds() -> None:
    payload = _parse_clause_payload(_CLAUSE_VALUE)
    assert payload is not None
    assert payload["target"] == "propose.rule.no_legal_binding"
    assert payload["new_clause"] == _NEW_CLAUSE
    body = _guidance_body(_h0())
    composed = _apply_clause_replacement(body, payload)
    assert _NEW_CLAUSE in composed
    assert "propose.rule.hypothesis_binding" in composed  # 其他 clause 未动
    # 固定段与其余 clause 逐字保留
    old_start = body.find("propose.rule.no_legal_binding: ")
    old_end = body.find("\npropose.rule.exploration_supply", old_start)
    assert composed[:old_start] == body[:old_start]
    assert composed[old_start + 1:].endswith(body[old_end:].strip("\n"))


def test_clause_payload_rejects_malformed() -> None:
    assert _parse_clause_payload("") is None
    assert _parse_clause_payload("ADD_CLAUSE\ntarget: x") is None
    assert _parse_clause_payload(
        "REPLACE_CLAUSE\ntarget: inspect.rule.x\nnew_clause: y") is None
    assert _parse_clause_payload(
        "REPLACE_CLAUSE\ntarget: propose.rule.x") is None
    assert _parse_clause_payload(
        "REPLACE_CLAUSE\ntarget: propose.rule.x\nnew_clause: a\nb") is None


def test_guidance_clause_patch_runtime_binds_sha_and_pends() -> None:
    h0 = _h0()
    old_body = _guidance_body(h0)
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        fake = _FakeSlow(_manifest(SURFACE, EditOperation.PATCH,
                                   _CLAUSE_VALUE, h0.harness_content_sha))
        method = TTHAMethod(object(), h0, ())
        ev = method.handle_group_guidance(
            {"workflow": "winsorize"}, {"n_episodes": 2},
            slow_agent=fake, controller=controller, store=store,
            card_builder=lambda g, c: _card(),
            confirmed_cause="WORKFLOW_GUIDANCE_GAP")
        assert ev["stage"] == "pending", ev
        assert ev["guidance_clause_proposed"]["target"] == \
            "propose.rule.no_legal_binding"
        assert ev["guidance_body_old"] == old_body
        pending_snap = method.pending_guidance_snapshot()
        assert pending_snap is not None
        assert pending_snap.harness_content_sha != h0.harness_content_sha
        new_body = _guidance_body(pending_snap)
        assert _NEW_CLAUSE in new_body
        # FIXED_CONTRACT / inspect / select 段逐字未动
        fixed = old_body[:old_body.find("[propose_construction_guidance]")]
        assert new_body.startswith(fixed)
        assert "propose.rule.hypothesis_binding" in new_body
        # 证据链缺一 → 拒绝；齐全 → 激活
        assert method.activate_pending_guidance(
            g3_behavior_verified=True, g4_support_passed=True,
            delayed_ok=False) is False
        assert method.activate_pending_guidance(
            g3_behavior_verified=True, g4_support_passed=True,
            delayed_ok=True) is True
        assert _guidance_body(method._active_snapshot()) == new_body


def test_guidance_clause_patch_rejects_wrong_surface_and_unknown_target() -> None:
    h0 = _h0()
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        # 错误 surface
        fake = _FakeSlow(_manifest(
            "bootstrap_skills.entries/inspect_and_localize.body",
            EditOperation.PATCH, _CLAUSE_VALUE, h0.harness_content_sha))
        method = TTHAMethod(object(), h0, ())
        ev = method.handle_group_guidance(
            {"workflow": "winsorize"}, {"n_episodes": 2},
            slow_agent=fake, controller=controller, store=store,
            card_builder=lambda g, c: _card(),
            confirmed_cause="WORKFLOW_GUIDANCE_GAP")
        assert ev["stage"] == "surface_mismatch_rejected", ev
        # unknown clause target
        bad_value = ("REPLACE_CLAUSE\n"
                     "target: propose.rule.not_a_clause\n"
                     "new_clause: something different")
        fake2 = _FakeSlow(_manifest(SURFACE, EditOperation.PATCH,
                                    bad_value, h0.harness_content_sha))
        method2 = TTHAMethod(object(), h0, ())
        ev2 = method2.handle_group_guidance(
            {"workflow": "winsorize"}, {"n_episodes": 2},
            slow_agent=fake2, controller=controller, store=store,
            card_builder=lambda g, c: _card(),
            confirmed_cause="WORKFLOW_GUIDANCE_GAP")
        assert ev2["stage"] == "clause_target_unknown", ev2


def test_guidance_clause_patch_rejects_malformed_and_empty() -> None:
    h0 = _h0()
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        # malformed payload → clause_payload_invalid
        fake = _FakeSlow(_manifest(SURFACE, EditOperation.PATCH,
                                   "not a clause payload",
                                   h0.harness_content_sha))
        method = TTHAMethod(object(), h0, ())
        ev = method.handle_group_guidance(
            {"workflow": "winsorize"}, {"n_episodes": 2},
            slow_agent=fake, controller=controller, store=store,
            card_builder=lambda g, c: _card(),
            confirmed_cause="WORKFLOW_GUIDANCE_GAP")
        assert ev["stage"] == "clause_payload_invalid", ev
        # 空载荷 → empty_patch_rejected
        fake2 = _FakeSlow(_manifest(SURFACE, EditOperation.PATCH, "   ",
                                    h0.harness_content_sha))
        method2 = TTHAMethod(object(), h0, ())
        ev2 = method2.handle_group_guidance(
            {"workflow": "winsorize"}, {"n_episodes": 2},
            slow_agent=fake2, controller=controller, store=store,
            card_builder=lambda g, c: _card(),
            confirmed_cause="WORKFLOW_GUIDANCE_GAP")
        assert ev2["stage"] == "empty_patch_rejected", ev2
        # no_manifest → 干净弃权
        method3 = TTHAMethod(object(), h0, ())
        ev3 = method3.handle_group_guidance(
            {"workflow": "winsorize"}, {"n_episodes": 2},
            slow_agent=_FakeSlow(None), controller=controller, store=store,
            card_builder=lambda g, c: _card(),
            confirmed_cause="WORKFLOW_GUIDANCE_GAP")
        assert ev3["stage"] == "no_manifest"
