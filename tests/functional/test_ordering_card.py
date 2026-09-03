"""E1 承重集成测试（2026-08-16 本地评审修订版）：Domain Ordering Card。

纵向切片——在 fork/active store 中生成一张 Target-local ordering card，编译、
**精确匹配 scope**、**重排同一候选池**、删除后恢复。零 LLM、零真实数据、**不修改 h0**。

被测的五条语义（对应评审要求）：
  1  卡由 Episode 计数**确定性**生成（同输入同输出，无 LLM）；
  2  卡编译进 snapshot 并被 Runtime 取到；
  3  卡**只重排 Fast 已供应的候选**——permutation 不变量：不增、不删、不替换；
  4  scope 不精确匹配（domain / downstream_model_class / task）→ **不生效**，
     因为 `resolve_harness_view` 不读 risk_guards，这一层必须由 Runtime 机械检查；
  5  卡**不供应候选**（body 无 `Frozen program steps:` marker）；删除后顺序恢复原状。

计数口径同时被测：合法但未探测 = UNKNOWN，排在已知算子之后，**不按零收益处理**。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha import ordering_card as oc  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway, extract_public_features)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor)

DATASET = "ordering_card_test"
CONSUMER = "ridge"          # 精确 consumer——**不**提前推广为 parametric_forecaster
FAMILY = "outlier"
OPS = ("winsorize", "outlier_mad", "hampel_filter")
CARD_ID = "ordering_outlier_forecast"
ORIGIN = 400


def _series() -> np.ndarray:
    t = np.arange(1024, dtype=np.float64)
    return np.sin(t / 7.0) + 0.1 * np.sin(t / 3.0) + 5.0


def _neutral_eval(roster, values, compiled, config, *, origin):
    """所有候选恒中性——本测试只看探测**顺序**，不看收益。"""
    return {"mean_smase": 1.0, "per_view_smase": [1.0], "behavior_point_count": 10}


def _scope(*, domain=DATASET, consumer=CONSUMER, task="forecast", family=FAMILY):
    return {"task": task, "domain": domain,
            "downstream_model_class": consumer, "program_family": family}


def _evidence_favouring_hampel():
    """hampel 最优、winsorize 次之、outlier_mad 有害；`denoise_median` 合法但从未探测。"""
    ev = oc.empty_evidence([*OPS, "denoise_median"])
    oc.accumulate(ev, [("hampel_filter", 0.50)] * 4)
    oc.accumulate(ev, [("winsorize", 0.20)] * 4)
    oc.accumulate(ev, [("outlier_mad", -0.30)] * 4)
    oc.accumulate(ev, [("denoise_median", None)] * 9)   # 合法机会，从未探测
    return ev


def _evidence_favouring_mad():
    """与基线顺序**相反**：outlier_mad 最优、winsorize 有害。

    基线（无卡）探测序是 winsorize → outlier_mad；用这份证据建卡后必须翻转，
    否则「卡改变了顺序」这条断言会空过（2026-08-16 诊断发现原证据与基线同序）。
    """
    ev = oc.empty_evidence([*OPS, "denoise_median"])
    oc.accumulate(ev, [("outlier_mad", 0.60)] * 4)
    oc.accumulate(ev, [("hampel_filter", 0.10)] * 4)
    oc.accumulate(ev, [("winsorize", -0.40)] * 4)
    oc.accumulate(ev, [("denoise_median", None)] * 9)
    return ev


def _install_card(store, snapshot, card_doc, *, edit_id):
    parent = store.materialize(snapshot)
    fork = store.fork(parent, edit_id=edit_id)
    learned = fork / "skills" / "learned"
    learned.mkdir(parents=True, exist_ok=True)
    (learned / ".gitkeep").unlink(missing_ok=True)
    (learned / f"{card_doc['skill_id']}.json").write_text(
        json.dumps(card_doc, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8")
    new_snap = compile_snapshot(fork, verify_lock=False)
    store.materialize(new_snap, parent_sha=snapshot.runtime_bundle_sha)
    store.set_active(new_snap.runtime_bundle_sha)
    return new_snap


def _remove_card(store, snapshot, skill_id, *, edit_id):
    parent = store.materialize(snapshot)
    fork = store.fork(parent, edit_id=edit_id)
    (fork / "skills" / "learned" / f"{skill_id}.json").unlink()
    new_snap = compile_snapshot(fork, verify_lock=False)
    store.materialize(new_snap, parent_sha=snapshot.runtime_bundle_sha)
    store.set_active(new_snap.runtime_bundle_sha)
    return new_snap


def _round(values, snapshot, *, scope_family=FAMILY, consumer=CONSUMER,
           domain=DATASET):
    series0 = values["s0"]
    backend = sealed.SealedProbeBackend(
        explore=True, operators=OPS, max_propose_candidates=len(OPS),
        force_pool=True)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), snapshot, ())
    executor = ScopeExecutor(
        [{"series_uid": "s0", "role": "train"},
         {"series_uid": "s0", "role": "eval"}],
        values, {"anchors": []}, evaluate_fn=_neutral_eval)
    request = runner._a5_request(series0, values, ORIGIN, DATASET)
    if consumer != request.task_spec.downstream_model_class:
        # TaskSpec/PreparationRequest 均为 frozen dataclass，且 task_context.task_spec
        # 必须与 task_spec 相等——三处一起 replace。
        _ts = dataclasses.replace(request.task_spec,
                                  downstream_model_class=consumer)
        request = dataclasses.replace(
            request, task_spec=_ts,
            task_context=dataclasses.replace(request.task_context,
                                             task_spec=_ts))
    feats = dict(extract_public_features(series0[:ORIGIN], task_kind="forecast"))
    return run_online_round(
        method, executor, request, values, origin=ORIGIN,
        slow_agent=None, controller=None, store=None,
        card_builder=runner._a5v2_card,
        round_name="ordering_card_test", budget=len(OPS), allow_slow=False,
        domain=domain, period=24, fast_features=feats,
        ordering_program_family=scope_family)


def test_card_generation_is_deterministic_and_respects_unknown():
    ev = _evidence_favouring_hampel()
    a = oc.build_ordering_card(skill_id=CARD_ID, scope=_scope(), evidence=ev,
                               lam=1.0, tie_break=[*OPS, "denoise_median"])
    b = oc.build_ordering_card(skill_id=CARD_ID, scope=_scope(),
                               evidence=_evidence_favouring_hampel(),
                               lam=1.0, tie_break=[*OPS, "denoise_median"])
    assert a == b, "卡必须由计数确定性生成（同输入同输出）"
    order = a["risk_guards"]["order"]
    assert order[:3] == ["hampel_filter", "winsorize", "outlier_mad"], order
    # 合法但从未探测 → UNKNOWN → 排在已知算子之后，且不被当成 0 收益抬到
    # outlier_mad(E_gain=-0.30) 之前
    assert order[-1] == "denoise_median", order
    row = a["risk_guards"]["evidence"]["denoise_median"]
    assert row["legal_opportunities"] == 9 and row["evaluated_attempts"] == 0
    # 卡不供应候选
    assert "Frozen program steps:" not in a["body"]
    assert a["risk_guards"]["authority"] == {
        "reorders_supplied_candidates": True, "supplies_candidates": False,
        "suppresses_operators": False, "grants_execution": False}


def test_ordering_card_reorders_pool_and_restores_after_removal(tmp_path):
    values = {"s0": _series()}
    store = SnapshotStore(tmp_path / "store")
    h0 = runner._h0_snapshot()

    # --- 基线：无卡 ---
    base = _round(values, h0)
    assert base.ordering_card_id is None
    assert base.probe_order_before_card == base.probe_order_after_card
    baseline_order = list(base.probe_order_after_card)
    assert len(baseline_order) >= 2, f"候选池太小，无法观测重排: {baseline_order}"

    # --- 装卡（不改 h0：只在 fork 上编译）---
    card = oc.build_ordering_card(
        skill_id=CARD_ID, scope=_scope(), evidence=_evidence_favouring_mad(),
        lam=1.0, tie_break=[*OPS, "denoise_median"])
    assert card["risk_guards"]["order"][0] == "outlier_mad"
    snap1 = _install_card(store, h0, card, edit_id="install_ordering_card")
    assert CARD_ID in [s.skill_id for s in snap1.skills], "卡未进 snapshot"
    assert CARD_ID not in [s.skill_id for s in h0.skills], "h0 不得被修改"

    r1 = _round(values, snap1)
    assert r1.ordering_card_id == CARD_ID, "卡未被 Runtime 取到/匹配"
    # 非空断言：顺序必须**真的**被改变，否则本测试形同虚设
    assert r1.probe_order_before_card != r1.probe_order_after_card, (
        f"卡未改变顺序（测试空过）: {r1.probe_order_before_card}")
    # permutation 不变量：不增、不删、不替换
    assert sorted(r1.probe_order_after_card) == sorted(r1.probe_order_before_card)
    # 卡不供应候选
    assert not any(str(c).startswith("cand_skill_")
                   for c in r1.probe_order_after_card), r1.probe_order_after_card
    # 顺序确实按卡改变了，且 hampel 在 outlier_mad 之前
    steps = dict((r1._method.last_trace.candidate_program_steps or {}))  # noqa: SLF001
    ops_after = [oc._leading_op(steps.get(c)) for c in r1.probe_order_after_card]
    known = [o for o in ops_after if o in card["risk_guards"]["order"]]
    ranks = [card["risk_guards"]["order"].index(o) for o in known]
    assert ranks == sorted(ranks), f"未按卡的 order 排列: {ops_after}"
    assert known[0] == "outlier_mad", f"卡的首选未生效: {ops_after}"

    # --- 删卡 → 恢复原状 ---
    snap2 = _remove_card(store, snap1, CARD_ID, edit_id="remove_ordering_card")
    assert CARD_ID not in [s.skill_id for s in snap2.skills]
    r2 = _round(values, snap2)
    assert r2.ordering_card_id is None
    assert r2.probe_order_after_card == baseline_order, "删卡后未恢复原顺序"


def test_card_does_not_apply_outside_its_exact_scope(tmp_path):
    """retrieval 不读 risk_guards——scope 必须由 Runtime 机械精确匹配。"""
    values = {"s0": _series()}
    store = SnapshotStore(tmp_path / "store")
    h0 = runner._h0_snapshot()
    card = oc.build_ordering_card(
        skill_id=CARD_ID, scope=_scope(), evidence=_evidence_favouring_hampel(),
        lam=1.0, tie_break=[*OPS, "denoise_median"])
    snap = _install_card(store, h0, card, edit_id="install_for_scope_test")

    # 同 scope → 生效（正控）
    assert _round(values, snap).ordering_card_id == CARD_ID

    # 三个维度各错一个 → 一律不生效
    assert _round(values, snap, domain="other_domain").ordering_card_id is None
    assert _round(values, snap, consumer="lstm_reporter").ordering_card_id is None
    assert _round(values, snap, scope_family="impute").ordering_card_id is None


def test_reorder_is_pure_permutation_on_unknown_operators():
    card = oc.build_ordering_card(
        skill_id=CARD_ID, scope=_scope(), evidence=_evidence_favouring_hampel(),
        lam=1.0, tie_break=[*OPS, "denoise_median"])
    steps_map = {
        "a": [{"op": "outlier_mad", "params": {}}],
        "b": [{"op": "hampel_filter", "params": {}}],
        "c": [{"op": "impute_linear", "params": {}}],   # 卡里没有的算子
        "d": "not-parseable",
    }
    out = oc.reorder_probe_order(["a", "b", "c", "d"], steps_map, card)
    assert sorted(out) == ["a", "b", "c", "d"]
    assert out.index("b") < out.index("a"), out          # hampel 优先于 mad
    assert out[-2:] == ["c", "d"], out                   # 未知算子保持原相对次序、殿后


# ===================== E2（2026-08-16）prequential 卡生命周期 =====================
# 新增的承重语义（E1 之外）：
#   6  Source / Target 计数**分开存**，Source 只以 pseudo-count 入场——
#      Target 反馈积累到一定量后**必须能翻转** Source 先验（评审第 7 条）；
#   7  一条 prequential 流上卡能**连续重建**：revision 单调递增，且新 revision
#      被 Runtime 取到、真的改变了探测顺序；
#   8  **同候选供给**：装卡与不装卡，Fast 供应的候选集合完全相同——
#      卡只重排，不改 Program Supply。

def test_source_and_target_counts_stay_separate_and_target_can_override():
    source = oc.empty_evidence([*OPS])
    # Source 大样本先验：winsorize 好、outlier_mad 差
    oc.accumulate(source, [("winsorize", 0.50)] * 100)
    oc.accumulate(source, [("outlier_mad", -0.20)] * 100)
    oc.accumulate(source, [("hampel_filter", 0.10)] * 100)
    target = oc.empty_evidence([*OPS])

    merged0 = oc.merge_evidence(source, target, source_pseudo_count=10)
    assert oc.rank_operators(merged0, lam=1.0, tie_break=OPS)[0] == "winsorize"
    # Source 的 100 条样本被压成 10 条伪观测——样本量不得传染
    assert merged0["winsorize"]["evaluated_attempts"] == 10

    # Target 反馈与 Source 相反
    oc.accumulate(target, [("outlier_mad", 0.60)] * 20)
    oc.accumulate(target, [("winsorize", -0.40)] * 20)
    merged1 = oc.merge_evidence(source, target, source_pseudo_count=10)
    assert oc.rank_operators(merged1, lam=1.0, tie_break=OPS)[0] == "outlier_mad", (
        "Target 反馈积累后必须能翻转 Source 先验——否则 Source 样本量压住 Target")

    card = oc.build_ordering_card(
        skill_id=CARD_ID, scope=_scope(), evidence=merged1, lam=1.0,
        tie_break=OPS, revision=3,
        evidence_blocks={"source": source, "target": target})
    blocks = card["risk_guards"]["evidence_blocks"]
    # 两块原始计数分别保留、永不合池
    assert blocks["source"]["winsorize"]["evaluated_attempts"] == 100
    assert blocks["target"]["winsorize"]["evaluated_attempts"] == 20
    assert card["revision"] == 3


def test_prequential_rebuild_changes_order_midstream_and_keeps_supply_identical(
        tmp_path):
    values = {"s0": _series()}
    store = SnapshotStore(tmp_path / "store")
    h0 = runner._h0_snapshot()

    # --- 无卡基线：记录 Fast 的候选供给 ---
    base = _round(values, h0)
    supply = sorted(base.probe_order_after_card)
    assert len(supply) >= 2

    source = oc.empty_evidence([*OPS])
    oc.accumulate(source, [("winsorize", 0.50)] * 60)
    oc.accumulate(source, [("outlier_mad", -0.20)] * 60)
    target = oc.empty_evidence([*OPS])

    orders: list[list[str]] = []
    snapshot = h0
    seen_first_ops: list[str] = []
    for revision in (1, 2):
        merged = oc.merge_evidence(source, target, source_pseudo_count=10)
        card = oc.build_ordering_card(
            skill_id=CARD_ID, scope=_scope(), evidence=merged, lam=1.0,
            tie_break=OPS, revision=revision,
            evidence_blocks={"source": source, "target": target})
        orders.append(list(card["risk_guards"]["order"]))
        snapshot = _install_card(store, snapshot, card,
                                 edit_id=f"prequential_rev{revision}")
        installed = [s for s in snapshot.skills if s.skill_id == CARD_ID]
        assert installed and installed[0].revision == revision, (
            "revision 必须单调递增地进 snapshot")

        r = _round(values, snapshot)
        assert r.ordering_card_id == CARD_ID
        # 8. 同候选供给——装卡不改 Program Supply
        assert sorted(r.probe_order_after_card) == supply, (
            f"卡改变了候选供给: {sorted(r.probe_order_after_card)} != {supply}")
        steps = dict(r._method.last_trace.candidate_program_steps or {})  # noqa: SLF001
        ops_now = [oc._leading_op(steps.get(c))
                   for c in r.probe_order_after_card]
        seen_first_ops.append(ops_now[0])

        # 批结束：折入与 Source 相反的 Target 反馈，下一 revision 重建
        oc.accumulate(target, [("outlier_mad", 0.60)] * 12)
        oc.accumulate(target, [("winsorize", -0.40)] * 12)

    assert orders[0] != orders[1], "prequential 重建后 order 必须变化"
    assert seen_first_ops[0] != seen_first_ops[1], (
        f"新 revision 未改变实际探测顺序: {seen_first_ops}")


def test_ordering_card_cannot_also_carry_a_frozen_program():
    """最小权限守卫（本地评审 2026-08-16 第 2 条）。

    Ordering Card 与 Executable Program Skill 共用 `skill-entry/1` 载体和
    `skills/learned/` 目录，唯一的权限分界是 body 里有没有
    `Frozen program steps:` marker。一张同时声明 ordering 身份又带 marker
    的卡会**同时**拿到「重排」和「供应候选」两种权限，
    后续「顺序改进 vs 程序改进」的归因就被污染了。
    守卫落在**加载期**（`load_learned_skill_entry`），不靠构造方自觉。
    """
    from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: PLC0415
        load_learned_skill_entry)

    good = oc.build_ordering_card(
        skill_id=CARD_ID, scope=_scope(), evidence=_evidence_favouring_mad(),
        lam=1.0, tie_break=[*OPS, "denoise_median"])
    load_learned_skill_entry(good)          # 正常卡照常加载

    smuggled = dict(good)
    smuggled["body"] = good["body"] + (
        ' Frozen program steps: [{"op": "outlier_mad", "params": {}}]')
    try:
        load_learned_skill_entry(smuggled)
    except ValueError as exc:
        assert "frozen program" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("同时带 ordering 身份与 frozen program 的卡必须被拒绝")

    # 守卫只针对 ordering 卡——普通 Executable Program Skill 不受影响
    program_skill = dict(good)
    program_skill["body"] = (
        'Frozen program steps: [{"op": "outlier_mad", "params": {}}]')
    program_skill["risk_guards"] = {"requires_target_support": True}
    load_learned_skill_entry(program_skill)
