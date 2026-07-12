"""tests/test_p6_split.py — P6 sequential split manifest 单测 + 真实文件集成测。

单元测试全部用合成 fixture（tmp_path / 内存），不依赖任何真实文件；
唯一的集成测试读真实 exposure_ledger / meta / U 探针，缺失则 skip。
红线：任何写盘只发生在 tmp_path，绝不触碰 results/。
"""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.p6.split_manifest import (
    PRECOMMIT_REQUIRED_KEYS,
    P6ManifestError,
    P6SplitManifest,
    P6StateError,
    SequentialGate,
    TERMINAL_VERDICTS,
    build_manifest,
    compute_event_sha,
    compute_manifest_sha,
    ledger_path,
    load_manifest,
    read_jsonl,
    select_u_items,
    select_virgin_items,
    validate_manifest,
    write_manifest,
)

BIG = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths")
SINGLE = ("us_births", "saugeenday", "sunspot")
U_EXCLUDED = [f"T{i}" for i in range(1, 25)]  # 合成的 24 条探针排除清单


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(config, item, exposure="confirmed_exposed"):
    return {
        "config": config,
        "item_id": item,
        "series_uid": f"{config}:{item}",
        "exposure_class": exposure,
        "evidence": ["synthetic"],
        "confidence": "high",
    }


def make_ledger():
    rows = [_row(c, f"T{i}") for c in BIG for i in range(1, 21)]
    rows += [_row(c, "T1") for c in SINGLE]
    return rows


@pytest.fixture()
def manifest():
    return build_manifest(make_ledger(), U_EXCLUDED)


# ---------------------------------------------------------------- a. 确定性

def test_determinism_same_sha():
    m1 = build_manifest(make_ledger(), U_EXCLUDED)
    rows = make_ledger()
    rows.reverse()  # 行序无关
    m2 = build_manifest(rows, list(reversed(U_EXCLUDED)))
    assert m1.manifest_sha == m2.manifest_sha
    assert m1.payload == m2.payload
    assert len(m1.manifest_sha) == 64 and all(c in "0123456789abcdef" for c in m1.manifest_sha)


def test_sha_ignores_volatile_fields(manifest):
    p = dict(manifest.payload)
    p["created_at"] = "2026-07-10T00:00:00Z"
    p["timestamp"] = 123456
    assert compute_manifest_sha(p) == manifest.manifest_sha
    # 非易变字段变动必须改变 sha
    p2 = copy.deepcopy(manifest.payload)
    p2["counts"]["legacy_total"] = 999
    assert compute_manifest_sha(p2) != manifest.manifest_sha


# ------------------------------------------------------------ b. 分块性质

def test_block_partition_and_quota(manifest):
    blocks = manifest.blocks
    c0, d1, d2 = blocks["C0"], blocks["D1"], blocks["D2"]
    assert (len(c0), len(d1), len(d2)) == (16, 32, 32)
    s0, s1, s2 = set(c0), set(d1), set(d2)
    assert not (s0 & s1) and not (s0 & s2) and not (s1 & s2)
    all_big = {f"{c}:T{i}" for c in BIG for i in range(1, 21)}
    assert (s0 | s1 | s2) == all_big and len(s0 | s1 | s2) == 80

    legacy = manifest.legacy_rows
    for cfg in BIG:
        assert sum(1 for u in c0 if legacy[u]["config"] == cfg) == 4
        assert sum(1 for u in d1 if legacy[u]["config"] == cfg) == 8
        assert sum(1 for u in d2 if legacy[u]["config"] == cfg) == 8

    # 手工重算一个域的 sha256 升序，核对 4/8/8 切片
    cfg = "nn5_daily"
    uids = [f"{cfg}:T{i}" for i in range(1, 21)]
    order = sorted(uids, key=lambda u: (_hex(f"p6|{cfg}|{u}"), u))
    assert [u for u in c0 if u.startswith(f"{cfg}:")] == order[:4]
    assert [u for u in d1 if u.startswith(f"{cfg}:")] == order[4:12]
    assert [u for u in d2 if u.startswith(f"{cfg}:")] == order[12:20]

    # 单条域隔离块
    assert blocks["C0_qualitative"] == sorted(f"{c}:T1" for c in SINGLE)


# ------------------------------------- c. 单条域隔离与放置规则违规 raise

def test_placement_violations_raise():
    # virgin 类 exposure_class 混入 legacy 账本
    rows = make_ledger()
    rows[0]["exposure_class"] = "virgin"
    with pytest.raises(P6ManifestError):
        build_manifest(rows, U_EXCLUDED)

    # uncertain_legacy_exposure 属于允许集（正例）
    rows = make_ledger()
    rows[5]["exposure_class"] = "uncertain_legacy_exposure"
    build_manifest(rows, U_EXCLUDED)

    # 未知 config
    with pytest.raises(P6ManifestError):
        build_manifest(make_ledger() + [_row("mystery_domain", "T1")], U_EXCLUDED)

    # series_uid 重复
    rows = make_ledger()
    rows.append(dict(rows[0]))
    with pytest.raises(P6ManifestError):
        build_manifest(rows, U_EXCLUDED)

    # 大域行数 != 20（配额 4+8+8 无法恰好用尽）
    rows = [r for r in make_ledger() if r["series_uid"] != "nn5_daily:T20"]
    with pytest.raises(P6ManifestError):
        build_manifest(rows, U_EXCLUDED)

    # 单条域出现 2 条
    with pytest.raises(P6ManifestError):
        build_manifest(make_ledger() + [_row("sunspot", "T2")], U_EXCLUDED)

    # 缺字段
    rows = make_ledger()
    del rows[3]["exposure_class"]
    with pytest.raises(P6ManifestError):
        build_manifest(rows, U_EXCLUDED)

    # legacy 大域被指定为 U 域（legacy 进 U 规格）
    with pytest.raises(P6ManifestError):
        build_manifest(make_ledger(), U_EXCLUDED, u_config="nn5_daily")

    # 单条域被指定为 U 域（单条域不得入 U）
    with pytest.raises(P6ManifestError):
        build_manifest(make_ledger(), U_EXCLUDED, u_config="sunspot")


def test_tampered_manifest_raises(manifest):
    def tampered():
        return copy.deepcopy(manifest.payload)

    # legacy item 从 V1 排除集中被移除 → legacy 可能进 V → raise
    p = tampered()
    p["virgin_specs"]["V1"]["exclusions"]["nn5_daily"].remove("T1")
    with pytest.raises(P6ManifestError):
        validate_manifest(P6SplitManifest(p))

    # 单条域 uid 被塞进 D1
    p = tampered()
    p["blocks"]["C0_qualitative"].remove("sunspot:T1")
    p["blocks"]["D1"][0] = "sunspot:T1"
    with pytest.raises(P6ManifestError):
        validate_manifest(P6SplitManifest(p))

    # C0/D1 相交（uid 重复放置）
    p = tampered()
    p["blocks"]["D1"][0] = p["blocks"]["C0"][0]
    with pytest.raises(P6ManifestError):
        validate_manifest(P6SplitManifest(p))

    # V2 丢失 "排除 V1 已选" 的显式声明
    p = tampered()
    p["virgin_specs"]["V2"]["additional_exclusions"] = []
    with pytest.raises(P6ManifestError):
        validate_manifest(P6SplitManifest(p))

    # 某 legacy 行的 exposure_class 被改成 virgin
    p = tampered()
    p["legacy_rows"]["fred_md:T1"]["exposure_class"] = "virgin"
    with pytest.raises(P6ManifestError):
        validate_manifest(P6SplitManifest(p))

    # virgin 规格被物化
    p = tampered()
    p["virgin_specs"]["U"]["materialized"] = True
    with pytest.raises(P6ManifestError):
        validate_manifest(P6SplitManifest(p))

    # 未篡改的 payload 原样通过（对照）
    validate_manifest(P6SplitManifest(tampered()))


# ---------------------------------------------------------------- d. 状态机 v2

C1 = {
    "candidate_edit_sha": "edit-c1",
    "harness_state_sha": "state-h0",
    "config_digest": "cfg-c1",
    "materialization_sha": "mat-2026",
    "code_sha": "code-abc",
}
C2 = {
    "candidate_edit_sha": "edit-c2",
    "harness_state_sha": "state-h1",
    "config_digest": "cfg-c2",
    "materialization_sha": "mat-2026",
    "code_sha": "code-abc",
}


def _events(manifest, root):
    return read_jsonl(ledger_path(manifest, root))


def test_canonical_ledger_path_enforced(manifest, tmp_path):
    canon = ledger_path(manifest, tmp_path)
    assert canon.parent == tmp_path
    assert canon.name == f"p6_gate_ledger_{manifest.manifest_sha}.jsonl"
    # 任意 log_path 的第二本台账被消灭
    with pytest.raises(P6StateError):
        SequentialGate(manifest, tmp_path / "events.jsonl")
    # 纯内存模式已废除
    with pytest.raises(P6StateError):
        SequentialGate(manifest, None)
    g = SequentialGate(manifest, tmp_path)  # 传目录 → 派生 canonical
    assert g.log_path == canon
    g.close()
    g2 = SequentialGate(manifest, canon)  # 显式 canonical 文件路径亦可
    assert g2.log_path == canon
    g2.close()


def test_ledger_lock_mutual_exclusion(manifest, tmp_path):
    g1 = SequentialGate(manifest, tmp_path)
    try:
        with pytest.raises(P6StateError):
            SequentialGate(manifest, tmp_path)  # 第二实例持锁 → raise（真互斥）
    finally:
        g1.close()
    g2 = SequentialGate(manifest, tmp_path)  # 锁释放后可再开
    g2.record_precommit(1, C1)
    g2.close()
    with pytest.raises(P6StateError):
        g2.record_precommit(2, C2)  # close 后禁写
    assert g2.state("V1") == "sealed"  # 查询仍可用


def test_precommit_payload_whitelist_and_once(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        for k in PRECOMMIT_REQUIRED_KEYS:
            bad = dict(C1)
            del bad[k]
            with pytest.raises(P6StateError):
                g.record_precommit(1, bad)  # 缺键 raise
            bad = dict(C1)
            bad[k] = "   "
            with pytest.raises(P6StateError):
                g.record_precommit(1, bad)  # 空白值同缺键
        with pytest.raises(P6StateError):
            g.record_precommit(1, "not-a-mapping")
        for cyc in (0, 3, True, "1"):
            with pytest.raises(P6StateError):
                g.record_precommit(cyc, C1)
        with pytest.raises(P6StateError):
            g.record_precommit(2, C2)  # cycle2 precommit 需 cycle1 terminal（严格顺序）
        sha1 = g.record_precommit(1, C1)
        pc = g.precommit(1)
        assert pc["event_sha"] == sha1 and pc["payload"] == C1
        with pytest.raises(P6StateError):
            g.record_precommit(1, C2)  # 一次性：禁止重新选择候选（不同五元组）


def test_f4_record_precommit_idempotent_on_identical_reentry(manifest, tmp_path):
    """F4/finding 34：pending precommit + 五元组逐字段一致 → 幂等返回既有 sha、不追加事件；
    五元组不一致 → 仍拒绝；已 terminal → 仍拒绝。"""
    with SequentialGate(manifest, tmp_path) as g:
        sha1 = g.record_precommit(1, C1)
        seq_after = g.seq
        tip_after = g.chain_tip
        # 逐字段一致的重入 → 幂等：返回同一 sha、seq/tip 不变（无新事件）
        sha_again = g.record_precommit(1, dict(C1))
        assert sha_again == sha1
        assert g.seq == seq_after and g.chain_tip == tip_after
        # 不一致 → 拒绝
        with pytest.raises(P6StateError):
            g.record_precommit(1, C2)
        # ledger 中只有一条 precommit 事件（无分叉）
        evs = _events(manifest, tmp_path)
        assert [e["event"] for e in evs].count("precommit") == 1
        # terminal 后即便同五元组也拒绝（不再 pending）
        g.open_block("V1")
        g.record_verdict("V1", "reject")
        g.record_cycle_terminal(1, "reject")
        with pytest.raises(P6StateError):
            g.record_precommit(1, dict(C1))


def test_precommit_must_precede_open(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        assert not g.can_open("V1")
        with pytest.raises(P6StateError):
            g.open_block("V1")  # precommit 必须先于 open
        g.record_precommit(1, C1)
        assert g.can_open("V1") and not g.can_open("V2") and not g.can_open("U")
        with pytest.raises(P6StateError):
            g.open_block("V1", bindings={"x": "y"})  # V 块绑定只能来自 precommit
        g.open_block("V1")
        pend = g.pending_open("V1")
        assert pend["precommit_event_sha"] == g.precommit(1)["event_sha"]
        assert pend["bindings"] == C1  # open 事件绑定 = precommit 五元组
        with pytest.raises(P6StateError):
            g.open_block("V1")  # open 一次性


def test_full_two_cycle_lifecycle_replay_and_chain(manifest, tmp_path):
    g = SequentialGate(manifest, tmp_path)
    g.record_precommit(1, C1)
    g.open_block("V1")
    g.record_verdict("V1", "promote")
    assert g.pending_open("V1") is None  # verdict 后不再 pending
    g.record_cycle_terminal(1, "promote")
    assert g.cycle_terminal(1) == "promote"
    g.record_precommit(2, C2)
    g.open_block("V2")
    g.record_verdict("V2", "reject", result_digest="sha256:deadbeef")
    g.record_cycle_terminal(2, "reject")
    assert g.can_open("U")  # reject 也是 terminal → U 解锁
    g.open_block("U", bindings={"materialization_sha": "mat-2026"})
    g.close()

    events = _events(manifest, tmp_path)
    assert [e["event"] for e in events] == [
        "precommit", "open", "verdict", "cycle_terminal",
        "precommit", "open", "verdict", "cycle_terminal", "open",
    ]
    assert [e["seq"] for e in events] == list(range(1, 10))
    assert all(e["manifest_sha"] == manifest.manifest_sha for e in events)
    # hash 链：genesis 派生自 manifest_sha，逐事件 prev/sha 闭合
    prev = hashlib.sha256(
        f"p6-gate-genesis|{manifest.manifest_sha}".encode("utf-8")
    ).hexdigest()
    for e in events:
        assert e["prev_event_sha"] == prev
        assert compute_event_sha(e) == e["event_sha"]
        prev = e["event_sha"]
    # open/verdict/terminal 均链回 precommit 且携带绑定
    assert events[1]["precommit_event_sha"] == events[0]["event_sha"]
    assert events[1]["bindings"] == C1
    assert events[2]["precommit_event_sha"] == events[0]["event_sha"]
    assert events[3]["precommit_event_sha"] == events[0]["event_sha"]
    assert events[3]["v_block"] == "V1" and events[3]["v_block_state"] == "verdict_recorded"
    assert events[5]["bindings"] == C2
    assert events[6]["result_digest"] == "sha256:deadbeef"
    assert events[8]["bindings"] == {"materialization_sha": "mat-2026"}

    # 重放重建：状态/判决/终局一致；U open 为 pending，可 resume 记 verdict
    g2 = SequentialGate(manifest, tmp_path)
    assert g2.state("V1") == "verdict_recorded" and g2.verdict("V1") == "promote"
    assert g2.state("V2") == "verdict_recorded" and g2.verdict("V2") == "reject"
    assert g2.state("U") == "open" and g2.pending_open("U") is not None
    assert g2.cycle_terminal(1) == "promote" and g2.cycle_terminal(2) == "reject"
    assert g2.seq == 9 and g2.chain_tip == events[-1]["event_sha"]
    with pytest.raises(P6StateError):
        g2.open_block("U")
    g2.record_verdict("U", "promote", result_digest="sha256:u-final")
    g2.close()


def test_cycle_terminal_whitelist_once_irreversible(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        assert sorted(TERMINAL_VERDICTS) == ["abstain", "promote", "reject"]
        for bad in ("PASS", "Promote", "", "  ", None, 1, b"promote"):
            with pytest.raises(P6StateError):
                g.record_cycle_terminal(1, bad)  # 白名单外一律 raise
        for cyc in (0, 3, True, "1"):
            with pytest.raises(P6StateError):
                g.record_cycle_terminal(cyc, "abstain")
        with pytest.raises(P6StateError):
            g.record_cycle_terminal(2, "abstain")  # cycle 顺序：先 1 后 2
        with pytest.raises(P6StateError):
            g.record_cycle_terminal(1, "promote")  # V1 sealed → 只能 abstain
        g.record_precommit(1, C1)
        g.open_block("V1")
        with pytest.raises(P6StateError):
            g.record_cycle_terminal(1, "promote")  # 开启未判决（pending）禁 terminal
        g.record_verdict("V1", "promote")
        with pytest.raises(P6StateError):
            g.record_cycle_terminal(1, "reject")  # terminal 必须与 V verdict 一致
        g.record_cycle_terminal(1, "promote")
        with pytest.raises(P6StateError):
            g.record_cycle_terminal(1, "promote")  # 每 cycle 只能一次（不可逆）
        with pytest.raises(P6StateError):
            g.record_precommit(1, C2)  # cycle 已 closed：不得再 precommit
        with pytest.raises(P6StateError):
            g.open_block("V1")  # terminal 后 V 永久封存


def test_v2_requires_cycle1_terminal_and_own_precommit(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        g.record_precommit(1, C1)
        with pytest.raises(P6StateError):
            g.open_block("V2")  # 无 cycle1 terminal
        g.open_block("V1")
        g.record_verdict("V1", "reject")
        with pytest.raises(P6StateError):
            g.record_precommit(2, C2)  # terminal 未记，cycle2 不得 precommit
        g.record_cycle_terminal(1, "reject")
        with pytest.raises(P6StateError):
            g.open_block("V2")  # 有 terminal 但缺 cycle2 precommit
        assert not g.can_open("V2")
        g.record_precommit(2, C2)
        assert g.can_open("V2")
        g.open_block("V2")
        assert g.pending_open("V2")["bindings"] == C2


def test_u_requires_cycle2_terminal_abstain_path(manifest, tmp_path):
    # abstain→U 死锁修复：V1/V2 双 abstain（永不 open）仍能解锁 U。
    with SequentialGate(manifest, tmp_path) as g:
        assert not g.can_open("U")
        with pytest.raises(P6StateError):
            g.open_block("U")
        g.record_cycle_terminal(1, "abstain")  # V1 保持 sealed，terminal 照记
        assert g.state("V1") == "sealed" and g.cycle_terminal(1) == "abstain"
        assert not g.can_open("U")  # cycle2 terminal 之前 U 仍锁
        with pytest.raises(P6StateError):
            g.open_block("U")
        g.record_cycle_terminal(2, "abstain")
        assert g.state("V2") == "sealed"
        assert g.can_open("U")  # 不要求 V2 verdict——abstain 也是 terminal
        g.open_block("U")
        assert g.state("U") == "open"
        # abstain 后 V 永久封存：不得 open/verdict/precommit
        with pytest.raises(P6StateError):
            g.record_verdict("V2", "promote")
        with pytest.raises(P6StateError):
            g.record_precommit(1, C1)
        with pytest.raises(P6StateError):
            g.open_block("V1")

    # 变体：precommit 后未开箱亦可 abstain（V 未消费）；terminal 链回该 precommit
    sub = tmp_path / "b"
    sub.mkdir()
    with SequentialGate(manifest, sub) as g2:
        g2.record_precommit(1, C1)
        g2.record_cycle_terminal(1, "abstain")
        assert g2.state("V1") == "sealed" and g2.cycle_terminal(1) == "abstain"
    events = _events(manifest, sub)
    assert events[-1]["precommit_event_sha"] == events[0]["event_sha"]
    assert events[-1]["v_block_state"] == "sealed"


def test_block_verdict_whitelist(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        g.record_precommit(1, C1)
        g.open_block("V1")
        for bad in ("PASS", "FAIL", "", "  ", None):
            with pytest.raises(P6StateError):
                g.record_verdict("V1", bad)
        with pytest.raises(P6StateError):
            g.record_verdict("V1", "abstain")  # abstain 只属于 cycle terminal
        with pytest.raises(P6StateError):
            g.record_verdict("V1", "promote", result_digest="  ")
        g.record_verdict("V1", "promote")
        with pytest.raises(P6StateError):
            g.record_verdict("V1", "reject")  # verdict 一次性


def test_wal_crash_resume_pending_open(manifest, tmp_path):
    # 崩溃点：precommit + open intent 已 durable 落盘、verdict 未记，进程死亡。
    # close() 不写任何字节 → 文件状态与 kill 等价。
    g = SequentialGate(manifest, tmp_path)
    g.record_precommit(1, C1)
    g.open_block("V1")
    lines = ledger_path(manifest, tmp_path).read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["event"] == "open"  # WAL：intent 已先于一切在盘上
    g.close()

    g2 = SequentialGate(manifest, tmp_path)  # 重放恢复
    assert g2.state("V1") == "open"
    pend = g2.pending_open("V1")
    assert pend is not None and pend["bindings"] == C1
    assert pend["precommit_event_sha"] == g2.precommit(1)["event_sha"]
    with pytest.raises(P6StateError):
        g2.open_block("V1")  # 不得重开
    with pytest.raises(P6StateError):
        g2.record_precommit(1, C2)  # 不得重新选择候选
    with pytest.raises(P6StateError):
        g2.record_cycle_terminal(1, "promote")  # pending 不得直接 terminal
    g2.record_verdict("V1", "promote")  # 唯一合法路径：按同一 precommit resume
    g2.record_cycle_terminal(1, "promote")
    g2.close()


def test_hash_chain_tamper_detection(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        g.record_precommit(1, C1)
        g.open_block("V1")
        g.record_verdict("V1", "promote")
        g.record_cycle_terminal(1, "promote")
    path = ledger_path(manifest, tmp_path)
    pristine = path.read_text(encoding="utf-8")

    def _rewrite(lines):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ① 篡改中间事件字段（verdict promote→reject）→ event_sha 校验失败
    lines = pristine.splitlines()
    evt = json.loads(lines[2])
    evt["verdict"] = "reject"
    lines[2] = json.dumps(evt, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    _rewrite(lines)
    with pytest.raises(P6StateError):
        SequentialGate(manifest, tmp_path)

    # ② 篡改 + 重算自身 sha → 下一事件 prev_event_sha 断链
    evt["event_sha"] = compute_event_sha(evt)
    lines[2] = json.dumps(evt, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    _rewrite(lines)
    with pytest.raises(P6StateError):
        SequentialGate(manifest, tmp_path)

    # ③ 抽掉中间事件 → seq/链断
    lines = pristine.splitlines()
    del lines[1]
    _rewrite(lines)
    with pytest.raises(P6StateError):
        SequentialGate(manifest, tmp_path)

    # ④ 尾部撕裂写（半行 JSON）= 日志损坏 → technical abort 语义
    path.write_text(pristine + '{"seq": 5, "event": "open', encoding="utf-8")
    with pytest.raises(P6StateError):
        SequentialGate(manifest, tmp_path)

    # ⑤ 原样恢复 → 通过（对照）
    path.write_text(pristine, encoding="utf-8")
    SequentialGate(manifest, tmp_path).close()


def test_ledger_bound_to_manifest(manifest, tmp_path):
    rows = make_ledger()
    rows[0]["exposure_class"] = "uncertain_legacy_exposure"
    other = build_manifest(rows, U_EXCLUDED)
    assert other.manifest_sha != manifest.manifest_sha
    with SequentialGate(manifest, tmp_path) as g:
        g.record_precommit(1, C1)
    # 把 manifest A 的台账字节改名成 B 的 canonical 文件 → 重放必 raise
    src = ledger_path(manifest, tmp_path)
    dst = ledger_path(other, tmp_path)
    dst.write_bytes(src.read_bytes())
    with pytest.raises(P6StateError):
        SequentialGate(other, tmp_path)


def test_gate_misc_queries(manifest, tmp_path):
    with SequentialGate(manifest, tmp_path) as g:
        assert not g.can_open("C0") and not g.can_open("nope")  # 数据块/未知块恒 False
        with pytest.raises(P6StateError):
            g.state("C0")
        with pytest.raises(P6StateError):
            g.verdict("nope")
        with pytest.raises(P6StateError):
            g.pending_open("D1")
        with pytest.raises(P6StateError):
            g.record_verdict("V9", "promote")  # 未知块
        with pytest.raises(P6StateError):
            g.open_block("D2")
        assert g.verdict("V1") is None and g.precommit(1) is None
        assert g.cycle_terminal(2) is None and g.pending_open("U") is None


# ------------------------------------------------------------ e. virgin 规则

def test_virgin_specs_frozen_rules(manifest):
    specs = manifest.virgin_specs
    v1, v2, u = specs["V1"], specs["V2"], specs["U"]
    legacy_items = sorted(f"T{i}" for i in range(1, 21))
    for cfg in BIG:
        assert v1["exclusions"][cfg] == legacy_items  # 排除集含全部 legacy
        assert v2["exclusions"][cfg] == legacy_items
    assert v1["hash_prefix"] == "p6v1" and v2["hash_prefix"] == "p6v2" and u["hash_prefix"] == "p6u"
    assert v1["per_config_n"] == 5 and v2["per_config_n"] == 5
    assert v1["additional_exclusions"] == []
    assert v2["additional_exclusions"] == ["V1_selected"]  # V2 显式声明排除 V1 已选
    assert v1["materialized"] is False and v2["materialized"] is False and u["materialized"] is False
    assert v1["requires_content_sha_at_download"] is True
    assert v2["requires_content_sha_at_download"] is True
    assert u["requires_content_sha_at_download"] is True
    assert u["config"] == "traffic_hourly" and u["n"] == 20
    assert u["exclusions"] == sorted(set(U_EXCLUDED))


def test_virgin_selection_hash_order_and_v2_excludes_v1(manifest):
    cfg = "fred_md"
    legacy_items = [f"T{i}" for i in range(1, 21)]
    fresh = [f"F{i}" for i in range(1, 13)]  # 12 条新下载候选
    universe = legacy_items + fresh

    sel1 = select_virgin_items(manifest, "V1", cfg, universe)
    exp1 = sorted(fresh, key=lambda it: (_hex(f"p6v1|{cfg}|{it}"), it))[:5]
    assert sel1 == exp1  # 排除集剔掉全部 legacy 后按 p6v1 hash 升序取前 5
    assert not set(sel1) & set(legacy_items)

    sel2 = select_virgin_items(manifest, "V2", cfg, universe, v1_selected=sel1)
    exp2 = sorted(set(fresh) - set(sel1), key=lambda it: (_hex(f"p6v2|{cfg}|{it}"), it))[:5]
    assert sel2 == exp2  # V2 用 p6v2 前缀且额外排除 V1 已选
    assert not set(sel2) & set(sel1) and not set(sel2) & set(legacy_items)

    with pytest.raises(P6ManifestError):
        select_virgin_items(manifest, "V2", cfg, universe)  # V2 必须显式给 v1_selected
    with pytest.raises(P6ManifestError):
        select_virgin_items(manifest, "V1", cfg, universe, v1_selected=sel1)  # V1 禁传
    with pytest.raises(P6ManifestError):
        select_virgin_items(manifest, "V1", cfg, legacy_items)  # 候选不足（全被排除）
    with pytest.raises(P6ManifestError):
        select_virgin_items(manifest, "V1", "traffic_hourly", universe)  # 非规格 config
    with pytest.raises(P6ManifestError):
        select_virgin_items(manifest, "U", cfg, universe)  # U 不走本函数

    # U：候选 = 全部 item_id 减去探针分析过的 24 条，p6u hash 升序前 20
    fresh_u = [f"N{i}" for i in range(1, 31)]
    sel_u = select_u_items(manifest, U_EXCLUDED + fresh_u)
    exp_u = sorted(fresh_u, key=lambda it: (_hex(f"p6u|{it}"), it))[:20]
    assert sel_u == exp_u and len(sel_u) == 20
    assert not set(sel_u) & set(U_EXCLUDED)
    with pytest.raises(P6ManifestError):
        select_u_items(manifest, U_EXCLUDED + fresh_u[:19])  # 排除后不足 20


# ------------------------------------------------------------------ f. 往返

def test_write_load_roundtrip(manifest, tmp_path):
    path = tmp_path / "p6_split_manifest.json"
    sha = write_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded.manifest_sha == manifest.manifest_sha == sha
    assert loaded.payload == manifest.payload

    # 文件内嵌易变字段不影响 sha（load 剔除）
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["created_at"] = "2026-07-10T12:00:00Z"
    p_vol = tmp_path / "with_volatile.json"
    p_vol.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    assert load_manifest(p_vol).manifest_sha == sha

    # 篡改内容 → 内嵌 sha 校验失败
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["blocks"]["C0"][0], doc["blocks"]["C0"][1] = doc["blocks"]["C0"][1], doc["blocks"]["C0"][0]
    p_bad = tmp_path / "tampered.json"
    p_bad.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(P6ManifestError):
        load_manifest(p_bad)

    # 即使抹掉内嵌 sha，放置规则重建比对仍能抓住篡改
    doc.pop("manifest_sha")
    p_bad2 = tmp_path / "tampered_nosha.json"
    p_bad2.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(P6ManifestError):
        load_manifest(p_bad2)


# ------------------------------------------------- g. 真实文件集成（可 skip）

_PKG_ROOT = Path(__file__).resolve().parents[1]
_PROBES = _PKG_ROOT / "results" / "Stage2" / "P6Probes"
_LEDGER = _PROBES / "exposure_ledger.jsonl"
_META = _PKG_ROOT / "data" / "_artifacts" / "monash_clean.meta.jsonl"
#: F3/finding 33：承重 U 排除集切换到 v2 全宇宙复检探针（首轮 24 条文件保留但不再被引用）。
_UPROBE_V2 = _PROBES / "u_admission_v2_traffic_hourly.json"
#: v2 探针文件冻结 sha256（codex 外审无法本机跑 pytest 的验证锚；prereg §7 冻结清单成员）。
_UPROBE_V2_SHA256 = "768a380d733db1ef0a1718112d56fab34d178d2fc03f8ec4b8f6a525a8947fb2"
_U_UNIVERSE_TOTAL = 862          # traffic_hourly 全量
_U_N_EXCLUDED = 56               # 首轮 24 + 全宇宙复检 32（零重叠）
_U_SAMPLE_UNIVERSE = 806         # 862 − 56 = 抽取宇宙


def test_f3_u_v2_probe_pinned():
    """F3/finding 33：承重 U 排除集 = v2 全宇宙复检探针；固定 sha256、排除 56、宇宙 806。"""
    assert _UPROBE_V2.exists(), "v2 U 探针缺失（承重工件）"
    raw = _UPROBE_V2.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _UPROBE_V2_SHA256      # 文件 sha256 固定
    probe = json.loads(raw)
    assert probe["config"] == "traffic_hourly"
    excl = probe["all_probe_consumed_item_ids"]
    assert len(excl) == _U_N_EXCLUDED == 56 and len(set(excl)) == 56
    acc = probe["exclusion_accounting"]
    assert acc["n_round1"] == 24 and acc["n_v2_capability"] == 32 and acc["n_overlap"] == 0
    assert acc["n_total"] == _U_N_EXCLUDED
    assert probe["report1_universe"]["n_loaded"] == _U_UNIVERSE_TOTAL == 862
    assert _U_UNIVERSE_TOTAL - _U_N_EXCLUDED == _U_SAMPLE_UNIVERSE == 806


@pytest.mark.skipif(
    not (_LEDGER.exists() and _META.exists() and _UPROBE_V2.exists()),
    reason="真实 P6 输入缺失（exposure_ledger / monash_clean.meta / u_admission_v2 探针）",
)
def test_integration_real_files(tmp_path):
    rows = read_jsonl(_LEDGER)
    assert len(rows) == 83
    cnt = Counter(r["config"] for r in rows)
    assert all(cnt[c] == 20 for c in BIG)
    assert all(cnt[c] == 1 for c in SINGLE)
    assert all(r["series_uid"] == f"{r['config']}:{r['item_id']}" for r in rows)
    assert all(r["exposure_class"] == "confirmed_exposed" for r in rows)

    # F3：承重排除集 = v2 全宇宙复检的 all_probe_consumed_item_ids（56 条）
    probe = json.loads(_UPROBE_V2.read_text(encoding="utf-8"))
    excl = probe["all_probe_consumed_item_ids"]
    assert len(excl) == 56 and len(set(excl)) == 56
    assert probe["config"] == "traffic_hourly"

    m = build_manifest(rows, excl)
    validate_manifest(m)  # 放置规则全过
    assert m.payload["counts"] == {
        "C0": 16, "D1": 32, "D2": 32, "C0_qualitative": 3, "legacy_total": 83,
    }
    for cfg in BIG:
        want = sorted(r["item_id"] for r in rows if r["config"] == cfg)
        assert m.payload["virgin_specs"]["V1"]["exclusions"][cfg] == want
        assert m.payload["virgin_specs"]["V2"]["exclusions"][cfg] == want
    assert m.payload["virgin_specs"]["U"]["exclusions"] == sorted(excl)
    assert m.block("C0_qualitative") == sorted(f"{c}:T1" for c in SINGLE)

    # meta 语料一致性（只读 sanity）：账本 (config,item) ⊆ meta；
    # 注意 meta 只含 83 条已用序列、无 traffic_hourly——它不是 virgin 候选全集。
    meta_rows = read_jsonl(_META)
    meta_pairs = {(r["config"], r["item_id"]) for r in meta_rows}
    ledger_pairs = {(r["config"], r["item_id"]) for r in rows}
    assert ledger_pairs <= meta_pairs
    assert all(cfg != "traffic_hourly" for cfg, _ in meta_pairs)

    # 确定性（行序无关）+ 往返（只写 tmp_path，不碰 results/）
    m2 = build_manifest(list(reversed(rows)), list(reversed(excl)))
    assert m2.manifest_sha == m.manifest_sha
    out = tmp_path / "real_manifest_roundtrip.json"
    sha = write_manifest(m, out)
    assert load_manifest(out).manifest_sha == sha == m.manifest_sha
