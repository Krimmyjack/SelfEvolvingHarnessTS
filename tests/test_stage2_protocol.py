"""tests/test_stage2_protocol.py — 2.0-⑥ 张量协议守卫（防静默漂移；v3 版）。

守：①v3 config_sha 可复算 + v2/v1/v0 原样保留（supersedes 链完整，sha 钉死）；
②action menu SHA 与 policy.action_menu_v1() 实时一致（resolved 语义身份）；
③holdout 解锁走独立 access log（协议内无 holdout_opened；log 缺失/为空=从未读）；
④动作 roles 三分并集=menu 全集；⑤分支规则数值+worst-group 安全预注册；
⑥v3=frozen_full：8 族全冻结、library 参数快照与 s2_corpus 单一真源一致（防协议↔生成器漂移）、
张量 gate 自 v3 起合格；⑦DLinear 主口径=within-domain pooled（v2 勘误沿袭）。
协议缺失（新 clone 未落盘）→ skip。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parent.parent / "results" / "Stage2"
PROTOCOL_V3 = _DIR / "tensor_protocol_v3.json"
CHAIN = ((_DIR / "tensor_protocol_v2.json", "f007d6cfaa04ec84"),
         (_DIR / "tensor_protocol_v1.json", "4cf04acb8d46299c"),
         (_DIR / "tensor_protocol.json", "ff79883f196200c3"))
ACCESS_LOG = _DIR / "holdout_access_log.jsonl"

pytestmark = pytest.mark.skipif(not PROTOCOL_V3.exists(), reason="tensor_protocol_v3.json 未落盘")


@pytest.fixture(scope="module")
def proto():
    return json.loads(PROTOCOL_V3.read_text("utf-8"))


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def test_config_sha_recomputes_and_chain_untouched(proto):
    assert proto["config_sha"] == _sha_obj({k: v for k, v in proto.items() if k != "config_sha"}), \
        "v3 协议文件被原地改动——改动须另存新版本文件"
    assert proto["version"] == "v3" and "f007d6cfaa04ec84" in proto["supersedes"]
    for path, sha in CHAIN:
        if path.exists():                                     # 旧版本原样保留（不删不改）
            old = json.loads(path.read_text("utf-8"))
            assert old["config_sha"] == _sha_obj({k: v for k, v in old.items() if k != "config_sha"})
            assert old["config_sha"] == sha


def test_v3_frozen_full_and_tensor_gate(proto):
    assert proto["status"] == "frozen_full"
    pv = proto["protocol_versioning"]
    assert "已全冻结" in pv["freeze_process"] and "另存 v4" in pv["freeze_process"]
    assert "v3 起合格" in pv["tensor_gate"] and "v0–v2 永久不合格" in pv["tensor_gate"]


def test_structure_library_matches_generator_source(proto):
    """协议快照 == s2_corpus 单一真源（防协议与生成器漂移——语料生成后任何一侧改动都在此炸响）。"""
    from SelfEvolvingHarnessTS import s2_corpus as sc
    lib = proto["structure_library_v3"]
    assert lib["frozen_families"] == list(sc.S2_FAMILIES) and len(lib["frozen_families"]) == 8
    assert lib["family_grid"] == {k: [dict(p) for p in v] for k, v in sc.S2_FAMILY_GRID.items()}
    assert lib["deg_grid"] == {k: dict(v) for k, v in sc.S2_DEG_GRID.items()}
    dev, hold = sc.s2_split()
    assert lib["wave"]["n_dev"] == len(dev) and lib["wave"]["n_holdout_reserved"] == len(hold)
    assert lib["wave"]["dev_j"] == list(sc.DEV_J)
    assert "structure_library_v2_superseded" in proto        # v2 draft 原文留档
    assert set(proto["real_domains_frozen"]["pick"]) == {"nn5_daily", "tourism_monthly", "fred_md"}


def test_action_menu_sha_matches_live_menu(proto):
    from SelfEvolvingHarnessTS.policy import action_menu_v1
    menu = action_menu_v1()
    ax = proto["tensor_axes"]["action"]
    assert ax["menu_version"] == menu.version
    assert ax["menu_sha256"] == menu.sha256, \
        "action menu 语义相对协议漂移（resolved params/动作集改动须走新 menu SHA + 新协议版本）"
    assert ax["n_actions"] == len(menu)


def test_holdout_sealed_via_access_log(proto):
    assert "holdout_opened" not in proto, "协议不可变——解锁记录不进协议文件"
    assert proto["holdout_access"]["log_file"].endswith("holdout_access_log.jsonl")
    if ACCESS_LOG.exists():                                   # log 在则必须为空（本阶段不得读 holdout）
        assert not ACCESS_LOG.read_text("utf-8").strip(), "Stage-2 holdout 在协议阶段被读取过！"


def test_action_roles_cover_full_menu(proto):
    from SelfEvolvingHarnessTS.policy import action_menu_v1
    roles = proto["tensor_axes"]["action"]["roles"]
    covered = (set(roles["core_pool"]) | set(roles["ablation"])
               | set(roles["dosage_diagnostic"]["actions"]))
    assert covered == set(action_menu_v1().actions), "存在归属不清的动作"
    assert len(roles["core_pool"]) == 10 and len(roles["ablation"]) == 3


def test_branch_rules_preregistered_with_worst_group(proto):
    br = proto["branch_rules"]
    assert br["eps"] == 0.03 and br["delta_safe"] == 0.05
    assert "90%" in br["dominance"]["rule"]
    assert "worst_group_safety" in br["dominance"] and "LCB" in br["dominance"]["worst_group_safety"]


def test_dlinear_training_unit_pooled_within_domain(proto):
    dl = next(m for m in proto["tensor_axes"]["model"]["pilot"] if m["id"] == "dlinear_scratch")
    assert "within-domain pooled" in dl["train"] and "共享" in dl["train"]
    diag = proto["tensor_axes"]["model"]["diagnostic_only"]
    assert any("dlinear_per_series" in x for x in diag)
    assert "utility_vs_report" in proto["measurement"] and "estimand" in proto["measurement"]


def test_pattern_spec_ref_matches_p0(proto):
    from SelfEvolvingHarnessTS.policy import pattern_spec_p0
    ref = proto["pattern_spec_ref"]
    assert ref["version"] == "P0" and ref["config_sha"] == pattern_spec_p0().config_sha()


def test_runtime_pinning_declared(proto):
    assert "fail-loud" in proto["runtime_pinning"]["rule"]
    assert "provenance" in proto["runtime_pinning"]["impl"]
