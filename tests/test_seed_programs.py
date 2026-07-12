"""P3 契约测试：种子供给 skill bank v1（prereg §3 冻结清单；grammar 合法 + menu 不可表达）。"""
import json

from SelfEvolvingHarnessTS.policy.program_edit import is_novel_v1, validate_v1
from SelfEvolvingHarnessTS.policy.seed_programs import (
    SEED_PROGRAMS_V1,
    seed_bank_manifest,
    seed_skill_cards,
)


def test_bank_size_and_task_split():
    assert len(SEED_PROGRAMS_V1) == 8
    tasks = [s.task_type for s in SEED_PROGRAMS_V1.values()]
    assert tasks.count("forecast") == 6
    assert tasks.count("anomaly_detection") == 2


def test_all_seeds_pass_grammar_v1():
    for name, spec in SEED_PROGRAMS_V1.items():
        ok, why = validate_v1(spec)
        assert ok, f"{name}: {why}"


def test_all_seeds_are_menu_inexpressible():
    # 供给主张的前提：种子的 resolved 链身份 ∉ 冻结 menu v1 的 15 动作
    for name, spec in SEED_PROGRAMS_V1.items():
        assert is_novel_v1(spec), f"{name} 与 menu 动作 resolved 身份重合"


def test_seed_identities_deterministic():
    shas = {name: spec.sha() for name, spec in SEED_PROGRAMS_V1.items()}
    assert len(set(shas.values())) == 8                     # 互不重复
    from SelfEvolvingHarnessTS.policy.seed_programs import SEED_PROGRAMS_V1 as again
    assert {n: s.sha() for n, s in again.items()} == shas   # 构造确定性


def test_guarded_seed_present():
    spec = SEED_PROGRAMS_V1["seed_period_stl"]
    assert spec.pattern_guard == (("seasonal_strength", ">=", 0.5),)
    assert spec.steps[0][0] == "period_complete"


def test_skill_cards_and_manifest_serializable():
    cards = seed_skill_cards()
    assert len(cards) == 8
    for card in cards:
        assert card["version"] == "bank_v1"
        assert card["program_spec"]["grammar"] == "v1"
        assert card["task_scope"] in ("forecast", "anomaly_detection")
    json.dumps(cards, allow_nan=False)
    manifest = seed_bank_manifest()
    assert all(entry["novel_vs_menu_v1"] for entry in manifest.values())
    json.dumps(manifest, allow_nan=False)
