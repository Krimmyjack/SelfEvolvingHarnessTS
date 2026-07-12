"""P5-A.3 前置④契约测试：ReadinessPlan → deterministic compiler（对症 L1 合规塌缩）。

架构主张：LLM 只产出**语义意图**（plan），合规由确定性编译器**按构造保证**——任何结构
合法的 plan 编译出的 ProgramSpecV1 必过 validate_v1；guard 只保留观测面有背书的谓词
（丢弃项留痕）。
"""
import itertools
import json

import pytest

from SelfEvolvingHarnessTS.policy.program_edit import validate_v1
from SelfEvolvingHarnessTS.policy.readiness_plan import (
    PLAN_SCHEMA_VERSION,
    PlanComposer,
    compile_plan,
    plan_from_dict,
)

FINGERPRINT = {"snr": 5.0, "missing_rate": 0.1,
               "struct_feats": {"seasonal_strength": 0.7, "period": 24.0}}


def _plan(**kw):
    base = {"plan": PLAN_SCHEMA_VERSION, "task_type": "forecast",
            "impute": "linear", "outlier_clip": False,
            "denoise": {"family": "median", "strength": "light"},
            "guard": [], "risk_budget_beta": 0.3, "rationale": "test"}
    base.update(kw)
    return base


def test_compile_valid_plan_passes_grammar():
    spec, info = compile_plan(plan_from_dict(_plan()), FINGERPRINT)
    ok, why = validate_v1(spec)
    assert ok, why
    assert [op for op, _ in spec.steps] == ["impute_linear", "denoise_median"]
    assert dict(spec.steps[1][1])["window"] == 9                     # light → w9


def test_compile_is_deterministic_and_full_pipeline_shape():
    plan = plan_from_dict(_plan(impute="period", outlier_clip=True,
                                denoise={"family": "ma", "strength": "heavy"}))
    s1, _ = compile_plan(plan, FINGERPRINT)
    s2, _ = compile_plan(plan, FINGERPRINT)
    assert s1.sha() == s2.sha()
    assert [op for op, _ in s1.steps] == ["period_complete", "winsorize", "smooth_ma"]
    assert dict(s1.steps[2][1])["window"] == 25                      # heavy → w25


def test_every_enumerable_plan_compiles_to_valid_program():
    # 合规按构造保证（穷举 4 impute × 2 clip × 6 family × 3 strength）
    for imp, clip, fam, stg in itertools.product(
            ("linear", "period", "fft", "ema"), (False, True),
            ("median", "ma", "savgol", "stl", "wavelet", "none"),
            ("light", "medium", "heavy")):
        plan = plan_from_dict(_plan(impute=imp, outlier_clip=clip,
                                    denoise={"family": fam, "strength": stg}))
        spec, _ = compile_plan(plan, FINGERPRINT)
        ok, why = validate_v1(spec)
        assert ok, f"{imp}/{clip}/{fam}/{stg}: {why}"


def test_guard_kept_only_when_backed_and_drops_recorded():
    plan = plan_from_dict(_plan(guard=[["seasonal_strength", ">=", 0.3],
                                       ["volatility_of_vibes", ">", 0.0],
                                       ["snr", "<", 10.0]]))
    spec, info = compile_plan(plan, FINGERPRINT)
    kept = [g[0] for g in spec.pattern_guard]
    assert kept == ["seasonal_strength", "snr"]                      # 有背书的保留
    assert info["dropped_guards"] == [["volatility_of_vibes", ">", 0.0]]
    ok, why = validate_v1(spec)
    assert ok, why


def test_anomaly_task_drops_destructive_steps_with_record():
    plan = plan_from_dict(_plan(task_type="anomaly_detection", outlier_clip=True,
                                denoise={"family": "median", "strength": "light"}, guard=[]))
    spec, info = compile_plan(plan, FINGERPRINT)
    assert [op for op, _ in spec.steps] == ["impute_linear"]         # registry 契约内合法面
    assert set(info["dropped_steps"]) == {"winsorize", "denoise_median"}
    ok, why = validate_v1(spec)
    assert ok, why


def test_malformed_plan_fails_loud():
    with pytest.raises(ValueError):
        plan_from_dict({"plan": "v9"})
    with pytest.raises(ValueError):
        plan_from_dict(_plan(impute="magic"))
    with pytest.raises(ValueError):
        plan_from_dict(_plan(denoise={"family": "median", "strength": "extreme"}))
    with pytest.raises(ValueError):
        plan_from_dict("not a mapping")


def test_plan_composer_stub_and_llm_itt():
    stub = PlanComposer(backend="stub")
    outcome = stub.compose({"task": {"task_type": "forecast"},
                            "pattern": FINGERPRINT, "episode_uid": "u"})
    assert outcome.candidate is not None
    assert outcome.candidate.program_spec["grammar"] == "v1"         # 编译产物=合法 ProgramSpec
    assert outcome.plan is not None

    valid_plan_json = json.dumps(_plan())
    llm_ok = PlanComposer(backend="llm", llm=lambda s, u, nonce=0: valid_plan_json)
    o2 = llm_ok.compose({"task": {"task_type": "forecast"}, "pattern": FINGERPRINT})
    assert o2.candidate is not None and o2.api_calls == 1

    prose = PlanComposer(backend="llm", llm=lambda s, u, nonce=0: "no json here")
    o3 = prose.compose({"task": {"task_type": "forecast"}, "pattern": FINGERPRINT})
    assert o3.candidate is None and o3.invalid_reason                # ITT no-op

    repaired = PlanComposer(backend="llm", repair_retries=1,
                            llm=(lambda calls=[]:
                                 lambda s, u, nonce=0: (calls.append(1),
                                                        "bad" if len(calls) == 1 else valid_plan_json)[1])())
    o4 = repaired.compose({"task": {"task_type": "forecast"}, "pattern": FINGERPRINT})
    assert o4.candidate is not None and o4.api_calls == 2
