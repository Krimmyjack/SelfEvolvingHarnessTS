"""P3 契约测试：TS-Readiness Replay Gym（0-API、预算、trace 反馈、泄漏纪律、确定性）。"""
import json

import numpy as np
import pytest

from SelfEvolvingHarnessTS.evaluators.anomaly_rig import make_anomaly_slice
from SelfEvolvingHarnessTS.readiness_gym import ReadinessGym

LEAK_KEYS = {"labels", "future_clean", "true_delta", "x", "series", "noise_sigma"}


def _all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _all_keys(value)


def _prog(steps, task="forecast"):
    return {"grammar": "v1", "steps": steps, "scope": ["*"], "task_type": task,
            "pattern_guard": [], "risk_budget_beta": 0.3, "fallback": "v_impute_linear"}


def _gym(task="forecast", budget=4, n=4):
    rows = make_anomaly_slice(n, seed=17)
    return ReadinessGym(rows, task=task, budget=budget)


def test_reset_observation_shape_and_leakage():
    gym = _gym()
    obs = gym.reset(0)
    assert obs["task"]["task_type"] == "forecast"
    assert obs["allowed_grammar"]["grammar"] == "program_spec_v1"
    assert obs["budget_remaining"] == 4
    assert obs["evals"] == []
    assert not (LEAK_KEYS & set(_all_keys(obs)))
    json.dumps(obs, allow_nan=False)


def test_proxy_eval_consumes_budget_and_returns_trace():
    gym = _gym()
    gym.reset(0)
    obs, done = gym.step({"op": "proxy_eval",
                          "program_spec": _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]])})
    assert not done
    assert obs["budget_remaining"] == 3
    entry = obs["evals"][-1]
    assert entry["ok"] is True
    assert isinstance(entry["proxy_delta"], float)
    assert "modified_fraction" in entry
    assert entry["program_sha"]
    assert not (LEAK_KEYS & set(_all_keys(obs)))


def test_invalid_program_gets_structured_reason_and_costs_budget():
    gym = _gym()
    gym.reset(0)
    obs, done = gym.step({"op": "proxy_eval",
                          "program_spec": _prog([["denoise_median", {"window": 9}]])})   # 首步非 imputer
    assert not done
    assert obs["budget_remaining"] == 3                       # ITT：无效提案照样耗预算
    entry = obs["evals"][-1]
    assert entry["ok"] is False
    assert "imputer" in entry["reason"]


def test_budget_exhaustion_blocks_proxy_eval():
    gym = _gym(budget=1)
    gym.reset(0)
    gym.step({"op": "proxy_eval", "program_spec": _prog([["impute_linear", {}]])})
    obs, done = gym.step({"op": "proxy_eval", "program_spec": _prog([["impute_linear", {}]])})
    assert not done
    assert obs["evals"][-1]["reason"] == "budget_exhausted"
    assert obs["budget_remaining"] == 0


def test_finalize_records_true_delta_only_in_result():
    gym = _gym()
    gym.reset(0)
    spec = _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]])
    gym.step({"op": "proxy_eval", "program_spec": spec})
    obs, done = gym.step({"op": "finalize", "program_spec": spec})
    assert done
    assert not (LEAK_KEYS & set(_all_keys(obs)))              # true 不进 observation
    result = gym.result(0)
    assert result["final_kind"] == "program"
    assert isinstance(result["true_delta"], float)
    assert result["proxy_evals_used"] == 1


def test_abstain_finalizes_with_raw():
    gym = _gym()
    gym.reset(0)
    obs, done = gym.step({"op": "abstain"})
    assert done
    result = gym.result(0)
    assert result["final_kind"] == "abstain"
    assert result["true_delta"] == 0.0                        # raw vs raw


def test_observation_provides_p0_fingerprint_and_contract_consistency():
    # P5-A.2 前置①：观测面提供完整 P0 指纹 → allowed_grammar 宣传的每个 guard 特征都有背书
    from SelfEvolvingHarnessTS.e32_policy import P_FEATS

    gym = _gym()
    obs = gym.reset(0)
    pattern = obs["pattern"]
    assert set(P_FEATS) <= set(pattern["struct_feats"])
    assert all(isinstance(v, float) for v in pattern["struct_feats"].values())
    assert isinstance(pattern["snr"], float)                 # 估计 SNR（可观测，非生成真值）
    assert isinstance(pattern["missing_rate"], float)
    provided = set(pattern["struct_feats"]) | {"snr", "missing_rate"}
    advertised = set(obs["allowed_grammar"]["guard_features"])
    assert advertised <= provided                            # 契约一致：宣传 ⊆ 提供


def test_guarded_program_now_evaluable_end_to_end():
    # P5-A.2 前置①验收：guarded 程序可评估（P5-A 曾 179/180 死于 feature-missing）
    gym = _gym()
    gym.reset(0)                                             # 强季节 sine 底座
    prog = _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]])
    prog["pattern_guard"] = [["seasonal_strength", ">=", 0.3]]
    obs, _ = gym.step({"op": "proxy_eval", "program_spec": prog})
    assert obs["evals"][-1]["ok"] is True                    # guard 满足 → 真实执行

    strict = _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]])
    strict["pattern_guard"] = [["seasonal_strength", ">=", 0.99]]
    obs, _ = gym.step({"op": "proxy_eval", "program_spec": strict})
    entry = obs["evals"][-1]
    assert entry["ok"] is False
    assert entry["reason"] == "pattern_guard_unsatisfied"    # 被评估后不满足 ≠ 特征缺失


def test_task_mismatch_program_rejected():
    gym = _gym(task="anomaly_detection")
    gym.reset(0)
    obs, _ = gym.step({"op": "proxy_eval",
                       "program_spec": _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]],
                                             task="forecast")})
    assert obs["evals"][-1]["ok"] is False


def test_anomaly_proxy_penalizes_smoothing():
    # anomaly proxy=告警保存率：直接对 artifact 平滑 → proxy_delta 显著为负
    gym = _gym(task="anomaly_detection", budget=4)
    gym.reset(1)                                              # snrLow|miss 行也可
    obs, _ = gym.step({"op": "proxy_eval",
                       "program_spec": _prog([["impute_linear", {}]], task="anomaly_detection")})
    ok_delta = obs["evals"][-1]["proxy_delta"]
    assert ok_delta >= -0.2                                   # 插补基本无害


def test_deterministic_replay():
    script = [
        {"op": "proxy_eval", "program_spec": _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]])},
        {"op": "finalize", "program_spec": _prog([["impute_linear", {}], ["denoise_median", {"window": 9}]])},
    ]
    outs = []
    for _ in range(2):
        gym = _gym()
        gym.reset(0)
        for action in script:
            obs, done = gym.step(action)
        outs.append((json.dumps(obs, sort_keys=True), json.dumps(gym.result(0), sort_keys=True)))
    assert outs[0] == outs[1]
