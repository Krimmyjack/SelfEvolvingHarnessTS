"""V1 actionable 机械正控闭环（零 LLM，2026-08-08）。

审查裁决（十）：用 exposed mechanical positive control（outlier_iqr——
Support +0.0439@928 / delayed +0.0272@976 已暴露）验证合法 Pipeline 的机械闭环：
  1. 正控 Episode 进 Memory → 真实 fast_agent → actual verifier；
  2. **用 Fast Agent 实际返回的 prepared values 做 Support**（不按 candidate ID
     重建 Workflow——2A 假 PASS 的教训）；
  3. 写 Episode、打开 delayed；
  4. 下一轮确认该局部经验被检索并影响行动。

结果只证明合法 Pipeline 的机械闭环（正控已暴露，不宣称 A5 跨域效果）。

Support 评估：单序列（真实入口语义）——Ridge 训练用 prepared（处理后）窗口、
eval 用原始 context/future（与 V1 gain 语义一致：处理后训练数据 → 原始未来评价）。

用法：
  python evaluation/functional/run_v1_actionable_mechanical_control.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402（策略/contract_params 复用）
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import build_episode  # noqa: E402

HORIZON = 48
CONTEXT_LENGTH = 192
PERIOD = 24
TARGET_DOMAIN = "gefcom"
# prepared 语义正控（全链扫描选定）：winsorize @832 +0.188（prepared 语义，
# 与 v6 语义 +0.144 一致——两语义可靠；outlier_iqr 在 prepared 语义下为负，弃用）
CONTROL_OP = "winsorize"
CONTROL_ORIGIN = 832
CONTROL_DELAYED = 880
MATERIAL = 0.005
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_actionable_mechanical_control_report.json")


def evaluate_prepared_single(
    raw_series: np.ndarray,
    prepared_series: np.ndarray,
    origin: int,
    period: int,
    anchors: Sequence[int] = (),
) -> float:
    """单序列 sMASE gain：Ridge 训练用 prepared（处理后）窗口、eval 用原始。

    与 V1 gain 语义一致（处理后训练数据 → 原始未来评价）；prepared 来自
    Fast Agent 实际返回的 PreparedSeries（不重建）。
    """
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale, smase
    from run_e2_cross_series_curation import _exact_weighted_ridge_prediction

    def ridge_fit(x_train: list[np.ndarray], y_train: list[np.ndarray],
                  x_eval: np.ndarray) -> np.ndarray:
        # 复用 v6 冻结 Ridge（alpha=1、unpenalized intercept、精确解）——
        # 与 V1 gain 评估完全一致，零重写偏差
        return _exact_weighted_ridge_prediction(
            np,
            x_train=np.vstack(x_train),
            targets=np.vstack(y_train),
            weights=np.ones(len(x_train), dtype=np.float64),
            x_eval=np.asarray(x_eval, dtype=np.float64),
        )

    def evaluate(use_prepared: bool) -> float:
        x_train: list[np.ndarray] = []
        y_train: list[np.ndarray] = []
        # raw 取到 truth 为止（部署可见历史 + 未来 truth 仅用于评估）
        raw = np.asarray(raw_series, dtype=np.float64)[: origin + HORIZON]
        # anchor 集合与 v6 一致（config["anchors"]）——评估语义对齐
        anchor_list = list(anchors) if anchors else \
            range(CONTEXT_LENGTH, origin - HORIZON + 1, period)
        for anchor in anchor_list:
            if anchor + HORIZON > origin:
                continue
            src = prepared_series if use_prepared else raw
            window = src[anchor - CONTEXT_LENGTH: anchor + HORIZON]
            prepared = np.asarray(window, dtype=np.float64)
            if not np.isfinite(prepared).all():
                # 最小完整性修复（与 v6 _linear_integrity 同语义）
                mask = ~np.isfinite(prepared)
                idx = np.arange(prepared.size)
                prepared[mask] = np.interp(idx[mask], idx[~mask], prepared[~mask])
            context = prepared[:CONTEXT_LENGTH]
            target = prepared[CONTEXT_LENGTH:]
            center = float(np.median(context))
            scale = float(1.4826 * np.median(np.abs(context - center)))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(context))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            x_train.append((context - center) / scale)
            y_train.append((target - center) / scale)
        # eval context（原始，不处理）
        eval_window = raw[origin - CONTEXT_LENGTH: origin]
        prepared_eval = np.asarray(eval_window, dtype=np.float64)
        if not np.isfinite(prepared_eval).all():
            mask = ~np.isfinite(prepared_eval)
            idx = np.arange(prepared_eval.size)
            prepared_eval[mask] = np.interp(idx[mask], idx[~mask], prepared_eval[~mask])
        center = float(np.median(prepared_eval))
        scale = float(1.4826 * np.median(np.abs(prepared_eval - center)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(prepared_eval))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        x_eval = np.asarray([(prepared_eval - center) / scale])
        truth = raw[origin: origin + HORIZON]
        pred = ridge_fit(x_train, y_train, x_eval)[0] * scale + center
        metric_scale = seasonal_scale(raw[:origin], np.isfinite(raw[:origin]),
                                      period=period, min_pairs=32)
        observed = np.isfinite(truth)
        if not observed.any():
            return 0.0  # 与 v6 一致：truth 全缺失则跳过该序列
        return float(smase(truth[observed], pred[observed], scale=metric_scale))

    baseline = evaluate(use_prepared=False)
    candidate = evaluate(use_prepared=True)
    return baseline - candidate


def build_control_episode(root: Path) -> Any:
    """winsorize 机械正控 Episode——support/delayed 用 prepared 语义实测
    （verify winsorize → prepared values → evaluate_prepared_single），
    与闭环评估同源（正控暴露但不硬编码数值）。"""
    from SelfEvolvingHarnessTS.contracts.candidate import Candidate
    from SelfEvolvingHarnessTS.contracts.program import Program
    from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    _roster, values = v6._fixed_roster(root, config)
    series = np.asarray(values[list(values.keys())[0]])
    anchors = tuple(int(a) for a in config.get("anchors", ()))
    r_values = np.asarray(series)[: CONTROL_ORIGIN]
    params = wiring.contract_params(CONTROL_OP, PERIOD)
    candidate = Candidate.program_candidate(
        f"probe_{CONTROL_OP}",
        Program.from_steps([(CONTROL_OP, params)], source="control_episode"),
        source="control_episode")
    artifact = verify_candidate(
        candidate, r_values, allowed_operators=(CONTROL_OP,),
        inspected_regions=((0, CONTROL_ORIGIN),),
        maximum_modified_fraction=1.0,
        preserve_outside_inspected_region=True,
        require_finite_output=False)
    assert artifact.selectable and artifact.prepared_values is not None
    prepared = np.asarray(artifact.prepared_values, dtype=np.float64)
    support_gain = evaluate_prepared_single(series, prepared, CONTROL_ORIGIN,
                                            PERIOD, anchors)
    # delayed：独立 verify @CONTROL_DELAYED（决策点 880 长度的 request——
    # 与 support 的 prepared 语义一致，非 concat 混合）
    r_delayed = np.asarray(series)[: CONTROL_DELAYED]
    artifact_d = verify_candidate(
        candidate, r_delayed, allowed_operators=(CONTROL_OP,),
        inspected_regions=((0, CONTROL_DELAYED),),
        maximum_modified_fraction=1.0,
        preserve_outside_inspected_region=True,
        require_finite_output=False)
    assert artifact_d.selectable and artifact_d.prepared_values is not None
    prepared_d = np.asarray(artifact_d.prepared_values, dtype=np.float64)
    delayed_gain = evaluate_prepared_single(series, prepared_d, CONTROL_DELAYED,
                                            PERIOD, anchors)
    f_support = resolver.window_context(values, CONTROL_ORIGIN, PERIOD)
    f_delayed = resolver.window_context(values, CONTROL_DELAYED, PERIOD)
    print(f"== control episode: {CONTROL_OP} support={support_gain:.5f} "
          f"delayed={delayed_gain:.5f}")
    return build_episode(
        episode_id=f"{TARGET_DOMAIN}_control_{CONTROL_OP}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=TARGET_DOMAIN,
        context_summary={
            "cohort": {"series_count": 1, "evaluation_series_count": 0},
            "local_pattern": {"support_gain": float(support_gain), **f_support},
            "delayed_pattern": dict(f_delayed),
            "program_geometry": {
                "scope": "training_rows",
                "program_steps": [{"op": CONTROL_OP, "params": params}],
            },
        },
        workflow_signature=CONTROL_OP,
        support_response={"gain": float(support_gain)},
        delayed_response={"evaluated": True, "gain": float(delayed_gain)},
        relation=("POSITIVE" if support_gain >= MATERIAL and delayed_gain >= MATERIAL
                  else "CONFLICT"),
        evidence_level="DELAYED",
        local_status=("LOCAL_ACTIVE" if support_gain >= MATERIAL and delayed_gain >= MATERIAL
                      else "RESTRICTED"),
        evidence_refs=["run_v1_actionable_mechanical_control"],
    )


def prepared_headroom_scan(
    root: Path, origins: Sequence[int],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """prepared 语义 headroom 扫描（零 LLM）：14 个 actionable 算子的默认候选
    → verify（得到 prepared values）→ evaluate_prepared_single → gain ≥ M 的
    才是 prepared 语义正控（机械闭环的正控必须以真实入口执行器为准——
    v6 语义正控在 prepared 语义下不成立，outlier_iqr 已证实）。"""
    from SelfEvolvingHarnessTS.contracts.candidate import Candidate
    from SelfEvolvingHarnessTS.contracts.program import Program
    from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _actionable_operators, _allowed_operators,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    _roster, values = v6._fixed_roster(root, config)
    series = np.asarray(values[list(values.keys())[0]])
    anchors = tuple(int(a) for a in config.get("anchors", ()))
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for origin in origins:
        r_values = np.asarray(series)[: origin]
        request = PreparationRequest(
            "prepared-scan",
            r_values,
            forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                                  metric=MetricSpec("sMASE", "lower_is_better")),
            {},
        )
        features = extract_public_features(r_values, task_kind="forecast")
        view = resolve_harness_view(h0, features, role="fast")
        actionable = _actionable_operators(request, r_values, view,
                                           _allowed_operators(request))
        for op in sorted(actionable):
            params = wiring.contract_params(op, PERIOD)
            candidate = Candidate.program_candidate(
                f"probe_{op}", Program.from_steps([(op, params)], source="prepared_scan"),
                source="prepared_scan")
            artifact = verify_candidate(
                candidate, r_values, allowed_operators=(op,),
                inspected_regions=((0, origin),),
                maximum_modified_fraction=1.0,
                preserve_outside_inspected_region=True,
                require_finite_output=False)
            if not artifact.selectable or artifact.prepared_values is None:
                results.append({"operator": op, "origin": origin, "selectable": False})
                continue
            gain = evaluate_prepared_single(
                series, np.asarray(artifact.prepared_values, dtype=np.float64),
                origin, PERIOD, anchors)
            delayed_gain = None
            delayed_origin = origin + HORIZON
            if gain >= MATERIAL and delayed_origin <= 1024:
                # support 正时测 delayed（独立 verify @origin+48，prepared 语义）
                r_d = np.asarray(series)[: delayed_origin]
                artifact_d = verify_candidate(
                    candidate, r_d, allowed_operators=(op,),
                    inspected_regions=((0, delayed_origin),),
                    maximum_modified_fraction=1.0,
                    preserve_outside_inspected_region=True,
                    require_finite_output=False)
                if artifact_d.selectable and artifact_d.prepared_values is not None:
                    delayed_gain = evaluate_prepared_single(
                        series, np.asarray(artifact_d.prepared_values, dtype=np.float64),
                        delayed_origin, PERIOD, anchors)
            results.append({"operator": op, "origin": origin, "selectable": True,
                            "support_gain": round(float(gain), 6),
                            "delayed_gain": (round(float(delayed_gain), 6)
                                             if delayed_gain is not None else None)})
            if (gain >= MATERIAL and delayed_gain is not None
                    and delayed_gain >= MATERIAL
                    and (best is None or gain > best["support_gain"])):
                best = {"operator": op, "origin": origin,
                        "support_gain": float(gain),
                        "delayed_gain": float(delayed_gain)}
    return best, results


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    series = np.asarray(values[list(values.keys())[0]])
    anchors = tuple(int(a) for a in config.get("anchors", ()))
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)

    # prepared 语义正控选择（机械闭环正控必须以真实入口执行器为准）
    chain_origins = [736, 784, 832, 880, 928, 976]
    control, scan_rows = prepared_headroom_scan(root, chain_origins)
    print("== prepared-headroom scan (14 actionable × 6 origins):")
    for row in scan_rows:
        if row.get("selectable") and row.get("support_gain", 0) != 0:
            print(f"   {row['operator']:24s} @{row['origin']} gain={row.get('support_gain')}")
    if control is None:
        print("== NO_PREPARED_HEADROOM：合法动作空间在 prepared 语义下全链无正控——"
              "执行器差异（run_pipeline vs v6）使 v6 语义正控失效——诚实记录，"
              "不靠放宽约束")
    else:
        print(f"== prepared 正控: {control['operator']} @{control['origin']} "
              f"gain={control['support_gain']}")
    control_episode = build_control_episode(root)
    memory = [control_episode]

    rounds: list[dict[str, Any]] = []
    for idx, origin in enumerate((CONTROL_ORIGIN, CONTROL_ORIGIN)):
        label = f"round{idx + 1}"
        observed = dict(resolver.window_context(values, origin, PERIOD))
        observed["bound_period"] = float(PERIOD)
        r_values = np.asarray(series)[:origin]
        backend = wiring.DeterministicStrategyBackend()
        request = PreparationRequest(
            "mechanical-control",
            r_values,
            forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                                  metric=MetricSpec("sMASE", "lower_is_better")),
            dict(observed),
        )
        core = TTHAAgentCore(
            backend,
            LocalPublicToolGateway(r_values, task_kind="forecast"),
        )
        result, trace = TTHAFastAgent(core).prepare(
            request, h0, experience_episodes=tuple(memory))
        # 用 Fast Agent 实际返回的 prepared values 做 Support（不重建；
        # raw 传完整 series——truth 是未来，仅评估用）
        support_gain = None
        if result.prepared is not None and trace.chosen_candidate_id != "identity":
            support_gain = evaluate_prepared_single(
                series, np.asarray(result.prepared.values, dtype=np.float64),
                origin, PERIOD, anchors)
        # delayed（同一 prepared 语义：delayed 窗口的处理后序列）
        delayed_gain = None
        if result.prepared is not None and trace.chosen_candidate_id != "identity":
            delayed_gain = evaluate_prepared_single(
                series,
                np.concatenate([np.asarray(result.prepared.values, dtype=np.float64),
                                np.asarray(series)[origin: origin + HORIZON]]),
                origin + HORIZON, PERIOD, anchors)
        rounds.append({
            "round": label,
            "chosen": trace.chosen_candidate_id,
            "compilation": trace.compilation_status,
            "prepared_status": result.status.name,
            "support_gain": support_gain,
            "delayed_gain": delayed_gain,
            "instruction_ref1": "Reference 1" in (
                wiring.DeterministicStrategyBackend.extract_instruction(
                    backend.requests[-1].messages) if backend.requests else ""),
        })
        print(f"[{label}] chosen={trace.chosen_candidate_id} "
              f"compile={trace.compilation_status} status={result.status.name} "
              f"support={support_gain if support_gain is None else round(support_gain, 5)} "
              f"delayed={delayed_gain if delayed_gain is None else round(delayed_gain, 5)}")
        if idx == 0:
            # 写回 + delayed（机械闭环：真实入口行动 → Episode）
            if support_gain is not None:
                from run_v1_target_local_loop import write_target_episode, update_delayed_status
                f_support = resolver.window_context(values, origin, PERIOD)
                ep = write_target_episode(
                    domain=TARGET_DOMAIN, op=CONTROL_OP,
                    program_steps=[{"op": CONTROL_OP,
                                    "params": wiring.contract_params(CONTROL_OP, PERIOD)}],
                    support_gain=support_gain, delayed_gain=None,
                    support_context=f_support)
                if delayed_gain is not None:
                    f_delayed = resolver.window_context(values, origin + HORIZON, PERIOD)
                    ep = update_delayed_status(ep, delayed_gain,
                                               delayed_context=f_delayed)
                memory.append(ep)

    r1 = rounds[0]
    checks: dict[str, bool] = {
        "legal_non_identity_action": (
            r1["chosen"] != "identity"
            and r1["compilation"] in ("ok", "compiled")),
        "verifier_passed": r1["prepared_status"] == "PREPARED",
        "support_positive": r1["support_gain"] is not None
        and r1["support_gain"] >= MATERIAL,
        "delayed_positive": r1["delayed_gain"] is not None
        and r1["delayed_gain"] >= MATERIAL,
        "memory_updated": len(memory) == 2,
        "next_round_retrieves": rounds[1]["instruction_ref1"]
        and rounds[1]["chosen"] == f"cand_{CONTROL_OP}",
    }
    all_pass = all(checks.values())
    verdict = "MECHANICAL_CLOSED_LOOP_PASS" if all_pass else "MECHANICAL_CLOSED_LOOP_PARTIAL"
    print(f"\n== checks: {checks}")
    print(f"== verdict: {verdict}")
    print("== 口径：正控已暴露（机械闭环验证，不宣称 A5 跨域效果）")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-actionable-mechanical-control",
            "control_operator": CONTROL_OP,
            "rounds": rounds,
            "checks": checks,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
