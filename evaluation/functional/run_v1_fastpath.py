"""Harness v1 阶段 B：历史 Episode 能否改善未来切片的等预算探测顺序。

这是零 LLM 的 Memory positive-control，不是自然 Agent/跨域能力证明：

1. 在较早的 Source support/delayed 切片上，对完整 forecast Operator inventory
   生成正向、负向和冲突 Episode；不按 Target outcome 筛选候选。
2. Target Fast Path 只能读取已经完成的 Source Episode。经验 Arm 通过
   SignedEpisodeRetriever 形成先验顺序；无经验 Arm 使用冻结的算子名字顺序。
3. 两个 Arm 使用相同的 Target Support 上限，首个正向即停。计划冻结后才打开
   Target delayed outcome，比较首次正向 probe、harm、abstain 和 delayed AUC。

默认 GEFCom 时间顺序：
  Source support=[832,880), Source delayed=[880,928),
  Target support=[928,976), Target delayed=[976,1024).

用法：
  python evaluation/functional/run_v1_fastpath.py --domain gefcom
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import SelfEvolvingHarnessTS.operators.registry as reg  # noqa: E402
from experience_memory import (  # noqa: E402
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    SignedEpisodeRetriever,
    build_episode,
)
from run_w2_operator_scan import _default_params  # noqa: E402
from skill_acquisition import attach_delayed_outcomes  # noqa: E402

HORIZON = 48
MAX_TARGET_SUPPORT_EVALS = 2
PATTERN_VIEW = "structural"
TASK_CONSUMER_KEY = "forecast|ridge|sMASE"  # 规范 key：task_type|model_class|metric
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_fastpath_report.json")  # 实际输出带 domain 后缀


def extract_F(
    values: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    origin: int,
) -> dict[str, float]:
    """保持当前已开发 structural 视角；本轮只改变 Memory 时间绑定。"""

    period = int(config.get("period", 1))
    all_run_lengths: list[int] = []
    all_acfs: list[float] = []
    all_seasonal: list[float] = []
    for array in values.values():
        window = np.asarray(array[:origin], dtype=np.float64)
        mask = ~np.isfinite(window)
        runs: list[tuple[int, int]] = []
        start = None
        for index, missing in enumerate(mask):
            if missing and start is None:
                start = index
            elif not missing and start is not None:
                runs.append((start, index))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        all_run_lengths.extend(stop - begin for begin, stop in runs)

        if period >= window.size:
            continue
        left = window[:-period]
        right = window[period:]
        valid = np.isfinite(left) & np.isfinite(right)
        left, right = left[valid], right[valid]
        if left.size < 3:
            continue
        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = float(
            np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        )
        if denominator <= 0.0:
            continue
        all_acfs.append(
            float(np.dot(left_centered, right_centered) / denominator)
        )
        observed = window[np.isfinite(window)]
        center = float(np.median(observed)) if observed.size else 0.0
        scale = float(1.4826 * np.median(np.abs(observed - center)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(observed)) if observed.size else 1.0
        if np.isfinite(scale) and scale > 1e-12:
            all_seasonal.append(float(np.median(np.abs(right - left)) / scale))

    return {
        "maximum_missing_run_length": (
            float(max(all_run_lengths)) if all_run_lengths else 0.0
        ),
        "median_acf_at_calendar_period": (
            float(statistics.median(all_acfs)) if all_acfs else 0.0
        ),
        "median_normalized_seasonal_residual": (
            float(statistics.median(all_seasonal)) if all_seasonal else 0.0
        ),
        "bound_period": float(period),
    }


def context_summary(
    F: Mapping[str, float],
    roster: Sequence[Mapping[str, object]],
    params: Mapping[str, object],
) -> dict[str, object]:
    return {
        "cohort": {"series_count": len(roster)},
        "local_pattern": dict(F),
        "program_geometry": {
            "scope": "training_rows",
            "period_binding": "calendar_period",
            **dict(params),
        },
    }


def make_compiled(op: str, params: Mapping[str, object]) -> Any:
    from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
    from SelfEvolvingHarnessTS.contracts.program import Program
    from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import CompiledWorkflow

    program = Program.from_steps([(op, dict(params))], source="v1_fastpath")
    candidate = Candidate(
        candidate_id=f"{op}_v1",
        kind=CandidateKind.PROGRAM,
        program=program,
        source="v1_fastpath",
    )
    return CompiledWorkflow(candidate, (), tuple(program.steps))


def gain_at(
    roster: list[dict[str, object]],
    values: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    compiled: Any,
    origin: int,
    baseline_cache: dict[int, float],
) -> float | None:
    """返回当前 origin 的确定性 Support gain；仪器失败不伪装成负经验。"""

    try:
        if origin not in baseline_cache:
            baseline = v6._evaluate(roster, values, None, config, origin=origin)
            baseline_cache[origin] = float(baseline["mean_smase"])
        candidate = v6._evaluate(roster, values, compiled, config, origin=origin)
    except Exception:
        return None
    return baseline_cache[origin] - float(candidate["mean_smase"])


# 审查⑤统一：material threshold + changes_target_space 排除（与 v6 生成路径一致）
MATERIAL_THRESHOLD = 0.005
CTS_EXCLUDED = tuple(
    n for n in v6.OPERATOR_NAMES
    if v6.OPERATOR_METADATA[n].get("changes_target_space")
)


def relation_of(support_gain: float, delayed_gain: float) -> str:
    if support_gain >= MATERIAL_THRESHOLD and delayed_gain >= MATERIAL_THRESHOLD:
        return RELATION_POSITIVE
    if support_gain < MATERIAL_THRESHOLD and delayed_gain < MATERIAL_THRESHOLD:
        return RELATION_NEGATIVE
    return RELATION_CONFLICT


def build_source_memory(
    *,
    domain: str,
    roster: list[dict[str, object]],
    values: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    operators: Sequence[str],
    source_support_origin: int,
    source_delayed_origin: int,
    baseline_cache: dict[int, float],
    context_fn: object | None = None,
) -> tuple[list[Any], list[dict[str, object]]]:
    period = int(config.get("period", 1))
    F = extract_F(values, config, source_support_origin)
    # signed 半径 resolver（run_v1_signed_radius）：context_fn 提供部署可见
    # recent/change Context（support + delayed 各一），与 target Episode 同一特征族。
    # 缺省 None → 保持历史行为（累计前缀 extract_F，仅 support）。
    support_ctx = dict(context_fn(source_support_origin)) if context_fn is not None else F
    delayed_ctx = dict(context_fn(source_delayed_origin)) if context_fn is not None else None
    episodes: list[Any] = []
    attempts: list[dict[str, object]] = []
    for op in operators:
        params = _default_params(op, period)
        compiled = make_compiled(op, params)
        support_gain = gain_at(
            roster,
            values,
            config,
            compiled,
            source_support_origin,
            baseline_cache,
        )
        delayed_gain = gain_at(
            roster,
            values,
            config,
            compiled,
            source_delayed_origin,
            baseline_cache,
        )
        if support_gain is None or delayed_gain is None:
            attempts.append({"operator": op, "status": "INSTRUMENT_INVALID"})
            continue
        relation = relation_of(support_gain, delayed_gain)
        cs = context_summary(support_ctx, roster, params)
        if delayed_ctx is not None:
            cs["delayed_pattern"] = delayed_ctx
        episode = build_episode(
            episode_id=f"v1_{domain}_source_{source_support_origin}_{op}",
            task_consumer_key=TASK_CONSUMER_KEY,
            domain_namespace=domain,
            context_summary=cs,
            workflow_signature=op,
            support_response={"gain": support_gain},
            delayed_response={"evaluated": True, "gain": delayed_gain},
            relation=relation,
            evidence_level="DELAYED",
            local_status=(
                "LOCAL_ACTIVE" if relation == RELATION_POSITIVE else "EPISODE_ONLY"
            ),
            pattern_view=PATTERN_VIEW,
            evidence_refs=["evaluation/functional/run_v1_fastpath.py"],
        )
        episodes.append(episode)
        attempts.append(
            {
                "operator": op,
                "status": "EXECUTED",
                "support_gain": support_gain,
                "delayed_gain": delayed_gain,
                "relation": relation,
            }
        )
    return episodes, attempts


def compile_experienced_order(
    *,
    episodes: Sequence[Any],
    operators: Sequence[str],
    target_F: Mapping[str, float],
    roster: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    domain: str,
) -> tuple[list[str], list[dict[str, object]]]:
    """逐算子调用现有 Retriever，再用历史 signed response 形成 probe prior。"""

    period = int(config.get("period", 1))
    rows: list[tuple[tuple[float, float, str], str, dict[str, object]]] = []
    relation_rank = {
        RELATION_POSITIVE: 0.0,
        RELATION_CONFLICT: 1.0,
        RELATION_NEGATIVE: 2.0,
    }
    for op in operators:
        params = _default_params(op, period)
        retriever = SignedEpisodeRetriever(
            episodes,
            task_consumer_key=TASK_CONSUMER_KEY,
            allowed_operators=(op,),
            pattern_view=PATTERN_VIEW,
        )
        pack = retriever.retrieve(
            context_summary(target_F, roster, params),
            domain_namespace=domain,
        )
        episode = pack.positive or pack.conflict or pack.negative
        if episode is None:
            rows.append(
                (
                    (3.0, 0.0, op),
                    op,
                    {
                        "operator": op,
                        "episode_id": None,
                        "relation": None,
                        "retrieval_note": pack.retrieval_note,
                    },
                )
            )
            continue
        support_gain = float(episode.support_response["gain"])
        delayed_gain = float(episode.delayed_response["gain"])
        stable_gain = min(support_gain, delayed_gain)
        rows.append(
            (
                (relation_rank[episode.relation], -stable_gain, op),
                op,
                {
                    "operator": op,
                    "episode_id": episode.episode_id,
                    "relation": episode.relation,
                    "source_support_gain": support_gain,
                    "source_delayed_gain": delayed_gain,
                    "retrieval_note": pack.retrieval_note,
                },
            )
        )
    rows.sort(key=lambda row: row[0])
    return [row[1] for row in rows], [row[2] for row in rows]


def plan_target_support(
    *,
    order: Sequence[str],
    roster: list[dict[str, object]],
    values: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    target_support_origin: int,
    baseline_cache: dict[int, float],
) -> dict[str, object]:
    """相同两次 Support 上限；计划阶段不读取 Target delayed。"""

    period = int(config.get("period", 1))
    selected = "IDENTITY"
    terminal = False
    observations: list[dict[str, object]] = []
    planning_trace: list[dict[str, object]] = [
        {
            "budget": 0,
            "selected_workflow": selected,
            "abstained": True,
            "terminal": False,
        }
    ]
    order_index = 0
    harm = 0
    for budget in range(1, MAX_TARGET_SUPPORT_EVALS + 1):
        while not terminal and order_index < len(order):
            op = order[order_index]
            order_index += 1
            compiled = make_compiled(op, _default_params(op, period))
            gain = gain_at(
                roster,
                values,
                config,
                compiled,
                target_support_origin,
                baseline_cache,
            )
            if gain is None:
                continue
            observations.append({"workflow_id": op, "support_gain": gain})
            if gain < 0.0:
                harm += 1
            if gain >= MATERIAL_THRESHOLD:
                selected = op
                terminal = True
            break
        planning_trace.append(
            {
                "budget": budget,
                "selected_workflow": selected,
                "abstained": selected == "IDENTITY",
                "terminal": terminal,
            }
        )
    return {
        "selected_workflow": selected,
        "abstained": selected == "IDENTITY",
        "probed_workflows": [row["workflow_id"] for row in observations],
        "support_observations": observations,
        "support_planning_trace": planning_trace,
        "control": "stop_on_first_positive",
        "first_positive_probe": (
            len(observations) if selected != "IDENTITY" else None
        ),
        "harm_probe_count": harm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V1 chronological local-Memory probe-order measurement"
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom", choices=("gefcom", "nn5"))
    parser.add_argument("--source-origin", type=int, default=None)
    parser.add_argument("--target-origin", type=int, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    max_len = max(int(len(array)) for array in values.values())
    period = int(config.get("period", 1))
    shift = min(period, max_len - int(config["support_origin"]) - 2 * HORIZON)
    default_target = int(config["support_origin"]) + shift
    target_support_origin = (
        args.target_origin if args.target_origin is not None else default_target
    )
    source_support_origin = (
        args.source_origin
        if args.source_origin is not None
        else target_support_origin - 2 * HORIZON
    )
    source_delayed_origin = source_support_origin + HORIZON
    target_delayed_origin = target_support_origin + HORIZON
    if source_support_origin < HORIZON:
        raise SystemExit("source origin leaves insufficient visible history")
    if source_delayed_origin + HORIZON > target_support_origin:
        raise SystemExit("Source delayed outcome overlaps Target Support")
    if target_delayed_origin + HORIZON > max_len:
        raise SystemExit("Target delayed segment exceeds available development data")

    operators = sorted(
        name
        for name in reg.OPERATOR_NAMES
        if "forecast"
        in (reg.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
    )
    baseline_cache: dict[int, float] = {}
    source_F = extract_F(values, config, source_support_origin)
    target_F = extract_F(values, config, target_support_origin)
    print(
        f"== {domain}: source=[{source_support_origin},{source_delayed_origin + HORIZON}) "
        f"target_support=[{target_support_origin},{target_delayed_origin}) "
        f"target_delayed=[{target_delayed_origin},{target_delayed_origin + HORIZON})"
    )
    print(f"== common candidate inventory: {len(operators)} operators")

    source_episodes, source_attempts = build_source_memory(
        domain=domain,
        roster=roster,
        values=values,
        config=config,
        operators=operators,
        source_support_origin=source_support_origin,
        source_delayed_origin=source_delayed_origin,
        baseline_cache=baseline_cache,
    )
    experienced_order, retrieval_trace = compile_experienced_order(
        episodes=source_episodes,
        operators=operators,
        target_F=target_F,
        roster=roster,
        config=config,
        domain=domain,
    )
    fixed_order = list(operators)
    print(f"== experienced first five: {experienced_order[:5]}")
    print(f"== fixed first five:       {fixed_order[:5]}")

    experienced_plan = plan_target_support(
        order=experienced_order,
        roster=roster,
        values=values,
        config=config,
        target_support_origin=target_support_origin,
        baseline_cache=baseline_cache,
    )
    fixed_plan = plan_target_support(
        order=fixed_order,
        roster=roster,
        values=values,
        config=config,
        target_support_origin=target_support_origin,
        baseline_cache=baseline_cache,
    )

    # 两个 Support 计划均冻结后，才打开其已探测 Workflow 的 Target delayed。
    probed = sorted(
        set(experienced_plan["probed_workflows"])
        | set(fixed_plan["probed_workflows"])
    )
    delayed_gains: dict[str, float] = {}
    for op in probed:
        compiled = make_compiled(op, _default_params(op, period))
        gain = gain_at(
            roster,
            values,
            config,
            compiled,
            target_delayed_origin,
            baseline_cache,
        )
        if gain is None:
            raise RuntimeError(f"Target delayed evaluation failed for {op}")
        delayed_gains[op] = gain

    experienced_result = attach_delayed_outcomes(experienced_plan, delayed_gains)
    fixed_result = attach_delayed_outcomes(fixed_plan, delayed_gains)
    experienced_result["first_positive_probe"] = experienced_plan[
        "first_positive_probe"
    ]
    experienced_result["harm_probe_count"] = experienced_plan["harm_probe_count"]
    fixed_result["first_positive_probe"] = fixed_plan["first_positive_probe"]
    fixed_result["harm_probe_count"] = fixed_plan["harm_probe_count"]

    exp_first = experienced_result["first_positive_probe"]
    fixed_first = fixed_result["first_positive_probe"]
    exp_auc = float(experienced_result["adaptation_auc"])
    fixed_auc = float(fixed_result["adaptation_auc"])
    exp_harm = int(experienced_result["harm_probe_count"])
    fixed_harm = int(fixed_result["harm_probe_count"])
    if (
        exp_first is not None
        and (fixed_first is None or int(exp_first) < int(fixed_first) or exp_harm < fixed_harm)
        and exp_auc >= fixed_auc
    ):
        verdict = "DEVELOPMENT_EXPERIENCE_VALUE_SUPPORTED"
    elif (
        experienced_result["probed_workflows"] == fixed_result["probed_workflows"]
        and exp_auc == fixed_auc
    ):
        verdict = "NO_NONTRIVIAL_BEHAVIOR_DIFFERENCE"
    elif exp_auc < fixed_auc or exp_harm > fixed_harm:
        verdict = "DEVELOPMENT_EXPERIENCE_VALUE_NOT_SUPPORTED"
    else:
        verdict = "MIXED_NO_STAGE_B_PASS"

    relation_counts = {
        relation: sum(1 for ep in source_episodes if ep.relation == relation)
        for relation in (
            RELATION_POSITIVE,
            RELATION_NEGATIVE,
            RELATION_CONFLICT,
        )
    }
    report = {
        "experiment_id": "v1-fastpath-chronological-local-memory",
        "evaluation_role": "EXPOSED_DEVELOPMENT_ZERO_LLM_MECHANISM",
        "domain": domain,
        "timeline": {
            "source_support_origin": source_support_origin,
            "source_delayed_origin": source_delayed_origin,
            "target_support_origin": target_support_origin,
            "target_delayed_origin": target_delayed_origin,
            "future_read_during_target_planning": False,
        },
        "pattern_view": PATTERN_VIEW,
        "source_F": {key: round(value, 6) for key, value in source_F.items()},
        "target_F": {key: round(value, 6) for key, value in target_F.items()},
        "common_candidate_inventory": operators,
        "target_support_budget_per_arm": MAX_TARGET_SUPPORT_EVALS,
        "source_episode_count": len(source_episodes),
        "source_relation_counts": relation_counts,
        "source_attempts": source_attempts,
        "retrieval_trace": retrieval_trace,
        "experienced_order": experienced_order,
        "fixed_order": fixed_order,
        "experienced": experienced_result,
        "fixed": fixed_result,
        "target_delayed_gains_for_probed_workflows": delayed_gains,
        "verdict": verdict,
        "claim_ceiling": (
            "Same-domain exposed-development Memory probe-order mechanism only; "
            "not cross-domain transfer, not natural Agent generation, and not fresh evidence."
        ),
        "llm_api_call_count": 0,
    }
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"== experienced probes={experienced_result['probed_workflows']} "
        f"first={exp_first} harm={exp_harm} delayed_auc={exp_auc:+.6f}"
    )
    print(
        f"== fixed probes={fixed_result['probed_workflows']} "
        f"first={fixed_first} harm={fixed_harm} delayed_auc={fixed_auc:+.6f}"
    )
    print(f"== verdict: {verdict}")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
