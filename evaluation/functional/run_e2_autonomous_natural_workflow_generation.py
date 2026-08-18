"""Exposed NN5 Development run for catalog-free autonomous Workflow generation.

This experiment tests only the GENERATIVE_SUPPLY_GAP.  Two LLM calls receive a
public Context and the complete canonical operator inventory; Support feedback
may revise the full Program once.  A generated Program prepares training
windows only, after which the retrained Consumer is evaluated on untouched
deployment contexts.  A frozen accepted Program is then evaluated on one later
temporal selection slice.  The run cannot promote or write Memory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes, canonical_sha256
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_cross_series_curation import (
    _center_scale,
    _exact_weighted_ridge_prediction,
    _missing_runs,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    _read_object,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import AgentRole, TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CompiledWorkflow,
    build_public_operator_inventory,
    compile_workflow_proposal,
    resolve_generated_acquisition_lifecycle,
    run_two_round_generation,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    CohortHistoryPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import EffectiveHarnessView
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgictoChatCompletionsBackend,
    AgentBackend,
    AgentRequest,
    AgentResponse,
    BudgetedAgentBackend,
    ReplayAgentBackend,
)


CONTEXT_LENGTH = 192
HORIZON = 48
TRAIN_SERIES_COUNT = 12
EVAL_SERIES_COUNT = 8
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_SLOW_MODEL = "gpt-5.5"
DEFAULT_BASE_URL = "https://api.agicto.cn/v1"
INDUCTION_REPORT_PATH = Path(
    "artifacts/functional/e2/autonomous_natural_workflow_contrast_induction_report.json"
)
INDUCTION_SOURCE_REPORTS = (
    Path("artifacts/functional/e2/autonomous_natural_workflow_generation_exploration_v2_report.json"),
    Path("artifacts/functional/e2/autonomous_natural_workflow_generation_gefcom_report.json"),
)
SCOPE_INDUCTION_REPORT_PATH = Path(
    "artifacts/functional/e2/autonomous_natural_workflow_scope_induction_v1_report.json"
)
AUTONOMOUS_CYCLE_REPORT_PATH = Path(
    "artifacts/functional/e2/autonomous_natural_acquisition_cycle_v1_report.json"
)
SCOPE_SOURCE_REPORTS = (
    Path("artifacts/functional/e2/autonomous_natural_workflow_generation_nn5_training_only_v2_report.json"),
    Path("artifacts/functional/e2/autonomous_natural_workflow_generation_gefcom_training_only_v1_report.json"),
)
DATASET_CONFIGS: dict[str, dict[str, object]] = {
    "nn5": {
        "dataset_id": "monash:nn5_daily",
        "sampling": "daily_regular",
        "period": 7,
        "anchors": (240, 300, 360, 420, 480, 540),
        "support_origin": 632,
        "selection_origin": 680,
        "report_path": "artifacts/functional/e2/autonomous_natural_workflow_generation_nn5_report.json",
    },
    "gefcom": {
        "dataset_id": "gefcom2012_load",
        "sampling": "hourly_regular",
        "period": 24,
        "anchors": (312, 372, 432, 492, 552, 612, 672, 732, 792, 852),
        "support_origin": 912,
        "selection_origin": 960,
        "report_path": "artifacts/functional/e2/autonomous_natural_workflow_generation_gefcom_report.json",
    },
    "noaa": {
        "dataset_id": "noaa_global_hourly",
        "sampling": "hourly_regular",
        "period": 24,
        "anchors": (240, 300, 360, 420, 480, 540, 600, 660),
        "support_origin": 720,
        "selection_origin": 768,
        "report_path": "artifacts/functional/e2/autonomous_natural_workflow_generation_noaa_report.json",
    },
}

Proposer = Callable[[Mapping[str, object]], Mapping[str, object]]


def _plain_public(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_public(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_public(nested) for nested in value]
    return value


def _empty_observation_harness_view() -> EffectiveHarnessView:
    instruction = (
        "Inspect deployment-visible time-series history before Program generation. "
        "Call the one allowed public observation tool exactly once, then return the "
        "required stage result. Do not select, name, rank, or recommend an operator, "
        "Workflow, Skill, threshold, or promotion decision."
    )
    controls = {
        "role": "fast",
        "observation_only": True,
        "required_tool": "compare_history_windows",
        "required_tool_call_count": 1,
    }
    return EffectiveHarnessView(
        instruction=instruction,
        skills=(),
        memories=(),
        controls=controls,
        effective_harness_view_sha=canonical_sha256(
            {
                "schema_version": "effective-harness-view/1",
                "instruction": instruction,
                "skills": [],
                "memories": [],
                "controls": controls,
            }
        ),
    )


class _RequiredHistoryToolBackend:
    """Give this observation stage one correction for a skipped required tool."""

    def __init__(self, delegate: AgentBackend):
        self.delegate = delegate
        self.correction_sent = False
        self.required_tool_executed = False

    def complete(self, request: AgentRequest) -> AgentResponse:
        try:
            latest_message = json.loads(str(request.messages[-1]["content"]))
        except (KeyError, TypeError, ValueError):
            latest_message = None
        if (
            isinstance(latest_message, Mapping)
            and latest_message.get("schema_version") == "tool-result/1"
            and latest_message.get("tool_name") == "compare_history_windows"
        ):
            self.required_tool_executed = True
        effective_request = (
            replace(request, call_index=request.call_index + 1)
            if self.correction_sent
            else request
        )
        response = self.delegate.complete(effective_request)
        envelope = response.parsed_envelope
        if (
            self.correction_sent
            or self.required_tool_executed
            or response.parse_status != "VALID_AGENT_ENVELOPE"
            or envelope is None
            or envelope.get("kind") != "stage_result"
        ):
            return response
        correction = {
            "schema_version": "stage-validation-error/2",
            "stage": "observe",
            "error_code": "REQUIRED_TOOL_MISSING",
            "public_message": (
                "The stage_result was returned before compare_history_windows was "
                "executed. The next envelope must be a tool_request for the missing "
                "required tool compare_history_windows."
            ),
            "required_outer_format": (
                '{"schema_version":"agent-envelope/1",'
                '"kind":"tool_request","call_id":"...",'
                '"tool_name":"compare_history_windows","arguments":{}}'
            ),
            "instruction": (
                "Return exactly one agent-envelope/1 tool_request for "
                "compare_history_windows; do not return a stage_result or explain "
                "the correction."
            ),
        }
        correction_request = replace(
            effective_request,
            call_index=effective_request.call_index + 1,
            messages=(
                *effective_request.messages,
                {"role": "assistant", "content": response.assistant_text},
                {
                    "role": "user",
                    "content": canonical_json_bytes(correction).decode("utf-8"),
                },
            ),
        )
        self.correction_sent = True
        return self.delegate.complete(correction_request)


def _augment_context_with_history_observation(
    public_context: Mapping[str, object],
    series_prefixes: Sequence[object],
    *,
    calendar_period: int,
    backend: AgentBackend,
    model: str,
    base_url: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute the existing agent-envelope tool loop and merge its public result."""

    gateway = CohortHistoryPublicToolGateway(
        series_prefixes,
        calendar_period=calendar_period,
        window_length=CONTEXT_LENGTH,
    )
    # The normal path is two calls. Two bounded corrections may coexist: one
    # malformed-envelope retry owned by the core and one skipped-tool retry here.
    budgeted = BudgetedAgentBackend(backend, maximum_calls=4)
    core = TTHAAgentCore(
        _RequiredHistoryToolBackend(budgeted),
        gateway,
        model=model,
        base_url=base_url,
    )
    result = core.run_stage(
        role=AgentRole.FAST,
        stage="observe",
        case_id="autonomous-natural-history-observation",
        public_input={"public_context": copy.deepcopy(dict(public_context))},
        harness_view=_empty_observation_harness_view(),
        output_schema_name="history_observation_complete_v1",
        output_schema={
            "type": "object",
            "properties": {"observation_complete": {"const": True}},
            "required": ["observation_complete"],
            "additionalProperties": False,
        },
        source_snapshot_sha="",
        validation_retries=1,
    )
    if result.payload.get("observation_complete") is not True:
        raise RuntimeError("observation stage did not confirm completion")
    if (
        len(result.tool_receipts) != 1
        or result.tool_receipts[0].tool_name != "compare_history_windows"
    ):
        raise RuntimeError("observation stage must execute compare_history_windows exactly once")
    augmented = copy.deepcopy(dict(public_context))
    observations = augmented.setdefault("observations", {})
    if not isinstance(observations, dict):
        raise ValueError("public Context observations field must be an object")
    observations["history_window_comparison"] = _plain_public(
        result.tool_receipts[0].public_result
    )
    metadata = {
        "enabled": True,
        "tool_name": "compare_history_windows",
        "agent_call_count": budgeted.calls,
        "tool_call_count": len(result.tool_receipts),
        "prompt_tokens": budgeted.prompt_tokens,
        "completion_tokens": budgeted.completion_tokens,
        "returned_models": sorted(budgeted.returned_models),
        "result_merged_at": "public_context.observations.history_window_comparison",
    }
    return augmented, metadata


class LiveJSONProposer:
    """Two-call OpenAI-compatible JSON adapter with no Workflow examples."""

    def __init__(self, *, api_key: str, model: str, base_url: str):
        if not api_key.strip():
            raise ValueError("API key is required")
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.model = model
        self.base_url = base_url
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    def __call__(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        stage = str(payload.get("stage", ""))
        system = (
            "You generate a typed time-series data-preparation Workflow. Read the "
            "complete operator inventory, including runtime effects, parameters and "
            "availability. Return exactly one JSON object matching workflow_schema, "
            "with one to four steps and IDENTITY fallback. Use only EXECUTABLE "
            "canonical operators. Do not invent a catalog, skill, dataset identity, "
            "Query result, clean truth, or promotion claim. On REVISION, use only the "
            "initial Action--Response trace and return a complete replacement Program, "
            "not a patch. If the initial candidate has a compilation error, zero "
            "behavior relative to baseline_integrity, or non-positive Support while "
            "an executable operator remains untried, the replacement must have a "
            "different Program AST and must be intended to produce non-zero behavior "
            "relative to baseline_integrity. ABSTAIN only for an explained legal risk "
            "or when no executable alternative remains. No markdown or commentary."
        )
        # Memory presentation（TIMECLAW 模式，2026-08-07）：pack 存在时把对照包
        # 程序化渲染成 fenced 前缀块（只含方向、不含数值——防锚定），
        # 放在任务指令之前；结构化 JSON 原样保留在 payload。
        if isinstance(payload.get("experience_contrast_pack"), dict):
            try:
                from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
                    render_experience_pack,
                )
                rendered = render_experience_pack(payload["experience_contrast_pack"])
            except Exception:  # 渲染失败不阻塞（回退弱契约已移除；pack 仍在 payload）
                rendered = ""
            if rendered:
                system = rendered + system
        self.last_system_prefix = system[:220]  # 调试：记录实际 system prompt 前缀
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                },
            ],
        )
        choice = completion.choices[0]
        proposal = json.loads(choice.message.content or "")
        if not isinstance(proposal, dict):
            raise ValueError("LLM proposer must return one JSON object")
        self.call_count += 1
        usage = getattr(completion, "usage", None)
        self.calls.append(
            {
                "stage": stage,
                "returned_model": str(getattr(completion, "model", "")),
                "finish_reason": str(getattr(choice, "finish_reason", "")),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        )
        return proposal


class LiveScopeProposer:
    """A bounded Slow-Path proposer for a typed applicability patch."""

    def __init__(self, *, api_key: str, model: str, base_url: str):
        if not api_key.strip():
            raise ValueError("API key is required")
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.model = model
        self.base_url = base_url
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    def __call__(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        is_revision = "original_typed_patch" in payload
        if is_revision:
            system = (
                "You are revising one typed applicability Scope for a self-evolving "
                "time-series Harness. The input contains the original anonymized Scope "
                "dossier, the original typed patch, and anonymous compact full-policy "
                "Support/Selection replay under a frozen risk gate. Return exactly one "
                "complete replacement JSON object matching required_output_json_schema: "
                "either RESTRICT_SCOPE or ABSTAIN. Use only schema-listed public numeric "
                "fields and one to three AND conditions. Do not change the Program, gate, "
                "compiler, or infer dataset identity. Do not claim promotion or Query "
                "evidence. No markdown or commentary."
            )
        else:
            system = (
                "You are the Slow Path of a self-evolving time-series Harness. The input "
                "contains anonymized, deployment-visible per-series history summaries and "
                "exact singleton Support responses for one Program discovered by an earlier "
                "LLM trace. Decide whether those observations support one portable typed "
                "RESTRICT_SCOPE patch. Return exactly one JSON object matching "
                "required_output_json_schema. Use only listed public numeric fields, one to "
                "three AND conditions, and no dataset identity. Singleton responses are "
                "proposal credit only: do not approve a Skill, claim policy utility, invent "
                "an operator, change the Program, or use later Selection/Query outcomes. "
                "Return ABSTAIN when no defensible deployment-visible predicate exists. No "
                "markdown or commentary."
            )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                },
            ],
            response_format={"type": "json_object"},
        )
        choice = completion.choices[0]
        proposal = json.loads(choice.message.content or "")
        if not isinstance(proposal, dict):
            raise ValueError("scope proposer must return one JSON object")
        self.call_count += 1
        usage = getattr(completion, "usage", None)
        self.calls.append(
            {
                "stage": "REVISION" if is_revision else "INITIAL",
                "returned_model": str(getattr(completion, "model", "")),
                "finish_reason": str(getattr(choice, "finish_reason", "")),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        )
        return proposal


class LiveBindingProposer:
    """One authorized Slow-Path call for a one-parameter public binding patch."""

    def __init__(self, *, api_key: str, model: str, base_url: str):
        if not api_key.strip():
            raise ValueError("API key is required")
        if model != DEFAULT_SLOW_MODEL:
            raise ValueError("Binding Slow Path is frozen to gpt-5.5")
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.model = model
        self.base_url = base_url
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    def __call__(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        system = (
            "You are the Binding Slow Path of a self-evolving time-series Harness. "
            "The input contains only anonymous environments, one common operator, "
            "the unique current template and compiled parameters for that operator, "
            "and allowed deployment-visible finite scalar Context fields. Return "
            "exactly one JSON object matching required_output_json_schema. You may "
            "only PATCH_BINDING for one currently static parameter to one listed "
            "public_context scalar path, or ABSTAIN. Do not change the operator, "
            "Scope, any other parameter, or infer dataset identity. No NOAA, future, "
            "Query, Selection, or outcome information is available. No markdown or "
            "commentary."
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
                },
            ],
            response_format={"type": "json_object"},
        )
        choice = completion.choices[0]
        proposal = json.loads(choice.message.content or "")
        if not isinstance(proposal, dict):
            raise ValueError("binding proposer must return one JSON object")
        self.call_count += 1
        usage = getattr(completion, "usage", None)
        self.calls.append(
            {
                "returned_model": str(getattr(completion, "model", "")),
                "finish_reason": str(getattr(choice, "finish_reason", "")),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        )
        return proposal


def _fixed_roster(
    root: Path, config: Mapping[str, object]
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    registry_path = root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_length = int(config["selection_origin"]) + HORIZON
    eligible = sorted(
        (
            row
            for row in rows
            if row.get("dataset_id") == config["dataset_id"]
            and int(row.get("length", 0)) >= required_length
        ),
        key=lambda row: str(row["series_uid"]),
    )
    selected = eligible[: TRAIN_SERIES_COUNT + EVAL_SERIES_COUNT]
    if len(selected) != TRAIN_SERIES_COUNT + EVAL_SERIES_COUNT:
        raise ValueError("fixed Development roster does not contain twenty eligible series")

    record_dirs: dict[str, Path] = {}
    wanted = {str(row["series_uid"]) for row in selected}
    for record_path in (root / "data/benchmark_v0_2/clean_base").glob("*/record.json"):
        record = _read_object(record_path)
        uid = str(record.get("series_uid", ""))
        if uid in wanted:
            record_dirs[uid] = record_path.parent
    if set(record_dirs) != wanted:
        raise ValueError("fixed Development roster is missing clean_base records")

    import numpy as np

    values = {
        uid: np.asarray(
            np.load(directory / "values.npy", allow_pickle=False), dtype=np.float64
        )
        for uid, directory in record_dirs.items()
    }
    roster = [
        {
            "series_uid": str(row["series_uid"]),
            "role": "train" if index < TRAIN_SERIES_COUNT else "eval",
        }
        for index, row in enumerate(selected)
    ]
    return roster, values


def _linear_integrity(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64).ravel()
    observed = np.flatnonzero(np.isfinite(array))
    if observed.size < 2:
        raise ValueError("minimal integrity handling requires two observed values")
    return np.interp(np.arange(array.size), observed, array[observed])


def _acf(values: Any, lag: int) -> float:
    import numpy as np

    array = _linear_integrity(values)
    left = array[:-lag] - float(np.mean(array[:-lag]))
    right = array[lag:] - float(np.mean(array[lag:]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 0.0 else float(np.dot(left, right) / denominator)


def _public_context(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
) -> dict[str, object]:
    import numpy as np

    prefixes = [
        np.asarray(
            values[str(row["series_uid"])][: int(config["support_origin"])],
            dtype=np.float64,
        )
        for row in roster
    ]
    masks = [~np.isfinite(array) for array in prefixes]
    runs = [run for mask in masks for run in _missing_runs(mask)]
    run_lengths = [stop - start for start, stop in runs]
    acf_by_lag = {
        lag: statistics.median(_acf(array, lag) for array in prefixes)
        for lag in range(2, 49)
    }
    dominant_lag = max(acf_by_lag, key=lambda lag: (acf_by_lag[lag], -lag))
    centers: list[float] = []
    scales: list[float] = []
    outlier_rates: list[float] = []
    for array in prefixes:
        center, scale, _method = _center_scale(np, array)
        observed = array[np.isfinite(array)]
        centers.append(center)
        scales.append(scale)
        outlier_rates.append(float(np.mean(np.abs((observed - center) / scale) > 6.0)))
    total_points = sum(array.size for array in prefixes)
    missing_points = sum(int(mask.sum()) for mask in masks)
    return {
        "task": {
            "type": "forecast",
            "sampling": str(config["sampling"]),
            "horizon": HORIZON,
            "context_length": CONTEXT_LENGTH,
        },
        "program_application_scope": "training_windows_only",
        "consumer": {
            "model": "ridge_alpha_1_with_intercept",
            "metric": "mean_per_series_smase",
            "seasonal_scale_period": int(config["period"]),
            "baseline_integrity": "linear_fill_only",
        },
        "cohort": {
            "series_count": len(roster),
            "training_series_count": TRAIN_SERIES_COUNT,
            "evaluation_series_count": EVAL_SERIES_COUNT,
            "training_window_count": TRAIN_SERIES_COUNT * len(config["anchors"]),  # type: ignore[arg-type]
        },
        "missingness": {
            "point_fraction": float(missing_points / total_points),
            "run_count": len(runs),
            "median_run_length": float(statistics.median(run_lengths)) if run_lengths else 0.0,
            "maximum_run_length": max(run_lengths, default=0),
            "affected_series_fraction": float(sum(bool(mask.any()) for mask in masks) / len(masks)),
        },
        "periodicity": {
            "dominant_lag": int(dominant_lag),
            "median_acf_at_dominant_lag": float(acf_by_lag[dominant_lag]),
            "calendar_period": int(config["period"]),
        },
        "scale": {
            "median_center": float(statistics.median(centers)),
            "median_robust_scale": float(statistics.median(scales)),
            "minimum_robust_scale": float(min(scales)),
            "maximum_robust_scale": float(max(scales)),
        },
        "outliers": {
            "median_robust_z_gt_6_fraction": float(statistics.median(outlier_rates)),
            "maximum_robust_z_gt_6_fraction": float(max(outlier_rates)),
        },
        "capability_memory": {"entry_count": 0},
    }


def _apply_program(raw: Any, compiled: CompiledWorkflow | None) -> tuple[Any, list[dict[str, object]]]:
    import numpy as np

    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

    array = np.asarray(raw, dtype=np.float64).ravel()
    if compiled is None:
        return _linear_integrity(array), []
    assert compiled.candidate.program is not None
    execution = run_pipeline(
        compiled.candidate.program.execution_steps(), array, source="llm_generated"
    )
    if not execution.ok or execution.artifact is None:
        raise RuntimeError(execution.error or "Pipeline execution failed")
    prepared = np.asarray(execution.artifact, dtype=np.float64).ravel()
    if prepared.shape != array.shape:
        raise RuntimeError("Pipeline changed series shape")
    prepared = _linear_integrity(prepared)
    if not np.isfinite(prepared).all():
        raise RuntimeError("Pipeline produced non-finite values after integrity handling")
    return prepared, copy.deepcopy(execution.trace)


def _evaluate(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    compiled: CompiledWorkflow | None,
    config: Mapping[str, object],
    *,
    origin: int,
    train_series_scope: set[str] | frozenset[str] | None = None,
) -> dict[str, object]:
    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale, smase

    train_rows = [row for row in roster if row["role"] == "train"]
    eval_rows = [row for row in roster if row["role"] == "eval"]
    x_train: list[Any] = []
    y_train: list[Any] = []
    behavior_count = 0
    execution_steps: list[dict[str, object]] = []
    for row in train_rows:
        series_uid = str(row["series_uid"])
        raw = np.asarray(values[series_uid], dtype=np.float64)
        for anchor in config["anchors"]:  # type: ignore[union-attr]
            anchor = int(anchor)
            # Reviewer HIGH-1 修复：训练窗口 target 不得覆盖评估 origin 之后的
            # horizon——anchor + HORIZON > origin 的行会让训练标签包含 eval truth
            # （lookahead 泄漏）。对原始 v6 配置（912/960、632/680）行为不变。
            if anchor + HORIZON > origin:
                continue
            window = raw[anchor - CONTEXT_LENGTH : anchor + HORIZON]
            baseline = _linear_integrity(window)
            if compiled is not None and (
                train_series_scope is None or series_uid in train_series_scope
            ):
                prepared, trace = _apply_program(window, compiled)
            else:
                prepared, trace = baseline, []
            behavior_count += int(np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True)))
            execution_steps.extend(trace)
            context = prepared[:CONTEXT_LENGTH]
            target = prepared[CONTEXT_LENGTH:]
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                raise RuntimeError("training context reached scale floor")
            x_train.append((context - center) / scale)
            y_train.append((target - center) / scale)

    x_eval: list[Any] = []
    truths: list[Any] = []
    eval_centers: list[float] = []
    eval_scales: list[float] = []
    metric_scales: list[float] = []
    for row in eval_rows:
        raw = np.asarray(values[str(row["series_uid"])], dtype=np.float64)
        window = raw[origin - CONTEXT_LENGTH : origin]
        prepared = _linear_integrity(window)
        center, scale, method = _center_scale(np, prepared)
        if method == "scale_floor_fallback":
            raise RuntimeError("evaluation context reached scale floor")
        x_eval.append((prepared - center) / scale)
        truths.append(raw[origin : origin + HORIZON])
        eval_centers.append(center)
        eval_scales.append(scale)
        metric_scales.append(
            seasonal_scale(
                raw[:origin],
                np.isfinite(raw[:origin]),
                period=int(config["period"]),
                min_pairs=32,
            )
        )

    prediction = _exact_weighted_ridge_prediction(
        np,
        x_train=np.asarray(x_train, dtype=np.float64),
        targets=np.asarray(y_train, dtype=np.float64),
        weights=np.ones(len(x_train), dtype=np.float64),
        x_eval=np.asarray(x_eval, dtype=np.float64),
    )
    prediction = prediction * np.asarray(eval_scales)[:, None] + np.asarray(eval_centers)[:, None]
    losses: list[float] = []
    for truth, predicted, scale in zip(truths, prediction, metric_scales):
        observed = np.isfinite(truth)
        if not observed.any():
            raise RuntimeError("evaluation future contains no observed truth")
        losses.append(smase(truth[observed], predicted[observed], scale=scale))
    failed_steps = [row for row in execution_steps if row.get("ok") is not True]
    return {
        "mean_smase": float(statistics.fmean(losses)),
        "per_view_smase": [float(value) for value in losses],
        "behavior_point_count": behavior_count,
        "execution_step_count": len(execution_steps),
        "failed_step_count": len(failed_steps),
    }


def _support_callback(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
) -> Callable[[CompiledWorkflow], Mapping[str, object]]:
    support_origin = int(config["support_origin"])
    baseline = _evaluate(roster, values, None, config, origin=support_origin)

    def evaluate(compiled: CompiledWorkflow) -> Mapping[str, object]:
        try:
            candidate = _evaluate(
                roster, values, compiled, config, origin=support_origin
            )
        except Exception as exc:
            return {
                "accepted": False,
                "support_gain": None,
                "per_view_gain": [],
                "behavior": "EXECUTION_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        gains = [
            float(reference - method)
            for reference, method in zip(
                baseline["per_view_smase"], candidate["per_view_smase"]
            )
        ]
        support_gain = float(baseline["mean_smase"] - candidate["mean_smase"])
        behavior_count = int(candidate["behavior_point_count"])
        return {
            "accepted": bool(support_gain > 0.0 and behavior_count > 0),
            "support_gain": support_gain,
            "per_view_gain": gains,
            "positive_view_fraction": float(sum(value > 0.0 for value in gains) / len(gains)),
            "behavior": "CHANGED_INPUT" if behavior_count else "ZERO_BEHAVIOR",
            "behavior_point_count": behavior_count,
            "execution_step_count": int(candidate["execution_step_count"]),
            "error": None,
        }

    return evaluate


def _proposal_rows(trace: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in trace:
        action = item.get("action", {})
        if not isinstance(action, Mapping):
            continue
        rows.append(
            {
                "stage": item.get("stage"),
                "candidate_id": item.get("candidate_id"),
                "workflow_steps": copy.deepcopy(
                    action.get("workflow_steps", action.get("proposed_workflow"))
                ),
                "compiled_program_steps": copy.deepcopy(action.get("program_steps")),
                "requested_observations": copy.deepcopy(action.get("requested_observations")),
                "experience_use": copy.deepcopy(action.get("experience_use")),
                "support_response": copy.deepcopy(item.get("support_response")),
            }
        )
    return rows


def run(
    root: Path,
    *,
    initial_proposer: Proposer | None = None,
    revision_proposer: Proposer | None = None,
    observation_backend: AgentBackend | None = None,
    observe_history: bool = False,
    dataset_key: str = "nn5",
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    report_path: Path | None = None,
    write_report: bool = True,
) -> dict[str, object]:
    root = Path(root)
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"unknown exposed Development dataset key: {dataset_key}")
    config = DATASET_CONFIGS[dataset_key]
    roster, values = _fixed_roster(root, config)
    public_context = _public_context(roster, values, config)

    observation_metadata: dict[str, object] | None = None
    if observe_history:
        if observation_backend is None:
            observation_api_key = (
                os.environ.get("OPENAI_API_KEY", "").strip()
                or os.environ.get("AGICTO_API_KEY", "").strip()
            )
            if not observation_api_key:
                raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
            observation_backend = AgictoChatCompletionsBackend(
                api_key=observation_api_key,
                base_url=base_url,
                timeout_seconds=120,
            )
        prefixes = [
            values[str(row["series_uid"])][: int(config["support_origin"])]
            for row in roster
        ]
        public_context, observation_metadata = _augment_context_with_history_observation(
            public_context,
            prefixes,
            calendar_period=int(config["period"]),
            backend=observation_backend,
            model=model,
            base_url=base_url,
        )

    live: LiveJSONProposer | None = None
    if initial_proposer is None or revision_proposer is None:
        if initial_proposer is not None or revision_proposer is not None:
            raise ValueError("initial and revision proposers must be supplied together")
        api_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("AGICTO_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
        live = LiveJSONProposer(api_key=api_key, model=model, base_url=base_url)
        initial_proposer = revision_proposer = live

    cycle = run_two_round_generation(
        public_context,
        "forecast",
        initial_proposer,
        revision_proposer,
        _support_callback(roster, values, config),
        capability_memory=(),
        forbidden_operators=tuple(
            name
            for name in OPERATOR_NAMES
            if OPERATOR_METADATA[name].get("changes_target_space") is True
        ),
    )
    trace = cycle["action_response_trace"]
    assert isinstance(trace, Sequence)
    selection: dict[str, object] | None = None
    final_status = "REJECTED"
    if cycle["status"] == "CANDIDATE":
        final_candidate = cycle["final_candidate"]
        selected = CompiledWorkflow(
            final_candidate,
            tuple(cycle["skill_draft"]["requested_observations"]),
            tuple(cycle["skill_draft"]["program_template"]["steps"]),
        )
        selection_origin = int(config["selection_origin"])
        baseline = _evaluate(
            roster, values, None, config, origin=selection_origin
        )
        try:
            candidate = _evaluate(
                roster, values, selected, config, origin=selection_origin
            )
            gains = [
                float(reference - method)
                for reference, method in zip(
                    baseline["per_view_smase"], candidate["per_view_smase"]
                )
            ]
            gain = float(baseline["mean_smase"] - candidate["mean_smase"])
            selection = {
                "opened_after_program_freeze": True,
                "baseline_mean_smase": baseline["mean_smase"],
                "candidate_mean_smase": candidate["mean_smase"],
                "selection_gain": gain,
                "per_view_gain": gains,
                "behavior_point_count": candidate["behavior_point_count"],
                "error": None,
            }
            if gain > 0.0 and int(candidate["behavior_point_count"]) > 0:
                final_status = "SOURCE_CANDIDATE"
        except Exception as exc:
            selection = {
                "opened_after_program_freeze": True,
                "selection_gain": None,
                "per_view_gain": [],
                "behavior_point_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    inventory = cycle["operator_inventory"]
    llm_report: dict[str, object] = {
        "api_integrated": live is not None,
        "api_call_count": live.call_count if live is not None else 0,
        "requested_model": model if live is not None else None,
        "base_url": base_url if live is not None else None,
        "calls": copy.deepcopy(live.calls) if live is not None else [],
    }
    if observation_metadata is not None:
        generation_call_count = live.call_count if live is not None else 0
        observation_call_count = int(observation_metadata["agent_call_count"])
        llm_report.update(
            {
                "api_call_count": generation_call_count + observation_call_count,
                "observation_api_call_count": observation_call_count,
                "generation_api_call_count": generation_call_count,
                "observation": copy.deepcopy(observation_metadata),
                "generation_calls": copy.deepcopy(live.calls) if live is not None else [],
            }
        )

    report = {
        "experiment_id": "E2-autonomous-natural-workflow-generation",
        "scientific_role": "exposed_development_generative_supply_mechanism",
        "substrate": {
            "dataset_key": dataset_key,
            "dataset_id": config["dataset_id"],
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "fixed_roster_rule": "registry eligible by length, sort series_uid, first 12 train + next 8 eval",
            "roster": roster,
            "support_origin": config["support_origin"],
            "selection_origin": config["selection_origin"],
        },
        "public_context_sent_to_llm": public_context,
        "program_application_scope": "training_windows_only",
        "capability_memory_entry_count": 0,
        "operator_inventory": inventory,
        "canonical_operator_count": len(inventory),
        "executable_operator_count": sum(row["availability"] == "EXECUTABLE" for row in inventory),
        "unavailable_operator_count": sum(row["availability"] == "UNAVAILABLE" for row in inventory),
        "llm": llm_report,
        "generation_proposals": _proposal_rows(trace),
        "cycle_status": cycle["status"],
        "cycle_reason_code": cycle["reason_code"],
        "skill_automatically_formed_from_trace": cycle["skill_draft"] is not None,
        "candidate_skill_draft": copy.deepcopy(cycle["skill_draft"]),
        "selection": selection,
        "final_status": final_status,
        "memory_written": False,
        "claim_limit": (
            "Exposed single-dataset Development evidence only. SOURCE_CANDIDATE means "
            "that an LLM-generated training-data Program survived one later temporal "
            "selection slice after Consumer retraining; evaluation contexts were not "
            "modified. It is not Promotion, fresh evidence, cross-dataset transfer, or "
            "proof that the LLM is better than deterministic search."
        ),
    }
    if write_report:
        output = root / (
            report_path if report_path is not None else Path(str(config["report_path"]))
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def _contrast_episode(label: str, report: Mapping[str, object]) -> dict[str, object]:
    context = report.get("public_context_sent_to_llm")
    skill = report.get("candidate_skill_draft")
    proposals = report.get("generation_proposals")
    if not isinstance(context, Mapping) or not isinstance(skill, Mapping):
        raise ValueError(f"contrast episode {label} lacks an accepted Candidate")
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise ValueError(f"contrast episode {label} lacks proposal history")
    source_id = skill.get("program", {}).get("source_candidate_id")  # type: ignore[union-attr]
    accepted = next(
        (
            row
            for row in proposals
            if isinstance(row, Mapping)
            and row.get("candidate_id") == source_id
            and isinstance(row.get("support_response"), Mapping)
            and row["support_response"].get("accepted") is True  # type: ignore[union-attr]
        ),
        None,
    )
    if accepted is None:
        raise ValueError(f"contrast episode {label} has no accepted proposal trace")
    selection = report.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"contrast episode {label} lacks selection response")
    episode = {
        "episode": label,
        "public_context": copy.deepcopy(dict(context)),
        "accepted_workflow": {
            "template_steps": copy.deepcopy(accepted.get("workflow_steps")),
            "compiled_steps": copy.deepcopy(accepted.get("compiled_program_steps")),
        },
        "support_response": copy.deepcopy(accepted.get("support_response")),
        "selection_response": copy.deepcopy(dict(selection)),
        "final_status": report.get("final_status"),
    }
    forbidden = {"dataset_id", "series_uid", "filename", "file_name", "path"}

    def check(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in forbidden:
                    raise ValueError(f"private field entered contrast dossier: {key}")
                check(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                check(nested)

    check(episode)
    return episode


def _joint_inventory(
    contexts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    forbidden = tuple(
        name
        for name in OPERATOR_NAMES
        if OPERATOR_METADATA[name].get("changes_target_space") is True
    )
    inventories = [
        build_public_operator_inventory(
            "forecast", context, forbidden_operators=forbidden
        )
        for context in contexts
    ]
    rows: list[dict[str, object]] = []
    for index, name in enumerate(OPERATOR_NAMES):
        variants = [inventory[index] for inventory in inventories]
        row = copy.deepcopy(variants[0])
        unavailable = [
            f"CONTEXT_{offset}:{variant['reason']}"
            for offset, variant in enumerate(variants, start=1)
            if variant["availability"] == "UNAVAILABLE"
        ]
        if unavailable:
            row["availability"] = "UNAVAILABLE"
            row["reason"] = ";".join(unavailable)
        rows.append(row)
        if row["name"] != name:
            raise AssertionError("operator inventory order changed")
    return tuple(rows)


def _live_induction_proposal(
    payload: Mapping[str, object], *, model: str, base_url: str
) -> tuple[Mapping[str, object], dict[str, object]]:
    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    system = (
        "You induce one complete typed replacement Workflow from two anonymized "
        "time-series Action--Response episodes. Use any EXECUTABLE canonical operator "
        "from the complete inventory. The same proposal must compile in both public "
        "Contexts. When a parameter should vary with public Context, use a dotted-path "
        "binding. Return exactly one JSON object matching required_output, with one to "
        "four steps and IDENTITY fallback. Do not diagnose a named fault, recommend a "
        "patch, identify a dataset, approve a Skill, or include markdown/commentary."
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, allow_nan=False),
            },
        ],
        response_format={"type": "json_object"},
    )
    choice = completion.choices[0]
    proposal = json.loads(choice.message.content or "")
    if not isinstance(proposal, dict):
        raise ValueError("induction proposer must return one JSON object")
    usage = getattr(completion, "usage", None)
    return proposal, {
        "returned_model": str(getattr(completion, "model", "")),
        "finish_reason": str(getattr(choice, "finish_reason", "")),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def _replay_compiled(
    root: Path, dataset_key: str, compiled: CompiledWorkflow
) -> dict[str, object]:
    config = DATASET_CONFIGS[dataset_key]
    roster, values = _fixed_roster(root, config)

    def score(origin: int) -> dict[str, object]:
        baseline = _evaluate(roster, values, None, config, origin=origin)
        candidate = _evaluate(roster, values, compiled, config, origin=origin)
        gains = [
            float(reference - method)
            for reference, method in zip(
                baseline["per_view_smase"], candidate["per_view_smase"]
            )
        ]
        return {
            "gain": float(baseline["mean_smase"] - candidate["mean_smase"]),
            "per_view_gain": gains,
            "behavior_point_count": candidate["behavior_point_count"],
            "error": None,
        }

    try:
        return {
            "support": score(int(config["support_origin"])),
            "selection": score(int(config["selection_origin"])),
        }
    except Exception as exc:
        return {
            "support": None,
            "selection": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_induce(
    root: Path,
    *,
    proposer: Proposer | None = None,
    model: str = DEFAULT_SLOW_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    report_path: Path = INDUCTION_REPORT_PATH,
    write_report: bool = True,
) -> dict[str, object]:
    root = Path(root)
    source_reports = [_read_object(root / path) for path in INDUCTION_SOURCE_REPORTS]
    dossier = {
        "episodes": [
            _contrast_episode("A", source_reports[0]),
            _contrast_episode("B", source_reports[1]),
        ]
    }
    contexts = [episode["public_context"] for episode in dossier["episodes"]]
    inventory = _joint_inventory(contexts)
    payload = {
        "contrast_dossier": dossier,
        "operator_inventory": copy.deepcopy(list(inventory)),
        "required_output": {
            "decision": "PROPOSE",
            "steps": [
                {
                    "op": "canonical operator name",
                    "params": {"fixed_parameter": "JSON value"},
                    "bindings": {
                        "portable_parameter": (
                            "relative dotted path; optional public_context. prefix accepted"
                        )
                    },
                }
            ],
            "requested_observations": ["public observation id"],
            "fallback": "IDENTITY",
        },
        "constraints": {
            "complete_replacement_program": True,
            "step_count": "one_to_four",
            "must_compile_in_both_public_contexts": True,
            "workflow_examples_supplied": False,
            "operator_shortlist_supplied": False,
            "llm_cannot_approve_candidate": True,
        },
    }
    provider: dict[str, object]
    if proposer is None:
        proposal, provider = _live_induction_proposal(
            payload, model=model, base_url=base_url
        )
        api_integrated = True
        api_call_count = 1
    else:
        proposal = proposer(copy.deepcopy(payload))
        provider = {}
        api_integrated = False
        api_call_count = 0
    if not isinstance(proposal, Mapping):
        raise ValueError("induction proposer must return an object")

    source_results: list[dict[str, object]] = []
    compiled_sources: list[CompiledWorkflow] = []
    for label, dataset_key, context in zip(("A", "B"), ("nn5", "gefcom"), contexts):
        try:
            compiled = compile_workflow_proposal(
                proposal, inventory, context, generation=1
            )
            replay = _replay_compiled(root, dataset_key, compiled)
            compiled_sources.append(compiled)
            source_results.append(
                {"episode": label, "compilation": "VALID", "replay": replay}
            )
        except Exception as exc:
            source_results.append(
                {
                    "episode": label,
                    "compilation": "INVALID",
                    "error": f"{type(exc).__name__}: {exc}",
                    "replay": None,
                }
            )
    source_gate = bool(
        len(compiled_sources) == 2
        and all(
            isinstance(row.get("replay"), Mapping)
            and isinstance(row["replay"].get("selection"), Mapping)  # type: ignore[union-attr]
            and float(row["replay"]["selection"]["gain"]) > 0.0  # type: ignore[index]
            and int(row["replay"]["selection"]["behavior_point_count"]) > 0  # type: ignore[index]
            for row in source_results
        )
    )

    noaa_confirmation: dict[str, object] | None = None
    final_status = "REJECTED"
    if source_gate:
        noaa_config = DATASET_CONFIGS["noaa"]
        noaa_roster, noaa_values = _fixed_roster(root, noaa_config)
        noaa_context = _public_context(noaa_roster, noaa_values, noaa_config)
        try:
            noaa_compiled = compile_workflow_proposal(
                proposal, inventory, noaa_context, generation=1
            )
            origin = int(noaa_config["selection_origin"])
            baseline = _evaluate(
                noaa_roster, noaa_values, None, noaa_config, origin=origin
            )
            candidate = _evaluate(
                noaa_roster, noaa_values, noaa_compiled, noaa_config, origin=origin
            )
            gain = float(baseline["mean_smase"] - candidate["mean_smase"])
            noaa_confirmation = {
                "context_visible_to_slow_path": False,
                "outcome_opened_after_source_gate": True,
                "historical_exposure": "OLD_PROJECT_EXPOSED",
                "selection_gain": gain,
                "per_view_gain": [
                    float(reference - method)
                    for reference, method in zip(
                        baseline["per_view_smase"], candidate["per_view_smase"]
                    )
                ],
                "behavior_point_count": candidate["behavior_point_count"],
                "error": None,
            }
            final_status = (
                "MULTISOURCE_CANDIDATE_CONFIRMED"
                if gain > 0.0 and int(candidate["behavior_point_count"]) > 0
                else "REJECTED_AFTER_CONFIRMATION"
            )
        except Exception as exc:
            final_status = "REJECTED_AFTER_CONFIRMATION"
            noaa_confirmation = {
                "context_visible_to_slow_path": False,
                "outcome_opened_after_source_gate": True,
                "historical_exposure": "OLD_PROJECT_EXPOSED",
                "selection_gain": None,
                "behavior_point_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    report = {
        "experiment_id": "E2-autonomous-natural-workflow-contrast-induction",
        "scientific_role": "exposed_multisource_induction_mechanism",
        "contrast_dossier_sent_to_llm": dossier,
        "operator_inventory": copy.deepcopy(list(inventory)),
        "canonical_operator_count": len(inventory),
        "llm": {
            "api_integrated": api_integrated,
            "api_call_count": api_call_count,
            "requested_model": model if api_integrated else None,
            "base_url": base_url if api_integrated else None,
            "provider": provider,
        },
        "replacement_proposal": copy.deepcopy(dict(proposal)),
        "source_replays": source_results,
        "source_gate_passed": source_gate,
        "template_frozen_after_source_gate": (
            copy.deepcopy(proposal.get("steps")) if source_gate else None
        ),
        "noaa_confirmation": noaa_confirmation,
        "final_status": final_status,
        "memory_written": False,
        "claim_limit": (
            "Mechanism evidence from historically exposed data. NOAA was held out from "
            "this induction prompt and opened only after the two-Source gate, but it was "
            "exposed in older project work; this is not fresh evidence, Promotion, or "
            "unseen cross-dataset transfer."
        ),
    }
    if write_report:
        output = root / report_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return report


def _common_generated_program(
    reports: Sequence[Mapping[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    per_environment: list[dict[str, list[dict[str, object]]]] = []
    for report in reports:
        proposals = report.get("generation_proposals")
        if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
            raise ValueError("generation report lacks proposal trace")
        by_operator: dict[str, list[dict[str, object]]] = {}
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                continue
            steps = proposal.get("compiled_program_steps")
            if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
                continue
            plain_steps = [
                copy.deepcopy(dict(step)) for step in steps if isinstance(step, Mapping)
            ]
            if len(plain_steps) != 1:
                continue
            op = plain_steps[0].get("op")
            params = plain_steps[0].get("params")
            if isinstance(op, str) and isinstance(params, Mapping):
                by_operator.setdefault(op, []).append(plain_steps[0])
        per_environment.append(by_operator)
    common = set(per_environment[0])
    for by_operator in per_environment[1:]:
        common &= set(by_operator)
    unique = sorted(
        op
        for op in common
        if all(len(by_operator[op]) == 1 for by_operator in per_environment)
    )
    if len(unique) != 1:
        raise ValueError("reports do not identify one unique common generated Program family")
    op = unique[0]
    return op, [copy.deepcopy(by_operator[op][0]) for by_operator in per_environment]


def _portable_template_from_current_generation(
    reports: Sequence[Mapping[str, object]], common_program: str
) -> Mapping[str, object] | None:
    templates: list[object] = []
    for report in reports:
        proposals = report.get("generation_proposals")
        if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
            return None
        matches: list[object] = []
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                continue
            compiled_steps = proposal.get("compiled_program_steps")
            workflow_steps = proposal.get("workflow_steps")
            if (
                isinstance(compiled_steps, Sequence)
                and not isinstance(compiled_steps, (str, bytes))
                and len(compiled_steps) == 1
                and isinstance(compiled_steps[0], Mapping)
                and compiled_steps[0].get("op") == common_program
                and isinstance(workflow_steps, Sequence)
                and not isinstance(workflow_steps, (str, bytes))
                and len(workflow_steps) == 1
                and isinstance(workflow_steps[0], Mapping)
                and workflow_steps[0].get("op") == common_program
            ):
                matches.append(copy.deepcopy(list(workflow_steps)))
        if len(matches) != 1:
            return None
        templates.append(matches[0])
    if len(templates) != 2 or templates[0] != templates[1]:
        return None
    return {
        "decision": "PROPOSE",
        "steps": copy.deepcopy(templates[0]),
        "fallback": "IDENTITY",
    }


_BINDING_DOSSIER_FORBIDDEN = frozenset(
    {
        "dataset",
        "dataset_id",
        "dataset_key",
        "file",
        "filename",
        "future",
        "outcome",
        "path",
        "query",
        "raw",
        "selection",
        "series_uid",
        "uid",
        "values",
    }
)


def _assert_binding_dossier_public(
    value: object, *, location: str = "binding_dossier"
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _BINDING_DOSSIER_FORBIDDEN:
                raise ValueError(f"private field entered {location}: {key}")
            _assert_binding_dossier_public(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, nested in enumerate(value):
            _assert_binding_dossier_public(nested, location=f"{location}[{index}]")


def _finite_public_context_scalars(
    public_context: Mapping[str, object],
) -> dict[str, int | float]:
    scalars: dict[str, int | float] = {}

    def visit(value: object, prefix: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = str(key).lower()
                if normalized in _BINDING_DOSSIER_FORBIDDEN:
                    raise ValueError(f"private field entered public Context: {key}")
                visit(nested, f"{prefix}.{key}" if prefix else str(key))
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            scalars[prefix] = value

    visit(public_context, "")
    return scalars


def _binding_trace_environment(
    report: Mapping[str, object], *, environment: str, common_program: str
) -> tuple[dict[str, object], Mapping[str, object], Mapping[str, object]]:
    proposals = report.get("generation_proposals")
    context = report.get("public_context_sent_to_llm")
    if not isinstance(context, Mapping):
        raise ValueError("generation trace lacks deployment-visible public Context")
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise ValueError("generation trace lacks proposal history")
    matches: list[Mapping[str, object]] = []
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        template_steps = proposal.get("workflow_steps")
        compiled_steps = proposal.get("compiled_program_steps")
        if (
            isinstance(template_steps, Sequence)
            and not isinstance(template_steps, (str, bytes))
            and len(template_steps) == 1
            and isinstance(template_steps[0], Mapping)
            and template_steps[0].get("op") == common_program
            and isinstance(compiled_steps, Sequence)
            and not isinstance(compiled_steps, (str, bytes))
            and len(compiled_steps) == 1
            and isinstance(compiled_steps[0], Mapping)
            and compiled_steps[0].get("op") == common_program
        ):
            matches.append(proposal)
    if len(matches) != 1:
        raise ValueError("binding requires one unique current trace for the common operator")
    template_step = matches[0]["workflow_steps"][0]  # type: ignore[index]
    compiled_step = matches[0]["compiled_program_steps"][0]  # type: ignore[index]
    assert isinstance(template_step, Mapping) and isinstance(compiled_step, Mapping)
    template_params = template_step.get("params")
    template_bindings = template_step.get("bindings", {})
    compiled_params = compiled_step.get("params")
    if (
        not isinstance(template_params, Mapping)
        or not isinstance(template_bindings, Mapping)
        or not isinstance(compiled_params, Mapping)
    ):
        raise ValueError("binding trace contains malformed parameters")
    public_scalars = _finite_public_context_scalars(context)
    return (
        {
            "environment": environment,
            "current_template": {
                "op": common_program,
                "params": copy.deepcopy(dict(template_params)),
                "bindings": copy.deepcopy(dict(template_bindings)),
            },
            "current_compiled_params": copy.deepcopy(dict(compiled_params)),
            "public_context_scalars": public_scalars,
        },
        context,
        matches[0],
    )


def _build_binding_dossier(
    reports: Sequence[Mapping[str, object]], common_program: str
) -> tuple[dict[str, object], list[Mapping[str, object]], list[Mapping[str, object]]]:
    if len(reports) != 2:
        raise ValueError("binding requires exactly two generation environments")
    rows: list[dict[str, object]] = []
    contexts: list[Mapping[str, object]] = []
    trace_rows: list[Mapping[str, object]] = []
    for label, report in zip(("A", "B"), reports):
        row, context, trace = _binding_trace_environment(
            report, environment=label, common_program=common_program
        )
        rows.append(row)
        contexts.append(context)
        trace_rows.append(trace)
    scalar_maps = [row["public_context_scalars"] for row in rows]
    assert all(isinstance(values, Mapping) for values in scalar_maps)
    shared_fields = sorted(
        set.intersection(*(set(values) for values in scalar_maps))
    )
    if not shared_fields:
        raise ValueError("binding dossier has no shared finite public Context scalar")
    # This is intentionally a small typed menu, not a Context snapshot or feature system.
    shared_fields = shared_fields[:64]
    for row, values in zip(rows, scalar_maps):
        assert isinstance(values, Mapping)
        row["public_context_scalars"] = [
            {"field": field, "value": values[field]} for field in shared_fields
        ]
    dossier = {
        "common_program": common_program,
        "allowed_public_context_scalar_fields": shared_fields,
        "environments": rows,
        "required_output_json_schema": {
            "oneOf": [
                {
                    "decision": "ABSTAIN",
                    "optional_reason": "text",
                },
                {
                    "decision": "PATCH_BINDING",
                    "program_op": common_program,
                    "parameter": "one currently static parameter",
                    "public_context_field": "one allowed scalar field",
                },
            ]
        },
    }
    _assert_binding_dossier_public(dossier)
    return dossier, contexts, trace_rows


def _compiled_program_steps(compiled: CompiledWorkflow) -> list[dict[str, object]]:
    assert compiled.candidate.program is not None
    return [
        {"op": op, "params": copy.deepcopy(params)}
        for op, params in compiled.candidate.program.execution_steps()
    ]


def _resolve_portable_binding_patch(
    reports: Sequence[Mapping[str, object]],
    common_program: str,
    proposer: Proposer,
) -> dict[str, object]:
    """Compile one typed binding patch and fail closed on every mismatch."""

    try:
        dossier, contexts, trace_rows = _build_binding_dossier(
            reports, common_program
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "ABSTAINED",
            "reason_code": "BINDING_DOSSIER_INVALID",
            "llm_invoked": False,
            "dossier": None,
            "patch": None,
            "validation": {"passed": False, "error": str(exc)},
            "portable_proposal": None,
        }
    try:
        proposal = proposer(copy.deepcopy(dossier))
    except (TypeError, ValueError) as exc:
        return {
            "status": "ABSTAINED",
            "reason_code": "BINDING_PROPOSER_OUTPUT_INVALID",
            "llm_invoked": True,
            "dossier": dossier,
            "patch": None,
            "validation": {"passed": False, "error": str(exc)},
            "portable_proposal": None,
        }
    if not isinstance(proposal, Mapping):
        proposal = {"decision": "INVALID_NON_OBJECT"}
    patch = copy.deepcopy(dict(proposal))

    def invalid(reason_code: str, message: str) -> dict[str, object]:
        return {
            "status": "ABSTAINED",
            "reason_code": reason_code,
            "llm_invoked": True,
            "dossier": dossier,
            "patch": patch,
            "validation": {"passed": False, "error": message},
            "portable_proposal": None,
        }

    if proposal.get("decision") == "ABSTAIN":
        if set(proposal) - {"decision", "reason"} or (
            "reason" in proposal and not isinstance(proposal["reason"], str)
        ):
            return invalid("BINDING_PATCH_INVALID", "malformed ABSTAIN response")
        return invalid("BINDING_PROPOSER_ABSTAINED", str(proposal.get("reason", "")))
    if set(proposal) != {
        "decision",
        "program_op",
        "parameter",
        "public_context_field",
    } or proposal.get("decision") != "PATCH_BINDING":
        return invalid(
            "BINDING_PATCH_INVALID",
            "binding response must be ABSTAIN or one typed PATCH_BINDING",
        )
    if proposal.get("program_op") != common_program:
        return invalid("BINDING_PATCH_INVALID", "binding patch changed the operator")
    parameter = proposal.get("parameter")
    field = proposal.get("public_context_field")
    allowed_fields = dossier["allowed_public_context_scalar_fields"]
    if (
        not isinstance(parameter, str)
        or not parameter
        or not isinstance(field, str)
        or field not in allowed_fields
    ):
        return invalid(
            "BINDING_PATCH_INVALID",
            "binding parameter or public Context field is unavailable",
        )

    environments = dossier["environments"]
    assert isinstance(environments, Sequence)
    templates = [row["current_template"] for row in environments]  # type: ignore[index]
    compiled_params = [
        row["current_compiled_params"] for row in environments  # type: ignore[index]
    ]
    template_params = [template["params"] for template in templates]  # type: ignore[index]
    template_bindings = [template["bindings"] for template in templates]  # type: ignore[index]
    if not all(
        isinstance(params, Mapping) and parameter in params
        for params in template_params
    ) or any(
        isinstance(bindings, Mapping) and parameter in bindings
        for bindings in template_bindings
    ):
        return invalid(
            "BINDING_PATCH_INVALID",
            "patched parameter must be static in every current template",
        )
    bound_values = [params[parameter] for params in template_params]  # type: ignore[index]
    if bound_values[0] == bound_values[1]:
        return invalid(
            "BINDING_VALIDATION_FAILED", "patched parameter does not vary by environment"
        )
    static_params = [
        {key: value for key, value in params.items() if key != parameter}  # type: ignore[union-attr]
        for params in template_params
    ]
    if static_params[0] != static_params[1] or template_bindings[0] != template_bindings[1]:
        return invalid(
            "BINDING_VALIDATION_FAILED",
            "static parameters or existing bindings differ across environments",
        )
    for row, expected, params in zip(environments, bound_values, compiled_params):
        scalar_rows = row["public_context_scalars"]  # type: ignore[index]
        observed = next(
            (
                scalar["value"]
                for scalar in scalar_rows
                if isinstance(scalar, Mapping) and scalar.get("field") == field
            ),
            None,
        )
        if observed != expected or not isinstance(params, Mapping) or params.get(parameter) != expected:
            return invalid(
                "BINDING_VALIDATION_FAILED",
                "public field does not equal the current bound parameter value",
            )

    portable_step = copy.deepcopy(dict(templates[0]))
    portable_params = portable_step["params"]
    portable_bindings = portable_step["bindings"]
    assert isinstance(portable_params, dict) and isinstance(portable_bindings, dict)
    del portable_params[parameter]
    portable_bindings[parameter] = field
    portable_proposal = {
        "decision": "PROPOSE",
        "steps": [portable_step],
        "fallback": "IDENTITY",
    }
    replay_equivalent: list[bool] = []
    try:
        inventory = _joint_inventory(contexts)
        for context, trace in zip(contexts, trace_rows):
            compiled = compile_workflow_proposal(
                portable_proposal, inventory, context, generation=1
            )
            original_steps = trace.get("compiled_program_steps")
            replay_equivalent.append(
                isinstance(original_steps, Sequence)
                and not isinstance(original_steps, (str, bytes))
                and _compiled_program_steps(compiled) == list(original_steps)
            )
    except Exception as exc:
        return invalid(
            "BINDING_VALIDATION_FAILED", f"portable recompilation failed: {exc}"
        )
    if replay_equivalent != [True, True]:
        return invalid(
            "BINDING_VALIDATION_FAILED",
            "portable recompilation is not equivalent to both current traces",
        )
    return {
        "status": "VALIDATED",
        "reason_code": "PATCH_BINDING_VALIDATED",
        "llm_invoked": True,
        "dossier": dossier,
        "patch": patch,
        "validation": {
            "passed": True,
            "static_parameters_equal": True,
            "bound_parameter_varies": True,
            "public_field_matches_bound_values": True,
            "source_recompile_equivalent": replay_equivalent,
        },
        "portable_proposal": portable_proposal,
    }


def _compiled_bound_program(
    step: Mapping[str, object], *, environment: str
) -> CompiledWorkflow:
    op = step.get("op")
    params = step.get("params")
    if not isinstance(op, str) or not isinstance(params, Mapping):
        raise ValueError("bound Program step is malformed")
    program = Program.from_steps([(op, copy.deepcopy(dict(params)))], source="llm_generated")
    candidate = Candidate.program_candidate(
        f"scope-source-{environment.lower()}", program, source="llm_generated"
    )
    return CompiledWorkflow(
        candidate,
        (),
        ({"op": op, "params": copy.deepcopy(dict(params)), "bindings": {}},),
    )


def _history_summary_for_series(
    values: object, *, cutoff: int, calendar_period: int
) -> dict[str, object]:
    gateway = CohortHistoryPublicToolGateway(
        [values[:cutoff]],  # type: ignore[index]
        calendar_period=calendar_period,
        window_length=CONTEXT_LENGTH,
    )
    receipt = gateway.call("compare_history_windows", {})
    summary = _plain_public(receipt.public_result)
    if not isinstance(summary, dict):
        raise TypeError("history comparison must be a public object")
    return summary


_SCOPE_DOSSIER_FORBIDDEN = frozenset(
    {
        "raw",
        "raw_values",
        "series_values",
        "values",
        "dataset_id",
        "dataset_key",
        "series_uid",
        "path",
        "file_path",
        "selection",
        "selection_outcome",
        "query",
        "query_outcome",
        "future",
        "future_values",
        "clean",
        "clean_truth",
    }
)


def _assert_scope_dossier_public(value: object, *, path: str = "scope_dossier") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).lower() in _SCOPE_DOSSIER_FORBIDDEN:
                raise ValueError(f"private field entered {path}: {key}")
            _assert_scope_dossier_public(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _assert_scope_dossier_public(nested, path=f"{path}[{index}]")


def _numeric_summary_fields(episodes: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    field_sets: list[set[str]] = []
    for episode in episodes:
        summary = episode.get("public_history_summary")
        if not isinstance(summary, Mapping):
            raise ValueError("scope episode lacks public history summary")
        fields: set[str] = set()
        for section_name in ("early", "recent", "early_to_recent_change"):
            section = summary.get(section_name)
            if not isinstance(section, Mapping):
                raise ValueError("history summary lacks a fixed comparison section")
            for name, value in section.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    fields.add(f"{section_name}.{name}")
        field_sets.append(fields)
    allowed = set.intersection(*field_sets) if field_sets else set()
    if not allowed:
        raise ValueError("no shared finite numeric history fields are available")
    return tuple(sorted(allowed))


def _scope_field(summary: Mapping[str, object], path: str) -> float:
    current: object = summary
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise KeyError(path)
        current = current[segment]
    if (
        not isinstance(current, (int, float))
        or isinstance(current, bool)
        or not math.isfinite(float(current))
    ):
        raise ValueError(f"scope field is not finite numeric: {path}")
    return float(current)


def _compile_scope_patch(
    proposal: Mapping[str, object],
    *,
    common_program: str,
    allowed_fields: Sequence[str],
) -> tuple[dict[str, object], ...] | None:
    if not isinstance(proposal, Mapping):
        raise ValueError("scope proposal must be an object")
    decision = proposal.get("decision")
    if decision == "ABSTAIN":
        if set(proposal) - {"decision", "reason"}:
            raise ValueError("ABSTAIN proposal has unsupported fields")
        if "reason" in proposal and not isinstance(proposal["reason"], str):
            raise ValueError("ABSTAIN reason must be text")
        return None
    if decision != "RESTRICT_SCOPE" or set(proposal) != {
        "decision",
        "program_op",
        "predicate",
    }:
        raise ValueError("scope proposal must be ABSTAIN or one typed RESTRICT_SCOPE patch")
    if proposal["program_op"] != common_program:
        raise ValueError("scope patch must target the automatically discovered common Program")
    predicate = proposal["predicate"]
    if not isinstance(predicate, Mapping) or set(predicate) != {"all"}:
        raise ValueError("scope predicate must contain one all conjunction")
    conditions = predicate["all"]
    if (
        not isinstance(conditions, list)
        or not 1 <= len(conditions) <= 3
        or not all(isinstance(condition, Mapping) for condition in conditions)
    ):
        raise ValueError("scope predicate requires one to three AND conditions")
    allowed = set(allowed_fields)
    compiled: list[dict[str, object]] = []
    seen_fields: set[str] = set()
    for condition in conditions:
        assert isinstance(condition, Mapping)
        if set(condition) != {"field", "op", "value"}:
            raise ValueError("scope condition has unsupported fields")
        field = condition["field"]
        op = condition["op"]
        value = condition["value"]
        if not isinstance(field, str) or field not in allowed or field in seen_fields:
            raise ValueError("scope condition field is unavailable or repeated")
        if op not in {"<=", ">="}:
            raise ValueError("scope condition operator must be <= or >=")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("scope threshold must be finite numeric")
        seen_fields.add(field)
        compiled.append({"field": field, "op": op, "value": float(value)})
    return tuple(compiled)


def _scope_matches(
    summary: Mapping[str, object], conditions: Sequence[Mapping[str, object]]
) -> bool:
    for condition in conditions:
        actual = _scope_field(summary, str(condition["field"]))
        threshold = float(condition["value"])
        if condition["op"] == "<=" and not actual <= threshold:
            return False
        if condition["op"] == ">=" and not actual >= threshold:
            return False
    return True


def _policy_score(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
    compiled: CompiledWorkflow | None,
    *,
    origin: int,
    scope: set[str] | None = None,
) -> dict[str, object]:
    baseline = _evaluate(roster, values, None, config, origin=origin)
    candidate = _evaluate(
        roster,
        values,
        compiled,
        config,
        origin=origin,
        train_series_scope=scope,
    )
    gains = [
        float(reference - method)
        for reference, method in zip(
            baseline["per_view_smase"], candidate["per_view_smase"]
        )
    ]
    return {
        "baseline_mean_smase": baseline["mean_smase"],
        "candidate_mean_smase": candidate["mean_smase"],
        "gain_vs_identity": float(baseline["mean_smase"] - candidate["mean_smase"]),
        "per_view_gain": gains,
        "behavior_point_count": candidate["behavior_point_count"],
    }


def _scope_induction_environment(
    root: Path,
    *,
    environment: str,
    dataset_key: str,
    bound_step: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    config = DATASET_CONFIGS[dataset_key]
    roster, values = _fixed_roster(root, config)
    compiled = _compiled_bound_program(bound_step, environment=environment)
    support_origin = int(config["support_origin"])
    baseline = _evaluate(roster, values, None, config, origin=support_origin)
    episodes: list[dict[str, object]] = []
    train_rows = [row for row in roster if row["role"] == "train"]
    summaries_by_uid: dict[str, dict[str, object]] = {}
    for ordinal, row in enumerate(train_rows):
        uid = str(row["series_uid"])
        summary = _history_summary_for_series(
            values[uid],
            cutoff=support_origin,
            calendar_period=int(config["period"]),
        )
        summaries_by_uid[uid] = summary
        singleton = _evaluate(
            roster,
            values,
            compiled,
            config,
            origin=support_origin,
            train_series_scope={uid},
        )
        gains = [
            float(reference - method)
            for reference, method in zip(
                baseline["per_view_smase"], singleton["per_view_smase"]
            )
        ]
        episodes.append(
            {
                "environment": environment,
                "within_environment_ordinal": ordinal,
                "program": copy.deepcopy(dict(bound_step)),
                "public_history_summary": summary,
                "support_exact_singleton_response": {
                    "credit_level": "PROPOSAL_ONLY_LOCAL_ACTION_EPISODE",
                    "cohort_support_gain": float(
                        baseline["mean_smase"] - singleton["mean_smase"]
                    ),
                    "per_view_gain": gains,
                    "behavior_point_count": singleton["behavior_point_count"],
                },
            }
        )
    public_environment = {"environment": environment, "episodes": episodes}
    _assert_scope_dossier_public(public_environment)
    return public_environment, {
        "roster": roster,
        "values": values,
        "config": config,
        "compiled": compiled,
        "summaries_by_uid": summaries_by_uid,
    }


def _scope_output_schema(
    common_program: str, allowed_fields: Sequence[str]
) -> dict[str, object]:
    return {
        "oneOf": [
            {
                "type": "object",
                "required": ["decision"],
                "properties": {
                    "decision": {"const": "ABSTAIN"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["decision", "program_op", "predicate"],
                "properties": {
                    "decision": {"const": "RESTRICT_SCOPE"},
                    "program_op": {"const": common_program},
                    "predicate": {
                        "type": "object",
                        "required": ["all"],
                        "properties": {
                            "all": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "required": ["field", "op", "value"],
                                    "properties": {
                                        "field": {"enum": list(allowed_fields)},
                                        "op": {"enum": ["<=", ">="]},
                                        "value": {"type": "number"},
                                    },
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        ]
    }


def _confirm_scoped_portable_program(
    root: Path,
    *,
    confirmation_dataset_key: str,
    portable_proposal: Mapping[str, object],
    portable_program_source: str,
    common_program: str,
    conditions: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], bool]:
    """Open the existing confirmation only after Scope and Program are frozen."""

    config = DATASET_CONFIGS[confirmation_dataset_key]
    roster, values = _fixed_roster(root, config)
    confirmation_context = _public_context(roster, values, config)
    confirmation_inventory = build_public_operator_inventory(
        "forecast",
        confirmation_context,
        forbidden_operators=tuple(
            name
            for name in OPERATOR_NAMES
            if OPERATOR_METADATA[name].get("changes_target_space") is True
        ),
    )
    compiled = compile_workflow_proposal(
        portable_proposal,
        confirmation_inventory,
        confirmation_context,
        generation=1,
    )
    compiled_ops = {step.op for step in compiled.candidate.program.steps}  # type: ignore[union-attr]
    if compiled_ops != {common_program}:
        raise ValueError("portable Program does not match the Scope target")
    train_uids = [str(row["series_uid"]) for row in roster if row["role"] == "train"]
    summaries = {
        uid: _history_summary_for_series(
            values[uid],
            cutoff=int(config["support_origin"]),
            calendar_period=int(config["period"]),
        )
        for uid in train_uids
    }
    eligible_uids = {
        uid for uid in train_uids if _scope_matches(summaries[uid], conditions)
    }
    support = _policy_score(
        roster,
        values,
        config,
        compiled,
        origin=int(config["support_origin"]),
        scope=eligible_uids,
    )
    selection = _policy_score(
        roster,
        values,
        config,
        compiled,
        origin=int(config["selection_origin"]),
        scope=eligible_uids,
    )
    passed = bool(
        eligible_uids
        and float(selection["gain_vs_identity"]) > 0.0
        and int(selection["behavior_point_count"]) > 0
    )
    return {
        "environment": "C",
        "context_sent_to_llm": False,
        "program_binding_source": portable_program_source,
        "outcome_opened_after_scope_freeze": True,
        "historical_exposure": "OLD_PROJECT_EXPOSED",
        "eligible_count": len(eligible_uids),
        "training_series_count": len(train_uids),
        "support": support,
        "selection": selection,
        "passed": passed,
    }, passed


def run_induce_scope(
    root: Path,
    *,
    proposer: Proposer,
    source_generation_results: Sequence[Mapping[str, object]] | None = None,
    portable_workflow_proposal: Mapping[str, object] | None = None,
    portable_proposal_path: Path | None = INDUCTION_REPORT_PATH,
    confirmation_dataset_key: str | None = None,
    proposal_source: str | None = None,
    report_path: Path = SCOPE_INDUCTION_REPORT_PATH,
    write_report: bool = True,
) -> dict[str, object]:
    """Generate one Scope patch through an injected, separately authorized proposer."""

    root = Path(root)
    source_reports = (
        [copy.deepcopy(dict(report)) for report in source_generation_results]
        if source_generation_results is not None
        else [_read_object(root / path) for path in SCOPE_SOURCE_REPORTS]
    )
    if len(source_reports) != 2:
        raise ValueError("Scope induction requires exactly two generation results")
    common_program, bound_steps = _common_generated_program(source_reports)
    environments: list[dict[str, object]] = []
    private_runtime: list[dict[str, object]] = []
    for label, dataset_key, step in zip(("A", "B"), ("nn5", "gefcom"), bound_steps):
        environment, runtime = _scope_induction_environment(
            root,
            environment=label,
            dataset_key=dataset_key,
            bound_step=step,
        )
        environments.append(environment)
        private_runtime.append(runtime)
    episodes = [
        episode
        for environment in environments
        for episode in environment["episodes"]  # type: ignore[union-attr]
    ]
    allowed_fields = _numeric_summary_fields(episodes)
    payload = {
        "scientific_question": (
            "Can one public per-series predicate improve a generated global "
            "training-data Program?"
        ),
        "common_generated_program": common_program,
        "allowed_public_numeric_fields": list(allowed_fields),
        "local_action_episodes": environments,
        "local_credit_semantics": (
            "Each response is an exact singleton training-data intervention at exposed "
            "Support and may propose Scope only; it is not final policy utility."
        ),
        "required_output_json_schema": _scope_output_schema(
            common_program, allowed_fields
        ),
    }
    _assert_scope_dossier_public(payload)
    initial_proposal = proposer(copy.deepcopy(payload))
    if not isinstance(initial_proposal, Mapping):
        raise ValueError("scope proposer must return an object")

    def evaluate_scope_attempt(
        candidate_proposal: Mapping[str, object], *, attempt_number: int
    ) -> dict[str, object]:
        compilation = "VALID"
        compilation_error: str | None = None
        try:
            conditions = _compile_scope_patch(
                candidate_proposal,
                common_program=common_program,
                allowed_fields=allowed_fields,
            )
        except ValueError as exc:
            conditions = None
            compilation = "INVALID"
            compilation_error = str(exc)

        replays: list[dict[str, object]] = []
        total_series = 0
        total_eligible = 0
        behavior_changed = False
        if compilation == "VALID" and conditions is not None:
            for label, runtime in zip(("A", "B"), private_runtime):
                roster = runtime["roster"]
                values = runtime["values"]
                config = runtime["config"]
                compiled = runtime["compiled"]
                summaries = runtime["summaries_by_uid"]
                assert isinstance(roster, Sequence) and isinstance(values, Mapping)
                assert isinstance(config, Mapping) and isinstance(
                    compiled, CompiledWorkflow
                )
                assert isinstance(summaries, Mapping)
                train_uids = [
                    str(row["series_uid"])
                    for row in roster
                    if isinstance(row, Mapping) and row.get("role") == "train"
                ]
                eligible_uids = {
                    uid
                    for uid in train_uids
                    if _scope_matches(summaries[uid], conditions)  # type: ignore[arg-type]
                }
                total_series += len(train_uids)
                total_eligible += len(eligible_uids)
                support_origin = int(config["support_origin"])
                selection_origin = int(config["selection_origin"])
                support_global = _policy_score(
                    roster, values, config, compiled, origin=support_origin
                )
                support_scoped = _policy_score(
                    roster,
                    values,
                    config,
                    compiled,
                    origin=support_origin,
                    scope=eligible_uids,
                )
                selection_global = _policy_score(
                    roster, values, config, compiled, origin=selection_origin
                )
                selection_scoped = _policy_score(
                    roster,
                    values,
                    config,
                    compiled,
                    origin=selection_origin,
                    scope=eligible_uids,
                )
                behavior_changed = behavior_changed or any(
                    scoped["behavior_point_count"]
                    != global_score["behavior_point_count"]
                    for scoped, global_score in (
                        (support_scoped, support_global),
                        (selection_scoped, selection_global),
                    )
                )
                replays.append(
                    {
                        "environment": label,
                        "eligible_ordinals": [
                            index
                            for index, uid in enumerate(train_uids)
                            if uid in eligible_uids
                        ],
                        "eligible_count": len(eligible_uids),
                        "training_series_count": len(train_uids),
                        "identity": {"support_gain": 0.0, "selection_gain": 0.0},
                        "global_program": {
                            "support": support_global,
                            "selection": selection_global,
                        },
                        "scoped_program": {
                            "support": support_scoped,
                            "selection": selection_scoped,
                        },
                    }
                )

        dead_patch = bool(
            compilation == "VALID"
            and conditions is not None
            and (
                total_eligible == 0
                or total_eligible == total_series
                or not behavior_changed
            )
        )
        replay_gate = bool(
            compilation == "VALID"
            and conditions is not None
            and not dead_patch
            and len(replays) == 2
            and all(
                float(row["scoped_program"][split]["gain_vs_identity"])  # type: ignore[index]
                > max(
                    0.0,
                    float(row["global_program"][split]["gain_vs_identity"]),  # type: ignore[index]
                )
                and int(row["scoped_program"][split]["behavior_point_count"]) > 0  # type: ignore[index]
                for row in replays
                for split in ("support", "selection")
            )
        )
        selection_pairs = [
            (
                float(row["global_program"]["selection"]["gain_vs_identity"]),  # type: ignore[index]
                float(row["scoped_program"]["selection"]["gain_vs_identity"]),  # type: ignore[index]
                int(row["scoped_program"]["selection"]["behavior_point_count"]),  # type: ignore[index]
            )
            for row in replays
        ]
        risk_patch_replay_passed = bool(
            compilation == "VALID"
            and conditions is not None
            and not dead_patch
            and len(selection_pairs) == 2
            and all(
                scoped >= 0.0 and scoped >= global_gain
                for global_gain, scoped, _ in selection_pairs
            )
            and any(
                scoped > global_gain
                for global_gain, scoped, _ in selection_pairs
            )
            and any(
                scoped > 0.0 and behavior > 0
                for _, scoped, behavior in selection_pairs
            )
        )
        return {
            "attempt": attempt_number,
            "proposal": copy.deepcopy(dict(candidate_proposal)),
            "compilation": compilation,
            "compilation_error": compilation_error,
            "conditions": conditions,
            "dead_patch": dead_patch,
            "policy_replays": replays,
            "replay_gate_passed": replay_gate,
            "risk_patch_replay_passed": risk_patch_replay_passed,
        }

    attempts = [evaluate_scope_attempt(initial_proposal, attempt_number=1)]
    first_attempt = attempts[0]
    if (
        first_attempt["compilation"] == "VALID"
        and first_attempt["conditions"] is not None
        and first_attempt["dead_patch"] is False
        and first_attempt["risk_patch_replay_passed"] is False
    ):
        compact_replay: list[dict[str, object]] = []
        for replay in first_attempt["policy_replays"]:  # type: ignore[union-attr]
            assert isinstance(replay, Mapping)
            compact: dict[str, object] = {
                "environment": replay["environment"],
                "eligible_count": replay["eligible_count"],
                "training_series_count": replay["training_series_count"],
            }
            for policy_name, report_key in (
                ("global", "global_program"),
                ("scoped", "scoped_program"),
            ):
                policy = replay[report_key]
                assert isinstance(policy, Mapping)
                for split_name in ("support", "selection"):
                    split = policy[split_name]
                    assert isinstance(split, Mapping)
                    compact[f"{policy_name}_{split_name}"] = {
                        "gain_vs_identity": split["gain_vs_identity"],
                        "behavior_point_count": split["behavior_point_count"],
                    }
            compact_replay.append(compact)
        revision_payload = {
            "original_scope_dossier": copy.deepcopy(environments),
            "original_typed_patch": copy.deepcopy(dict(initial_proposal)),
            "anonymous_full_policy_replay": compact_replay,
            "frozen_risk_gate_semantics": {
                "each_environment": [
                    "scoped_selection_gain >= 0",
                    "scoped_selection_gain >= global_selection_gain",
                ],
                "across_environments": [
                    "at_least_one_scoped_selection_gain > global_selection_gain",
                    (
                        "at_least_one_scoped_selection_gain > 0 with "
                        "scoped_selection_behavior_point_count > 0"
                    ),
                ],
            },
            "required_output_json_schema": _scope_output_schema(
                common_program, allowed_fields
            ),
        }
        _assert_scope_dossier_public(revision_payload)
        revised_proposal = proposer(copy.deepcopy(revision_payload))
        if not isinstance(revised_proposal, Mapping):
            raise ValueError("scope proposer must return an object")
        attempts.append(evaluate_scope_attempt(revised_proposal, attempt_number=2))

    final_attempt = attempts[-1]
    proposal = final_attempt["proposal"]
    compilation = str(final_attempt["compilation"])
    compilation_error = final_attempt["compilation_error"]
    conditions = final_attempt["conditions"]
    dead_patch = bool(final_attempt["dead_patch"])
    replays = final_attempt["policy_replays"]
    replay_gate = bool(final_attempt["replay_gate_passed"])
    risk_patch_replay_passed = bool(final_attempt["risk_patch_replay_passed"])

    confirmation: dict[str, object] | None = None
    confirmation_passed = False
    confirmation_status = "NOT_REQUESTED"
    portable_program_source: str | None = None
    if confirmation_dataset_key is not None and risk_patch_replay_passed:
        if confirmation_dataset_key not in DATASET_CONFIGS:
            raise ValueError("unknown Scope confirmation dataset")
        assert conditions is not None
        portable_proposal = portable_workflow_proposal
        if portable_proposal is not None:
            portable_program_source = "CURRENT_GENERATION_TRACES"
        elif portable_proposal_path is not None:
            portable_report = _read_object(root / portable_proposal_path)
            cached_proposal = portable_report.get("replacement_proposal")
            if isinstance(cached_proposal, Mapping):
                portable_proposal = cached_proposal
                portable_program_source = "EXPLICIT_CACHED_REPORT_COMPATIBILITY"
        if portable_proposal is None:
            confirmation_status = "SKIPPED_NO_CURRENT_PORTABLE_TEMPLATE"
        else:
            confirmation_status = "OPENED_AFTER_RISK_PATCH_REPLAY"
            confirmation, confirmation_passed = _confirm_scoped_portable_program(
                root,
                confirmation_dataset_key=confirmation_dataset_key,
                portable_proposal=portable_proposal,
                portable_program_source=portable_program_source,
                common_program=common_program,
                conditions=conditions,
            )
    elif confirmation_dataset_key is not None:
        confirmation_status = "SKIPPED_RISK_PATCH_REPLAY_NOT_PASSED"
    final_status = (
        "DEVELOPMENT_MULTISOURCE_SCOPE_CANDIDATE"
        if confirmation_passed
        else "REJECTED_AFTER_CONFIRMATION"
        if confirmation is not None
        else "SOURCE_SCOPE_CANDIDATE"
        if replay_gate
        else "RISK_PATCH_REPLAY_PASSED"
        if risk_patch_replay_passed
        else "ABSTAINED"
        if compilation == "VALID" and conditions is None
        else "REJECTED_DEAD_PATCH"
        if dead_patch
        else "REJECTED"
    )
    report = {
        "experiment_id": "E2-autonomous-natural-workflow-scope-induction-v1",
        "scientific_role": "exposed_development_scope_generation_mechanism",
        "causal_hypothesis": "GLOBAL_PROGRAM_SCOPE_TOO_COARSE",
        "program_application_scope": "training_windows_only",
        "common_program_discovered_from_generation_traces": common_program,
        "scope_dossier_sent_to_proposer": environments,
        "allowed_public_numeric_fields": list(allowed_fields),
        "scope_proposal": copy.deepcopy(dict(proposal)),
        "scope_attempts": [
            {
                "attempt": attempt["attempt"],
                "proposal": copy.deepcopy(attempt["proposal"]),
                "compilation": attempt["compilation"],
                "compilation_error": attempt["compilation_error"],
                "compiled_conditions": (
                    copy.deepcopy(list(attempt["conditions"]))
                    if attempt["conditions"]
                    else None
                ),
                "dead_patch": attempt["dead_patch"],
                "policy_replays": copy.deepcopy(attempt["policy_replays"]),
                "replay_gate_passed": attempt["replay_gate_passed"],
                "risk_patch_replay_passed": attempt[
                    "risk_patch_replay_passed"
                ],
            }
            for attempt in attempts
        ],
        "revision_invoked": len(attempts) == 2,
        "proposal_source": proposal_source,
        "llm": {
            "api_integrated": isinstance(proposer, LiveScopeProposer),
            "api_call_count": int(getattr(proposer, "call_count", 0)),
            "requested_model": getattr(proposer, "model", None),
            "base_url": getattr(proposer, "base_url", None),
            "calls": copy.deepcopy(getattr(proposer, "calls", [])),
        },
        "compilation": compilation,
        "compilation_error": compilation_error,
        "compiled_conditions": copy.deepcopy(list(conditions)) if conditions else None,
        "dead_patch": dead_patch,
        "policy_replays": replays,
        "replay_gate_passed": replay_gate,
        "risk_patch_replay_passed": risk_patch_replay_passed,
        "confirmation": confirmation,
        "confirmation_status": confirmation_status,
        "portable_program_source": portable_program_source,
        "final_status": final_status,
        "memory_written": False,
        "evidence_semantics": {
            "local_singleton": "proposal_credit_only",
            "full_scoped_retrain": "policy_evidence",
        },
        "claim_limit": (
            "Exposed Development evidence only. Local singleton responses propose a "
            "per-series Scope; only full scoped Consumer retraining supplies policy "
            "evidence. This run cannot promote or write Memory."
        ),
    }
    if write_report:
        output = root / report_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return report


def run_autonomous_acquisition_cycle(
    root: Path,
    *,
    generation_runner: Callable[..., dict[str, object]] = run,
    scope_runner: Callable[..., dict[str, object]] = run_induce_scope,
    scope_proposer: Proposer | None = None,
    binding_proposer: Proposer | None = None,
    generation_model: str = DEFAULT_MODEL,
    scope_model: str = DEFAULT_SLOW_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    report_path: Path = AUTONOMOUS_CYCLE_REPORT_PATH,
    write_report: bool = True,
) -> dict[str, object]:
    """Run the exposed Development acquisition lifecycle without manual phases."""

    root = Path(root)
    generation_results: list[dict[str, object]] = []
    generation_stages: list[dict[str, object]] = []

    def stop_after_generation_fault(
        *, reason_code: str, environment: str, message: str
    ) -> dict[str, object]:
        resolution = {
            "status": "ABSTAINED",
            "reason_code": reason_code,
            "rejected_capability_version": None,
            "contextual_episode": None,
            "operator_blacklisted": False,
            "program_family_closed": False,
            "memory_write_authorized": False,
            "memory_write_count": 0,
            "generated_skill_card": None,
        }
        completed_api_calls = sum(
            int(stage["llm_api_call_count"])
            for stage in generation_stages
            if isinstance(stage.get("llm_api_call_count"), int)
        )
        terminal_report = {
            "experiment_id": "E2-autonomous-natural-acquisition-cycle-v1",
            "scientific_role": (
                "exposed_development_autonomous_acquisition_lifecycle"
            ),
            "first_fault": {
                "stage": "generation",
                "environment": environment,
                "reason_code": reason_code,
                "message": message,
            },
            "stages": {
                "generation": copy.deepcopy(generation_stages),
                "scope_and_full_policy_replay": {
                    "final_status": "NOT_RUN",
                    "reason_code": "STOPPED_AFTER_GENERATION_FAULT",
                    "llm_api_call_count": 0,
                    "intermediate_report_written": False,
                },
                "binding": {
                    "status": "NOT_RUN",
                    "reason_code": "STOPPED_AFTER_GENERATION_FAULT",
                    "llm_invoked": False,
                    "llm_api_call_count": 0,
                    "patch": None,
                    "validation": None,
                    "intermediate_report_written": False,
                },
                "confirmation": {
                    "status": "NOT_RUN",
                    "context_sent_to_llm": False,
                    "passed": None,
                },
            },
            "llm_api_call_count": completed_api_calls,
            "resolution": resolution,
            "final_status": "ABSTAINED",
            "staged_memory": [],
            "staged_memory_count": 0,
            "persistent_memory_written": False,
            "intermediate_reports_written": False,
            "operator_blacklisted": False,
            "program_family_closed": False,
            "claim_limit": (
                "Exposed Development lifecycle stopped at the first generation "
                "contract fault; no downstream evidence was opened and no Memory "
                "write was authorized."
            ),
        }
        if write_report:
            output = root / report_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    terminal_report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return terminal_report

    observation_contract_faults = {
        "observation stage did not confirm completion",
        "observation stage must execute compare_history_windows exactly once",
    }
    for environment, dataset_key in (("A", "nn5"), ("B", "gefcom")):
        try:
            result = generation_runner(
                root,
                observe_history=True,
                dataset_key=dataset_key,
                model=generation_model,
                base_url=base_url,
                write_report=False,
            )
        except RuntimeError as exc:
            if str(exc) not in observation_contract_faults:
                raise
            generation_stages.append(
                {
                    "environment": environment,
                    "cycle_status": "ABSTAINED",
                    "final_status": "NOT_COMPLETED",
                    "reason_code": "OBSERVATION_CONTRACT_VIOLATION",
                    "accepted_program_steps": None,
                    "final_program_steps": None,
                    "support_gain": None,
                    "support_status": None,
                    "selection_gain": None,
                    "selection_status": "NOT_RUN",
                    "llm_api_call_count": None,
                    "intermediate_report_written": False,
                }
            )
            return stop_after_generation_fault(
                reason_code="OBSERVATION_CONTRACT_VIOLATION",
                environment=environment,
                message=str(exc),
            )
        if not isinstance(result, dict):
            raise TypeError("generation stage must return a report object")
        generation_results.append(result)
        llm = result.get("llm")
        proposals = result.get("generation_proposals")
        proposal_rows = (
            [row for row in proposals if isinstance(row, Mapping)]
            if isinstance(proposals, Sequence) and not isinstance(proposals, (str, bytes))
            else []
        )
        skill = result.get("candidate_skill_draft")
        skill_program = skill.get("program") if isinstance(skill, Mapping) else None
        accepted_id = (
            skill_program.get("source_candidate_id")
            if isinstance(skill_program, Mapping)
            else None
        )
        final_proposal = next(
            (row for row in proposal_rows if row.get("candidate_id") == accepted_id),
            proposal_rows[-1] if proposal_rows else None,
        )
        support = (
            final_proposal.get("support_response")
            if isinstance(final_proposal, Mapping)
            else None
        )
        selection = result.get("selection")
        generation_stages.append(
            {
                "environment": environment,
                "cycle_status": result.get("cycle_status"),
                "final_status": result.get("final_status"),
                "accepted_program_steps": (
                    copy.deepcopy(skill_program.get("steps"))
                    if isinstance(skill_program, Mapping)
                    else None
                ),
                "final_program_steps": (
                    copy.deepcopy(final_proposal.get("compiled_program_steps"))
                    if isinstance(final_proposal, Mapping)
                    else None
                ),
                "support_gain": (
                    support.get("support_gain")
                    if isinstance(support, Mapping)
                    else None
                ),
                "support_status": (
                    support.get("accepted") if isinstance(support, Mapping) else None
                ),
                "selection_gain": (
                    selection.get("selection_gain")
                    if isinstance(selection, Mapping)
                    else None
                ),
                "selection_status": result.get("final_status"),
                "llm_api_call_count": (
                    llm.get("api_call_count") if isinstance(llm, Mapping) else None
                ),
                "intermediate_report_written": False,
            }
        )
        if result.get("cycle_status") == "ABSTAIN":
            reason = result.get("cycle_reason_code")
            reason_code = (
                str(reason) if isinstance(reason, str) and reason else "GENERATION_ABSTAINED"
            )
            return stop_after_generation_fault(
                reason_code=reason_code,
                environment=environment,
                message="generation stage returned a compile-safe abstention",
            )

    common_program: str | None = None
    try:
        common_program, _bound_steps = _common_generated_program(generation_results)
    except ValueError as exc:
        if str(exc) != "reports do not identify one unique common generated Program family":
            raise
        current_portable_proposal = None
    else:
        current_portable_proposal = _portable_template_from_current_generation(
            generation_results, common_program
        )

    if scope_proposer is None:
        api_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("AGICTO_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
        scope_proposer = LiveScopeProposer(
            api_key=api_key,
            model=scope_model,
            base_url=base_url,
        )

    no_common_reason = "reports do not identify one unique common generated Program family"
    binding_result: dict[str, object] = {
        "status": "NOT_RUN",
        "reason_code": "CURRENT_PORTABLE_TEMPLATE_AVAILABLE",
        "llm_invoked": False,
        "dossier": None,
        "patch": None,
        "validation": None,
        "portable_proposal": None,
    }
    first_fault: dict[str, object] | None = None
    try:
        slow_result = scope_runner(
            root,
            proposer=scope_proposer,
            source_generation_results=generation_results,
            portable_workflow_proposal=current_portable_proposal,
            portable_proposal_path=None,
            confirmation_dataset_key="noaa",
            write_report=False,
        )
    except ValueError as exc:
        if str(exc) != no_common_reason:
            raise
        slow_result = {
            "final_status": "ABSTAINED",
            "reason_code": "NO_UNIQUE_COMMON_LLM_GENERATED_PROGRAM",
            "llm": {"api_call_count": 0},
            "confirmation": None,
            "confirmation_status": "NOT_OPENED_NO_COMMON_PROGRAM",
        }
        resolution = {
            "status": "ABSTAINED",
            "reason_code": "NO_UNIQUE_COMMON_LLM_GENERATED_PROGRAM",
            "rejected_capability_version": None,
            "contextual_episode": None,
            "operator_blacklisted": False,
            "program_family_closed": False,
            "memory_write_authorized": False,
            "memory_write_count": 0,
            "generated_skill_card": None,
        }
        staged_memory: list[dict[str, object]] = []
    else:
        if (
            slow_result.get("risk_patch_replay_passed") is True
            and current_portable_proposal is None
            and common_program is not None
        ):
            first_fault = {
                "stage": "binding",
                "reason_code": "NO_CURRENT_PORTABLE_TEMPLATE",
                "message": (
                    "Source traces share one operator but not one portable parameter "
                    "template after the Scope risk replay passed."
                ),
                "resolved": False,
            }
            if binding_proposer is None:
                api_key = (
                    os.environ.get("OPENAI_API_KEY", "").strip()
                    or os.environ.get("AGICTO_API_KEY", "").strip()
                )
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
                binding_proposer = LiveBindingProposer(
                    api_key=api_key,
                    model=scope_model,
                    base_url=base_url,
                )
            binding_result = _resolve_portable_binding_patch(
                generation_results,
                common_program,
                binding_proposer,
            )
            if binding_result.get("status") == "VALIDATED":
                portable_proposal = binding_result.get("portable_proposal")
                conditions = slow_result.get("compiled_conditions")
                if not isinstance(portable_proposal, Mapping) or not isinstance(
                    conditions, Sequence
                ) or isinstance(conditions, (str, bytes)):
                    raise RuntimeError("validated binding lacks a frozen Scope or Program")
                confirmation, confirmation_passed = _confirm_scoped_portable_program(
                    root,
                    confirmation_dataset_key="noaa",
                    portable_proposal=portable_proposal,
                    portable_program_source="CURRENT_TRACE_PATCH_BINDING",
                    common_program=common_program,
                    conditions=[
                        condition
                        for condition in conditions
                        if isinstance(condition, Mapping)
                    ],
                )
                slow_result["confirmation"] = confirmation
                slow_result["confirmation_status"] = "OPENED_AFTER_BINDING_VALIDATION"
                slow_result["portable_program_source"] = "CURRENT_TRACE_PATCH_BINDING"
                slow_result["final_status"] = (
                    "DEVELOPMENT_MULTISOURCE_SCOPE_CANDIDATE"
                    if confirmation_passed
                    else "REJECTED_AFTER_CONFIRMATION"
                )
                first_fault["resolved"] = True
            else:
                slow_result["confirmation"] = None
                slow_result["confirmation_status"] = (
                    "SKIPPED_BINDING_PATCH_NOT_VALIDATED"
                )
                slow_result["portable_program_source"] = None
                slow_result["final_status"] = "RISK_PATCH_REPLAY_PASSED"
        elif current_portable_proposal is None:
            binding_result["reason_code"] = "SCOPE_RISK_REPLAY_NOT_PASSED"
        confirmation = slow_result.get("confirmation")
        staged_memory = []
        resolution = resolve_generated_acquisition_lifecycle(
            generation_results,
            slow_result,
            confirmation if isinstance(confirmation, Mapping) else None,
            memory_writer=staged_memory.append,
        )

    slow_llm = slow_result.get("llm")
    generation_api_calls = sum(
        int(stage["llm_api_call_count"])
        for stage in generation_stages
        if isinstance(stage.get("llm_api_call_count"), int)
    )
    scope_api_calls = (
        int(slow_llm["api_call_count"])
        if isinstance(slow_llm, Mapping)
        and isinstance(slow_llm.get("api_call_count"), int)
        else 0
    )
    binding_api_calls = (
        int(getattr(binding_proposer, "call_count", 0))
        if binding_proposer is not None
        else 0
    )
    policy_replay_summary: list[dict[str, object]] = []
    policy_replays = slow_result.get("policy_replays")
    if isinstance(policy_replays, Sequence) and not isinstance(
        policy_replays, (str, bytes)
    ):
        for replay in policy_replays:
            if not isinstance(replay, Mapping):
                continue
            summary: dict[str, object] = {
                "environment": replay.get("environment"),
                "eligible_count": replay.get("eligible_count"),
                "training_series_count": replay.get("training_series_count"),
            }
            for program_label, report_key in (
                ("global", "global_program"),
                ("scoped", "scoped_program"),
            ):
                program_result = replay.get(report_key)
                compact_splits: dict[str, object] = {}
                for split in ("support", "selection"):
                    split_result = (
                        program_result.get(split)
                        if isinstance(program_result, Mapping)
                        else None
                    )
                    compact_splits[split] = {
                        "gain_vs_identity": (
                            split_result.get("gain_vs_identity")
                            if isinstance(split_result, Mapping)
                            else None
                        ),
                        "behavior_point_count": (
                            split_result.get("behavior_point_count")
                            if isinstance(split_result, Mapping)
                            else None
                        ),
                    }
                summary[program_label] = compact_splits
            policy_replay_summary.append(summary)
    report = {
        "experiment_id": "E2-autonomous-natural-acquisition-cycle-v1",
        "scientific_role": "exposed_development_autonomous_acquisition_lifecycle",
        "first_fault": copy.deepcopy(first_fault),
        "stages": {
            "generation": generation_stages,
            "scope_and_full_policy_replay": {
                "final_status": slow_result.get("final_status"),
                "reason_code": slow_result.get("reason_code"),
                "compilation": slow_result.get("compilation"),
                "dead_patch": slow_result.get("dead_patch"),
                "risk_patch_replay_passed": slow_result.get(
                    "risk_patch_replay_passed"
                ),
                "common_program": slow_result.get(
                    "common_program_discovered_from_generation_traces"
                ),
                "compiled_scope": copy.deepcopy(
                    slow_result.get("compiled_conditions")
                ),
                "scope_attempts": copy.deepcopy(slow_result.get("scope_attempts")),
                "revision_invoked": slow_result.get("revision_invoked", False),
                "policy_replays": policy_replay_summary,
                "portable_program_source": slow_result.get(
                    "portable_program_source"
                ),
                "confirmation_status": slow_result.get("confirmation_status"),
                "llm_api_call_count": scope_api_calls,
                "llm_calls": (
                    copy.deepcopy(slow_llm.get("calls"))
                    if isinstance(slow_llm, Mapping)
                    else None
                ),
                "intermediate_report_written": False,
            },
            "binding": {
                "status": binding_result.get("status"),
                "reason_code": binding_result.get("reason_code"),
                "llm_invoked": binding_result.get("llm_invoked"),
                "llm_api_call_count": binding_api_calls,
                "requested_model": getattr(binding_proposer, "model", None),
                "calls": copy.deepcopy(getattr(binding_proposer, "calls", [])),
                "patch": copy.deepcopy(binding_result.get("patch")),
                "validation": copy.deepcopy(binding_result.get("validation")),
                "dossier": copy.deepcopy(binding_result.get("dossier")),
                "intermediate_report_written": False,
            },
            "confirmation": {
                "status": slow_result.get("confirmation_status"),
                "context_sent_to_llm": False,
                "passed": (
                    slow_result["confirmation"].get("passed")
                    if isinstance(slow_result.get("confirmation"), Mapping)
                    else None
                ),
            },
        },
        "llm_api_call_count": generation_api_calls + scope_api_calls + binding_api_calls,
        "resolution": copy.deepcopy(resolution),
        "final_status": resolution["status"],
        "staged_memory": copy.deepcopy(staged_memory),
        "staged_memory_count": len(staged_memory),
        "persistent_memory_written": False,
        "intermediate_reports_written": False,
        "operator_blacklisted": False,
        "program_family_closed": False,
        "claim_limit": (
            "Exposed Development autonomous lifecycle evidence only. A positive result "
            "may stage one generated Capability in process but cannot persist Memory or "
            "claim fresh transfer. Rejection applies only to the exact contextual "
            "Capability version resolved from these traces."
        ),
    }
    if write_report:
        output = root / report_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument(
        "--phase",
        choices=("generate", "induce", "induce-scope", "autonomous-cycle"),
        default="generate",
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), default="nn5")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--scope-proposal",
        type=Path,
        help="a direct proposal JSON or a prior Scope report containing scope_proposal",
    )
    parser.add_argument(
        "--scope-confirmation",
        choices=tuple(DATASET_CONFIGS),
        help="optional frozen development confirmation dataset; never sent to the LLM",
    )
    parser.add_argument(
        "--observe-history",
        action="store_true",
        help="run one public historical-window tool loop before Program generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise injected proposer plumbing without API calls or artifact writes",
    )
    args = parser.parse_args()
    if args.phase == "autonomous-cycle":
        if args.dry_run:
            raise SystemExit(
                "autonomous-cycle dry-run uses the focused injected orchestration test; "
                "the CLI phase is live only"
            )
        report = run_autonomous_acquisition_cycle(
            args.root,
            base_url=args.base_url,
            report_path=args.report or AUTONOMOUS_CYCLE_REPORT_PATH,
        )
        print(
            json.dumps(
                {
                    "final_status": report["final_status"],
                    "llm_api_call_count": report["llm_api_call_count"],
                    "staged_memory_count": report["staged_memory_count"],
                    "persistent_memory_written": False,
                    "report": str(args.report or AUTONOMOUS_CYCLE_REPORT_PATH),
                },
                ensure_ascii=False,
            )
        )
        return
    if args.phase == "induce-scope":
        proposal_source: str | None = None
        if args.dry_run:
            scope_proposal: Mapping[str, object] = {"decision": "ABSTAIN"}
            scope_proposer: Proposer = lambda _payload: copy.deepcopy(
                dict(scope_proposal)
            )
        elif args.scope_proposal is not None:
            loaded_scope_proposal = _read_object(args.scope_proposal)
            if not isinstance(loaded_scope_proposal, Mapping):
                raise SystemExit("scope proposal must be one JSON object")
            if isinstance(loaded_scope_proposal.get("scope_proposal"), Mapping):
                loaded_scope_proposal = loaded_scope_proposal["scope_proposal"]  # type: ignore[assignment]
            scope_proposal = loaded_scope_proposal
            proposal_source = str(args.scope_proposal)
            scope_proposer = lambda _payload: copy.deepcopy(dict(scope_proposal))
        else:
            api_key = (
                os.environ.get("OPENAI_API_KEY", "").strip()
                or os.environ.get("AGICTO_API_KEY", "").strip()
            )
            if not api_key:
                raise SystemExit("OPENAI_API_KEY or AGICTO_API_KEY is required")
            scope_proposer = LiveScopeProposer(
                api_key=api_key,
                model=args.model or DEFAULT_SLOW_MODEL,
                base_url=args.base_url,
            )

        report = run_induce_scope(
            args.root,
            proposer=scope_proposer,
            confirmation_dataset_key=args.scope_confirmation,
            proposal_source=proposal_source,
            report_path=args.report or SCOPE_INDUCTION_REPORT_PATH,
            write_report=not args.dry_run,
        )
        print(
            json.dumps(
                {
                    "dry_run": bool(args.dry_run),
                    "final_status": report["final_status"],
                    "common_program": report[
                        "common_program_discovered_from_generation_traces"
                    ],
                    "llm_api_call_count": report["llm"]["api_call_count"],
                    "report_written": not args.dry_run,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.phase == "induce":
        slow_model = args.model or DEFAULT_SLOW_MODEL
        if args.dry_run:
            def discarded_mock(_payload: Mapping[str, object]) -> Mapping[str, object]:
                return {
                    "decision": "PROPOSE",
                    "steps": [
                        {
                            "op": "period_median_complete",
                            "params": {"cycles": 3, "min_donors": 2},
                            "bindings": {"period": "periodicity.calendar_period"},
                        }
                    ],
                    "requested_observations": [],
                    "fallback": "IDENTITY",
                }

            report = run_induce(
                args.root,
                proposer=discarded_mock,
                model=slow_model,
                write_report=False,
            )
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "phase": "induce",
                        "source_gate_passed": report["source_gate_passed"],
                        "final_status": report["final_status"],
                        "canonical_operator_count": report["canonical_operator_count"],
                        "report_written": False,
                        "mock_proposal_discarded": True,
                    },
                    ensure_ascii=False,
                )
            )
            return
        report = run_induce(
            args.root,
            model=slow_model,
            base_url=args.base_url,
            report_path=args.report or INDUCTION_REPORT_PATH,
        )
        print(
            json.dumps(
                {
                    "final_status": report["final_status"],
                    "llm_api_call_count": report["llm"]["api_call_count"],
                    "report": str(args.report or INDUCTION_REPORT_PATH),
                },
                ensure_ascii=False,
            )
        )
        return
    if args.dry_run:
        def initial_mock(_payload: Mapping[str, object]) -> Mapping[str, object]:
            return {
                "decision": "PROPOSE",
                "steps": [{"op": "impute_linear", "params": {}, "bindings": {}}],
                "requested_observations": ["missing_run_topology"],
                "fallback": "IDENTITY",
            }

        def revision_mock(_payload: Mapping[str, object]) -> Mapping[str, object]:
            return {
                "decision": "PROPOSE",
                "steps": [{"op": "smooth_ema", "params": {}, "bindings": {}}],
                "requested_observations": ["local_variation_summary"],
                "fallback": "IDENTITY",
            }

        observation_backend = None
        if args.observe_history:
            observation_backend = ReplayAgentBackend(
                [
                    AgentResponse.valid(
                        {
                            "schema_version": "agent-envelope/1",
                            "kind": "tool_request",
                            "call_id": "history-call",
                            "tool_name": "compare_history_windows",
                            "arguments": {},
                        },
                        raw_response={"id": "dry-history-tool"},
                    ),
                    AgentResponse.valid(
                        {
                            "schema_version": "agent-envelope/1",
                            "kind": "stage_result",
                            "stage": "observe",
                            "payload": {"observation_complete": True},
                        },
                        raw_response={"id": "dry-history-result"},
                    ),
                ]
            )
        report = run(
            args.root,
            initial_proposer=initial_mock,
            revision_proposer=revision_mock,
            observation_backend=observation_backend,
            observe_history=args.observe_history,
            dataset_key=args.dataset,
            write_report=False,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "cycle_status": report["cycle_status"],
                    "proposal_count": len(report["generation_proposals"]),
                    "canonical_operator_count": report["canonical_operator_count"],
                    "dataset_key": args.dataset,
                    "report_written": False,
                    "mock_skill_discarded": True,
                },
                ensure_ascii=False,
            )
        )
        return
    report = run(
        args.root,
        observe_history=args.observe_history,
        dataset_key=args.dataset,
        model=args.model or DEFAULT_MODEL,
        base_url=args.base_url,
        report_path=args.report,
    )
    print(json.dumps({
        "final_status": report["final_status"],
        "llm_api_call_count": report["llm"]["api_call_count"],
        "report": str(
            args.report
            if args.report is not None
            else DATASET_CONFIGS[args.dataset]["report_path"]
        ),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
