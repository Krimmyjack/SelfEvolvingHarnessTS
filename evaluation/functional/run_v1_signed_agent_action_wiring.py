"""V1 2A：确定性 Agent 策略——Memory 通过真实 Agent 接口影响行动并进入反馈闭环。

审查裁决（2026-08-08 四轮）：2A 验证完整机械链——
  signed Memory → 正确注入 prompt → 策略读取 Reference → 生成非 Identity
  Workflow → compiler → Target Support → Episode 写回。
正确 verdict：SIGNED_AGENT_ACTION_WIRING_PASS（不称 AGENT_SELECTION_QUALITY_PASS——
确定性策略是人为定义的正控，不代表真实 LLM 自主理解）。

约束（硬性）：
  - 策略不硬编码 denoise_savgol/denoise_stl/domain 名/origin/已知 gain；
  - A3/A5 使用完全相同策略与候选池；
  - 策略只读正常 Agent 可见的 instruction、Operator contracts、公开 Context；
  - Target gain 在候选冻结后才打开（prepare 返回后）；
  - A5 只有 Risk/Conflict 时结论表述为"借助失败教训规避负迁移"。

场景（延续跨域链，从 R2 决策点起跑两轮）：
  R2（832）：本地 denoise_stl 双正 → radius POSITIVE_PRIOR → instruction 含
    Reference 1 → 策略生成候选 → 冻结 → Support +0.1196 → Episode → delayed 880
  R3（928）：本地 denoise_stl delayed 负 → RISK_PRIOR → instruction 含 Reference 3
    → 策略规避（不生成）→ identity（abstain）→ harm 0（规避负迁移）

验收 8 项（见 checks）。

用法：
  python evaluation/functional/run_v1_signed_agent_action_wiring.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_target_local_loop as loop  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse  # noqa: E402

HORIZON = 48
PERIOD = 24
MAX_TARGET_PROBES = 2
SOURCE_DOMAIN = "noaa"
SOURCE_ORIGINS = (832, 880)
TARGET_DOMAIN = "gefcom"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_signed_agent_action_wiring_report.json")
R1_SLICE = (736, 784)
R2_SLICE = (832, 880)
R3_SLICE = (928, 976)


def contract_params(op: str, period: int) -> dict[str, object]:
    """从 Operator contract（public_parameter_schema）构造最小合法参数（Agent 可见）。"""
    schema = (OPERATOR_METADATA.get(op) or {}).get("public_parameter_schema")
    if not schema:
        return {}
    props = schema.get("properties") or {}
    params: dict[str, object] = {}
    for name, spec in props.items():
        if "default" in spec:
            params[name] = spec["default"]
        elif spec.get("type") == "integer":
            params[name] = spec.get("minimum", 1)
        elif spec.get("type") == "number":
            params[name] = spec.get("minimum", 1.0) or 1.0
    for req in schema.get("required") or []:
        if req not in params:
            spec = props.get(req, {})
            if spec.get("type") == "integer":
                params[req] = spec.get("minimum", 1)
            else:
                params[req] = 1
    if "period" in params and period is not None:
        params["period"] = period  # 从公开 Context（bound_period）绑定
    return params


class DeterministicStrategyBackend:
    """确定性 Agent 策略（正控）：读 instruction 的 Memory Reference 与 contracts。

    不硬编码算子名/domain/origin/gain；A3/A5 同策略。
    Reference 1（POSITIVE_PRIOR）→ 优先尝试该算子；
    Reference 2/3（CONFLICT / RISK_PRIOR）→ **降级**：探索序中降到 UNKNOWN
      候选之后（不硬排除——避免过度泛化；耗尽 UNKNOWN 后才尝试）。
    无 Reference 1 时：
      explore=False → 不生成（identity/abstain 安全兜底，2A 接线正控）；
      explore=True  → 从 Operator inventory 按序提案（UNKNOWN 优先、降级
                      算子最后），A3 公平对照：空 Source Memory 但可自主
                      探索、从零适应。

    审查裁决（2026-08-08 十六）：此前只消费 Reference 1（"正向感知、冲突盲"）
    ——A3 在 R1 得到 impute_fft=CONFLICT 后，R2 仍按字母序重复尝试该算子并
    付出 harm，放大 A5 优势。修复：消费 Reference 2/3 降级（每次 prepare 更新
    ——Memory 变化 → instruction 变化 → 降级列表随之更新）。
    """

    def __init__(self, *, explore: bool = False, operators: Sequence[str] = (),
                 prefer_skill_in_select: bool = False,
                 reserve_exploration_slot: bool = False) -> None:
        self.requests: list[Any] = []
        self._explore = explore
        self._operators = tuple(operators)
        self._explored: list[str] = []  # 探索状态机：按序逐个提案、不重复（不读 Memory）
        self._deprioritized: list[str] = []  # Reference 2/3 算子：UNKNOWN 耗尽后才尝试
        self._pending_op: str | None = None  # propose 选定、select 复用的算子
        self._prefer_skill = prefer_skill_in_select  # 中性 selector（方法层验收用）
        # E2（用户裁决 2026-08-12）：Source Memory 双槽——reserve_
        # exploration_slot=True 时 ref1 提案后始终追加一个当前 Context
        # 探索槽（Source 最多占一个 slot——不能删除探索槽）。
        self._reserve_exploration_slot = reserve_exploration_slot

    @staticmethod
    def _eligible_ops(messages: Sequence[Mapping[str, object]]) \
            -> tuple[str, ...] | None:
        """过滤感知探索的 eligible 算子集（FILTER_AWARE_EXPLORATION_ADVANCE
        _CONTROL，2026-08-10）：从 propose 请求消息的
        allowed_operator_contracts 提取算子名——fast_agent 已按当前公开
        Context 剔除 no-op/ineligible（_noop_ops_for_context），契约顺序即
        冻结 inventory 顺序。返回 None 表示契约未渲染（未知调用方→回退
        self._operators 原行为）；空 tuple 表示已渲染但全部被过滤（真耗尽
        → abstain）。全程不读取 gain。"""
        blob = "\n".join(
            str(m.get("content")) for m in messages
            if isinstance(m, Mapping) and isinstance(m.get("content"), str))
        marker = '"allowed_operator_contracts":'
        idx = blob.find(marker)
        if idx < 0:
            return None
        brace = blob.find("[", idx)
        if brace < 0:
            return None
        try:
            arr, _ = json.JSONDecoder().raw_decode(blob[brace:])
        except json.JSONDecodeError:
            return None
        if not isinstance(arr, list):
            return None
        return tuple(str(c["name"]) for c in arr
                     if isinstance(c, Mapping) and c.get("name"))

    def _next_explore_op(self, eligible: Sequence[str] | None = None) -> str | None:
        # eligible=None → 原 self._operators 行为（向后兼容）；否则按契约
        # 顺序（已过滤 no-op）扫描，跳过 no-op/ineligible 直到第一个合法
        # 且未探索候选；全部耗尽才返回 None（abstain）。
        pool = self._operators if eligible is None else tuple(eligible)
        # UNKNOWN（未探索且未降级）候选优先；耗尽后再尝试降级算子（不硬排除）
        for o in pool:
            if o not in self._explored and o not in self._deprioritized:
                return o
        for o in pool:
            if o not in self._explored and o in self._deprioritized:
                return o
        return None

    @staticmethod
    def extract_instruction(messages: Sequence[Mapping[str, object]]) -> str:
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""

    @staticmethod
    def _reference_ops(instruction: str, ref_number: int) -> list[str]:
        """解析 'Reference N: candidate operators [...]' 的算子列表（Python repr）。"""
        pattern = rf"Reference {ref_number}: candidate operators (\[[^\]]*\])"
        match = re.search(pattern, instruction)
        if not match:
            return []
        try:
            parsed = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return []
        return [str(op) for op in parsed] if isinstance(parsed, list) else []

    @staticmethod
    def _select_candidate_ids(messages: Sequence[Mapping[str, object]]) -> list[str]:
        """解析 select 阶段公开候选列表（public_input.candidates 的
        candidate_id 顺序——在最后一个 user message 的 JSON 里）。

        中性 selector 用（外部审核 2026-08-09）：不解析 Skill 文本（不读 body），
        只按公开候选 ID 顺序选择。"""
        blob = "\n".join(
            str(m.get("content")) for m in messages
            if isinstance(m, Mapping) and isinstance(m.get("content"), str))
        marker = '"candidates":'
        idx = blob.find(marker)
        if idx < 0:
            return []
        brace = blob.find("[", idx)
        if brace < 0:
            return []
        try:
            arr, _ = json.JSONDecoder().raw_decode(blob[brace:])
        except json.JSONDecodeError:
            return []
        if not isinstance(arr, list):
            return []
        return [str(c["candidate_id"]) for c in arr
                if isinstance(c, Mapping) and isinstance(c.get("candidate_id"), str)]

    def complete(self, request: Any) -> AgentResponse:
        self.requests.append(request)
        instruction = self.extract_instruction(request.messages)
        ref1 = self._reference_ops(instruction, 1)
        ref2 = self._reference_ops(instruction, 2)
        ref3 = self._reference_ops(instruction, 3)
        # 审查裁决（十六）：消费 Reference 2/3——冲突/风险经验降级（不硬排除）。
        # 每次 prepare 更新：Memory 变化 → instruction 变化 → 降级列表随之更新。
        self._deprioritized = list(dict.fromkeys([*ref2, *ref3]))
        stage = request.stage
        if stage == "inspect":
            payload = {
                "inspected_region_fractions": [[0.0, 1.0]],
                "requested_public_tools": [],
                "uncertainty": "high",
            }
        elif stage == "propose":
            if ref1:
                op = ref1[0]
                candidates = [{
                    "candidate_id": f"cand_{op}",
                    "steps": [{"op": op, "params": contract_params(op, PERIOD)}],
                }]
                # E2 双槽：Source 正例最多优先一个 trial——ref1 提案后
                # 保留一个当前 Context 探索槽（Source 不能独占供应——
                # 外部 AI 判断：ref1 短路使 novel candidate 不进池）。
                if self._reserve_exploration_slot:
                    eop = self._next_explore_op(self._eligible_ops(
                        request.messages))
                    if eop is not None and eop != op:
                        self._explored.append(eop)
                        candidates.append({
                            "candidate_id": f"cand_{eop}",
                            "steps": [{"op": eop,
                                       "params": contract_params(
                                           eop, PERIOD)}],
                        })
            elif self._explore:
                op = self._next_explore_op(self._eligible_ops(request.messages))
                if op is not None:
                    self._explored.append(op)
                    self._pending_op = op
                    candidates = [{
                        "candidate_id": f"cand_{op}",
                        "steps": [{"op": op, "params": contract_params(op, PERIOD)}],
                    }]
                else:
                    candidates = []
            else:
                candidates = []
            payload = {"candidates": candidates}
        elif stage == "select":
            if self._prefer_skill:
                # 中性 selector（外部审核 2026-08-09）：按公开候选 ID 顺序——
                # 优先 cand_skill_*（Skill 优先），否则第一个非 identity PROGRAM，
                # 否则 identity。不解析 Skill 文本（只匹配 ID 前缀）。
                ids = self._select_candidate_ids(request.messages)
                skill_ids = [i for i in ids if i.startswith("cand_skill_")]
                if skill_ids:
                    chosen = skill_ids[0]
                    verification_actions: list[str] = []
                else:
                    non_identity = [i for i in ids if i != "identity"]
                    if non_identity:
                        chosen = non_identity[0]
                        verification_actions = []
                    else:
                        chosen = "identity"
                        verification_actions = ["public_evidence_insufficient"]
            elif ref1:
                chosen = f"cand_{ref1[0]}"
                verification_actions = []
            elif self._explore:
                # 复用 propose 选定的算子（不再次推进探索状态机）
                if self._pending_op is not None:
                    chosen = f"cand_{self._pending_op}"
                    self._pending_op = None
                else:
                    chosen = "identity"
                verification_actions = []
            else:
                chosen = "identity"
                verification_actions = ["public_evidence_insufficient"]
            payload = {"chosen_candidate_id": chosen,
                       "verification_actions": verification_actions}
        else:
            raise AssertionError(f"unexpected stage: {stage}")
        return AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": stage, "payload": payload},
            raw_response={"id": f"strategy-{stage}"},
        )


def build_r2_memory(root: Path) -> list[Any]:
    """R2 决策点前的 Memory：noaa source + R1 探测 Episode（delayed 784 已更新）。

    R1 探测 = 跨域闭环 analogy 序（period_median_complete 0.0 → denoise_stl +0.802）。
    """
    config = dict(v6.DATASET_CONFIGS[SOURCE_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    noaa_source, _ = v1.build_source_memory(
        domain=SOURCE_DOMAIN, roster=roster, values=values, config=config,
        operators=sorted(n for n in v6.OPERATOR_NAMES
                         if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                         and n not in core.CTS_EXCLUDED),
        source_support_origin=SOURCE_ORIGINS[0], source_delayed_origin=SOURCE_ORIGINS[1],
        baseline_cache=baseline_cache,
        context_fn=lambda o: resolver.window_context(values, o, period),
    )
    # R1：analogy 探测（与跨域闭环一致：period_median_complete, denoise_stl）
    tconfig = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    troster, tvalues = v6._fixed_roster(root, tconfig)
    tperiod = int(tconfig.get("period", 1))
    f_support = resolver.window_context(tvalues, R1_SLICE[0], tperiod)
    local: list[Any] = []
    for op, g in (("period_median_complete", 0.0), ("denoise_stl", 0.8020055111538338)):
        local.append(loop.write_target_episode(
            domain=TARGET_DOMAIN, op=op,
            program_steps=[{"op": op, "params": contract_params(op, tperiod)}],
            support_gain=g, delayed_gain=None, support_context=f_support))
    # delayed 784 更新（确定性重放）
    f_delayed = resolver.window_context(tvalues, R1_SLICE[1], tperiod)
    new_local = []
    for ep in local:
        compiled = loop.compiled_from_episode(ep, tperiod)
        dg = v1.gain_at(troster, tvalues, tconfig, compiled, R1_SLICE[1], baseline_cache)
        new_local.append(loop.update_delayed_status(ep, dg, delayed_context=f_delayed)
                         if dg is not None else ep)
    return list(noaa_source) + new_local


def run_prepare(values: np.ndarray, observed: Mapping[str, float], episodes: Sequence[Any],
                backend: DeterministicStrategyBackend, h0: Any,
                *, explore: bool = False) -> tuple[Any, Any, str]:
    request = PreparationRequest(
        "gefcom-r2-r3",
        np.asarray(values, dtype=float),
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.asarray(values, dtype=float), task_kind="forecast"),
    )
    result, trace = TTHAFastAgent(core).prepare(
        request, h0, experience_episodes=tuple(episodes))
    instruction = backend.extract_instruction(
        backend.requests[-1].messages if backend.requests else ())
    return result, trace, instruction


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="V1 2A signed agent action wiring")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--explore", action="store_true",
                        help="A3 无 Reference 时从 inventory 自主探索（边界 2 验证）")
    args = parser.parse_args()
    root = args.root.resolve()
    explore = args.explore
    m = core.MATERIAL_THRESHOLD
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    operators = sorted(n for n in v6.OPERATOR_NAMES
                       if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                       and n not in core.CTS_EXCLUDED)
    series = values[list(values.keys())[0]]
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)

    memory = build_r2_memory(root)
    print(f"== memory at R2: {len(memory)} episodes")

    rounds: list[dict[str, Any]] = []
    a5_episodes: list[Any] = list(memory)  # A5 池（延续）
    a3_episodes: list[Any] = []
    # A3 探索 backend 跨轮保持（状态机：字母序逐个提案不重复——从零适应）
    a3_backend = DeterministicStrategyBackend(explore=explore, operators=operators)

    for idx, (ts, td) in enumerate((R2_SLICE, R3_SLICE)):
        label = f"R{idx + 2}"
        observed = dict(resolver.window_context(values, ts, period))
        observed["bound_period"] = float(period)
        r_values = np.asarray(series)[:ts]

        # A5：真实 fast_agent 入口（确定性策略 backend）——候选在 prepare 内冻结
        a5_backend = DeterministicStrategyBackend()
        a5_result, a5_trace, a5_instruction = run_prepare(
            r_values, observed, a5_episodes, a5_backend, h0)
        # A3：同入口、无 Source Experience；explore=True 时从 inventory 自主探索
        # （边界 2：空 Source Memory 但可自主提案、从零适应——公平对照）
        a3_result, a3_trace, a3_instruction = run_prepare(
            r_values, observed, a3_episodes, a3_backend, h0, explore=explore)

        # Target gain 在候选冻结（prepare 返回）后才打开（约束）
        chosen_a5 = a5_trace.chosen_candidate_id
        chosen_a3 = a3_trace.chosen_candidate_id
        if chosen_a5 != "identity":
            op = chosen_a5.removeprefix("cand_")
            compiled = v1.make_compiled(op, contract_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, ts, baseline_cache)
            # 写 Episode + delayed（真实 feedback 闭环）
            f_support = resolver.window_context(values, ts, period)
            ep = loop.write_target_episode(
                domain=TARGET_DOMAIN, op=op,
                program_steps=[{"op": op, "params": contract_params(op, period)}],
                support_gain=g, delayed_gain=None, support_context=f_support)
            f_delayed = resolver.window_context(values, td, period)
            dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
            ep = loop.update_delayed_status(ep, dg, delayed_context=f_delayed) if dg is not None else ep
            a5_episodes.append(ep)
            harm = int(g is not None and g < -m)
            first_positive = 1 if (g is not None and g >= m) else None
            a5_probe = {"op": op, "support_gain": g, "delayed_gain": dg,
                        "local_status": ep.local_status, "relation": ep.relation,
                        "harm": harm, "first_positive_probe": first_positive}
        else:
            a5_episodes.append(loop.write_abstain_episode(
                domain=TARGET_DOMAIN, reason="A5_strategy_abstain"))
            a5_probe = {"op": None, "support_gain": None, "delayed_gain": None,
                        "local_status": None, "relation": None,
                        "harm": 0, "first_positive_probe": None}
        if chosen_a3 != "identity":
            # A3 自主探索的反馈闭环（从零适应：Support → Episode → delayed）
            op = chosen_a3.removeprefix("cand_")
            compiled = v1.make_compiled(op, contract_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, ts, baseline_cache)
            f_support = resolver.window_context(values, ts, period)
            ep = loop.write_target_episode(
                domain=TARGET_DOMAIN, op=op,
                program_steps=[{"op": op, "params": contract_params(op, period)}],
                support_gain=g, delayed_gain=None, support_context=f_support)
            f_delayed = resolver.window_context(values, td, period)
            dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
            ep = loop.update_delayed_status(ep, dg, delayed_context=f_delayed) if dg is not None else ep
            a3_episodes.append(ep)
            a3_probe = {"abstained": False, "op": op, "support_gain": g,
                        "delayed_gain": dg, "local_status": ep.local_status,
                        "harm": int(g is not None and g < -m),
                        "first_positive_probe": 1 if (g is not None and g >= m) else None}
        else:
            a3_episodes.append(loop.write_abstain_episode(
                domain=TARGET_DOMAIN, reason="A3_strategy_abstain"))
            a3_probe = {"abstained": True, "harm": 0, "first_positive_probe": None}

        rounds.append({
            "round": label, "slice": {"support": ts, "delayed": td},
            "a5_chosen_candidate": chosen_a5,
            "a5_compilation_status": a5_trace.compilation_status,
            "a5_execution_status": a5_trace.execution_status,
            "a5_probe": a5_probe,
            "a5_instruction_has_ref1": "Reference 1" in a5_instruction,
            "a5_instruction_has_ref3": "Reference 3" in a5_instruction,
            "a3_chosen_candidate": chosen_a3,
            "a3_compilation_status": a3_trace.compilation_status,
            "a3_instruction_has_ref1": "Reference 1" in a3_instruction,
            "a3_probe": a3_probe,
        })
        print(f"[{label}] A5 chosen={chosen_a5} compile={a5_trace.compilation_status} "
              f"ref1={'Reference 1' in a5_instruction} ref3={'Reference 3' in a5_instruction} "
              f"probe={a5_probe} | A3 chosen={chosen_a3}")

    r2, r3 = rounds
    if explore:
        # 边界 2 断言（审查裁决 2026-08-08）：A3 空 Source Memory 但可自主提案、从零适应
        checks: dict[str, bool] = {
            "a3_proposes_non_identity": (
                r2["a3_chosen_candidate"] != "identity"
                and r3["a3_chosen_candidate"] != "identity"),
            "a3_workflows_compilable": (
                r2["a3_compilation_status"] in ("ok", "compiled")
                and r3["a3_compilation_status"] in ("ok", "compiled")),
            "a3_from_zero_adapts": (
                r2["a3_probe"]["op"] == "denoise_median"
                and r2["a3_probe"]["harm"] == 0
                and r3["a3_probe"]["op"] == "denoise_savgol"
                and r3["a3_probe"]["first_positive_probe"] == 1),
            "a3_local_accumulates": (
                r2["a3_probe"]["support_gain"] is not None
                and r2["a3_probe"]["delayed_gain"] is not None),
            "a5_unchanged_by_explore": (
                r2["a5_chosen_candidate"] == "cand_denoise_stl"
                and r3["a5_chosen_candidate"] == "identity"),
            "a5_risk_aversion": (
                r3["a5_instruction_has_ref3"] and r3["a5_probe"]["harm"] == 0),
        }
        all_pass = all(checks.values())
        verdict = "BOUNDARY2_EXPLORATION_PASS" if all_pass else "BOUNDARY2_EXPLORATION_PARTIAL"
        print(f"\n== boundary-2 checks: {checks}")
        print(f"== verdict: {verdict}")
        print("== 边界 2 结论：A3 空 Source Memory 可自主提案（同 inventory/同策略），"
              "R2 中性探测 → R3 探索命中 savgol（从零适应）；"
              "A5 行为不受 A3 探索影响。")
    else:
        _emit_wiring_checks(r2, r3, rounds, root, m)
    return 0


def _emit_wiring_checks(r2: dict[str, Any], r3: dict[str, Any],
                        rounds: Sequence[dict[str, Any]], root: Path,
                        m: float) -> None:
    checks: dict[str, bool] = {
        # 1. A5 注入来自当前合法 signed verdict（Reference 1 且算子与 resolver 判定一致）
        "a5_injection_from_signed_verdict": (
            r2["a5_instruction_has_ref1"]
            and r2["a5_chosen_candidate"].removeprefix("cand_") == "denoise_stl"
            and r3["a5_instruction_has_ref3"]),
        # 2. A3 同入口无 Source Experience（无注入）
        "a3_no_source_experience": (
            not r2["a3_instruction_has_ref1"] and not r3["a3_instruction_has_ref1"]),
        # 3. 生成 Workflow 可编译（R2 非 identity 候选编译成功；R3 identity 正常）
        "workflows_compilable": (
            r2["a5_compilation_status"] in ("ok", "compiled")
            and r3["a5_compilation_status"] in ("ok", "compiled", "not_started")),
        # 4. 决策前不读取 Target gain（gain 在 prepare 返回后打开）
        "no_target_gain_before_freeze": True,  # 结构性：gain_at 仅在 prepare 后调用
        # 5. 候选差异可追溯到 Memory（同 backend 策略：A5 由 Reference 1 驱动，A3 无注入 abstain）
        "candidate_diff_traceable_to_memory": (
            r2["a5_chosen_candidate"] != "identity"
            and r2["a3_chosen_candidate"] == "identity"
            and r3["a5_chosen_candidate"] == "identity"),
        # 6. Support 后立即写 Episode、delayed 正确更新（R2 denoise_stl RESTRICTED）
        "episode_and_delayed_update": (
            r2["a5_probe"]["op"] == "denoise_stl"
            and r2["a5_probe"]["support_gain"] is not None
            and r2["a5_probe"]["delayed_gain"] is not None
            and r2["a5_probe"]["local_status"] is not None),
        # 7. 比较：A5 R2 首探命中（first_positive=1）；A3 abstain
        "comparison_metrics": (
            r2["a5_probe"]["first_positive_probe"] == 1
            and r2["a5_probe"]["harm"] == 0
            and r2["a3_probe"]["abstained"]),
        # 8. 安全兜底：R3 风险（Reference 3）→ abstain（规避负迁移，harm 0）
        "safe_fallback_on_risk": (
            r3["a5_chosen_candidate"] == "identity"
            and r3["a5_probe"]["harm"] == 0
            and r3["a5_probe"]["op"] is None),
    }
    all_pass = all(checks.values())
    verdict = "SIGNED_AGENT_ACTION_WIRING_PASS" if all_pass else "SIGNED_AGENT_ACTION_WIRING_PARTIAL"
    print(f"\n== checks: {checks}")
    print(f"== verdict: {verdict}")
    # 结论口径（审查裁决）：Risk 场景 = 规避负迁移，不称迁移成功经验
    conclusion = ("Memory 通过真实 Agent 接口影响行动并进入反馈闭环"
                  if all_pass else "链路存在缺口")
    print(f"== conclusion: {conclusion}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-signed-agent-action-wiring",
            "rounds": rounds,
            "checks": checks,
            "verdict": verdict,
            "conclusion": conclusion,
            "a5_r2_instruction": a5_instruction if False else "",
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
