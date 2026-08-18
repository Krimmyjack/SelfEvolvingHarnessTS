"""Sealed-Target A5/A3（外部审核第七轮条件批准，docs/V1_SEALED_TARGET_A5A3_
FROZEN_SLICE.md）。

核心验证：同一 dataset（monash:traffic_hourly）的**两个互斥 certified-virgin
series cohort**——Source cohort @600/@648 探测冻结 ≤2 Episode（固定计划不挑
正）→ Target cohort @792/@840/@888/@936 保持 sealed 直到 Source 冻结 →
A5（Source Memory）vs A3（空）在相同 Agent/动作空间/预算下比较。

口径："同 dataset、跨 series-cohort Experience 复用"（不是跨域，不是同
cohort 早期反馈）。

冻结纪律（审核第七轮）：
- Runner 局部过滤 exposure_class == certified_virgin（不改 _fixed_roster）；
- 两组 UID 冻结（前 20 = Source，次 20 = Target，互斥）；
- R1 Support → LOCAL_DRAFT → delayed @840 更新（LOCAL_ACTIVE/CONFLICT/
  RESTRICTED）→ R2 prepare @888 → R2 delayed @936 全生命周期；
- A5/A3 各自独立 Episode / Skill fork / method 实例，禁止跨臂写回；
- verdict 六档（PASS / SAME_NO_BENEFIT / PARTIAL / NEGATIVE /
  NO_APPLICABLE_SOURCE_MEMORY / INFEASIBLE_NO_HEADROOM）；
- abstention 单独报告，不因次数高自动判负；
- proposal 数与 first-positive Support receipt index 分开报告（REJ 不计
  receipt）。

零 LLM。

用法：
  python evaluation/functional/run_v1_sealed_a5_a3.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent, _actionable_operators, _allowed_operators,
    _noop_ops_for_context)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway, extract_public_features  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

DOMAIN = "monash:traffic_hourly"  # 默认；--domain 可换 uci_electricity_load_diagrams / metr_la
PERIOD = 24
HORIZON = 48


def _set_domain(domain: str) -> None:
    """运行时切换 dataset_id（--domain；模块级常量，main 内首行调用）。"""
    global DOMAIN  # noqa: PLW0603
    DOMAIN = domain
CONTEXT_LENGTH = 192
SOURCE_ORIGIN = 600
SOURCE_DELAYED = 648
R1_ORIGIN = 792
R1_DELAYED = 840
R2_ORIGIN = 888
R2_DELAYED = 936
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005
H0_ROOT = Path("methods/ttha/harness/h0")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_sealed_a5_a3_report.json")
REGISTRY = Path("artifacts/frozen/benchmark_v02/series_registry.jsonl")
CLEAN_BASE = Path("data/benchmark_v0_2/clean_base")

ANCHORS = (312, 372, 432, 492, 552, 612, 672, 732, 792, 852)


def _config() -> dict[str, object]:
    return {
        "dataset_id": DOMAIN,
        "sampling": "hourly_regular",
        "period": PERIOD,
        "anchors": ANCHORS,
        "support_origin": R1_ORIGIN,
        "selection_origin": R1_ORIGIN,
    }


def _virgin_roster(root: Path, n_source: int = 20, n_target: int = 20,
                   offset: int = 0):
    """Runner 局部 virgin 过滤（审核 P0-2）：不混入 probe_consumed。
    offset：消费下一组 virgin series（实验 2 用 offset=40——前 40 支已在
    第一次 sealed 运行消费）。"""
    rows = [
        json.loads(line)
        for line in (root / REGISTRY).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = sorted(
        (
            r for r in rows
            if r.get("dataset_id") == DOMAIN
            and r.get("exposure_class") == "certified_virgin"
            and int(r.get("length", 0)) >= R2_ORIGIN + 2 * HORIZON
        ),
        key=lambda r: str(r["series_uid"]),
    )
    assert len(eligible) >= offset + n_source + n_target, (
        f"not enough virgin series: {len(eligible)} needed "
        f"{offset + n_source + n_target}")
    source_uids = [str(r["series_uid"])
                   for r in eligible[offset:offset + n_source]]
    target_uids = [str(r["series_uid"])
                   for r in eligible[offset + n_source:offset + n_source + n_target]]
    assert set(source_uids).isdisjoint(target_uids)

    record_dirs: dict[str, Path] = {}
    for record_path in (root / CLEAN_BASE).glob("*/record.json"):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        uid = str(record.get("series_uid", ""))
        if uid in set(source_uids) | set(target_uids):
            record_dirs[uid] = record_path.parent
    values = {
        uid: np.asarray(np.load(record_dirs[uid] / "values.npy",
                                allow_pickle=False), dtype=np.float64)
        for uid in set(source_uids) | set(target_uids)
    }

    def roster(uids: Sequence[str]) -> list[dict[str, object]]:
        return [
            {"series_uid": uid,
             "role": "train" if index < 12 else "eval"}
            for index, uid in enumerate(uids)
        ]

    return (roster(source_uids), {uid: values[uid] for uid in source_uids},
            roster(target_uids), {uid: values[uid] for uid in target_uids})


class ProviderTransportError(TypeError):
    """provider/transport 故障（空 choices 或调用失败重试后仍失败）——
    继承 TypeError（fast_agent.prepare 的 except 捕获列表内）→ prepare
    FAILED + error 含类名 → runner 判 INCONCLUSIVE_PROVIDER_FAILURE
    （P0.1 用户裁决 2026-08-11）。"""


class SealedProbeBackend(wiring.DeterministicStrategyBackend):
    """探测 backend（预注册规则）：ref1 算子未试过 → 优先；已试过/REJ →
    explore 下一个未试算子（预算内不重复提案——ref1 分支也记录 explored）。

    实验装置参数（2026-08-10，LEVEL_SHIFT_CANDIDATE_AVAILABILITY_TEST）：
      max_propose_candidates：一次提案最多 N 个未探索候选（默认 1=原语义）；
      bound_params：候选参数覆盖（公开特征绑定值——post_validator 要求
        repair_level_shift 参数必须等于公开特征值）；
      force_pool：True 时从 self._operators 提案（绕过 actionable 契约
        过滤——测"修复供应后"的候选可用性，Treatment 装置）。
    """

    def __init__(self, *, explore: bool, operators: Sequence[str],
                 max_propose_candidates: int = 2,
                 bound_params: Mapping[str, Mapping[str, object]] | None = None,
                 force_pool: bool = False,
                 prefer_skill_in_select: bool = False,
                 reserve_exploration_slot: bool = False) -> None:
        super().__init__(explore=explore, operators=operators,
                         prefer_skill_in_select=prefer_skill_in_select,
                         reserve_exploration_slot=reserve_exploration_slot)
        self._max_propose = int(max_propose_candidates)
        self._bound_params = dict(bound_params or {})
        self._force_pool = bool(force_pool)

    @staticmethod
    def _public_features_from(request: Any) -> dict[str, object]:
        """从 propose 请求消息解析 public_input.features（部署可见特征，
        CONTEXT_BOUND_PROGRAM_SUPPLY 配套：绑定参数候选的数值来源）。"""
        blob = "\n".join(
            str(m.get("content")) for m in request.messages
            if isinstance(m, Mapping) and isinstance(m.get("content"), str))
        marker = '"features":'
        idx = blob.find(marker)
        if idx < 0:
            return {}
        brace = blob.find("{", idx)
        if brace < 0:
            return {}
        try:
            obj, _ = json.JSONDecoder().raw_decode(blob[brace:])
        except json.JSONDecodeError:
            return {}
        return dict(obj) if isinstance(obj, Mapping) else {}

    def _cand(self, request: Any, op: str) -> dict[str, object]:
        """候选构造：带 public_parameter_bindings 的算子用公开特征值绑定
        参数（post_validator 硬约束：键=绑定键、值=特征值）；绑定不完整
        → 回退 bound_params 装置/contract_params（不产生非法候选）。"""
        bindings = (OPERATOR_METADATA.get(op) or {})             .get("public_parameter_bindings") or {}
        if bindings:
            features = self._public_features_from(request)
            params = {p: features[f] for p, f in bindings.items()
                      if f in features}
            if len(params) == len(bindings):
                return {"candidate_id": f"cand_{op}",
                        "steps": [{"op": op, "params": dict(params)}]}
        params = (self._bound_params.get(op)
                  or wiring.contract_params(op, PERIOD))
        return {"candidate_id": f"cand_{op}",
                "steps": [{"op": op, "params": dict(params)}]}

    def complete(self, request: Any) -> Any:
        self.requests.append(request)
        instruction = self.extract_instruction(request.messages)
        ref1 = self._reference_ops(instruction, 1)
        ref2 = self._reference_ops(instruction, 2)
        ref3 = self._reference_ops(instruction, 3)
        self._deprioritized = list(dict.fromkeys([*ref2, *ref3]))
        stage = request.stage
        if stage == "inspect":
            payload = {
                "inspected_region_fractions": [[0.0, 1.0]],
                "requested_public_tools": [],
                "uncertainty": "high",
            }
        elif stage == "propose":
            op: str | None = None
            for candidate in ref1:
                if candidate not in self._explored:
                    op = candidate
                    break
            if op is not None:
                # 审核 first fault 修复：propose 只记录 pending——**不**记为
                # explored（探测未发生前不消耗；LLM/selector abstain 后
                # 下一轮仍可提案）
                self._pending_op = op
                candidates = [self._cand(request, op)]
                # E2 双槽（用户裁决 2026-08-12）：Source 正例最多优先一个
                # trial——ref1 提案后保留一个当前 Context 探索槽（Source
                # 不能独占供应——外部 AI 判断 ref1 短路使 novel candidate
                # 不进池）。探索槽跳过 ref1 算子本身。
                if self._reserve_exploration_slot:
                    eligible = (None if self._force_pool
                                else self._eligible_ops(request.messages))
                    pool = self._operators if self._force_pool else eligible
                    eop: str | None = None
                    if pool is not None:
                        for o in pool:
                            if o != op and o not in self._explored \
                                    and o not in self._deprioritized:
                                eop = o
                                break
                        if eop is None:
                            for o in pool:
                                if o != op and o not in self._explored \
                                        and o in self._deprioritized:
                                    eop = o
                                    break
                    if eop is not None:
                        candidates.append(self._cand(request, eop))
            elif self._explore:
                eligible = (None if self._force_pool
                            else self._eligible_ops(request.messages))
                pool = self._operators if self._force_pool else eligible
                if self._max_propose > 1 and pool is not None:
                    ops_list = [o for o in pool
                                if o not in self._explored
                                and o not in self._deprioritized]
                    if not ops_list:
                        ops_list = [o for o in pool
                                    if o not in self._explored
                                    and o in self._deprioritized]
                    ops_list = ops_list[:self._max_propose]
                    self._pending_op = ops_list[0] if ops_list else None
                    candidates = [self._cand(request, o) for o in ops_list]
                else:
                    op = self._next_explore_op(pool)
                    if op is not None:
                        self._pending_op = op
                        candidates = [self._cand(request, op)]
                    else:
                        candidates = []
            else:
                candidates = []
            payload = {"candidates": candidates}
        elif stage == "select":
            ids = self._select_candidate_ids(request.messages)
            non_identity = [i for i in ids if i != "identity"]
            if non_identity:
                chosen = non_identity[0]
                verification_actions: list[str] = []
                # 审核 first fault 修复：实际选中 → 才记为 explored（探测
                # 发生）；abstain（identity）→ 不消耗，下一轮仍可提案
                if chosen.startswith("cand_"):
                    _op = str(chosen[len("cand_"):])
                    if _op and _op not in self._explored:
                        self._explored.append(_op)
            else:
                chosen = "identity"
                verification_actions = ["public_evidence_insufficient"]
            self._pending_op = None
            payload = {"chosen_candidate_id": chosen,
                       "verification_actions": verification_actions}
        else:
            raise AssertionError(f"unexpected stage: {stage}")
        return wiring.AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": stage, "payload": payload},
            raw_response={"id": f"sealed-probe-{stage}"},
        )


def _actionable_at(root: Path, series: np.ndarray, origin: int) -> tuple[str, ...]:
    """verifier 实测可行动算子（动作合法性；不含 Context 前提过滤）。"""
    h0 = compile_snapshot(root / H0_ROOT, verify_lock=False)
    request = PreparationRequest(
        "sealed-a5a3",
        series[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series[:origin], task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    return _actionable_operators(request, series[:origin], view,
                                 _allowed_operators(request))


def _actionable_ops(root: Path, series: np.ndarray, origin: int,
                    observed: Mapping[str, object]) -> tuple[str, ...]:
    """探测动作空间 = verifier 实测可行动 − Context 前提 no-op（实验 1：
    与 propose contracts 过滤一致，backend 探索序不浪费在 no-op 上）。"""
    ops = _actionable_at(root, series, origin)
    request = PreparationRequest(
        "sealed-a5a3",
        series[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )
    noop = _noop_ops_for_context(request)
    return tuple(o for o in ops if o not in noop)


def _request(series0: np.ndarray, values: Mapping[str, Any], origin: int,
             observed_extra: Mapping[str, object] | None = None) -> PreparationRequest:
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    if observed_extra:
        observed.update(observed_extra)
    return PreparationRequest(
        "sealed-a5a3",
        series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )


def probe_round(method: TTHAMethod, executor: ScopeExecutor,
                series0: np.ndarray, values: Mapping[str, Any], origin: int,
                *, round_name: str, budget: int = 2) -> dict[str, Any]:
    """一决策点探测循环（预注册）：每探测 = prepare（正常入口）→ 沿 chosen
    Program 的 evaluate → Support receipt。gain ≥ M → first-positive 停止；
    REJ 不计 receipt 不写 Episode；每有效 Support 立即写本臂 Episode。"""
    probes: list[dict[str, Any]] = []
    for i in range(budget):
        result = method.prepare(_request(series0, values, origin))
        trace = method.last_trace
        chosen = trace.chosen_candidate_id
        steps = None
        if result.program is not None:
            steps = tuple((op, dict(pr))
                          for op, pr in result.program.execution_steps())
        entry: dict[str, Any] = {"proposal": i + 1, "chosen": chosen}
        if chosen == "identity" or not steps:
            entry["kind"] = "abstain"
            probes.append(entry)
            break
        receipt = executor.evaluate(tuple(steps), origin)
        gain = (float(receipt.gain) if receipt.gain is not None else None)
        entry.update({"op": steps[0][0], "gain": gain,
                      "passed": receipt.verification.passed})
        if not receipt.verification.passed:
            entry["kind"] = "reject"  # REJ：不计 receipt，不写 Episode
            # 审核 first fault 修复：verifier 拒绝 → 记 explored/rejected
            # （不可行动，下一轮不重复提案）
            try:
                method.fast_agent.core.backend._explored.append(steps[0][0])  # noqa: SLF001
            except Exception:
                pass
            probes.append(entry)
            continue
        entry["kind"] = "support"
        entry["first_positive"] = bool(gain is not None and gain >= MATERIAL)
        # 立即写本臂 Episode（Support 后）
        ep = tll.write_target_episode(
            domain=DOMAIN, op=steps[0][0],
            episode_id_suffix=f"_{round_name}_p{i + 1}",
            program_steps=[{"op": s[0], "params": dict(s[1])} for s in steps],
            support_gain=float(gain) if gain is not None else 0.0,
            delayed_gain=None,
            support_context=resolver.window_context(values, origin, PERIOD))
        method.append_experience_episode(ep)
        entry["episode_id"] = ep.episode_id
        entry["relation"] = ep.relation
        probes.append(entry)
        if gain is not None and gain >= MATERIAL:
            break  # first-positive 命中，停止该点探测
    return {"round": round_name, "origin": origin, "probes": probes}


def open_delayed(method: TTHAMethod, executor: ScopeExecutor,
                 series0: np.ndarray, values: Mapping[str, Any],
                 origin: int, *, round_name: str,
                 episode_id: str | None = None,
                 steps_override: Sequence[tuple[str, Mapping]] | None = None,
                 ) -> dict[str, Any] | None:
    """打开 delayed（预注册）：evaluate(steps, origin+HORIZON) → 更新**指定
    本轮 Episode**（原位替换）。
    episode_id：本轮 support Episode 的 id（审核 first fault 修复：必须
    显式接收本轮 episode_id；本轮无 Episode（如 abstain）→ 返回 None、
    不更新任何历史/Source Episode——**不允许默认取 Memory 最后一条**）；
    steps_override 非空 = 只评估指定 steps 不更新 Episode（Skill delayed
    utility）。两者都无 → 返回 None。"""
    if steps_override is None and episode_id is None:
        return None
    eps = list(tuple(method._experience_episodes))  # noqa: SLF001
    ep: Any | None = None
    if steps_override is not None:
        steps = tuple((str(op), dict(pr)) for op, pr in steps_override)
    else:
        ep = next((e for e in eps if getattr(e, "episode_id", "") == episode_id),
                  None)
        if ep is None:
            return None
        steps_raw = ep.context_summary["program_geometry"]["program_steps"]
        steps = tuple((str(s["op"]), dict(s["params"])) for s in steps_raw)
    receipt = executor.evaluate(steps, origin + HORIZON)
    gain = (float(receipt.gain) if receipt.gain is not None else None)
    ep2 = None
    if ep is not None:
        ep2 = tll.update_delayed_status(
            ep, gain if gain is not None else 0.0,
            delayed_context=resolver.window_context(values, origin + HORIZON,
                                                    PERIOD))
        method.update_experience_episode(ep2)
    return {"round": round_name,
            "episode_id": ep2.episode_id if ep2 is not None else None,
            "op": str(steps[0][0]), "delayed_gain": gain,
            "relation": ep2.relation if ep2 is not None else None,
            "status": ep2.local_status if ep2 is not None else None,
            "delayed_origin": origin + HORIZON}


def write_skill(root: Path, snapshot: Any, steps: Sequence[tuple[str, Mapping]],
                skill_id: str, status: str, *, rationale: str) -> tuple[Any, SnapshotStore, Path]:
    """正向 Workflow 写为 Skill（审核：R1 delayed 后状态写盘）。状态（
    LOCAL_ACTIVE / RESTRICTED 等）写入 body 文本（不新增 schema 字段——
    编译器 schema 不变）。返回 (patched_snapshot, store, fork_root)。"""
    store = SnapshotStore(root)
    parent = store.materialize(snapshot)
    fork_root = store.fork(parent, edit_id=skill_id)
    learned_dir = fork_root / "skills" / "learned"
    learned_dir.mkdir(parents=True, exist_ok=True)
    skill_body = (
        "Sealed A5/A3 Target-local Workflow.\n"
        f"Status: {status}\n"
        f"Rationale: {rationale}\n"
        "Frozen program steps:\n"
        + json.dumps([{"op": op, "params": dict(pr)} for op, pr in steps],
                     sort_keys=True) + "\n"
    )
    skill_entry = {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": 1,
        "body": skill_body,
        "observable_applicability": {"const": True},
        "allowed_tools": [],
        "risk_guards": {"max_modified_fraction": 0.35,
                        "preserve_outside_candidate_region": True},
    }
    (learned_dir / f"{skill_id}.json").write_text(
        json.dumps(skill_entry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    patched = compile_snapshot(fork_root, verify_lock=False)
    return patched, store, fork_root


def _metric_summary(arm_rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """预注册指标（§5）：proposal 数（含 REJ）、first-positive Support
    receipt index（REJ 不计）、harm、abstention。"""
    proposals = 0
    first_pos: int | None = None
    harm_count = 0
    harm_sum = 0.0
    abstentions = 0
    support_idx = 0
    for rd in arm_rounds:
        for p in rd["probes"]:
            proposals += 1
            if p.get("kind") == "abstain":
                abstentions += 1
            elif p.get("kind") == "reject":
                abstentions += 1
            elif p.get("kind") == "support":
                support_idx += 1
                g = p.get("gain")
                if g is not None and g < -MATERIAL:
                    harm_count += 1
                    harm_sum += -g
                if p.get("first_positive") and first_pos is None:
                    first_pos = support_idx
    return {"proposal_count": proposals,
            "first_positive_support_receipt_index": first_pos,
            "harm_count": harm_count, "harm_magnitude_sum": round(harm_sum, 6),
            "abstention_count": abstentions}


class LLMSelectBackend(SealedProbeBackend):
    """inspect/propose 确定性（继承）；select 用真实 LLM（实验 4 冻结接口：
    候选/顺序/Context 固定、Memory 渲染进 instruction、luna temp=0）。
    构造时传入 context_plain（deployment-visible Context）。"""

    def __init__(self, *, explore: bool, operators: Sequence[str],
                 client: Any, context_plain: Mapping[str, object],
                 model: str = "gpt-5.6-luna",
                 max_propose_candidates: int = 2,
                 bound_params: Mapping[str, Mapping[str, object]] | None = None,
                 force_pool: bool = False) -> None:
        super().__init__(explore=explore, operators=operators,
                         max_propose_candidates=max_propose_candidates,
                         bound_params=bound_params, force_pool=force_pool)
        self._client = client
        self._model = model
        self._context = dict(context_plain)
        self._select_logs: list[dict[str, Any]] = []

    def complete(self, request: Any) -> Any:
        if request.stage != "select":
            return super().complete(request)
        instruction = self.extract_instruction(request.messages)
        # 提取 Reference 段（渲染前缀——只含 signed Memory，不含 TTHA 系统
        # 指令："abstain when public evidence does not justify a repair" 等
        # 系统规则重复出现会让 LLM 保守 abstain——实验 4 的 prompt 只放
        # Reference 段且选 Skill 成功）
        _ref_start = instruction.find("The following references")
        if _ref_start >= 0:
            _ref_end = instruction.find("You are the TTHA")
            if _ref_end > _ref_start:
                memory_text = instruction[_ref_start:_ref_end].strip()
            else:
                memory_text = instruction[_ref_start:].strip()
        else:
            memory_text = ""
        ids = self._select_candidate_ids(request.messages)
        # 解析候选 steps（public_input.candidates JSON）
        blob = "\n".join(
            str(m.get("content")) for m in request.messages
            if isinstance(m, Mapping) and isinstance(m.get("content"), str))
        marker = '"candidates":'
        idx = blob.find(marker)
        cands: list[dict[str, Any]] = []
        if idx >= 0:
            brace = blob.find("[", idx)
            if brace >= 0:
                try:
                    arr, _ = json.JSONDecoder().raw_decode(blob[brace:])
                except json.JSONDecodeError:
                    arr = []
                for c in arr if isinstance(arr, list) else []:
                    if isinstance(c, Mapping) and c.get("candidate_id"):
                        steps = c.get("steps") or []
                        op = (str(steps[0]["op"]) if steps
                              and isinstance(steps[0], Mapping) else "?")
                        cands.append({"candidate_id": str(c["candidate_id"]),
                                     "op": op,
                                     "params": (dict(steps[0]["params"])
                                                if steps
                                                and isinstance(steps[0], Mapping)
                                                else {})})
        lines = [
            "You are the fast-path selector of a time-series preprocessing "
            "harness.",
            "Choose exactly one candidate to execute at the support decision "
            "point.",
            "",
            "== Candidate programs (fixed order) ==",
        ]
        for i, c in enumerate(cands, start=1):
            lines.append(f"  {i}. {c['candidate_id']}: {c['op']} "
                         f"params={json.dumps(c['params'], sort_keys=True)}")
        lines += [
            "",
            "== Public context at the decision point "
            "(deployment-visible) ==",
            json.dumps({k: round(float(v), 6) for k, v in sorted(
                self._context.items())}, indent=2, sort_keys=True),
            "",
            "== Experience memory (signed) ==",
            memory_text if memory_text else
            "  (no applicable signed experience)",
            "",
            "== Probe semantics (important) ==",
            "Choosing a candidate means requesting one budgeted Support "
            "probe — it does NOT mean final deployment. The current Support "
            "of a Reference 1 operator has not yet been confirmed; that is "
            "not a reason to abstain — the Support probe itself IS the "
            "confirmation process.",
            "",
            "== Risk semantics (important) ==",
            "- Weak risk (negative evidence from an uncalibrated context, "
            "'weak reference'): the operator is downgraded below UNKNOWN "
            "candidates, but this is NOT a global prohibition. After "
            "UNKNOWN candidates have been exhausted and Support budget "
            "remains, one bounded probe of a weak-risk operator is allowed "
            "— the probe itself is the confirmation.",
            "- Strong risk (context-matched or target-local negative "
            "evidence): avoid or abstain — do not probe.",
            "",
            "== Your task ==",
            f"Choose exactly one candidate ID from {ids} or 'identity' "
            "(ABSTAIN). ABSTAIN is valid when evidence is insufficient or "
            "contradictory — abstaining is not a failure.",
            'Output JSON only: {"chosen_candidate_id": "<id|identity>", '
            '"evidence_refs": ["..."], "rationale": "..."}',
        ]
        prompt = "\n".join(lines)
        chosen = "identity"
        raw_last = ""
        # 审核 first fault 修复：保留 propose 阶段设置的 _pending_op——LLM
        # 实际选中候选 → 记 explored（探测发生）；abstain（identity）→ 不
        # 消耗，下一轮仍可提案。（不要在此重设 pending：候选列表第一个是
        # identity，基于 ids[0] 重设会让 pending 恒 None → 选中不消耗 →
        # 下一轮重复提案——offset=80 运行暴露的遗漏。）
        for _ in range(2):  # 格式纠正 1 次余量（validation_retries=1 语义）
            kwargs = {"model": self._model,
                      "messages": [{"role": "user", "content": prompt}]}
            # P0.1 provider 故障处理（用户裁决 2026-08-11）：空 choices /
            # transport 异常是 provider 问题不是合法 abstain——对完全相同
            # request 自动重试一次；连续失败两次 → ProviderTransportError
            # （runner 判 INCONCLUSIVE_PROVIDER_FAILURE）。禁止对合法
            # identity/abstain/错误选择重试、禁止投票。
            resp = None
            try:
                resp = self._client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"})
            except Exception:
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception:
                    resp = None
            if resp is None or not (getattr(resp, "choices", None) or []):
                # provider 异常 → 完全相同 request 重试一次
                try:
                    resp = self._client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"})
                except Exception:
                    resp = None
                if resp is None or not (getattr(resp, "choices", None) or []):
                    raise ProviderTransportError(
                        "provider returned empty choices / transport failure "
                        "after one retry")
            raw = resp.choices[0].message.content or ""
            raw_last = raw
            try:
                chosen = str(json.loads(raw).get("chosen_candidate_id", ""))
            except json.JSONDecodeError:
                chosen = ""
            if chosen in (*ids, "identity"):
                break
            chosen = "identity"
        if chosen.startswith("cand_"):
            _op = str(chosen[len("cand_"):])
            if _op and _op not in self._explored:
                self._explored.append(_op)
        self._pending_op = None
        self._select_logs.append({"prompt": prompt, "raw": raw_last,
                                  "chosen": chosen})
        return wiring.AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": "select",
             "payload": {"chosen_candidate_id": chosen,
                         "verification_actions": []}},
            raw_response={"id": "llm-select"},
        )


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="sealed A5/A3")
    parser.add_argument("--offset", type=int, default=0,
                        help="virgin series offset（实验 2 用 --offset 40："
                             "消费下一组互斥 cohort）")
    parser.add_argument("--domain", default=DOMAIN,
                        help="dataset_id（PASS 分支：uci_electricity_load_"
                             "diagrams 第二个 sealed 确认）")
    parser.add_argument("--llm-select", action="store_true",
                        help="select 用真实 LLM（实验 4 冻结接口；默认"
                             "确定性）——metr_la sealed 真实 LLM 确认")
    args = parser.parse_args()
    root = PROJECT_ROOT
    domain = args.domain
    if domain != DOMAIN:
        _set_domain(domain)
    config = _config()
    cohort_offset = args.offset
    llm_select = args.llm_select

    # ---- 窗口/互斥断言（程序计算）----
    assert R2_ORIGIN - R1_ORIGIN == 2 * HORIZON
    assert SOURCE_ORIGIN + 2 * HORIZON <= R1_ORIGIN  # Source 窗口 ≤ R1（无未来）
    assert R2_ORIGIN + 2 * HORIZON <= 1024           # R2 delayed 窗口在数据内
    print(f"== windows: source [{SOURCE_ORIGIN},{SOURCE_ORIGIN + 2 * HORIZON}) "
          f"<= R1={R1_ORIGIN}; R1={R1_ORIGIN} R1D={R1_ORIGIN + HORIZON} "
          f"R2={R2_ORIGIN} R2D={R2_ORIGIN + HORIZON}")

    # ---- 互斥 virgin 双 cohort ----
    (src_roster, src_values, tgt_roster, tgt_values) = _virgin_roster(
        root, offset=cohort_offset)
    src_series0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                             dtype=np.float64)
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    disjoint = set(r["series_uid"] for r in src_roster).isdisjoint(
        r["series_uid"] for r in tgt_roster)
    print(f"== cohorts: source={len(src_roster)} target={len(tgt_roster)} "
          f"disjoint={disjoint}")
    h0_snapshot = compile_snapshot(root / H0_ROOT, verify_lock=False)

    # ---- 真实 LLM select（--llm-select；实验 4 冻结接口）----
    import os as _os
    from run_v1_llm_signed_selection import CountingClient as _CountingClient
    llm_client: Any | None = None
    if llm_select:
        api_key = next((_os.environ.get(k, "").strip() for k in
                        ("OPENAI_API_KEY", "AGICTO_API_KEY")
                        if _os.environ.get(k, "").strip()), None)
        if not api_key:
            print("== no api key — falling back to deterministic select")
            llm_select = False
        else:
            import openai
            llm_client = _CountingClient(
                openai.OpenAI(api_key=api_key,
                              base_url="https://api.agicto.cn/v1"),
                max_calls=20)  # 预算：4 决策点 × 2 臂 × ≤2 格式纠正 + 余量

    def _make_backend(operators: Sequence[str],
                      context_plain: Mapping[str, object]) -> Any:
        if llm_select:
            return LLMSelectBackend(explore=True, operators=operators,
                                    client=llm_client,
                                    context_plain=context_plain)
        return SealedProbeBackend(explore=True, operators=operators)

    # ---- Source 阶段（Target 未打开；固定计划不挑正）----
    # 混合形态（实验 5，审核裁决）：Source 探测始终确定性（空 Memory 时
    # Harness Control 负责最低限度的确定性探索，保证产生 Action–Response
    # Experience）——即使 --llm-select（LLM 只用于有经验的 Target select）。
    src_executor = ScopeExecutor(src_roster, src_values, config,
                                 evaluate_fn=v6._evaluate)
    src_observed = dict(resolver.window_context(src_values, SOURCE_ORIGIN,
                                                PERIOD))
    src_observed["bound_period"] = float(PERIOD)
    actionable_src = _actionable_ops(root, src_series0, SOURCE_ORIGIN,
                                     src_observed)
    src_backend = SealedProbeBackend(explore=True, operators=tuple(actionable_src))
    src_method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            src_backend,
            LocalPublicToolGateway(src_series0[:SOURCE_ORIGIN],
                                   task_kind="forecast"))),
        h0_snapshot, ())
    src_round = probe_round(src_method, src_executor, src_series0, src_values,
                            SOURCE_ORIGIN, round_name="src600")
    src_delayed: list[dict[str, Any]] = []
    for i, p in enumerate(src_round["probes"]):
        if p.get("kind") == "support" and p.get("episode_id"):
            # 每条 Source Episode 用**自己的**冻结 Workflow 更新（审核实验 1
            # 修复：不再重复更新最后一条）
            d = open_delayed(src_method, src_executor, src_series0, src_values,
                             SOURCE_ORIGIN, round_name=f"src600_p{i + 1}",
                             episode_id=p["episode_id"])
            if d:
                src_delayed.append(d)
    src_memory = tuple(src_method._experience_episodes)  # noqa: SLF001
    print(f"== source probes: {json.dumps(src_round['probes'], ensure_ascii=False)}")
    print(f"== source memory frozen: {len(src_memory)} episodes "
          f"({[getattr(e, 'relation', '?') for e in src_memory]})")

    # ---- Target 阶段（Source 冻结后打开）----
    tgt_executor = ScopeExecutor(tgt_roster, tgt_values, config,
                                 evaluate_fn=v6._evaluate)
    tgt_observed = dict(resolver.window_context(tgt_values, R1_ORIGIN, PERIOD))
    tgt_observed["bound_period"] = float(PERIOD)
    actionable_tgt = _actionable_ops(root, tgt_series0, R1_ORIGIN, tgt_observed)
    print(f"== target actionable @{R1_ORIGIN}: n={len(actionable_tgt)} "
          f"noop_filtered={len(_actionable_at(root, tgt_series0, R1_ORIGIN)) - len(actionable_tgt)}")

    gw = LocalPublicToolGateway(tgt_series0[:R1_ORIGIN], task_kind="forecast")
    backend_a5 = _make_backend(actionable_tgt, tgt_observed)
    backend_a3 = _make_backend(actionable_tgt, tgt_observed)
    method_a5 = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(backend_a5, gw)), h0_snapshot, src_memory)
    method_a3 = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(backend_a3, gw)), h0_snapshot, ())
    assert set(tuple(backend_a5._explored)) == set(tuple(backend_a3._explored))  # noqa: SLF001

    # R1 探测循环（每臂独立，预算 2）
    r1_a5 = probe_round(method_a5, tgt_executor, tgt_series0, tgt_values,
                        R1_ORIGIN, round_name="r1_a5")
    r1_a3 = probe_round(method_a3, tgt_executor, tgt_series0, tgt_values,
                        R1_ORIGIN, round_name="r1_a3")
    print(f"== R1 A5: {json.dumps(r1_a5['probes'], ensure_ascii=False)}")
    print(f"== R1 A3: {json.dumps(r1_a3['probes'], ensure_ascii=False)}")

    # R1 delayed @840（审核 first fault 修复：显式本轮 support Episode；
    # 本轮 abstain（无 Episode）→ open_delayed 返回 None、不更新历史）
    def _round_last_support_episode(rd: dict[str, Any]) -> str | None:
        for p in reversed(rd["probes"]):
            if p.get("kind") == "support" and p.get("episode_id"):
                return str(p["episode_id"])
        return None

    d1_a5 = open_delayed(method_a5, tgt_executor, tgt_series0, tgt_values,
                         R1_ORIGIN, round_name="r1_a5",
                         episode_id=_round_last_support_episode(r1_a5))
    d1_a3 = open_delayed(method_a3, tgt_executor, tgt_series0, tgt_values,
                         R1_ORIGIN, round_name="r1_a3",
                         episode_id=_round_last_support_episode(r1_a3))
    print(f"== R1 delayed A5: {d1_a5}")
    print(f"== R1 delayed A3: {d1_a3}")

    # Skill 写盘（每臂独立 fork）：R1 首次正向 chosen → 写盘；状态取该
    # Episode delayed 后的状态（R1 探测到 first positive 即停止——本臂最后
    # 一条 Episode 即 skill 那条）。
    def skill_from_method(method: TTHAMethod, rd: dict[str, Any],
                          arm_label: str, d1: dict[str, Any] | None,
                          ) -> dict[str, Any] | None:
        for p in rd["probes"]:
            if not p.get("first_positive"):
                continue
            eps = list(tuple(method._experience_episodes))  # noqa: SLF001
            for ep in reversed(eps):
                if getattr(ep, "episode_id", "") == p.get("episode_id"):
                    steps_raw = ep.context_summary["program_geometry"]["program_steps"]
                    steps = tuple((str(s["op"]), dict(s["params"]))
                                  for s in steps_raw)
                    skill_id = f"{steps[0][0][:8]}-sealed-{arm_label}"
                    status = (d1 or {}).get("status") or "LOCAL_DRAFT"
                    patched, store, fork_root = write_skill(
                        root, h0_snapshot, steps, skill_id, status,
                        rationale=f"R1 first-positive @{R1_ORIGIN} arm {arm_label}")
                    return {"skill_id": skill_id,
                            "status": status,
                            "steps": [{"op": s[0], "params": dict(s[1])}
                                      for s in steps],
                            "patched_snapshot": patched,
                            "store": store, "fork_root": fork_root}
            break
        return None

    skill_a5 = skill_from_method(method_a5, r1_a5, "a5", d1_a5)
    skill_a3 = skill_from_method(method_a3, r1_a3, "a3", d1_a3)
    print(f"== skill A5: {skill_a5 and skill_a5['skill_id']} "
          f"status={skill_a5 and skill_a5['status']}")
    print(f"== skill A3: {skill_a3 and skill_a3['skill_id']} "
          f"status={skill_a3 and skill_a3['status']}")

    # R2 @888（本臂 fork 快照；bind_round_data 到 888）
    snap_a5 = skill_a5["patched_snapshot"] if skill_a5 else h0_snapshot
    snap_a3 = skill_a3["patched_snapshot"] if skill_a3 else h0_snapshot
    method_a5.bind_round_data(tgt_series0[:R2_ORIGIN], task_kind="forecast")
    method_a3.bind_round_data(tgt_series0[:R2_ORIGIN], task_kind="forecast")
    # prepare 用快照：method 的 snapshot 是构造时传入——重建 method（保留
    # Memory/backend？method 的 snapshot 是构造参数。bind_round_data 只换
    # gateway。需要换 snapshot → 新建 method 会丢 Memory。改法：prepare 后
    # 我们通过 fast_agent 换 snapshot？TTHAFastAgent.prepare(request, snapshot)
    # 接受 snapshot 参数——但 method.prepare 用 self._snapshot。
    # 最小处理：把本臂 method 的 _snapshot 换成 fork 快照（同实例内）。
    method_a5._snapshot = snap_a5  # noqa: SLF001
    method_a3._snapshot = snap_a3  # noqa: SLF001
    r2_a5 = probe_round(method_a5, tgt_executor, tgt_series0, tgt_values,
                        R2_ORIGIN, round_name="r2_a5")
    r2_a3 = probe_round(method_a3, tgt_executor, tgt_series0, tgt_values,
                        R2_ORIGIN, round_name="r2_a3")
    print(f"== R2 A5: {json.dumps(r2_a5['probes'], ensure_ascii=False)}")
    print(f"== R2 A3: {json.dumps(r2_a3['probes'], ensure_ascii=False)}")

    # R2 delayed @936（收尾 2：更新 R2 实际 chosen 的 Episode——修复
    # steps_override 路径 episode_id=null；Skill delayed utility = 同 steps
    # 的 delayed gain）
    def _r2_last_support_episode(rd: dict[str, Any]) -> str | None:
        for p in reversed(rd["probes"]):
            if p.get("kind") == "support" and p.get("episode_id"):
                return str(p["episode_id"])
        return None

    r2_ep_a5 = _r2_last_support_episode(r2_a5)
    r2_ep_a3 = _r2_last_support_episode(r2_a3)
    skill_steps_a5 = (tuple((s["op"], s["params"]) for s in skill_a5["steps"])
                      if skill_a5 else None)
    skill_steps_a3 = (tuple((s["op"], s["params"]) for s in skill_a3["steps"])
                      if skill_a3 else None)
    d2_a5 = open_delayed(method_a5, tgt_executor, tgt_series0, tgt_values,
                         R2_ORIGIN, round_name="r2_a5",
                         episode_id=r2_ep_a5,
                         steps_override=(None if r2_ep_a5 else skill_steps_a5))
    d2_a3 = open_delayed(method_a3, tgt_executor, tgt_series0, tgt_values,
                         R2_ORIGIN, round_name="r2_a3",
                         episode_id=r2_ep_a3,
                         steps_override=(None if r2_ep_a3 else skill_steps_a3))

    # 收尾 2：Skill 状态同步（R2 delayed 后的 relation/status → fork 内
    # skill json 的 Status 更新 + 重编译——LOCAL_ACTIVE 保持 /
    # CONFLICT→RESTRICTED）
    def _sync_skill_status(skill: dict[str, Any] | None,
                           delayed_info: dict[str, Any] | None,
                           ) -> dict[str, Any] | None:
        if not skill or delayed_info is None or not delayed_info.get("status"):
            return skill
        status = str(delayed_info["status"])
        if skill["status"] == status:
            return skill
        import re as _re
        skill_path = (skill["fork_root"] / "skills" / "learned"
                      / f"{skill['skill_id']}.json")
        entry = json.loads(skill_path.read_text(encoding="utf-8"))
        entry["body"] = _re.sub(r"(?m)^Status: .*$", f"Status: {status}",
                                entry["body"])
        entry["revision"] = int(entry.get("revision", 1)) + 1
        skill_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2)
                              + "\n", encoding="utf-8")
        patched = compile_snapshot(skill["fork_root"], verify_lock=False)
        skill["patched_snapshot"] = patched
        skill["status"] = status
        print(f"== skill status synced: {skill['skill_id']} -> {status}")
        return skill

    skill_a5 = _sync_skill_status(skill_a5, d2_a5)
    skill_a3 = _sync_skill_status(skill_a3, d2_a3)

    # ---- 实验 3（审核：自然 delayed feedback 控制）----
    # plan-only prepare @984（不评估 future——不调用 evaluate）：验证
    # delayed 反馈是否改变下一决策点行动——traffic（LOCAL_ACTIVE）Skill
    # 保持优先；uci（RESTRICTED）Skill 降到 UNKNOWN 后或 abstain。
    def _plan_only_prepare(method: TTHAMethod, series0: np.ndarray,
                           values: Mapping[str, Any], origin: int, *,
                           round_name: str) -> dict[str, Any]:
        method.bind_round_data(series0[:origin], task_kind="forecast")
        method.prepare(_request(series0, values, origin))
        trace = method.last_trace
        return {"round": round_name, "origin": origin,
                "chosen": trace.chosen_candidate_id,
                "pool": list((trace.candidate_program_steps or {}).keys()),
                "evaluated": False}

    r3_a5 = _plan_only_prepare(method_a5, tgt_series0, tgt_values, 984,
                               round_name="r3_a5")
    r3_a3 = _plan_only_prepare(method_a3, tgt_series0, tgt_values, 984,
                               round_name="r3_a3")
    print(f"== R3(plan-only @984) A5: chosen={r3_a5['chosen']} "
          f"pool={r3_a5['pool']}")
    print(f"== R3(plan-only @984) A3: chosen={r3_a3['chosen']} "
          f"pool={r3_a3['pool']}")
    print(f"== R2 delayed A5: {d2_a5}")
    print(f"== R2 delayed A3: {d2_a3}")

    # ---- 指标与 verdict（预注册 §5/§6）----
    m_a5 = _metric_summary([r1_a5, r2_a5])
    m_a3 = _metric_summary([r1_a3, r2_a3])
    skill_delayed_a5 = (d2_a5 or {}).get("delayed_gain")
    skill_delayed_a3 = (d2_a3 or {}).get("delayed_gain")
    fp5 = m_a5["first_positive_support_receipt_index"]
    fp3 = m_a3["first_positive_support_receipt_index"]
    harm5, harm3 = m_a5["harm_count"], m_a3["harm_count"]

    a5_ok = skill_a5 is not None and skill_delayed_a5 is not None \
        and skill_delayed_a5 >= MATERIAL
    a3_ok = skill_a3 is not None and skill_delayed_a3 is not None \
        and skill_delayed_a3 >= MATERIAL
    if not src_memory:
        verdict = "SEALED_A5A3_NO_APPLICABLE_SOURCE_MEMORY"
    elif not actionable_tgt:
        verdict = "SEALED_A5A3_INFEASIBLE_NO_HEADROOM"
    elif fp5 is None and fp3 is None and not a5_ok and not a3_ok:
        # 预算内无正向 Workflow（两臂均未形成 Skill）——如实报告数据约束
        verdict = "SEALED_A5A3_INFEASIBLE_NO_HEADROOM"
    else:
        faster = fp5 is not None and (fp3 is None or fp5 < fp3)
        safer = harm5 < harm3
        not_worse = (
            (fp5 is None or fp3 is None or fp5 <= fp3)
            and harm5 <= harm3
            and m_a5["proposal_count"] <= m_a3["proposal_count"])
        if (faster or safer) and not_worse and a5_ok:
            verdict = "SEALED_A5A3_SOURCE_GUIDANCE_PASS"
        elif (faster or safer) and not a5_ok:
            verdict = "SEALED_A5A3_PARTIAL"
        elif fp5 == fp3 and harm5 == harm3 and a5_ok == a3_ok:
            verdict = "SEALED_A5A3_SAME_NO_BENEFIT"
        else:
            verdict = "SEALED_A5A3_NEGATIVE"

    print(f"== metrics A5: {m_a5}")
    print(f"== metrics A3: {m_a3}")
    print(f"== skill delayed A5={skill_delayed_a5} A3={skill_delayed_a3}")
    print(f"== verdict: {verdict}")

    # ---- 收尾 2 + 实验 3 验收（lifecycle closure：delayed → Skill 状态 →
    #      下一决策点行动）----
    skill_cand_a5 = f"cand_skill_{skill_a5['skill_id']}" if skill_a5 else None
    skill_cand_a3 = f"cand_skill_{skill_a3['skill_id']}" if skill_a3 else None

    def _skill_pos(pool: Sequence[str], cand: str | None) -> int | None:
        return pool.index(cand) if cand is not None and cand in pool else None

    pos_a5 = _skill_pos(r3_a5["pool"], skill_cand_a5)
    pos_a3 = _skill_pos(r3_a3["pool"], skill_cand_a3)
    n_eps_a5_before_r3 = len(tuple(method_a5._experience_episodes))  # noqa: SLF001
    n_eps_a3_before_r3 = len(tuple(method_a3._experience_episodes))  # noqa: SLF001
    # 期望由 R2 delayed 的真实 relation 决定（预注册：delayed 双正 → 保持；
    # delayed 负 → 降级/abstain；不硬编码 domain）
    r2_rel = {"a5": (d2_a5 or {}).get("relation"),
              "a3": (d2_a3 or {}).get("relation")}
    expected = "keep" if r2_rel.get("a5") == "POSITIVE" else "degrade"
    lifecycle: dict[str, Any] = {
        "r2_episode_delayed_updated": bool(
            d2_a5 and d2_a5.get("episode_id") and d2_a3
            and d2_a3.get("episode_id")),
        "r2_relations": r2_rel,
        "skill_status": {"a5": skill_a5 and skill_a5["status"],
                         "a3": skill_a3 and skill_a3["status"]},
        "r3_plan_only": {"a5": r3_a5, "a3": r3_a3},
        "skill_pool_position_at_984": {"a5": pos_a5, "a3": pos_a3},
        "expected_from_delayed_relation": expected,
    }
    if expected == "keep":
        # 只对形成 Skill 的臂检查（abstain 臂无 Skill 不参与）
        lifecycle["skill_kept_priority_at_984"] = bool(
            (pos_a5 == 0 if skill_a5 else True)
            and (pos_a3 == 0 if skill_a3 else True))
    else:
        lifecycle["skill_degraded_or_abstain_at_984"] = bool(
            ((pos_a5 is None or pos_a5 > 0 or r3_a5["chosen"] == "identity")
             if skill_a5 else True)
            and ((pos_a3 is None or pos_a3 > 0 or r3_a3["chosen"] == "identity")
                 if skill_a3 else True))
    lifecycle["no_episode_deletion"] = bool(
        len(tuple(method_a5._experience_episodes)) == n_eps_a5_before_r3  # noqa: SLF001
        and len(tuple(method_a3._experience_episodes)) == n_eps_a3_before_r3)  # noqa: SLF001
    lifecycle["no_future_read_plan_only"] = bool(
        r3_a5["evaluated"] is False and r3_a3["evaluated"] is False)
    print(f"== lifecycle: {json.dumps(lifecycle, ensure_ascii=False)}")

    report = {
        "experiment_id": "v1-sealed-a5a3",
        "cohort_offset": cohort_offset,
        "dataset": DOMAIN,
        "cohorts": {
            "source_uids": [r["series_uid"] for r in src_roster],
            "target_uids": [r["series_uid"] for r in tgt_roster],
            "disjoint": True,
            "exposure_class": "certified_virgin",
        },
        "origins": {"source": SOURCE_ORIGIN, "source_delayed": SOURCE_DELAYED,
                    "r1": R1_ORIGIN, "r1_delayed": R1_DELAYED,
                    "r2": R2_ORIGIN, "r2_delayed": R2_DELAYED,
                    "assertions": {
                        "r2_minus_r1_equals_2h": bool(R2_ORIGIN - R1_ORIGIN == 2 * HORIZON),
                        "source_window_before_r1": bool(SOURCE_ORIGIN + 2 * HORIZON <= R1_ORIGIN),
                        "r2_delayed_in_data": bool(R2_ORIGIN + 2 * HORIZON <= 1024)}},
        "source": {"round": src_round, "delayed": src_delayed,
                   "memory_episodes": len(src_memory),
                   "relations": [getattr(e, "relation", "?") for e in src_memory]},
        "arms": {
            "a5": {"r1": r1_a5, "r1_delayed": d1_a5,
                   "skill": skill_a5 and {k: v for k, v in skill_a5.items()
                                          if k not in ("patched_snapshot", "store", "fork_root")},
                   "r2": r2_a5, "r2_delayed": d2_a5, "metrics": m_a5,
                   "select_logs": [{"chosen": s["chosen"],
                                    "memory_excerpt": s["prompt"][
                                        s["prompt"].find("== Experience memory")
                                        :s["prompt"].find("== Your task")][:300],
                                    "raw": s["raw"][:200]}
                                   for s in getattr(backend_a5, "_select_logs", [])]},
            "a3": {"r1": r1_a3, "r1_delayed": d1_a3,
                   "skill": skill_a3 and {k: v for k, v in skill_a3.items()
                                          if k not in ("patched_snapshot", "store", "fork_root")},
                   "r2": r2_a3, "r2_delayed": d2_a3, "metrics": m_a3,
                   "select_logs": [{"chosen": s["chosen"],
                                    "memory_excerpt": s["prompt"][
                                        s["prompt"].find("== Experience memory")
                                        :s["prompt"].find("== Your task")][:300],
                                    "raw": s["raw"][:200]}
                                   for s in getattr(backend_a3, "_select_logs", [])]},
        },
        "verdict": verdict,
        "lifecycle_closure": lifecycle,
        "llm_api_call_count": (llm_client.calls if llm_client is not None
                               else 0),
        "selector": "llm-select" if llm_select else "deterministic",
    }
    # 收尾 1：按 dataset + cohort_offset 分文件保存（traffic PASS 与 UCI
    # PARTIAL 各自独立承重证据）
    _san = "".join(c if c.isalnum() else "_" for c in DOMAIN)
    out = root / Path(
        f"artifacts/functional/e2/w1_sealed_a5_a3_{_san}_{cohort_offset}"
        f"_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    for s in (skill_a5, skill_a3):
        if s:
            s["store"].discard_fork(s["fork_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
