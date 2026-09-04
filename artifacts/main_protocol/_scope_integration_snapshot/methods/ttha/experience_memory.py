"""Experience Memory — 工作包 1 最小接线（deepseek 副本，2026-08-06）。

三个组件：
- ExperienceEpisode：一次合法 Action–Response 的最小记录（relation/validity/evidence_level/local_status）
- SignedEpisodeRetriever：确定性四步检索（硬过滤 → 同域分开 → 轻量 Context 距离 → 对照包）
- CurrentHarnessState：当前视图（RESTRICTED/REJECTED 覆盖旧 ACTIVE），Fast Path 唯一读取入口

设计依据：docs/WORK_PACK_1_EXPERIENCE_RUNTIME_DESIGN.md（审核稿 §5.2 + 评议最小框架）。
约束：不建平台、不依赖原项目深层模块（仅标准库）、不读取 outcome/Query future。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# 1. ExperienceEpisode
# ---------------------------------------------------------------------------

# relation 枚举
RELATION_POSITIVE = "POSITIVE"
RELATION_NEGATIVE = "NEGATIVE"
RELATION_CONFLICT = "CONFLICT"
RELATION_ABSTAIN = "ABSTAIN"
# T4 (#40) A2：生命周期分类升级需要第五格——"聚合近零"既不是成功也不是失败，
# 此前只能被挤进 POSITIVE/NEGATIVE，会让"聚合正、局部有害"与"什么也没发生"
# 无法分辨。NEUTRAL 不进对照包（既非正例亦非负例亦非冲突），只如实存档。
RELATION_NEUTRAL = "NEUTRAL"
RELATIONS = (
    RELATION_POSITIVE,
    RELATION_NEGATIVE,
    RELATION_CONFLICT,
    RELATION_ABSTAIN,
    RELATION_NEUTRAL,
)

# local_status 枚举（近期只启用 4 个；SHARED_* 跨域实验时再启用）
STATUS_EPISODE_ONLY = "EPISODE_ONLY"
STATUS_LOCAL_DRAFT = "LOCAL_DRAFT"
STATUS_LOCAL_ACTIVE = "LOCAL_ACTIVE"
STATUS_RESTRICTED = "RESTRICTED"
LOCAL_STATUSES = (STATUS_EPISODE_ONLY, STATUS_LOCAL_DRAFT, STATUS_LOCAL_ACTIVE, STATUS_RESTRICTED)

# evidence_level 枚举
EVIDENCE_SUPPORT = "SUPPORT"
EVIDENCE_FULL_POLICY = "FULL_POLICY"
EVIDENCE_DELAYED = "DELAYED"

# response_validity 枚举：仪器故障（API timeout/fit crash/compile failure/metric 无效）
# 不算负向经验，默认不参与检索
VALIDITY_VALID = "VALID"
VALIDITY_INSTRUMENT_INVALID = "INSTRUMENT_INVALID"

# 禁止进入 Episode 的私有字段（复用 v6 _contrast_episode 的 forbidden 检查）
_FORBIDDEN_KEYS = frozenset(
    {"dataset_id", "series_uid", "filename", "file_name", "path", "query_future", "future"}
)


# ---------------------------------------------------------------------------
# 0. 键：一处铸造，别处不得手拼（T4 #40 A1）
# ---------------------------------------------------------------------------
# 收束裁决（fast_agent.py 的运行时规范键）：Memory 的任务硬键 =
# task_type|downstream_model_class|metric.name。此前这个格式在 fast_agent 内联
# f-string、ssi 的一个死常量、guidance_evolution 的一处手拼里各写了一遍，而 ssi
# 的 Episode 写入用的是第三种格式 batch:<cohort>|consumer:<variant>——同一个
# Memory 里两种方言，检索必然错过。这里把铸造收进一处：
#   * task_consumer_key(spec)  —— 任务硬键，唯一进 Episode.task_consumer_key 的串；
#   * cell_key(cohort, variant) —— cohort×consumer 单元标识，**不是**任务键。
# cohort/domain 只进 domain_namespace 与 context_summary，不进任务硬键：跨域检索
# （T6/X）要的正是"同任务、异 cohort"能命中。
TASK_CONSUMER_KEY_RULE = "task_type|downstream_model_class|metric.name"
TASK_CONSUMER_KEY_FALLBACK = "forecast|ridge|sMASE"


def task_consumer_key(task_spec: object) -> str:
    """Episode/检索共用的任务硬键。传 None 回落到历史默认（fast_agent 原语义）。

    只接受带 task_type / downstream_model_class / metric.name 的 TaskSpec 对象；
    传字符串会被原样退回（调用方已持有铸好的键），传别的形状直接报错——静默
    造出第四种方言比崩掉更贵。
    """
    if task_spec is None:
        return TASK_CONSUMER_KEY_FALLBACK
    if isinstance(task_spec, str):
        return task_spec
    try:
        task_type = task_spec.task_type
        model_class = task_spec.downstream_model_class
        metric_name = task_spec.metric.name
    except AttributeError as exc:  # pragma: no cover - 形状错误应当暴露
        raise TypeError(
            "task_consumer_key expects a TaskSpec-shaped object "
            "(task_type / downstream_model_class / metric.name); got %r"
            % (type(task_spec).__name__,)
        ) from exc
    return "%s|%s|%s" % (task_type, model_class, metric_name)


def cell_key(cohort: object, consumer_variant: object) -> str:
    """cohort x consumer 单元标识（报告字段、卡的编译落点、LOCO withhold 口径）。

    与任务硬键**分立**且不可互换：这个串带 cohort，正因为它标的是"哪一批数据的
    哪个 Consumer 单元"；任务硬键带 cohort 会把同任务跨 cohort 的经验切断。
    """
    return "batch:%s|consumer:%s" % (cohort, consumer_variant)


@dataclass(frozen=True)
class ExperienceEpisode:
    episode_id: str
    schema_version: str
    task_consumer_key: str
    domain_namespace: str
    context_summary: Mapping[str, object]
    workflow_signature: str
    support_response: Mapping[str, object]
    delayed_response: Mapping[str, object]
    relation: str
    evidence_level: str
    response_validity: str
    local_status: str
    pattern_view: str = "default"
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "experience-episode/1":
            raise ValueError("ExperienceEpisode schema_version must be experience-episode/1")
        if self.relation not in RELATIONS:
            raise ValueError(f"invalid relation: {self.relation}")
        if self.local_status not in LOCAL_STATUSES:
            raise ValueError(f"invalid local_status: {self.local_status}")
        if self.response_validity not in (VALIDITY_VALID, VALIDITY_INSTRUMENT_INVALID):
            raise ValueError(f"invalid response_validity: {self.response_validity}")
        if self.evidence_level not in (EVIDENCE_SUPPORT, EVIDENCE_FULL_POLICY, EVIDENCE_DELAYED):
            raise ValueError(f"invalid evidence_level: {self.evidence_level}")

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "schema_version": self.schema_version,
            "task_consumer_key": self.task_consumer_key,
            "domain_namespace": self.domain_namespace,
            "context_summary": dict(self.context_summary),
            "workflow_signature": self.workflow_signature,
            "support_response": dict(self.support_response),
            "delayed_response": dict(self.delayed_response),
            "relation": self.relation,
            "evidence_level": self.evidence_level,
            "response_validity": self.response_validity,
            "local_status": self.local_status,
            "pattern_view": self.pattern_view,
            "evidence_refs": list(self.evidence_refs),
        }


def _check_private_fields(obj: object, *, path: str = "") -> None:
    """私有字段检查：dataset_id/series_uid/path/query_future 等不得进入 Episode。"""
    if isinstance(obj, Mapping):
        for key, nested in obj.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"private field entered episode: {key}")
            _check_private_fields(nested, path=f"{path}.{key}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for nested in obj:
            _check_private_fields(nested, path=path)


def workflow_signature_of(program_steps: object) -> str:
    """算子序列的确定性指纹（只含算子名与顺序，不含参数值）。

    真实 v6 steps 结构为 {"op": ..., "params": {...}}——`op` 是首要键，
    兼容 operator/name/program 旧键。steps 非空但解析不到任何算子时返回
    "unknown"（显式暴露"声明≠执行"问题），只有 steps 为空/None 才返回
    "identity"（真正什么都没做）。
    """
    names: list[str] = []
    if isinstance(program_steps, Sequence) and not isinstance(program_steps, (str, bytes)):
        for step in program_steps:
            if isinstance(step, Mapping):
                op = (
                    step.get("op")
                    or step.get("operator")
                    or step.get("name")
                    or step.get("program")
                )
                if isinstance(op, str) and op:
                    names.append(op)
    if names:
        return "|".join(names)
    if program_steps is None or (
        isinstance(program_steps, Sequence) and len(program_steps) == 0
    ):
        return "identity"
    return "unknown"


def canonical_sha256(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 2. SignedEpisodeRetriever（确定性四步）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastPack:
    """Fast Path 得到的对照包：最相似的成功/失败/冲突/未动作 + 证据充分性。

    T4b (#40b) A2：第四格 abstain。#40 的读数指认了这个缺口——AD 臂正确答案
    是"什么都不做"，而 identity 判 ABSTAIN 后三格里无处安放，能渲染的只剩
    "winsorize 退化"与"hampel 聚合改善"，孤立的聚合改善卡把谨慎剖面推翻了。
    字段名 abstain 唯一；卡面可称 no-action baseline。默认 None，位置在末尾，
    既有位置构造保持可用。
    """

    positive: ExperienceEpisode | None
    negative: ExperienceEpisode | None
    conflict: ExperienceEpisode | None
    evidence_sufficient: bool
    retrieval_note: str
    abstain: ExperienceEpisode | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "positive": self.positive.to_dict() if self.positive else None,
            "negative": self.negative.to_dict() if self.negative else None,
            "conflict": self.conflict.to_dict() if self.conflict else None,
            "abstain": self.abstain.to_dict() if self.abstain else None,
            "evidence_sufficient": self.evidence_sufficient,
            "retrieval_note": self.retrieval_note,
        }


def _context_distance(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    """轻量 Context 距离：三个维度（cohort/local_pattern/program_geometry）的数值 L1。

    维度缺失时按 0 计（保守：缺特征不惩罚，由硬过滤兜底）。
    """
    total = 0.0
    for dim in ("cohort", "local_pattern", "program_geometry"):
        av = a.get(dim)
        bv = b.get(dim)
        if not isinstance(av, Mapping) or not isinstance(bv, Mapping):
            continue
        keys = set(av) & set(bv)
        for key in keys:
            x, y = av[key], bv[key]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                total += abs(float(x) - float(y))
            elif isinstance(x, str) and isinstance(y, str) and x != y:
                total += 1.0
    return total


class SignedEpisodeRetriever:
    """确定性四步检索：硬过滤 → 同域分开 → 轻量距离 → 对照包。不训练、不读取 outcome。"""

    def __init__(
        self,
        episodes: Sequence[ExperienceEpisode],
        *,
        task_consumer_key: str,
        allowed_operators: Sequence[str] = (),
        pattern_view: str | None = None,
    ) -> None:
        self._episodes = tuple(episodes)
        self._task_consumer_key = task_consumer_key
        self._allowed_operators = tuple(allowed_operators)
        self._pattern_view = pattern_view

    def _hard_filter(self, episode: ExperienceEpisode) -> bool:
        if episode.response_validity != VALIDITY_VALID:
            return False  # 仪器故障不参与检索
        # task/consumer 精确匹配（规范 key 统一后不用前缀兼容——避免不同
        # Consumer 的经验混用）
        if episode.task_consumer_key != self._task_consumer_key:
            return False
        if self._pattern_view is not None and episode.pattern_view != self._pattern_view:
            return False  # 区别化：只在匹配的 pattern 视角内检索
        # T4b (#40b) A1：ABSTAIN + identity 的 Episode 只绕过下面这一条
        # informative-operator membership 检查，上面的 response_validity /
        # task_consumer_key / pattern_view 全部照过。绕过的理由：这条经验说的
        # 就是"没有动用任何算子"，用"算子是否在允许集里"去筛它是范畴错误；
        # 且真实 Runtime 的 allowed_operators 来自 Operator registry，identity
        # 作为 no-op 未必在册——挂靠 identity ∈ allowed_operators 会做成考试
        # 专用通道。unknown 签名不在此列：那是"声明≠执行"的解析失败，照滤。
        if (
            self._allowed_operators
            and episode.relation == RELATION_ABSTAIN
            and episode.workflow_signature == "identity"
        ):
            return True
        if self._allowed_operators:
            ops = tuple(episode.workflow_signature.split("|"))
            # identity/unknown 无信息量（什么都没做 / 解析失败），显式排除；
            # 其余 token（算子名或 workflow 名，词汇表由调用方与签名保持一致）
            informative = [op for op in ops if op and op not in ("identity", "unknown")]
            if not informative:
                return False
            if not any(op in self._allowed_operators for op in informative):
                return False
        return True

    def retrieve(
        self,
        context_summary: Mapping[str, object],
        domain_namespace: str,
    ) -> ContrastPack:
        """按公开 Context 检索，返回对照包。不读取 outcome/Query future。"""
        candidates = [ep for ep in self._episodes if self._hard_filter(ep)]
        if not candidates:
            return ContrastPack(None, None, None, False, "no valid episodes")

        # 同域优先：同 domain 的 episode 先排序；跨域仅作补充
        def key(ep: ExperienceEpisode) -> tuple[int, float, str]:
            same_domain = 0 if ep.domain_namespace == domain_namespace else 1
            return (same_domain, _context_distance(ep.context_summary, context_summary), ep.episode_id)

        ranked = sorted(candidates, key=key)
        positive = next((ep for ep in ranked if ep.relation == RELATION_POSITIVE), None)
        negative = next((ep for ep in ranked if ep.relation == RELATION_NEGATIVE), None)
        conflict = next((ep for ep in ranked if ep.relation == RELATION_CONFLICT), None)
        abstain = next((ep for ep in ranked if ep.relation == RELATION_ABSTAIN), None)
        count = len(ranked)
        note = (
            f"{count} valid episodes; same-domain-first; signed pack "
            "(pos/neg/conflict/abstain)"
        )
        # 证据充分性口径不动（正或负才算充分）：abstain 是"未动作"的读数，
        # 它让谨慎选择可被看见，但本身不构成"有足够证据去做什么"。
        sufficient = positive is not None or negative is not None
        return ContrastPack(positive, negative, conflict, sufficient, note, abstain)


# ---------------------------------------------------------------------------
# 3. CurrentHarnessState（当前视图）
# ---------------------------------------------------------------------------

@dataclass
class CurrentHarnessState:
    """Fast Path 唯一读取的当前视图。RESTRICTED/REJECTED 覆盖旧 ACTIVE。"""

    local_skills: dict[str, str] = field(default_factory=dict)  # skill_id -> status
    restrictions: list[dict[str, object]] = field(default_factory=list)
    rejected_bets: list[dict[str, object]] = field(default_factory=list)
    schema_version: str = "current-harness-state/1"

    def apply_episode_status(self, episode: ExperienceEpisode) -> None:
        """按 Episode 的 local_status 更新视图；RESTRICTED 覆盖 ACTIVE。"""
        key = episode.episode_id
        if episode.local_status == STATUS_RESTRICTED:
            self.local_skills[key] = STATUS_RESTRICTED
            self.restrictions.append(
                {
                    "skill_id": key,
                    "reason": str(episode.support_response.get("restriction_reason") or "restricted"),
                    "evidence_ref": episode.evidence_refs[0] if episode.evidence_refs else None,
                }
            )
        elif episode.relation == RELATION_NEGATIVE and episode.local_status == STATUS_EPISODE_ONLY:
            self.rejected_bets.append(
                {
                    "skill_id": key,
                    "relation": episode.relation,
                    "evidence_ref": episode.evidence_refs[0] if episode.evidence_refs else None,
                }
            )
        else:
            # 已有 RESTRICTED 记录时，后到的 ACTIVE 不覆盖
            if self.local_skills.get(key) != STATUS_RESTRICTED:
                self.local_skills[key] = episode.local_status

    def is_restricted(self, skill_id: str) -> bool:
        return self.local_skills.get(skill_id) == STATUS_RESTRICTED

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "local_skills": dict(self.local_skills),
            "restrictions": [dict(r) for r in self.restrictions],
            "rejected_bets": [dict(r) for r in self.rejected_bets],
        }


# ---------------------------------------------------------------------------
# 4. 报告加载（工作包 1 数据源：v6 报告 + e288 + target_local_v2）
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3b. 生命周期机械分类（T4 #40 A2）
# ---------------------------------------------------------------------------
# 冻结阈值 ±0.005（与 bch.MATERIAL_THRESHOLD、guard 的 harm 线同一条）。规则是
# 机械的，不看任务、不看程序名、不看任何答案表：
#   identity                                      -> ABSTAIN（什么也没做）
#   agg >= +t 且 min(per-series) >= -t            -> POSITIVE（聚合正且无局部伤害）
#   agg >= +t 且 存在 per-series < -t             -> CONFLICT（聚合正但局部有害）
#   agg <  -t                                     -> NEGATIVE
#   其余（近零）                                   -> NEUTRAL
# 第三格是这次升级的全部理由：没有它，"聚合正、局部有害"会被记成 POSITIVE，
# Memory 反而会去加固当初那个错误选择。
MEASURED_EFFECT_KEY = "measured_effect"
CLASSIFICATION_MATERIAL_THRESHOLD = 0.005


def classify_relation(
    *,
    aggregate_gain: object,
    per_series_gains: Mapping[str, object] | None = None,
    is_identity: bool = False,
    consumer_id: str | None = None,
    material_threshold: float = CLASSIFICATION_MATERIAL_THRESHOLD,
) -> dict[str, object]:
    """机械推导 relation + 卡要用的事实摘要。不读 outcome，不看程序名。"""
    threshold = float(material_threshold)
    per_series = {
        str(uid): float(value)
        for uid, value in dict(per_series_gains or {}).items()
    }
    harmed = sorted(uid for uid, value in per_series.items() if value < -threshold)
    minimum = min(per_series.values()) if per_series else None
    aggregate = None if aggregate_gain is None else float(aggregate_gain)
    if is_identity:
        relation = RELATION_ABSTAIN
        basis = "identity: the batch was left as it is, so there is no effect to judge"
    elif aggregate is None:
        relation = RELATION_NEUTRAL
        basis = "no aggregate reading, so no direction can be claimed"
    elif aggregate >= threshold and not harmed:
        relation = RELATION_POSITIVE
        basis = (
            "aggregate >= +%g and every per-series reading >= -%g"
            % (threshold, threshold)
        )
    elif aggregate >= threshold:
        relation = RELATION_CONFLICT
        basis = (
            "aggregate >= +%g but %d per-series reading(s) < -%g"
            % (threshold, len(harmed), threshold)
        )
    elif aggregate < -threshold:
        relation = RELATION_NEGATIVE
        basis = "aggregate < -%g" % threshold
    else:
        relation = RELATION_NEUTRAL
        basis = "aggregate within +/-%g of zero" % threshold
    if aggregate is None:
        direction = "unmeasured"
    elif aggregate >= threshold:
        direction = "improved"
    elif aggregate < -threshold:
        direction = "degraded"
    else:
        direction = "unchanged"
    return {
        "relation": relation,
        "classification_basis": basis,
        "material_threshold": threshold,
        "consumer_id": consumer_id,
        "aggregate_gain": aggregate,
        "aggregate_direction": direction,
        "series_read": len(per_series),
        "harmed_series_count": len(harmed),
        "harmed_series": harmed,
        "min_per_series_gain": minimum,
    }


def build_episode(
    *,
    episode_id: str,
    task_consumer_key: str,
    domain_namespace: str,
    context_summary: Mapping[str, object],
    workflow_signature: str,
    support_response: Mapping[str, object],
    delayed_response: Mapping[str, object],
    relation: str,
    evidence_level: str,
    response_validity: str = VALIDITY_VALID,
    local_status: str = STATUS_EPISODE_ONLY,
    pattern_view: str = "default",
    evidence_refs: Sequence[str] = (),
) -> ExperienceEpisode:
    """构造 Episode 并执行私有字段检查（构造即检查，防泄漏进 Memory）。"""
    raw = {
        "context_summary": dict(context_summary),
        "support_response": dict(support_response),
        "delayed_response": dict(delayed_response),
    }
    _check_private_fields(raw)
    return ExperienceEpisode(
        episode_id=episode_id,
        schema_version="experience-episode/1",
        task_consumer_key=task_consumer_key,
        domain_namespace=domain_namespace,
        context_summary=dict(context_summary),
        workflow_signature=workflow_signature,
        support_response=dict(support_response),
        delayed_response=dict(delayed_response),
        relation=relation,
        evidence_level=evidence_level,
        response_validity=response_validity,
        local_status=local_status,
        pattern_view=pattern_view,
        evidence_refs=tuple(evidence_refs),
    )


def load_episodes_from_v6_reports(reports_dir: Path) -> list[ExperienceEpisode]:
    """从三个已暴露报告构造 Episode（工作包 1 数据源，机制重放用）。

    - v6 report generation A（NN5）：POSITIVE（support gain > 0）
    - target_local_v2（REJECTED）：NEGATIVE（Target 验证拒绝）
    - e288（v1 RESTRICTED，POSITIVE_SUPPORT_FALSE_CONFIRMED_HARM_ON_FRESH_TARGET）：CONFLICT
    """
    episodes: list[ExperienceEpisode] = []

    def read(name: str) -> dict[str, object]:
        path = reports_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing report: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # --- Episode 1: POSITIVE（v6 generation A / NN5）---
    v6 = read("autonomous_natural_acquisition_cycle_v6_report.json")
    gen = v6.get("stages", {}).get("generation", [])
    item_a = next((g for g in gen if isinstance(g, dict) and g.get("environment") == "A"), None)
    if item_a is not None:
        steps = item_a.get("accepted_program_steps") or item_a.get("final_program_steps")
        episodes.append(
            build_episode(
                episode_id="v6_nn5_support_positive",
                task_consumer_key="forecast|ridge_smase",
                domain_namespace="nn5",
                context_summary={
                    "cohort": {"series_count": 32, "evaluation_series_count": 8},
                    "local_pattern": {"support_gain": float(item_a.get("support_gain") or 0.0)},
                    "program_geometry": {"scope": "training_rows"},
                },
                workflow_signature=workflow_signature_of(steps),
                support_response={
                    "gain": item_a.get("support_gain"),
                    "accepted": True,
                    "selection_gain": item_a.get("selection_gain"),
                },
                delayed_response={"evaluated": False, "gain": None},
                relation=RELATION_POSITIVE,
                evidence_level=EVIDENCE_SUPPORT,
                local_status=STATUS_LOCAL_DRAFT,
                evidence_refs=["artifacts/functional/e2/autonomous_natural_acquisition_cycle_v6_report.json"],
            )
        )

    # --- Episode 2: NEGATIVE（target_local_v2 REJECTED）---
    rejected = read("historical_policy_episode_workflow_target_local_v2_rejected.json")
    if rejected.get("status") == "REJECTED":
        episodes.append(
            build_episode(
                episode_id="historical_policy_episode_workflow_target_local_v2",
                task_consumer_key="forecast|ridge_smase",
                domain_namespace="target_local",
                context_summary={
                    "cohort": {"series_count": 12, "evaluation_series_count": 4},
                    "local_pattern": {"support_gain": None},
                    "program_geometry": {"scope": "historical_origins"},
                },
                workflow_signature="W_rowblock|W_curation|W_temporal_origin",
                support_response={
                    "gain": None,
                    "accepted": False,
                    "rejection": str(rejected.get("status")),
                },
                delayed_response={"evaluated": False, "gain": None},
                relation=RELATION_NEGATIVE,
                evidence_level=EVIDENCE_FULL_POLICY,
                local_status=STATUS_EPISODE_ONLY,
                evidence_refs=["artifacts/functional/e2/historical_policy_episode_workflow_target_local_v2_rejected.json"],
            )
        )

    # --- Episode 3: CONFLICT（e288：v1 曾 ACTIVE → Support 正但 fresh Target 有害 → RESTRICTED）---
    e288 = read("historical_policy_episode_workflow_state_update_e288.json")
    if e288.get("status") == "RESTRICTED":
        episodes.append(
            build_episode(
                episode_id="historical_policy_episode_workflow_v1",
                task_consumer_key="forecast|ridge_smase",
                domain_namespace="multi_source",
                context_summary={
                    "cohort": {"series_count": 12, "evaluation_series_count": 4},
                    "local_pattern": {"support_gain": None},
                    "program_geometry": {"scope": "historical_origins"},
                },
                workflow_signature="W_rowblock|W_curation|W_temporal_origin",
                support_response={
                    "gain": None,
                    "accepted": True,
                    "restriction_reason": str(e288.get("reason")),
                },
                delayed_response={
                    "evaluated": True,
                    "gain": None,
                    "harm_on_fresh_target": True,
                },
                relation=RELATION_CONFLICT,
                evidence_level=EVIDENCE_DELAYED,
                local_status=STATUS_RESTRICTED,
                evidence_refs=["artifacts/functional/e2/historical_policy_episode_workflow_state_update_e288.json"],
            )
        )

    return episodes


def episode_from_dict(d: Mapping[str, object]) -> "ExperienceEpisode":
    """从 to_dict() 产物恢复 Episode（episodes.json round-trip）。"""
    return ExperienceEpisode(
        episode_id=str(d["episode_id"]),
        schema_version=str(d["schema_version"]),
        task_consumer_key=str(d["task_consumer_key"]),
        domain_namespace=str(d["domain_namespace"]),
        context_summary=dict(d.get("context_summary") or {}),
        workflow_signature=str(d.get("workflow_signature") or ""),
        support_response=dict(d.get("support_response") or {}),
        delayed_response=dict(d.get("delayed_response") or {}),
        relation=str(d["relation"]),
        evidence_level=str(d.get("evidence_level") or "SUPPORT"),
        response_validity=str(d.get("response_validity") or VALIDITY_VALID),
        local_status=str(d.get("local_status") or STATUS_EPISODE_ONLY),
        pattern_view=str(d.get("pattern_view") or "default"),
        evidence_refs=tuple(str(x) for x in (d.get("evidence_refs") or [])),
    )


def load_experience_episodes(path: Path) -> list["ExperienceEpisode"]:
    """从 JSON 文件加载 Episode 列表（工作包 1 的持久化产物）。"""
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return [episode_from_dict(d) for d in raw if isinstance(d, Mapping)]


def resolve_experience_contrast_pack(
    episodes: Sequence["ExperienceEpisode"],
    features: Mapping[str, object],
    task_consumer_key: str,
    *,
    pattern_view: str | None = None,
    allowed_operators: Sequence[str] = (),
) -> "ContrastPack | None":
    """方法层接线（fast_agent.prepare 注入点）：按当前特征检索经验对照包。

    审查修订（2026-08-08）：
    - allowed_operators 过滤（只检索当前允许算子的经验，与 compile 一致性）；
    - 无 episodes / 无共同可识别特征时返回 None（安全不注入）；
    - 特征 = extract_public_features 的输出；与 Episode.context_summary.local_pattern
      使用相同键（审查 ③——特征键对齐）。
    """
    if not episodes:
        return None
    retriever = SignedEpisodeRetriever(
        episodes,
        task_consumer_key=task_consumer_key,
        pattern_view=pattern_view,
        allowed_operators=allowed_operators,
    )
    # 特征键对齐：Episode 的 context_summary.local_pattern 用数值标量特征
    local_pattern = {k: v for k, v in dict(features).items()
                     if isinstance(v, (int, float))}
    if not local_pattern:
        return None  # 无共同可识别特征 → 不注入（审查 ③）
    query_context = {
        "cohort": {"series_count": 1, "evaluation_series_count": 0},
        "local_pattern": local_pattern,
        "program_geometry": {"scope": "training_rows"},
    }
    return retriever.retrieve(query_context, domain_namespace="")


def _measured_effect(episode: object) -> Mapping[str, object] | None:
    """T4 (#40) A3：卡词汇的 Consumer 特征来源。

    只有携带机械分类事实块的 Episode 才走事实卡；旧 Episode 一个字节不变，
    继续走原来的方向句（向后兼容，另一线的替换/回放读数不动）。
    """
    if not isinstance(episode, Mapping):
        return None
    # delayed 优先：读数落在哪个窗口由写入方声明，delayed 是更强的证据等级。
    for slot in ("delayed_response", "support_response"):
        response = episode.get(slot)
        if not isinstance(response, Mapping):
            continue
        facts = response.get(MEASURED_EFFECT_KEY)
        if isinstance(facts, Mapping):
            return facts
    return None


def _fact_sentence(episode: Mapping[str, object], facts: Mapping[str, object],
                   *, index: int) -> str:
    """事实摘要句：Consumer / 聚合方向 / harmed count / min gain。

    只陈述测到了什么，不含"应选 X"或"避免 X"——方向由读者从事实推，不由卡代推
    （章程：Episode 与卡只载事实）。数值口径故意保留 min gain：这一轮考的正是
    风险层，而风险层是一个量级问题，抹掉量级卡就没有可读的风险信息。
    """
    op = str(episode.get("workflow_signature", "") or "")
    consumer = facts.get("consumer_id")
    consumer_text = (
        " for Consumer `%s`" % consumer if consumer else ""
    )
    direction = str(facts.get("aggregate_direction") or "unmeasured")
    harmed = facts.get("harmed_series_count")
    read = facts.get("series_read")
    minimum = facts.get("min_per_series_gain")
    parts = [
        "Reference %d: candidate operator(s) [%s] were measured on held-in "
        "data in a similar context%s. Aggregate direction: %s."
        % (index, op, consumer_text, direction)
    ]
    if isinstance(harmed, int) and isinstance(read, int) and read:
        parts.append(
            " Individual series harmed beyond the %+.3f line: %d of %d."
            % (-float(facts.get("material_threshold") or 0.005), harmed, read)
        )
    if isinstance(minimum, (int, float)):
        parts.append(" Worst single-series reading: %+.4f." % float(minimum))
    return "".join(parts)


def _abstain_sentence(episode: Mapping[str, object],
                      facts: Mapping[str, object], *, index: int) -> str:
    """未动作读数的事实句（T4b #40b A3）。零祈使。

    句内顺序与其余事实句一致（做了什么 -> 聚合 -> harmed -> 来源）：本轮禁动
    卡序，聚合先于风险的排法留作唯一后备面。这一句唯一要说清的是：把这一批
    原样留下，也是同一来源、同一 Consumer 下记录在案的一个读数。
    """
    consumer = facts.get("consumer_id")
    consumer_text = " for Consumer `%s`" % consumer if consumer else ""
    read = facts.get("series_read")
    harmed = facts.get("harmed_series_count")
    threshold = -float(facts.get("material_threshold") or 0.005)
    parts = [
        "Reference %d: leaving the batch as it is -- no operator applied -- is "
        "itself a recorded outcome%s in a similar context. Aggregate reading: "
        "0.0000, unchanged by definition." % (index, consumer_text)
    ]
    if isinstance(harmed, int) and isinstance(read, int) and read:
        parts.append(
            " Individual series harmed beyond the %+.3f line: %d of %d."
            % (threshold, harmed, read)
        )
    parts.append(
        " It comes from the same source and the same Consumer as the "
        "references above."
    )
    _ = episode
    return "".join(parts)


def render_experience_pack(pack: Mapping[str, object]) -> str:
    """把对照包渲染成 TIMECLAW 风格 fenced 前缀块（prompts.py:18 模式借鉴）。

    设计（TIMECLAW 验证 + 用户裁决）：
    - fenced 前缀、任务指令之前（LLM 注意力必然经过的位置）；
    - 祈使指令（"Use them as a guide"），弱化忽略许可为安全阀；
    - 内容=可行动的句子（方向），**不含 gain 数值**——TIMECLAW 消融发现
      GT 答案导致 answer-anchoring 损害精度（summarize_trajectory 故意省略）；
    - 结构化 JSON 原样保留在 payload（不手工改写、不排序）。
    """
    entries: list[str] = []
    pos = pack.get("positive")
    neg = pack.get("negative")
    conf = pack.get("conflict")
    # Memory 语义修正（用户裁决 2026-08-13）：
    #  1) 同一 workflow 同时命中正负（Context 无法区分）→ 合并为
    #     AMBIGUOUS 块——不得同时给出"优先尝试"和"避免"两个方向；
    #  2) Support-only 正例（delayed 未评估）不得表述成 delayed-positive
    #     （夸大证据等级）。
    if (isinstance(pos, Mapping) and isinstance(neg, Mapping)
            and pos.get("workflow_signature")
            == neg.get("workflow_signature")):
        op = pos.get("workflow_signature", "")
        entries.append(
            f"Reference 1: candidate operator(s) [{op}] produced both positive and "
            "negative Support outcomes in similar contexts, and the current "
            "observations cannot distinguish the two conditions. Treat the candidate "
            "as AMBIGUOUS — confirm on the current Support before relying on it, and "
            "do not treat it as a strong prior in either direction."
        )
    elif isinstance(pos, Mapping) and _measured_effect(pos) is not None:
        entries.append(_fact_sentence(pos, _measured_effect(pos), index=1))
    elif isinstance(pos, Mapping):
        op = pos.get("workflow_signature", "")
        dr = pos.get("delayed_response") or {}
        if bool(dr.get("evaluated")) and isinstance(dr.get("gain"), (int, float)) \
                and float(dr["gain"]) >= 0:
            tail = "Support and delayed segments both positive"
        else:
            tail = ("Support segment positive; delayed segment not yet evaluated"
                    if not bool(dr.get("evaluated"))
                    else "Support segment positive; delayed segment pending")
        entries.append(
            f"Reference 1: candidate operator(s) [{op}] were verified beneficial on "
            f"held-in data in a similar context ({tail}). Consider them as priors "
            "to confirm again on the current Support."
        )
    elif isinstance(neg, Mapping) and _measured_effect(neg) is not None:
        entries.append(_fact_sentence(neg, _measured_effect(neg), index=2))
    elif isinstance(neg, Mapping):
        op = neg.get("workflow_signature", "")
        entries.append(
            f"Reference 2: candidate operator(s) [{op}] were verified harmful on held-in "
            "data in this domain (Support segment negative). Avoid them."
        )
    if isinstance(conf, Mapping) and _measured_effect(conf) is not None:
        facts = _measured_effect(conf)
        entries.append(
            _fact_sentence(conf, facts, index=3)
            + " The aggregate reading and the per-series readings disagree."
        )
    elif isinstance(conf, Mapping):
        op = conf.get("workflow_signature", "")
        entries.append(
            f"Reference 3: candidate operator(s) [{op}] showed a Support-positive but "
            "delayed-negative flip. Treat as risk; confirm on the delayed segment before "
            "relying on it."
        )
    # T4b (#40b) A3：第四格 no-action baseline，附在既有三条之后——既有
    # Reference 编号（正 1 / 负 2 / 冲突 3）一个不动，本轮禁动卡序。
    abst = pack.get("abstain")
    if isinstance(abst, Mapping):
        abst_facts = _measured_effect(abst)
        if abst_facts is not None:
            entries.append(_abstain_sentence(abst, abst_facts, index=4))
    if not entries:
        return ""
    body = "\n\n".join(entries)
    return (
        "=== EXPERIENCE REFERENCES FROM PRIOR TRIALS ===\n"
        "Below are similar prior trials in this context and how their candidate "
        "operators behaved on held-in data. Use them as a guide for operator choice. "
        "The correct choice for the current data may differ; do not copy blindly, and "
        "ignore them if the context clearly does not match.\n\n"
        f"{body}\n"
        "=== END REFERENCES ===\n\n"
    )


__all__ = [
    "ExperienceEpisode",
    "ContrastPack",
    "SignedEpisodeRetriever",
    "CurrentHarnessState",
    "build_episode",
    "load_episodes_from_v6_reports",
    "workflow_signature_of",
    "canonical_sha256",
    "render_experience_pack",
    "episode_from_dict",
    "load_experience_episodes",
    "resolve_experience_contrast_pack",
    "RELATION_POSITIVE",
    "RELATION_NEGATIVE",
    "RELATION_CONFLICT",
    "RELATION_ABSTAIN",
    "RELATION_NEUTRAL",
    "task_consumer_key",
    "cell_key",
    "classify_relation",
    "CLASSIFICATION_MATERIAL_THRESHOLD",
    "MEASURED_EFFECT_KEY",
    "STATUS_EPISODE_ONLY",
    "STATUS_LOCAL_DRAFT",
    "STATUS_LOCAL_ACTIVE",
    "STATUS_RESTRICTED",
    "VALIDITY_VALID",
    "VALIDITY_INSTRUMENT_INVALID",
]
