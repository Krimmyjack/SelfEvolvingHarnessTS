"""Canonical fast-path mechanics for preparation methods and paired validation.

capability matrix 第 3/4 项：P6 的部署面就是 P6 自己的 fast path——候选供给（det/random/llm）
→ 效果去重 → 风险剔除 → selector 选择（空候选 abstain=None），以及三个组件各自的
**配对语义**（paired_*_run：除被测组件外身份一致 assert + 池不变性 assert，violation 一律 raise）。

K/slot 语义（prereg §4，冻结）：
  - K 是**请求 slot 预算**（== sampler.expected_total，冻结常量），不是候选数承诺；
  - det 阶梯固定 3 程序：det 分配 >3 时**如实短缺、不回填**（预算是上限不是配额承诺，
    其他 supplier 不得补位）；
  - 跨 supplier sha 去重后 realized unique pool size 可 < K，**逐 episode 落账**：
    generate_candidates_with_stats 返回 (pool, stats)，run_fast_path 返回 FastPathResult
    （dict 语义向后兼容，另挂 .pool_stats）。

RiskRule scope 闸（prereg §3.3，冻结）：when 是条件列表（**同规则内 AND**，prereg §4——
bin scope 的 lo/hi 两原子同规则合取；规则之间才是并集 ban），每个条件的 feature 强制
∈ P0_FEATURE_ALLOWLIST（{"snr","missing_rate"} ∪ P_FEATS；定义与来源注记见
H_ref config）。spec 级 RiskRuleSpec.validate() 为第一道闸；apply_risk 为第二道
同 allowlist 闸（纵深防御，逐条件），非 allowlist 特征 / 空条件列表一律 raise ValueError。

scope 类 edit 的端到端校验（prereg §4 门④）：paired_risk_run 不只比较 kept-mask，
对**作用域外每个 episode** 还比较两臂最终 prepared artifact 的字节级一致
（shape+dtype+tobytes；prepared_artifact 复用 runtime.executor.run_pipeline），
不一致 → FastPathPairingError。

程序执行复用统一 runtime executor，算子默认参数来自冻结 H_ref 配置；
步骤参数解析 = {**defaults[op], **explicit_params}。

确定性红线：
  - random 供给器无 RNG——伪随机流 = sha256(f"{uid}|{state_sha}|{i}")，其中 state_sha
    由 generate_candidates 传入 `state.sha_excluding("selector", "risk_rules")`（只随 sampler
    组件变化）——这使"只改 selector / 只改 risk 的两个 state 候选池按构造 bit 级一致"
    成为配对语义的结构保证，而非巧合。
  - llm_supplier 仅接受注入的 callable（须自身确定性），默认 None → 贡献 0 候选；
    本模块**不得联网**、不 import 任何 LLM client。
  - proxy_score 是 toy 版选择输入（见其 docstring），确定性，不承载科学意义。

红线：模块级依赖 stdlib + numpy + 对现有项目模块的只读 import；不读 results/ 与 data/。
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .executor import ExecutionResult, run_pipeline
from ..methods.h_ref_v02.config import (
    DET_PROGRAM_STEPS,
    GRAMMAR_DENOISERS,
    GRAMMAR_IMPUTERS,
    GRAMMAR_OUTLIERS,
    GRAMMAR_WINDOWS,
    GUARD_OPS,
    HREF_OPERATOR_DEFAULTS,
    P0_FEATURE_ALLOWLIST,
    PRESET_SCOPE_FEATURE,
    HRefState,
    RiskRuleSpec,
    canonical_json,
)

__all__ = [
    "Candidate",
    "FAILED_PROXY",
    "FastPathResult",
    "FastPathPairingError",
    "GRAMMAR_DENOISERS",
    "GRAMMAR_IMPUTERS",
    "GRAMMAR_OUTLIERS",
    "GRAMMAR_WINDOWS",
    "GUARD_OPS",
    "P0_FEATURE_ALLOWLIST",
    "P6PairingError",
    "apply_risk",
    "det_ladder",
    "enrich_candidate",
    "execute_candidate",
    "generate_candidates",
    "generate_candidates_with_stats",
    "make_candidate",
    "merge_preset_fingerprints",
    "paired_risk_run",
    "paired_sampler_run",
    "paired_selector_run",
    "prepared_artifact",
    "program_sha",
    "proxy_score",
    "random_grammar_sampler",
    "resolve_steps",
    "run_fast_path",
    "select",
    "toy_fingerprint",
]

FAILED_PROXY = -1.0e9        # 程序执行失败的 proxy 哨兵（有限值，排序稳定）


class FastPathPairingError(RuntimeError):
    """配对语义违规（tamper / 池不变性破坏），一律 raise。"""


P6PairingError = FastPathPairingError


# ════════════════════════════ Candidate ════════════════════════════
@dataclass(frozen=True)
class Candidate:
    """一个候选程序。program_steps = ((op, params), ...)；sha = 程序身份（canonical JSON
    sha256[:16]，只覆盖 program_steps——features/source 不进 sha）。features 勿原地改，
    enrich_candidate 返回新实例。"""

    program_steps: Tuple[Tuple[str, Dict[str, Any]], ...]
    source: str
    features: Dict[str, Any] = field(default_factory=dict)
    sha: str = ""

    def __post_init__(self) -> None:
        steps = tuple((str(op), dict(params or {})) for op, params in self.program_steps)
        object.__setattr__(self, "program_steps", steps)
        object.__setattr__(self, "features", dict(self.features or {}))
        object.__setattr__(self, "source", str(self.source))

    def op_names(self) -> Tuple[str, ...]:
        return tuple(op for op, _p in self.program_steps)


def program_sha(steps: Sequence[Tuple[str, Mapping[str, Any]]]) -> str:
    """程序身份 sha：canonical JSON（[[op, params], ...]，sort_keys）sha256[:16]。"""
    payload = [[str(op), dict(params or {})] for op, params in steps]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _static_features(steps: Sequence[Tuple[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    """执行无关的候选特征：n_steps（程序步数）、has_guard（含 outlier 类守卫算子 = 1.0）。"""
    ops = [op for op, _p in steps]
    return {
        "n_steps": float(len(ops)),
        "has_guard": 1.0 if any(op in GUARD_OPS for op in ops) else 0.0,
    }


def make_candidate(
    steps: Sequence[Tuple[str, Mapping[str, Any]]],
    source: str,
    features: Optional[Mapping[str, Any]] = None,
) -> Candidate:
    """构造候选：sha 一律由本函数从 steps 计算（供给器给的 sha 不被信任），静态特征自动填充。"""
    feats = _static_features(steps)
    feats.update(dict(features or {}))
    return Candidate(
        program_steps=tuple((op, dict(p or {})) for op, p in steps),
        source=source,
        features=feats,
        sha=program_sha(steps),
    )


# ════════════════════════════ 程序执行（复用现有模块） ════════════════════════════
@lru_cache(maxsize=1)
def _operator_defaults() -> Mapping[str, Dict[str, Any]]:
    return {name: dict(params) for name, params in HREF_OPERATOR_DEFAULTS.items()}


def resolve_steps(
    steps: Sequence[Tuple[str, Mapping[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """参数解析：{**H_ref defaults[op], **explicit_params}（不改缓存的默认）。"""
    defaults = _operator_defaults()
    return [(op, {**defaults.get(op, {}), **dict(p or {})}) for op, p in steps]


def execute_candidate(candidate: Candidate, series: np.ndarray) -> ExecutionResult:
    """执行候选程序（run_pipeline，异常围堵在 executor 内）。"""
    return run_pipeline(resolve_steps(candidate.program_steps), series, source=candidate.source)


def _aligned_finite(series: np.ndarray, artifact: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(series, dtype=float).ravel()
    art = np.asarray(artifact, dtype=float).ravel()
    m = min(raw.size, art.size)
    raw, art = raw[:m], art[:m]
    return raw, art, np.isfinite(raw) & np.isfinite(art)


def _proxy_from_result(res: ExecutionResult, series: np.ndarray) -> float:
    if not res.ok or res.artifact is None:
        return FAILED_PROXY
    raw, art, mask = _aligned_finite(series, res.artifact)
    if not mask.any():
        return 0.0
    d = art[mask] - raw[mask]
    rmsd = float(np.sqrt(np.mean(d * d)))
    return 1.0 / (1.0 + rmsd)


def _modified_fraction_from_result(res: ExecutionResult, series: np.ndarray) -> float:
    if not res.ok or res.artifact is None:
        return 1.0                                    # 失败 = 最大改动哨兵
    raw, art, _ = _aligned_finite(series, res.artifact)
    if raw.size == 0:
        return 1.0
    raw_finite = np.isfinite(raw)
    changed = ~raw_finite | ~np.isclose(
        np.where(raw_finite, raw, 0.0), np.where(raw_finite, art, 0.0), atol=1e-12, rtol=0.0
    )
    return float(changed.mean())


def proxy_score(candidate: Candidate, series: np.ndarray) -> float:
    """toy 版 proxy：1 / (1 + RMSD(执行产物, raw))，只在双方有限的位置上算；执行失败 →
    FAILED_PROXY。**声明：这只是 selector 的确定性选择输入，不承载任何科学意义**
    （它天然偏好改动小的程序——toy 部署面只需要一个廉价、确定、可区分的排序信号）。"""
    return _proxy_from_result(execute_candidate(candidate, series), series)


def enrich_candidate(candidate: Candidate, series: np.ndarray) -> Candidate:
    """执行一次，写入执行派生特征（proxy_score / modified_fraction / exec_ok）→ 新实例。"""
    res = execute_candidate(candidate, series)
    feats = dict(candidate.features)
    feats.update(_static_features(candidate.program_steps))
    feats["exec_ok"] = 1.0 if res.ok else 0.0
    feats["proxy_score"] = _proxy_from_result(res, series)
    feats["modified_fraction"] = _modified_fraction_from_result(res, series)
    return dataclasses.replace(candidate, features=feats)


def prepared_artifact(chosen: Optional[Candidate], series: np.ndarray) -> Optional[np.ndarray]:
    """chosen 执行后的最终 prepared artifact（部署语义；复用 runtime.executor.run_pipeline）。

    - chosen = None（abstain）→ 原序列 float64 ravel **副本**（不动数据的部署缺省）；
    - 执行失败（executor 异常围堵）→ None 哨兵；
    - 成功 → float64 ravel 的产物数组。
    paired_risk_run 的 prereg §4 门④ 端到端字节级校验以本函数输出为准。"""
    x = np.asarray(series, dtype=float).ravel()
    if chosen is None:
        return x.copy()
    res = execute_candidate(chosen, series)
    if not res.ok or res.artifact is None:
        return None
    return np.asarray(res.artifact, dtype=float).ravel()


def _artifacts_byte_equal(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> bool:
    """字节级一致：双 None 视为一致；否则 shape+dtype+tobytes 全等（np.ndarray.tobytes）。"""
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ════════════════════════════ 供给器（全确定性） ════════════════════════════
def det_ladder() -> List[Candidate]:
    """固定 3 程序阶梯（确定性现任供给；prereg §4 冻结）：identity 插补 → 守卫+平滑 →
    中值(w9)。det 分配 >3 时供给到 3 为止——**如实短缺不回填**（见 generate_candidates_with_stats）。"""
    return [make_candidate(list(program), source="det") for program in DET_PROGRAM_STEPS]


def random_grammar_sampler(
    uid: str,
    state_sha: str,
    n: int,
    params: Optional[Mapping[str, Any]] = None,
) -> List[Candidate]:
    """确定性伪随机文法采样：第 i 个候选的全部选择由 sha256(f"{uid}|{state_sha}|{i}")
    的字节驱动（byte0=imputer、byte1=outlier 含 None、byte2=denoiser 含 None、byte3=窗口）。
    采样内按 sha 去重；凑不满 n 个唯一程序时在 16n 次尝试后停止（返回 ≤ n）。
    params 可覆盖文法（imputers/outliers/denoisers/windows，空/缺省用模块默认）。"""
    p = dict(params or {})
    imputers = tuple(p.get("imputers") or GRAMMAR_IMPUTERS)
    outliers = tuple(p.get("outliers") or GRAMMAR_OUTLIERS)
    denoisers = tuple(p.get("denoisers") or GRAMMAR_DENOISERS)
    windows = tuple(int(w) for w in (p.get("windows") or GRAMMAR_WINDOWS))

    out: List[Candidate] = []
    seen: set = set()
    n = int(n)
    i = 0
    cap = max(16 * n, 16)
    while len(out) < n and i < cap:
        h = hashlib.sha256(f"{uid}|{state_sha}|{i}".encode("utf-8")).digest()
        steps: List[Tuple[str, Dict[str, Any]]] = [(imputers[h[0] % len(imputers)], {})]
        o = ((None,) + outliers)[h[1] % (len(outliers) + 1)]
        if o is not None:
            steps.append((o, {}))
        d = ((None,) + denoisers)[h[2] % (len(denoisers) + 1)]
        if d is not None:
            steps.append((d, {"window": windows[h[3] % len(windows)]}))
        cand = make_candidate(steps, source="random")
        if cand.sha not in seen:
            seen.add(cand.sha)
            out.append(cand)
        i += 1
    return out


LlmSupplier = Callable[[str, HRefState, int], Sequence[Any]]


def _collect_llm(
    llm_supplier: Optional[LlmSupplier], uid: str, state: HRefState, n: int
) -> List[Candidate]:
    """llm 供给：仅接受注入的 callable（须自身确定性、离线）；默认 None → 0 候选。
    返回项可以是 Candidate 或 steps 序列；sha 一律重算（不信任供给方声明的 sha）。"""
    if n <= 0 or llm_supplier is None:
        return []
    raw = list(llm_supplier(uid, state, n) or [])[: int(n)]
    out: List[Candidate] = []
    for item in raw:
        steps = item.program_steps if isinstance(item, Candidate) else item
        out.append(make_candidate(steps, source="llm"))
    return out


# ════════════════════════════ serving consumer ════════════════════════════
def generate_candidates_with_stats(
    uid: str,
    state: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier] = None,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """generate_candidates 的落账版：返回 (pool, stats)（prereg §4 K/slot 语义 + §6 披露）。

    供给：按 sampler.allocation 分配名额（det 取阶梯前 k1 个——阶梯固定 3 程序，k1>3 时
    **如实短缺不回填**；random 走确定性文法流；llm 走注入 callable），按 det→random→llm
    次序汇总并**按 sha 跨 supplier 效果去重**（保留首次出现）→ ≤ K 个。

    stats（JSON-native，逐 episode 落账）：
      uid / requested_K / allocation（快照）/
      supplied = {det, random, llm}（各 supplier 去重前实际产出条数——短缺在此可见）/
      det_shortfall = max(0, allocation.det − supplied.det)（阶梯只有 3 程序）/
      pre_dedup_size / **realized_pool_size**（跨 supplier sha 去重后 unique pool 大小）/
      dedup_removed = pre_dedup_size − realized_pool_size。

    预算完整性：K 必须 == state.sampler.expected_total（冻结常量），否则 raise ValueError。
    random 流的 state_sha = state.sha_excluding("selector", "risk_rules")：候选池只随
    sampler 组件变化 → selector/risk 配对的"池不变"是结构保证。"""
    if int(K) != state.sampler.expected_total:
        raise ValueError(
            f"预算不一致：K={K} ≠ sampler.expected_total={state.sampler.expected_total}"
        )
    alloc = state.sampler.allocation
    stream_sha = state.sha_excluding("selector", "risk_rules")

    det_part = det_ladder()[: int(alloc.get("det", 0))]
    random_part = random_grammar_sampler(
        uid, stream_sha, int(alloc.get("random", 0)), state.sampler.random_params
    )
    llm_part = _collect_llm(llm_supplier, uid, state, int(alloc.get("llm", 0)))
    pooled: List[Candidate] = [*det_part, *random_part, *llm_part]

    merged: List[Candidate] = []
    seen: set = set()
    for c in pooled:
        if c.sha in seen:
            continue
        seen.add(c.sha)
        merged.append(c)

    stats: Dict[str, Any] = {
        "uid": str(uid),
        "requested_K": int(K),
        "allocation": dict(alloc),
        "supplied": {"det": len(det_part), "random": len(random_part), "llm": len(llm_part)},
        "det_shortfall": max(0, int(alloc.get("det", 0)) - len(det_part)),
        "pre_dedup_size": len(pooled),
        "realized_pool_size": len(merged),
        "dedup_removed": len(pooled) - len(merged),
    }
    return merged, stats


def generate_candidates(
    uid: str,
    state: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier] = None,
) -> List[Candidate]:
    """向后兼容入口：= generate_candidates_with_stats(...)[0]（语义见该函数 docstring）。"""
    return generate_candidates_with_stats(uid, state, K, llm_supplier)[0]


def _rule_hits(rule: RiskRuleSpec, candidate: Candidate) -> bool:
    """then.target 命中语义：候选 sha == target，或 target ∈ 程序 op 名集合。"""
    target = rule.then["target"]
    return candidate.sha == target or target in candidate.op_names()


def apply_risk(
    candidates: Sequence[Candidate],
    fingerprint: Mapping[str, Any],
    state: HRefState,
) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
    """风险剔除：规则触发 = 其 when 条件列表**全部**在 fingerprint（toy dict）上成立
    （同规则 AND，prereg §4 冻结；任一条件的 feature 缺失 → 该条件不成立 → 规则不触发，
    保守方向——语义在 RiskRuleSpec.matches）。触发规则的 target 命中候选被剔除；
    多规则之间独立求值 = 并集 ban（那是本来正确的层）。返回 (kept, banned)；banned 记录
    {sha, source, rule_id, target}（首个命中规则，按 state.risk_rules 次序，确定性）。
    kept 保持输入次序。

    第二道 allowlist 闸（prereg §3.3，纵深防御——第一道在 RiskRuleSpec.validate）：
    任何规则的任一 when 条件 feature ∉ P0_FEATURE_ALLOWLIST → raise ValueError（不触发
    也拦，绕过构造期校验的 outcome/response/series id/domain id 类特征在此兜底）；
    空条件列表同样 raise（空合取 = 无条件全局 ban，绕过 validate 走私进 state 也在此拦）。"""
    for r in state.risk_rules:
        if not r.when:
            raise ValueError(
                f"apply_risk 拒绝：规则 {r.rule_id!r} 的 when 条件列表为空"
                f"（空合取 = 无条件全局 ban，禁止）"
            )
        for cond in r.when:
            feat = cond.get("feature") if isinstance(cond, Mapping) else None
            # preset 成员资格是 RiskRule scope 的保留维（F7），与 P0 数值 allowlist 并列合法。
            if feat != PRESET_SCOPE_FEATURE and feat not in P0_FEATURE_ALLOWLIST:
                raise ValueError(
                    f"apply_risk 拒绝：规则 {r.rule_id!r} 的 when 条件 feature {feat!r} "
                    f"不在冻结 P0 allowlist（prereg §3.3：{sorted(P0_FEATURE_ALLOWLIST)}）"
                    f"、也非 preset 成员资格保留维 {PRESET_SCOPE_FEATURE!r}"
                )
    fired = [r for r in state.risk_rules if r.matches(fingerprint)]
    kept: List[Candidate] = []
    banned: List[Dict[str, Any]] = []
    for c in candidates:
        hit = next((r for r in fired if _rule_hits(r, c)), None)
        if hit is None:
            kept.append(c)
        else:
            banned.append(
                {"sha": c.sha, "source": c.source, "rule_id": hit.rule_id,
                 "target": hit.then["target"]}
            )
    return kept, banned


def select(candidates: Sequence[Candidate], state: HRefState) -> Optional[Candidate]:
    """按 selector spec 选一个；空候选 → abstain = None。

    proxy_rank：features["proxy_score"] 降序（缺失按 0.0），tie-break program sha 升序。
    weighted_features：score = Σ w_f · float(features.get(f, 0.0)) 降序，tie-break sha 升序。"""
    if not candidates:
        return None
    sel = state.selector
    sel.validate()
    if sel.kind == "proxy_rank":
        def key(c: Candidate) -> Tuple[float, str]:
            return (-float(c.features.get("proxy_score", 0.0)), c.sha)
    else:  # weighted_features（SELECTOR_KINDS 全集 = 两种；validate 已挡未知）
        weights = sel.weights

        def key(c: Candidate) -> Tuple[float, str]:
            score = sum(w * float(c.features.get(f, 0.0)) for f, w in weights.items())
            return (-score, c.sha)

    return min(candidates, key=key)


def toy_fingerprint(series: np.ndarray) -> Dict[str, float]:
    """toy 指纹（风险规则的 when 求值域）：只发 allowlist 键 {snr, missing_rate}
    （prereg §3.3；P_FEATS 结构特征由正式 fingerprint 计算器供给，缺失 → 规则不触发）。

    确定性、无科学意义的 toy 估计：missing_rate = 1 − 有限值占比；snr =
    max(var(x)−var(diff(x))/2, 0) / max(var(diff(x))/2, 1e-12)（一阶差分噪声估计；
    有限值 <3 个 → 0.0）。"""
    x = np.asarray(series, dtype=float).ravel()
    finite = np.isfinite(x)
    missing_rate = float(1.0 - finite.mean()) if x.size else 1.0
    xf = x[finite]
    if xf.size >= 3:
        noise_var = float(np.var(np.diff(xf))) / 2.0
        signal_var = max(float(np.var(xf)) - noise_var, 0.0)
        snr = signal_var / max(noise_var, 1e-12)
    else:
        snr = 0.0
    return {"snr": float(snr), "missing_rate": missing_rate}


def _iter_views(views: Any) -> Iterable[Tuple[str, np.ndarray]]:
    """接受 Mapping[uid → series]，或可迭代的 (uid, series) / 带 .uid/.history 的对象
    （如 judge_closed_form.SeriesView）。保持给定次序（确定性来自调用方输入）。"""
    if isinstance(views, Mapping):
        for uid, series in views.items():
            yield str(uid), series
        return
    for v in views:
        if hasattr(v, "uid") and hasattr(v, "history"):
            yield str(v.uid), v.history
        else:
            uid, series = v
            yield str(uid), series


def _fp_for(
    uid: str, series: np.ndarray, fingerprints: Optional[Mapping[str, Mapping[str, Any]]]
) -> Mapping[str, Any]:
    if fingerprints is not None and uid in fingerprints:
        return fingerprints[uid]
    return toy_fingerprint(series)


def merge_preset_fingerprints(
    episodes: Iterable[Any],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """把 episode.preset 并入 per-uid fingerprint（F7）：RiskRule 的 preset 成员资格 scope
    在 apply_risk/matches 里对 fingerprint["preset"] 求值，故 arm 运行前须让 preset 可见。

    每 episode（须有 .uid/.history/.preset）→ {**(fingerprints[uid] 或 toy_fingerprint(history)),
    "preset": preset}：数值特征原样保留（与 _fp_for 无 preset 时口径 bit 级一致），仅追加
    保留维 "preset"。返回新 dict，不改输入。调用方把它当作 fingerprints 传给 fast path 即可。"""
    out: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        uid = str(ep.uid)
        base = _fp_for(uid, ep.history, fingerprints)
        merged = dict(base)
        merged[PRESET_SCOPE_FEATURE] = str(ep.preset)
        out[uid] = merged
    return out


@dataclass
class _UidRun:
    pool: List[Candidate]            # enriched，风险剔除前
    kept: List[Candidate]
    banned: List[Dict[str, Any]]
    chosen: Optional[Candidate]
    stats: Dict[str, Any]            # generate_candidates_with_stats 落账 + serving 侧字段


def _run_one(
    uid: str,
    series: np.ndarray,
    state: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
) -> _UidRun:
    raw_pool, stats = generate_candidates_with_stats(uid, state, K, llm_supplier)
    pool = [enrich_candidate(c, series) for c in raw_pool]
    kept, banned = apply_risk(pool, _fp_for(uid, series, fingerprints), state)
    chosen = select(kept, state)
    stats = dict(stats)
    stats["kept_pool_size"] = len(kept)
    stats["n_banned"] = len(banned)
    stats["abstained"] = chosen is None
    return _UidRun(pool=pool, kept=kept, banned=banned, chosen=chosen, stats=stats)


class FastPathResult(Dict[str, Optional[Candidate]]):
    """run_fast_path 返回类型：dict 语义 = {uid: chosen | None}（**向后兼容**——迭代/取值/
    len/set 一如从前）；新增属性 `pool_stats: {uid: stats}`（generate_candidates_with_stats
    的逐 episode 落账 + kept_pool_size/n_banned/abstained；prereg §4 realized pool size
    逐 episode 落账、§6 "realized pool size 分布" 披露的数据源）。"""

    def __init__(
        self,
        chosen: Mapping[str, Optional[Candidate]],
        pool_stats: Mapping[str, Dict[str, Any]],
    ) -> None:
        super().__init__(chosen)
        self.pool_stats: Dict[str, Dict[str, Any]] = {u: dict(s) for u, s in pool_stats.items()}


def run_fast_path(
    views: Any,
    state: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier] = None,
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> FastPathResult:
    """serving consumer：逐 uid 供给→enrich→风险剔除→选择。返回 FastPathResult
    （dict 语义 {uid: chosen | None} 向后兼容；.pool_stats 逐 episode 落账 realized
    unique pool size 等，见 FastPathResult docstring）。"""
    runs = {uid: _run_one(uid, series, state, K, llm_supplier, fingerprints)
            for uid, series in _iter_views(views)}
    return FastPathResult(
        {uid: r.chosen for uid, r in runs.items()},
        {uid: r.stats for uid, r in runs.items()},
    )


# ════════════════════════════ paired validator（capability matrix 第 4 项） ════════════════════════════
def _assert_same_except(stateA: HRefState, stateB: HRefState, varying: str) -> None:
    if stateA.sha_excluding(varying) != stateB.sha_excluding(varying):
        raise FastPathPairingError(
            f"配对违规：两 state 除 {varying} 外的组件 sha 不一致"
            f"（{stateA.sha_excluding(varying)} ≠ {stateB.sha_excluding(varying)}）——"
            f"疑似 {varying} 之外的组件被篡改"
        )


def _run_both(
    views: Any,
    stateA: HRefState,
    stateB: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
) -> Tuple[Dict[str, _UidRun], Dict[str, _UidRun]]:
    pairs = list(_iter_views(views))
    ra = {uid: _run_one(uid, s, stateA, K, llm_supplier, fingerprints) for uid, s in pairs}
    rb = {uid: _run_one(uid, s, stateB, K, llm_supplier, fingerprints) for uid, s in pairs}
    return ra, rb


def _shas(cands: Sequence[Candidate]) -> Tuple[str, ...]:
    return tuple(c.sha for c in cands)


def paired_selector_run(
    views: Any,
    stateA: HRefState,
    stateB: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier] = None,
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """selector 配对：①assert 两 state 除 selector 外 sha 一致；②逐 uid assert 候选池
    sha 集合完全相同（剔除前后都查——两侧只许 selector 不同，池必须 bit 级一致）；
    ③返回两套 chosen：{"A": {uid: cand|None}, "B": {...}, "pool_shas": {uid: (...,)}}。"""
    _assert_same_except(stateA, stateB, "selector")
    ra, rb = _run_both(views, stateA, stateB, K, llm_supplier, fingerprints)
    pool_shas: Dict[str, Tuple[str, ...]] = {}
    for uid in ra:
        if _shas(ra[uid].pool) != _shas(rb[uid].pool):
            raise FastPathPairingError(f"selector 配对违规：uid={uid} 剔除前候选池 sha 不一致")
        if _shas(ra[uid].kept) != _shas(rb[uid].kept):
            raise FastPathPairingError(f"selector 配对违规：uid={uid} 剔除后候选池 sha 不一致")
        pool_shas[uid] = _shas(ra[uid].kept)
    return {
        "A": {uid: r.chosen for uid, r in ra.items()},
        "B": {uid: r.chosen for uid, r in rb.items()},
        "pool_shas": pool_shas,
    }


def paired_sampler_run(
    views: Any,
    stateA: HRefState,
    stateB: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier] = None,
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """sampler 配对：①assert 两 state 除 sampler 外 sha 一致；②assert 两侧总 K 相同且 == K
    （预算冻结）；③返回两套池+chosen：{"A": {"pools": {uid: (cand,...)}, "chosen": {...}},
    "B": {...}}（pools = enriched 剔除前池；组成允许不同——那正是被测变量）。"""
    _assert_same_except(stateA, stateB, "sampler")
    if stateA.sampler.expected_total != stateB.sampler.expected_total:
        raise FastPathPairingError(
            f"sampler 配对违规：两侧总 K 不同（{stateA.sampler.expected_total} ≠ "
            f"{stateB.sampler.expected_total}）"
        )
    if int(K) != stateA.sampler.expected_total:
        raise FastPathPairingError(
            f"sampler 配对违规：K={K} ≠ 冻结总预算 {stateA.sampler.expected_total}"
        )
    ra, rb = _run_both(views, stateA, stateB, K, llm_supplier, fingerprints)
    return {
        "A": {"pools": {u: tuple(r.pool) for u, r in ra.items()},
              "chosen": {u: r.chosen for u, r in ra.items()}},
        "B": {"pools": {u: tuple(r.pool) for u, r in rb.items()},
              "chosen": {u: r.chosen for u, r in rb.items()}},
    }


def paired_risk_run(
    views: Any,
    stateA: HRefState,
    stateB: HRefState,
    K: int,
    llm_supplier: Optional[LlmSupplier] = None,
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """risk 配对：①assert 两 state 除 risk_rules 外 sha 一致；②逐 uid assert 剔除前池
    bit 级一致；③mask 级 assert：**非 scope 内候选集不变**——kept 掩码只允许在
    "两侧规则差集（对称差）在该 uid 指纹上触发且命中"的候选上不同，其余任何差异 raise；
    ④**端到端字节级 assert（prereg §4 门④，升级：不只比较 kept-mask）**：作用域外每个
    episode（= 差集规则未在该 uid 触发或未命中池内任何候选，scope 集为空）比较两臂最终
    prepared artifact（prepared_artifact：chosen 执行产物；abstain → 原序列）的字节级
    一致（shape+dtype+tobytes），不一致 → raise。
    返回 {"A": {"kept": {...}, "banned": {...}, "chosen": {...}}, "B": {...},
    "out_of_scope_verified": (uid, ...)}（通过门④字节级校验的作用域外 episode 清单）。"""
    _assert_same_except(stateA, stateB, "risk_rules")
    ra, rb = _run_both(views, stateA, stateB, K, llm_supplier, fingerprints)

    canon_a = {r.rule_id: r for r in stateA.risk_rules}
    canon_b = {r.rule_id: r for r in stateB.risk_rules}
    ja = {rid: r.to_dict() for rid, r in canon_a.items()}
    jb = {rid: r.to_dict() for rid, r in canon_b.items()}
    diff_rules = [r for rid, r in canon_a.items() if jb.get(rid) != ja[rid]] + [
        r for rid, r in canon_b.items() if ja.get(rid) != jb[rid]
    ]

    view_map = {uid: series for uid, series in _iter_views(views)}
    out_of_scope_verified: List[str] = []
    for uid in ra:
        if _shas(ra[uid].pool) != _shas(rb[uid].pool):
            raise FastPathPairingError(f"risk 配对违规：uid={uid} 剔除前候选池 sha 不一致")
        fp = _fp_for(uid, view_map[uid], fingerprints)
        scope = {
            c.sha
            for c in ra[uid].pool
            if any(r.matches(fp) and _rule_hits(r, c) for r in diff_rules)
        }
        kept_a = set(_shas(ra[uid].kept))
        kept_b = set(_shas(rb[uid].kept))
        for c in ra[uid].pool:
            if (c.sha in kept_a) != (c.sha in kept_b) and c.sha not in scope:
                raise FastPathPairingError(
                    f"risk 配对违规：uid={uid} 非 scope 候选 {c.sha} 的 kept 掩码在两侧不同"
                )
        if not scope:                       # 作用域外 episode → 门④ 端到端字节级校验
            art_a = prepared_artifact(ra[uid].chosen, view_map[uid])
            art_b = prepared_artifact(rb[uid].chosen, view_map[uid])
            if not _artifacts_byte_equal(art_a, art_b):
                raise FastPathPairingError(
                    f"risk 配对违规（门④）：uid={uid} 作用域外 episode 的两臂最终 "
                    f"prepared artifact 字节级不一致"
                )
            out_of_scope_verified.append(uid)
    return {
        "A": {"kept": {u: tuple(r.kept) for u, r in ra.items()},
              "banned": {u: tuple(r.banned) for u, r in ra.items()},
              "chosen": {u: r.chosen for u, r in ra.items()}},
        "B": {"kept": {u: tuple(r.kept) for u, r in rb.items()},
              "banned": {u: tuple(r.banned) for u, r in rb.items()},
              "chosen": {u: r.chosen for u, r in rb.items()}},
        "out_of_scope_verified": tuple(out_of_scope_verified),
    }
