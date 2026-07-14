"""p6/c0_runner.py — P6 C0 校准 runner（prereg §3：identity gate / ε / δ / P0 cutpoints）。

外审 NO-GO 兑现件：C0 只用 C0 块；**identity gate FAIL → P6TechnicalStop**（判官身份未确立，
D/V/U 不开启、本 freeze 作废；不设 Adam fallback 分支）。全部承重环节依赖注入，机械测试
不读真实数据、不联网、不 import torch（真 trainer 由 make_adam_trainer 惰性加载）。

组件（prereg §3.1–§3.3 的算法冻结；数值以 C0_FREEZE 记录追加）：
  1. episode 构造 build_episodes：底层序列 × 4 preset（load_real 语义——per-series z-score 的
     clean 为信号源、degrade_fn 注入退化、future=clean 末 H=48）；corruption seed =
     int(sha256("p6deg|{config}|{item}|{preset}").hexdigest()[:8], 16)（prereg §2 字面规则，
     前 8 个 hex 字符解析为整数）。
  2. P_gate 8 程序（§3.1 冻结清单，raw_identity=不处理）→ op 步骤映射；执行 =
     sandbox.executor.run_pipeline + fast_path.resolve_steps（= harness.layers.minimal_l2
     的 operator_defaults 参数解析），不自铸执行器。
  3. 闭式判官（per-domain shared、series_weight="equal"）；**每次正式拟合双路对拍自检
     paired_judge_fit（prereg §1 最新判据）**：per-episode loss 与 batch utility 的
     max|Δ| ≤ 1e-9（主判据）∧ W 相对差 max|ΔW|/max|W| ≤ 1e-6（辅助），任一超限 →
     P6TechnicalAbort。fit/rebuild 可注入（构造性测试篡改路径）。
  4. Adam 参照（trainer 注入；真实现 = make_adam_trainer：_torch_models.train_forecaster 的
     DLinear、CPU、每 fit 前 seed_all(seed)、torch.use_deterministic_algorithms(True)）。
     单 fit 超时 15 min → abort；Adam 任一拟合产 NaN/发散 → abort。
  5. identity gate（§3.1）：**PASS = ① ∧ ② ∧ ④ ∧ ③′**；ε 先于 gate 计算（near-zero 带 ε/4、
     top-1 容差 ε/2 都依赖它）。
  6. ε = 0.02·J_raw_C0（J_raw = raw 臂 per-domain batch loss 的域等权均值）；δ_safe = 2.5·ε；
     P0 bin cutpoints = snr / missing_rate 的 C0 四分位（quantile linear 插值；fingerprint
     提取注入）。
  7. C0_FREEZE record dict + write_c0_freeze；gate FAIL → 记录（若给 out_path 先写盘）后
     raise P6TechnicalStop（record 挂在异常 .record 上）。

——冻结释义（实现裁量点，全部机械可测）——
  A. **判官 ingestion fill**：raw_identity 在 miss preset 上把 NaN 直喂闭式判官会 NaN 塌缩；
     项目先例（evaluators/report_target.py:85 "raw(含 miss)有 NaN → 线性填补后才能窗化/训练"）
     = 评估端最小 ingestion 步。冻结为 judge_ingest：非有限（含 ±inf）→ NaN → 线性插补
     （首尾最近值钳制，= operators._common.interp_nan 语义）；**统一作用于全部程序臂**
     （带 impute 的臂本就有限 → no-op）；全非有限 → P6TechnicalAbort。
  B. **Adam 训练窗 pooled、无 series 等权**：train_forecaster 无样本权重接口；prereg §1 只
     冻结 epochs/lr/bs——Adam 是"部署真实口径"参照，按窗 pooled 训练；series_weight="equal"
     是闭式 ridge estimand 的属性。评估协议与判官逐字一致（history-only z-score、末 48 窗、
     future_norm RMSE）→ 判据①可比。
  C. **near-zero 带（④）**：程序对的 |gain| < ε/4 剔出符号分母——**任一侧**（闭式或 Adam）
     落带即剔（近零符号是噪声，与哪个判官产生无关）；带是严格 <（|gain| 恰 = ε/4 保留）。
     ④ 分母为 0 → 符号一致率无法建立 → 判据 FAIL（不缺省通过）。
  D. **top-1 一致（③′）**：Adam top-1 ∈ 闭式 top-2 **且** 闭式尺度 loss 差
     cf_loss(adam_top1) − cf_loss(cf_top1) ≤ ε/2（ε-tolerant 的连贯读法：跨判官的水准差
     由判据①管，③′ 管排序）。argmin/top-2 并列按冻结程序次序（确定性）。
  E. **Spearman 退化**：任一侧秩全并列（分母 0）→ ρ 冻结为 0.0（保守趋 FAIL）；ties 取均秩。
  F. **预算**：8 程序 × 4 域 × 3 seeds = 96 次 Adam 拟合 ≤ prereg §3.1 预算行 108
     （其 "(8 程序+1 raw)" 与冻结清单首项 raw_identity 重复计数；本 runner 以清单为准，
     raw 只拟合一次，实耗 96 落成本账）。
  G. **超时**：Windows 无 SIGALRM（sandbox/executor.py 同注）；15 min 为事后墙钟检查
     （软超时）：单 fit 耗时 > 上限 → P6TechnicalAbort。
  H. prereg §3.2 "全 84 raw-退化 episode" 与 manifest 算术（C0=16 底层 × 4 preset=64）不合；
     本 runner 不硬编码任何 episode 数，J_raw 从实际提供的 C0 episodes 计算并落账。

红线：不改任何现有文件；模块级不 import torch/网络/data 文件读取（data.load_real 只取
FORECAST_PRESETS 与 _zscore 常量语义，import 无 IO）。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.load_real import FORECAST_PRESETS, _zscore
from ..sandbox.executor import run_pipeline
from .fast_path import resolve_steps
from .judge_closed_form import (
    DomainFit,
    HORIZON,
    PROTOCOL_ID,
    SeriesView,
    fit_domain,
    fit_domain_rebuild,
)
from .materializer import P6TechnicalAbort
from .metrics import gain

__all__ = [
    "ADAM_BS",
    "ADAM_EPOCHS",
    "ADAM_FIT_TIMEOUT_SECONDS",
    "ADAM_LR",
    "C0_PRESETS",
    "C0_SEEDS",
    "DELTA_SAFE_MULTIPLIER",
    "DUAL_PATH_LOSS_ATOL",
    "DUAL_PATH_W_RTOL",
    "EPSILON_RATE",
    "GATE1_REL_TOL",
    "GATE2_MEDIAN_RHO_MIN",
    "GATE3_TOP1_RATE_MIN",
    "GATE4_SIGN_RATE_MIN",
    "JUDGE_CFG_FROZEN",
    "P6Episode",
    "P6TechnicalAbort",
    "P6TechnicalStop",
    "P_GATE_PROGRAMS",
    "P_GATE_PROGRAM_IDS",
    "RAW_PROGRAM_ID",
    "build_episodes",
    "compute_cutpoints",
    "degradation_seed",
    "FROZEN_TRAINER_FIT_TIMEOUT_SECONDS",
    "P6FrozenParamError",
    "assert_c0_frozen_params",
    "c0_frozen_literals",
    "evaluate_identity_gate",
    "frozen_literals_digest",
    "judge_ingest",
    "make_adam_trainer",
    "make_real_degrade_fn",
    "make_torch_forecaster_trainer",
    "paired_judge_fit",
    "prepared_views_for_program",
    "run_c0_formal",
    "run_c0_unfrozen",
    "run_gate_program",
    "spearman_rho",
    "write_c0_freeze",
]

# ── 冻结常量（prereg §1/§3） ─────────────────────────────────────────────
C0_PRESETS: Tuple[str, ...] = tuple(FORECAST_PRESETS)   # load_real 网格 4 preset（单一真源）
C0_SEEDS: Tuple[int, ...] = (0, 1, 2)                   # prereg §3.1：3 seeds

DUAL_PATH_LOSS_ATOL = 1e-9        # prereg §1 主判据：per-episode loss 与 batch utility max|Δ|
DUAL_PATH_W_RTOL = 1e-6           # prereg §1 辅助：max|ΔW|/max|W|

EPSILON_RATE = 0.02               # ε = 0.02 · J_raw_C0（prereg §3.2）
DELTA_SAFE_MULTIPLIER = 2.5       # δ_safe = 2.5 · ε（prereg §3.2）
CUTPOINT_QUANTILES: Tuple[float, ...] = (0.25, 0.5, 0.75)   # §3.3 四分位 cutpoints

GATE1_REL_TOL = 0.10              # ①：|U_cf(raw) − U_adam(raw)| ≤ 0.10·U_adam(raw)
GATE2_MEDIAN_RHO_MIN = 0.7        # ②：4 域 Spearman 中位数 ≥ 0.7
GATE3_TOP1_RATE_MIN = 0.6         # ③′：episode 级 ε-tolerant top-1 一致率 ≥ 0.6
GATE4_SIGN_RATE_MIN = 0.75        # ④：preset 级符号一致率 ≥ 0.75（near-zero 带外）

ADAM_EPOCHS, ADAM_LR, ADAM_BS = 120, 1e-2, 256    # prereg §1 Adam-DLinear 协议
ADAM_FIT_TIMEOUT_SECONDS = 900.0                  # 单 fit 15 min（软超时，见 docstring G）

#: 判官协议冻结默认（prereg §1）：λ=1e-3、stride=4、window_cap=None、series_weight="equal"。
JUDGE_CFG_FROZEN: Mapping[str, Any] = {
    "lam": 1e-3, "stride": 4, "window_cap": None, "series_weight": "equal",
}
_JUDGE_CFG_KEYS = frozenset(JUDGE_CFG_FROZEN)

RAW_PROGRAM_ID = "raw_identity"
#: P_gate 8 程序（prereg §3.1 冻结清单；raw_identity=不处理，其余经 resolve_steps 补
#: minimal_l2 defaults 后由 run_pipeline 执行）。
P_GATE_PROGRAMS: Tuple[Tuple[str, Tuple[Tuple[str, Dict[str, Any]], ...]], ...] = (
    (RAW_PROGRAM_ID, ()),
    ("impute_linear", (("impute_linear", {}),)),
    ("il+winsorize", (("impute_linear", {}), ("winsorize", {}))),
    ("il+winsorize+savgol",
     (("impute_linear", {}), ("winsorize", {}), ("denoise_savgol", {}))),
    ("il+median_w9", (("impute_linear", {}), ("denoise_median", {"window": 9}))),
    ("il+median_w15", (("impute_linear", {}), ("denoise_median", {"window": 15}))),
    ("il+median_w25", (("impute_linear", {}), ("denoise_median", {"window": 25}))),
    ("il+smooth_ma_w5", (("impute_linear", {}), ("smooth_ma", {"window": 5}))),
)
P_GATE_PROGRAM_IDS: Tuple[str, ...] = tuple(g for g, _ in P_GATE_PROGRAMS)
_PROGRAM_STEPS: Mapping[str, Tuple[Tuple[str, Dict[str, Any]], ...]] = dict(P_GATE_PROGRAMS)

MIN_EPISODE_LEN = 144             # = materializer.MIN_SERIES_LEN（history ≥ 96 + future 48）


class P6TechnicalStop(RuntimeError):
    """C0 identity gate FAIL → technical stop（prereg §3）：判官身份未确立，
    D/V/U 不开启、本 freeze 作废。record 挂在 .record。"""

    def __init__(self, message: str, record: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.record = record


class P6FrozenParamError(RuntimeError):
    """正式入口检出偏离 prereg 冻结字面量（G4 / codex finding 36）。

    run_c0_formal/run_cycle_formal/run_u_eval_formal（**唯一合法正式入口**）机械断言全部
    prereg §1/§3/§4/§5 冻结字面量（seeds=(0,1,2)、bootstrap_b=2000、cycle boot
    seed=20260711+cycle、U boot seed=20260714、trainer 超时=900.0、K=8、C0=64 episodes/
    4 域/每 series 4 preset/96 Adam fits）；任一不符 → raise。run_*_unfrozen（测试专用）
    不做断言，正式运行禁止调用。"""


# ── 正式入口冻结字面量（prereg §1/§3/§4/§5；drift-guard 测试对照 prereg 文本） ──
FROZEN_PAIRED_SEEDS: Tuple[int, ...] = (0, 1, 2)     # prereg §1：paired seeds {0,1,2}
FROZEN_BOOTSTRAP_B: int = 2000                        # prereg §4：B=2000
FROZEN_CYCLE_BOOTSTRAP_SEED_BASE: int = 20260711      # prereg §4：PRNG=default_rng(20260711+cycle)
FROZEN_U_BOOTSTRAP_SEED: int = 20260714               # prereg §4：U 终评 seed=20260714
FROZEN_K_SLOT_BUDGET: int = 8                         # prereg §4：K=8 slot 预算
FROZEN_TRAINER_FIT_TIMEOUT_SECONDS: float = 900.0     # prereg §3.1：单 fit 软超时 15 min（G4 补入断言）
FROZEN_C0_N_EPISODES: int = 64                        # prereg §3.2：16 series × 4 preset
FROZEN_C0_N_DOMAINS: int = 4                          # prereg §3.1：4 域
FROZEN_C0_PRESETS_PER_SERIES: int = 4                 # prereg §2：同 series 4 preset
FROZEN_C0_ADAM_FITS: int = 96                         # prereg §3.1：8 程序 × 4 域 × 3 seeds


def _assert_timeout(timeout: float, where: str) -> None:
    if float(timeout) != 900.0:
        raise P6FrozenParamError(f"formal {where}：trainer fit 超时必须 == 900.0，得到 {timeout!r}")


def frozen_literals_digest(entrypoint: str, literals: Mapping[str, Any]) -> str:
    """正式入口冻结字面量指纹（写入输出记录/台账/final packet，证明数字出自正式入口）。"""
    payload = {"entrypoint": str(entrypoint),
               "frozen_literals": {str(k): literals[k] for k in sorted(literals)}}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_c0_frozen_params(
    episodes: Sequence["P6Episode"],
    seeds: Sequence[int],
    adam_fit_timeout_seconds: float = FROZEN_TRAINER_FIT_TIMEOUT_SECONDS,
) -> None:
    """run_c0_formal 断言（prereg §3；值写死字面量）。任一不符 → P6FrozenParamError。"""
    if tuple(int(s) for s in seeds) != (0, 1, 2):
        raise P6FrozenParamError(f"formal C0：seeds 必须 == (0, 1, 2)，得到 {tuple(seeds)!r}")
    if len(episodes) != 64:
        raise P6FrozenParamError(
            f"formal C0：episodes 必须 == 64（16 series × 4 preset），得到 {len(episodes)}"
        )
    domains = {ep.config for ep in episodes}
    if len(domains) != 4:
        raise P6FrozenParamError(f"formal C0：域数必须 == 4，得到 {sorted(domains)}")
    per_series: Dict[str, set] = {}
    for ep in episodes:
        per_series.setdefault(ep.series_uid, set()).add(ep.preset)
    bad = {s: sorted(p) for s, p in per_series.items() if len(p) != 4}
    if bad:
        raise P6FrozenParamError(f"formal C0：每 series 必须恰 4 preset，违规 {bad}")
    n_adam = len(P_GATE_PROGRAM_IDS) * len(domains) * len(list(seeds))
    if n_adam != 96:
        raise P6FrozenParamError(
            f"formal C0：Adam 拟合预算必须 == 96（8 程序 × 4 域 × 3 seeds），得到 {n_adam}"
        )
    _assert_timeout(adam_fit_timeout_seconds, "C0")


# ════════════════════════════ 1. episode 构造 ════════════════════════════
@dataclass(frozen=True)
class P6Episode:
    """一个 episode = 底层序列 × preset。history=退化后（可含 NaN）、future=clean 末 H（有限）。"""

    uid: str            # f"{config}:{item_id}:{preset}"
    series_uid: str     # 底层序列身份 f"{config}:{item_id}"（bootstrap cluster / 4 preset 同抽）
    config: str
    preset: str
    history: np.ndarray
    future: np.ndarray


def degradation_seed(config: str, item_id: str, preset: str) -> int:
    """corruption seed = sha256("p6deg|{config}|{item}|{preset}") 前 8 个 hex 字符 → int（prereg §2）。"""
    digest = hashlib.sha256(f"p6deg|{config}|{item_id}|{preset}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _coerce_series(item: Any, idx: int) -> Tuple[str, str, np.ndarray]:
    """series_list 条目 → (config, item_id, clean)。接受 (config, item_id, array) 元组、
    含同名键的 mapping、或含同名属性的对象。"""
    if isinstance(item, Mapping):
        try:
            return str(item["config"]), str(item["item_id"]), np.asarray(item["clean"], float)
        except KeyError as exc:
            raise ValueError(f"series_list[{idx}] 缺键 {exc}") from None
    if hasattr(item, "config") and hasattr(item, "item_id") and hasattr(item, "clean"):
        return str(item.config), str(item.item_id), np.asarray(item.clean, float)
    try:
        config, item_id, clean = item
    except Exception:
        raise ValueError(
            f"series_list[{idx}] 须为 (config, item_id, clean) / mapping / 对象，"
            f"got {type(item).__name__}"
        ) from None
    return str(config), str(item_id), np.asarray(clean, float)


def build_episodes(
    series_list: Sequence[Any],
    degrade_fn: Callable[[np.ndarray, str, int], np.ndarray],
    *,
    presets: Sequence[str] = C0_PRESETS,
    standardize: bool = True,
) -> List[P6Episode]:
    """底层序列 × preset → episodes（load_real 语义；degrade_fn 注入）。

    - standardize=True：clean 先 per-series z-score（load_real._zscore，nanmean/nanstd、
      护栏 1e-9）——真实 Monash 跨域尺度差 5 个数量级，z-score 是 load_real 语义的必需部分；
    - degrade_fn(clean_z, preset, seed) → 同长退化序列（可含 NaN）；seed = degradation_seed；
    - history = degraded[:len-H]、future = clean_z[len-H:]（H = 判官 HORIZON = 48）。
    abort（P6TechnicalAbort）：clean 非有限 / 长度 < 144 / degrade_fn 改变长度。
    """
    episodes: List[P6Episode] = []
    for idx, item in enumerate(series_list):
        config, item_id, clean = _coerce_series(item, idx)
        clean = clean.ravel()
        if clean.size < MIN_EPISODE_LEN:
            raise P6TechnicalAbort(
                f"{config}:{item_id}: 长度 {clean.size} < {MIN_EPISODE_LEN}（prereg §2 eligibility）"
            )
        if not np.all(np.isfinite(clean)):
            raise P6TechnicalAbort(f"{config}:{item_id}: clean 含非有限值（下载完整性故障）")
        sig = _zscore(clean) if standardize else clean.astype(float)
        cut = sig.size - HORIZON
        for preset in presets:
            seed = degradation_seed(config, item_id, str(preset))
            degraded = np.asarray(degrade_fn(sig, str(preset), seed), dtype=float).ravel()
            if degraded.shape != sig.shape:
                raise P6TechnicalAbort(
                    f"{config}:{item_id}:{preset}: degrade_fn 改变长度 "
                    f"{sig.shape} → {degraded.shape}（退化必须保长）"
                )
            episodes.append(
                P6Episode(
                    uid=f"{config}:{item_id}:{preset}",
                    series_uid=f"{config}:{item_id}",
                    config=config,
                    preset=str(preset),
                    history=degraded[:cut].copy(),
                    future=sig[cut:].copy(),
                )
            )
    return episodes


def make_real_degrade_fn() -> Callable[[np.ndarray, str, int], np.ndarray]:
    """真跑用 degrade_fn：load_real 的 preset 网格语义 = synthetic_gen._degrade(clean,
    PATTERN_PARAMS[preset], seed)（惰性 import；测试不调用即不触及）。"""
    from ..data.synthetic_gen import PATTERN_PARAMS, _degrade

    def degrade(clean: np.ndarray, preset: str, seed: int) -> np.ndarray:
        if preset not in PATTERN_PARAMS:
            raise P6TechnicalAbort(f"未知退化 preset {preset!r}（PATTERN_PARAMS 冻结网格）")
        return _degrade(np.asarray(clean, float), PATTERN_PARAMS[preset], int(seed))

    return degrade


# ════════════════════════════ 2. 程序执行 + 判官 ingestion ════════════════════════════
def judge_ingest(x: np.ndarray) -> np.ndarray:
    """判官 ingestion fill（冻结释义 A）：非有限 → NaN → 线性插补（首尾最近值钳制，
    = operators._common.interp_nan 语义）；全非有限 → P6TechnicalAbort。有限输入原样副本。"""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise P6TechnicalAbort("judge_ingest: 空序列")
    finite = np.isfinite(arr)
    if finite.all():
        return arr.copy()
    if not finite.any():
        raise P6TechnicalAbort("judge_ingest: 全非有限序列（无法评估，不得凭空造数据）")
    y = arr.copy()
    y[~finite] = np.nan
    idx = np.arange(y.size)
    m = np.isnan(y)
    y[m] = np.interp(idx[m], idx[~m], y[~m])
    return y


def run_gate_program(
    steps: Sequence[Tuple[str, Mapping[str, Any]]], history: np.ndarray
) -> np.ndarray:
    """执行一个 P_gate 程序（空 steps = raw_identity = 不处理）。执行失败 → abort
    （冻结程序在 C0 数据上失败是基础设施故障）。返回 prepared 序列（未过 ingestion fill）。"""
    x = np.asarray(history, dtype=float).ravel()
    if not steps:
        return x.copy()
    res = run_pipeline(resolve_steps(list(steps)), x)
    if not res.ok or res.artifact is None:
        raise P6TechnicalAbort(f"P_gate 程序执行失败：steps={list(steps)!r} error={res.error!r}")
    return np.asarray(res.artifact, dtype=float).ravel()


def prepared_views_for_program(
    episodes: Sequence[P6Episode], program_id: str
) -> List[SeriesView]:
    """一个域（或任意 episode 集）在指定 P_gate 程序下的判官视图（程序执行 → ingestion fill）。"""
    if program_id not in _PROGRAM_STEPS:
        raise ValueError(f"未知 P_gate 程序 {program_id!r}（冻结清单：{P_GATE_PROGRAM_IDS}）")
    steps = _PROGRAM_STEPS[program_id]
    return [
        SeriesView(
            uid=ep.uid,
            history=judge_ingest(run_gate_program(steps, ep.history)),
            future=ep.future,
        )
        for ep in episodes
    ]


# ════════════════════════════ 3. 双路对拍判官拟合（prereg §1） ════════════════════════════
def _resolve_judge_cfg(judge_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = dict(JUDGE_CFG_FROZEN)
    if judge_cfg is not None:
        unknown = sorted(set(judge_cfg) - _JUDGE_CFG_KEYS)
        if unknown:
            raise ValueError(f"judge_cfg 未知键 {unknown}（可用：{sorted(_JUDGE_CFG_KEYS)}）")
        cfg.update(judge_cfg)
    return cfg


def paired_judge_fit(
    views: Sequence[SeriesView],
    judge_cfg: Optional[Mapping[str, Any]] = None,
    *,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
) -> DomainFit:
    """正式判官拟合 = 双路对拍自检（prereg §1 冻结判据）。

    主判据：per-episode loss 与 batch utility 的 max|Δ| ≤ 1e-9；
    辅助：W 相对差 max|ΔW|/max|W| ≤ 1e-6；任一超限 → P6TechnicalAbort。
    fit_fn / rebuild_fn 可注入（默认 = judge_closed_form.fit_domain / fit_domain_rebuild；
    测试用被篡改的 rebuild 对照构造性触发 abort）。返回主路 DomainFit。
    """
    cfg = _resolve_judge_cfg(judge_cfg)
    fit = (fit_fn or fit_domain)(views, **cfg)
    rb = (rebuild_fn or fit_domain_rebuild)(views, **cfg)
    r_a, r_b = np.asarray(fit.per_series_rmse, float), np.asarray(rb.per_series_rmse, float)
    if r_a.shape != r_b.shape:
        raise P6TechnicalAbort(
            f"对拍失败：per-episode loss 形状不一致 {r_a.shape} vs {r_b.shape}"
        )
    loss_diff = float(np.max(np.abs(r_a - r_b))) if r_a.size else 0.0
    util_diff = float(abs(float(fit.utility) - float(rb.utility)))
    if not (np.isfinite(loss_diff) and np.isfinite(util_diff)):
        raise P6TechnicalAbort("对拍失败：loss/utility 非有限（NaN 塌缩）")
    if loss_diff > DUAL_PATH_LOSS_ATOL or util_diff > DUAL_PATH_LOSS_ATOL:
        raise P6TechnicalAbort(
            f"对拍超限（主判据）：per-episode loss max|Δ|={loss_diff:.3e}，"
            f"utility |Δ|={util_diff:.3e} > {DUAL_PATH_LOSS_ATOL:g} → technical abort"
        )
    w_a, w_b = np.asarray(fit.W, float), np.asarray(rb.W, float)
    w_diff = float(np.max(np.abs(w_a - w_b)))
    w_max = float(np.max(np.abs(w_a)))
    w_rel = (w_diff / w_max) if w_max > 0.0 else (0.0 if w_diff == 0.0 else float("inf"))
    if not np.isfinite(w_rel) or w_rel > DUAL_PATH_W_RTOL:
        raise P6TechnicalAbort(
            f"对拍超限（辅助判据）：max|ΔW|/max|W|={w_rel:.3e} > {DUAL_PATH_W_RTOL:g}"
            " → technical abort"
        )
    return fit


# ════════════════════════════ 4. Adam 参照（真实现工厂；惰性 torch） ════════════════════════════
def _torch_windows_from_views(
    views: Sequence[SeriesView], stride: int, window_cap: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """与判官逐字一致的窗协议（judge_closed_form._prepare 减去 phi）：history-only z-score、
    滑窗 stride、X=(n,48) 原始 z 窗（DLinear 自行分解）、Y=(n,48)；
    eval_inputs=(m,48) 末窗、futures_norm=(m,48)。"""
    from .judge_closed_form import CONTEXT_LEN, window_starts, zscore_state

    total = CONTEXT_LEN + HORIZON
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    evals: List[np.ndarray] = []
    futs: List[np.ndarray] = []
    for v in views:
        h = np.asarray(v.history, float).ravel()
        f = np.asarray(v.future, float).ravel()
        mean, std = zscore_state(h)
        hn = (h - mean) / std
        for t in window_starts(h.size, stride=stride, window_cap=window_cap):
            xs.append(hn[t: t + CONTEXT_LEN])
            ys.append(hn[t + CONTEXT_LEN: t + total])
        evals.append(hn[-CONTEXT_LEN:])
        futs.append((f[:HORIZON] - mean) / std)
    if not xs:
        raise P6TechnicalAbort("Adam 参照：域内没有任何训练窗口")
    return (np.asarray(xs, float), np.asarray(ys, float),
            np.asarray(evals, float), np.asarray(futs, float))


def make_torch_forecaster_trainer(
    model: str = "dlinear",
    *,
    epochs: int = ADAM_EPOCHS,
    lr: float = ADAM_LR,
    bs: int = ADAM_BS,
    hidden: int = 64,
    stride: Optional[int] = None,
    window_cap: Optional[int] = None,
) -> Callable[[Sequence[SeriesView], int], np.ndarray]:
    """真实 torch trainer 工厂（惰性 import torch；prereg §1 协议冻结）。

    trainer(views, seed) → per-episode losses（判官同协议 z 空间 RMSE，np.ndarray）。
    - model="dlinear"（Adam co-gate/参照）| "lstm"（U roster 的 LSTM-scratch，hidden=64）；
    - **CPU**（临时置 _torch_models.DEVICE="cpu"，finally 还原）、每 fit 前 seed_all(seed)、
      torch.use_deterministic_algorithms(True)；
    - 训练窗 pooled（冻结释义 B）；stride 默认取判官冻结 stride（=4）；
    - NaN/发散 → P6TechnicalAbort（超时检查在 runner 侧，见 run_c0/run_u_eval）。
    """
    if model not in ("dlinear", "lstm"):
        raise ValueError(f"model 须为 'dlinear' 或 'lstm'，got {model!r}")
    eff_stride = int(JUDGE_CFG_FROZEN["stride"] if stride is None else stride)

    def trainer(views: Sequence[SeriesView], seed: int) -> np.ndarray:
        import torch

        from ..evaluators import _torch_models as tm
        from .judge_closed_form import CONTEXT_LEN

        X, Y, evals, futs = _torch_windows_from_views(views, eff_stride, window_cap)
        old_device = tm.DEVICE
        tm.DEVICE = "cpu"                       # prereg §1：正式 runner 全 CPU
        try:
            torch.use_deterministic_algorithms(True)
            tm.seed_all(int(seed))              # 每次拟合前重播种（prereg §1）
            if model == "dlinear":
                net = tm.DLinear(CONTEXT_LEN, HORIZON)
            else:
                net = tm.LSTMForecaster(CONTEXT_LEN, HORIZON, hidden=int(hidden))
            tm.train_forecaster(net, X, Y, epochs=int(epochs), lr=float(lr), bs=int(bs))
            preds = tm.forecast_predict(net, evals)
        finally:
            tm.DEVICE = old_device
        err = np.asarray(preds, float) - futs
        losses = np.sqrt(np.mean(err * err, axis=1))
        if not np.all(np.isfinite(losses)):
            raise P6TechnicalAbort(f"{model} 拟合产出 NaN/发散 per-episode loss → technical abort")
        return losses

    return trainer


def make_adam_trainer(**kwargs: Any) -> Callable[[Sequence[SeriesView], int], np.ndarray]:
    """Adam-DLinear 参照 trainer（prereg §1：DLinear、epochs=120、lr=1e-2、bs=256、CPU、
    seed_all、deterministic）。= make_torch_forecaster_trainer("dlinear", **kwargs)。"""
    return make_torch_forecaster_trainer("dlinear", **kwargs)


def _checked_trainer_call(
    trainer: Callable[[Sequence[SeriesView], int], Any],
    views: Sequence[SeriesView],
    seed: int,
    timeout_seconds: float,
    label: str,
) -> np.ndarray:
    """runner 侧统一护栏：软超时（事后墙钟，冻结释义 G）+ 形状 + NaN/发散 abort。"""
    t0 = time.perf_counter()
    losses = np.asarray(trainer(views, int(seed)), dtype=float).ravel()
    elapsed = time.perf_counter() - t0
    if elapsed > float(timeout_seconds):
        raise P6TechnicalAbort(
            f"{label} 单 fit 超时：{elapsed:.1f}s > {float(timeout_seconds):g}s → technical abort"
        )
    if losses.shape != (len(views),):
        raise ValueError(
            f"{label} trainer 返回形状 {losses.shape} != ({len(views)},)（per-episode loss 契约）"
        )
    if not np.all(np.isfinite(losses)):
        raise P6TechnicalAbort(f"{label} 拟合产出 NaN/发散 → technical abort")
    return losses


# ════════════════════════════ 5. identity gate（prereg §3.1） ════════════════════════════
def _avg_ranks(v: np.ndarray) -> np.ndarray:
    """均秩（ties 取平均；1-based），确定性。"""
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=float)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman ρ = 均秩上的 Pearson；退化（任一侧全并列 → 分母 0）→ 0.0（冻结释义 E）。"""
    va, vb = np.asarray(a, float), np.asarray(b, float)
    if va.shape != vb.shape or va.ndim != 1 or va.size < 2:
        raise ValueError(f"spearman_rho 需要两条等长 1-D（≥2）序列，got {va.shape} vs {vb.shape}")
    if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
        raise ValueError("spearman_rho 输入含非有限值")
    ra = _avg_ranks(va)
    rb = _avg_ranks(vb)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(ra * rb) / denom)


def _check_loss_table(
    table: Mapping[str, Mapping[str, Sequence[float]]],
    name: str,
    programs: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for d in sorted(table):
        row: Dict[str, np.ndarray] = {}
        n = None
        for g in programs:
            if g not in table[d]:
                raise ValueError(f"{name}[{d!r}] 缺程序 {g!r}")
            v = np.asarray(table[d][g], dtype=float).ravel()
            if v.size == 0 or not np.all(np.isfinite(v)):
                raise ValueError(f"{name}[{d!r}][{g!r}] 为空或含非有限值")
            if n is None:
                n = v.size
            elif v.size != n:
                raise ValueError(f"{name}[{d!r}] 各程序 episode 数不一致")
            row[g] = v
        out[str(d)] = row
    return out


def evaluate_identity_gate(
    cf_losses: Mapping[str, Mapping[str, Sequence[float]]],
    adam_losses: Mapping[str, Mapping[str, Sequence[float]]],
    presets_by_domain: Mapping[str, Sequence[str]],
    eps: float,
    *,
    programs: Sequence[str] = P_GATE_PROGRAM_IDS,
    raw_id: str = RAW_PROGRAM_ID,
) -> Dict[str, Any]:
    """identity gate（prereg §3.1 精确定义）纯函数：输入 per-episode loss 表，输出四判据 + PASS。

    cf_losses / adam_losses：{domain: {program: per-episode losses}}（adam 已 seed 均值）；
    presets_by_domain：{domain: 逐 episode preset 标签}；eps：§3.2 的 ε（先算 ε 再进本函数）。
    判据（PASS = ① ∧ ② ∧ ④ ∧ ③′）：
      ① per-domain raw-level **non-inferiority**（Amendment A1，见 prereg §11；原双侧
         equivalence 门 |U_cf−U_adam|≤0.10·U_adam 已改）：U_cf(raw) ≤ 1.10·U_adam(raw)
         ——闭式判官相对 Adam 参照不显著更差即过；不再确立两估计器绝对水平等价；
      ② per-domain Spearman ρ（8 程序 gain vs raw）的域中位数 ≥ 0.7；
      ③′ episode 级 ε-tolerant top-1 一致率 ≥ 0.6（Adam top-1 ∈ 闭式 top-2 且
         cf_loss(adam_top1) − cf_loss(cf_top1) ≤ ε/2；冻结释义 D）；
      ④ preset 级 gain 符号一致率 ≥ 0.75（near-zero 带 |gain| < ε/4 任一侧落带即出分母，
         带为严格 <；分母 0 → FAIL；冻结释义 C）。
    """
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps 必须是正有限数（先算 §3.2 的 ε），got {eps!r}")
    programs = list(programs)
    if raw_id not in programs:
        raise ValueError(f"raw 程序 {raw_id!r} 必须在 programs 中")
    cf = _check_loss_table(cf_losses, "cf_losses", programs)
    ad = _check_loss_table(adam_losses, "adam_losses", programs)
    if sorted(cf) != sorted(ad):
        raise ValueError(f"cf/adam 域集合不一致：{sorted(cf)} vs {sorted(ad)}")
    domains = sorted(cf)
    presets: Dict[str, List[str]] = {}
    for d in domains:
        if d not in presets_by_domain:
            raise ValueError(f"presets_by_domain 缺域 {d!r}")
        labels = [str(p) for p in presets_by_domain[d]]
        if len(labels) != cf[d][raw_id].size:
            raise ValueError(f"presets_by_domain[{d!r}] 长度与 episode 数不一致")
        presets[d] = labels

    # ① raw 水准 non-inferiority（Amendment A1；见 prereg §11）：闭式判官相对 Adam 参照
    # 不显著更差即过——U_cf ≤ (1+0.10)·U_adam。原双侧 equivalence 门（|U_cf−U_adam|≤tol）
    # 已改为一侧；闭式 loss 更低（更优）恒过，仅当闭式比 Adam 更差 >10% 才 FAIL（门未整体
    # 削弱，见 test_gate_criterion1_noninferiority_a1）。签名口径同 diagnostics/D10。
    crit1_per: Dict[str, Any] = {}
    for d in domains:
        u_cf = float(np.mean(cf[d][raw_id]))
        u_ad = float(np.mean(ad[d][raw_id]))
        upper = (1.0 + GATE1_REL_TOL) * u_ad
        signed = u_cf - u_ad
        noninf = bool(u_cf <= upper)
        crit1_per[d] = {
            "u_cf_raw": u_cf, "u_adam_raw": u_ad,
            "signed_offset": signed,                                 # U_cf − U_adam（负=闭式更优）
            "signed_relative_offset": (float(signed / u_ad) if u_ad != 0.0 else float("inf")),
            "upper_bound": upper, "rel_tol": GATE1_REL_TOL,
            "upper_noninferiority_pass": noninf, "pass": noninf,
            "criterion_semantics": "closed_form_not_worse_than_adam_by_more_than_10pct",
        }
    crit1_pass = all(v["pass"] for v in crit1_per.values())

    # ② per-domain Spearman（8 程序 gain vs raw；raw 自身 gain=0 两侧同入秩）
    crit2_rho: Dict[str, float] = {}
    for d in domains:
        g_cf = [gain(float(np.mean(cf[d][raw_id])), float(np.mean(cf[d][g]))) for g in programs]
        g_ad = [gain(float(np.mean(ad[d][raw_id])), float(np.mean(ad[d][g]))) for g in programs]
        crit2_rho[d] = spearman_rho(g_cf, g_ad)
    median_rho = float(np.median(list(crit2_rho.values())))
    crit2_pass = bool(median_rho >= GATE2_MEDIAN_RHO_MIN)

    # ③′ episode 级 ε-tolerant top-1 一致率（冻结释义 D）
    n_episodes = 0
    n_agree = 0
    half_eps = eps / 2.0
    for d in domains:
        n_d = cf[d][raw_id].size
        for i in range(n_d):
            cf_vec = np.array([cf[d][g][i] for g in programs], dtype=float)
            ad_vec = np.array([ad[d][g][i] for g in programs], dtype=float)
            adam_top1 = int(np.argmin(ad_vec))          # 并列取冻结程序次序靠前（argmin 首个）
            order_cf = np.argsort(cf_vec, kind="stable")
            top2_cf = {int(order_cf[0]), int(order_cf[1])} if order_cf.size >= 2 else {int(order_cf[0])}
            loss_gap = float(cf_vec[adam_top1] - cf_vec[int(order_cf[0])])
            n_episodes += 1
            if adam_top1 in top2_cf and loss_gap <= half_eps:
                n_agree += 1
    top1_rate = n_agree / n_episodes if n_episodes else 0.0
    crit3_pass = bool(n_episodes > 0 and top1_rate >= GATE3_TOP1_RATE_MIN)

    # ④ preset 级 gain 符号一致率（near-zero 带外；冻结释义 C）
    all_presets = sorted({p for d in domains for p in presets[d]})
    band = eps / 4.0
    pairs_total = 0
    pairs_excluded = 0
    pairs_agree = 0
    preset_gains: Dict[str, Dict[str, Dict[str, float]]] = {}
    for p in all_presets:
        preset_gains[p] = {}
        # preset batch loss = 该 preset 全 episode（跨域 pooled）等权均值
        def _preset_loss(table: Dict[str, Dict[str, np.ndarray]], g: str) -> float:
            vals: List[float] = []
            for d in domains:
                mask = [lab == p for lab in presets[d]]
                vals.extend(table[d][g][mask].tolist())
            if not vals:
                raise ValueError(f"preset {p!r} 无 episode")
            return float(np.mean(vals))

        raw_cf_p = _preset_loss(cf, raw_id)
        raw_ad_p = _preset_loss(ad, raw_id)
        for g in programs:
            g_cf_p = gain(raw_cf_p, _preset_loss(cf, g))
            g_ad_p = gain(raw_ad_p, _preset_loss(ad, g))
            preset_gains[p][g] = {"gain_cf": g_cf_p, "gain_adam": g_ad_p}
            pairs_total += 1
            if abs(g_cf_p) < band or abs(g_ad_p) < band:     # 严格 <：恰 = ε/4 保留
                pairs_excluded += 1
                continue
            if (g_cf_p > 0.0) == (g_ad_p > 0.0):
                pairs_agree += 1
    n_eval = pairs_total - pairs_excluded
    sign_rate = (pairs_agree / n_eval) if n_eval > 0 else None
    crit4_pass = bool(n_eval > 0 and sign_rate >= GATE4_SIGN_RATE_MIN)

    gate_pass = bool(crit1_pass and crit2_pass and crit4_pass and crit3_pass)
    return {
        "epsilon": eps,
        "criterion1_raw_level": {"per_domain": crit1_per, "rel_tol": GATE1_REL_TOL,
                                 "criterion_semantics": "raw_level_non_inferiority_A1",
                                 "pass": crit1_pass},
        "criterion2_spearman": {"per_domain_rho": crit2_rho, "median_rho": median_rho,
                                "threshold": GATE2_MEDIAN_RHO_MIN, "pass": crit2_pass},
        "criterion3_prime_top1": {"n_episodes": n_episodes, "n_agree": n_agree,
                                  "rate": top1_rate, "loss_tol": half_eps,
                                  "threshold": GATE3_TOP1_RATE_MIN, "pass": crit3_pass},
        "criterion4_preset_sign": {"n_pairs_total": pairs_total,
                                   "n_excluded_near_zero": pairs_excluded,
                                   "n_evaluated": n_eval, "n_agree": pairs_agree,
                                   "rate": sign_rate, "near_zero_band": band,
                                   "threshold": GATE4_SIGN_RATE_MIN, "pass": crit4_pass,
                                   "preset_gains": preset_gains},
        "pass": gate_pass,
    }


# ════════════════════════════ 6. ε/δ 与 P0 cutpoints（prereg §3.2/§3.3） ════════════════════════════
def compute_cutpoints(
    episodes: Sequence[P6Episode],
    fingerprint_fn: Callable[[np.ndarray], Mapping[str, Any]],
) -> Dict[str, Any]:
    """P0 bin cutpoints：snr 与 missing_rate 的 C0 四分位（np.quantile linear 插值）。

    fingerprint_fn 注入：history → 至少含 {"snr", "missing_rate"}；缺键 → ValueError，
    非有限 → P6TechnicalAbort（数据/提取器故障）。
    """
    snr_vals: List[float] = []
    miss_vals: List[float] = []
    for ep in episodes:
        fp = fingerprint_fn(ep.history)
        for key in ("snr", "missing_rate"):
            if key not in fp:
                raise ValueError(f"fingerprint 缺键 {key!r}（episode {ep.uid}）")
        s, m = float(fp["snr"]), float(fp["missing_rate"])
        if not (np.isfinite(s) and np.isfinite(m)):
            raise P6TechnicalAbort(f"fingerprint 非有限（episode {ep.uid}）→ technical abort")
        snr_vals.append(s)
        miss_vals.append(m)
    q = list(CUTPOINT_QUANTILES)
    return {
        "quantiles": q,
        "n_episodes": len(episodes),
        "snr": [float(v) for v in np.quantile(snr_vals, q, method="linear")],
        "missing_rate": [float(v) for v in np.quantile(miss_vals, q, method="linear")],
    }


# ════════════════════════════ 7. C0 orchestrator + C0_FREEZE ════════════════════════════
def _group_by_domain(episodes: Sequence[P6Episode]) -> Dict[str, List[P6Episode]]:
    uids = [ep.uid for ep in episodes]
    if len(set(uids)) != len(uids):
        raise ValueError("episodes uid 重复（判官/台账都按 uid 对齐）")
    by_dom: Dict[str, List[P6Episode]] = {}
    for ep in episodes:
        by_dom.setdefault(ep.config, []).append(ep)
    return {d: by_dom[d] for d in sorted(by_dom)}


def run_c0_unfrozen(
    episodes: Sequence[P6Episode],
    adam_trainer: Callable[[Sequence[SeriesView], int], Any],
    fingerprint_fn: Callable[[np.ndarray], Mapping[str, Any]],
    *,
    judge_cfg: Optional[Mapping[str, Any]] = None,
    seeds: Sequence[int] = C0_SEEDS,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
    out_path: Optional[Any] = None,
    adam_fit_timeout_seconds: float = ADAM_FIT_TIMEOUT_SECONDS,
    _entrypoint: Optional[str] = None,
    _frozen_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """C0 校准主流程核心（**测试专用；正式运行禁止调用——用 run_c0_formal**）。

    返回 C0_FREEZE record dict；gate FAIL → （若给 out_path 先写盘）raise P6TechnicalStop。
    对拍超限 / Adam NaN / 超时 / 数据故障 → P6TechnicalAbort。冻结字面量断言不在此，在
    run_c0_formal（G4/finding 36）；`_entrypoint`/`_frozen_digest` 由正式入口注入以入账。
    """
    t_start = time.perf_counter()
    episodes = list(episodes)
    if not episodes:
        raise ValueError("episodes 不能为空")
    if not seeds:
        raise ValueError("seeds 不能为空")
    cfg = _resolve_judge_cfg(judge_cfg)
    by_dom = _group_by_domain(episodes)
    domains = list(by_dom)

    # —— 闭式判官：per-domain × per-program（每次正式拟合都过双路对拍） ——
    views_cache: Dict[Tuple[str, str], List[SeriesView]] = {}
    cf_losses: Dict[str, Dict[str, np.ndarray]] = {d: {} for d in domains}
    cf_utils: Dict[str, Dict[str, float]] = {d: {} for d in domains}
    n_cf_fits = 0
    for d in domains:
        for g in P_GATE_PROGRAM_IDS:
            views = prepared_views_for_program(by_dom[d], g)
            fit = paired_judge_fit(views, cfg, fit_fn=fit_fn, rebuild_fn=rebuild_fn)
            n_cf_fits += 1
            views_cache[(d, g)] = views
            cf_losses[d][g] = np.asarray(fit.per_series_rmse, float).copy()
            cf_utils[d][g] = float(fit.utility)

    # —— §3.2：先算 ε（gate 的 near-zero 带 / top-1 容差依赖它） ——
    j_raw = float(np.mean([cf_utils[d][RAW_PROGRAM_ID] for d in domains]))   # 域等权再合并
    if not np.isfinite(j_raw) or j_raw <= 0.0:
        raise P6TechnicalAbort(f"J_raw_C0={j_raw!r} 非正有限——判官 raw 臂塌缩 → technical abort")
    epsilon = EPSILON_RATE * j_raw
    delta_safe = DELTA_SAFE_MULTIPLIER * epsilon

    # —— Adam 参照：per-domain × per-program × per-seed（runner 侧软超时 + NaN abort） ——
    adam_mean: Dict[str, Dict[str, np.ndarray]] = {d: {} for d in domains}
    adam_utils: Dict[str, Dict[str, float]] = {d: {} for d in domains}
    n_adam_fits = 0
    for d in domains:
        for g in P_GATE_PROGRAM_IDS:
            per_seed: List[np.ndarray] = []
            for s in seeds:
                losses = _checked_trainer_call(
                    adam_trainer, views_cache[(d, g)], int(s),
                    adam_fit_timeout_seconds, f"Adam[{d}/{g}/seed={s}]",
                )
                per_seed.append(losses)
                n_adam_fits += 1
            adam_mean[d][g] = np.mean(np.stack(per_seed, axis=0), axis=0)
            adam_utils[d][g] = float(np.mean(adam_mean[d][g]))

    # —— identity gate（ε 已定） ——
    presets_by_domain = {d: [ep.preset for ep in by_dom[d]] for d in domains}
    gate = evaluate_identity_gate(cf_losses, adam_mean, presets_by_domain, epsilon)

    # —— P0 cutpoints（§3.3；fingerprint 注入） ——
    cutpoints = compute_cutpoints(episodes, fingerprint_fn)

    record: Dict[str, Any] = {
        "schema_version": "p6-c0-freeze/1",
        "protocol": {
            "judge": {"protocol_id": PROTOCOL_ID, **cfg},
            "dual_path_check": {"loss_atol": DUAL_PATH_LOSS_ATOL, "w_rtol": DUAL_PATH_W_RTOL},
            "p_gate_programs": {g: [[op, dict(p)] for op, p in steps]
                                for g, steps in P_GATE_PROGRAMS},
            "raw_program": RAW_PROGRAM_ID,
            "seeds": [int(s) for s in seeds],
            "epsilon_rate": EPSILON_RATE,
            "delta_safe_multiplier": DELTA_SAFE_MULTIPLIER,
            "degradation_seed_rule":
                "int(sha256('p6deg|{config}|{item}|{preset}').hexdigest()[:8], 16)",
            "judge_ingest": "non-finite→NaN→linear interp（首尾钳制）；全非有限→abort",
            "adam": {"model": "dlinear", "epochs": ADAM_EPOCHS, "lr": ADAM_LR, "bs": ADAM_BS,
                     "device": "cpu", "deterministic": True,
                     "fit_timeout_seconds": float(adam_fit_timeout_seconds),
                     "windows": "judge-identical (history-only z-score, stride=judge, pooled)"},
        },
        "domains": domains,
        "n_episodes": len(episodes),
        "episodes_per_domain": {d: len(by_dom[d]) for d in domains},
        "presets_observed": sorted({ep.preset for ep in episodes}),
        "j_raw_c0": j_raw,
        "epsilon": epsilon,
        "delta_safe": delta_safe,
        "identity_gate": gate,
        "per_domain_utilities": {
            "closed_form": {d: dict(cf_utils[d]) for d in domains},
            "adam_mean_seed": {d: dict(adam_utils[d]) for d in domains},
        },
        "p0_cutpoints": cutpoints,
        "provenance": {                             # G4：正式入口证据（entrypoint + 冻结字面量 digest）
            "entrypoint": _entrypoint or "run_c0_unfrozen",
            "frozen_literals_digest": _frozen_digest,
        },
        "costs": {
            "closed_form_fits": n_cf_fits,          # 正式拟合数（每次内部另跑 1 次 rebuild 对拍）
            "adam_fits": n_adam_fits,
            "wall_clock_seconds": round(time.perf_counter() - t_start, 3),
        },
    }
    if out_path is not None:
        write_c0_freeze(record, out_path)
    if not gate["pass"]:
        failed = [k for k in ("criterion1_raw_level", "criterion2_spearman",
                              "criterion3_prime_top1", "criterion4_preset_sign")
                  if not gate[k]["pass"]]
        raise P6TechnicalStop(
            f"C0 identity gate FAIL（{failed}）→ technical stop：判官身份未确立，"
            "D/V/U 不开启、本 freeze 作废（prereg §3）",
            record=record,
        )
    return record


def c0_frozen_literals(seeds: Sequence[int], adam_fit_timeout_seconds: float) -> Dict[str, Any]:
    """run_c0_formal 冻结字面量集合（用于 provenance digest）。"""
    return {
        "seeds": [int(s) for s in seeds],
        "n_episodes": FROZEN_C0_N_EPISODES, "n_domains": FROZEN_C0_N_DOMAINS,
        "presets_per_series": FROZEN_C0_PRESETS_PER_SERIES, "adam_fits": FROZEN_C0_ADAM_FITS,
        "adam_fit_timeout_seconds": float(adam_fit_timeout_seconds),
    }


def run_c0_formal(
    episodes: Sequence[P6Episode],
    adam_trainer: Callable[[Sequence[SeriesView], int], Any],
    fingerprint_fn: Callable[[np.ndarray], Mapping[str, Any]],
    *,
    judge_cfg: Optional[Mapping[str, Any]] = None,
    seeds: Sequence[int] = C0_SEEDS,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
    out_path: Optional[Any] = None,
    adam_fit_timeout_seconds: float = ADAM_FIT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """**C0 唯一合法正式入口**（G4/finding 36）：先机械断言全部 prereg §3 冻结字面量
    （seeds/64ep/4域/4preset/96fits/timeout=900），把 entrypoint + 冻结字面量 digest 写入
    C0_FREEZE record（入 record_sha），再委托 run_c0_unfrozen。任一漂移 → P6FrozenParamError。"""
    assert_c0_frozen_params(episodes, seeds, adam_fit_timeout_seconds)
    digest = frozen_literals_digest(
        "run_c0_formal", c0_frozen_literals(seeds, adam_fit_timeout_seconds)
    )
    return run_c0_unfrozen(
        episodes, adam_trainer, fingerprint_fn, judge_cfg=judge_cfg, seeds=seeds,
        fit_fn=fit_fn, rebuild_fn=rebuild_fn, out_path=out_path,
        adam_fit_timeout_seconds=adam_fit_timeout_seconds,
        _entrypoint="run_c0_formal", _frozen_digest=digest,
    )


def write_c0_freeze(record: Mapping[str, Any], path) -> str:
    """写 C0_FREEZE JSON（含内嵌 record_sha = canonical JSON sha256），返回 sha。"""
    doc = dict(record)
    doc.pop("record_sha", None)
    canonical = json.dumps(doc, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                           default=float)
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    doc["record_sha"] = sha
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return sha
