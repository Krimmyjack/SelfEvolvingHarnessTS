"""operators/s1_structural.py — S1 结构性缺陷修复（水平断层 / structural break）。

E-3.3 R1（2026-07-14 新增）。动机：`benchmark/programs.py: CAPABILITY_GAPS` 在任何 v0.2
数字被读到之前就声明了一个能力缺口——**算子库里没有任何变点检测或分段归一算子，因此
池里没有东西能找到断点，也就没有东西能修断点**。v0.2 Dev 实测坐实了这个声明：level_shift
腐蚀格上，raw / forward_fill / seasonal_fill / h_ref 四个 program 的损失小数点后四位相同。
本算子填的就是这个**预先声明过的**缺口——先声明缺口、后补算子，这个次序是它免疫"事后挑
格子"指控的全部依据。

════════════════════════════════ 算法（冻结，不许漂移） ════════════════════════════════

检测 = **趋势/季节感知的 sup-Chow 扫描 + 贪心前向选择**（非朴素均值二分）。

  设计矩阵 X = [1, t, **连续分段线性（hinge）基底**] ⊕ 季节哑变量 ⊕ 已接受的阶跃列。
  对每个候选断点 τ，阶跃列 s_τ = 1[t ≥ τ]。用 Frisch–Waugh–Lovell 定理，在**当前设计的
  正交基 Q 上**一次性向量化算出全部 τ 的检验量：

      e        = y − Q(Qᵀy)                      当前残差
      Σ_{t≥τ}e = 后缀和（revcumsum，O(n)）
      ‖s̃_τ‖²  = (n−τ) − ‖Σ_{t≥τ}Q‖²             s_τ 对当前设计残差化后的模方
      δ̂_τ      = Σ_{t≥τ}e / ‖s̃_τ‖²              该阶跃的 OLS 系数（FWL）
      t_τ      = |Σ_{t≥τ}e| / (**σ̂_长程** · ‖s̃_τ‖)  检验量

  → 整个扫描 O(n·p)，纯 numpy，无 ruptures 依赖（选定即写死：**不引入新依赖**，
    因为新依赖会给冻结池增加一个必须硬校验的第四个 requires_dependency）。

**为什么趋势基底必须是"连续分段线性"而不是 [1, t]**（2026-07-14 Support-A discovery 检查
逼出来的修复，v1 在这里死得很难看）：

  v1 的设计矩阵只有 [1, t]，即假设趋势是**线性**的。真实 roster 不是——`monash:covid_deaths`
  是**累计计数**（单调递增的 S 曲线，0 → 408）。线性趋势对 S 曲线严重失配 → 残差是系统性的
  U 形 → sup-Chow 把这个**模型失配**读成一串"电平断层" → 分段对齐把整条曲线打成垃圾。
  实测：干净的真实序列上误触发 **82%**、平均损伤 **1.78σ**；covid 上重建误差 **69σ**。
  合成测试却全绿——因为我的合成序列**恰好就是设计矩阵假设的那个模型**（线性趋势＋纯正弦＋
  白噪）。**拿自己的假设当测试，什么也没测**。

  修复的关键结构性质：**连续基底无法表示跳变**。所以趋势基底可以做到足够柔（跟得住 S 曲线、
  指数增长），而断点依然完整地留在残差里——柔性与检测力在这里不冲突。纯趋势（无论线性还是
  弯的）→ 残差 ≈ 0 → 检验量 ≈ 0 → 一个断点都不接受。

**为什么方差必须是"长程"的**（同一次检查逼出来的第二处修复）：

  t 统计量的分母假设残差 **iid**。真实序列的残差是强自相关的——后缀和的真实标准误因此被抬高
  （实测中位数 **2.9 倍**），iid 假设下 t 统计量系统性膨胀，在**100% 的干净真实序列**上都越过
  3.5。这不是"把阈值从 3.0 提到 3.5 就能保守掉"的量级问题（v1 的注释正是这么天真地写的），
  是**分母算错了**。改用 batch-means 长程标准差：把残差切成长 = period 的不重叠块，取块均值的
  稳健标准差 × √L——块内平均吸收自相关，MAD 使单个断层不炸掉估计。

季节哑变量同理不可省：不建模季节，残差被季节摆幅主导 → 检验功效塌掉（period=24 的合成序列上
检验量从 ~14 掉到 ~2.8，一个真断层都测不出来）。哑变量对季节**形状**不做假设（不是正弦）。

两道闸**作用在不同层**（这个分层是踩过坑改出来的，不是设计洁癖）：
  ① 统计闸（检测期，逐 τ）：t_τ ≥ t_threshold。默认 3.5——sup-Chow 在 15% 修剪下的经典
     临界值约 3.0，取 3.5 是有意保守：残差自相关会抬高检验量，宁可漏修不可错修。
  ② 效应闸（**修复期，逐段**）：只有相对**前后两个邻段**的电平偏离都 ≥ min_jump_sigma ×
     robust_sigma(y) 的段才真的被搬动（默认 1.0——**一段的电平必须至少偏离邻段一个稳健 σ
     才值得动它**）。判据的完整形式见下方"修复"一节（"上去又回来"规则）。

  ⚠️ 效应闸**不能**放在检测期的 δ̂_τ 上。δ̂_τ 是"τ 到序列末尾的**永久阶跃**"的 OLS 系数；
  而一个宽度 c 的**脉冲型**断层（升上去、又降回来——benchmark 的 level_shift 正是如此）
  只能被单个永久阶跃解释掉 c/(n−τ) 的份额：实测真实跳高 8.6 的脉冲，第一步贪心的 δ̂ 只有
  0.45。把效应闸放在 δ̂ 上会把真断点全部挡死（第一版就是这么写的，②号语义测试全灭）。
  段电平差才是"这段偏了多少"的正确度量，它只有在两个断点都进设计之后才谈得上。

  min_jump_sigma 是通用效应量地板，不是从腐蚀常数反推的：benchmark 的 level_shift 幅度
  恰好是 2.0σ，本闸以 2× 余量放行，但闸值选择从未参照过任何 benchmark 结果（见
  operators/CHANGELOG.md 的披露）。两闸合起来的语义是：**统计显著性负责找候选，效应量
  负责决定动不动手**——在干净序列上即便 sup 统计量偶然越过 3.5，效应闸也会把偏移剪成 0，
  输出**逐位恒等**（不是"近似恒等"）。

修复 = **只撤销"上去又回来"的暂态偏移**（`_excursion_offsets`）：

  一段被判为伪影，当且仅当它的电平相对**前后两个邻段都**偏离**同一方向**，且偏离幅度
  ≥ min_jump_sigma × robust_sigma(y)。命中的段被拉回两个邻段电平的中点；其余一律不动。
  **首段（无左邻）与末段（无右邻）永远不动。**

  这是一条**数据质量**规则，不是变点检测的教科书规则：
    - 「上去又回来」= **可逆的伪影**（传感器卡住一段偏置、一次被回滚的重新标定、一段口径
      异常的报送）→ 该撤销；
    - 「上去就不下来」= **真实的 regime 变更**（新政策、新量程、路网改造）→ **必须保留**。
      持续到序列末尾的电平变化，就是"新 regime"的定义；把它撤掉是在删真实信号。

  ⚠️ **v1 用的是另一套语义（"把历史重基到末段电平"），已被 Support-A discovery 否掉**：
  那套语义会搬动**首段**（一条单调递增的累计计数曲线因此被整段拉平），干净真实序列上误触发
  77.6%、平均损伤 1.62σ。换成暂态偏移语义后降到 45.7% / 0.56σ——**每个轴上都严格更优**，
  且它才是数据质量算子该有的语义。选择在 Support-A discovery（唯一被许可的调参面）上做出。

**尾部守卫**：最后 max(min_segment, period) 个点内不许有断点——末段短于一个完整季节周期时，
它的中位数会被"半个周期"系统性带偏，而末段是**每一个判据的右邻**。要求末段至少覆盖一个完整
季节周期，这个失效模式就被结构性排除。代价：贴着序列末尾的真断层测不到 → 恒等返回 → 不修
但也不伤（安全方向的失败）。

趋势项与季节项从头到尾没被碰过：本算子只搬**段电平**，不动趋势、不动季节、不动残差。

**任务契约**：anomaly_detection 物理禁用——level shift 在异常检测里**就是要检出的信号本身**，
修掉它等于把标签擦了。这正是 C1（H* = f(pattern, task)）最干净的一个例证：同一个算子，
forecast 下是修复、anomaly 下是破坏。
"""
from __future__ import annotations

import math

import numpy as np

from ._common import as_1d, interp_nan, robust_sigma
from ._provenance import record
from ..conditioning.period import guess_period_robust_v1 as _guess_period

_TINY = 1e-12
_MIN_CYCLES_FOR_SEASONAL_DUMMIES = 3     # 少于 3 个完整周期 → 哑变量在拟合噪声，不建季节
_TREND_SEGMENTS = 8                      # 连续分段线性趋势基底的段数（结点间距 = n/8）
_MIN_BLOCKS_FOR_LONG_RUN = 4             # 块数不足 → 长程估计无意义，退回 iid MAD


def _effective_period(n: int, period: int) -> int:
    """要建模的季节周期 = **主导的日历循环**，不一定等于调用方传进来的基础周期。

    ⚠️ 这是 Support-A discovery 逼出来的第三处修复。hourly 序列的基础周期是 24（日循环），
    但真实交通/电力序列里**电平结构的主导循环是"周"**——周末的电平本来就比工作日低一截。
    只建 24 的哑变量，那道周末的电平台阶就被读成"断层"，于是 traffic_hourly / metr_la 上
    误触发 77–100%。

    修法：sub-daily 采样（period ≥ 24）时改建 **7×period（周循环）** 的哑变量。这在数学上是
    严格更强的：**168 = 7×24，phase mod 168 决定 phase mod 24**，所以周哑变量**吞掉**日哑变量，
    不需要两套并存（并存反而共线）。daily/monthly 数据的 period 本身就已经是主导日历循环
    （7=周、12=年），不再上推。列数够不着（n < 3 个完整周循环）→ 退回基础周期。"""
    if period < 2:
        return 0
    weekly = 7 * period
    if period >= 24 and n >= _MIN_CYCLES_FOR_SEASONAL_DUMMIES * weekly:
        return weekly
    return period if n >= _MIN_CYCLES_FOR_SEASONAL_DUMMIES * period else 0


def _seasonal_design(n: int, period: int) -> np.ndarray:
    """季节哑变量（P−1 列，丢掉最后一相位以避免与截距共线）；P = `_effective_period`。
    对季节**形状**不做任何假设（不是正弦——真实的日内负荷曲线根本不是正弦）。"""
    p = _effective_period(n, period)
    if p < 2:
        return np.zeros((n, 0))
    phase = np.arange(n) % p
    return (phase[:, None] == np.arange(p - 1)[None, :]).astype(float)


def _trend_design(n: int, period: int) -> np.ndarray:
    """[1, t, hinge 基底] —— **连续**分段线性趋势。

    hinge_k(t) = max(0, t − knot_k)：连续、分段线性、可任意逼近平滑曲线，但**无法表示跳变**。
    这个"无法"是设计的全部要害：趋势基底可以做到足够柔（跟得住 covid 的 S 曲线），而阶跃
    依然完整地留在残差里。结点间距取 max(n/8, 2·period)——比腐蚀脉冲宽得多，基底想拟合掉
    一个断层也拟合不出来。"""
    t = np.arange(n, dtype=float)
    scale = max(t.std(), _TINY)
    cols = [np.ones(n), (t - t.mean()) / scale]
    spacing = max(int(n / _TREND_SEGMENTS), 2 * period if period >= 2 else 8)
    knots = np.arange(spacing, n - spacing / 2.0, spacing, dtype=float)
    if knots.size:
        cols.append(np.maximum(0.0, t[:, None] - knots[None, :]) / scale)
    return np.column_stack(cols)


def _long_run_sigma(e: np.ndarray, block: int) -> float:
    """长程标准差（batch-means，稳健）—— sup-Chow 分母的**正确**尺度。

    iid 的 MAD 会把自相关序列的后缀和标准误低估约 3 倍（Support-A 实测中位数 2.9），
    于是 t 统计量在 100% 的干净真实序列上都"显著"。把残差切成长 L 的不重叠块、取块均值的
    稳健标准差 × √L：块内平均吸收自相关，MAD 使单个真断层不炸掉这个估计。"""
    block = max(2, int(block))
    n_blocks = e.size // block
    if n_blocks < _MIN_BLOCKS_FOR_LONG_RUN:
        return robust_sigma(e)
    means = e[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    return robust_sigma(means) * float(np.sqrt(block))


def _candidate_mask(n: int, breaks: list[int], min_segment: int, min_tail: int) -> np.ndarray:
    """候选 τ 掩码：距序列头 ≥ min_segment、距序列尾 ≥ min_tail、距每个已接受断点 ≥ min_segment。"""
    ok = np.zeros(n, dtype=bool)
    ok[min_segment : max(min_segment, n - min_tail)] = True
    for b in breaks:
        lo, hi = max(0, b - min_segment + 1), min(n, b + min_segment)
        ok[lo:hi] = False
    return ok


def _scan(y: np.ndarray, design: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray, float]:
    """对所有 τ 一次性算 (t 统计量, 跳变幅度 δ̂, 长程尺度)。见模块 docstring 的推导。"""
    n = y.size
    q, _ = np.linalg.qr(design)                       # n×p 正交基（design 满秩由调用方保证）
    e = y - q @ (q.T @ y)
    sigma = _long_run_sigma(e, block)                 # **长程**，不是 iid

    suffix_e = np.cumsum(e[::-1])[::-1]               # suffix_e[τ] = Σ_{t≥τ} e_t
    suffix_q = np.cumsum(q[::-1], axis=0)[::-1]       # suffix_q[τ] = Σ_{t≥τ} q_t
    count = (n - np.arange(n)).astype(float)          # ‖s_τ‖²
    norm2 = count - np.einsum("ij,ij->i", suffix_q, suffix_q)   # ‖s̃_τ‖²

    safe = norm2 > _TINY
    delta = np.zeros(n)
    tstat = np.zeros(n)
    delta[safe] = suffix_e[safe] / norm2[safe]
    if sigma > _TINY:
        tstat[safe] = np.abs(suffix_e[safe]) / (sigma * np.sqrt(norm2[safe]))
    return tstat, delta, sigma


def _excursion_offsets(n: int, edges: list[tuple[int, int]], levels: list[float],
                       gate: float) -> tuple[np.ndarray, int]:
    """只给"上去又回来"的段算偏移量；其余段偏移为 0。

    命中条件（三条全要）：① 相对前后两个邻段偏离**同一方向**（d_prev·d_next > 0）；
    ② 两侧偏离幅度的**较小者**都 ≥ gate（不是取较大者——只偏离一侧的段是 regime 变更的
    一半，不是暂态偏移）；③ 该段既非首段也非末段（首段无左邻、末段无右邻 → 无从判断
    "是否回来了" → **一律不动**，这正是"持续到末尾的电平变化 = 真实 regime 变更"的落点）。
    命中段被拉回两个邻段电平的中点。"""
    offset = np.zeros(n)
    moved = 0
    for i in range(1, len(edges) - 1):
        d_prev = levels[i] - levels[i - 1]
        d_next = levels[i] - levels[i + 1]
        if d_prev * d_next > 0 and min(abs(d_prev), abs(d_next)) >= gate:
            a, b = edges[i]
            offset[a:b] = 0.5 * (d_prev + d_next)
            moved += 1
    return offset, moved


def repair_level_shift(x, period: int = 0, t_threshold: float = 3.5,
                       min_jump_sigma: float = 1.0, min_segment: int = 10,
                       max_breaks: int = 5,
                       region_start_fraction: float | None = None,
                       region_end_fraction: float | None = None,
                       estimated_offset: float | None = None,
                       **_) -> np.ndarray:
    """检测水平断层，**撤销其中"上去又回来"的暂态偏移**；持续型 regime 变更保留。无命中 → 恒等。

    契约：destructive=True（改动已观测点取值）、preserves_observed=False、
    changes_target_space=False（只在原值域搬电平，不改目标空间）、anomaly 物理禁用。
    NaN 先线性插补（与 winsorize/denoise_* 同惯例——池要求输出全有限）。

    🚩 **本算子未通过 benchmark 冻结前的真实数据检查，不得进入 pool_v3**：在未经腐蚀的真实
    roster 序列上它仍会改动 45.7%。它作为**方法侧**动作可用（selector 的职责就是学会何时用它、
    何时不用），但 oracle 池的天花板不能建在一个到处乱动的算子上。详见 operators/CHANGELOG.md。
    """
    y = interp_nan(as_1d(x))
    n = y.size

    explicit = (
        region_start_fraction,
        region_end_fraction,
        estimated_offset,
    )
    if any(value is not None for value in explicit):
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in explicit
        ):
            raise ValueError(
                "observable level repair requires three finite numeric parameters"
            )
        start_fraction = float(region_start_fraction)
        end_fraction = float(region_end_fraction)
        offset = float(estimated_offset)
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError("observable level repair fractions are out of bounds")
        start = min(n - 1, max(0, int(np.floor(start_fraction * n))))
        end = min(n, max(start + 1, int(np.ceil(end_fraction * n))))
        if offset == 0.0:
            record(
                "repair_level_shift",
                "repair_level_shift",
                "observable_zero_offset_identity",
            )
            return y
        output = y.copy()
        output[start:end] -= offset
        record(
            "repair_level_shift",
            "repair_level_shift",
            "observable_parameterized_excursion",
        )
        return output

    p = int(period) if isinstance(period, (int, float)) and not isinstance(period, bool) and period >= 2 else 0
    if p < 2:
        p = _guess_period(y)                          # A0 共享周期模块（单一定义点，不另起估计器）
    min_segment = max(2, int(min_segment))
    min_tail = max(min_segment, p if p >= 2 else 0)

    if n < 2 * min_segment + min_tail:
        record("repair_level_shift", "repair_level_shift", "series_too_short_identity")
        return y

    level_sigma = robust_sigma(y)                     # 效应闸的尺度（序列自身的稳健 σ）
    if level_sigma <= _TINY:
        record("repair_level_shift", "repair_level_shift", "degenerate_scale_identity")
        return y                                      # 常数序列：任何"几个 σ"的判据都无定义

    t = np.arange(n, dtype=float)
    base = np.column_stack([_trend_design(n, p), _seasonal_design(n, p)])
    block = p if p >= 2 else min_segment              # 长程方差的块长 = 一个季节周期

    # ── 检测期：贪心前向选择，只过统计闸 ──────────────────────────────────────
    breaks: list[int] = []
    steps = np.zeros((n, 0))
    for _ in range(max(0, int(max_breaks))):
        design = np.column_stack([base, steps]) if steps.shape[1] else base
        tstat, _delta, _sigma = _scan(y, design, block)
        ok = _candidate_mask(n, breaks, min_segment, min_tail) & (tstat >= t_threshold)
        if not ok.any():
            break
        tau = int(np.argmax(np.where(ok, tstat, -np.inf)))
        breaks.append(tau)
        steps = np.column_stack([steps, (t >= tau).astype(float)])

    if not breaks:
        record("repair_level_shift", "repair_level_shift", "no_break_detected_identity")
        return y

    # ── 修复期：只撤销暂态偏移（见模块 docstring 的"上去又回来"规则）────────────
    breaks.sort()
    design = np.column_stack([base, steps])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    n_smooth = base.shape[1]
    smooth = design[:, :n_smooth] @ coef[:n_smooth]   # 趋势 + 季节（**不含**阶跃列）
    resid = y - smooth                                # 去平滑成分（趋势在含阶跃的设计里估过，未被断层污染）

    edges = list(zip([0, *breaks], [*breaks, n]))
    levels = [float(np.median(resid[a:b])) for a, b in edges]
    offset, moved = _excursion_offsets(n, edges, levels, min_jump_sigma * level_sigma)

    if not moved:
        record("repair_level_shift", "repair_level_shift", "no_transient_excursion_identity")
        return y                                      # 逐位恒等

    record("repair_level_shift", "repair_level_shift",
           f"repaired_{moved}_of_{len(edges)}_segments")
    return y - offset
