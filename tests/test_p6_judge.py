"""tests/test_p6_judge.py — P6 闭式可归因判官（dlinear_closed_form_v1）toy-only 测试。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_judge.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent）

全部确定性：只用固定 rng 的合成 toy 数据（正弦+噪声），不读 results/ 或 data/ 下任何文件。
torch 只在本测试文件里 import，带 skip-if-unavailable 守卫。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.p6.judge_closed_form import (
    CONTEXT_LEN,
    DEFAULT_LAM,
    HORIZON,
    INTERCEPT_IDX,
    KERNEL,
    PHI_DIM,
    PROTOCOL_ID,
    EffectPair,
    SeriesView,
    evaluate,
    fit_domain,
    fit_domain_rebuild,
    moving_average_replicate,
    phi,
    replacement_effects,
    ridge_matrix,
    series_stats,
    solve_from_design,
    solve_weights,
    torch_dlinear_state,
    window_starts,
)

try:  # torch 只在测试里 import；不可用则跳过相关测试
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch 不可用")


# ============================== toy 数据（固定 rng，合成） ==============================
def _toy_view(seed: int, n_hist: int = 240, uid: str | None = None) -> SeriesView:
    """正弦 + 弱趋势 + 噪声的合成序列；history 长 n_hist，future 长 HORIZON。"""
    rng = np.random.default_rng(seed)
    n = n_hist + HORIZON
    t = np.arange(n, dtype=np.float64)
    period = 24 + (seed % 5)
    amp = 2.0 + 0.5 * (seed % 3)
    x = (
        amp * np.sin(2.0 * np.pi * t / period)
        + 0.01 * (seed % 4) * t
        + rng.normal(0.0, 0.4, size=n)
        + 5.0
    )
    return SeriesView(uid=uid or f"toy{seed}", history=x[:n_hist], future=x[n_hist:])


def _toy_domain(n_series: int = 6, n_hist: int = 240, seed0: int = 100) -> list[SeriesView]:
    return [_toy_view(seed0 + k, n_hist=n_hist) for k in range(n_series)]


def _perturbed_view(v: SeriesView, seed: int = 777) -> SeriesView:
    """v 的一个 'prepared' 替代版本：history 平滑 + 固定 rng 微噪声，future 不变。"""
    rng = np.random.default_rng(seed)
    h = moving_average_replicate(v.history, kernel=5) + rng.normal(0.0, 0.05, size=v.history.shape[0])
    return SeriesView(uid=v.uid, history=h, future=v.future.copy())


# ============================== a. φ 语义对拍 torch ==============================
@requires_torch
def test_phi_trend_matches_torch_pad_avgpool():
    """numpy trend vs torch F.pad(replicate)+avg_pool1d，atol 1e-9（double 双侧）。"""
    rng = np.random.default_rng(0)
    for length in (CONTEXT_LEN, 61):  # 48 为协议长度；61 验证 helper 的通用语义
        for _ in range(5):
            w = rng.normal(size=length)
            trend_np = moving_average_replicate(w, KERNEL)
            xt = torch.from_numpy(w).view(1, 1, -1)  # float64
            xp = F.pad(xt, (KERNEL // 2, KERNEL // 2), mode="replicate")
            trend_t = F.avg_pool1d(xp, KERNEL, stride=1).view(-1)[:length].numpy()
            np.testing.assert_allclose(trend_np, trend_t, rtol=0.0, atol=1e-9)

    # phi 布局：[trend; season; 1]
    w = rng.normal(size=CONTEXT_LEN)
    f = phi(w)
    assert f.shape == (PHI_DIM,)
    trend = moving_average_replicate(w, KERNEL)
    np.testing.assert_array_equal(f[:CONTEXT_LEN], trend)
    np.testing.assert_array_equal(f[CONTEXT_LEN : 2 * CONTEXT_LEN], w - trend)
    assert f[INTERCEPT_IDX] == 1.0


@requires_torch
def test_weights_unpack_to_torch_dlinear():
    """全模型等价：W* 拆回 torch DLinear 的 lin_trend/lin_season（截距并入 bias），
    同输入预测 allclose(atol 1e-6)。"""
    from SelfEvolvingHarnessTS.evaluators._torch_models import DLinear

    views = _toy_domain(4, n_hist=200, seed0=300)
    fit = fit_domain(views)
    sd = torch_dlinear_state(fit.W)

    model = DLinear(CONTEXT_LEN, HORIZON, kernel=KERNEL).double()
    model.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
    model.eval()

    X = np.stack([s.eval_input for s in fit.stats])  # (B, 48) z-scored
    with torch.no_grad():
        pred_torch = model(torch.from_numpy(X)).numpy()
    pred_np = np.stack([phi(x) @ fit.W for x in X])
    np.testing.assert_allclose(pred_np, pred_torch, rtol=0.0, atol=1e-6)


# ============================== b. 双路一致 ==============================
def test_dual_path_consistency():
    """fit_domain（充分统计量）vs fit_domain_rebuild（全量堆叠）：W* 与 utility allclose(1e-9)。"""
    views = _toy_domain(6, n_hist=240)
    for kwargs in (
        {},  # 默认 lam=1e-3, stride=4, cap=None
        {"lam": 1e-2, "stride": 1, "window_cap": 12},
        {"stride": 7},
    ):
        f1 = fit_domain(views, **kwargs)
        f2 = fit_domain_rebuild(views, **kwargs)
        np.testing.assert_allclose(f1.W, f2.W, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(
            f1.per_series_rmse, f2.per_series_rmse, rtol=0.0, atol=1e-9
        )
        assert abs(f1.utility - f2.utility) <= 1e-9
        assert f1.n_windows_total == f2.n_windows_total
        assert f1.protocol == f2.protocol == PROTOCOL_ID


# ============================== c. 反事实 vs 暴力重建 ==============================
def test_replacement_effects_match_bruteforce():
    """toy 6 序列域，替换第 2 条（index 2）：三效应逐一与"从头重建数据再拟合"allclose(1e-9)。"""
    views = _toy_domain(6, n_hist=240)
    i = 2
    view_e = _perturbed_view(views[i])

    eff = replacement_effects(views, i, view_e)

    # ---- 暴力重建：从原始 views 重新构造一切 ----
    stats_H = [series_stats(v) for v in views]
    W_H = solve_weights(stats_H)
    rmse_H, u_H = evaluate(W_H, stats_H)

    views_repl = list(views)
    views_repl[i] = view_e
    stats_R = [series_stats(v) for v in views_repl]     # 从头重建（充分统计量路）
    W_R = solve_weights(stats_R)
    W_R2 = fit_domain_rebuild(views_repl).W             # 从头重建（全量堆叠路，独立实现）
    np.testing.assert_allclose(W_R, W_R2, rtol=0.0, atol=1e-9)

    # train：模型=替换域重拟合，eval 全保持 H
    rmse_T, u_T = evaluate(W_R, stats_H)
    # context：模型保持 H 拟合，eval 中序列 i 换成 e
    rmse_C, u_C = evaluate(W_H, stats_R)
    # joint：两者都换
    rmse_J, u_J = evaluate(W_R, stats_R)

    assert eff.baseline_utility == pytest.approx(u_H, abs=1e-12)
    assert eff.baseline_self_rmse == pytest.approx(float(rmse_H[i]), abs=1e-12)

    np.testing.assert_allclose(eff.train_effect.batch_delta, u_T - u_H, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        eff.train_effect.self_delta, float(rmse_T[i] - rmse_H[i]), rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(eff.context_effect.batch_delta, u_C - u_H, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        eff.context_effect.self_delta, float(rmse_C[i] - rmse_H[i]), rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(eff.joint_effect.batch_delta, u_J - u_H, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(
        eff.joint_effect.self_delta, float(rmse_J[i] - rmse_H[i]), rtol=0.0, atol=1e-9
    )

    # 交叉验证：train 效应也应与"全量堆叠路重建的模型"一致（独立第二条对拍路径）
    rmse_T2, u_T2 = evaluate(W_R2, stats_H)
    np.testing.assert_allclose(eff.train_effect.batch_delta, u_T2 - u_H, rtol=0.0, atol=1e-9)

    # context 不动模型：j≠i 的序列 RMSE 应严格不变
    for j in range(len(views)):
        if j != i:
            assert float(rmse_C[j]) == float(rmse_H[j])

    # 替换确实产生了非零效应（toy 扰动足够大）
    assert abs(eff.joint_effect.batch_delta) > 1e-12


# ============================== d. 截距不受罚 ==============================
def test_ridge_matrix_intercept_unpenalized():
    R = ridge_matrix()
    expect = np.eye(PHI_DIM)
    expect[INTERCEPT_IDX, INTERCEPT_IDX] = 0.0
    np.testing.assert_array_equal(R, expect)
    assert R[INTERCEPT_IDX, INTERCEPT_IDX] == 0.0


def test_target_shift_absorbed_by_intercept():
    """目标整体平移 c：截距行 +c 吸收，非截距权重不变，预测同步平移。
    （截距受罚时该性质不成立——正是"截距不受罚"的行为学验证。）"""
    rng = np.random.default_rng(7)
    Phi = np.stack([phi(rng.normal(size=CONTEXT_LEN)) for _ in range(150)])
    Y = rng.normal(size=(150, HORIZON))
    c = 3.7

    W0 = solve_from_design(Phi, Y, lam=DEFAULT_LAM)
    W1 = solve_from_design(Phi, Y + c, lam=DEFAULT_LAM)

    np.testing.assert_allclose(W1[:INTERCEPT_IDX], W0[:INTERCEPT_IDX], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        W1[INTERCEPT_IDX], W0[INTERCEPT_IDX] + c, rtol=0.0, atol=1e-8
    )
    x = rng.normal(size=CONTEXT_LEN)
    np.testing.assert_allclose(phi(x) @ W1, phi(x) @ W0 + c, rtol=0.0, atol=1e-8)


# ============================== e. 确定性 ==============================
def test_fit_deterministic_bytewise():
    """同输入两次拟合：W* 字节级一致；反事实两次调用逐字段相等。"""
    views = _toy_domain(5, n_hist=200, seed0=500)
    f1 = fit_domain(views)
    f2 = fit_domain(views)
    assert f1.W.tobytes() == f2.W.tobytes()
    assert f1.per_series_rmse.tobytes() == f2.per_series_rmse.tobytes()
    assert f1.utility == f2.utility

    view_e = _perturbed_view(views[1])
    e1 = replacement_effects(views, 1, view_e)
    e2 = replacement_effects(views, 1, view_e)
    assert e1 == e2  # frozen dataclass：全 float/str/int 字段精确相等


# ============================== f. stride 与窗口配额 ==============================
def test_window_starts_stride_and_cap():
    n_hist = 240
    for stride in (1, 4, 7):
        starts = window_starts(n_hist, stride=stride)
        assert len(starts) == (n_hist - 96) // stride + 1
        assert starts[0] == 0
        assert all(b - a == stride for a, b in zip(starts, starts[1:]))
        assert starts[-1] + 96 <= n_hist  # 全在 history 内

    assert window_starts(95, stride=4) == []   # 不足一个窗口
    assert window_starts(96, stride=4) == [0]  # 恰好一个
    assert window_starts(240, stride=4, window_cap=10) == window_starts(240, stride=4)[:10]
    assert len(window_starts(240, stride=4, window_cap=10)) == 10
    assert window_starts(240, stride=4, window_cap=0) == []


def test_series_stats_window_counts():
    v = _toy_view(3, n_hist=240)
    s_all = series_stats(v, stride=4)
    assert s_all.n_windows == (240 - 96) // 4 + 1  # 37

    s1 = series_stats(v, stride=1)
    assert s1.n_windows == 240 - 96 + 1  # 145

    s_cap = series_stats(v, stride=4, window_cap=5)
    assert s_cap.n_windows == 5
    assert not np.array_equal(s_cap.G, s_all.G)  # 配额确实截断了统计量

    s_big = series_stats(v, stride=4, window_cap=10_000)
    assert s_big.n_windows == 37  # 配额大于可用窗口数 → 全部

    # history 不足 96：0 个训练窗口，但 eval 状态仍然可用
    v_short = _toy_view(9, n_hist=95)
    s_short = series_stats(v_short, stride=4)
    assert s_short.n_windows == 0
    assert s_short.eval_input.shape == (CONTEXT_LEN,)
    assert s_short.future_norm.shape == (HORIZON,)
    np.testing.assert_array_equal(s_short.G, np.zeros((PHI_DIM, PHI_DIM)))

    # 配额影响域拟合的窗口总数
    views = _toy_domain(3, n_hist=240)
    fit = fit_domain(views, window_cap=8)
    assert fit.n_windows_total == 3 * 8
    assert all(s.n_windows == 8 for s in fit.stats)


# ============================== g. raw-vs-raw 替换 → 三效应精确为 0 ==============================
def test_identity_replacement_exactly_zero():
    views = _toy_domain(6, n_hist=240)
    i = 3
    view_e = SeriesView(
        uid=views[i].uid,
        history=views[i].history.copy(),
        future=views[i].future.copy(),
    )
    eff = replacement_effects(views, i, view_e)
    assert eff.train_effect == EffectPair(0.0, 0.0)
    assert eff.context_effect == EffectPair(0.0, 0.0)
    assert eff.joint_effect == EffectPair(0.0, 0.0)


# ============================== h. series_weight="equal"（等权序列，v4 外审要求） ==============================
def _mixed_length_domain() -> list:
    """窗口数不等的 toy 域：n_hist 差异 → n_windows 差异（长序列支配训练的场景）。"""
    return [
        _toy_view(900, n_hist=400, uid="long_a"),
        _toy_view(901, n_hist=400, uid="long_b"),
        _toy_view(902, n_hist=152, uid="short_a"),
        _toy_view(903, n_hist=160, uid="short_b"),
    ]


def test_equal_weight_dual_path_consistency():
    views = _mixed_length_domain()
    fit_a = fit_domain(views, series_weight="equal")
    fit_b = fit_domain_rebuild(views, series_weight="equal")
    assert np.allclose(fit_a.W, fit_b.W, atol=1e-9)
    assert abs(fit_a.utility - fit_b.utility) < 1e-12
    assert fit_a.series_weight == "equal" and fit_b.series_weight == "equal"


def test_equal_weight_changes_solution_iff_windows_unequal():
    # 窗口数不等：等权序列必须改变解
    mixed = _mixed_length_domain()
    W_none = fit_domain(mixed).W
    W_eq = fit_domain(mixed, series_weight="equal").W
    assert not np.allclose(W_none, W_eq, atol=1e-9)
    # 窗口数全等：w_i ≡ 1.0（精确），解 bit 级一致
    same = _toy_domain(4, n_hist=240)
    W_none2 = fit_domain(same).W
    W_eq2 = fit_domain(same, series_weight="equal").W
    assert W_none2.tobytes() == W_eq2.tobytes()


def test_equal_weight_identity_replacement_exactly_zero():
    views = _mixed_length_domain()
    i = 2
    view_e = SeriesView(
        uid=views[i].uid,
        history=views[i].history.copy(),
        future=views[i].future.copy(),
    )
    eff = replacement_effects(views, i, view_e, series_weight="equal")
    assert eff.train_effect == EffectPair(0.0, 0.0)
    assert eff.context_effect == EffectPair(0.0, 0.0)
    assert eff.joint_effect == EffectPair(0.0, 0.0)


def test_equal_weight_replacement_matches_bruteforce():
    views = _mixed_length_domain()
    i = 1
    view_e = _perturbed_view(views[i], seed=888)
    eff = replacement_effects(views, i, view_e, series_weight="equal")

    # 暴力路：train = 用替换后 views 重新加权拟合，但 eval 全保持 H 版
    views_repl = list(views)
    views_repl[i] = view_e
    fit_H = fit_domain(views, series_weight="equal")
    fit_T = fit_domain_rebuild(views_repl, series_weight="equal")   # 独立全量路
    rmses_train, u_train = evaluate(fit_T.W, fit_H.stats)
    assert abs(eff.train_effect.batch_delta - (u_train - fit_H.utility)) < 1e-9
    assert abs(
        eff.train_effect.self_delta - float(rmses_train[i] - fit_H.per_series_rmse[i])
    ) < 1e-9
    # joint = 替换后模型 + 替换后 eval
    rmses_joint, u_joint = evaluate(fit_T.W, fit_domain(views_repl, series_weight="equal").stats)
    assert abs(eff.joint_effect.batch_delta - (u_joint - fit_H.utility)) < 1e-9


def test_series_weight_invalid_mode_raises():
    views = _toy_domain(3, n_hist=240)
    try:
        fit_domain(views, series_weight="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 series_weight 应 raise ValueError")
