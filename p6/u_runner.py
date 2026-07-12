"""p6/u_runner.py — P6 U 终评 runner（prereg §5：cycle2 terminal 后一次性、**非门**）。

外审 NO-GO 兑现件：H0 vs H_final paired 于 U（20 virgin × 4 preset）：
闭式判官三效应（train/context/joint 全披露）+ Adam-DLinear(3 seeds) + LSTM-scratch(3 seeds)。
**本 runner 不做任何晋升 gate 决策**（无 promote/reject/abstain）：报告只含数值与"成功描述子"
（train_gain LCB ≥ −δ_safe 且方向为正 → 允许 §0 的转移限定词）。但一次性 open 的记账由本
runner 自己承担（F5/finding 35）——见流程 1。

流程：
  1. `gate.can_open("U")` 前置校验（False → raise P6StateError），随后**本 runner 自己**
     `gate.open_block("U", bindings)`（bindings 必含 materialization sha；空 bindings 拒绝）——
     在读取任何 U episode 数据之前，一次性开箱由 gate 强制（二次 open 被拒 → 无法重复窥视）；
  2. paired 两臂 fast path：同一 episodes、同 K（两 state 的 sampler.expected_total 必须
     相等，否则 P6PairingError）→ 每臂逐 episode chosen → prepared artifact
     （fast_path.prepared_artifact；执行失败 None → P6TechnicalAbort）→ 判官 ingestion fill
     （c0_runner.judge_ingest，冻结释义同 C0）；
  3. 闭式判官三效应（batch 级 H0→H_final 替换反事实，语义 = judge_closed_form.
     replacement_effects 推广到整批替换）：
       loss_00 = fit(H0), eval(H0)      —— 基线
       loss_10 = fit(FINAL), eval(H0)   —— train effect（主判据口径）
       loss_01 = fit(H0), eval(FINAL)   —— context effect
       loss_11 = fit(FINAL), eval(FINAL)—— joint effect（Adam co-gate 同口径）
     两次正式拟合都过 paired_judge_fit 双路对拍（prereg §1）；gain 一律经 p6.metrics 词汇表
     （gain(loss_00, loss_arm)，调用方不手写符号）；
  4. Adam(3 seeds) + LSTM(3 seeds)（trainer 注入；契约 = trainer(views, seed) →
     per-episode losses；每 seed 两臂配对调用 → 各 3×2=6 次拟合）；报告 per-seed 两臂
     loss 与 joint 口径 gain（torch 报告器无 train/context 分解——三效应分解是闭式判官的
     attribution-exact 专属，prereg §0）；
  5. 成功描述子：train_gain LCB ≥ −δ_safe（无害，≥ 含边界）∧ 方向为正（overall train_gain
     > 0，严格）；LCB = metrics.cluster_lcb90，cluster = 底层 series_uid（4 preset 同抽）。

——冻结释义——
  A. bootstrap seed：prereg §4 只冻结 V 用 20260711+cycle（cycle1→+1、cycle2→+2）；
     U 终评顺延 U_BOOTSTRAP_SEED = 20260711+3（可经参数覆盖，正式跑用默认）。
  B. per-preset 披露：gain/lcb90/n/direction 按 episode 的 preset 分组（loss 取该组等权均值，
     gain 经词汇表）；LCB 的 cluster 仍为 series_uid。
  C. judge_cfg 必须携带 delta_safe（来自 C0_FREEZE；缺失 → ValueError——描述子没有 δ 不成立）；
     判官协议键（lam/stride/window_cap/series_weight）缺省用 C0 冻结默认。

红线：不改任何现有文件；无网络/LLM（llm_supplier 只接受注入 callable，默认 None）；
模块级不 import torch（make_lstm_trainer 惰性）。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .c0_runner import (
    ADAM_FIT_TIMEOUT_SECONDS,
    FROZEN_TRAINER_FIT_TIMEOUT_SECONDS,
    JUDGE_CFG_FROZEN,
    P6Episode,
    P6FrozenParamError,
    _checked_trainer_call,
    _resolve_judge_cfg,
    frozen_literals_digest,
    judge_ingest,
    make_torch_forecaster_trainer,
    paired_judge_fit,
)
from .fast_path import (
    P6PairingError,
    merge_preset_fingerprints,
    prepared_artifact,
    run_fast_path,
)
from .final_packet import write_final_packet
from .judge_closed_form import DomainFit, SeriesView, evaluate
from .loaders import BoundUEpisodes
from .materializer import P6TechnicalAbort, validate_materialization
from .metrics import cluster_lcb90, gain
from .split_manifest import P6StateError

__all__ = [
    "U_BOOTSTRAP_SEED",
    "assert_u_frozen_params",
    "direction_of",
    "make_lstm_trainer",
    "run_u_eval_formal",
    "run_u_eval_unfrozen",
    "train_gain_descriptor",
    "u_frozen_literals",
]

#: 冻结释义 A：U 终评 bootstrap seed = 20260711 + 3（V 为 +cycle；U 顺延）。
U_BOOTSTRAP_SEED = 20260711 + 3
U_PAIRED_SEEDS: Tuple[int, ...] = (0, 1, 2)      # prereg §1/§5：paired seeds {0,1,2}
DEFAULT_BOOTSTRAP_B = 2000                       # prereg §4：B=2000


def _check_u_bindings(bindings: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """U open bindings 校验（F5）：必含非空 materialization_sha；全部值须为非空 str。
    空 bindings / 缺 materialization_sha → P6StateError（拒绝开箱，无法重复窥视）。"""
    b = dict(bindings or {})
    mat = b.get("materialization_sha")
    if not isinstance(mat, str) or not mat.strip():
        raise P6StateError(
            "run_u_eval：U open bindings 必须含非空 materialization_sha（空 bindings 拒绝）"
        )
    out: Dict[str, str] = {}
    for k, v in b.items():
        if not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip():
            raise P6StateError(f"U open bindings 项 {k!r} 必须为非空 str → 非空 str")
        out[k] = v
    return out


def assert_u_frozen_params(
    seeds: Sequence[int], bootstrap_b: int, bootstrap_seed: int, K: int,
    trainer_fit_timeout_seconds: float = FROZEN_TRAINER_FIT_TIMEOUT_SECONDS,
) -> None:
    """run_u_eval_formal 断言（prereg §1/§4/§5；值写死字面量，G4）。任一漂移 → P6FrozenParamError。"""
    if tuple(int(s) for s in seeds) != (0, 1, 2):
        raise P6FrozenParamError(f"formal U：seeds 必须 == (0, 1, 2)，得到 {tuple(seeds)!r}")
    if int(bootstrap_b) != 2000:
        raise P6FrozenParamError(f"formal U：bootstrap_b 必须 == 2000，得到 {bootstrap_b}")
    if int(bootstrap_seed) != 20260714:
        raise P6FrozenParamError(
            f"formal U：U 终评 bootstrap seed 必须 == 20260714，得到 {bootstrap_seed}"
        )
    if int(K) != 8:
        raise P6FrozenParamError(f"formal U：K slot 预算必须 == 8，得到 {K}")
    if float(trainer_fit_timeout_seconds) != 900.0:
        raise P6FrozenParamError(
            f"formal U：trainer fit 超时必须 == 900.0，得到 {trainer_fit_timeout_seconds!r}"
        )


def u_frozen_literals(
    seeds: Sequence[int], bootstrap_b: int, bootstrap_seed: int, K: int,
    trainer_fit_timeout_seconds: float,
) -> Dict[str, Any]:
    """run_u_eval_formal 冻结字面量集合（用于 provenance digest）。"""
    return {
        "seeds": [int(s) for s in seeds], "bootstrap_b": int(bootstrap_b),
        "bootstrap_seed": int(bootstrap_seed), "K": int(K),
        "trainer_fit_timeout_seconds": float(trainer_fit_timeout_seconds),
    }


def _verify_u_loaded(loaded: Any, expected_materialization_sha: str) -> List[P6Episode]:
    """U loader 返回值 manifest-bound 验证（G3/finding 35）。任一不符 → P6TechnicalAbort。

    loaded 必须是 BoundUEpisodes；逐条 series_uid ∈ materialization、content_sha 复算一致、
    config/preset 一致、materialization 实际 sha == open 绑定值。返回验证过的 episodes。"""
    if not isinstance(loaded, BoundUEpisodes):
        raise P6TechnicalAbort(
            f"U loader 必须返回 BoundUEpisodes，得到 {type(loaded).__name__}"
            "（U 已 open，事故如实留台账）"
        )
    sm = loaded.materialization
    if str(sm.materialization_sha) != str(expected_materialization_sha):
        raise P6TechnicalAbort(
            f"U 物化实际 sha 与 open 绑定不一致：{sm.materialization_sha} != {expected_materialization_sha}"
        )
    validate_materialization(sm, loaded.series_by_uid)      # content_sha 复算一致
    mat_uids = set(sm.uids())
    mat_cfg = sm.config
    episodes = list(loaded.episodes)
    if not episodes:
        raise ValueError("U episodes 不能为空")
    for ep in episodes:
        if ep.series_uid not in mat_uids:
            raise P6TechnicalAbort(
                f"U episode {ep.uid} 的 series_uid {ep.series_uid!r} ∉ materialization record"
            )
        if str(ep.config) != mat_cfg:
            raise P6TechnicalAbort(
                f"U episode {ep.uid} config {ep.config!r} != materialization config {mat_cfg!r}"
            )
        if not ep.preset or ep.uid != f"{ep.series_uid}:{ep.preset}":
            raise P6TechnicalAbort(f"U episode {ep.uid} preset/uid 格式不一致（config:item:preset）")
    return episodes


def direction_of(g: float) -> str:
    """方向词（§6 结果表）：正/负/零。"""
    g = float(g)
    if g > 0.0:
        return "positive"
    if g < 0.0:
        return "negative"
    return "zero"


def make_lstm_trainer(**kwargs: Any) -> Callable[[Sequence[SeriesView], int], np.ndarray]:
    """LSTM-scratch reporter trainer（prereg §1 roster；from-scratch，hidden=64）。

    = c0_runner.make_torch_forecaster_trainer("lstm", **kwargs)：CPU、每 fit 前 seed_all、
    torch deterministic、判官同协议窗与评估。prereg 未单列 LSTM 超参 → 冻结为与
    Adam co-gate 相同的训练协议（epochs=120、lr=1e-2、bs=256）。LSTM 禁参与任何选择/否决。
    """
    return make_torch_forecaster_trainer("lstm", **kwargs)


def train_gain_descriptor(
    per_episode_train_gains: Sequence[float],
    clusters: Sequence[Any],
    overall_train_gain: float,
    delta_safe: float,
    *,
    b: int = DEFAULT_BOOTSTRAP_B,
    seed: int = U_BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """成功描述子（prereg §5，纯函数）：train_gain LCB ≥ −δ_safe（≥ 含边界）∧ 方向为正（严格 >0）。

    LCB = metrics.cluster_lcb90（cluster = 底层 series_uid，4 preset 同抽）。**非门**：
    描述子只决定措辞（是否允许 §0 转移限定词），不产生任何 promote/reject。
    """
    d = float(delta_safe)
    if not np.isfinite(d) or d < 0.0:
        raise ValueError(f"delta_safe 必须是非负有限数，got {delta_safe!r}")
    lcb = cluster_lcb90(per_episode_train_gains, clusters, int(b), seed=int(seed))
    g = float(overall_train_gain)
    non_harm = bool(lcb >= -d)
    positive = bool(g > 0.0)
    return {
        "train_gain": g,
        "train_gain_lcb90": lcb,
        "delta_safe": d,
        "non_harm": non_harm,                    # LCB ≥ −δ_safe（含边界）
        "direction_positive": positive,          # 严格 > 0
        "success": bool(non_harm and positive),  # 允许 "directional transfer" 限定词
    }


# ── 内部：效应块 / preset 分解 ─────────────────────────────────────────────
def _per_episode_gains(loss_base: np.ndarray, loss_arm: np.ndarray) -> List[float]:
    return [gain(float(a), float(b)) for a, b in zip(loss_base, loss_arm)]


def _effect_block(
    loss_base: np.ndarray,
    loss_arm: np.ndarray,
    u_base: float,
    u_arm: float,
    episodes: Sequence[P6Episode],
    clusters: Sequence[str],
    b: int,
    seed: int,
) -> Dict[str, Any]:
    gains = _per_episode_gains(loss_base, loss_arm)
    overall = gain(float(u_base), float(u_arm))
    block: Dict[str, Any] = {
        "overall_gain": overall,
        "lcb90": cluster_lcb90(gains, clusters, b, seed=seed),
        "direction": direction_of(overall),
        "per_preset": {},
    }
    presets = sorted({ep.preset for ep in episodes})
    for p in presets:
        idx = [i for i, ep in enumerate(episodes) if ep.preset == p]
        base_p = float(np.mean([float(loss_base[i]) for i in idx]))
        arm_p = float(np.mean([float(loss_arm[i]) for i in idx]))
        g_p = gain(base_p, arm_p)
        block["per_preset"][p] = {
            "gain": g_p,
            "lcb90": cluster_lcb90([gains[i] for i in idx], [clusters[i] for i in idx],
                                   b, seed=seed),
            "n": len(idx),
            "direction": direction_of(g_p),
        }
    return block


def _reporter_block(
    trainer: Callable[[Sequence[SeriesView], int], Any],
    views_h0: Sequence[SeriesView],
    views_final: Sequence[SeriesView],
    seeds: Sequence[int],
    episodes: Sequence[P6Episode],
    timeout_seconds: float,
    label: str,
) -> Tuple[Dict[str, Any], int]:
    """torch reporter（Adam/LSTM）：每 seed 两臂配对（H0 先、FINAL 后）→ per-seed joint gain；
    per-episode seed 均值 → per-preset joint gain 披露。返回 (block, n_fits)。"""
    per_seed: List[Dict[str, Any]] = []
    losses_h0: List[np.ndarray] = []
    losses_fin: List[np.ndarray] = []
    n_fits = 0
    for s in seeds:
        la = _checked_trainer_call(trainer, views_h0, int(s), timeout_seconds,
                                   f"{label}[h0/seed={s}]")
        n_fits += 1
        lf = _checked_trainer_call(trainer, views_final, int(s), timeout_seconds,
                                   f"{label}[final/seed={s}]")
        n_fits += 1
        losses_h0.append(la)
        losses_fin.append(lf)
        u_a, u_f = float(np.mean(la)), float(np.mean(lf))
        per_seed.append({"seed": int(s), "loss_h0": u_a, "loss_final": u_f,
                         "gain": gain(u_a, u_f)})
    mean_h0 = np.mean(np.stack(losses_h0, axis=0), axis=0)
    mean_fin = np.mean(np.stack(losses_fin, axis=0), axis=0)
    overall = gain(float(np.mean(mean_h0)), float(np.mean(mean_fin)))
    per_preset: Dict[str, Any] = {}
    for p in sorted({ep.preset for ep in episodes}):
        idx = [i for i, ep in enumerate(episodes) if ep.preset == p]
        g_p = gain(float(np.mean(mean_h0[idx])), float(np.mean(mean_fin[idx])))
        per_preset[p] = {"gain": g_p, "n": len(idx), "direction": direction_of(g_p)}
    block = {
        "per_seed": per_seed,
        "mean_gain": float(np.mean([r["gain"] for r in per_seed])),
        "overall_gain_seed_mean_losses": overall,
        "per_preset_gain": per_preset,
        "direction": direction_of(overall),
    }
    return block, n_fits


# ════════════════════════════ U 终评主流程 ════════════════════════════
def run_u_eval_unfrozen(
    u_loader: Callable[[], Any],
    state_h0: Any,
    state_final: Any,
    judge_cfg: Mapping[str, Any],
    adam_trainer: Callable[[Sequence[SeriesView], int], Any],
    lstm_trainer: Callable[[Sequence[SeriesView], int], Any],
    gate: Any,
    *,
    seeds: Sequence[int] = U_PAIRED_SEEDS,
    llm_supplier: Optional[Callable[..., Any]] = None,
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = U_BOOTSTRAP_SEED,
    trainer_fit_timeout_seconds: float = ADAM_FIT_TIMEOUT_SECONDS,
    bindings: Optional[Mapping[str, str]] = None,
    _entrypoint: Optional[str] = None,
    _frozen_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """U 终评核心（**测试专用；正式运行禁止调用——用 run_u_eval_formal**）。返回 §6 结果表 dict。

    **次序机械固定（G3/finding 35）**：`gate.open_block("U", bindings 含 materialization_sha)`
    → 调用零参 `u_loader()`（延迟加载）→ manifest-bound 验证（loaded 是 BoundUEpisodes；
    逐条 series_uid ∈ materialization、content_sha 复算一致、config/preset 一致、materialization
    实际 sha == open 绑定值）→ 评估。任何验证失败 → P6TechnicalAbort（U 已 open，事故如实
    留台账——特性非缺陷）。空/缺 materialization_sha 的 bindings → open 前拒绝。仍**非门**。
    冻结字面量断言不在此（在 run_u_eval_formal）。
    """
    t_start = time.perf_counter()
    if not bool(gate.can_open("U")):
        raise P6StateError(
            "U 未解锁：gate.can_open('U') = False（U 开启需 cycle2 terminal 已记录，"
            "prereg §4/§5）——U 终评拒绝运行"
        )
    if not seeds:
        raise ValueError("seeds 不能为空")

    # judge_cfg：delta_safe 必备（冻结释义 C），其余键并入冻结判官协议
    cfg_in = dict(judge_cfg or {})
    if "delta_safe" not in cfg_in:
        raise ValueError("judge_cfg 必须携带 delta_safe（来自 C0_FREEZE）——成功描述子依赖 δ_safe")
    delta_safe = float(cfg_in.pop("delta_safe"))
    epsilon = cfg_in.pop("epsilon", None)            # 允许携带（只披露，不判决）
    judge_proto = _resolve_judge_cfg(cfg_in)

    k_h0 = int(state_h0.sampler.expected_total)
    k_final = int(state_final.sampler.expected_total)
    if k_h0 != k_final:
        raise P6PairingError(f"U paired 两臂 K 不一致：H0={k_h0} vs H_final={k_final}（预算冻结）")
    b = int(bootstrap_b)
    seed = int(bootstrap_seed)
    entrypoint = _entrypoint or "run_u_eval_unfrozen"

    # G3：读取任何 U 数据之前先 open（含 materialization_sha 绑定 + 正式入口 provenance）。
    u_bindings = _check_u_bindings(bindings)
    open_bindings = dict(u_bindings)
    open_bindings["entrypoint"] = entrypoint
    if _frozen_digest is not None:
        open_bindings["frozen_literals_digest"] = _frozen_digest
    gate.open_block("U", open_bindings)

    # G3：延迟加载 + manifest-bound 验证（U 已 open，任何失败留台账 → technical abort）
    episodes = _verify_u_loaded(u_loader(), u_bindings["materialization_sha"])
    uids = [ep.uid for ep in episodes]
    if len(set(uids)) != len(uids):
        raise ValueError("U episodes uid 重复")

    # —— paired 两臂 fast path（同 episodes、同 K） ——
    fp_views = [(ep.uid, ep.history) for ep in episodes]
    fp_run = merge_preset_fingerprints(episodes, fingerprints)   # F7：preset scope 可求值
    fp_h0 = run_fast_path(fp_views, state_h0, k_h0, llm_supplier, fp_run)
    fp_final = run_fast_path(fp_views, state_final, k_final, llm_supplier, fp_run)

    def _prepared_views(fp_result: Mapping[str, Any], arm: str) -> List[SeriesView]:
        out: List[SeriesView] = []
        for ep in episodes:
            art = prepared_artifact(fp_result[ep.uid], ep.history)
            if art is None:
                raise P6TechnicalAbort(
                    f"U/{arm}: episode {ep.uid} 的 chosen 执行失败（prepared=None）→ technical abort"
                )
            out.append(SeriesView(uid=ep.uid, history=judge_ingest(art), future=ep.future))
        return out

    views_h0 = _prepared_views(fp_h0, "H0")
    views_final = _prepared_views(fp_final, "H_final")

    # —— 闭式判官三效应（2 次正式拟合，均过双路对拍；交叉评估复用 stats） ——
    fit0 = paired_judge_fit(views_h0, judge_proto, fit_fn=fit_fn, rebuild_fn=rebuild_fn)
    fit1 = paired_judge_fit(views_final, judge_proto, fit_fn=fit_fn, rebuild_fn=rebuild_fn)
    n_cf_fits = 2
    loss_00 = np.asarray(fit0.per_series_rmse, float)
    loss_11 = np.asarray(fit1.per_series_rmse, float)
    u_00, u_11 = float(fit0.utility), float(fit1.utility)
    loss_10, u_10 = evaluate(fit1.W, fit0.stats)     # train：模型换 FINAL、eval 保持 H0
    loss_01, u_01 = evaluate(fit0.W, fit1.stats)     # context：模型保持 H0、eval 换 FINAL

    clusters = [ep.series_uid for ep in episodes]   # b/seed/formal/open 已在 fast path 前处理
    effects = {
        "train": _effect_block(loss_00, loss_10, u_00, u_10, episodes, clusters, b, seed),
        "context": _effect_block(loss_00, loss_01, u_00, u_01, episodes, clusters, b, seed),
        "joint": _effect_block(loss_00, loss_11, u_00, u_11, episodes, clusters, b, seed),
    }
    per_episode = [
        {
            "uid": ep.uid, "series_uid": ep.series_uid, "preset": ep.preset,
            "loss_00": float(loss_00[i]), "loss_10": float(loss_10[i]),
            "loss_01": float(loss_01[i]), "loss_11": float(loss_11[i]),
            "train_gain": gain(float(loss_00[i]), float(loss_10[i])),
            "context_gain": gain(float(loss_00[i]), float(loss_01[i])),
            "joint_gain": gain(float(loss_00[i]), float(loss_11[i])),
        }
        for i, ep in enumerate(episodes)
    ]

    # —— Adam / LSTM reporter（注入 trainer；每 seed 两臂配对） ——
    adam_block, n_adam = _reporter_block(adam_trainer, views_h0, views_final, seeds,
                                         episodes, trainer_fit_timeout_seconds, "Adam")
    lstm_block, n_lstm = _reporter_block(lstm_trainer, views_h0, views_final, seeds,
                                         episodes, trainer_fit_timeout_seconds, "LSTM")

    # —— 成功描述子（非门） ——
    train_gains = _per_episode_gains(loss_00, loss_10)
    descriptor = train_gain_descriptor(
        train_gains, clusters, effects["train"]["overall_gain"], delta_safe,
        b=b, seed=seed,
    )

    return {
        "schema_version": "p6-u-eval/1",
        "block": "U",
        "non_gate_note": ("U 终评非门（prereg §5）：本报告不产生任何 promote/reject/abstain "
                          "决策；成功描述子只决定 §0 转移限定词措辞"),
        "protocol": {
            "judge": dict(judge_proto),
            "delta_safe": delta_safe,
            "epsilon": (float(epsilon) if epsilon is not None else None),
            "paired_seeds": [int(s) for s in seeds],
            "bootstrap": {"b": b, "seed": seed, "cluster": "series_uid（4 preset 同抽）",
                          "lcb": "5% 分位（quantile linear）= LCB90"},
            "K": k_h0,
            "state_h0_sha": state_h0.sha(),
            "state_final_sha": state_final.sha(),
        },
        "n_episodes": len(episodes),
        "n_series": len(set(clusters)),
        "presets_observed": sorted({ep.preset for ep in episodes}),
        "arms": {
            "h0": {"utility": u_00,
                   "n_abstained": sum(1 for ep in episodes if fp_h0[ep.uid] is None),
                   "pool_stats": {u: dict(s) for u, s in fp_h0.pool_stats.items()}},
            "final": {"utility": u_11,
                      "n_abstained": sum(1 for ep in episodes if fp_final[ep.uid] is None),
                      "pool_stats": {u: dict(s) for u, s in fp_final.pool_stats.items()}},
        },
        "judge_effects": effects,          # train/context/joint 全披露（prereg §5）
        "per_episode": per_episode,
        "adam": adam_block,
        "lstm": lstm_block,
        "success_descriptor": descriptor,
        "provenance": {                    # G4：正式入口证据（entrypoint + 冻结字面量 digest + 物化 sha）
            "entrypoint": entrypoint,
            "frozen_literals_digest": _frozen_digest,
            "materialization_sha": u_bindings["materialization_sha"],
        },
        "costs": {
            "closed_form_fits": n_cf_fits,
            "adam_fits": n_adam,
            "lstm_fits": n_lstm,
            "wall_clock_seconds": round(time.perf_counter() - t_start, 3),
        },
    }


def run_u_eval_formal(
    u_loader: Callable[[], Any],
    state_h0: Any,
    state_final: Any,
    judge_cfg: Mapping[str, Any],
    adam_trainer: Callable[[Sequence[SeriesView], int], Any],
    lstm_trainer: Callable[[Sequence[SeriesView], int], Any],
    gate: Any,
    *,
    bindings: Mapping[str, str],
    seeds: Sequence[int] = U_PAIRED_SEEDS,
    llm_supplier: Optional[Callable[..., Any]] = None,
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]] = None,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = U_BOOTSTRAP_SEED,
    trainer_fit_timeout_seconds: float = ADAM_FIT_TIMEOUT_SECONDS,
    final_packet_path: Optional[Any] = None,
    freeze_shas: Optional[Mapping[str, str]] = None,
    selection_manifest_sha: Optional[str] = None,
    claim_branch: Optional[str] = None,
) -> Dict[str, Any]:
    """**U 唯一合法正式入口**（G4/finding 36）：断言全部 prereg §1/§4/§5 冻结字面量
    （seeds/bootstrap_b=2000/U seed=20260714/K=8/timeout=900），entrypoint + 冻结字面量 digest
    写入 open bindings（台账）与报告 provenance；U 评估完成后（G5/backlog 41）——若给
    final_packet_path——外锚 final packet（chain_tip + freeze SHA 集 + selection/materialization
    SHA + claim 分支 + U 转移限定词）。任一漂移 → P6FrozenParamError。"""
    k_h0 = int(state_h0.sampler.expected_total)
    assert_u_frozen_params(seeds, bootstrap_b, bootstrap_seed, k_h0, trainer_fit_timeout_seconds)
    digest = frozen_literals_digest(
        "run_u_eval_formal",
        u_frozen_literals(seeds, bootstrap_b, bootstrap_seed, k_h0, trainer_fit_timeout_seconds),
    )
    report = run_u_eval_unfrozen(
        u_loader, state_h0, state_final, judge_cfg, adam_trainer, lstm_trainer, gate,
        seeds=seeds, llm_supplier=llm_supplier, fingerprints=fingerprints, fit_fn=fit_fn,
        rebuild_fn=rebuild_fn, bootstrap_b=bootstrap_b, bootstrap_seed=bootstrap_seed,
        trainer_fit_timeout_seconds=trainer_fit_timeout_seconds, bindings=bindings,
        _entrypoint="run_u_eval_formal", _frozen_digest=digest,
    )
    # G5/backlog 41：U open 事件落账后外锚结果包（证明数字出自正式入口 + 台账 chain_tip）。
    if final_packet_path is not None:
        packet_sha = write_final_packet(
            final_packet_path, chain_tip=gate.chain_tip, manifest_sha=selection_manifest_sha,
            materialization_sha=report["provenance"]["materialization_sha"],
            freeze_shas=freeze_shas, claim_branch=claim_branch,
            u_transfer=bool(report["success_descriptor"]["success"]),
            extra={"u_entrypoint": "run_u_eval_formal", "frozen_literals_digest": digest},
        )
        report["final_packet"] = {"path": str(final_packet_path), "packet_sha256": packet_sha}
    return report
