"""p6/cycle_runner.py — P6 cycle runner（prereg §4 cycle t 步骤 1-7 的机械转写）。

入口 `run_cycle(...) -> CycleResult`：discovery → 归因(probe) → signature → miner →
内部门 → precommit → V 晋升门 → cycle terminal 的完整回路。全部承重环节依赖注入
（adam_trainer / llm_supplier / fit_fn / rebuild_fn / episodes_v_loader / gate），机械
测试不读真实数据、不联网、不 import torch。台账层 = split_manifest.SequentialGate
（本模块只调用其 API，状态机是权威）；判官 = judge_closed_form（每次正式拟合经
c0_runner.paired_judge_fit 双路对拍，prereg §1）；效应换算一律经 p6.metrics 词汇表。

与 prereg §4 cycle t 步骤 1-7 的对应：
  步骤 1  discovery：D_t 上以 H_t 跑 fast path（复用 fast_path._run_one，与
          run_fast_path bit 级同路径）；uid 消费 manifest 先落盘 out_dir
          （write_consumption_manifest）；supplier counterfactual chosen-set（≤3 配置，
          det-only / det+random / 现任，只在 D 上）入 discovery 摘要。
  步骤 2  归因：候选池程序（全 episode kept 池的并集）按行为等价类去重后取 ≤16 个
          probe 变体；对每 (episode, 变体) 用 judge 的 replacement_effects 精确计算
          train/context/joint gain（经 metrics.gain_from_batch_delta）。
  步骤 3  signature：metrics.s1_selector / s2_supply / s3_scope_harm + activate；
          ε/δ_safe/P0 bins 从 c0_freeze 读；S3 cohort 清单 = 4 preset + P0 8 bin；
          activate → None ⇒ abstain：record_cycle_terminal(cycle,"abstain")、V 不开。
  步骤 4  miner：miner.mine(family, evidence, state, c0_bins) ≤3 候选；空 ⇒ abstain 同上。
  步骤 5  内部选择：≤12 次完整配置 paired 重评（apply_edit → 新 state → D 双臂
          fast path（fast_path.paired_*_run 按被测组件分派）→ prepared → per-domain
          paired_judge_fit（对拍判据同 §1）→ train_gain）；选最优；内部门
          train_gain ≥ ε，否则 abstain。
  步骤 6  precommit：gate.record_precommit(cycle, {candidate_edit_sha,
          harness_state_sha, config_digest, materialization_sha, code_sha})。
  步骤 7  V_t 一发：precommit → gate.open_block(f"V{t}") → **再** episodes_v_loader()
          （先 open 后读取；F1/finding 31：一次性 open 是 V 隔离屏障，先于任何 V 读取）；双臂完整评估；
          晋升门六条（evaluate_promotion_gates，见下）全过才 promote；
          gate.record_verdict(f"V{t}", verdict, result_digest) →
          gate.record_cycle_terminal(cycle, 同 verdict)；promote → 返回新 state。
  V 隔离  V 详细指标只写 `out_dir/sealed_V{t}/`（README 声明 H_final 冻结前禁读）；
          CycleResult 只携带 {terminal verdict, 新 state, 内部门/预算账目, discovery
          摘要}，不携带任何 V 数字。

晋升门（prereg §4 步骤 5 六条；边界语义钉死在 evaluate_promotion_gates）：
  ① train_gain ≥ ε（含边界）∧ cluster LCB90 > 0（严格；metrics.cluster_lcb90，
     cluster = 底层 series_uid，seed = 20260711+cycle）；
  ② preset worst-group train_gain LCB90 ≥ −δ_safe（含边界）；
  ③ Adam co-gate：joint 臂 = 直接在两臂 prepared 数据上训 Adam（per-domain、
     3 paired seeds、每 seed 先 H 臂后 edit 臂、域按名升序）；seed 均值 gain ≥ −ε（含边界）；
  ④ scope 类 edit（risk 族）：作用域外 episode prepared artifact 字节级不变——复用
     fast_path.paired_risk_run 的端到端字节级校验；非 scope 类自动过；
  ⑤ 台账干净：precommit 已在、V 首开（状态机自动强制；runner 另行断言绑定一致）；
  ⑥ joint 安全门：overall joint_gain LCB90 ≥ −δ_safe ∧ preset worst-group joint_gain
     LCB90 ≥ −δ_safe（均含边界）。

——冻结释义（实现裁量点；全部机械可测）——
  A. probe 选择（prereg「候选池程序按行为等价类去重后取 ≤16 个 probe 变体」的唯一冻结
     释义）：候选池 = 全 D episode 的 kept 池程序并集；每程序的去重 loss = 其「per-episode
     候选 loss」（释义 B）在全 D episode 上的等权均值；等价类 = metrics.effect_classes 的
     tol 语义（|loss 差| ≤ 1e-9 union-find 成类）；类代表 = 类内 canonical program sha
     最小者；类数 > 16 时按「类内最优 loss 升序」取前 16（effect_classes 返回序即此序）。
     在任一 D episode 上执行失败（prepared=None）的程序退出 probe 候选（无法精确归因），
     数目入 discovery 摘要。
  B. per-episode 候选 loss 口径 = replacement context-effect self loss：series_rmse(基线
     per-domain W, 该 episode 换成候选 prepared 后的评估视图)——与 judge_closed_form.
     replacement_effects(...).context_effect 的 self 路径数值恒等（W 不换、只换 eval 侧），
     不重解即可精确得到；chosen 自身的该 loss ≡ 基线 per-episode loss（直接复用）。
     S1 的 loss_chosen/loss_pool_min 与 S2 的 episode 级等价类数、池上限/det 阶梯上限
     全部用此口径（det 阶梯 3 程序逐 episode 单独执行计入，det 执行失败 = 基础设施
     故障 → P6TechnicalAbort）。
  C. S3 estimand（chosen-vs-raw train gain）：对每 episode 以 raw 视图（不处理、只过
     judge_ingest）做一次 replacement_effects，chosen 的 train gain =
     metrics.harm(metrics.gain_from_batch_delta(train_effect.batch_delta))（把「换回 raw」
     的 gain 取负 = chosen 相对 raw 的 gain）；cohort 内按底层 series 求均值（每 series
     一个值 = 一个 bootstrap cluster）。raw 参照是固定 comparator，不占 probe 预算，
     replacement 调用数照实入成本账。
  D. preset cohort 的 scope（miner 期望的结构）：**preset 成员资格**（F7/finding 37）——
     miner_cohort 携带 preset 名，miner 产出单原子 scope (preset,"==",名)；apply_risk/matches
     对 fingerprint["preset"] 求成员判定。arm 运行前 runner 用 merge_preset_fingerprints 把
     episode.preset 并入 fingerprint（数值特征原样保留），使 preset scope 可求值。旧「C0 中位数
     半平面近似 preset」已删除：它把 preset 误当 snr/missing 数值条件，噪声下 ban 错集。
  E. S3 accused：worst cohort 成员 episode 的 chosen 程序 sha 众数（并列取 sha 升序
     最小；全员 abstain 无 chosen → 无被告 → 按 miner 空处理 = abstain("miner_empty")）。
  F. precommit 的 harness_state_sha = 现任 H_t 的 state.sha()（edit 由 candidate_edit_sha
     标识；resume 语义 = 新 state 可由 (H_t, edit) 重建）。
  G. 门④ 复用 paired_risk_run：其 P6PairingError（含字节级校验）在 V 上被捕获 →
     门④ FAIL → verdict=reject（其余门无从计算，sealed 报告记违规原因）；在 D 内部重评
     上不捕获（结构性篡改 = technical 层，原样 raise）。selector/sampler 族的 pairing
     violation 任何阶段都原样 raise。
  H. 总体 batch = 全 episode 等权均值（跨域直接平均；per-domain shared 拟合不变）；
     bootstrap seed = 20260711+cycle（prereg §4），D 侧 signature 与 V 侧晋升门同源。
  I. sampler 族 evidence 不被 miner 读取（single source = state.sampler）；本 runner
     照传 S2 摘要（零信息损失、零自由度）。
  J. V episodes 的 uid 必须与 D 不相交（virgin 纪律的机械断言，违反 → ValueError）。
  K. 预算（冻结常量，CycleBudget 强制，超限 raise P6BudgetError）：probe ≤16、miner
     候选 ≤3、内部重评 ≤12、LLM HTTP ≤60（llm_supplier 包 CountingLlmSupplier，每次
     调用 = 1 request）、discovery 轮 ≤2（本实现单轮，轮数记账）、counterfactual ≤3 配置。
  L. CycleResult.digest()：对 {cycle, terminal, abstain_reason, state sha, signature,
     internal, discovery(剔除 *_path 键), precommit, cost(剔除 wall_clock_seconds)} 的
     canonical JSON 取 sha256——同输入两次 run_cycle（新 gate/新 out_dir）digest 恒等。

红线：不改任何现有文件；无网络/真 LLM/git；文件写只发生在调用方给的 out_dir；
模块级不 import torch。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .c0_runner import (
    ADAM_FIT_TIMEOUT_SECONDS,
    FROZEN_TRAINER_FIT_TIMEOUT_SECONDS,
    P6Episode,
    P6FrozenParamError,
    _checked_trainer_call,
    _resolve_judge_cfg,
    frozen_literals_digest,
    judge_ingest,
    paired_judge_fit,
)
from .edit_surfaces import compile_proposal
from .fast_path import (
    Candidate,
    P6PairingError,
    _fp_for,
    _run_one,
    det_ladder,
    enrich_candidate,
    merge_preset_fingerprints,
    paired_risk_run,
    paired_sampler_run,
    paired_selector_run,
    prepared_artifact,
    run_fast_path,
)
from .harness_state import P6HarnessState, SamplerSpec, apply_edit, canonical_json
from .loaders import BoundVEpisodes, UnboundEpisodes
from .judge_closed_form import (
    DomainFit,
    SeriesView,
    evaluate,
    replacement_effects,
    series_rmse,
    series_stats,
)
from .materializer import P6TechnicalAbort
from .metrics import (
    DEFAULT_BOOTSTRAP_B,
    activate,
    cluster_lcb90,
    effect_classes,
    gain,
    gain_from_batch_delta,
    harm,
    normalized_headroom,
    s1_selector,
    s2_supply,
    s3_scope_harm,
)
from .miner import MinedCandidate, mine
from .split_manifest import PRECOMMIT_REQUIRED_KEYS, P6StateError

__all__ = [
    "ABSTAIN_INTERNAL_GATE",
    "ABSTAIN_MINER_EMPTY",
    "ABSTAIN_NO_SIGNATURE",
    "BOOTSTRAP_SEED_BASE",
    "COUNTERFACTUAL_CONFIG_BUDGET",
    "CYCLE_PAIRED_SEEDS",
    "CountingLlmSupplier",
    "CycleBudget",
    "CycleResult",
    "DISCOVERY_ROUND_BUDGET",
    "INTERNAL_REEVAL_BUDGET",
    "LLM_REQUEST_BUDGET",
    "MINER_CANDIDATE_BUDGET",
    "P6BudgetError",
    "P6FrozenParamError",
    "PROBE_BUDGET",
    "PRECOMMIT_SIDECAR_SCHEMA",
    "assert_cycle_frozen_params",
    "bootstrap_seed_for",
    "build_cohorts",
    "cycle_frozen_literals",
    "evaluate_promotion_gates",
    "paired_arm_run",
    "precommit_sidecar_path",
    "run_cycle_formal",
    "run_cycle_unfrozen",
    "sealed_v_dir",
    "select_probe_variants",
    "write_consumption_manifest",
]

# ── 冻结常量（prereg §4 预算/协议；释义 K） ─────────────────────────────────
BOOTSTRAP_SEED_BASE = 20260711          # prereg §4：PRNG = default_rng(20260711+cycle)
PROBE_BUDGET = 16                       # probe 变体 ≤16
MINER_CANDIDATE_BUDGET = 3              # miner ≤3 候选/族
INTERNAL_REEVAL_BUDGET = 12             # 内部重评 ≤12
LLM_REQUEST_BUDGET = 60                 # LLM HTTP requests ≤60/cycle
DISCOVERY_ROUND_BUDGET = 2              # discovery 轮 ≤2（本实现单轮）
COUNTERFACTUAL_CONFIG_BUDGET = 3        # supplier counterfactual ≤3 配置
CYCLE_PAIRED_SEEDS: Tuple[int, ...] = (0, 1, 2)   # prereg §1：Adam co-gate paired seeds

ABSTAIN_NO_SIGNATURE = "no_signature"
ABSTAIN_MINER_EMPTY = "miner_empty"
ABSTAIN_INTERNAL_GATE = "internal_gate"

_COHORT_FEATURES: Tuple[str, ...] = ("snr", "missing_rate")   # §3.3 P0 粗 bin 的两特征
_N_BINS = 4                                                    # 四分位 → 4 bin

CONSUMPTION_SCHEMA = "p6-uid-consumption/1"
V_REPORT_SCHEMA = "p6-v-eval/1"

SEALED_README = """# sealed V block — DO NOT READ before H_final freeze

本目录是 prereg_p6 §4 步骤 7 的 V 隔离区（V 详细指标）。

- **H_final 冻结前，任何 miner / runner / 人工不得读取本目录内容**；
- cycle t+1 的合法输入只有：新 harness state + cycle terminal verdict；
- V 拒绝后不得以 bugfix 名义重试（预定义 technical-failure 分支除外，
  且该分支不暴露任何效用数字）；
- 本目录由 cycle_runner 一次性写入，结果包外锚 = 台账 verdict 事件的 result_digest。
"""


class P6BudgetError(RuntimeError):
    """cycle 预算超限（prereg §4 预算行；冻结常量，超限一律 raise）。"""


def sealed_v_dir(out_dir: Any, cycle: int) -> Path:
    """V 隔离区路径（确定性重算；F1/finding 31）：out_dir/sealed_V{cycle}/。

    CycleResult 不携带此路径——本纯函数是定位 sealed 目录的唯一权威（读取受 sealed
    README 纪律约束：H_final 冻结前禁读）。"""
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle not in (1, 2):
        raise ValueError(f"cycle 必须 ∈ {{1, 2}}，得到 {cycle!r}")
    return Path(out_dir) / f"sealed_V{cycle}"


def bootstrap_seed_for(cycle: int) -> int:
    """prereg §4 冻结：bootstrap PRNG seed = 20260711 + cycle（cycle ∈ {1,2}）。"""
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle not in (1, 2):
        raise ValueError(f"cycle 必须 ∈ {{1, 2}}，得到 {cycle!r}")
    return BOOTSTRAP_SEED_BASE + cycle


def assert_cycle_frozen_params(
    seeds: Sequence[int],
    bootstrap_b: int,
    boot_seed: int,
    cycle: int,
    K: int,
    trainer_fit_timeout_seconds: float = FROZEN_TRAINER_FIT_TIMEOUT_SECONDS,
) -> None:
    """run_cycle_formal 断言（prereg §1/§4；值写死字面量，G4）。任一漂移 → P6FrozenParamError。"""
    if tuple(int(s) for s in seeds) != (0, 1, 2):
        raise P6FrozenParamError(f"formal cycle：seeds 必须 == (0, 1, 2)，得到 {tuple(seeds)!r}")
    if int(bootstrap_b) != 2000:
        raise P6FrozenParamError(f"formal cycle：bootstrap_b 必须 == 2000，得到 {bootstrap_b}")
    if int(boot_seed) != 20260711 + int(cycle):
        raise P6FrozenParamError(
            f"formal cycle：bootstrap seed 必须 == 20260711+cycle（{20260711 + int(cycle)}），"
            f"得到 {boot_seed}"
        )
    if int(K) != 8:
        raise P6FrozenParamError(f"formal cycle：K slot 预算必须 == 8，得到 {K}")
    if float(trainer_fit_timeout_seconds) != 900.0:
        raise P6FrozenParamError(
            f"formal cycle：trainer fit 超时必须 == 900.0，得到 {trainer_fit_timeout_seconds!r}"
        )


def _resolve_v_loader_result(
    raw_v: Any, materialization_sha: str, *, require_bound: bool
) -> Sequence[P6Episode]:
    """V loader 返回值收口（G1/finding 32）：只接受 BoundVEpisodes / UnboundEpisodes，
    杜绝裸序列静默路径。BoundVEpisodes 的 materialization_sha 必须 == precommit 绑定值；
    formal（require_bound=True）拒绝 UnboundEpisodes。任一不符 → P6TechnicalAbort。"""
    if isinstance(raw_v, BoundVEpisodes):
        if str(raw_v.materialization_sha) != str(materialization_sha):
            raise P6TechnicalAbort(
                f"V 载入 materialization_sha 与 precommit 绑定不一致："
                f"{raw_v.materialization_sha} != {materialization_sha}"
            )
        return _validate_episodes(raw_v.episodes, "episodes_v")
    if isinstance(raw_v, UnboundEpisodes):
        if require_bound:
            raise P6TechnicalAbort(
                "formal 模式 V loader 必须返回 BoundVEpisodes（manifest-bound）——"
                "UnboundEpisodes 被拒（无静默裸序列路径）"
            )
        return _validate_episodes(raw_v.episodes, "episodes_v")
    raise P6TechnicalAbort(
        f"V loader 必须返回 BoundVEpisodes 或 UnboundEpisodes，得到 "
        f"{type(raw_v).__name__}（G1：无裸序列路径）"
    )


# ── G2：precommit sidecar（冻结候选持久化；resume 不重跑 discovery） ──
PRECOMMIT_SIDECAR_SCHEMA = "p6-precommit-sidecar/1"


def precommit_sidecar_path(out_dir: Any, cycle: int) -> Path:
    """precommit sidecar 路径：out_dir/precommit_payload_cycle{cycle}.json。"""
    return Path(out_dir) / f"precommit_payload_cycle{cycle}.json"


def _build_sidecar_core(
    cycle: int, winner_cand: MinedCandidate, winner_kind: str, best: Mapping[str, Any],
    state_sha: str, config_digest: str, materialization_sha: str, code_sha: str,
    seeds: Sequence[int], signature: Any, internal_native: Any, discovery: Any,
    entrypoint: str, frozen_digest: Optional[str],
) -> Dict[str, Any]:
    return _native({
        "schema": PRECOMMIT_SIDECAR_SCHEMA,
        "cycle": int(cycle),
        "candidate": {
            "proposal_dicts": [dict(pd) for pd in winner_cand.proposal_dicts],
            "edit_kind": str(winner_kind),
        },
        "best": dict(best),
        "harness_state_sha": str(state_sha),
        "config_digest": str(config_digest),
        "materialization_sha": str(materialization_sha),
        "code_sha": str(code_sha),
        "seeds": [int(s) for s in seeds],
        "entrypoint": str(entrypoint),
        "frozen_literals_digest": frozen_digest,
        "signature": signature,
        "internal": internal_native,
        "discovery": discovery,
    })


def _write_precommit_sidecar(out_dir: Any, core: Mapping[str, Any]) -> str:
    """落盘 sidecar（含内嵌 sidecar_sha），返回 sidecar_sha。"""
    sidecar_sha = _sha256_json(core)
    doc = dict(core)
    doc["sidecar_sha"] = sidecar_sha
    path = precommit_sidecar_path(out_dir, core["cycle"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar_sha


def _load_precommit_sidecar(
    out_dir: Any, cycle: int, pending_pc: Mapping[str, Any], state: P6HarnessState,
    config_digest: str, materialization_sha: str, code_sha: str, seeds: Sequence[int],
) -> Tuple[Dict[str, Any], P6HarnessState, str, Dict[str, Any], Any, Any, Any,
           Optional[str], Optional[str]]:
    """resume（G2/finding 34）：从 sidecar 加载冻结候选并逐项验证与 precommit 五元组一致，
    重放 EditOp 重建 winner_state，返回 (best, winner_state, winner_kind, precommit_payload,
    signature, internal, discovery, entrypoint, frozen_digest)。任一不符 → P6TechnicalAbort。
    **不重跑 discovery/miner/内部选择**（不再调用 LLM/采样）。"""
    path = precommit_sidecar_path(out_dir, cycle)
    if not path.exists():
        raise P6TechnicalAbort(f"resume: precommit sidecar 缺失 {path}（无法不经 discovery 恢复）")
    doc = json.loads(path.read_text(encoding="utf-8"))
    embedded = doc.get("sidecar_sha")
    core = {k: v for k, v in doc.items() if k != "sidecar_sha"}
    recomputed = _sha256_json(core)
    if embedded != recomputed:
        raise P6TechnicalAbort(f"resume: sidecar sha 漂移（{embedded} != {recomputed}）")
    pc_pay = dict(pending_pc["payload"])
    if pc_pay.get("sidecar_sha") != embedded:
        raise P6TechnicalAbort("resume: sidecar sha 未被 precommit 事件锚定（台账/sidecar 不一致）")
    if core.get("harness_state_sha") != state.sha():
        raise P6TechnicalAbort("resume: sidecar harness_state_sha 与当前 H 不一致")
    if (core.get("config_digest") != config_digest
            or core.get("materialization_sha") != materialization_sha
            or core.get("code_sha") != code_sha):
        raise P6TechnicalAbort("resume: sidecar 上下文 sha（config/materialization/code）漂移")
    if [int(s) for s in core.get("seeds", [])] != [int(s) for s in seeds]:
        raise P6TechnicalAbort("resume: sidecar seeds 漂移")
    best = dict(core["best"])
    if pc_pay.get("candidate_edit_sha") != best.get("candidate_sha"):
        raise P6TechnicalAbort("resume: sidecar 候选 sha 与 precommit 五元组不一致")
    if pc_pay.get("harness_state_sha") != core.get("harness_state_sha"):
        raise P6TechnicalAbort("resume: sidecar/precommit state sha 不一致")
    winner_state = state
    for pd in core["candidate"]["proposal_dicts"]:
        op = compile_proposal(pd)
        if op is None:
            raise P6TechnicalAbort("resume: sidecar 含不可编译提案")
        winner_state = apply_edit(winner_state, op)
    if winner_state.sha() != best.get("new_state_sha"):
        raise P6TechnicalAbort(
            f"resume: 重建 winner_state sha {winner_state.sha()} != sidecar {best.get('new_state_sha')}"
        )
    winner_kind = str(core["candidate"]["edit_kind"])
    return (best, winner_state, winner_kind, pc_pay, core["signature"],
            core["internal"], core["discovery"],
            core.get("entrypoint"), core.get("frozen_literals_digest"))


# ════════════════════════════ 预算强制（释义 K） ════════════════════════════
@dataclass
class CycleBudget:
    """cycle 预算计数器：全部实耗累加，超冻结上限 raise P6BudgetError。"""

    probe_variants: int = 0
    internal_reevals: int = 0
    llm_requests: int = 0
    discovery_rounds: int = 0
    counterfactual_configs: int = 0

    def _charge(self, field_name: str, n: int, cap: int, label: str) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError(f"{label} 计数必须是非负 int，得到 {n!r}")
        new = getattr(self, field_name) + n
        if new > cap:
            raise P6BudgetError(f"{label} 预算超限：{new} > {cap}（prereg §4 冻结上限）")
        setattr(self, field_name, new)

    def charge_probes(self, n: int) -> None:
        self._charge("probe_variants", n, PROBE_BUDGET, "probe 变体")

    def charge_internal_reeval(self) -> None:
        self._charge("internal_reevals", 1, INTERNAL_REEVAL_BUDGET, "内部重评")

    def charge_llm_request(self) -> None:
        self._charge("llm_requests", 1, LLM_REQUEST_BUDGET, "LLM HTTP request")

    def charge_discovery_round(self) -> None:
        self._charge("discovery_rounds", 1, DISCOVERY_ROUND_BUDGET, "discovery 轮")

    def charge_counterfactual(self) -> None:
        self._charge(
            "counterfactual_configs", 1, COUNTERFACTUAL_CONFIG_BUDGET,
            "supplier counterfactual 配置",
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "probe_variants": self.probe_variants,
            "internal_reevals": self.internal_reevals,
            "llm_requests": self.llm_requests,
            "discovery_rounds": self.discovery_rounds,
            "counterfactual_configs": self.counterfactual_configs,
        }


class CountingLlmSupplier:
    """llm_supplier 计数包装：每次调用 = 1 次 HTTP request（prereg §4 Sampler(b) 语义：
    1 request/episode 返回该 episode 全部 llm slot 候选），经 budget.charge_llm_request()
    计账，第 61 次 raise P6BudgetError。"""

    def __init__(self, supplier: Callable[..., Any], budget: CycleBudget) -> None:
        if supplier is None:
            raise ValueError("CountingLlmSupplier 需要非 None supplier（None 直接不包装）")
        self._supplier = supplier
        self._budget = budget

    def __call__(self, uid: str, state: P6HarnessState, n: int) -> Any:
        self._budget.charge_llm_request()
        return self._supplier(uid, state, n)


# ════════════════════════════ 小工具 ════════════════════════════
def _native(obj: Any) -> Any:
    """递归转 JSON 原生（np 标量/数组/元组 → python 标量/列表）；canonical_json 的前置。"""
    if isinstance(obj, dict):
        return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(_native(obj)).encode("utf-8")).hexdigest()


def _check_nonempty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串，得到 {value!r}")
    return value


def _require_c0_freeze(c0_freeze: Any) -> Tuple[float, float, Dict[str, List[float]]]:
    """从 C0_FREEZE record 读 ε/δ_safe/P0 cutpoints（prereg §3.2/§3.3；缺失/非法 raise）。"""
    if not isinstance(c0_freeze, Mapping):
        raise ValueError(f"c0_freeze 必须是 Mapping（C0_FREEZE record），得到 {type(c0_freeze).__name__}")
    try:
        eps = float(c0_freeze["epsilon"])
        delta_safe = float(c0_freeze["delta_safe"])
        cut = c0_freeze["p0_cutpoints"]
    except KeyError as exc:
        raise ValueError(f"c0_freeze 缺键 {exc}（需要 epsilon/delta_safe/p0_cutpoints）") from None
    if not (np.isfinite(eps) and eps > 0.0):
        raise ValueError(f"c0_freeze.epsilon 必须是正有限数，得到 {eps!r}")
    if not (np.isfinite(delta_safe) and delta_safe > 0.0):
        raise ValueError(f"c0_freeze.delta_safe 必须是正有限数，得到 {delta_safe!r}")
    bins: Dict[str, List[float]] = {}
    for feature in _COHORT_FEATURES:
        if feature not in cut:
            raise ValueError(f"c0_freeze.p0_cutpoints 缺特征 {feature!r}")
        cuts = [float(v) for v in cut[feature]]
        if len(cuts) != 3 or sorted(cuts) != cuts or not all(np.isfinite(c) for c in cuts):
            raise ValueError(
                f"c0_freeze.p0_cutpoints[{feature!r}] 必须是 3 个升序有限四分位 cutpoints，"
                f"得到 {cuts!r}"
            )
        bins[feature] = cuts
    return eps, delta_safe, bins


def _validate_episodes(episodes: Sequence[P6Episode], name: str) -> List[P6Episode]:
    eps = list(episodes)
    if not eps:
        raise ValueError(f"{name} 不能为空")
    uids = [ep.uid for ep in eps]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{name} uid 重复")
    return eps


def _group_by_domain(episodes: Sequence[P6Episode]) -> Dict[str, List[int]]:
    """{config: [全局 episode 索引]}，域按名升序（确定性）。"""
    by_dom: Dict[str, List[int]] = {}
    for i, ep in enumerate(episodes):
        by_dom.setdefault(ep.config, []).append(i)
    return {d: by_dom[d] for d in sorted(by_dom)}


def _views_for_arm(
    episodes: Sequence[P6Episode], chosen_map: Mapping[str, Optional[Candidate]]
) -> List[SeriesView]:
    """arm 的判官视图（对齐 episodes 次序）：prepared_artifact（chosen=None → 原序列的
    部署缺省）→ judge_ingest。执行失败（prepared=None）→ P6TechnicalAbort（u_runner 同口径）。"""
    out: List[SeriesView] = []
    for ep in episodes:
        art = prepared_artifact(chosen_map[ep.uid], ep.history)
        if art is None:
            raise P6TechnicalAbort(
                f"episode {ep.uid} 的 chosen 执行失败（prepared=None）→ technical abort"
            )
        out.append(SeriesView(uid=ep.uid, history=judge_ingest(art), future=ep.future))
    return out


def _fit_domains(
    episodes: Sequence[P6Episode],
    views: Sequence[SeriesView],
    cfg: Mapping[str, Any],
    fit_fn: Optional[Callable[..., DomainFit]],
    rebuild_fn: Optional[Callable[..., DomainFit]],
) -> Tuple[np.ndarray, float, Dict[str, DomainFit], int]:
    """per-domain paired_judge_fit（prereg §1 双路对拍）→ (对齐 episodes 的 per-episode loss,
    全 episode 等权 batch, {domain: DomainFit}, 拟合数)。"""
    by_dom = _group_by_domain(episodes)
    losses = np.empty(len(episodes), dtype=float)
    fits: Dict[str, DomainFit] = {}
    n_fits = 0
    for dom, idxs in by_dom.items():
        fit = paired_judge_fit([views[i] for i in idxs], cfg, fit_fn=fit_fn, rebuild_fn=rebuild_fn)
        n_fits += 1
        for k, i in enumerate(idxs):
            losses[i] = float(fit.per_series_rmse[k])
        fits[dom] = fit
    return losses, float(losses.mean()), fits, n_fits


def _cross_eval(
    episodes: Sequence[P6Episode],
    fits_model: Mapping[str, DomainFit],
    fits_eval: Mapping[str, DomainFit],
) -> Tuple[np.ndarray, float]:
    """交叉评估：用 fits_model 的 W 评 fits_eval 的 stats（per-domain 对齐；批 = 全 episode 等权）。"""
    by_dom = _group_by_domain(episodes)
    losses = np.empty(len(episodes), dtype=float)
    for dom, idxs in by_dom.items():
        rmses, _u = evaluate(fits_model[dom].W, fits_eval[dom].stats)
        for k, i in enumerate(idxs):
            losses[i] = float(rmses[k])
    return losses, float(losses.mean())


# ════════════════════════════ 步骤 1：uid 消费 manifest ════════════════════════════
def write_consumption_manifest(
    path: Any,
    *,
    block: str,
    cycle: int,
    episodes: Sequence[P6Episode],
    state: P6HarnessState,
    K: int,
    code_sha: str,
    materialization_sha: str,
    config_digest: str,
) -> str:
    """uid 级消费 manifest 落盘（prereg §2 运行纪律 / §4 步骤 1）。返回 manifest sha。"""
    doc: Dict[str, Any] = {
        "schema": CONSUMPTION_SCHEMA,
        "block": str(block),
        "cycle": int(cycle),
        "state_sha": state.sha(),
        "state_version": state.version,
        "K": int(K),
        "code_sha": code_sha,
        "materialization_sha": materialization_sha,
        "config_digest": config_digest,
        "n_episodes": len(episodes),
        "episode_uids": sorted(ep.uid for ep in episodes),
        "series_uids": sorted({ep.series_uid for ep in episodes}),
        "configs": sorted({ep.config for ep in episodes}),
    }
    sha = _sha256_json(doc)
    doc["manifest_sha"] = sha
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_native(doc), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha


# ════════════════════════════ 步骤 2：probe 选择（释义 A） ════════════════════════════
def select_probe_variants(
    pool_losses: Mapping[str, float], budget: int = PROBE_BUDGET
) -> List[str]:
    """行为等价类去重 → ≤budget 个 probe 变体（释义 A）。

    等价类 = metrics.effect_classes（tol=1e-9 union-find）；代表 = 类内 canonical sha
    最小者；effect_classes 返回类按类内最小 loss 升序 ⇒ 直接取前 budget 即
    「>16 时按类内最优 loss 升序取前 16」。空池 → []。"""
    classes = effect_classes(dict(pool_losses))
    reps = [min(cls) for cls in classes]
    return reps[: int(budget)]


# ════════════════════════════ 步骤 3：cohort 清单（§3.3） ════════════════════════════
def build_cohorts(
    episodes: Sequence[P6Episode],
    fps: Sequence[Mapping[str, Any]],
    bins: Mapping[str, Sequence[float]],
) -> List[Dict[str, Any]]:
    """§3.3 冻结 cohort 清单 = {观测到的 preset} ∪ {snr, missing_rate 各 4 个 C0 四分位 bin}。

    每项：{"cohort_id", "member_idx"（episode 全局索引）, "miner_cohort"（miner risk 族期望
    的两种冻结形之一：preset 形带 preset 名（成员资格 scope，F7/finding 37——不再由 C0 中位数
    半平面近似）；bin 形带 lo/hi）}。bin 归属 = bisect_right(cutpoints, v)（左闭右开，与
    miner._first_dominant_bin 一致）；缺该特征的 episode 不入任何 bin cohort。"""
    if len(fps) != len(episodes):
        raise ValueError(f"fps（{len(fps)}）与 episodes（{len(episodes)}）长度不一致")
    cohorts: List[Dict[str, Any]] = []
    for p in sorted({ep.preset for ep in episodes}):
        idx = [i for i, ep in enumerate(episodes) if ep.preset == p]
        cohorts.append(
            {
                "cohort_id": f"preset:{p}",
                "member_idx": idx,
                "miner_cohort": {
                    "cohort_id": f"preset:{p}",
                    "preset": p,
                },
            }
        )
    for feature in _COHORT_FEATURES:
        cuts = [float(c) for c in bins[feature]]
        for b_idx in range(_N_BINS):
            idx = [
                i
                for i, fp in enumerate(fps)
                if feature in fp and bisect_right(cuts, float(fp[feature])) == b_idx
            ]
            lo = None if b_idx == 0 else cuts[b_idx - 1]
            hi = None if b_idx == _N_BINS - 1 else cuts[b_idx]
            cid = f"bin:{feature}:{b_idx}"
            cohorts.append(
                {
                    "cohort_id": cid,
                    "member_idx": idx,
                    "miner_cohort": {
                        "cohort_id": cid,
                        "bin": {"feature": feature, "lo": lo, "hi": hi},
                    },
                }
            )
    return cohorts


# ════════════════════════════ 步骤 5/7：双臂 paired run 分派 ════════════════════════════
def paired_arm_run(
    episodes: Sequence[P6Episode],
    state_a: P6HarnessState,
    state_b: P6HarnessState,
    edit_kind: str,
    K: int,
    llm_supplier: Optional[Callable[..., Any]],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
    *,
    on_scope_violation: str = "raise",
) -> Dict[str, Any]:
    """双臂 fast path：按被测组件分派 fast_path.paired_*_run（配对/池不变性 assert 复用）。

    edit_kind ∈ {selector_patch, sampler_patch, risk_rule_patch}。risk 族的
    P6PairingError（含 prereg §4 门④ 端到端字节级校验）：
      - on_scope_violation="gate"（V 晋升门语境）→ 捕获，返回 gate4_ok=False + 违规原因；
      - on_scope_violation="raise"（D 内部重评语境）→ 原样 raise（结构性篡改 = technical 层）。
    selector/sampler 族的 pairing violation 一律原样 raise（门④ 对非 scope 类自动过）。
    返回 {"chosen_a", "chosen_b", "gate4_ok", "gate4_note", "out_of_scope_verified"}。"""
    if on_scope_violation not in ("raise", "gate"):
        raise ValueError(f"on_scope_violation 须为 'raise'|'gate'，得到 {on_scope_violation!r}")
    pairs = [(ep.uid, ep.history) for ep in episodes]
    if edit_kind == "selector_patch":
        res = paired_selector_run(pairs, state_a, state_b, K, llm_supplier, fingerprints)
        return {
            "chosen_a": dict(res["A"]), "chosen_b": dict(res["B"]),
            "gate4_ok": True, "gate4_note": "non-scope edit（selector）：门④ 自动过",
            "out_of_scope_verified": None,
        }
    if edit_kind == "sampler_patch":
        res = paired_sampler_run(pairs, state_a, state_b, K, llm_supplier, fingerprints)
        return {
            "chosen_a": dict(res["A"]["chosen"]), "chosen_b": dict(res["B"]["chosen"]),
            "gate4_ok": True, "gate4_note": "non-scope edit（sampler）：门④ 自动过",
            "out_of_scope_verified": None,
        }
    if edit_kind == "risk_rule_patch":
        try:
            res = paired_risk_run(pairs, state_a, state_b, K, llm_supplier, fingerprints)
        except P6PairingError as exc:
            if on_scope_violation == "gate":
                return {
                    "chosen_a": None, "chosen_b": None,
                    "gate4_ok": False, "gate4_note": f"门④ FAIL：{exc}",
                    "out_of_scope_verified": None,
                }
            raise
        return {
            "chosen_a": dict(res["A"]["chosen"]), "chosen_b": dict(res["B"]["chosen"]),
            "gate4_ok": True,
            "gate4_note": "risk 族门④：作用域外 prepared artifact 字节级校验通过",
            "out_of_scope_verified": [str(u) for u in res["out_of_scope_verified"]],
        }
    raise ValueError(f"未知 edit_kind {edit_kind!r}（可分派：selector/sampler/risk_rule patch）")


def _apply_candidate(
    state: P6HarnessState, cand: MinedCandidate
) -> Tuple[P6HarnessState, str]:
    """候选 → 新 state（bundle 全量按序 apply_edit）+ edit_kind（bundle 必须同 kind）。"""
    kinds = {str(pd.get("kind")) for pd in cand.proposal_dicts}
    if len(kinds) != 1:
        raise ValueError(f"候选 {cand.recipe_id!r} bundle 跨 kind {sorted(kinds)}（冻结配方不产生此形态）")
    new_state = state
    for pd in cand.proposal_dicts:
        op = compile_proposal(pd)
        if op is None:
            raise ValueError(f"候选 {cand.recipe_id!r} 含不可编译提案 kind={pd.get('kind')!r}")
        new_state = apply_edit(new_state, op)
    return new_state, kinds.pop()


# ════════════════════════════ 步骤 7：晋升门（六条；边界钉死） ════════════════════════════
def evaluate_promotion_gates(
    train_gains: Sequence[float],
    joint_gains: Sequence[float],
    clusters: Sequence[Any],
    presets: Sequence[str],
    *,
    eps: float,
    delta_safe: float,
    adam_gains: Sequence[float],
    risk_scope_ok: bool,
    ledger_ok: bool,
    b: int = DEFAULT_BOOTSTRAP_B,
    seed: int,
) -> Dict[str, Any]:
    """prereg §4 步骤 5 晋升门六条的纯函数（全过才 promote）。

    边界语义（冻结）：① train_gain ≥ ε（含边界）∧ cluster LCB90 > 0（严格）；
    ② preset worst-group train LCB90 ≥ −δ_safe（含边界）；③ Adam seed 均值 gain ≥ −ε
    （含边界）；④=risk_scope_ok、⑤=ledger_ok（布尔直通，语义在调用方）；
    ⑥ overall joint LCB90 ≥ −δ_safe ∧ preset worst-group joint LCB90 ≥ −δ_safe（均含边界）。
    LCB = metrics.cluster_lcb90（cluster = 底层 series_uid；同一 seed 贯穿全部 LCB）。"""
    tg = [float(x) for x in train_gains]
    jg = [float(x) for x in joint_gains]
    cl = list(clusters)
    pr = [str(p) for p in presets]
    if not (len(tg) == len(jg) == len(cl) == len(pr)) or not tg:
        raise ValueError("train_gains/joint_gains/clusters/presets 必须等长且非空")
    ag = [float(x) for x in adam_gains]
    if not ag:
        raise ValueError("adam_gains 不能为空（Adam co-gate 是硬门）")
    e = float(eps)
    d = float(delta_safe)

    overall_train = float(np.mean(tg))
    lcb_train = cluster_lcb90(tg, cl, int(b), seed=int(seed))
    gate1 = {
        "train_gain": overall_train, "lcb90": lcb_train, "eps": e,
        "pass": bool(overall_train >= e and lcb_train > 0.0),
    }

    def _preset_worst(values: List[float]) -> Tuple[Dict[str, Any], float]:
        per: Dict[str, Any] = {}
        worst: Optional[float] = None
        for p in sorted(set(pr)):
            idx = [i for i, q in enumerate(pr) if q == p]
            lcb = cluster_lcb90([values[i] for i in idx], [cl[i] for i in idx],
                                int(b), seed=int(seed))
            per[p] = {"lcb90": lcb, "n": len(idx)}
            worst = lcb if worst is None else min(worst, lcb)
        assert worst is not None
        return per, worst

    per_t, worst_t = _preset_worst(tg)
    gate2 = {
        "per_preset": per_t, "worst_lcb90": worst_t, "threshold": -d,
        "pass": bool(worst_t >= -d),
    }

    adam_mean = float(np.mean(ag))
    gate3 = {
        "per_seed_gain": ag, "mean_gain": adam_mean, "threshold": -e,
        "pass": bool(adam_mean >= -e),
    }

    gate4 = {"pass": bool(risk_scope_ok)}
    gate5 = {"pass": bool(ledger_ok)}

    overall_joint = float(np.mean(jg))
    lcb_joint = cluster_lcb90(jg, cl, int(b), seed=int(seed))
    per_j, worst_j = _preset_worst(jg)
    gate6 = {
        "joint_gain": overall_joint, "overall_lcb90": lcb_joint,
        "per_preset": per_j, "worst_lcb90": worst_j, "threshold": -d,
        "pass": bool(lcb_joint >= -d and worst_j >= -d),
    }

    promote = bool(
        gate1["pass"] and gate2["pass"] and gate3["pass"]
        and gate4["pass"] and gate5["pass"] and gate6["pass"]
    )
    return {
        "gate1_train": gate1, "gate2_preset_train": gate2, "gate3_adam_cogate": gate3,
        "gate4_scope_bytes": gate4, "gate5_ledger": gate5, "gate6_joint_safety": gate6,
        "promote": promote,
    }


# ════════════════════════════ CycleResult ════════════════════════════
@dataclass(frozen=True)
class CycleResult:
    """cycle 结果（prereg §4 步骤 7 V 隔离：**不携带任何 V 数字、也不携带 sealed 目录路径**
    ——V 详细指标只在 out_dir/sealed_V{t}/v_report.json；本载体只有 terminal verdict、
    新 state、D 侧 signature/内部门/预算账目与 discovery 摘要）。cycle t+1 的合法输入只有
    新 state + terminal verdict。sealed 目录路径由 `sealed_v_dir(out_dir, cycle)` 纯函数
    确定性重算（F1/finding 31：不经 CycleResult 泄漏，读取受 sealed README 纪律约束）。
    digest() 语义见模块 docstring 释义 L。"""

    cycle: int
    terminal: str                       # promote | reject | abstain
    abstain_reason: Optional[str]       # no_signature | miner_empty | internal_gate | None
    new_state: P6HarnessState
    state_changed: bool
    signature: Dict[str, Any]
    internal: Dict[str, Any]
    discovery: Dict[str, Any]
    precommit: Optional[Dict[str, Any]]
    cost: Dict[str, Any]
    entrypoint: str                     # G4：产出此结果的入口（run_cycle_formal / run_cycle_unfrozen）
    frozen_literals_digest: Optional[str]   # G4：正式入口冻结字面量指纹（unfrozen 时为 None）

    def digest(self) -> str:
        payload = {
            "cycle": self.cycle,
            "terminal": self.terminal,
            "abstain_reason": self.abstain_reason,
            "state_sha": self.new_state.sha(),
            "state_changed": self.state_changed,
            "signature": self.signature,
            "internal": self.internal,
            "discovery": {
                k: v for k, v in self.discovery.items() if not str(k).endswith("_path")
            },
            # sidecar_sha 是 out_dir 位置相关的锚（sidecar 含 discovery 路径），从 digest 剔除，
            # 使两次全新 run（不同 out_dir）digest 恒等（G2 引入 sidecar 后的确定性修复）。
            "precommit": (
                {k: v for k, v in self.precommit.items() if k != "sidecar_sha"}
                if self.precommit is not None else None
            ),
            "cost": {k: v for k, v in self.cost.items()
                     if k not in ("wall_clock_seconds", "resumed")},
            "entrypoint": self.entrypoint,
            "frozen_literals_digest": self.frozen_literals_digest,
        }
        return _sha256_json(payload)


# ════════════════════════════ 步骤 7 收口（fresh + resume 共用） ════════════════════════════
@dataclass
class _FinalizeCtx:
    """步骤 7 共用配置上下文（fresh 与 resume 两路同用；避免 ~20 参数散落）。"""
    cfg: Mapping[str, Any]
    seeds: Tuple[int, ...]
    b: int
    boot_seed: int
    eps_threshold: float
    delta_safe: float
    K: int
    adam_trainer: Callable[..., Any]
    wrapped_llm: Any
    fingerprints: Any
    fit_fn: Any
    rebuild_fn: Any
    trainer_fit_timeout_seconds: float
    code_sha: str
    materialization_sha: str
    config_digest: str
    out: Path
    t_start: float
    budget: CycleBudget
    require_bound_v: bool


def _finalize_v(
    fctx: _FinalizeCtx, cycle: int, state: P6HarnessState, episodes: Sequence[P6Episode],
    episodes_v_loader: Callable[[], Any], gate: Any, winner_state: P6HarnessState,
    winner_kind: str, best: Mapping[str, Any], precommit_payload: Mapping[str, Any],
    signature_summary: Any, internal_native: Any, discovery_summary: Any,
    counters: Dict[str, int], entrypoint: str, frozen_digest: Optional[str], resumed: bool,
) -> CycleResult:
    """步骤 7：open V（一次性，loader 之前）→ 加载并收口 V loader（G1 严格载体）→ 双臂评估
    → 晋升门 → verdict/terminal → CycleResult。fresh 与 resume 两路共用（resume 时 D 侧摘要
    来自 sidecar、V 侧重新评估）。"""
    out = fctx.out
    cfg = fctx.cfg
    seeds = fctx.seeds
    n_adam_fits_local = 0
    v_block = f"V{cycle}"
    # F1/finding 31 + F4：precommit → open V（若已 pending-open 则不重复 open）→ 再 loader。
    if gate.pending_open(v_block) is None:
        gate.open_block(v_block)
    raw_v = episodes_v_loader()
    # G1/finding 32：V loader 返回值必须是 BoundVEpisodes / UnboundEpisodes（无裸序列静默路径）。
    episodes_v = _resolve_v_loader_result(
        raw_v, fctx.materialization_sha, require_bound=fctx.require_bound_v
    )
    d_uids = {ep.uid for ep in episodes}
    overlap = sorted(d_uids & {ep.uid for ep in episodes_v})
    if overlap:
        raise ValueError(f"V episodes 与 D 重叠（virgin 纪律违约）：{overlap[:5]}")
    v_manifest_path = out / f"consumed_uids_cycle{cycle}_V.json"
    write_consumption_manifest(
        v_manifest_path, block=v_block, cycle=cycle, episodes=episodes_v, state=state,
        K=fctx.K, code_sha=fctx.code_sha, materialization_sha=fctx.materialization_sha,
        config_digest=fctx.config_digest,
    )

    sealed_dir = sealed_v_dir(out, cycle)
    fingerprints_run_v = merge_preset_fingerprints(episodes_v, fctx.fingerprints)   # F7
    pr_v = paired_arm_run(
        episodes_v, state, winner_state, winner_kind, fctx.K, fctx.wrapped_llm,
        fingerprints_run_v, on_scope_violation="gate",
    )
    if not pr_v["gate4_ok"]:
        report = {
            "schema": V_REPORT_SCHEMA, "block": v_block, "cycle": cycle,
            "verdict": "reject",
            "edit": {"candidate_sha": best["candidate_sha"], "recipe_id": best["recipe_id"],
                     "kind": winner_kind, "new_state_sha": winner_state.sha()},
            "gate4": {"ok": False, "note": pr_v["gate4_note"]},
            "gates": {"gate4_scope_bytes": {"pass": False}, "promote": False},
            "precommit": dict(precommit_payload),
            "n_episodes": len(episodes_v),
        }
        verdict = "reject"
    else:
        views_v_a = _views_for_arm(episodes_v, pr_v["chosen_a"])
        views_v_b = _views_for_arm(episodes_v, pr_v["chosen_b"])
        vloss_00, vu_00, vfits_a, nf_a = _fit_domains(episodes_v, views_v_a, cfg, fctx.fit_fn, fctx.rebuild_fn)
        vloss_11, vu_11, vfits_b, nf_b = _fit_domains(episodes_v, views_v_b, cfg, fctx.fit_fn, fctx.rebuild_fn)
        counters["judge_fits"] = counters.get("judge_fits", 0) + nf_a + nf_b
        vloss_10, vu_10 = _cross_eval(episodes_v, vfits_b, vfits_a)   # train
        vloss_01, vu_01 = _cross_eval(episodes_v, vfits_a, vfits_b)   # context（披露）
        train_gains_v = [gain(float(a), float(t)) for a, t in zip(vloss_00, vloss_10)]
        joint_gains_v = [gain(float(a), float(j)) for a, j in zip(vloss_00, vloss_11)]
        clusters_v = [ep.series_uid for ep in episodes_v]
        presets_v = [ep.preset for ep in episodes_v]

        by_dom_v = _group_by_domain(episodes_v)
        adam_gains: List[float] = []
        adam_per_seed: List[Dict[str, Any]] = []
        for s in seeds:
            la = np.empty(len(episodes_v), dtype=float)
            lb = np.empty(len(episodes_v), dtype=float)
            for dom, idxs in by_dom_v.items():
                la[idxs] = _checked_trainer_call(
                    fctx.adam_trainer, [views_v_a[i] for i in idxs], s,
                    fctx.trainer_fit_timeout_seconds, f"AdamCoGate[{dom}/h/seed={s}]",
                )
                n_adam_fits_local += 1
                lb[idxs] = _checked_trainer_call(
                    fctx.adam_trainer, [views_v_b[i] for i in idxs], s,
                    fctx.trainer_fit_timeout_seconds, f"AdamCoGate[{dom}/edit/seed={s}]",
                )
                n_adam_fits_local += 1
            g_s = gain(float(la.mean()), float(lb.mean()))
            adam_gains.append(g_s)
            adam_per_seed.append({"seed": int(s), "loss_h": float(la.mean()),
                                  "loss_edit": float(lb.mean()), "joint_gain": g_s})

        pc = gate.precommit(cycle)
        ledger_ok = bool(
            pc is not None
            and {k: pc["payload"].get(k) for k in PRECOMMIT_REQUIRED_KEYS}
            == {k: precommit_payload[k] for k in PRECOMMIT_REQUIRED_KEYS}
            and gate.state(v_block) == "open"
            and gate.pending_open(v_block) is not None
        )
        gates = evaluate_promotion_gates(
            train_gains_v, joint_gains_v, clusters_v, presets_v,
            eps=fctx.eps_threshold, delta_safe=fctx.delta_safe, adam_gains=adam_gains,
            risk_scope_ok=True, ledger_ok=ledger_ok, b=fctx.b, seed=fctx.boot_seed,
        )
        verdict = "promote" if gates["promote"] else "reject"
        report = {
            "schema": V_REPORT_SCHEMA, "block": v_block, "cycle": cycle,
            "verdict": verdict,
            "edit": {"candidate_sha": best["candidate_sha"], "recipe_id": best["recipe_id"],
                     "kind": winner_kind, "new_state_sha": winner_state.sha()},
            "protocol": {
                "judge": dict(cfg), "eps": fctx.eps_threshold, "delta_safe": fctx.delta_safe,
                "bootstrap": {"b": fctx.b, "seed": fctx.boot_seed}, "paired_seeds": list(seeds),
                "K": fctx.K, "state_h_sha": state.sha(),
            },
            "n_episodes": len(episodes_v),
            "arms": {"h": {"utility": vu_00}, "edit": {"utility": vu_11}},
            "effects": {
                "train": {"overall_gain": gain(vu_00, vu_10)},
                "context": {"overall_gain": gain(vu_00, vu_01)},
                "joint": {"overall_gain": gain(vu_00, vu_11)},
            },
            "per_episode": [
                {
                    "uid": ep.uid, "series_uid": ep.series_uid, "preset": ep.preset,
                    "loss_00": float(vloss_00[i]), "loss_10": float(vloss_10[i]),
                    "loss_01": float(vloss_01[i]), "loss_11": float(vloss_11[i]),
                    "train_gain": train_gains_v[i], "joint_gain": joint_gains_v[i],
                }
                for i, ep in enumerate(episodes_v)
            ],
            "adam_cogate": {"per_seed": adam_per_seed,
                            "mean_gain": float(np.mean(adam_gains))},
            "gate4": {"ok": True, "note": pr_v["gate4_note"],
                      "out_of_scope_verified": pr_v["out_of_scope_verified"]},
            "gates": gates,
            "precommit": dict(precommit_payload),
            "consumed_manifest_path": str(v_manifest_path),
        }

    counters["adam_fits"] = counters.get("adam_fits", 0) + n_adam_fits_local
    sealed_dir.mkdir(parents=True, exist_ok=True)
    result_digest = _sha256_json(report)
    sealed_doc = dict(_native(report))
    sealed_doc["result_digest"] = result_digest
    (sealed_dir / "v_report.json").write_text(
        json.dumps(sealed_doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (sealed_dir / "README.md").write_text(SEALED_README, encoding="utf-8")

    gate.record_verdict(v_block, verdict, result_digest=result_digest)
    gate.record_cycle_terminal(cycle, verdict)

    cost = dict(fctx.budget.as_dict())
    cost.update({
        "replacement_effect_calls": counters.get("repl_calls", 0),
        "surrogate_evals": counters.get("surrogate", 0),
        "judge_paired_fits": counters.get("judge_fits", 0),
        "adam_cogate_fits": counters.get("adam_fits", 0),
        "resumed": bool(resumed),
        "wall_clock_seconds": round(time.perf_counter() - fctx.t_start, 3),
    })
    promoted = verdict == "promote"
    return CycleResult(
        cycle=cycle, terminal=verdict, abstain_reason=None,
        new_state=winner_state if promoted else state, state_changed=promoted,
        signature=signature_summary, internal=internal_native, discovery=discovery_summary,
        precommit=_native(dict(precommit_payload)), cost=cost,
        entrypoint=entrypoint, frozen_literals_digest=frozen_digest,
    )


# ════════════════════════════ run_cycle ════════════════════════════
def run_cycle_unfrozen(
    cycle: int,
    episodes_d: Sequence[P6Episode],
    episodes_v_loader: Callable[[], Any],
    state: P6HarnessState,
    gate: Any,
    c0_freeze: Mapping[str, Any],
    *,
    adam_trainer: Callable[[Sequence[SeriesView], int], Any],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
    llm_supplier: Optional[Callable[..., Any]] = None,
    judge_cfg: Optional[Mapping[str, Any]],
    seeds: Sequence[int] = CYCLE_PAIRED_SEEDS,
    out_dir: Any,
    code_sha: str,
    materialization_sha: str,
    config_digest: str,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
    trainer_fit_timeout_seconds: float = ADAM_FIT_TIMEOUT_SECONDS,
    _require_bound_v: bool = False,
    _entrypoint: Optional[str] = None,
    _frozen_digest: Optional[str] = None,
) -> CycleResult:
    """P6 cycle t 完整回路核心（**测试专用；正式运行禁止调用——用 run_cycle_formal**）。

    episodes_v_loader 只在步骤 7（precommit/open 之后）被调用一次；其返回值必须是
    BoundVEpisodes / UnboundEpisodes（G1：无裸序列静默路径）。
    **崩溃恢复（G2/finding 34）**：入口检测到本 cycle 已有 pending precommit → 从 sidecar 加载
    冻结候选、逐项验证、重放 EditOp 重建 winner → **跳过步骤 1-6（不重跑 discovery/LLM/采样）**
    直达步骤 7。abstain 三条路（无 signature / miner 空 / 内部门不过）都记 cycle terminal 且 V 不开。
    冻结字面量断言不在此（在 run_cycle_formal）；`_*` 内部参数由正式入口注入。
    """
    t_start = time.perf_counter()
    if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle not in (1, 2):
        raise ValueError(f"cycle 必须 ∈ {{1, 2}}，得到 {cycle!r}")
    if not isinstance(state, P6HarnessState):
        raise ValueError(f"state 必须是 P6HarnessState，得到 {type(state).__name__}")
    code_sha = _check_nonempty_str(code_sha, "code_sha")
    materialization_sha = _check_nonempty_str(materialization_sha, "materialization_sha")
    config_digest = _check_nonempty_str(config_digest, "config_digest")
    episodes = _validate_episodes(episodes_d, "episodes_d")
    eps_threshold, delta_safe, bins = _require_c0_freeze(c0_freeze)
    cfg = _resolve_judge_cfg(judge_cfg)
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("seeds 不能为空")
    b = int(bootstrap_b)
    boot_seed = bootstrap_seed_for(cycle)

    # —— 状态机前置（等价检查；record_* 处状态机仍是权威） ——
    if gate.cycle_terminal(cycle) is not None:
        raise P6StateError(
            f"cycle{cycle} 已 terminal（{gate.cycle_terminal(cycle)!r}）：不得重跑"
        )
    if cycle == 2 and gate.cycle_terminal(1) is None:
        raise P6StateError(
            "cycle2 需 cycle1 terminal 已记录（prereg §4 步骤 6：cycle 严格顺序）"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    budget = CycleBudget()
    wrapped_llm = (
        CountingLlmSupplier(llm_supplier, budget) if llm_supplier is not None else None
    )
    K = int(state.sampler.expected_total)
    entrypoint = _entrypoint or "run_cycle_unfrozen"
    fctx = _FinalizeCtx(
        cfg=cfg, seeds=seeds, b=b, boot_seed=boot_seed, eps_threshold=eps_threshold,
        delta_safe=delta_safe, K=K, adam_trainer=adam_trainer, wrapped_llm=wrapped_llm,
        fingerprints=fingerprints, fit_fn=fit_fn, rebuild_fn=rebuild_fn,
        trainer_fit_timeout_seconds=trainer_fit_timeout_seconds, code_sha=code_sha,
        materialization_sha=materialization_sha, config_digest=config_digest, out=out,
        t_start=t_start, budget=budget, require_bound_v=_require_bound_v,
    )

    # ════ G2/finding 34：崩溃恢复——pending precommit → 从 sidecar 恢复、跳过步骤 1-6 ════
    pending_pc = gate.precommit(cycle)
    if pending_pc is not None and gate.cycle_terminal(cycle) is None:
        (best_r, winner_state_r, winner_kind_r, precommit_payload_r, sig_r, intern_r, disc_r,
         ep_r, fd_r) = _load_precommit_sidecar(
            out, cycle, pending_pc, state, config_digest, materialization_sha, code_sha, seeds
        )
        counters_r = {"judge_fits": 0, "adam_fits": 0, "repl_calls": 0, "surrogate": 0}
        # 恢复结果沿用 sidecar 记录的原始入口/字面量指纹（决策出自该处）。
        return _finalize_v(
            fctx, cycle, state, episodes, episodes_v_loader, gate, winner_state_r,
            winner_kind_r, best_r, precommit_payload_r, sig_r, intern_r, disc_r,
            counters_r, ep_r or entrypoint, fd_r, resumed=True,
        )

    n_ep = len(episodes)
    clusters_d = [ep.series_uid for ep in episodes]
    # preset 成员资格 scope（F7）：arm/discovery 的 apply_risk 需 fingerprint 携带 preset。
    # 数值特征 fps（cohort/S3 evidence）仍用无 preset 的原始 fingerprints（不引 provenance 漂移）。
    fingerprints_run = merge_preset_fingerprints(episodes, fingerprints)
    n_judge_fits = 0
    n_repl_calls = 0
    n_surrogate = 0
    n_adam_fits = 0

    # ════ 步骤 1：discovery（uid 消费 manifest 先落盘） ════
    budget.charge_discovery_round()
    d_manifest_path = out / f"consumed_uids_cycle{cycle}_D.json"
    write_consumption_manifest(
        d_manifest_path, block=f"D{cycle}", cycle=cycle, episodes=episodes, state=state,
        K=K, code_sha=code_sha, materialization_sha=materialization_sha,
        config_digest=config_digest,
    )
    runs = {
        ep.uid: _run_one(ep.uid, ep.history, state, K, wrapped_llm, fingerprints_run)
        for ep in episodes
    }
    chosen_h: Dict[str, Optional[Candidate]] = {uid: r.chosen for uid, r in runs.items()}
    views_h = _views_for_arm(episodes, chosen_h)
    loss_00, u_00, fits_h, nf = _fit_domains(episodes, views_h, cfg, fit_fn, rebuild_fn)
    n_judge_fits += nf
    fps = [dict(_fp_for(ep.uid, ep.history, fingerprints)) for ep in episodes]
    by_dom = _group_by_domain(episodes)
    dom_views: Dict[str, List[SeriesView]] = {
        d: [views_h[i] for i in idxs] for d, idxs in by_dom.items()
    }
    local_idx: Dict[int, Tuple[str, int]] = {}
    for d, idxs in by_dom.items():
        for k, i in enumerate(idxs):
            local_idx[i] = (d, k)

    # supplier counterfactual chosen-set（步骤 10；≤3 配置、只在 D）
    counterfactual = _counterfactual_chosen_sets(
        episodes, state, K, wrapped_llm, fingerprints_run, budget, chosen_h
    )

    # ════ 步骤 2：归因——per-episode 候选 loss（释义 B）+ probe 选择（释义 A）+ 精确归因 ════
    union: Dict[str, Candidate] = {}
    for ep in episodes:
        for c in runs[ep.uid].kept:
            union.setdefault(c.sha, c)
    det_progs = det_ladder()
    det_shas = {c.sha for c in det_progs}
    all_progs: Dict[str, Candidate] = dict(union)
    for c in det_progs:
        all_progs.setdefault(c.sha, c)

    ell: Dict[Tuple[int, str], float] = {}
    ingested: Dict[Tuple[int, str], np.ndarray] = {}
    failed_shas: set = set()
    for i, ep in enumerate(episodes):
        ch = chosen_h[ep.uid]
        for sha in sorted(all_progs):
            if ch is not None and sha == ch.sha:
                ell[(i, sha)] = float(loss_00[i])            # chosen 的候选 loss ≡ 基线（释义 B）
                ingested[(i, sha)] = views_h[i].history
                continue
            art = prepared_artifact(all_progs[sha], ep.history)
            if art is None:
                if sha in det_shas:
                    raise P6TechnicalAbort(
                        f"det 阶梯程序 {sha} 在 episode {ep.uid} 执行失败 → technical abort"
                    )
                failed_shas.add(sha)
                continue
            ing = judge_ingest(art)
            st = series_stats(
                SeriesView(uid=ep.uid, history=ing, future=ep.future),
                stride=cfg["stride"], window_cap=cfg["window_cap"],
            )
            ell[(i, sha)] = float(series_rmse(fits_h[ep.config].W, st))
            ingested[(i, sha)] = ing
            n_surrogate += 1

    probe_pool_losses = {
        sha: float(np.mean([ell[(i, sha)] for i in range(n_ep)]))
        for sha in union
        if sha not in failed_shas and all((i, sha) in ell for i in range(n_ep))
    }
    probe_shas = select_probe_variants(probe_pool_losses, PROBE_BUDGET)
    budget.charge_probes(len(probe_shas))

    selector_rows: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes):
        d, k = local_idx[i]
        for sha in probe_shas:
            view_e = SeriesView(uid=ep.uid, history=ingested[(i, sha)], future=ep.future)
            re = replacement_effects(dom_views[d], k, view_e, **cfg)
            n_repl_calls += 1
            selector_rows.append(
                {
                    "episode_uid": ep.uid,
                    "features": dict(enrich_candidate(all_progs[sha], ep.history).features),
                    "train_gain": gain_from_batch_delta(re.train_effect.batch_delta),
                }
            )
    attribution_path = out / f"attribution_cycle{cycle}_D.json"
    attribution_path.write_text(
        json.dumps(
            _native({"schema": "p6-attribution/1", "cycle": cycle,
                     "probe_shas": probe_shas, "rows": selector_rows}),
            sort_keys=True, ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # S3 estimand：chosen-vs-raw train gain（释义 C；raw 参照不占 probe 预算）
    chosen_vs_raw: List[float] = []
    for i, ep in enumerate(episodes):
        d, k = local_idx[i]
        raw_view = SeriesView(uid=ep.uid, history=judge_ingest(ep.history), future=ep.future)
        re = replacement_effects(dom_views[d], k, raw_view, **cfg)
        n_repl_calls += 1
        chosen_vs_raw.append(harm(gain_from_batch_delta(re.train_effect.batch_delta)))

    # ════ 步骤 3：signature（metrics 冻结实现 + activate） ════
    per_ep_s1: List[Dict[str, float]] = []
    class_counts: List[int] = []
    pool_min = np.empty(n_ep, dtype=float)
    det_min = np.empty(n_ep, dtype=float)
    for i, ep in enumerate(episodes):
        losses_i = {
            c.sha: ell[(i, c.sha)] for c in runs[ep.uid].kept if (i, c.sha) in ell
        }
        pmin = min(losses_i.values()) if losses_i else float(loss_00[i])
        pool_min[i] = min(pmin, float(loss_00[i]))           # chosen ∈ 池（regret ≥ 0 保证）
        per_ep_s1.append({"loss_chosen": float(loss_00[i]), "loss_pool_min": float(pool_min[i])})
        class_counts.append(
            len(effect_classes(losses_i)) if losses_i else 1
        )
        det_min[i] = min(ell[(i, sha)] for sha in det_shas)
    s1 = dict(s1_selector(per_ep_s1, clusters_d, eps_threshold, b, seed=boot_seed))
    s1["headroom"] = normalized_headroom(s1["regret_mean"], eps_threshold)
    s2 = dict(
        s2_supply(
            class_counts,
            gain(u_00, float(pool_min.mean())),
            gain(u_00, float(det_min.mean())),
            eps_threshold,
        )
    )
    s2["headroom"] = max(
        -normalized_headroom(s2["mean_classes"], 2.0),
        normalized_headroom(-s2["ceiling_gap"], eps_threshold),
    )
    cohorts = build_cohorts(episodes, fps, bins)
    cohort_gains: Dict[str, List[float]] = {}
    for co in cohorts:
        by_series: Dict[str, List[float]] = {}
        for i in co["member_idx"]:
            by_series.setdefault(episodes[i].series_uid, []).append(chosen_vs_raw[i])
        cohort_gains[co["cohort_id"]] = [
            float(np.mean(by_series[s])) for s in sorted(by_series)
        ]
    s3 = dict(s3_scope_harm(cohort_gains, delta_safe, b=b, seed=boot_seed))
    s3["headroom"] = (
        normalized_headroom(s3["harm_lcb90"], delta_safe)
        if s3["harm_lcb90"] is not None
        else 0.0
    )
    activated = activate({"S1": s1, "S2": s2, "S3": s3})
    signature_summary = _native(
        {
            "eps": eps_threshold, "delta_safe": delta_safe,
            "s1": s1, "s2": s2, "s3": s3, "activated": activated,
            "bootstrap": {"b": b, "seed": boot_seed},
        }
    )

    discovery_summary = _native(
        {
            "block": f"D{cycle}",
            "n_episodes": n_ep,
            "n_series": len(set(clusters_d)),
            "configs": sorted(by_dom),
            "baseline_utility": u_00,
            "n_abstained": sum(1 for c in chosen_h.values() if c is None),
            "pool_stats": {uid: dict(r.stats) for uid, r in runs.items()},
            "probe": {
                "n_union_programs": len(union),
                "n_failed_excluded": len(failed_shas & set(union)),
                "n_classes": len(effect_classes(probe_pool_losses)),
                "probe_shas": list(probe_shas),
                "n_probes": len(probe_shas),
                "n_selector_rows": len(selector_rows),
            },
            "counterfactual_chosen_sets": counterfactual,
            "consumed_manifest_path": str(d_manifest_path),
            "attribution_path": str(attribution_path),
        }
    )

    def _cost() -> Dict[str, Any]:
        c = dict(budget.as_dict())
        c.update(
            {
                "replacement_effect_calls": n_repl_calls,
                "surrogate_evals": n_surrogate,
                "judge_paired_fits": n_judge_fits,
                "adam_cogate_fits": n_adam_fits,
                "wall_clock_seconds": round(time.perf_counter() - t_start, 3),
            }
        )
        return c

    def _abstain(reason: str, internal_summary: Dict[str, Any]) -> CycleResult:
        gate.record_cycle_terminal(cycle, "abstain")
        return CycleResult(
            cycle=cycle, terminal="abstain", abstain_reason=reason,
            new_state=state, state_changed=False,
            signature=signature_summary, internal=_native(internal_summary),
            discovery=discovery_summary, precommit=None, cost=_cost(),
            entrypoint=entrypoint, frozen_literals_digest=_frozen_digest,
        )

    if activated is None:
        return _abstain(
            ABSTAIN_NO_SIGNATURE,
            {"family": None, "n_candidates": 0, "candidates": [], "winner": None,
             "internal_gate_pass": None, "eps": eps_threshold},
        )

    # ════ 步骤 4：miner（冻结代码；≤3 候选） ════
    if activated == "S1":
        if not selector_rows:
            return _abstain(
                ABSTAIN_MINER_EMPTY,
                {"family": activated, "n_candidates": 0, "candidates": [], "winner": None,
                 "internal_gate_pass": None, "eps": eps_threshold,
                 "note": "S1 激活但无 probe 归因行（池全空）→ 按 miner 空处理"},
            )
        evidence: Any = {"rows": selector_rows}
    elif activated == "S2":
        evidence = {"summary": {k: s2[k] for k in ("mean_classes", "ceiling_gap", "fired")}}
    else:  # S3
        worst = s3["worst_cohort"]
        co = next(c for c in cohorts if c["cohort_id"] == worst)
        tally: Dict[str, int] = {}
        accused_cand: Dict[str, Candidate] = {}
        for i in co["member_idx"]:
            c = chosen_h[episodes[i].uid]
            if c is None:
                continue
            tally[c.sha] = tally.get(c.sha, 0) + 1
            accused_cand[c.sha] = c
        if not tally:
            return _abstain(
                ABSTAIN_MINER_EMPTY,
                {"family": activated, "n_candidates": 0, "candidates": [], "winner": None,
                 "internal_gate_pass": None, "eps": eps_threshold,
                 "note": f"S3 worst cohort {worst!r} 无任何 chosen（全 abstain）→ 无被告"},
            )
        accused_sha = min(tally, key=lambda s: (-tally[s], s))   # 众数并列取 sha 升序（释义 E）
        evidence = {
            "cohort": co["miner_cohort"],
            "fingerprints": [fps[i] for i in co["member_idx"]],
            "accused_sha": accused_sha,
            "accused_ops": list(accused_cand[accused_sha].op_names()),
        }
    candidates = mine(activated, evidence, state, c0_bins=bins)
    if len(candidates) > MINER_CANDIDATE_BUDGET:
        raise P6BudgetError(
            f"miner 候选 {len(candidates)} > {MINER_CANDIDATE_BUDGET}（冻结上限）"
        )
    if not candidates:
        return _abstain(
            ABSTAIN_MINER_EMPTY,
            {"family": activated, "n_candidates": 0, "candidates": [], "winner": None,
             "internal_gate_pass": None, "eps": eps_threshold},
        )

    # ════ 步骤 5：内部选择（≤12 次完整配置 paired 重评）+ 内部门 ════
    evals: List[Dict[str, Any]] = []
    cand_states: Dict[str, Tuple[P6HarnessState, str, MinedCandidate]] = {}
    for cand in candidates:
        new_state_c, kind = _apply_candidate(state, cand)
        budget.charge_internal_reeval()
        pr = paired_arm_run(
            episodes, state, new_state_c, kind, K, wrapped_llm, fingerprints_run,
            on_scope_violation="raise",
        )
        views_a = _views_for_arm(episodes, pr["chosen_a"])
        views_b = _views_for_arm(episodes, pr["chosen_b"])
        _l00, u00_c, fits_a, nf_a = _fit_domains(episodes, views_a, cfg, fit_fn, rebuild_fn)
        _l11, _u11_c, fits_b, nf_b = _fit_domains(episodes, views_b, cfg, fit_fn, rebuild_fn)
        n_judge_fits += nf_a + nf_b
        _l10, u10_c = _cross_eval(episodes, fits_b, fits_a)     # train：模型换 edit、eval 保持 H
        evals.append(
            {
                "candidate_sha": cand.candidate_sha,
                "recipe_id": cand.recipe_id,
                "edit_kind": kind,
                "train_gain_d": gain(u00_c, u10_c),
                "new_state_sha": new_state_c.sha(),
            }
        )
        cand_states[cand.candidate_sha] = (new_state_c, kind, cand)
    best = min(evals, key=lambda ev: (-ev["train_gain_d"], ev["candidate_sha"]))
    internal_gate_pass = bool(best["train_gain_d"] >= eps_threshold)
    internal_summary = {
        "family": activated,
        "n_candidates": len(candidates),
        "candidates": evals,
        "winner": dict(best),
        "internal_gate_pass": internal_gate_pass,
        "eps": eps_threshold,
    }
    if not internal_gate_pass:
        return _abstain(ABSTAIN_INTERNAL_GATE, internal_summary)
    winner_state, winner_kind, winner_cand = cand_states[best["candidate_sha"]]

    # ════ 步骤 6：precommit（五元组 + sidecar；释义 F / G2 finding 34） ════
    internal_native = _native(internal_summary)
    sidecar_core = _build_sidecar_core(
        cycle, winner_cand, winner_kind, best, state.sha(), config_digest,
        materialization_sha, code_sha, seeds, signature_summary, internal_native,
        discovery_summary, entrypoint, _frozen_digest,
    )
    sidecar_sha = _write_precommit_sidecar(out, sidecar_core)
    precommit_payload = {
        "candidate_edit_sha": best["candidate_sha"],
        "harness_state_sha": state.sha(),
        "config_digest": config_digest,
        "materialization_sha": materialization_sha,
        "code_sha": code_sha,
        "sidecar_sha": sidecar_sha,   # G2：sidecar sha 入 precommit 事件 → hash 链锚定冻结候选
    }
    gate.record_precommit(cycle, precommit_payload)   # F4：崩溃后同五元组幂等续跑（不重复落账）

    # ════ 步骤 7（fresh）：交由 _finalize_v 收口（与 resume 路共用） ════
    counters = {"judge_fits": n_judge_fits, "adam_fits": n_adam_fits,
                "repl_calls": n_repl_calls, "surrogate": n_surrogate}
    return _finalize_v(
        fctx, cycle, state, episodes, episodes_v_loader, gate, winner_state, winner_kind,
        best, precommit_payload, signature_summary, internal_native, discovery_summary,
        counters, entrypoint, _frozen_digest, resumed=False,
    )


def cycle_frozen_literals(
    seeds: Sequence[int], bootstrap_b: int, cycle: int, K: int,
    trainer_fit_timeout_seconds: float,
) -> Dict[str, Any]:
    """run_cycle_formal 冻结字面量集合（用于 provenance digest）。"""
    return {
        "seeds": [int(s) for s in seeds], "bootstrap_b": int(bootstrap_b),
        "bootstrap_seed": bootstrap_seed_for(cycle), "K": int(K),
        "trainer_fit_timeout_seconds": float(trainer_fit_timeout_seconds),
    }


def run_cycle_formal(
    cycle: int,
    episodes_d: Sequence[P6Episode],
    episodes_v_loader: Callable[[], Any],
    state: P6HarnessState,
    gate: Any,
    c0_freeze: Mapping[str, Any],
    *,
    adam_trainer: Callable[[Sequence[SeriesView], int], Any],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
    llm_supplier: Optional[Callable[..., Any]] = None,
    judge_cfg: Optional[Mapping[str, Any]],
    seeds: Sequence[int] = CYCLE_PAIRED_SEEDS,
    out_dir: Any,
    code_sha: str,
    materialization_sha: str,
    config_digest: str,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    fit_fn: Optional[Callable[..., DomainFit]] = None,
    rebuild_fn: Optional[Callable[..., DomainFit]] = None,
    trainer_fit_timeout_seconds: float = ADAM_FIT_TIMEOUT_SECONDS,
) -> CycleResult:
    """**cycle 唯一合法正式入口**（G4/finding 36）：先机械断言全部 prereg §1/§4 冻结字面量
    （seeds/bootstrap_b=2000/boot seed=20260711+cycle/K=8/timeout=900），把 entrypoint +
    冻结字面量 digest 写入 CycleResult 与 sidecar（hash 链锚定），再委托 run_cycle_unfrozen
    并强制 V loader 为 BoundVEpisodes（require_bound=True）。任一漂移 → P6FrozenParamError。"""
    seeds_t = tuple(int(s) for s in seeds)
    boot_seed = bootstrap_seed_for(cycle)
    K = int(state.sampler.expected_total)
    assert_cycle_frozen_params(seeds_t, bootstrap_b, boot_seed, cycle, K, trainer_fit_timeout_seconds)
    digest = frozen_literals_digest(
        "run_cycle_formal",
        cycle_frozen_literals(seeds_t, bootstrap_b, cycle, K, trainer_fit_timeout_seconds),
    )
    return run_cycle_unfrozen(
        cycle, episodes_d, episodes_v_loader, state, gate, c0_freeze,
        adam_trainer=adam_trainer, fingerprints=fingerprints, llm_supplier=llm_supplier,
        judge_cfg=judge_cfg, seeds=seeds, out_dir=out_dir, code_sha=code_sha,
        materialization_sha=materialization_sha, config_digest=config_digest,
        bootstrap_b=bootstrap_b, fit_fn=fit_fn, rebuild_fn=rebuild_fn,
        trainer_fit_timeout_seconds=trainer_fit_timeout_seconds,
        _require_bound_v=True, _entrypoint="run_cycle_formal", _frozen_digest=digest,
    )


# ════════════════════════════ 步骤 10：supplier counterfactual chosen-set ════════════════════════════
def _counterfactual_chosen_sets(
    episodes: Sequence[P6Episode],
    state: P6HarnessState,
    K: int,
    llm_supplier: Optional[Callable[..., Any]],
    fingerprints: Optional[Mapping[str, Mapping[str, Any]]],
    budget: CycleBudget,
    incumbent_chosen: Mapping[str, Optional[Candidate]],
) -> Dict[str, Any]:
    """三配置 chosen-set（det-only / det+random / 现任；≤3、只在 D）。

    现任配置直接复用 discovery 的 chosen（同 state 同数据 bit 级恒等，不重跑）；
    与现任 allocation 相同的其它配置也复用（≤3 配置计数照收，fast path 不重复消耗）。"""
    det_k = min(3, K)
    specs = [
        ("det_only", {"det": K, "random": 0, "llm": 0}),
        ("det_random", {"det": det_k, "random": K - det_k, "llm": 0}),
        ("incumbent", dict(state.sampler.allocation)),
    ]
    incumbent_alloc = canonical_json(_native(dict(state.sampler.allocation)))
    incumbent_set = {
        uid: (c.sha if c is not None else None) for uid, c in incumbent_chosen.items()
    }
    cache: Dict[str, Dict[str, Optional[str]]] = {incumbent_alloc: incumbent_set}
    out: Dict[str, Any] = {}
    for name, alloc in specs:
        budget.charge_counterfactual()
        key = canonical_json(_native(dict(alloc)))
        if key not in cache:
            st = dataclasses.replace(
                state,
                sampler=SamplerSpec(
                    allocation=dict(alloc), expected_total=K,
                    random_params=dict(state.sampler.random_params),
                ),
            )
            fp = run_fast_path(
                [(ep.uid, ep.history) for ep in episodes], st, K, llm_supplier, fingerprints
            )
            cache[key] = {uid: (c.sha if c is not None else None) for uid, c in fp.items()}
        chosen = cache[key]
        out[name] = {
            "allocation": dict(alloc),
            "chosen": dict(chosen),
            "n_abstained": sum(1 for v in chosen.values() if v is None),
            "n_distinct_programs": len({v for v in chosen.values() if v is not None}),
        }
    return out
