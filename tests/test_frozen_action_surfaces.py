"""tests/test_frozen_action_surfaces.py — E-3.3 扩算子的**不变性证明**（不是断言，是证明）。

背景：E-3.3（2026-07-14）往 `operators/registry.py` 加了 4 个算子（repair_level_shift /
hampel_filter / impute_ssm / impute_ar）。registry 是一张全局表，谁都读它——所以"加算子会不会
悄悄改掉别的东西"必须被机械回答，而不是靠人保证。本模块守三个**已经有数字挂在上面**的面：

  ① **benchmark-v0.2 冻结池的行为**（results/Benchmark_v0_2/ 的全部读数建立在它上面）。
     `program_pool.json` 用 SHA 钉了 `operators/registry.py` 等源文件——而注册新算子**必然**
     改动 registry.py 的字节，所以那个 digest **一定会变**。programs.py 自己的注释早就点破了
     这个机制的局限：

         "Over-pinning -- hashing modules the pool never reaches -- would invalidate a frozen
          pool for edits that provably cannot move a single number, and a freeze that cries
          wolf gets ignored."

     文件 digest 回答不了"数字有没有动"。**行为 digest 可以**：下面对 8 个 value-domain program
     在一条固定合成序列上的输出做逐字节 SHA256。这些 digest 是在**改 registry 之前**采下来的。
     它们不变 ⇔ v0.2 的每一个 program 仍然算出同样的值 ⇔ v0.2 的读数依然有效。

  ② **H_ref 的候选文法**（`p6/fast_path.py`）。v0.2 里 H_ref 是被度量的现任者；它的候选池由
     `det_ladder()` 与 `GRAMMAR_*` **硬编码**给出，不从 registry 推导——这是好事（加算子不会
     偷偷扩大 H_ref 的动作空间），但"好事"必须被钉住，否则哪天有人"顺手"把 GRAMMAR_DENOISERS
     改成从 registry 生成，H_ref 就变成了另一个东西，而所有 v0.2 的数字还挂在旧名字上。

  ③ **menu v1 的 SHA**（P0 冻结动作面）与 `minimal_l2().operator_defaults`。后者是前者 meta 里
     的一个字段，**也**是 P6 `resolve_steps` 的参数来源——给新算子加一条 default，会同时
     (a) 改掉 menu v1 的 SHA，(b) 静默改掉 H_ref 的参数解析。所以新算子的参数一律显式写死在
     menu v2 的 ActionSpec 里，`operator_defaults` 一个键都不许加。这条测试就是那道闸。
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.programs import (
    PROGRAM_SPECS,
    RUNNER_EXECUTED,
    _code_digests,
    apply_program,
)
from SelfEvolvingHarnessTS.harness.layers import minimal_l2
from SelfEvolvingHarnessTS.p6.fast_path import (
    GRAMMAR_DENOISERS,
    GRAMMAR_IMPUTERS,
    GRAMMAR_OUTLIERS,
    GRAMMAR_WINDOWS,
    det_ladder,
)
from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1, action_menu_v2

_FIXED_PERIOD = 24

_RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "Benchmark_v0_2"
_POOL_MANIFEST_PATH = _RESULTS / "program_pool.json"
_RECONCILIATION_PATH = _RESULTS / "pool_code_pin_reconciliation.json"

# ── ① v0.2 池的行为 digest（采于 E-3.3 改动 registry 之前） ────────────────────
_POOL_OUTPUT_SHA256 = {
    "raw": "9a3049d734abb175214c6a0c6e6d78d78834cc9985bdb38939319bd0c1cd4a5e",
    "forward_fill": "7b8db1242f0e5e9ac50d8ae7e01a9213a9266ccafb539a9a472e2729bcf0076f",
    "seasonal_fill": "66ab500a54e5237a231ace534c0bb1628ca6e61d892827646b02f7c1a5182c5e",
    "winsorize": "91dc1811f007b531cd0d09ec8cb8afd28a5abf65ba53537321e41cc22b95796e",
    "denoise_median": "0f6ec9313b4eed97ed58bbe51c375093000d94d26d90173d7bf526c04ea13aca",
    "denoise_stl": "baab3d011566037bf2c15fc23aa8b6e62c519b7103367ee7ebd6b79186acc189",
    "denoise_savgol": "36d853525c3fc3ec6ae4e9e9cd49ddd20af8530243f755c862e35825512043e7",
    "denoise_wavelet": "f2b80def798039f3b94cb34e9086d31ee69140f740d099454c01ce895f4918c7",
}

# ── ③ P0 冻结面（同样采于改动之前） ───────────────────────────────────────────
_MENU_V1_SHA256 = "a4c0f83191e5bd18fb8b7b9f52854e356268ef3cf0e7e3420e3a0221514db457"

_OPERATOR_DEFAULTS = {
    "denoise_savgol": {"window": 11, "order": 3},
    "denoise_median": {"window": 5},
    "smooth_ma": {"window": 5},
    "stl_decompose": {"period": 0},
}

_DET_LADDER_SHAS = ("a6a6db644a7b61c0", "c0f66a51e987f8a7", "bee33065e1b25757")


def _probe_series() -> np.ndarray:
    """固定探针序列：趋势 + 季节 + 噪声 + 一段缺失 + 一个 spike + 一个电平断层。

    刻意让它同时踩到全部四类缺陷——这样任何一个 program 的行为漂移都会在 digest 上现形，
    而不是恰好落在探针够不着的分支里。构造必须逐位可复现（固定 seed、固定长度）。"""
    rng = np.random.default_rng(20260714)
    n = 240
    t = np.arange(n)
    x = 10.0 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / _FIXED_PERIOD) + rng.normal(0, 0.5, n)
    x[30:36] = np.nan
    x[100] += 18.0
    x[150:170] += 6.0
    return x


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


@pytest.mark.parametrize("program_id", sorted(_POOL_OUTPUT_SHA256))
def test_v0_2_pool_programs_are_behaviourally_unchanged(program_id):
    """v0.2 池的 8 个 value-domain program 逐字节产出不变 ⇒ v0.2 的读数没有被扩算子动过。

    这条测试**取代**了"registry.py 的文件 digest 不变"这个（做不到、也不该要求的）条件：
    注册新算子必然改 registry.py 的字节，但**不改任何一个 program 算出来的值**。
    行为不变才是"数字还成立"的充要条件。
    """
    out = apply_program(program_id, _probe_series(), period=_FIXED_PERIOD)
    assert _digest(out) == _POOL_OUTPUT_SHA256[program_id], (
        f"program {program_id!r} 的输出变了——benchmark-v0.2 的全部读数以它为基础。"
        f"若这是有意的算子行为变更，它就是一次 **benchmark 版本 bump**，不是一次重构。"
    )


def test_pool_membership_did_not_grow():
    """新算子**没有**偷偷进池。池成员的变更是 amendment + 版本 bump，不是 registry 的副作用。"""
    value_domain = {s.program_id for s in PROGRAM_SPECS if s.program_id not in RUNNER_EXECUTED}
    assert value_domain == set(_POOL_OUTPUT_SHA256)
    assert len(PROGRAM_SPECS) == 9          # 8 value-domain + h_ref（runner 执行）


def test_pool_code_pins_are_intact_or_explicitly_reconciled():
    """`program_pool.json` 用 SHA 钉了实现它的源文件。这道闸让那个 pin **要么成立、要么被显式调和**。

    为什么需要这道闸：注册一个新算子**必然**改动 `operators/registry.py` 的字节，于是 pin 必然
    对不上——而在 E-3.3 之前，**没有任何东西检查它**。一个没人检查的 pin 不是"红的"，它只是一句
    躺在 JSON 里的假话；而一个总是红着又没人管的检查，会教会每个下游开发者无视红色。两者都致命。

    所以规则是：源文件 digest 与冻结 pin 不一致时，**必须**存在一份调和记录，且它必须
      ① 恰好列出漂移的那些文件（不多不少——"顺手"改了别的文件不许蒙混过去）；
      ② 记下冻结时的旧 digest 与当前的新 digest（= 重新 pin）；
      ③ 携带**行为等价证据**，且该证据与本模块守的那组 digest 逐位一致（单一真源，不许两处漂移）。

    ⚠️ 下次改动 `_POOL_CODE_FILES` 里的文件时，这条测试会失败。正确做法是**更新调和记录并重新
    证明行为等价**，不是放宽这条测试。
    """
    frozen = json.loads(_POOL_MANIFEST_PATH.read_text(encoding="utf-8"))["code_sha256"]
    current = _code_digests()
    drifted = sorted(path for path in frozen if frozen[path] != current[path])

    if not drifted:
        return                                   # pin 本身成立，无需调和

    assert _RECONCILIATION_PATH.is_file(), (
        f"源文件 digest 漂移了 {drifted}，但没有调和记录。"
        f"带着一个失效的冻结 pin 发布 = 教下游开发者无视红色检查。"
    )
    record = json.loads(_RECONCILIATION_PATH.read_text(encoding="utf-8"))

    assert record["reconciles"] == "results/Benchmark_v0_2/program_pool.json"
    assert record["pool_membership_changed"] is False
    assert sorted(record["reconciled_files"]) == drifted          # ① 不多不少
    assert record["pinned_at_freeze"] == {p: frozen[p] for p in drifted}   # ② 旧 pin
    assert record["current"] == {p: current[p] for p in drifted}           # ② 重新 pin
    assert record["behavioural_equivalence"]["program_output_sha256"] == _POOL_OUTPUT_SHA256  # ③


# ── ② H_ref 的候选文法 ────────────────────────────────────────────────────────
def test_h_ref_candidate_grammar_is_frozen_and_registry_independent():
    """H_ref 的动作空间是**硬编码**的，不随 registry 增长。

    这既是事实也是要求：v0.2 里 H_ref 是被度量的现任者，它的候选池扩了 = 被度量的对象换了人，
    而所有 v0.2 的数字还挂在 "h_ref" 这个名字上。任何"让文法从 registry 自动生成"的重构都会
    在这里炸——那正是它该炸的地方。新算子要进程序生成面，走 **menu v2**，不走 P6 文法。
    """
    assert GRAMMAR_IMPUTERS == ("impute_linear", "impute_ema")
    assert GRAMMAR_OUTLIERS == ("winsorize", "outlier_iqr", "outlier_mad")
    assert GRAMMAR_DENOISERS == ("denoise_median", "smooth_ma", "denoise_savgol")
    assert GRAMMAR_WINDOWS == (5, 9, 15, 25)

    new_operators = {"repair_level_shift", "hampel_filter", "impute_ssm", "impute_ar"}
    grammar = set(GRAMMAR_IMPUTERS) | set(GRAMMAR_OUTLIERS) | set(GRAMMAR_DENOISERS)
    assert not (grammar & new_operators), "E-3.3 的新算子泄进了 H_ref 的候选文法"

    assert tuple(c.sha for c in det_ladder()) == _DET_LADDER_SHAS
    for candidate in det_ladder():
        assert not (set(candidate.op_names()) & new_operators)


# ── ③ menu v1 与 operator_defaults ────────────────────────────────────────────
def test_menu_v1_sha_is_unchanged_by_the_new_operators():
    assert action_menu_v1().sha256 == _MENU_V1_SHA256


def test_operator_defaults_gained_no_key():
    """新算子**不得**进 operator_defaults。

    它是 menu v1 的 meta 字段（加键 → v1 的 SHA 变），**同时**是 P6 `resolve_steps` 的参数来源
    （加键 → 静默改掉 H_ref 的参数解析）。一个 dict，两个冻结面在吃它。新算子的参数一律显式
    写死在 menu v2 的 ActionSpec 里。"""
    assert minimal_l2().operator_defaults == _OPERATOR_DEFAULTS


def test_menu_v2_extends_v1_without_mutating_it():
    v1, v2 = action_menu_v1(), action_menu_v2()
    assert set(v1.actions) < set(v2.actions)                 # 真扩张
    for action_id, spec in v1.actions.items():
        assert v2.actions[action_id].to_dict() == spec.to_dict()   # v1 的动作逐字节不变
    assert v2.sha256 != v1.sha256                            # 改动作集 = 新 SHA
    assert v2.meta["extends_sha256"] == v1.sha256            # 且新菜单自陈它扩的是哪一版
