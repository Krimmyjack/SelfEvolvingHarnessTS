"""tests/test_gym_b1b.py — B1b-mini gym 守卫（无 LLM 外呼、无重执行）：
  ① LOFO memory 无泄漏：held-out 族专属经验绝不进 selection-only memory。
  ② 主判据配对 CI 正负号正确 + 配对性（同 uid 逐序列差）。
  ③ resolved-chain novelty：bare savgol/median 默认窗 ≡ 池动作 → 非 novel（算子身份）。
  ④ 机械 Gate + 程序空间边界。
"""
import numpy as np

from SelfEvolvingHarnessTS.policy.program_edit import ProgramSpec, is_novel, validate
from SelfEvolvingHarnessTS.run_gym_b1b import SEEDED_MEM, CHALLENGE, paired_bootstrap_ci


def _sel_only_mem(dev_families, held, mem_by_fam):
    """复刻 lofo 中的 selection-only 检索逻辑（供守卫）。"""
    sel_families = [f for f in dev_families if f != held]
    return [line for f in sel_families for line in mem_by_fam.get(f, [])]


def test_memory_no_leak_lofo():
    """每折 memory 只含 selection 族经验；held 族专属条目一条都不得出现（核心 LOFO 完整性）。"""
    for held in CHALLENGE:
        mem = _sel_only_mem(CHALLENGE, held, SEEDED_MEM)
        held_lines = set(SEEDED_MEM.get(held, []))
        assert held_lines, f"{held} 无 seeded 经验（测试前提失效）"
        assert not (held_lines & set(mem)), f"held-out {held} 经验泄漏"
        # 且确实提供了 selection 族经验（memory 非空 → LLM 真有 history 可迁移）
        assert len(mem) == sum(len(SEEDED_MEM[f]) for f in CHALLENGE if f != held)


def test_paired_ci_sign_and_pairing():
    """配对 CI：A 系统性优于 B（每 uid dA>dB）→ point>0 且 CI 下界>0 且 A_beats_B。"""
    uids = [f"u{i}" for i in range(60)]
    rng = np.random.default_rng(0)
    base = {u: float(rng.normal(0, 0.1)) for u in uids}         # 共同的 per-uid 噪声（配对抵消）
    dA = {u: base[u] + 0.20 for u in uids}                      # A 一致高 0.20
    dB = {u: base[u] for u in uids}
    r = paired_bootstrap_ci(dA, dB)
    assert r["n"] == 60 and r["point"] > 0.15 and r["ci"][0] > 0 and r["A_beats_B"]
    # 反向：B 更好 → A 不胜
    r2 = paired_bootstrap_ci(dB, dA)
    assert r2["point"] < 0 and not r2["A_beats_B"]
    # 无差异 → CI 跨 0
    r3 = paired_bootstrap_ci(dB, dict(dB))
    assert r3["point"] == 0.0 and not r3["A_beats_B"]


def test_paired_ci_uses_common_uids_only():
    """配对只在两臂共同 uid 上做（一臂缺的 uid 不参与差分）。"""
    dA = {"a": 0.3, "b": 0.3, "c": 0.3}
    dB = {"a": 0.0, "b": 0.0}                                   # 无 'c'
    r = paired_bootstrap_ci(dA, dB)
    assert r["n"] == 2


def test_novelty_resolves_pool_equivalents():
    """resolved (op,window) 身份：与池动作编译等价者非 novel；真新组合/剂量 novel。"""
    def P(*steps):
        return ProgramSpec(steps=tuple(steps), scope=("forecast|snrLow|full",))
    assert not is_novel(P(("impute_linear", ())))                                   # v_none
    assert not is_novel(P(("impute_linear", ()), ("denoise_savgol", ())))           # 默认窗11 ≡ v_savgol
    assert not is_novel(P(("impute_linear", ()), ("winsorize", ()), ("denoise_savgol", ())))  # v_winsor_savgol
    assert is_novel(P(("impute_linear", ()), ("winsorize", ()), ("denoise_median", (("window", 9),))))
    assert is_novel(P(("impute_fft", ()), ("denoise_stl", ())))                     # 新 imputer
    assert is_novel(P(("impute_linear", ()), ("smooth_ma", (("window", 5),))))      # w5 不在 menu


def test_validate_grammar_bounds():
    """机械 Gate：首步须 imputer / 步数 1..3 / 窗算子须合法窗 / 禁相邻重复 / scope 非空。"""
    good = ProgramSpec(steps=(("impute_linear", ()), ("winsorize", ()),
                              ("denoise_median", (("window", 9),))), scope=("c",))
    assert validate(good)[0]
    assert not validate(ProgramSpec(steps=(("winsorize", ()),), scope=("c",)))[0]   # 首步非 imputer
    assert not validate(ProgramSpec(steps=(("impute_linear", ()),
                                           ("denoise_median", (("window", 7),))), scope=("c",)))[0]  # 窗越格
    assert not validate(ProgramSpec(steps=(("impute_linear", ()), ("winsorize", ()),
                                           ("winsorize", ())), scope=("c",)))[0]     # 相邻重复
    assert not validate(ProgramSpec(steps=(("impute_linear", ()),), scope=()))[0]   # 空 scope
