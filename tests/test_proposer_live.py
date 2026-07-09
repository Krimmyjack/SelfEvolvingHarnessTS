"""Proposer 实测（联网）：真实 DeepSeek 出候选 → 解析 → editable_surfaces 校验通过。

只验"LLM 提议 → 合法 EditPatch"链路（不要求 LLM 一定命中最优编辑）。K=2 控成本。
运行：  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.tests.test_proposer_live
"""
from __future__ import annotations

from SelfEvolvingHarnessTS.harness import HarnessState, EditPatch, Manifest
from SelfEvolvingHarnessTS.harness.editable_surfaces import validate as surface_validate
from SelfEvolvingHarnessTS.slow_path import BatchBuilder, Proposer, mine_weakness, Validator
from SelfEvolvingHarnessTS.data import make_forecast_batch, make_anomaly_batch

NMIN = 6


def _setup():
    h = HarnessState.from_minimal()
    for op in ["winsorize", "outlier_iqr", "outlier_mad"]:        # 降级：关离群算子
        h.apply_edit(EditPatch("L2", "set", f"l2.active_operators.{op}", False,
                               Manifest("w", "outliers leak", "lower nRMSE", "", "")))
    bb = BatchBuilder(h, n_min=NMIN)
    for rs in make_forecast_batch("P1", 2 * NMIN, seed0=0):
        bb.add_raw_series(rs)
    for rs in make_anomaly_batch("P1", NMIN, seed0=300):
        bb.add_raw_series(rs)
    cell = next(c for c in bb.triggerable_cells() if c.startswith("forecast|"))
    return h, bb, cell


def test_live_proposer_emits_valid_patches():
    h, bb, cell = _setup()
    weakness = mine_weakness(cell, bb.splits(cell)[0], h)
    proposer = Proposer(model="flash", k=2, temperature=0.7)
    candidates = proposer.propose(h, weakness, strength=None, rejection_log=[])
    print(f"    proposer returned {len(candidates)} valid candidate(s):")
    for c in candidates:
        print(f"      {c.edited_layer} {c.op} {c.path} = {c.value!r}  | {c.manifest.expected_effect!r}")
        assert surface_validate(c, h).ok            # 落在可编辑面、类型/引用合法
        assert c.cell_id == cell
    assert len(candidates) >= 1, "expected at least one parseable+valid candidate from the LLM"


def test_live_proposer_into_validator():
    """额外信息：把 LLM 候选喂 validator，看是否有被接受的（不作硬断言）。"""
    h, bb, cell = _setup()
    weakness = mine_weakness(cell, bb.splits(cell)[0], h)
    cands = Proposer(model="flash", k=3, temperature=0.7).propose(h, weakness, None, [])
    v = Validator()
    n_acc = 0
    for c in cands:
        out = v.validate(c, h, cell, bb.splits(cell))
        print(f"      {c.path}={c.value!r} -> {out.reason}")
        n_acc += int(out.accept)
    print(f"    {n_acc}/{len(cands)} LLM candidates accepted by grounded gate")


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
