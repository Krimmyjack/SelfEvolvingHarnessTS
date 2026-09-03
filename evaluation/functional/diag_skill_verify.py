"""诊断：fast_winner_repair_level_shift 候选在 R2 @888 的 verify 拒绝原因
（preserve_outside / max_modified_fraction / 参数绑定）。零 LLM。"""

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "evaluation" / "functional"))
sys.path.insert(0, str(root / "methods" / "ttha"))

import numpy as np  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _config as _kdd_config  # noqa: E402
from SelfEvolvingHarnessTS.contracts.candidate import (  # noqa: E402
    Candidate,
    CandidateKind,
)
from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.runtime.candidate_verification import (  # noqa: E402
    verify_candidate,
)

rows = [json.loads(line)
        for line in (root / "artifacts/functional/e2"
                     / "w1_kdd2018_frozen_cohort_p41.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()]
cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
names = [str(n) for n in cache["names"]]
values = cache["values"]
roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
          for r in rows]
vals = {str(r["series_name"]): np.asarray(
    values[names.index(str(r["series_name"]))], dtype=np.float64)
    for r in rows}
series0 = vals["T117"]

params = {"estimated_offset": 56.0, "region_end_fraction": 0.1717171717171717,
          "region_start_fraction": 0.04671717171717172}
program = Program.from_steps([("repair_level_shift", dict(params))],
                             source="diag")
cand = Candidate(candidate_id="diag_skill", kind=CandidateKind.PROGRAM,
                 program=program, source="diag")
ex = ScopeExecutor(roster, vals, _kdd_config(),
                   evaluate_fn=None)  # verify 不需要 evaluate
for preserve in (True, False):
    for mf in (0.35, 0.5, 1.0):
        rej = []
        for uid, anchor, window in ex.training_windows(888):
            art = verify_candidate(
                cand, window, allowed_operators=("repair_level_shift",),
                inspected_regions=((0, int(window.size)),),
                maximum_modified_fraction=mf,
                preserve_outside_inspected_region=preserve,
                require_finite_output=False)
            if not art.selectable:
                rej.append((uid, anchor, art.receipt.rejection_code))
        print(f"preserve={preserve} max_modified={mf}: "
              f"rejected {len(rej)}/108", rej[0] if rej else "")
