"""P0 案例构造 dry-run（零 LLM）——验证注册表枚举与 expected 计算。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_v1_p0_batch_evidence_causal_dev as p0  # noqa: E402


def main() -> int:
    gold = p0._load_gold()
    for t in gold:
        print(t, "candidates:", list(gold[t]["candidates"].keys()))
    cases, navail = p0._build_cases(gold)
    print("n patch available:", navail, "| chosen:", len(cases))
    for ci, c in enumerate(cases):
        exp_full = p0._expected(c, "full")
        exp_swap = p0._expected(c, "swap")
        exp_mean = p0._expected(c, "mean_only")
        exp_single = p0._expected(c, "single_episode")
        print(
            f"case {ci + 1} [{c['kind']}] table={c['table']} "
            f"labels={c['labels']} subset={c['subset']} "
            f"winner={c['winner']} natural={c['natural']} "
            f"biased={c.get('single_episode_biased')}")
        print(f"   expected: full={exp_full} swap={exp_swap} "
              f"mean={exp_mean} single={exp_single}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
