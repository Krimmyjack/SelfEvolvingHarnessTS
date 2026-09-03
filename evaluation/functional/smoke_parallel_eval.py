"""parallel_eval 冒烟（顺序保持/加速/异常隔离——零 LLM 零评估）。"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

from parallel_eval import run_parallel  # noqa: E402


def main() -> int:
    def f(i: int) -> int:
        time.sleep(0.3)
        return i * 2

    t0 = time.time()
    r = run_parallel([lambda i=i: f(i) for i in range(8)], workers=4)
    dt = time.time() - t0
    vals = [x for ok, x in r if ok]
    assert vals == [0, 2, 4, 6, 8, 10, 12, 14], r
    print(f"order preserved: {vals}")
    print(f"8 x 0.3s / 4 workers: {dt:.2f}s (sequential 2.4s)")

    def g(i: int) -> int:
        raise ValueError(f"boom {i}")

    r2 = run_parallel([lambda i=i: g(i) for i in range(3)], workers=2)
    assert [ok for ok, _ in r2] == [False, False, False], r2
    print("exceptions isolated per task: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
