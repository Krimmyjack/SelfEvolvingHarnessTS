# SelfEvolvingHarnessTS

SelfEvolvingHarnessTS is an active preparation-method harness with one contract layer,
one operator registry, one runtime, and one evolving method line.

## Active structure

- `contracts/` — task, program, and method contracts.
- `operators/` and `conditioning/` — canonical preparation mechanics.
- `runtime/` — the sole executor and fast path.
- `methods/h_ref_v02/` — frozen reference method.
- `methods/ttha/` — the next active method line when implemented.
- `evaluation/benchmark_v02/` — auxiliary frozen evaluator.
- `artifacts/frozen/benchmark_v02/` — immutable benchmark evidence.

## Verification

From the directory containing `SelfEvolvingHarnessTS`:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Benchmark CLI:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe \
  -m SelfEvolvingHarnessTS.evaluation.benchmark_v02 --help
```

## Historical recovery

P1–P6, E32, confirmatory runs, former runners, and historical result trees are available
from Git tag `pre-architecture-convergence-2026-07-17`. They are intentionally absent from
the active checkout.
