# SelfEvolvingHarnessTS

SelfEvolvingHarnessTS is an active preparation-method harness with one contract layer,
one operator registry, one runtime, and one evolving method line.

## Current project framework

```text
SelfEvolvingHarnessTS/
├── contracts/       # Task, Program, and Method public contracts
├── conditioning/    # Time-series features, period detection, and condition routing
├── operators/       # Canonical operator implementations and the sole registry
├── runtime/         # Sole executor, fast path, trace, and error model
├── methods/         # Current H_ref_v02 reference; future TTHA active method line
├── evaluation/      # Auxiliary Benchmark-v0.2 evaluation environment
├── artifacts/       # Frozen evidence and architecture-cleanup manifests
├── experiments/     # Git recovery instructions; no importable historical source
├── tests/           # Functional tests grouped by contracts, components, and integration
└── docs/            # Current architecture designs and implementation records
```

The active execution path is:

```text
TaskSpec -> Method -> Runtime -> Operators
                   ^
              Conditioning
```

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
