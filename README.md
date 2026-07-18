# SelfEvolvingHarnessTS

SelfEvolvingHarnessTS is an agent-centric, self-evolving time-series preparation
harness. TTHA is the sole active method: its fast path inspects data, writes candidate
programs and chooses one (including identity); its slow path attributes recurring
failures, proposes one-surface Harness edits, and promotes only edits supported by
paired replay.

## Current project framework

```text
SelfEvolvingHarnessTS/
├── contracts/       # Task, Program, and Method public contracts
├── conditioning/    # Time-series features, period detection, and condition routing
├── operators/       # Canonical operator implementations and the sole registry
├── runtime/         # Generic executor, candidate pool, trace, cache, and LLM backend
├── methods/ttha/    # Sole active Agent method and versioned Harness snapshots
├── evaluation/      # Mini-pipeline plus frozen Benchmark-v0.2 environment
├── artifacts/       # Frozen evidence and architecture-cleanup manifests
├── experiments/     # Git recovery instructions; no importable historical source
├── tests/           # Functional tests grouped by contracts, components, and integration
└── docs/            # Current architecture designs and implementation records
```

The active execution path is:

```text
public case -> TTHA Agent -> identity or Program -> Runtime -> Operators
                    ^                                  |
                    |--- versioned Harness <--- paired replay
```

H0 is procedurally complete but domain-naïve: it contains the workflow, safety
contracts and identity option, while learned capability skills and memory start empty.
The retired fixed reference is isolated under
`evaluation/benchmark_v02/_frozen_reference/` only to reproduce historical benchmark
numbers. It is not an Agent input, candidate source, or active method.

## Run the mini-pipeline

The checked-in replay tape runs two complete cycles without network access:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m \
  SelfEvolvingHarnessTS.cli.minipipe run \
  --backend replay \
  --replay-file SelfEvolvingHarnessTS/evaluation/minipipe/fixtures/m0_offline_replay_v1.jsonl \
  --cycles 2 \
  --run-dir SelfEvolvingHarnessTS/runs/minipipe/offline-demo
```

For a live run, provide the secret only through the environment. The default relay and
model are `https://api.agicto.cn/v1` and `gpt-5.5`:

```bash
export AGICTO_API_KEY='...'
/mnt/d/Anaconda_envs/envs/project/python.exe -m \
  SelfEvolvingHarnessTS.cli.minipipe run \
  --backend agicto \
  --cycles 2 \
  --run-dir SelfEvolvingHarnessTS/runs/minipipe/live
```

The live path uses the relay for Agent decisions and the frozen Chronos manifest for
valuation. Secrets are neither written into snapshots/artifacts nor included in request
hashes.

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
the active method surface. The small private benchmark fossil retained in this checkout
exists only for byte-for-byte regression checks.
