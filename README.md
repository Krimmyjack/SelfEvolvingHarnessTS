# SelfEvolvingHarnessTS

SelfEvolvingHarnessTS is the executable substrate for a **cross-domain, self-evolving
time-series data-preparation Harness**. The project-level goal is to accumulate
receipt-backed processing capabilities across source datasets and safely reuse or
revise them on unseen target datasets under a fixed target-feedback budget.

TTHA remains the sole executable Agent-method identifier in the current tree for
compatibility. Its fast path inspects data, writes candidate programs and chooses one
(including identity); its slow path attributes recurring failures, proposes one-surface
Harness edits, and promotes only edits supported by paired replay. TTHA is now a
target-local adaptation component, not the complete project-level research claim.

Current authority and routing, in order:

1. [`AGENTS.md`](AGENTS.md) — long-term method and execution boundary;
2. [`PROJECT_STATE_AND_DATA_MAP_2026-08-23.md`](docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md)
   — compact system, evidence, and data-exposure map;
3. [`DATA_QUALITY_AND_FEEDBACK_MODEL.md`](docs/DATA_QUALITY_AND_FEEDBACK_MODEL.md)
   — task/consumer-relative quality and feedback semantics;
4. [`ROADMAP_POST_V1_2026-08-22.md`](docs/ROADMAP_POST_V1_2026-08-22.md) — active route;
5. [`STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md`](docs/STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md)
   — detailed adjudication history.

The four files under the workspace-level `idea/` directory are historical design inputs,
not current execution instructions. They must not override the sources above.

## Current project framework

```text
SelfEvolvingHarnessTS/
├── contracts/       # Task, Program, and Method public contracts
├── conditioning/    # Time-series features, period detection, and condition routing
├── operators/       # Canonical operator implementations and the sole registry
├── runtime/         # Generic executor, candidate pool, trace, cache, and LLM backend
├── methods/ttha/    # Current executable Agent method and versioned Harness snapshots
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

## Current evidence and next gate

The executable lifecycle remains:

```text
observe -> propose typed Program -> execute -> attribute first fault
        -> propose one-surface edit -> paired replay -> versioned snapshot
```

Current evidence and route (2026-08-24):

- Forecasting has one bounded natural positive result: accumulated experience reduced
  pooled first-positive cost from 123 to 69 retrains while terminal utility and harm tied.
  This evidence remains frozen and is not rewritten by the AD work.
- Multi-task Task/Consumer isolation, the shared lifecycle, AD adapter, task-context
  fail-closed wiring, and H0 compatibility are connected through #42k/#42k-b.
- #42j did not qualify the sixth IForest fit policy (`FIT_POLICY_NOT_QUALIFIED`). The
  current IForest × program/feedback slice is stopped; a borderline `f1_pooled` diagnostic
  is not authorized as a Support or promotion signal.
- Yahoo S5 A1 contains 67 downloaded files and a 65-series roster. Outcomes for the first
  24 are development-exposed; the remaining 41 stay sealed for a later frozen evaluation.
- #42l is closed and #43 M0-C completed with no safe global cleaning headroom on the
  exposed Yahoo-24 slice across the three tested AD Consumers. This closes only that
  data/Consumer/menu probe; it does not imply that AD has no adaptable data problem.
- The current method gate is #44a: on a controlled held-in contamination, first verify
  that a known repair improves independent delayed task utility and then test whether an
  early Support signal predicts that recovery. Agent behavior, Experience replay, and
  the 41 sealed Yahoo series remain downstream of this feedback-validity gate.
- The product remains A5: audited accumulated knowledge plus multi-round Target held-in
  calibration. A3 and Static are attribution ablations, not replacement product goals.

Historical E0/E1/E2 evidence remains in `artifacts/` and the stage report; it is not the
active routing authority.

## Run the mini-pipeline

The checked-in replay tape runs two complete cycles without network access. On native
Windows, the existing environment can be activated with `conda activate project`; this
is a convenience, not a requirement imposed on other shells:

```bash
conda activate project
python -m SelfEvolvingHarnessTS.cli.minipipe run --backend replay --replay-file SelfEvolvingHarnessTS/evaluation/minipipe/fixtures/m0_offline_replay_v1.jsonl --cycles 2 --run-dir SelfEvolvingHarnessTS/runs/minipipe/offline-demo
```

This deterministic replay is contract evidence: it proves that candidate generation,
fault routing, one-surface edits, paired replay, and lineage work end to end. Its
synthetic valuation receipts are explicitly labeled
`DETERMINISTIC_CONTRACT_FIXTURE`; an edit promoted by this replay is not, by itself,
evidence of improvement under the frozen Chronos judge.

For a live run, provide secrets only through the environment. Backend, relay, and model
must come from the frozen experiment configuration rather than this README:

```bash
conda activate project
# Set AGICTO_API_KEY with the native shell's environment-variable syntax first.
python -m SelfEvolvingHarnessTS.cli.minipipe run --backend agicto --cycles 2 --run-dir SelfEvolvingHarnessTS/runs/minipipe/live
```

The live path uses the relay for Agent decisions and the frozen Chronos manifest for
valuation. Secrets are neither written into snapshots/artifacts nor included in request
hashes. Scientific runs should report `valuation_source=PINNED_FROZEN_CHRONOS` and
retain the immutable Agent-response cache used by paired replay.

## Verification

From the directory containing `SelfEvolvingHarnessTS` (after activating the desired
environment if needed):

```bash
python -m pytest SelfEvolvingHarnessTS/tests -q --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Benchmark CLI:

```bash
python -m SelfEvolvingHarnessTS.evaluation.benchmark_v02 --help
```

## Historical recovery

P1–P6, E32, confirmatory runs, former runners, and historical result trees are available
from Git tag `pre-architecture-convergence-2026-07-17`. They are intentionally absent from
the active method surface. The small private benchmark fossil retained in this checkout
exists only for byte-for-byte regression checks.
