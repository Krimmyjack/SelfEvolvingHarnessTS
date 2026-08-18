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

The current research truth is:

1. [`Cross_Domain_Self_Evolving_Harness_Method.md`](../idea/Cross_Domain_Self_Evolving_Harness_Method.md)
2. [`Time_Series_Workspace_and_Receipt_Contract.md`](../idea/Time_Series_Workspace_and_Receipt_Contract.md)
3. [`Cross_Domain_Evidence_Memory_Design.md`](../idea/Cross_Domain_Evidence_Memory_Design.md)
4. [`Cross_Domain_Experiment_Protocol.md`](../idea/Cross_Domain_Experiment_Protocol.md)

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

The checked-in M0/E0 path establishes the executable contract loop, and E1 provides a
mixed controlled cross-generator result:

```text
observe -> propose typed Program -> execute -> attribute first fault
        -> propose one-surface edit -> paired replay -> versioned snapshot
```

The natural-data E2 work has not promoted a Source Capability. The important actual
results are:

- the first `period >= 25 -> seasonal` candidate failed fresh promotion (mean gain
  `-0.00628`, harm `0.50`);
- an all-Linear cohort policy retained positive mean direction on two fresh Source
  cohorts, but harm was `0.375`, above the frozen `0.25` limit;
- the fixed pseudo-gap Witness harmed half of its evaluation series and was stopped;
- exact restoration of a target-adjacent coherent missingness artifact was readable by
  the fixed Ridge Consumer on FRED-MD and NN5, so that defect has downstream headroom;
- the proposed level-adjusted multi-cycle Program passed local recovery only on FRED,
  then failed a frozen `gap/period <= 0.5` extension on both Traffic and METR-LA;
- even a broader grader-only best-donor oracle failed the conjunctive donor premise:
  Traffic passed, but METR recovery was `0.32282`, below the frozen `0.50` gate;
- the Outlier corruption degraded the fixed Ridge Consumer on METR-LA but not on Traffic.

The promotion gate therefore prevented an unsupported capability from entering Source
Memory. UCI Target Query remains unopened. This is evidence that fail-closed promotion
has operational value, not evidence that Source Memory improves Target adaptation.

The current development ladder is deliberately smaller than a full Memory/Agent system:

```text
P0 local recoverability
 -> P1 downstream Consumer sensitivity to the defect
 -> P2 observable Witness selectivity and clean-data safety
 -> P3 fresh cross-dataset capability promotion
 -> only then: A3/A4/A5 Target adaptation
```

The recent periodic-donor family is now retired before P1. The defect is readable under
exact restoration, but neither the frozen Program nor a broader truth-selected donor
oracle supplied enough cross-Source P0 recovery. Public donor descriptors remain useful
future observation candidates, but no threshold is compiled from these exposed outcomes.

The structural provenance positive control is now complete and stopped. Key-only rebind
exactly repaired injected whole-`TargetRow` misbinding before fitting on every tested
dataset. It also improved fresh Traffic validation (`6/6` positive pairs), but fresh
COVID evidence contradicted stable Consumer utility: only `3/6` pairs improved and the
median gain was negative. The frozen cross-dataset promotion verdict is therefore FAIL.

The signed ledger preserves this conflict instead of averaging it away: Traffic policy
evidence is `supported`, COVID is `contradicted`, and contradiction priority compiles to
`ABSTAIN_DO_NOT_REGISTER`. Key rebind may remain a candidate minimal integrity adapter
because it restores schema semantics, but it is not a promoted utility capability or a
Memory entry. No additional fit on this broad structural family is justified, and UCI
Target Query remains unopened. The next scientific gate must start from a newly frozen
numerical/data-semantics intervention family with a reliable Consumer/metric premise;
it cannot tune a scope from these exposed outcomes. Detailed evidence is in
`artifacts/functional/e2/` and `CLAIMS_FROM_RESULTS.md`.

E2-J0 has now calibrated that Consumer/metric premise on exposed Source data. A frozen
22-fit Ridge plus original-unit sMASE run used three fixed doses of standardized training-
target block corruption. Traffic and FRED-MD both passed the response sub-gate with
strictly increasing dataset-mean degradation (`0.175/0.250/0.329` and
`0.121/0.290/0.501`), but failed the pre-registered `epsilon=0.02` resolution gate:
MDE80 was `0.227` and `0.382`. The correct status is therefore
`READABLE_AT_INJECTED_DOSE_BUT_UNDERPOWERED_FOR_EPSILON`, not E2-ready. P0 and exact-
repair controls passed, the fit budget was exact, and UCI/Target Query stayed closed.

This creates a protocol decision rather than a new Capability: either treat J0 as a
strong-effect readability instrument and require future candidate families to clear
their own large, pre-registered headroom gates, or pre-register a new lower-variance
effect measure/Consumer protocol. Merely adding the currently available exposed groups
cannot reach absolute sMASE resolution `0.02`; no new Capability, promotion, Memory, or
Target experiment is authorized until that decision is frozen.

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

This deterministic replay is contract evidence: it proves that candidate generation,
fault routing, one-surface edits, paired replay, and lineage work end to end. Its
synthetic valuation receipts are explicitly labeled
`DETERMINISTIC_CONTRACT_FIXTURE`; an edit promoted by this replay is not, by itself,
evidence of improvement under the frozen Chronos judge.

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
hashes. Scientific runs should report `valuation_source=PINNED_FROZEN_CHRONOS` and
retain the immutable Agent-response cache used by paired replay.

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
