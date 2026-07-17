# Historical experiment recovery

The active tree does not store importable historical source. Full P0-P6, E32,
confirmatory, and former harness implementations are recovered from Git tag
`pre-architecture-convergence-2026-07-17` at commit
`1e75305770815c256d5b295b7ad6b8cb6cffe4b4`.

The exact cleanup boundary and benchmark relocation map are recorded in
`artifacts/manifests/active_tree_cleanup.json`. Frozen Benchmark-v0.2 evidence lives under
`artifacts/frozen/benchmark_v02/`.

Do not copy an old package back wholesale. Recover the needed mechanism, characterize its
required behavior, and promote it into the canonical contracts/operators/runtime/methods
structure.
