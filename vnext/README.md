# TSharness vNext protocol boundary

`vnext/` is the non-destructive implementation track for benchmark-v0.2.  It does not
modify legacy P6/H_ref, the legacy Final ledger schema, or frozen files under
`results/Benchmark_v0_2`.

## Current machine state

The repository has a verified frozen environment and has reproduced the M0 data path
through L2, but an open protocol erratum blocks L3-L9:

```text
M0_READY_FOR_REPRODUCTION_WITH_OPEN_ERRATUM
TASK_G_AUTHORIZED=false
H0=NOT_FROZEN
Final=SEALED
```

All 53 raw/legacy assets and all 1919 canonical v0.2 clean-base records verify. Python
3.10.19 and every declared dependency match `vnext/environment/uv.lock`; 79 vNext and
benchmark regression tests pass. A shadow rebuild reproduced the frozen registry,
clean-base content, split, Support-A subsplit, corruption grid, dataset manifest, and
METR-LA blocks.

The stop is `results/vnext/m0/ProtocolErratumV1.json`: the frozen program-pool behaviour
test did not preserve its synthetic probe input bytes or bind the platform that generated
them. The mandated environment reconstructs a different input even though NumPy 1.26.4,
2.2.6, and 2.3.5 agree locally. Therefore its 8/8 output mismatches cannot currently
separate fixture drift from operator drift. L3-L9 and Task G remain closed until the exact
probe input or its originating environment is recovered; observed hashes must not be
accepted as replacement pins.

The formal H0 does not exist until the 136-series Init Corpus has produced all four
mandatory components in an `InitHarnessArtifactV1`.  M3a selects a runtime supplier
control after H0 and cannot redefine it.  `HarnessArtifactV1.h0()` is a compatibility-era
engineering value, not the research baseline.

## Hardened components

- `protocol.py`: authority resolution, frozen data views, historical exposure disclosure,
  group-atomic discovery folds, Method input contract, formal Init-only H0 boundary, and separate
  LLM trial/runtime/evolution qualifications.
- `init_harness.py`: exact 80+56 Init Corpus, forbidden-view guard, four mandatory H0
  components, and the formal Init-only H0 artifact.
- `access.py`: canonical-path, hash-chained, fsync'd access-before-read WAL for SA-V,
  Support-B, and the vNext Final authorization bridge.  A loader requires a durable
  `AccessReservationV1` receipt.
- `lifecycle.py`: receipt-driven M0→Task G→M2→Init H0→M3 supplier control→evolution→
  SA-V→Dev→Support-B→Final state machine.
- `preflight.py`: artifact integrity, recovery inventory, 1919-record clean-base audit,
  environment lock/probe, L0–L9 reproduction contract, and the binary M0 verdict.
- `recovery.py`: local SHA scan, clean-base candidate rejection, quarantine receipts,
  SHA-gated official downloads, displaced-byte backup, and exact legacy promotion.
- `gates.py`: fixed M3 primary comparison, runtime supplier selection without H0 mutation,
  bounded three-cycle evolution budget, factorial LLM qualification, and exact H0 SA-V
  comparator.
- `method.py` / `evaluator.py`: UID-independent semantics, explicit invalid-input terminal,
  effective-operator provenance, and history/inner call-order testing.

## Safe commands

From the repository parent with `PYTHONPATH` configured:

```bash
python -m SelfEvolvingHarnessTS.vnext protocol-audit --root SelfEvolvingHarnessTS
python -m SelfEvolvingHarnessTS.vnext m0-audit --root SelfEvolvingHarnessTS
python -m SelfEvolvingHarnessTS.vnext init-harness-prereg --root SelfEvolvingHarnessTS
python -m SelfEvolvingHarnessTS.vnext recovery-scan --root SelfEvolvingHarnessTS \
  --search-root /path/to/backup --quarantine SelfEvolvingHarnessTS/data/benchmark_v0/recovery_quarantine
python -m SelfEvolvingHarnessTS.vnext pinned-recovery --root SelfEvolvingHarnessTS \
  --quarantine SelfEvolvingHarnessTS/data/benchmark_v0/recovery_quarantine
```

Both commands are read-only unless `--out` is supplied. `m0-audit` records readiness of
raw data, exact clean-base, and the locked environment. Readiness does not override an
open protocol erratum and never authorizes Task G. A final `M0ReproductionVerdictV1` is
not created until a complete shadow raw-to-result run exists.

## What has not run

No Task G, M3/M4 search, SA-V, Dev vNext query, Support-B, Final, or U evaluation has been
executed.  M0 cannot pass by reading old reports; it requires exact asset recovery and a
shadow raw-to-result reproduction.
