# Benchmark v0 Data, Metrics, and Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download and register all approved datasets, freeze a leakage-safe benchmark with a defined Dev-Query, and implement the complete method-to-report pipeline without modifying frozen P6 behavior.

**Architecture:** A new `benchmark/` package owns source provenance, registry, split, materialization, corruption, ingestion, Method API, downstream trainers, metrics, aggregation, baselines, campaign ledger, and runners. Existing operator and torch model classes are reused behind new benchmark contracts. Final-Query is accessible only through a frozen one-shot evaluation campaign; all development validation and saturation diagnosis run on an independent repeatable Dev-Query.

**Tech Stack:** Python 3.10, numpy, pandas, pyarrow, huggingface_hub, statsmodels STL, PyTorch, pytest, JSON/JSONL/YAML artifacts, stdlib SHA256/file locking/fsync.

## Global Constraints

- Do not modify any file under `SelfEvolvingHarnessTS/p6/` or change P6 outputs.
- Canonical interpreter: `D:/Anaconda_envs/envs/project/python.exe`.
- Run commands from `C:/Users/辉/Desktop/Agent` so `SelfEvolvingHarnessTS` imports resolve.
- Every shell command is prefixed with `rtk`; every test command uses the canonical interpreter.
- Headline protocol: daily/hourly only, `L=48`, `H=48`, `MIN_LEN=207`.
- Model seeds are exactly `(0, 1, 2)`; corruption replicates are exactly `(0, 1)`.
- Harm threshold is conventional `delta=0.05 sMASE`, not a calibrated estimate. The manifest field is `harm_threshold_kind: conventional`.
- Bootstrap is `B=2000`, master seed `20260713`; sub-seeds are canonical SHA256 derivations.
- Corruption key is exactly `(benchmark_version, clean_content_sha, scenario, dose, replicate_idx)`.
- Normalization is benchmark-owned; `znorm`, `minmax_norm`, and all `changes_target_space=True` operators are forbidden.
- Raw means `No-op + canonical ingestion`.
- Final method failures are terminal results. Only exact-hash, exact-checkpoint infrastructure interruptions may resume.
- Existing dirty P6 and result files belong to the user; stage and commit only task-specific files.

## Protocol Definitions Added Before Coding

### Dev-Query

Dev-Query is a fifth explicit outer role, not an alias for Support-B. For every fresh, eligible, non-U atomic overlap group, compute:

```text
u = uint64_be(sha256(f"{benchmark_version}|outer|{split_salt}|{group_key}")[:8]) / 2^64
```

where `group_key = overlap_group` when present, otherwise `series_uid`. Assign:

```text
[0.00, 0.25) -> Support-A-fresh
[0.25, 0.45) -> Support-B
[0.45, 0.65) -> Dev-Query
[0.65, 1.00) -> Final-Query
```

Legacy/confirmed-exposed groups are forced to Support-A. Frozen U selections are forced to U. Any group that is simultaneously forced to Support-A and U is a manifest error. Dev-Query is repeatable and returns full metrics; it cannot select best-fixed, train `oracle_transfer`, substitute for Support-B confirmation, or enter Final. Each dataset-by-regime Dev-Query cell needs at least 12 uid for saturation reporting; smaller cells remain present with `diagnostic_unavailable`, with no refill or next-item substitution.

### Saturation/Discrimination Location

After benchmark freeze and the complete Dev pipeline dry-run, run Raw, best-fixed, H_ref, `oracle_transfer`, and `oracle_insample` on Dev-Query. Write `dev_discrimination_report.json` and tag a cell `saturated_under_pool_v1` when its Dev-Query `oracle_insample - H_ref` absolute sMASE gain is `<= 0.02` (`saturation_gap_kind: conventional`). This report is frozen before the Final campaign manifest. Final oracle results are post-hoc confirmations only and cannot change the saturation tag, pool, dose, or method roster.

### External Design Provenance

The version-control-external addendum currently has SHA256:

```text
468c65fbcb36f48a47a351597f99d9ccebd876fff39d3378923500a8c3ed45ff
```

Mirror it into the repository and record both this SHA and design commit `9e57da9` in the benchmark manifest.

## File Map

```text
SelfEvolvingHarnessTS/benchmark/
  __init__.py          public protocol constants
  sources.py           automatic/manual source specifications and acquisition
  registry.py          series registry schema, JSONL IO, admission checks
  probe.py             read-only structural census and regime features
  materialize.py       raw immutability, clean-base conversion, content hashes
  split.py             Support-A/B/Dev-Query/Final-Query/U manifest
  corruption.py        content-keyed corruption and CRN artifacts
  ingestion.py         canonical NaN handling and fill-rate accounting
  method_api.py        prepare/adapt/feedback contracts and validation
  trainers.py          closed-form adapter, weighted Adam-DLinear, weighted LSTM
  metrics.py           sMASE and unique gain semantics
  aggregate.py         repeat folding, cells, macro aggregation, bootstrap
  baselines.py         Raw, best-fixed, H_ref, two privileged oracles
  ledger.py            campaign WAL, locks, hash chain, resume validation
  runner.py            phase orchestration and Final access gating
  report.py            Dev discrimination and Final reports
SelfEvolvingHarnessTS/run_benchmark.py
SelfEvolvingHarnessTS/tests/test_benchmark_*.py
SelfEvolvingHarnessTS/data/benchmark_v0/incoming/README.md
SelfEvolvingHarnessTS/docs/benchmark/Benchmark_v0_Forecast_Design_v3_Addendum_2026-07-13.md
```

---

### Task 1: Freeze protocol constants, design provenance, and Dev-Query split

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/split.py`
- Create: `tests/test_benchmark_split.py`
- Create: `docs/benchmark/Benchmark_v0_Forecast_Design_v3_Addendum_2026-07-13.md`

**Interfaces:**
- Produces: `SplitRole`, `SplitCandidate`, `SplitAssignment`, `SplitManifest`, `build_split_manifest(...)`, `validate_split_manifest(...)`.
- Consumes later: registry rows converted to `SplitCandidate`; U selected uid set.

- [ ] **Step 1: Write failing split and provenance tests**

```python
def test_dev_query_is_independent_repeatable_role():
    roles = role_from_unit_interval
    assert roles(0.24) is SplitRole.SUPPORT_A
    assert roles(0.25) is SplitRole.SUPPORT_B
    assert roles(0.45) is SplitRole.DEV_QUERY
    assert roles(0.65) is SplitRole.FINAL_QUERY

def test_overlap_group_is_atomic_and_forced_roles_win():
    rows = [
        SplitCandidate("a", "d", "r", "g", "certified_virgin"),
        SplitCandidate("b", "d", "r", "g", "certified_virgin"),
        SplitCandidate("legacy", "d", "r", "legacy-g", "confirmed_exposed"),
    ]
    manifest = build_split_manifest(rows, "benchmark-v0", "split-salt-v0", set())
    assert manifest.assignment("a").role == manifest.assignment("b").role
    assert manifest.assignment("legacy").role is SplitRole.SUPPORT_A

def test_dev_query_policy_cannot_alias_support_b():
    policy = SplitManifest.role_policies()[SplitRole.DEV_QUERY]
    assert policy == {
        "repeatable": True,
        "utility_visible": True,
        "may_select_best_fixed": False,
        "may_train_oracle_transfer": False,
        "may_confirm_method": False,
        "final_eligible": False,
    }
```

- [ ] **Step 2: Run the split tests and verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_split.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'SelfEvolvingHarnessTS.benchmark'`.

- [ ] **Step 3: Implement the split interfaces and mirror the addendum**

```python
class SplitRole(str, Enum):
    SUPPORT_A = "support_a"
    SUPPORT_B = "support_b"
    DEV_QUERY = "dev_query"
    FINAL_QUERY = "final_query"
    U = "u"

def role_from_unit_interval(u: float) -> SplitRole:
    if not 0.0 <= u < 1.0:
        raise ValueError("u must be in [0,1)")
    if u < 0.25:
        return SplitRole.SUPPORT_A
    if u < 0.45:
        return SplitRole.SUPPORT_B
    if u < 0.65:
        return SplitRole.DEV_QUERY
    return SplitRole.FINAL_QUERY

def group_hash_value(version: str, salt: str, group_key: str) -> float:
    raw = hashlib.sha256(f"{version}|outer|{salt}|{group_key}".encode()).digest()
    return int.from_bytes(raw[:8], "big") / float(1 << 64)
```

`build_split_manifest` groups rows by overlap group, rejects Support-A/U force conflicts, applies forced roles before the hash role, records chronological boundaries, and stores the Dev-Query role policy verbatim. Mirror the approved addendum bytes into `docs/benchmark/` and add its external SHA plus commit `9e57da9` to manifest provenance.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_split.py -q`

Expected: PASS; boundaries 0.25/0.45/0.65 and atomic groups are covered.

- [ ] **Step 5: Commit Task 1**

```text
rtk git add benchmark/__init__.py benchmark/split.py tests/test_benchmark_split.py docs/benchmark/Benchmark_v0_Forecast_Design_v3_Addendum_2026-07-13.md
rtk git commit -m "feat: define benchmark split and dev query"
```

### Task 2: Implement source specifications, registry, and read-only probe

**Files:**
- Create: `benchmark/sources.py`
- Create: `benchmark/registry.py`
- Create: `benchmark/probe.py`
- Create: `tests/test_benchmark_sources_registry.py`
- Create: `tests/test_benchmark_probe.py`

**Interfaces:**
- Produces: `SourceSpec`, `SeriesRecord`, `Admission`, `probe_series(...)`, `probe_registry(...)`.
- Consumes: local raw source files only; no trainers, metrics, or method implementations.

- [ ] **Step 1: Write failing schema and probe tests**

```python
def test_registry_keeps_natural_missing_mask_and_legacy_83():
    row = SeriesRecord.from_values(
        dataset_id="x", entity_id="e", values=np.array([1.0, np.nan, 3.0]),
        source_revision="rev", license_id="cc-by-4.0", exposure_class="certified_virgin",
    )
    assert row.natural_missing_rate == pytest.approx(1 / 3)
    assert row.natural_missing_count == 1

def test_probe_returns_structure_without_loss_fields():
    result = probe_series(np.sin(np.arange(240) * 2 * np.pi / 24), period=24)
    assert set(result) >= {"seasonal_strength", "trend_strength", "spectral_entropy"}
    assert not ({"loss", "utility", "gain"} & set(result))
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_sources_registry.py SelfEvolvingHarnessTS/tests/test_benchmark_probe.py -q`

Expected: FAIL because source, registry, and probe modules do not exist.

- [ ] **Step 3: Implement exact source and registry contracts**

```python
@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    access: Literal["automatic", "manual"]
    official_url: str
    license_id: str
    expected_frequency: str
    overlap_family: str

@dataclass(frozen=True)
class SeriesRecord:
    series_uid: str
    dataset_id: str
    entity_id: str
    content_sha: str
    frequency: str
    length: int
    natural_missing_count: int
    natural_missing_rate: float
    overlap_group: str
    exposure_class: str
    regime_tag: str
    roles_allowed: tuple[str, ...]
```

Register official endpoints: pinned Monash HF; DCRNN author's METR-LA Drive object `10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX`; UCI dataset 321; NOAA Global Hourly; ENTSO-E Transparency Platform; Kaggle GEFCom 2012/2014. `probe_series` uses STL on clean inner-train, preserves NOAA irregular/missing diagnostics, and never imports `trainers`, `metrics`, or `baselines`.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_sources_registry.py SelfEvolvingHarnessTS/tests/test_benchmark_probe.py -q`

Expected: PASS, including natural-missing and no-loss probe assertions.

- [ ] **Step 5: Commit Task 2**

```text
rtk git add benchmark/sources.py benchmark/registry.py benchmark/probe.py tests/test_benchmark_sources_registry.py tests/test_benchmark_probe.py
rtk git commit -m "feat: add benchmark sources registry and probe"
```

### Task 3: Implement immutable acquisition and clean-base materialization

**Files:**
- Create: `benchmark/materialize.py`
- Create: `tests/test_benchmark_materialize.py`
- Create: `data/benchmark_v0/incoming/README.md`

**Interfaces:**
- Produces: `write_raw_once(...)`, `verify_raw_asset(...)`, `resample_hourly(...)`, `materialize_clean_base(...)`.
- Consumes: `SourceSpec`; outputs `SeriesRecord` candidates plus arrays/masks.

- [ ] **Step 1: Write failing immutability and resampling tests**

```python
def test_raw_write_is_immutable(tmp_path):
    asset = write_raw_once(tmp_path / "raw.bin", b"abc", source_revision="r1")
    assert asset.sha256 == hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(RawMutationError):
        write_raw_once(tmp_path / "raw.bin", b"abd", source_revision="r1")

def test_hourly_mean_keeps_raw_missing_mask():
    index = pd.date_range("2020-01-01", periods=8, freq="15min", tz="UTC")
    values = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0], index=index)
    out, mask = resample_hourly(values)
    assert out.tolist() == [pytest.approx(8 / 3), 8.0]
    assert mask.tolist() == [True, False]
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_materialize.py -q`

Expected: FAIL because materialization interfaces are missing.

- [ ] **Step 3: Implement acquisition/materialization and manual instructions**

```python
def write_raw_once(path: Path, payload: bytes, *, source_revision: str) -> RawAsset:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != digest:
            raise RawMutationError(f"immutable raw asset mismatch: {path}")
        return RawAsset(path, current, source_revision, len(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    return RawAsset(path, digest, source_revision, len(payload))
```

Automatic acquisition is resumable to a temporary file, hash-verified, then atomically promoted. `incoming/README.md` names the official ENTSO-E Actual Total Load export and both Kaggle competition pages, exact incoming roots, timezone requirement (UTC), and importer status command. Clean-base writes values, timestamps, and the natural-missing mask separately.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_materialize.py -q`

Expected: PASS; a changed raw payload fails and hourly mask behavior is exact.

- [ ] **Step 5: Commit Task 3**

```text
rtk git add benchmark/materialize.py tests/test_benchmark_materialize.py data/benchmark_v0/incoming/README.md
rtk git commit -m "feat: materialize immutable benchmark data"
```

### Task 4: Implement content-keyed corruption and canonical ingestion

**Files:**
- Create: `benchmark/corruption.py`
- Create: `benchmark/ingestion.py`
- Create: `tests/test_benchmark_corruption.py`
- Create: `tests/test_benchmark_ingestion.py`

**Interfaces:**
- Produces: `corruption_seed(...)`, `apply_corruption(...)`, `canonical_ingest(...)`, `IngestionResult`.

- [ ] **Step 1: Write failing reorder/subset and ingestion tests**

```python
def test_corruption_is_reorder_and_subset_invariant():
    specs = [("a", "aa" * 32), ("b", "bb" * 32), ("c", "cc" * 32)]
    full = materialize_corruptions(specs, "block", 0.12, 0, "benchmark-v0")
    rev = materialize_corruptions(list(reversed(specs)), "block", 0.12, 0, "benchmark-v0")
    sub = materialize_corruptions([specs[1]], "block", 0.12, 0, "benchmark-v0")
    assert np.array_equal(full["b"], rev["b"], equal_nan=True)
    assert np.array_equal(full["b"], sub["b"], equal_nan=True)

def test_ingestion_counts_fill_and_rejects_infinity():
    got = canonical_ingest(np.array([np.nan, 2.0, np.nan, 4.0, np.nan]))
    assert got.values.tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]
    assert got.fill_rate == pytest.approx(0.6)
    with pytest.raises(IngestionInvalid):
        canonical_ingest(np.array([1.0, np.inf]))
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_corruption.py SelfEvolvingHarnessTS/tests/test_benchmark_ingestion.py -q`

Expected: FAIL because corruption and ingestion modules are missing.

- [ ] **Step 3: Implement the unique key and ingestion rule**

```python
def corruption_seed(version: str, content_sha: str, scenario: str,
                    dose: float, replicate_idx: int) -> int:
    payload = json.dumps(
        [version, content_sha, scenario, format(float(dose), ".17g"), int(replicate_idx)],
        ensure_ascii=True, separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")

def canonical_ingest(values: np.ndarray) -> IngestionResult:
    x = np.asarray(values, dtype=np.float64).ravel()
    if np.isinf(x).any() or x.size == 0 or np.isnan(x).all():
        raise IngestionInvalid("input must contain finite values and no infinity")
    missing = np.isnan(x)
    y = x.copy()
    idx = np.arange(y.size)
    y[missing] = np.interp(idx[missing], idx[~missing], y[~missing])
    return IngestionResult(y, int(missing.sum()), float(missing.mean()), missing.mean() > 0.01)
```

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_corruption.py SelfEvolvingHarnessTS/tests/test_benchmark_ingestion.py -q`

Expected: PASS, including order/subset invariance.

- [ ] **Step 5: Commit Task 4**

```text
rtk git add benchmark/corruption.py benchmark/ingestion.py tests/test_benchmark_corruption.py tests/test_benchmark_ingestion.py
rtk git commit -m "feat: add keyed corruption and canonical ingestion"
```

### Task 5: Implement Method API, feedback accounting, and contract gates

**Files:**
- Create: `benchmark/method_api.py`
- Create: `tests/test_benchmark_method_api.py`

**Interfaces:**
- Produces: `BenchmarkMethod`, `PreparedSeries`, `FeedbackAPI`, `ContractVerdict`, `validate_prepared(...)`, `run_support_dry_run(...)`, `run_support_b_confirmation(...)`.

- [ ] **Step 1: Write failing visibility, normalization, and dry-run tests**

```python
def test_method_view_excludes_private_fields():
    view = MethodSeriesView.from_episode(private_episode_fixture())
    assert not hasattr(view, "future")
    assert not hasattr(view, "regime_tag")
    assert not hasattr(view, "split_role")

def test_target_space_transform_is_forbidden():
    candidate = PreparedSeries("u", np.arange(10.0), ("znorm",), "original_units")
    verdict = validate_prepared(candidate, expected_length=10)
    assert verdict.code == "forbidden_target_space_transform"

def test_query_requires_support_artifact_shas():
    with pytest.raises(MethodGateError):
        require_final_eligibility("m", dry_run_sha=None, confirmation_sha=None)
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_method_api.py -q`

Expected: FAIL because Method API is missing.

- [ ] **Step 3: Implement method views, validation, and feedback ledger**

```python
@runtime_checkable
class BenchmarkMethod(Protocol):
    method_id: str
    seed: int
    def prepare(self, train_series: MethodSeriesView,
                task_spec: TaskSpec,
                observed_pattern_spec: Mapping[str, float]) -> PreparedSeries: ...

FORBIDDEN_TARGET_SPACE_OPS = frozenset(
    name for name, meta in OPERATOR_METADATA.items()
    if bool(meta.get("changes_target_space"))
)
```

`FeedbackAPI.evaluate` accepts only Support-A inner-val handles and the closed-form channel, decrements a frozen budget, and writes call records. Support-A dry-run hashes method/code/config/contract output. Support-B confirmation is one-shot per frozen method SHA and cannot be overwritten.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_method_api.py -q`

Expected: PASS; private fields and target-space transforms are mechanically blocked.

- [ ] **Step 5: Commit Task 5**

```text
rtk git add benchmark/method_api.py tests/test_benchmark_method_api.py
rtk git commit -m "feat: define benchmark method contract"
```

### Task 6: Implement shared windows and series-equal downstream trainers

**Files:**
- Create: `benchmark/trainers.py`
- Create: `tests/test_benchmark_trainers.py`

**Interfaces:**
- Produces: `NormalizationState`, `WindowBatch`, `build_windows(...)`, `series_equal_batch_loss(...)`, `train_adam_dlinear(...)`, `train_lstm_reporter(...)`, `fit_closed_form(...)`.

- [ ] **Step 1: Write failing exact-weight and deterministic-order tests**

```python
def test_series_equal_full_objective():
    per_window = torch.tensor([1.0, 3.0, 9.0])
    series = np.array(["a", "a", "b"])
    assert series_equal_full_loss(per_window, series).item() == pytest.approx(5.5)

def test_batch_formula_handles_short_final_batch():
    losses = torch.tensor([2.0])
    weights = torch.tensor([0.5])
    assert series_equal_batch_loss(losses, weights, n_windows=3, n_series=2).item() == pytest.approx(1.5)

def test_same_seed_replays_window_order():
    assert window_order(17, 11, 0) == window_order(17, 11, 0)
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_trainers.py -q`

Expected: FAIL because benchmark trainers are missing.

- [ ] **Step 3: Implement the frozen objective**

```python
def series_equal_batch_loss(losses: torch.Tensor, weights: torch.Tensor,
                            *, n_windows: int, n_series: int) -> torch.Tensor:
    batch_n = int(losses.shape[0])
    if batch_n < 1 or n_windows < batch_n or n_series < 1:
        raise ValueError("invalid batch dimensions")
    return (float(n_windows) / (batch_n * n_series)) * torch.sum(weights * losses)

def window_weights(series_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(series_ids)
    return np.array([1.0 / counts[s] for s in series_ids], dtype=np.float64)
```

Build normalization from finite pre-method degraded inner-train, freeze it, ingest before normalization, use inner-train-only stride-4 windows, and reuse existing `DLinear`/`LSTMForecaster` classes. Freeze Adam defaults explicitly: epochs 120, lr `1e-2`, batch 256, betas `(0.9,0.999)`, eps `1e-8`, weight decay 0, deterministic CPU.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_trainers.py -q`

Expected: PASS; weighted sampling is absent and both loss formulas are exact.

- [ ] **Step 5: Commit Task 6**

```text
rtk git add benchmark/trainers.py tests/test_benchmark_trainers.py
rtk git commit -m "feat: add series equal benchmark trainers"
```

### Task 7: Implement sMASE, unique gains, repeat folding, and bootstrap

**Files:**
- Create: `benchmark/metrics.py`
- Create: `benchmark/aggregate.py`
- Create: `tests/test_benchmark_metrics.py`
- Create: `tests/test_benchmark_aggregate.py`

**Interfaces:**
- Produces: `seasonal_scale(...)`, `smase(...)`, `gain(...)`, `LossRow`, `collapse_uid_gains(...)`, `aggregate_cells(...)`, `bootstrap_ci90(...)`.

- [ ] **Step 1: Write failing metric and axis-completeness tests**

```python
def test_smase_uses_only_observed_training_pairs():
    y = np.array([1., 2., 3., 2., 3., 4.])
    observed = np.array([1, 1, 1, 1, 0, 1], dtype=bool)
    scale = seasonal_scale(y, observed, period=3, min_pairs=2)
    assert scale == pytest.approx(1.0)

def test_collapse_requires_3_by_2_by_all_doses():
    rows = complete_loss_rows(model_seeds=(0, 1), replicates=(0, 1), doses=(0.12, 0.24))
    with pytest.raises(AggregationContractError, match="model seeds"):
        collapse_uid_gains(rows, expected_model_seeds=(0, 1, 2), expected_replicates=(0, 1))

def test_bootstrap_accepts_one_row_per_uid_only():
    with pytest.raises(AggregationContractError, match="one row per uid"):
        bootstrap_ci90([("u", 0.1), ("u", 0.2)], b=2000, seed=20260713)
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_metrics.py SelfEvolvingHarnessTS/tests/test_benchmark_aggregate.py -q`

Expected: FAIL because metric and aggregation modules are missing.

- [ ] **Step 3: Implement the one-way folding state machine**

```python
FOLD_AXES = ("model_seed", "corruption_replicate", "scenario_dose", "uid")

def gain(reference_loss: float, method_loss: float) -> float:
    values = np.asarray([reference_loss, method_loss], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("gain inputs must be finite")
    return float(reference_loss - method_loss)

def bootstrap_ci90(uid_gain: Mapping[str, float], *, b: int = 2000,
                   seed: int = 20260713) -> tuple[float, float]:
    if len(uid_gain) != len(set(uid_gain)) or not uid_gain:
        raise AggregationContractError("bootstrap requires one row per uid")
    vals = np.fromiter(uid_gain.values(), dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(vals), size=(b, len(vals)))
    means = vals[draw].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.05, 0.95], method="linear"))
```

`collapse_uid_gains` validates the complete expected cartesian axes, folds model seeds, then replicates, then equal scenario-by-dose values, and returns one uid row. Manifest writes `harm_threshold=0.05` and `harm_threshold_kind=conventional`.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_metrics.py SelfEvolvingHarnessTS/tests/test_benchmark_aggregate.py -q`

Expected: PASS; duplicate uid bootstrap and incomplete repeat axes fail loud.

- [ ] **Step 5: Commit Task 7**

```text
rtk git add benchmark/metrics.py benchmark/aggregate.py tests/test_benchmark_metrics.py tests/test_benchmark_aggregate.py
rtk git commit -m "feat: add benchmark metrics and aggregation"
```

### Task 8: Implement baselines, privileged oracles, and Dev saturation report

**Files:**
- Create: `benchmark/baselines.py`
- Create: `benchmark/report.py`
- Create: `tests/test_benchmark_baselines.py`
- Create: `tests/test_benchmark_discrimination.py`

**Interfaces:**
- Produces: `RawBaseline`, `select_best_fixed(...)`, `HRefBaseline`, `oracle_transfer(...)`, `oracle_insample(...)`, `build_dev_discrimination_report(...)`.

- [ ] **Step 1: Write failing oracle privilege and execution-location tests**

```python
def test_oracles_are_not_public_methods():
    assert not isinstance(OracleTransfer(), BenchmarkMethod)
    assert not isinstance(OracleInSample(), BenchmarkMethod)

def test_saturation_is_created_only_from_dev_query():
    report = build_dev_discrimination_report(dev_rows_fixture(), saturation_gap=0.02)
    assert report["split_role"] == "dev_query"
    assert report["cells"]["d|r"]["tag"] == "saturated_under_pool_v1"
    with pytest.raises(ReportProtocolError):
        build_dev_discrimination_report(final_rows_fixture(), saturation_gap=0.02)
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_baselines.py SelfEvolvingHarnessTS/tests/test_benchmark_discrimination.py -q`

Expected: FAIL because baseline/report modules are missing.

- [ ] **Step 3: Implement baseline selection boundaries and both oracles**

```python
def oracle_transfer(support_a_losses: Sequence[ProgramLoss],
                    query_losses: Sequence[ProgramLoss]) -> list[ProgramLoss]:
    mapping = best_program_per_cell(support_a_losses)
    return [row for row in query_losses if row.program_id == mapping[row.cell_id]]

def oracle_insample(query_losses: Sequence[ProgramLoss]) -> list[ProgramLoss]:
    mapping = best_program_per_cell(query_losses)
    return [row for row in query_losses if row.program_id == mapping[row.cell_id]]
```

Raw wraps no-op plus ingestion. best-fixed can consume only Support-A. H_ref filters `changes_target_space=True` actions and freezes its random seed/budget. Dev saturation requires at least 12 uid in the cell and sets `saturation_gap=0.02`, `saturation_gap_kind=conventional`; otherwise tag `diagnostic_unavailable`.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_baselines.py SelfEvolvingHarnessTS/tests/test_benchmark_discrimination.py -q`

Expected: PASS; Final rows cannot generate the pre-campaign saturation report.

- [ ] **Step 5: Commit Task 8**

```text
rtk git add benchmark/baselines.py benchmark/report.py tests/test_benchmark_baselines.py tests/test_benchmark_discrimination.py
rtk git commit -m "feat: add benchmark baselines and dev diagnostics"
```

### Task 9: Implement campaign ledger, terminal failures, and exact resume

**Files:**
- Create: `benchmark/ledger.py`
- Create: `tests/test_benchmark_ledger.py`

**Interfaces:**
- Produces: `CampaignManifest`, `CampaignLedger`, `MethodResultStatus`, `ResumeBinding`.
- Consumes: complete Final roster including methods, baselines, and both oracle diagnostics.

- [ ] **Step 1: Write failing WAL-order, roster, and resume tests**

```python
def test_unseal_and_access_are_durable_before_read(tmp_path):
    ledger = frozen_ledger(tmp_path)
    ledger.unseal()
    ledger.record_access("m", "run-1")
    assert [e["event"] for e in ledger.events()] == [
        "campaign_freeze", "unseal", "method_access"
    ]

def test_unrostered_oracle_cannot_access_final(tmp_path):
    ledger = frozen_ledger(tmp_path, roster=("raw", "method"))
    ledger.unseal()
    with pytest.raises(CampaignStateError, match="roster"):
        ledger.record_access("oracle_insample", "run-o")

def test_method_invalid_is_terminal_but_exact_resume_is_idempotent(tmp_path):
    ledger = frozen_ledger(tmp_path)
    ledger.record_result("m", "run-1", MethodResultStatus.INVALID, "digest")
    with pytest.raises(CampaignStateError):
        ledger.record_access("m", "run-2")
    assert ledger.resume(exact_resume_binding()) == exact_resume_binding().run_id
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_ledger.py -q`

Expected: FAIL because ledger module is missing.

- [ ] **Step 3: Adapt the P6 WAL mechanics without importing mutable P6 state**

```python
class MethodResultStatus(str, Enum):
    COMPLETE = "complete"
    INVALID = "invalid"
    FAILED_TIMEOUT = "failed_timeout"
    INFRA_INTERRUPTED = "infra_interrupted"

REQUIRED_RESUME_FIELDS = (
    "campaign_id", "run_id", "entry_id", "method_code_sha", "runner_code_sha",
    "input_manifest_sha", "materialization_sha", "checkpoint_sha",
)
```

Implement exclusive lock, canonical event JSON, `prev_event_sha`, `event_sha`, append+fsync-before-state, replay validation, full-roster freeze, one unseal, pre-read access, terminal method result, and close. Resume accepts only an `INFRA_INTERRUPTED` run with byte-equal required bindings; any field drift fails.

- [ ] **Step 4: Verify GREEN**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_ledger.py -q`

Expected: PASS, including tamper/chain tests and complete roster enforcement.

- [ ] **Step 5: Commit Task 9**

```text
rtk git add benchmark/ledger.py tests/test_benchmark_ledger.py
rtk git commit -m "feat: add final evaluation campaign ledger"
```

### Task 10: Integrate runner, Final loader gate, CLI, and report artifacts

**Files:**
- Create: `benchmark/runner.py`
- Create: `run_benchmark.py`
- Create: `tests/test_benchmark_runner.py`
- Create: `tests/test_run_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkRunner.probe/freeze/dry_run/confirm/run_dev/freeze_campaign/run_final`, CLI subcommands `acquire`, `probe`, `freeze`, `dry-run`, `confirm`, `dev-eval`, `campaign-freeze`, `final-eval`.

- [ ] **Step 1: Write failing phase-order and Final-read-gate tests**

```python
def test_final_loader_commits_access_before_loading(monkeypatch, frozen_runner):
    calls = []
    monkeypatch.setattr(frozen_runner.ledger, "record_access", lambda *a: calls.append("access"))
    monkeypatch.setattr(frozen_runner.final_store, "load", lambda *a: calls.append("read") or [])
    frozen_runner.load_final_for_entry("method", "run-1")
    assert calls == ["access", "read"]

def test_final_requires_dev_report_support_gates_and_full_roster(frozen_runner):
    frozen_runner.dev_discrimination_sha = None
    with pytest.raises(RunnerGateError, match="Dev discrimination"):
        frozen_runner.freeze_campaign(complete_roster_fixture())
```

- [ ] **Step 2: Verify RED**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_runner.py SelfEvolvingHarnessTS/tests/test_run_benchmark.py -q`

Expected: FAIL because runner and CLI are missing.

- [ ] **Step 3: Implement orchestration and artifact contracts**

```python
FINAL_ROSTER_REQUIRED = frozenset({
    "raw", "best_fixed", "h_ref", "oracle_transfer", "oracle_insample"
})

def validate_campaign_roster(entries: Sequence[CampaignEntry]) -> None:
    ids = {entry.entry_id for entry in entries}
    missing = FINAL_ROSTER_REQUIRED - ids
    if missing:
        raise RunnerGateError(f"Final roster missing required entries: {sorted(missing)}")
```

Runner writes the §11 artifact tree, records exact protocol doc SHA/commit, computes numeric prepare/trainer timeouts from same-hardware Dev `2*p95`, refuses campaign freeze without `dev_discrimination_report.json`, and converts method-dependent exceptions to terminal ledger results while preserving exact infrastructure-resume bindings.

- [ ] **Step 4: Run focused and full benchmark tests**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests/test_benchmark_*.py SelfEvolvingHarnessTS/tests/test_run_benchmark.py -q`

Expected: PASS with no Final fixture read before a ledger access event.

- [ ] **Step 5: Commit Task 10**

```text
rtk git add benchmark/runner.py run_benchmark.py tests/test_benchmark_runner.py tests/test_run_benchmark.py
rtk git commit -m "feat: integrate benchmark runner and cli"
```

### Task 11: Acquire data, run probe, freeze v0, and validate on Dev-Query

**Files:**
- Create: `data/benchmark_v0/raw/**`
- Create: `data/benchmark_v0/clean_base/**`
- Create: `results/Benchmark_v0/benchmark_manifest_v0.yaml`
- Create: `results/Benchmark_v0/data_card.md`
- Create: `results/Benchmark_v0/series_registry.jsonl`
- Create: `results/Benchmark_v0/split_manifest.json`
- Create: `results/Benchmark_v0/training_evaluation_protocol.md`
- Create: `results/Benchmark_v0/virgin_ledger.jsonl`
- Create: `results/Benchmark_v0/dev_discrimination_report.json`
- Create: `results/Benchmark_v0/baseline_report.md`

**Interfaces:**
- Consumes all prior modules.
- Produces the runnable frozen v0-core plus v1 candidate registry; does not open Final-Query.

- [ ] **Step 1: Run automatic acquisition**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_benchmark acquire --automatic --root SelfEvolvingHarnessTS/data/benchmark_v0`

Expected: Monash, METR-LA, UCI ELD, and NOAA raw assets are present with source revisions and SHA256. ENTSO-E/GEFCom report `manual_required` with exact incoming paths, not failure.

- [ ] **Step 2: Run registry/probe and inspect NOAA admission**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_benchmark probe --root SelfEvolvingHarnessTS/data/benchmark_v0 --out SelfEvolvingHarnessTS/results/Benchmark_v0`

Expected: registry includes natural missingness and irregular-sampling diagnostics for NOAA; no loss/utility fields exist.

- [ ] **Step 3: Freeze split and protocol without opening Final**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_benchmark freeze --root SelfEvolvingHarnessTS/data/benchmark_v0 --out SelfEvolvingHarnessTS/results/Benchmark_v0`

Expected: explicit Support-A, Support-B, Dev-Query, Final-Query, and U roles; overlap groups are atomic; Final files are sealed; no `unseal` event exists.

- [ ] **Step 4: Run all baseline paths on Dev-Query and freeze saturation report**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_benchmark dev-eval --split dev_query --baselines raw,best_fixed,h_ref,oracle_transfer,oracle_insample --out SelfEvolvingHarnessTS/results/Benchmark_v0`

Expected: `dev_discrimination_report.json` exists before any campaign manifest, small cells are `diagnostic_unavailable`, and eligible cells have frozen saturation tags.

- [ ] **Step 5: Run full verification and commit non-bulk protocol artifacts**

Run: `rtk test D:/Anaconda_envs/envs/project/python.exe -m pytest SelfEvolvingHarnessTS/tests -q`

Expected: all tests pass. Inspect repository policy before staging raw data; commit code, tests, manifests, registry hashes, data card, and reports, but do not commit large raw archives unless repository policy explicitly tracks them.

```text
rtk git add benchmark run_benchmark.py tests/test_benchmark_*.py tests/test_run_benchmark.py docs/benchmark data/benchmark_v0/incoming/README.md results/Benchmark_v0
rtk git commit -m "feat: freeze benchmark v0 core pipeline"
```

## Verification Matrix

| Approved requirement | Primary test file |
|---|---|
| Dev-Query membership, semantics, disjoint roles | `tests/test_benchmark_split.py` |
| Raw immutability and resampling | `tests/test_benchmark_materialize.py` |
| Registry, overlap, exposure, legacy 83 | `tests/test_benchmark_sources_registry.py` |
| NOAA missingness and loss-free probe | `tests/test_benchmark_probe.py` |
| Corruption reorder/subset invariance and CRN | `tests/test_benchmark_corruption.py` |
| Canonical ingestion and fill-rate accounting | `tests/test_benchmark_ingestion.py` |
| Method visibility, normalization ownership, support gates | `tests/test_benchmark_method_api.py` |
| Series-equal weighted loss and deterministic shuffle | `tests/test_benchmark_trainers.py` |
| sMASE and unique gain semantics | `tests/test_benchmark_metrics.py` |
| Model seed -> replicate -> scenario/dose -> uid folding | `tests/test_benchmark_aggregate.py` |
| Baseline boundaries and oracle privilege | `tests/test_benchmark_baselines.py` |
| Dev-only saturation/discrimination report | `tests/test_benchmark_discrimination.py` |
| WAL order, roster completeness, tamper detection, resume | `tests/test_benchmark_ledger.py` |
| Final preconditions and ledger-before-read | `tests/test_benchmark_runner.py` |
| CLI phase ordering and artifact contracts | `tests/test_run_benchmark.py` |
