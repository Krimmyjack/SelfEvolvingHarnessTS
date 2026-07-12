# Benchmark v0 Implementation Plan Amendment 1

> Normative self-review corrections to
> `2026-07-13-benchmark-v0-data-metrics-pipeline.md`. All other tasks remain unchanged.

## Task 2 test addition: assert the real legacy inventory

Add this test to `tests/test_benchmark_sources_registry.py` before implementing
the registry importer:

```python
def test_existing_legacy_inventory_is_exactly_83():
    path = Path("SelfEvolvingHarnessTS/data/_artifacts/monash_clean.meta.jsonl")
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
    counts = Counter(row["config"] for row in rows)
    assert counts == {
        "nn5_daily": 20,
        "fred_md": 20,
        "tourism_monthly": 20,
        "covid_deaths": 20,
        "us_births": 1,
        "saugeenday": 1,
        "sunspot": 1,
    }
    assert len(rows) == 83
```

The imported rows must all receive `exposure_class="confirmed_exposed"` and
Support-A-only roles.

## Task 4 correction: corruption invariance fixture includes values

Replace the original invariance test with:

```python
def test_corruption_is_reorder_and_subset_invariant():
    values = {
        "a": np.linspace(0.0, 1.0, 240),
        "b": np.linspace(1.0, 2.0, 240),
        "c": np.linspace(2.0, 3.0, 240),
    }
    hashes = {uid: hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()
              for uid, x in values.items()}
    full = materialize_corruptions(values, hashes, "block", 0.12, 0, "benchmark-v0")
    reversed_values = dict(reversed(list(values.items())))
    reversed_hashes = {uid: hashes[uid] for uid in reversed_values}
    rev = materialize_corruptions(
        reversed_values, reversed_hashes, "block", 0.12, 0, "benchmark-v0"
    )
    sub = materialize_corruptions(
        {"b": values["b"]}, {"b": hashes["b"]}, "block", 0.12, 0, "benchmark-v0"
    )
    assert np.array_equal(full["b"], rev["b"], equal_nan=True)
    assert np.array_equal(full["b"], sub["b"], equal_nan=True)
```

The production signature is fixed as:

```python
def materialize_corruptions(
    values_by_uid: Mapping[str, np.ndarray],
    content_sha_by_uid: Mapping[str, str],
    scenario: str,
    dose: float,
    replicate_idx: int,
    benchmark_version: str,
) -> dict[str, np.ndarray]:
    ...
```

## Task 9 correction: terminal method failure and infrastructure resume use valid event sequences

Replace the combined ledger test with two tests:

```python
def test_method_invalid_is_terminal(tmp_path):
    ledger = frozen_ledger(tmp_path, roster=("m",))
    ledger.unseal()
    ledger.record_access("m", "run-1")
    ledger.record_result("m", "run-1", MethodResultStatus.INVALID, "digest")
    with pytest.raises(CampaignStateError, match="terminal"):
        ledger.record_access("m", "run-2")

def test_exact_infrastructure_resume_is_idempotent(tmp_path):
    binding = exact_resume_binding()
    ledger = frozen_ledger(tmp_path, roster=(binding.entry_id,))
    ledger.unseal()
    ledger.record_access(binding.entry_id, binding.run_id)
    ledger.record_result(
        binding.entry_id,
        binding.run_id,
        MethodResultStatus.INFRA_INTERRUPTED,
        binding.checkpoint_sha,
        resume_binding=binding,
    )
    assert ledger.resume(binding) == binding.run_id
    changed = dataclasses.replace(binding, runner_code_sha="different")
    with pytest.raises(CampaignStateError, match="resume binding"):
        ledger.resume(changed)
```

This preserves the approved distinction: method invalidity never creates a
resume path; only an explicitly recorded infrastructure interruption can resume.
