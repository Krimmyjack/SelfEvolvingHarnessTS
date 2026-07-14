"""The frozen benchmark-v0.2 baseline program pool.

v0.1's public pool held four programs -- Raw, forward fill, seasonal fill, and the P6
reference ladder -- and every one of them fills missing values.  The v0.1 Dev-Query run
made the consequence unmissable: on the `spike`, `gaussian`, `level_shift`, and
`local_permutation` lanes, all four programs scored the same loss to four decimal places,
because not one of them can touch an outlier, a noise floor, a structural break, or a
shuffled ordering.  A pool that cannot act differently cannot make "choosing" mean
anything, so C1 -- the claim that the best preparation depends on the condition -- was
untestable against it.  The saturation was a property of the pool, not of the data.

This module widens the pool by **mechanism coverage**, not by hunting for a winner.  Every
program is frozen with its operator identity and parameters before any v0.2 result is
looked at, and `pool_manifest` pins the SHA256 of the code that implements it, so the pool
cannot drift underneath a published number.

Two mechanisms are left uncovered and declared as capability gaps rather than papered over
with a program that cannot address them (see `CAPABILITY_GAPS`).  A pool that has no answer
for a defect class is a finding about the operator library, and reporting it is the point;
inventing a weak candidate so that every column has an entry would only hide it.

Silent fallback is the failure this project has been bitten by before: an operator that
quietly becomes a different operator forges the conditioning signal the benchmark is built
to measure.  `denoise_stl` and `denoise_wavelet` both carry documented fallbacks to savgol,
and `denoise_savgol`/`denoise_median` silently degrade to numpy equivalents when scipy is
absent.  `assert_pool_dependencies` therefore refuses to build the pool at all unless every
declared dependency imports, and `run_pool_with_provenance` re-executes the pool under the
operator ledger so that any fallback that does fire is counted, not assumed away.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..operators import _provenance
from ..operators._common import _HAS_SCIPY
from ..operators.registry import OPERATOR_METADATA, canonicalize, get_operator
from .ingestion import canonical_ingest
from .method_api import MethodSeriesView, PreparedSeries, validate_prepared

__all__ = [
    "CAPABILITY_GAPS",
    "POOL_SCHEMA_VERSION",
    "PROGRAM_IDS",
    "PoolProgram",
    "ProgramPoolError",
    "ProgramSpec",
    "PROGRAM_SPECS",
    "apply_program",
    "assert_pool_dependencies",
    "mechanism_of",
    "pool_manifest",
    "run_pool_with_provenance",
]

POOL_SCHEMA_VERSION = "benchmark-program-pool/2"


class ProgramPoolError(RuntimeError):
    """The frozen program pool cannot be built or executed as declared."""


@dataclass(frozen=True)
class ProgramSpec:
    """One frozen pool member: an identity, a mechanism, and pinned parameters."""

    program_id: str
    mechanism: str
    operator: str | None
    params: Mapping[str, Any]
    period_param: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "mechanism": self.mechanism,
            "operator": self.operator,
            "params": dict(self.params),
            "period_param": self.period_param,
            "note": self.note,
        }


# Mechanism coverage is the organizing principle.  `reference` is not a mechanism: H_ref is
# the incumbent ladder being measured, and it is in the pool as the thing to beat, not as a
# tool for any particular defect.
PROGRAM_SPECS: tuple[ProgramSpec, ...] = (
    ProgramSpec(
        program_id="raw",
        mechanism="none",
        operator=None,
        params=MappingProxyType({}),
        period_param=None,
        note="No-op + canonical ingestion. The honest floor: what a method must beat.",
    ),
    ProgramSpec(
        program_id="forward_fill",
        mechanism="missing",
        operator=None,
        params=MappingProxyType({}),
        period_param=None,
        note="Benchmark-owned last-observation-carried-forward, leading NaNs back-filled.",
    ),
    ProgramSpec(
        program_id="seasonal_fill",
        mechanism="missing",
        operator=None,
        params=MappingProxyType({}),
        period_param="period",
        note="Benchmark-owned lag-period fill, canonical ingestion for what it cannot reach.",
    ),
    ProgramSpec(
        program_id="winsorize",
        mechanism="outlier_spike",
        operator="winsorize",
        params=MappingProxyType({"limits": 0.05}),
        period_param=None,
        note="Symmetric quantile clipping. Registry default limits, frozen unchanged.",
    ),
    ProgramSpec(
        program_id="denoise_median",
        mechanism="outlier_spike",
        operator="denoise_median",
        params=MappingProxyType({"window": 5}),
        period_param=None,
        note=(
            "Sliding median, symmetric boundary. This is the operator that project results "
            "before the benchmark called `v_median`; the canonical registry name is used "
            "here so the pool cannot drift from the operator library's single source."
        ),
    ),
    ProgramSpec(
        program_id="denoise_stl",
        mechanism="additive_noise",
        operator="denoise_stl",
        params=MappingProxyType({}),
        period_param="period",
        note=(
            "STL trend+seasonal reconstruction, residual dropped. The period is passed "
            "explicitly from the frozen frequency so the operator never reaches its "
            "`_guess_period` branch, whose no-seasonality path falls back to savgol."
        ),
    ),
    ProgramSpec(
        program_id="denoise_savgol",
        mechanism="additive_noise",
        operator="denoise_savgol",
        params=MappingProxyType({"window": 11, "order": 3}),
        period_param=None,
        note="Savitzky-Golay, explicit `interp` endpoints. Registry defaults, frozen.",
    ),
    ProgramSpec(
        program_id="denoise_wavelet",
        mechanism="additive_noise",
        operator="denoise_wavelet",
        params=MappingProxyType({"wavelet": "db4"}),
        period_param=None,
        note="Bounded-level VisuShrink soft threshold, symmetric mode. Registry defaults.",
    ),
    ProgramSpec(
        program_id="h_ref",
        mechanism="reference",
        operator=None,
        params=MappingProxyType({}),
        period_param=None,
        note=(
            "The frozen P6 det+random fast path. Executed by the runner, not by "
            "`apply_program`, because its action is a function of the whole role's episode "
            "state rather than of one series' values."
        ),
    ),
)

PROGRAM_IDS: tuple[str, ...] = tuple(spec.program_id for spec in PROGRAM_SPECS)

_SPEC_OF_ID: dict[str, ProgramSpec] = {spec.program_id: spec for spec in PROGRAM_SPECS}

# Executed by the runner rather than by `apply_program`.
RUNNER_EXECUTED: frozenset[str] = frozenset({"h_ref"})

# Declared before any v0.2 number was read.  These are findings about the operator library,
# not omissions to be quietly filled later: any program added to close one of these gaps is
# a pool change, and a pool change is a benchmark version bump.
CAPABILITY_GAPS: tuple[Mapping[str, str], ...] = (
    MappingProxyType(
        {
            "defect_mechanism": "timestamp_irregularity",
            "corruption_scenario": "local_permutation",
            "why_uncovered": (
                "The corruption is realized in the value domain -- values are permuted "
                "inside disjoint tiles -- because the legacy Monash series carry no "
                "timestamps at all, so a real timestamp array cannot be perturbed "
                "uniformly across the roster without breaking CRN. A 're-align the "
                "timestamps' operator therefore has nothing to grip: the index is already "
                "uniform and correct. Median and smoothing programs are weak candidates at "
                "best, and they are in the pool for other mechanisms, not for this one."
            ),
        }
    ),
    MappingProxyType(
        {
            "defect_mechanism": "structural_break",
            "corruption_scenario": "level_shift",
            "why_uncovered": (
                "`operators/registry.py` holds no changepoint-detection or "
                "segment-normalization operator. Nothing in the pool can find a break, so "
                "nothing in the pool can repair one. This is a gap in the operator library "
                "and is reported as such."
            ),
        }
    ),
)

# Only the files whose code actually executes on the pool path are pinned.  Over-pinning --
# hashing modules the pool never reaches -- would invalidate a frozen pool for edits that
# provably cannot move a single number, and a freeze that cries wolf gets ignored.
_POOL_CODE_FILES: tuple[str, ...] = (
    "benchmark/programs.py",
    "operators/_common.py",
    "operators/registry.py",
    "operators/s1_denoise.py",
    "operators/s1_outlier.py",
)

_DECLARED_DEPENDENCIES: tuple[str, ...] = ("scipy", "statsmodels", "pywt")


def mechanism_of(program_id: str) -> str:
    spec = _SPEC_OF_ID.get(program_id)
    if spec is None:
        raise ProgramPoolError(f"unknown program: {program_id!r}")
    return spec.mechanism


def assert_pool_dependencies() -> dict[str, str]:
    """Refuse to build the pool unless every operator can run as declared.

    scipy absent does not make `denoise_savgol` fail -- it makes it silently become a
    moving average, and `denoise_median` a numpy median filter, with no ledger entry.  A
    pool frozen under that condition would publish numbers attributed to operators that
    never ran.  So the dependency check is a hard gate, not a warning.
    """
    missing: list[str] = []
    for module in _DECLARED_DEPENDENCIES:
        try:
            __import__(module)
        except Exception:  # pragma: no cover - environment-dependent
            missing.append(module)
    if missing:
        raise ProgramPoolError(
            "frozen program pool requires "
            + ", ".join(_DECLARED_DEPENDENCIES)
            + f"; missing: {', '.join(missing)}"
        )
    if not _HAS_SCIPY:  # pragma: no cover - environment-dependent
        raise ProgramPoolError(
            "operators/_common reports scipy unavailable; denoise_savgol and "
            "denoise_median would silently degrade to numpy equivalents"
        )
    return _provenance.dependency_fingerprint()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _code_digests() -> dict[str, str]:
    root = _project_root()
    digests: dict[str, str] = {}
    for relative in _POOL_CODE_FILES:
        path = root / relative
        if not path.is_file():
            raise ProgramPoolError(f"pool code file is missing: {relative}")
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _forward_fill(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    finite = np.flatnonzero(np.isfinite(result))
    if not finite.size:
        raise ProgramPoolError("forward_fill has no finite anchor")
    first = int(finite[0])
    result[:first] = result[first]
    for index in range(first + 1, len(result)):
        if not np.isfinite(result[index]):
            result[index] = result[index - 1]
    return result


def _seasonal_fill(values: np.ndarray, period: int) -> np.ndarray:
    result = values.copy()
    for index in range(len(result)):
        if (
            not np.isfinite(result[index])
            and index >= period
            and np.isfinite(result[index - period])
        ):
            result[index] = result[index - period]
    return canonical_ingest(result).values.copy()


def apply_program(program_id: str, values: np.ndarray, *, period: int) -> np.ndarray:
    """Run one pool program over one series' values, in original physical units.

    Bit-compatible with v0.1's `apply_fixed_program` for `raw`, `forward_fill`, and
    `seasonal_fill`, so those three carry over unchanged and remain comparable at the
    program level across the two arenas.
    """
    spec = _SPEC_OF_ID.get(program_id)
    if spec is None:
        raise ProgramPoolError(f"unknown program: {program_id!r}")
    if program_id in RUNNER_EXECUTED:
        raise ProgramPoolError(
            f"{program_id!r} is executed by the runner, not by apply_program"
        )
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or np.isinf(source).any():
        raise ProgramPoolError("program input must be a non-empty finite-or-NaN vector")
    if isinstance(period, bool) or not isinstance(period, int) or period < 2:
        raise ProgramPoolError("program period must be an integer of at least 2")

    if program_id == "raw":
        return source.copy()
    if program_id == "forward_fill":
        return _forward_fill(source)
    if program_id == "seasonal_fill":
        return _seasonal_fill(source, period)

    operator = get_operator(str(spec.operator))
    params = dict(spec.params)
    if spec.period_param is not None:
        params[spec.period_param] = period
    prepared = np.asarray(operator(source, **params), dtype=np.float64)
    if prepared.shape != source.shape:
        raise ProgramPoolError(
            f"{program_id!r} changed the series length, violating the Method contract"
        )
    if not np.isfinite(prepared).all():
        raise ProgramPoolError(f"{program_id!r} produced non-finite values")
    return prepared


class PoolProgram:
    """A frozen pool member exposed through the public `BenchmarkMethod` surface.

    Baselines and methods are held to one contract and one execution path.  If a pool
    program could reach the trainer by a private route that skips `validate_prepared`, then
    "the baseline" and "a method" would not be the same kind of object, and every
    method-versus-baseline comparison would be measuring the route as well as the program.
    """

    def __init__(self, program_id: str) -> None:
        if program_id not in _SPEC_OF_ID:
            raise ProgramPoolError(f"unknown program: {program_id!r}")
        if program_id in RUNNER_EXECUTED:
            raise ProgramPoolError(f"{program_id!r} has no standalone Method form")
        self.method_id = program_id
        self._spec = _SPEC_OF_ID[program_id]

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: Any,
        observed_pattern_spec: Mapping[str, float],
    ) -> PreparedSeries:
        period = observed_pattern_spec.get("period")
        if period is None:
            raise ProgramPoolError("observed_pattern_spec must carry the frozen period")
        values = apply_program(
            self.method_id,
            np.asarray(series_view.degraded_inner_train, dtype=np.float64),
            period=int(period),
        )
        operators: tuple[str, ...] = (
            () if self._spec.operator is None else (canonicalize(self._spec.operator),)
        )
        prepared = PreparedSeries(
            series_uid=series_view.series_uid,
            values=values,
            operators=operators,
            units="original_units",
        )
        verdict = validate_prepared(
            prepared, expected_length=len(series_view.degraded_inner_train)
        )
        if not verdict.valid:
            raise ProgramPoolError(
                f"pool program {self.method_id!r} failed its own contract: {verdict.code}"
            )
        return prepared


def _contract_violations() -> list[dict[str, str]]:
    """Check every operator-backed program against the registry contract before freezing."""
    violations: list[dict[str, str]] = []
    for spec in PROGRAM_SPECS:
        if spec.operator is None:
            continue
        canonical = canonicalize(spec.operator)
        metadata = OPERATOR_METADATA.get(canonical)
        if metadata is None:
            violations.append({"program_id": spec.program_id, "code": "unknown_operator"})
            continue
        if bool(metadata.get("changes_target_space")):
            violations.append(
                {"program_id": spec.program_id, "code": "changes_target_space"}
            )
        if "forecast" not in tuple(metadata.get("allowed_tasks", ())):
            violations.append(
                {"program_id": spec.program_id, "code": "forecast_not_allowed"}
            )
        if bool(metadata.get("shape_changing")):
            violations.append({"program_id": spec.program_id, "code": "shape_changing"})
    return violations


def run_pool_with_provenance(
    values: np.ndarray, *, period: int
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Execute every value-domain program once under the operator ledger.

    The returned summary is `{requested_operator: {effective_operator: count}}`.  Any entry
    whose effective operator differs from the requested one is an operator that did not do
    what its program name says -- exactly the masquerade that would forge a conditioning
    signal -- and callers are expected to treat it as a hard failure, not a footnote.
    """
    _provenance.start_recording()
    try:
        outputs = {
            spec.program_id: apply_program(spec.program_id, values, period=period)
            for spec in PROGRAM_SPECS
            if spec.program_id not in RUNNER_EXECUTED
        }
        summary = _provenance.fallback_summary()
    finally:
        _provenance.stop_recording()
    return outputs, summary


def pool_manifest() -> dict[str, Any]:
    """The frozen pool identity. Written before any v0.2 result is read."""
    violations = _contract_violations()
    if violations:
        raise ProgramPoolError(f"pool violates the operator contract: {violations}")
    fingerprint = assert_pool_dependencies()
    by_mechanism: dict[str, list[str]] = {}
    for spec in PROGRAM_SPECS:
        by_mechanism.setdefault(spec.mechanism, []).append(spec.program_id)
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "frozen_before_any_v0_2_result_was_read": True,
        "selection_principle": (
            "mechanism coverage, not expected performance -- the pool exists so that "
            "'choosing' has something to choose between, which is the precondition for C1 "
            "being measurable at all"
        ),
        "programs": [spec.to_dict() for spec in PROGRAM_SPECS],
        "programs_by_mechanism": {
            key: sorted(value) for key, value in sorted(by_mechanism.items())
        },
        "runner_executed": sorted(RUNNER_EXECUTED),
        "capability_gaps": [dict(gap) for gap in CAPABILITY_GAPS],
        "deliberately_excluded_operators": {
            "outlier_iqr": "redundant mechanism with winsorize; excluded before results",
            "outlier_mad": "redundant mechanism with winsorize; excluded before results",
            "smooth_ma": "redundant mechanism with denoise_savgol; excluded before results",
            "smooth_ema": "redundant mechanism with denoise_savgol; excluded before results",
            "impute_linear": "identical to canonical ingestion, which Raw already applies",
            "impute_ema": "redundant mechanism with forward_fill; excluded before results",
            "impute_fft": "redundant mechanism with seasonal_fill; excluded before results",
            "period_complete": "redundant mechanism with seasonal_fill; excluded before results",
            "znorm": "changes_target_space; banned from the action surface by the frozen spec",
            "minmax_norm": "changes_target_space; banned from the action surface",
            "note": (
                "This exclusion list is part of the freeze. Deciding after the fact which "
                "operators 'count' is the path by which a null becomes a discovery."
            ),
        },
        "code_sha256": _code_digests(),
        "dependency_fingerprint": fingerprint,
        "operator_contract_checked": [
            "changes_target_space is False",
            "forecast in allowed_tasks",
            "shape_changing is False",
            "output length equals input length",
            "output is finite",
        ],
    }
