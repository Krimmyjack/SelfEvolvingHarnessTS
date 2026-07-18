from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import (
    canonical_sha256,
    parse_json_document,
)
from SelfEvolvingHarnessTS.runtime.errors import InfrastructureError


_MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
_EXPECTED_MANIFEST = {
    "schema_version": "m0-valuator/1",
    "model_id": "amazon/chronos-bolt-small",
    "revision": "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
    "chronos_forecasting": "2.3.0",
    "torch": "2.12.0+cu126",
    "transformers": "5.12.1",
    "device": "cpu",
    "dtype": "float32",
    "context_length": 192,
    "prediction_length": 48,
    "point_forecast": "mean",
    "loss": "nrmse_clean_context_scale",
    "utility": "negative_loss",
    "ingestion_policy": "chronos_native_nan_mask/v1",
}


class FrozenModelUnavailable(InfrastructureError):
    """The exact, locally pinned frozen model cannot be loaded."""


def _array(values: object, *, length: int, field: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size != length:
        raise ValueError(f"{field} must be a one-dimensional array of length {length}")
    return result.copy()


def _array_sha(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").copy()
    canonical[np.isnan(canonical)] = np.nan
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _finite_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("scale context must contain at least one finite value")
    return max(float(np.std(finite, dtype=np.float64)), 1e-8)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _predict_mean(
    pipeline: object,
    contexts: object,
    *,
    prediction_length: int,
) -> np.ndarray:
    try:
        import torch

        with torch.inference_mode():
            _, mean = pipeline.predict_quantiles(
                contexts,
                prediction_length=prediction_length,
                quantile_levels=[0.5],
            )
    except (ImportError, AttributeError, TypeError, RuntimeError, ValueError) as exc:
        raise InfrastructureError("frozen model inference failed") from exc
    forecast = _to_numpy(mean)
    if forecast.ndim == 1:
        forecast = forecast[None, :]
    if forecast.ndim != 2 or forecast.shape[1] != prediction_length:
        raise InfrastructureError("frozen model returned an invalid mean forecast shape")
    if not np.all(np.isfinite(forecast)):
        raise InfrastructureError("frozen model returned a non-finite mean forecast")
    return forecast


def load_model_manifest(path: Path | None = None) -> tuple[dict[str, object], str]:
    manifest_path = path or _MANIFEST_PATH
    value = parse_json_document(manifest_path.read_bytes())
    if not isinstance(value, dict):
        raise InfrastructureError("frozen model manifest must be a JSON object")
    manifest = dict(value)
    embedded_sha = manifest.pop("manifest_sha", None)
    if not isinstance(embedded_sha, str) or canonical_sha256(manifest) != embedded_sha:
        raise InfrastructureError("frozen model manifest SHA mismatch")
    if manifest != _EXPECTED_MANIFEST:
        raise InfrastructureError("frozen model manifest differs from the M0 lock")
    return dict(value), embedded_sha


def _load_local_pipeline(manifest: dict[str, object]) -> object:
    identity = f"{manifest['model_id']}@{manifest['revision']}"
    try:
        import chronos
        import torch
        import transformers
        from chronos import BaseChronosPipeline

        versions = {
            "chronos_forecasting": getattr(chronos, "__version__", None),
            "torch": getattr(torch, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
        }
        for dependency, actual in versions.items():
            if actual != manifest[dependency]:
                raise FrozenModelUnavailable(
                    f"pinned dependency unavailable for frozen model: {dependency}"
                )
        return BaseChronosPipeline.from_pretrained(
            str(manifest["model_id"]),
            revision=str(manifest["revision"]),
            device_map="cpu",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
    except FrozenModelUnavailable:
        raise
    except Exception as exc:  # provider libraries expose several loader exceptions
        raise FrozenModelUnavailable(f"pinned frozen model unavailable: {identity}") from exc


@dataclass(frozen=True)
class ValuationReceipt:
    valuation_source: str
    ingestion_policy_id: str
    model_manifest_sha: str
    input_sha: str
    filled_context_sha: str
    future_sha: str
    forecast_sha: str
    loss_j: float
    utility_u: float
    missing_count: int
    missing_fraction: float
    fill_fraction: float
    scale: float
    prediction_length: int
    status: str


class FrozenChronosValuator:
    """Private frozen-model evaluator. Its receipt is never Agent-facing."""

    valuation_source = "PINNED_FROZEN_CHRONOS"

    def __init__(
        self,
        *,
        pipeline: object | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        manifest, manifest_sha = load_model_manifest(manifest_path)
        self.manifest = manifest
        self.model_manifest_sha = manifest_sha
        self.ingestion_policy_id = str(manifest["ingestion_policy"])
        self.pipeline = pipeline if pipeline is not None else _load_local_pipeline(manifest)

    def evaluate(
        self,
        context: object,
        clean_future: object,
        *,
        scale_context: object,
    ) -> ValuationReceipt:
        context_length = int(self.manifest["context_length"])
        prediction_length = int(self.manifest["prediction_length"])
        raw = _array(context, length=context_length, field="context")
        future = _array(clean_future, length=prediction_length, field="clean_future")
        scale_values = _array(
            scale_context,
            length=context_length,
            field="scale_context",
        )
        if not np.all(np.isfinite(future)):
            raise ValueError("clean_future must contain only finite values")
        if np.any(np.isinf(raw)):
            raise ValueError("context may contain NaN missing values but not infinity")
        finite = np.isfinite(raw)
        if not np.any(finite):
            raise ValueError("context must contain at least one observed value")
        # Chronos-Bolt 2.3.0 derives an observation mask from NaNs.  Preserve
        # that native representation so the identity arm is not silently
        # converted into the canonical impute_linear repair arm.
        model_context = raw.copy()
        missing_count = int(raw.size - np.count_nonzero(finite))
        scale = _finite_scale(scale_values)

        try:
            import torch

            tensor = torch.as_tensor(
                model_context,
                dtype=torch.float32,
                device="cpu",
            ).reshape(1, -1)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            raise InfrastructureError("CPU float32 tensor construction failed") from exc
        forecast = _predict_mean(
            self.pipeline,
            tensor,
            prediction_length=prediction_length,
        )[0]
        loss = float(np.sqrt(np.mean(np.square(forecast - future))) / scale)
        if not math.isfinite(loss):
            raise InfrastructureError("valuation loss is non-finite")
        return ValuationReceipt(
            valuation_source=self.valuation_source,
            ingestion_policy_id=self.ingestion_policy_id,
            model_manifest_sha=self.model_manifest_sha,
            input_sha=_array_sha(raw),
            # Retained as a compatibility/debugging instrument: it now hashes
            # the exact model input, including the canonical NaN mask.
            filled_context_sha=_array_sha(model_context),
            future_sha=_array_sha(future),
            forecast_sha=_array_sha(forecast),
            loss_j=loss,
            utility_u=-loss,
            missing_count=missing_count,
            missing_fraction=missing_count / context_length,
            fill_fraction=0.0,
            scale=scale,
            prediction_length=prediction_length,
            status="OK",
        )


__all__ = [
    "FrozenChronosValuator",
    "FrozenModelUnavailable",
    "ValuationReceipt",
    "load_model_manifest",
]
