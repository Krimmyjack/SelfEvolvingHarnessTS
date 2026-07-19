from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .chronos import (
    FrozenChronosValuator,
    _array,
    _array_sha,
    _finite_scale,
    _predict_mean,
)


PrefixTransform = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class RollingObservedReceipt:
    valuation_source: str
    ingestion_policy_id: str
    model_manifest_sha: str
    origins: tuple[int, ...]
    excluded_origins: tuple[int, ...]
    input_shas: tuple[str, ...]
    target_shas: tuple[str, ...]
    forecast_shas: tuple[str, ...]
    per_origin_losses: tuple[float, ...]
    mean_public_utility: float | None
    status: str
    horizon: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": "rolling-observed-receipt/1",
            "model_manifest_sha": self.model_manifest_sha,
            "origins": list(self.origins),
            "excluded_origins": list(self.excluded_origins),
            "input_shas": list(self.input_shas),
            "target_shas": list(self.target_shas),
            "forecast_shas": list(self.forecast_shas),
            "per_origin_losses": list(self.per_origin_losses),
            "mean_public_utility": self.mean_public_utility,
            "status": self.status,
            "horizon": self.horizon,
        }


class RollingObservedValuator:
    """Deployment-computable rolling utility over already-observed slices."""

    def __init__(
        self,
        *,
        pipeline: object | None = None,
        origins: tuple[int, ...] = (96, 120, 144, 168),
        horizon: int = 24,
        min_finite_targets: int = 12,
        manifest_path: Path | None = None,
        model_manifest_sha: str | None = None,
        valuation_source: str | None = None,
        ingestion_policy_id: str | None = None,
    ) -> None:
        base = FrozenChronosValuator(pipeline=pipeline, manifest_path=manifest_path)
        self.pipeline = base.pipeline
        self.model_manifest_sha = model_manifest_sha or base.model_manifest_sha
        self.valuation_source = valuation_source or "PINNED_FROZEN_CHRONOS"
        self.ingestion_policy_id = (
            ingestion_policy_id or str(base.manifest["ingestion_policy"])
        )
        self.origins = tuple(int(origin) for origin in origins)
        self.horizon = int(horizon)
        self.min_finite_targets = int(min_finite_targets)
        if self.horizon <= 0 or self.min_finite_targets <= 0:
            raise ValueError("rolling horizon and finite-target minimum must be positive")

    def evaluate(
        self,
        series: object,
        *,
        prefix_transform: PrefixTransform | None = None,
    ) -> RollingObservedReceipt:
        values = _array(series, length=192, field="rolling series")
        surviving: list[int] = []
        excluded: list[int] = []
        model_prefixes: list[np.ndarray] = []
        transformed_prefixes: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for origin in self.origins:
            if origin <= 0 or origin + self.horizon > values.size:
                excluded.append(origin)
                continue
            target = values[origin : origin + self.horizon].copy()
            if int(np.count_nonzero(np.isfinite(target))) < self.min_finite_targets:
                excluded.append(origin)
                continue
            prefix = values[:origin].copy()
            if prefix_transform is not None:
                transformed = np.asarray(
                    prefix_transform(prefix.copy(), origin),
                    dtype=np.float64,
                )
                if transformed.shape != prefix.shape:
                    raise ValueError("prefix transform must preserve prefix shape")
                prefix = transformed.copy()
            if np.any(np.isinf(prefix)) or not np.any(np.isfinite(prefix)):
                excluded.append(origin)
                continue
            surviving.append(origin)
            transformed_prefixes.append(prefix)
            # Chronos-Bolt consumes NaN as an observation mask.  Do not apply
            # an ingestion repair that overlaps the Agent's operator space.
            model_prefixes.append(prefix.copy())
            targets.append(target)

        if not surviving:
            return RollingObservedReceipt(
                valuation_source=self.valuation_source,
                ingestion_policy_id=self.ingestion_policy_id,
                model_manifest_sha=self.model_manifest_sha,
                origins=(),
                excluded_origins=tuple(excluded),
                input_shas=(),
                target_shas=(),
                forecast_shas=(),
                per_origin_losses=(),
                mean_public_utility=None,
                status="UNKNOWN",
                horizon=self.horizon,
            )

        try:
            import torch

            contexts = [
                torch.as_tensor(prefix, dtype=torch.float32, device="cpu")
                for prefix in model_prefixes
            ]
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            from SelfEvolvingHarnessTS.runtime.errors import InfrastructureError

            raise InfrastructureError("rolling CPU tensor construction failed") from exc
        forecasts = _predict_mean(
            self.pipeline,
            contexts,
            prediction_length=self.horizon,
        )
        if forecasts.shape[0] != len(surviving):
            from SelfEvolvingHarnessTS.runtime.errors import InfrastructureError

            raise InfrastructureError("rolling forecast batch size mismatch")

        losses: list[float] = []
        for index, target in enumerate(targets):
            finite = np.isfinite(target)
            scale = _finite_scale(values[: surviving[index]])
            error = forecasts[index, finite] - target[finite]
            losses.append(float(np.sqrt(np.mean(np.square(error))) / scale))

        return RollingObservedReceipt(
            valuation_source=self.valuation_source,
            ingestion_policy_id=self.ingestion_policy_id,
            model_manifest_sha=self.model_manifest_sha,
            origins=tuple(surviving),
            excluded_origins=tuple(excluded),
            input_shas=tuple(_array_sha(prefix) for prefix in transformed_prefixes),
            target_shas=tuple(_array_sha(target) for target in targets),
            forecast_shas=tuple(_array_sha(forecast) for forecast in forecasts),
            per_origin_losses=tuple(losses),
            mean_public_utility=-float(np.mean(losses)),
            status="OK",
            horizon=self.horizon,
        )


__all__ = ["RollingObservedReceipt", "RollingObservedValuator"]
