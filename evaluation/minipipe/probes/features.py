"""Compatibility import for the shared deployment-visible feature extractor."""

from SelfEvolvingHarnessTS.runtime.public_features import (
    PublicFeatureExtraction,
    extract_public_features,
)


__all__ = ["PublicFeatureExtraction", "extract_public_features"]
