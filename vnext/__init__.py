"""Benchmark-native, evidence-gated TSharness vNext implementation.

The package intentionally avoids eager imports: the repository's historical policy
package imports optional scientific/LLM dependencies at module import time.  vNext
artifact and lifecycle tooling must remain usable for preflight even when those runtime
dependencies have not yet passed M0.
"""

__all__ = [
    "AccessReservationV1", "ActionEligibilityManifestV1", "AtomicHarnessEdit",
    "CandidateGrammarV1", "ConfirmationArtifactV2", "ConfirmationArtifactV3",
    "DeterministicSupplier", "HBaseArtifactV1", "H0LineageArtifactV2",
    "HarnessArtifactV1", "HarnessArtifactV2", "MethodEvaluationArtifact",
    "InitCorpusManifestV1", "InitHarnessArtifactV1", "InitHarnessPreregV1",
    "MethodInputContractV1", "OneShotAccessControllerV1", "PatternCard",
    "SIX_ARM_ROSTER", "VNextBenchmarkMethod", "VNextMethodArtifact",
    "VNextMethodArtifactV2", "VNextPatternBindingV1",
]
