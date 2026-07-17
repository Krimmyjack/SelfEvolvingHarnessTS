import numpy as np

from SelfEvolvingHarnessTS.operators.registry import TOOL_REGISTRY, canonicalize
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline


def test_registry_alias_and_executor_share_the_canonical_operator():
    assert canonicalize("fill_gaps") == "impute_linear"
    assert TOOL_REGISTRY["fill_gaps"] is TOOL_REGISTRY["impute_linear"]
    result = run_pipeline([("impute_linear", {})], np.array([1.0, np.nan, 3.0]))
    assert result.ok is True
    assert np.isfinite(result.artifact).all()
