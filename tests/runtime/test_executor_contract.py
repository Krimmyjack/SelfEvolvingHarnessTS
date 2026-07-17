import numpy as np

from SelfEvolvingHarnessTS.runtime import executor as canonical
from SelfEvolvingHarnessTS.runtime.errors import ContractError, ExecutionError, RuntimeFailure


def test_executor_records_unknown_operator_without_silent_fallback():
    result = canonical.run_pipeline([("does_not_exist", {})], np.arange(4.0))
    assert result.ok is False
    assert result.artifact is None
    assert result.error == "unknown op 'does_not_exist'"
    assert result.trace[-1]["error"] == "op not in registry"


def test_runtime_error_taxonomy_is_typed():
    assert issubclass(ContractError, RuntimeFailure)
    assert issubclass(ExecutionError, RuntimeFailure)
