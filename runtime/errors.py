class RuntimeFailure(RuntimeError):
    """Base class for canonical runtime failures."""


class ContractError(RuntimeFailure):
    """A request, program, or prepared result violates a public contract."""


class ExecutionError(RuntimeFailure):
    """A declared runtime operation failed."""


class ProtocolViolation(RuntimeFailure):
    """A frozen or information-boundary protocol was violated."""


class InfrastructureError(RuntimeFailure):
    """Required infrastructure was unavailable."""
