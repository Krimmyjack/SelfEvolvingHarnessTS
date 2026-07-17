"""Compatibility import for the canonical task contract.

New code imports :mod:`SelfEvolvingHarnessTS.contracts.task` directly.
"""

from ..contracts.task import *  # noqa: F401,F403
from ..contracts.task import __all__
