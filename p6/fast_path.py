"""Compatibility module alias for the canonical fast-path runtime."""

import sys

from ..runtime import fast_path as _canonical

sys.modules[__name__] = _canonical
