"""Compatibility module alias for the canonical frozen H_ref configuration."""

import sys

from ..methods.h_ref_v02 import config as _canonical

sys.modules[__name__] = _canonical
