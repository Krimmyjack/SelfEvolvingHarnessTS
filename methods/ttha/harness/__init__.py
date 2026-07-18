"""Immutable Harness authoring, compilation, and snapshot storage.

Public objects live in ``compiler`` and ``store``.  Keeping this package initializer
side-effect free also makes ``python -m ...harness.compiler`` deterministic.
"""
