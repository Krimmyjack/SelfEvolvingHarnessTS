"""Compatibility boundary for the frozen benchmark-v0.2 evaluator."""

from .method_compat import BenchmarkMethodAdapter, run_h_ref_batch

__all__ = ["BenchmarkMethodAdapter", "run_h_ref_batch"]
