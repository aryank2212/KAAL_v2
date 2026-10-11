"""Tensor input-transform defenses (JPEG, bit depth, blur) — legacy; benchmark.py re-implements them in NumPy."""

from .preprocess import defense_pipeline

__all__ = ["defense_pipeline"]
