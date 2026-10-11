"""Gradient attacks against YOLOv8 (FGSM, PGD) — legacy research code, see Docs/FILE_REFERENCE.md."""

from .fgsm import fgsm_attack
from .pgd import pgd_attack

__all__ = ["fgsm_attack", "pgd_attack"]
