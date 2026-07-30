"""Controlled object-coordinate verifier benchmark construction."""

from .aligned import generate_aligned
from .ambiguous import generate_ambiguous
from .partial_coverage import generate_partial_coverage
from .unsupported import generate_unsupported
from .wrong_object import generate_wrong_object

__all__ = [
    'generate_aligned',
    'generate_ambiguous',
    'generate_partial_coverage',
    'generate_unsupported',
    'generate_wrong_object',
]
