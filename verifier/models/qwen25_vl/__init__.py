"""Qwen2.5-VL reusable model capabilities.

The box predictor is imported lazily so importing lightweight contracts never
forces model-specific adapters to initialize.  Model code has no dependency on
the verifier implementations that consume these capabilities.
"""

from .runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLRunner,
)

__all__ = [
    'DEFAULT_MAX_PIXELS',
    'DEFAULT_MIN_PIXELS',
    'LocalQwen25VLRunner',
    'Qwen25VLBoxPredictor',
    'Qwen25VLRunner',
]


def __getattr__(name):
    if name == 'Qwen25VLBoxPredictor':
        from .box_predictor import Qwen25VLBoxPredictor
        return Qwen25VLBoxPredictor
    raise AttributeError(name)
