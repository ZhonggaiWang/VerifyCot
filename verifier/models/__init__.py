"""Reusable model capabilities hidden behind verifier/expert adapters."""

from .box_predictor import (
    BoxPrediction,
    BoxPredictionRequest,
    BoxPredictor,
    PixelBox,
)

__all__ = [
    'BoxPrediction',
    'BoxPredictionRequest',
    'BoxPredictor',
    'PixelBox',
]
