"""Internal object-to-box model capability.

This is deliberately not a controller role named ``Localizer``.  A single
model capability can be adapted into a localization-based verifier, a
GrounderBackend, or (later) a BoxRefinerBackend without conflating those
system-level responsibilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

from PIL import Image


PixelBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxPredictionRequest:
    image: Image.Image
    object_reference: str
    sample_id: str = ''


@dataclass(frozen=True)
class BoxPrediction:
    bbox_pixel_xyxy: Optional[PixelBox]
    confidence: Optional[float]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BoxPredictor(Protocol):
    """Predict one reference box in original-image pixel coordinates."""

    def predict(self, request: BoxPredictionRequest) -> BoxPrediction:
        ...
