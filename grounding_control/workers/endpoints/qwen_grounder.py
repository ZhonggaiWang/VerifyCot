"""JSON request adapter for a Qwen2.5-VL relocation expert.

The endpoint deliberately stops at original-image pixel coordinates.  The
consumer adapting this worker to VoCoT is responsible for the model-specific
square-padding conversion.
"""

from typing import Any, Dict, Mapping

from ...models import BoxPredictionRequest
from ...models.qwen25_vl import Qwen25VLBoxPredictor
from ...transport.grounder_wire import (
    ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
    serialize_grounder_output,
)
from ...transport.request_io import (
    finite_pixel_box,
    load_image,
    required_string,
)

QWEN_GROUNDER_SOURCE = 'qwen25_vl_grounder'


class QwenGrounderEndpoint:
    """Run one clean-image Qwen grounding request."""

    def __init__(self, predictor: Qwen25VLBoxPredictor):
        self.predictor = predictor

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        image = load_image(payload.get('image_path'))
        reference = required_string(payload, 'object_reference')
        prediction = self.predictor.predict(BoxPredictionRequest(
            image=image,
            object_reference=reference,
            sample_id=str(payload.get('sample_id') or ''),
        ))
        metadata = dict(prediction.metadata or {})
        available = prediction.bbox_pixel_xyxy is not None
        if prediction.bbox_pixel_xyxy is None:
            # A model-format/parse failure is an expected expert-unavailable
            # result, not a malformed JSONL request.  Keeping it as an
            # ``ok=true, available=false`` response lets the caller apply its
            # configured fail-open/fail-closed policy while retaining the raw
            # model output for audit.
            return serialize_grounder_output(
                available=False,
                source=QWEN_GROUNDER_SOURCE,
                bbox=None,
                image_size=image.size,
                confidence=None,
                error=prediction.error or 'grounder_unavailable',
                metadata=metadata,
            )

        bbox = finite_pixel_box(
            prediction.bbox_pixel_xyxy,
            image.size,
            'bbox',
        )
        return serialize_grounder_output(
            available=available,
            source=QWEN_GROUNDER_SOURCE,
            bbox=tuple(bbox),
            image_size=image.size,
            confidence=prediction.confidence,
            error=None,
            metadata=metadata,
        )


__all__ = [
    'ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM',
    'QWEN_GROUNDER_SOURCE',
    'QwenGrounderEndpoint',
]
