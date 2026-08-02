"""Generic GrounderBackend backed by an internal BoxPredictor."""

import math
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image

from ...contracts import (
    GrounderBackend,
    GroundingRequest,
    GroundingResult,
)
from ...coordinates import original_pixel_box_to_normalized_square_box
from ...contracts.errors import ExpertUnavailableError
from ...models import BoxPrediction, BoxPredictionRequest, BoxPredictor


ORIGINAL_PIXEL_COORDINATE_SYSTEM = 'absolute_xyxy_on_original_image'


def request_image(request: GroundingRequest) -> Image.Image:
    """Read the immutable source image supplied with a verifier request."""

    image: Any = request.visual.image
    if isinstance(image, Image.Image):
        return image.convert('RGB').copy()
    image_path = request.visual.image_path
    if isinstance(image_path, (str, Path)):
        path = Path(image_path)
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert('RGB').copy()
    raise ValueError(
        'grounder requires sample_context["image"] as a PIL image or '
        'sample_context["image_path"] as a readable path'
    )


def pixel_prediction_to_grounding_result(
        prediction: BoxPrediction,
        *,
        image_width: int,
        image_height: int,
        source: str,
        extra_metadata: Optional[Mapping[str, Any]] = None,
) -> GroundingResult:
    """Convert one original-image pixel prediction to VoCoT coordinates.

    Both in-process predictors and remote worker adapters use this boundary so
    that VoCoT's center-padding transform has a single implementation path.
    Transport- and model-specific code must supply a box in the immutable
    original image's pixel frame.
    """

    if not isinstance(prediction, BoxPrediction):
        raise TypeError('prediction must be a BoxPrediction')
    normalized_source = str(source).strip()
    if not normalized_source:
        raise ValueError('source must be a non-empty string')
    if prediction.bbox_pixel_xyxy is None:
        raise RuntimeError(
            f'{normalized_source} failed to ground the object: '
            f'{prediction.error or "no box"}'
        )
    normalized = original_pixel_box_to_normalized_square_box(
        prediction.bbox_pixel_xyxy,
        image_width,
        image_height,
    )
    confidence = prediction.confidence
    if confidence is not None:
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError('prediction confidence must be in [0, 1]')

    metadata = dict(prediction.metadata)
    metadata.update(dict(extra_metadata or {}))
    metadata.update({
        'router_action': 'routed_to_grounder',
        'bbox_coordinate_system': ORIGINAL_PIXEL_COORDINATE_SYSTEM,
        'bbox_original_pixel_xyxy': list(prediction.bbox_pixel_xyxy),
        'bbox_vocot_normalized_padded_xyxy': list(normalized),
        'prediction_error': prediction.error,
        'prediction_confidence_available': confidence is not None,
    })
    return GroundingResult(
        bbox=normalized,
        source=normalized_source,
        confidence=0.0 if confidence is None else confidence,
        metadata=metadata,
    )


class PredictorGrounderBackend(GrounderBackend):
    """Expose a model's object-to-box capability as a relocation expert."""

    def __init__(self, predictor: BoxPredictor, source: str):
        self.predictor = predictor
        self.source = str(source)

    def ground(self, request: GroundingRequest) -> GroundingResult:
        if not isinstance(request, GroundingRequest):
            raise TypeError('Grounder requires a GroundingRequest')
        image = request_image(request)
        try:
            prediction = self.predictor.predict(BoxPredictionRequest(
                image=image,
                object_reference=request.object_reference,
                sample_id=request.sample_id,
            ))
        except Exception as error:
            raise ExpertUnavailableError(
                f'{self.source} failed while locating the object: '
                f'{type(error).__name__}: {error}',
                metadata={
                    'grounder_source': self.source,
                    'grounder_exception': True,
                    'grounder_exception_type': type(error).__name__,
                },
            ) from error
        if prediction.bbox_pixel_xyxy is None:
            raise ExpertUnavailableError(
                f'{self.source} could not locate the object: '
                f'{prediction.error or "no box"}',
                metadata={
                    'grounder_source': self.source,
                    'prediction_error': prediction.error,
                    'prediction_metadata': dict(prediction.metadata),
                },
            )
        return pixel_prediction_to_grounding_result(
            prediction,
            image_width=image.width,
            image_height=image.height,
            source=self.source,
        )
