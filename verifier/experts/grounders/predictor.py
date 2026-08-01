"""Generic GrounderBackend backed by an internal BoxPredictor."""

from pathlib import Path
from typing import Any

from PIL import Image

from ...contracts import (
    GrounderBackend,
    GroundingResult,
    ActionVerifierOutput,
    VerificationRequest,
)
from ...coordinates import original_pixel_box_to_normalized_square_box
from ...models import BoxPredictionRequest, BoxPredictor


def request_image(request: VerificationRequest) -> Image.Image:
    """Read the immutable source image supplied with a verifier request."""

    image: Any = request.sample_context.get('image')
    if isinstance(image, Image.Image):
        return image.convert('RGB').copy()
    image_path = request.sample_context.get('image_path')
    if isinstance(image_path, (str, Path)):
        path = Path(image_path)
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert('RGB').copy()
    raise ValueError(
        'grounder requires sample_context["image"] as a PIL image or '
        'sample_context["image_path"] as a readable path'
    )


class PredictorGrounderBackend(GrounderBackend):
    """Expose a model's object-to-box capability as a relocation expert."""

    def __init__(self, predictor: BoxPredictor, source: str):
        self.predictor = predictor
        self.source = str(source)

    def ground(
            self,
            request: VerificationRequest,
            verification: ActionVerifierOutput) -> GroundingResult:
        image = request_image(request)
        prediction = self.predictor.predict(BoxPredictionRequest(
            image=image,
            object_reference=request.object_reference,
            sample_id=request.sample_id,
        ))
        if prediction.bbox_pixel_xyxy is None:
            raise RuntimeError(
                f'{self.source} failed to ground the object: '
                f'{prediction.error or "no box"}'
            )
        normalized = original_pixel_box_to_normalized_square_box(
            prediction.bbox_pixel_xyxy,
            image.width,
            image.height,
        )
        metadata = dict(prediction.metadata)
        metadata.update({
            'router_action': 'routed_to_grounder',
            'bbox_original_pixel_xyxy': list(
                prediction.bbox_pixel_xyxy
            ),
            'bbox_vocot_normalized_padded_xyxy': list(normalized),
            'prediction_error': prediction.error,
            'prediction_confidence_available': (
                prediction.confidence is not None
            ),
        })
        confidence = (
            0.0
            if prediction.confidence is None
            else float(prediction.confidence)
        )
        return GroundingResult(
            bbox=normalized,
            source=self.source,
            confidence=confidence,
            metadata=metadata,
        )
