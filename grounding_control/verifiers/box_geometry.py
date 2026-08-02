"""Model-independent verification by comparing two boxes geometrically."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image

from ..coordinates import (
    COORDINATE_SYSTEM,
    original_pixel_box_to_normalized_square_box,
)
from ..models import BoxPredictionRequest, BoxPredictor


PixelBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class GeometryVerificationInput:
    """Clean image, object reference, and candidate in original pixels."""

    image: Image.Image
    object_reference: str
    candidate_bbox_pixel_xyxy: PixelBox
    sample_id: str = ''


@dataclass(frozen=True)
class PaddedGeometryVerificationInput:
    """Clean image and candidate in VoCoT's normalized padded-square frame."""

    image: Image.Image
    object_reference: str
    candidate_bbox_padded_normalized_xyxy: Tuple[float, float, float, float]
    sample_id: str = ''


@dataclass(frozen=True)
class GeometryVerificationLookup:
    status: Optional[str]
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class BoxGeometryMeasurement:
    """Policy-free overlap measurements for two validated ``xyxy`` boxes.

    The value deliberately contains no routing action, threshold, or verifier
    confidence.  Binary alignment scoring and legacy four-way routing can
    therefore consume the same geometry without depending on one another.
    """

    intersection_area: float
    candidate_area: float
    grounding_area: float
    iou: float
    candidate_coverage_by_grounding: float
    grounding_coverage_by_candidate: float


@dataclass(frozen=True)
class PaddedGeometryComparison:
    """One predictor result compared in VoCoT's padded coordinate frame."""

    candidate_bbox: PixelBox
    grounding_bbox: Optional[PixelBox]
    measurement: Optional[BoxGeometryMeasurement]
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]


def _box(values: Sequence[float], name: str) -> PixelBox:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f'{name} must contain four coordinates')
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f'{name} contains non-finite coordinates: {box}')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f'{name} is empty: {box}')
    return box  # type: ignore[return-value]


def _candidate_box(
        values: Sequence[float],
        image_size: Tuple[int, int],
) -> PixelBox:
    box = _box(values, 'candidate_bbox')
    width, height = image_size
    if not (
        0.0 <= box[0] < box[2] <= width
        and 0.0 <= box[1] < box[3] <= height
    ):
        raise ValueError(
            f'candidate pixel bbox is outside image {image_size}: {box}'
        )
    return box


def measure_box_geometry(
        candidate_bbox: Sequence[float],
        grounding_bbox: Sequence[float],
) -> BoxGeometryMeasurement:
    """Measure overlap without assigning any verifier or routing semantics."""

    candidate = _box(candidate_bbox, 'candidate_bbox')
    grounding = _box(grounding_bbox, 'grounding_bbox')
    intersection_width = max(
        0.0,
        min(candidate[2], grounding[2]) - max(candidate[0], grounding[0]),
    )
    intersection_height = max(
        0.0,
        min(candidate[3], grounding[3]) - max(candidate[1], grounding[1]),
    )
    intersection = intersection_width * intersection_height
    candidate_area = (
        (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    )
    grounding_area = (
        (grounding[2] - grounding[0]) * (grounding[3] - grounding[1])
    )
    union = candidate_area + grounding_area - intersection
    return BoxGeometryMeasurement(
        intersection_area=intersection,
        candidate_area=candidate_area,
        grounding_area=grounding_area,
        iou=intersection / union,
        candidate_coverage_by_grounding=intersection / candidate_area,
        grounding_coverage_by_candidate=intersection / grounding_area,
    )


class PaddedGeometryComparator:
    """Predict and measure a VoCoT candidate without routing semantics.

    The predictor sees only the clean, unpadded source image. Its output is
    mapped into VoCoT's center-padded normalized frame before the policy-free
    overlap measurement, so neither padding nor the candidate box is exposed
    to the grounding model.
    """

    def __init__(
            self,
            predictor: BoxPredictor,
            backend_name: str = 'padded_box_predictor_geometry_comparator'):
        self.predictor = predictor
        self.backend_name = str(backend_name)

    def compare(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> PaddedGeometryComparison:
        if not isinstance(candidate.image, Image.Image):
            raise TypeError('candidate.image must be a PIL.Image.Image')
        source = candidate.image.convert('RGB')
        candidate_box = tuple(
            float(value)
            for value in candidate.candidate_bbox_padded_normalized_xyxy
        )
        # Reuse the public VoCoT box validator through the conversion helper's
        # downstream geometry validation.
        _box(candidate_box, 'candidate_bbox_padded_normalized_xyxy')
        if not all(0.0 <= value <= 1.0 for value in candidate_box):
            raise ValueError(
                'candidate padded-normalized bbox must be inside [0, 1]'
            )

        prediction = self.predictor.predict(BoxPredictionRequest(
            image=source,
            object_reference=candidate.object_reference,
            sample_id=candidate.sample_id,
        ))
        metadata = {
            **dict(prediction.metadata),
            'backend': self.backend_name,
            'sample_id': candidate.sample_id,
            'object_reference': candidate.object_reference,
            'coordinate_system': COORDINATE_SYSTEM,
            'original_image_size': [source.width, source.height],
            'candidate_padded_normalized_bbox_xyxy': list(candidate_box),
        }
        if prediction.bbox_pixel_xyxy is None:
            metadata['selected_grounding_padded_normalized_bbox_xyxy'] = None
            return PaddedGeometryComparison(
                candidate_bbox=candidate_box,
                grounding_bbox=None,
                measurement=None,
                confidence=prediction.confidence,
                error=prediction.error or 'box_prediction_failed',
                metadata=metadata,
            )

        grounding_box = original_pixel_box_to_normalized_square_box(
            prediction.bbox_pixel_xyxy,
            source.width,
            source.height,
        )
        metadata.update({
            'selected_grounding_original_pixel_bbox_xyxy': list(
                prediction.bbox_pixel_xyxy
            ),
            'selected_grounding_padded_normalized_bbox_xyxy': list(
                grounding_box
            ),
        })
        return PaddedGeometryComparison(
            candidate_bbox=candidate_box,
            grounding_bbox=grounding_box,
            measurement=measure_box_geometry(candidate_box, grounding_box),
            confidence=prediction.confidence,
            error=prediction.error,
            metadata=metadata,
        )

