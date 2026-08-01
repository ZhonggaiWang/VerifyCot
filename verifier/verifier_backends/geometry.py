"""Model-independent verification by comparing two boxes geometrically."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image

from ..contracts import ActionVerifierOutput
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
class GroundingGeometryDecision:
    """One auditable four-action decision derived only from box geometry."""

    action: str
    reason: str
    intersection_area: float
    candidate_area: float
    grounding_area: float
    iou: float
    candidate_coverage_by_grounding: float
    grounding_coverage_by_candidate: float
    accept_iou_threshold: float
    containment_threshold: float


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


def route_from_grounding_geometry(
        candidate_bbox: Sequence[float],
        grounding_bbox: Sequence[float],
        accept_iou_threshold: float = 0.5,
        containment_threshold: float = 0.7,
) -> GroundingGeometryDecision:
    """Map candidate-versus-prediction geometry to a specialist action."""

    if not 0.0 < accept_iou_threshold <= 1.0:
        raise ValueError('accept_iou_threshold must be in (0, 1]')
    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError('containment_threshold must be in (0, 1]')
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
    iou = intersection / union
    candidate_coverage = intersection / candidate_area
    grounding_coverage = intersection / grounding_area

    if iou >= accept_iou_threshold:
        action = 'no_action'
        reason = 'candidate_grounding_iou_meets_accept_threshold'
    elif (
        candidate_coverage >= containment_threshold
        and grounding_coverage < containment_threshold
    ):
        action = 'expand'
        reason = 'candidate_is_mostly_inside_larger_grounding_box'
    elif (
        grounding_coverage >= containment_threshold
        and candidate_coverage < containment_threshold
    ):
        action = 'tighten'
        reason = 'grounding_box_is_mostly_inside_larger_candidate'
    else:
        action = 'relocate'
        reason = 'low_iou_without_directional_containment'

    return GroundingGeometryDecision(
        action=action,
        reason=reason,
        intersection_area=intersection,
        candidate_area=candidate_area,
        grounding_area=grounding_area,
        iou=iou,
        candidate_coverage_by_grounding=candidate_coverage,
        grounding_coverage_by_candidate=grounding_coverage,
        accept_iou_threshold=float(accept_iou_threshold),
        containment_threshold=float(containment_threshold),
    )


class GeometryVerifier:
    """Turn any internal box predictor into a four-action verifier."""

    def __init__(
            self,
            predictor: BoxPredictor,
            accept_iou_threshold: float = 0.5,
            containment_threshold: float = 0.7,
            backend_name: str = 'box_predictor_geometry_router'):
        if not 0.0 < accept_iou_threshold <= 1.0:
            raise ValueError('accept_iou_threshold must be in (0, 1]')
        if not 0.0 < containment_threshold <= 1.0:
            raise ValueError('containment_threshold must be in (0, 1]')
        self.predictor = predictor
        self.accept_iou_threshold = float(accept_iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.backend_name = str(backend_name)

    def classify(
            self,
            candidate: GeometryVerificationInput,
            image_mode: str = 'raw_image') -> GeometryVerificationLookup:
        if image_mode != 'raw_image':
            raise ValueError(
                'geometry verification supports only raw_image because the '
                'candidate must stay hidden from the box predictor'
            )
        if not isinstance(candidate.image, Image.Image):
            raise TypeError('candidate.image must be a PIL.Image.Image')
        image_size = (candidate.image.width, candidate.image.height)
        candidate_box = _candidate_box(
            candidate.candidate_bbox_pixel_xyxy,
            image_size,
        )
        prediction = self.predictor.predict(BoxPredictionRequest(
            image=candidate.image,
            object_reference=candidate.object_reference,
            sample_id=candidate.sample_id,
        ))
        metadata = dict(prediction.metadata)
        metadata.setdefault('backend', self.backend_name)
        metadata.update({
            'image_mode': image_mode,
            'model_image_count': 1,
            'sample_id': candidate.sample_id,
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'original_image_size': list(image_size),
            'object_reference': candidate.object_reference,
            'candidate_original_pixel_bbox_xyxy': list(candidate_box),
        })
        if prediction.bbox_pixel_xyxy is None:
            metadata.setdefault('localization_failed', True)
            metadata.setdefault('parse_failed', True)
            metadata['geometry'] = None
            return GeometryVerificationLookup(
                status=None,
                confidence=prediction.confidence,
                error=prediction.error or 'box_prediction_failed',
                metadata=metadata,
            )

        grounding_box = _candidate_box(
            prediction.bbox_pixel_xyxy,
            image_size,
        )
        decision = route_from_grounding_geometry(
            candidate_box,
            grounding_box,
            accept_iou_threshold=self.accept_iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        metadata['geometry'] = asdict(decision)
        return GeometryVerificationLookup(
            status=decision.action,
            confidence=prediction.confidence,
            error=prediction.error,
            metadata=metadata,
        )

    def classify_action(
            self,
            candidate: GeometryVerificationInput,
            image_mode: str = 'raw_image') -> ActionVerifierOutput:
        """Expose deterministic geometry through the canonical action schema."""

        lookup = self.classify(candidate, image_mode=image_mode)
        metadata = {
            **dict(lookup.metadata),
            'probability_source': 'unavailable_geometry_hard_label',
        }
        if lookup.status is None:
            return ActionVerifierOutput.unknown(
                error=lookup.error,
                confidence=0.0,
                metadata=metadata,
            )
        return ActionVerifierOutput(
            predicted_action=lookup.status,
            action_probabilities=None,
            confidence=(
                0.0
                if lookup.confidence is None
                else float(lookup.confidence)
            ),
            abstained=False,
            error=lookup.error,
            metadata=metadata,
        )


class PaddedGeometryVerifier:
    """Compare a VoCoT candidate with a raw-image box prediction.

    The predictor sees only the clean, unpadded source image. Its output is
    mapped into VoCoT's center-padded normalized frame before the deterministic
    geometry decision, so neither padding nor the candidate box is exposed to
    the grounding model.
    """

    def __init__(
            self,
            predictor: BoxPredictor,
            accept_iou_threshold: float = 0.5,
            containment_threshold: float = 0.7,
            backend_name: str = 'padded_box_predictor_geometry_router'):
        if not 0.0 < accept_iou_threshold <= 1.0:
            raise ValueError('accept_iou_threshold must be in (0, 1]')
        if not 0.0 < containment_threshold <= 1.0:
            raise ValueError('containment_threshold must be in (0, 1]')
        self.predictor = predictor
        self.accept_iou_threshold = float(accept_iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.backend_name = str(backend_name)

    def classify_action(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> ActionVerifierOutput:
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
            'probability_source': 'unavailable_geometry_hard_label',
        }
        if prediction.bbox_pixel_xyxy is None:
            metadata['selected_grounding_padded_normalized_bbox_xyxy'] = None
            metadata['geometry'] = None
            return ActionVerifierOutput.unknown(
                error=prediction.error or 'box_prediction_failed',
                confidence=0.0,
                metadata=metadata,
            )

        grounding_box = original_pixel_box_to_normalized_square_box(
            prediction.bbox_pixel_xyxy,
            source.width,
            source.height,
        )
        decision = route_from_grounding_geometry(
            candidate_box,
            grounding_box,
            accept_iou_threshold=self.accept_iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        metadata.update({
            'selected_grounding_original_pixel_bbox_xyxy': list(
                prediction.bbox_pixel_xyxy
            ),
            'selected_grounding_padded_normalized_bbox_xyxy': list(
                grounding_box
            ),
            'geometry': asdict(decision),
        })
        return ActionVerifierOutput(
            predicted_action=decision.action,
            action_probabilities=None,
            confidence=(
                0.0
                if prediction.confidence is None
                else float(prediction.confidence)
            ),
            abstained=False,
            error=prediction.error,
            metadata=metadata,
        )
