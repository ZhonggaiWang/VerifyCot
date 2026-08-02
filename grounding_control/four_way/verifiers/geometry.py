"""Legacy four-action routing derived from policy-free box geometry."""

from dataclasses import asdict, dataclass
from typing import Sequence

from PIL import Image

from ..contracts import ActionVerifierOutput
from ...models import BoxPredictionRequest, BoxPredictor
from ...verifiers.box_geometry import (
    BoxGeometryMeasurement,
    GeometryVerificationInput,
    GeometryVerificationLookup,
    PaddedGeometryComparator,
    PaddedGeometryVerificationInput,
    _candidate_box,
    measure_box_geometry,
)


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


def _route_geometry_measurement(
        measurement: BoxGeometryMeasurement,
        accept_iou_threshold: float,
        containment_threshold: float,
) -> GroundingGeometryDecision:
    """Apply the archived action policy to a neutral overlap measurement."""

    iou = measurement.iou
    candidate_coverage = measurement.candidate_coverage_by_grounding
    grounding_coverage = measurement.grounding_coverage_by_candidate
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
        intersection_area=measurement.intersection_area,
        candidate_area=measurement.candidate_area,
        grounding_area=measurement.grounding_area,
        iou=measurement.iou,
        candidate_coverage_by_grounding=(
            measurement.candidate_coverage_by_grounding
        ),
        grounding_coverage_by_candidate=(
            measurement.grounding_coverage_by_candidate
        ),
        accept_iou_threshold=float(accept_iou_threshold),
        containment_threshold=float(containment_threshold),
    )


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
    return _route_geometry_measurement(
        measure_box_geometry(candidate_bbox, grounding_bbox),
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
    """Legacy four-way adapter over the neutral padded comparator."""

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
        self.comparator = PaddedGeometryComparator(
            predictor,
            backend_name=self.backend_name,
        )

    def classify_action(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> ActionVerifierOutput:
        comparison = self.comparator.compare(candidate)
        metadata = {
            **dict(comparison.metadata),
            'probability_source': 'unavailable_geometry_hard_label',
        }
        if comparison.measurement is None:
            metadata['geometry'] = None
            return ActionVerifierOutput.unknown(
                error=comparison.error or 'box_prediction_failed',
                confidence=0.0,
                metadata=metadata,
            )

        decision = _route_geometry_measurement(
            comparison.measurement,
            accept_iou_threshold=self.accept_iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        metadata['geometry'] = asdict(decision)
        return ActionVerifierOutput(
            predicted_action=decision.action,
            action_probabilities=None,
            confidence=(
                0.0
                if comparison.confidence is None
                else float(comparison.confidence)
            ),
            abstained=False,
            error=comparison.error,
            metadata=metadata,
        )


__all__ = [
    'GeometryVerifier',
    'GroundingGeometryDecision',
    'PaddedGeometryVerifier',
    'route_from_grounding_geometry',
]
