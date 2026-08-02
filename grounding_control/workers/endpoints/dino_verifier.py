"""JSON endpoint for the binary Grounding DINO alignment scorer."""

from typing import Any, Dict, Mapping

from ...contracts import validate_normalized_box
from ...coordinates import COORDINATE_SYSTEM
from ...transport import WorkerRequestError
from ...transport.request_io import load_image, required_string
from ...verifiers.dino import GroundingDinoAlignmentScorer
from ...verifiers.box_geometry import PaddedGeometryVerificationInput
from .alignment_response import serialize_alignment_response


DINO_ALIGNMENT_MODE = 'binary_alignment'


class DinoVerifierEndpoint:
    """Validate one candidate and return a threshold-free alignment score."""

    def __init__(self, alignment_scorer: GroundingDinoAlignmentScorer):
        self.alignment_scorer = alignment_scorer

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        mode = str(payload.get('verifier_mode') or DINO_ALIGNMENT_MODE)
        if mode != DINO_ALIGNMENT_MODE:
            raise WorkerRequestError(
                'binary DINO verifier_mode must equal '
                f'{DINO_ALIGNMENT_MODE!r}; four-way geometry verification '
                'lives under grounding_control.four_way'
            )
        image = load_image(payload.get('image_path'))
        reference = required_string(payload, 'object_reference')
        coordinate_system = str(
            payload.get('coordinate_system') or COORDINATE_SYSTEM
        )
        if coordinate_system != COORDINATE_SYSTEM:
            raise WorkerRequestError(
                'DINO alignment verifier requires candidate_bbox in '
                f'{COORDINATE_SYSTEM!r}, got {coordinate_system!r}'
            )
        candidate_bbox = payload.get('candidate_bbox')
        if not isinstance(candidate_bbox, (list, tuple)):
            raise WorkerRequestError(
                'candidate_bbox must be a four-element list'
            )
        try:
            normalized_bbox = validate_normalized_box(candidate_bbox)
            candidate = PaddedGeometryVerificationInput(
                image=image,
                object_reference=reference,
                candidate_bbox_padded_normalized_xyxy=normalized_bbox,
                sample_id=str(payload.get('sample_id') or ''),
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError(str(error)) from error
        return serialize_alignment_response(
            self.alignment_scorer.classify_padded_alignment(candidate),
            DINO_ALIGNMENT_MODE,
        )


__all__ = [
    'DINO_ALIGNMENT_MODE',
    'DinoVerifierEndpoint',
]
