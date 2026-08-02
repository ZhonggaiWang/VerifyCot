"""JSON endpoint for the isolated four-way DINO geometry verifier."""

from typing import Any, Dict, Mapping

from grounding_control.contracts import validate_normalized_box
from grounding_control.coordinates import COORDINATE_SYSTEM
from grounding_control.four_way.verifiers.dino_geometry import (
    GroundingDinoGeometryClassifier,
)
from grounding_control.verifiers.box_geometry import (
    PaddedGeometryVerificationInput,
)
from grounding_control.transport import WorkerRequestError
from grounding_control.transport.request_io import load_image, required_string
from .action_response import serialize_action_output


DINO_GEOMETRY_MODE = 'grounding_dino_geometry'


class DinoGeometryVerifierEndpoint:
    def __init__(self, action_classifier: GroundingDinoGeometryClassifier):
        self.action_classifier = action_classifier

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        image = load_image(payload.get('image_path'))
        reference = required_string(payload, 'object_reference')
        coordinate_system = str(
            payload.get('coordinate_system') or COORDINATE_SYSTEM
        )
        if coordinate_system != COORDINATE_SYSTEM:
            raise WorkerRequestError(
                'DINO geometry verifier requires candidate_bbox in '
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
        mode = str(payload.get('verifier_mode') or DINO_GEOMETRY_MODE)
        if mode != DINO_GEOMETRY_MODE:
            raise WorkerRequestError(
                'four-way DINO verifier_mode must equal '
                f'{DINO_GEOMETRY_MODE!r}'
            )
        return serialize_action_output(
            self.action_classifier.classify_padded_action(candidate),
            DINO_GEOMETRY_MODE,
        )

__all__ = [
    'DINO_GEOMETRY_MODE',
    'DinoGeometryVerifierEndpoint',
]
