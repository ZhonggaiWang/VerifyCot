"""JSON adapter for a Grounding DINO plus geometry verifier."""

from typing import Any, Dict, Mapping

from ...coordinates import COORDINATE_SYSTEM
from ...runtime import WorkerRequestError
from ...runtime.request_io import load_image, required_string
from ...verifier_backends import (
    GroundingDinoGeometryClassifier,
    PaddedGeometryVerificationInput,
)
from .action_output import serialize_action_output


DINO_GEOMETRY_MODE = 'grounding_dino_geometry'


class DinoGeometryVerifierEndpoint:
    def __init__(self, classifier: GroundingDinoGeometryClassifier):
        self.classifier = classifier

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
            output = self.classifier.classify_padded_action(
                PaddedGeometryVerificationInput(
                    image=image,
                    object_reference=reference,
                    candidate_bbox_padded_normalized_xyxy=tuple(
                        float(value) for value in candidate_bbox
                    ),
                    sample_id=str(payload.get('sample_id') or ''),
                )
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError(str(error)) from error
        return serialize_action_output(output, DINO_GEOMETRY_MODE)
