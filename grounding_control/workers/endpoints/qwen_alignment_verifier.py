"""JSON endpoint for the binary Qwen object--region alignment verifier."""

from typing import Any, Dict, Mapping

from ...contracts import validate_normalized_box
from ...transport import WorkerRequestError
from ...transport.request_io import load_image, required_string
from ...verifiers.qwen25_vl.backend import (
    binary_lookup_to_alignment_output,
)
from ...verifiers.qwen25_vl.classifier import (
    Qwen25VLBinaryAlignmentClassifier,
)
from ...verifiers.qwen25_vl.inputs import CandidateVerificationInput
from ...verifiers.qwen25_vl.prompt import BINARY_IMAGE_MODES
from ...verifiers.qwen25_vl.rendering import COORDINATE_SYSTEM
from .alignment_response import serialize_alignment_response


QWEN_ALIGNMENT_MODE = 'binary_alignment'


class QwenAlignmentVerifierEndpoint:
    """Serve only the binary-alignment protocol used by the mainline."""

    def __init__(
            self,
            alignment_classifier: Qwen25VLBinaryAlignmentClassifier,
            default_image_mode: str = 'bbox_image_only'):
        if default_image_mode not in BINARY_IMAGE_MODES:
            raise ValueError(
                f'default_image_mode must be one of {BINARY_IMAGE_MODES}'
            )
        self.alignment_classifier = alignment_classifier
        self.default_image_mode = str(default_image_mode)

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        mode = str(payload.get('verifier_mode') or QWEN_ALIGNMENT_MODE)
        if mode != QWEN_ALIGNMENT_MODE:
            raise WorkerRequestError(
                'binary Qwen verifier_mode must equal '
                f'{QWEN_ALIGNMENT_MODE!r}; four-way verification lives '
                'under grounding_control.four_way'
            )
        image = load_image(payload.get('image_path'))
        reference = required_string(payload, 'object_reference')
        sample_id = str(payload.get('sample_id') or '')
        image_mode = str(
            payload.get('image_mode') or self.default_image_mode
        )
        if image_mode not in BINARY_IMAGE_MODES:
            raise WorkerRequestError(
                f'image_mode must be one of {BINARY_IMAGE_MODES}'
            )
        coordinate_system = str(
            payload.get('coordinate_system') or COORDINATE_SYSTEM
        )
        if coordinate_system != COORDINATE_SYSTEM:
            raise WorkerRequestError(
                'Qwen alignment verifier requires candidate_bbox in '
                f'{COORDINATE_SYSTEM!r}, got {coordinate_system!r}'
            )
        try:
            normalized = validate_normalized_box(
                payload.get('candidate_bbox')
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError(str(error)) from error
        lookup = self.alignment_classifier.classify(
            CandidateVerificationInput(
                image=image,
                object_reference=reference,
                candidate_bbox=normalized,
                sample_id=sample_id,
                coordinate_system=coordinate_system,
            ),
            image_mode=image_mode,
        )
        output = binary_lookup_to_alignment_output(
            lookup,
            extra_metadata={
                'alignment_backend': 'qwen25_vl_binary_alignment',
                'requested_image_mode': image_mode,
            },
        )
        return serialize_alignment_response(
            output,
            mode,
            legacy_aligned_label=lookup.aligned,
            legacy_label_confidence=lookup.confidence,
        )


__all__ = [
    'QWEN_ALIGNMENT_MODE',
    'QwenAlignmentVerifierEndpoint',
]
