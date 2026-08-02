"""JSON endpoint for retained four-way Qwen verifier modes.

The binary branch is kept only so the historical mixed Qwen/DINO worker can
replay old runs.  The binary-mainline endpoint lives in
``grounding_control.workers.endpoints.qwen_alignment_verifier`` and never
imports this module.
"""

from typing import Any, Dict, Mapping

from grounding_control.contracts import validate_normalized_box
from grounding_control.four_way.contracts.action_verifier import (
    ActionVerifierOutput,
)
from grounding_control.four_way.verifiers.qwen25_vl.backend import (
    Qwen25VLVerifierBackend,
)
from grounding_control.four_way.verifiers.qwen25_vl.geometry import (
    Qwen25VLGroundingGeometryClassifier,
)
from grounding_control.four_way.verifiers.qwen25_vl.inputs import (
    GroundingActionInput,
)
from grounding_control.models.qwen25_vl.preprocessing import (
    GROUNDING_ACTION_IMAGE_MODES,
)
from grounding_control.verifiers.qwen25_vl.backend import (
    binary_lookup_to_alignment_output,
)
from grounding_control.verifiers.qwen25_vl.classifier import (
    Qwen25VLBinaryAlignmentClassifier,
)
from grounding_control.verifiers.qwen25_vl.inputs import (
    CandidateVerificationInput,
)
from grounding_control.verifiers.qwen25_vl.prompt import BINARY_IMAGE_MODES
from grounding_control.verifiers.qwen25_vl.rendering import (
    COORDINATE_SYSTEM,
)
from grounding_control.transport import WorkerRequestError
from grounding_control.transport.request_io import (
    finite_pixel_box,
    load_image,
    required_string,
)
from grounding_control.workers.endpoints.alignment_response import (
    serialize_alignment_response,
)
from .action_response import serialize_action_output


VERIFIER_MODES = (
    'binary_alignment',
    'routing_four_way',
    'grounding_geometry',
)


class QwenFourWayVerifierEndpoint:
    def __init__(
            self,
            backend: Qwen25VLVerifierBackend,
            geometry_classifier: Qwen25VLGroundingGeometryClassifier,
            alignment_classifier: Qwen25VLBinaryAlignmentClassifier = None,
            default_mode: str = 'routing_four_way',
            default_image_mode: str = 'bbox_image_only'):
        if default_mode not in VERIFIER_MODES:
            raise ValueError(
                f'default_mode must be one of {VERIFIER_MODES}'
            )
        self.backend = backend
        self.alignment_classifier = (
            backend.binary_classifier
            if alignment_classifier is None
            else alignment_classifier
        )
        self.geometry_classifier = geometry_classifier
        self.default_mode = default_mode
        self.default_image_mode = default_image_mode

    @staticmethod
    def _response(
            output: ActionVerifierOutput,
            mode: str) -> Dict[str, Any]:
        return serialize_action_output(output, mode)

    @classmethod
    def _unknown(cls, error, metadata, mode):
        canonical_metadata = {
            **dict(metadata),
            'probability_source': 'unavailable_abstained',
        }
        return cls._response(ActionVerifierOutput.unknown(
            error=error,
            confidence=0.0,
            metadata=canonical_metadata,
        ), mode)

    @classmethod
    def _hard_action(
            cls,
            action,
            confidence,
            error,
            metadata,
            mode):
        canonical_metadata = {
            **dict(metadata),
            'probability_source': 'unavailable_hard_label',
        }
        output = ActionVerifierOutput(
            predicted_action=action,
            action_probabilities=None,
            confidence=0.0 if confidence is None else float(confidence),
            abstained=False,
            error=error,
            metadata=canonical_metadata,
        )
        return cls._response(output, mode)

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        image = load_image(payload.get('image_path'))
        reference = required_string(payload, 'object_reference')
        sample_id = str(payload.get('sample_id') or '')
        mode = str(payload.get('verifier_mode') or self.default_mode)
        image_mode = str(
            payload.get('image_mode') or self.default_image_mode
        )
        if mode not in VERIFIER_MODES:
            raise WorkerRequestError(
                f'verifier_mode must be one of {VERIFIER_MODES}'
            )

        if mode == 'grounding_geometry':
            if image_mode not in GROUNDING_ACTION_IMAGE_MODES:
                raise WorkerRequestError(
                    'grounding_geometry image_mode must be one of '
                    f'{GROUNDING_ACTION_IMAGE_MODES}'
                )
            candidate_box = finite_pixel_box(
                payload.get('candidate_bbox_original_pixel_xyxy'),
                image.size,
                'candidate_bbox_original_pixel_xyxy',
            )
            lookup = self.geometry_classifier.classify(
                GroundingActionInput(
                    image=image,
                    object_reference=reference,
                    candidate_bbox_pixel_xyxy=tuple(candidate_box),
                    sample_id=sample_id,
                ),
                image_mode=image_mode,
            )
            if lookup.status is None:
                return self._unknown(lookup.error, lookup.metadata, mode)
            return self._hard_action(
                lookup.status,
                lookup.confidence,
                lookup.error,
                lookup.metadata,
                mode,
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
                'Qwen verifier requires candidate_bbox in '
                f'{COORDINATE_SYSTEM!r}, got {coordinate_system!r}'
            )
        try:
            normalized = validate_normalized_box(
                payload.get('candidate_bbox')
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError(str(error)) from error
        candidate = CandidateVerificationInput(
            image=image,
            object_reference=reference,
            candidate_bbox=normalized,
            sample_id=sample_id,
            coordinate_system=coordinate_system,
        )

        if mode == 'binary_alignment':
            lookup = self.alignment_classifier.classify(
                candidate,
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

        lookup = self.backend.classify_routing_candidate(
            candidate,
            image_mode=image_mode,
        )
        if lookup.status is None:
            return self._unknown(lookup.error, lookup.metadata, mode)
        return self._hard_action(
            lookup.status,
            lookup.confidence,
            lookup.error,
            lookup.metadata,
            mode,
        )


# Compatibility name retained for phase-one callers.
QwenVerifierEndpoint = QwenFourWayVerifierEndpoint


__all__ = [
    'QwenFourWayVerifierEndpoint',
    'QwenVerifierEndpoint',
    'VERIFIER_MODES',
]
