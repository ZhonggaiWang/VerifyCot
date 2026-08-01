"""JSON request adapter for Qwen verifier protocols."""

from typing import Any, Dict, Mapping

from ...contracts import (
    ActionVerifierOutput,
    validate_normalized_box,
)
from ...models.qwen25_vl.preprocessing import GROUNDING_ACTION_IMAGE_MODES
from ...verifier_backends.qwen25_vl import (
    BINARY_IMAGE_MODES,
    CandidateVerificationInput,
    GroundingActionInput,
    Qwen25VLGroundingGeometryClassifier,
    Qwen25VLVerifierBackend,
)
from ...runtime import WorkerRequestError
from ...runtime.request_io import (
    finite_pixel_box,
    load_image,
    required_string,
)
from .action_output import serialize_action_output


VERIFIER_MODES = (
    'binary_alignment',
    'routing_four_way',
    'grounding_geometry',
)


class QwenVerifierEndpoint:
    def __init__(
            self,
            backend: Qwen25VLVerifierBackend,
            geometry_classifier: Qwen25VLGroundingGeometryClassifier,
            default_mode: str = 'routing_four_way',
            default_image_mode: str = 'bbox_image_only'):
        if default_mode not in VERIFIER_MODES:
            raise ValueError(
                f'default_mode must be one of {VERIFIER_MODES}'
            )
        self.backend = backend
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
        )

        if mode == 'binary_alignment':
            lookup = self.backend.verify_binary_alignment_candidate(
                candidate,
                image_mode=image_mode,
            )
            if lookup.aligned is None:
                return self._unknown(lookup.error, lookup.metadata, mode)
            action = (
                'no_action' if lookup.aligned else 'relocate'
            )
            return self._hard_action(
                action,
                lookup.confidence,
                lookup.error,
                lookup.metadata,
                mode,
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
