"""Neutral request describing one uncommitted object--coordinate claim."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .boxes import Box
from .requests import (
    CandidateAlignmentRequest,
    CandidateGenerationTrace,
    GroundingRequest,
    VisualInput,
)


@dataclass(frozen=True)
class VerificationRequest:
    """One uncommitted VoCoT coordinate presented to a verifier.

    The bbox uses VoCoT's normalized padded-image ``xyxy`` coordinate system.
    Model adapters are responsible for converting it to their native geometry.
    """

    sample_id: str
    grounding_step: int
    object_reference: str
    candidate_bbox: Box
    candidate_coordinate_text: str
    generated_ids: Tuple[int, ...]
    candidate_span: Tuple[int, int]
    sample_context: Mapping[str, Any] = field(default_factory=dict)

    def alignment_request(self) -> CandidateAlignmentRequest:
        """Return the sanitized request visible to a binary verifier."""

        return CandidateAlignmentRequest(
            sample_id=self.sample_id,
            grounding_step=self.grounding_step,
            object_reference=self.object_reference,
            candidate_bbox=self.candidate_bbox,
            visual=VisualInput(
                image=self.sample_context.get('image'),
                image_path=self.sample_context.get('image_path'),
            ),
            coordinate_system=str(self.sample_context.get(
                'coordinate_system',
                'normalized_xyxy_on_center_padded_square',
            )),
            image_mode=(
                None
                if self.sample_context.get('verifier_image_mode') is None
                else str(self.sample_context.get('verifier_image_mode'))
            ),
        )

    def grounding_request(self) -> GroundingRequest:
        """Return a candidate-free, oracle-free request for a Grounder."""

        return GroundingRequest(
            sample_id=self.sample_id,
            grounding_step=self.grounding_step,
            object_reference=self.object_reference,
            visual=VisualInput(
                image=self.sample_context.get('image'),
                image_path=self.sample_context.get('image_path'),
            ),
        )

    def generation_trace(self) -> CandidateGenerationTrace:
        """Return token-only state retained inside the VoCoT controller."""

        return CandidateGenerationTrace(
            candidate_coordinate_text=self.candidate_coordinate_text,
            generated_ids=tuple(self.generated_ids),
            candidate_span=tuple(self.candidate_span),
        )


__all__ = ['VerificationRequest']
