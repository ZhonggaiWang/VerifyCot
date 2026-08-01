"""Contract for judging one uncommitted object--coordinate claim."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from ..types import Box, VerificationLookup


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


class VerifierBackend(ABC):
    """Judge whether a candidate region supports its object reference."""

    @abstractmethod
    def verify(self, request: VerificationRequest) -> VerificationLookup:
        raise NotImplementedError
