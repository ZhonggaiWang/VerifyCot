"""Backend contracts for online object--coordinate verification and routing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from .types import Box, VerificationLookup


@dataclass(frozen=True)
class VerificationRequest:
    """One uncommitted VoCoT coordinate presented to a verifier.

    ``generated_ids`` and ``candidate_span`` are included for deterministic
    oracle replay and auditing.  A real VLM verifier normally consumes only
    ``object_reference``, ``candidate_bbox``, and immutable sample context
    such as the source image.
    """

    sample_id: str
    grounding_step: int
    object_reference: str
    candidate_bbox: Box
    candidate_coordinate_text: str
    generated_ids: Tuple[int, ...]
    candidate_span: Tuple[int, int]
    sample_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingResult:
    """A replacement region returned by a routed grounding expert."""

    bbox: Box
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VerifierBackend(ABC):
    """Judge whether an uncommitted candidate supports its object reference."""

    @abstractmethod
    def verify(self, request: VerificationRequest) -> VerificationLookup:
        raise NotImplementedError


class GrounderBackend(ABC):
    """Return a replacement region after the verifier rejects a candidate."""

    @abstractmethod
    def ground(
            self,
            request: VerificationRequest,
            verification: VerificationLookup) -> GroundingResult:
        raise NotImplementedError


def validate_normalized_box(values: Sequence[float]) -> Box:
    """Validate and normalize an ``xyxy`` box in the unit square."""

    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('bbox must be a four-element list or tuple')
    box = tuple(float(value) for value in values)
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'invalid normalized bbox: {box}')
    return box  # type: ignore[return-value]
