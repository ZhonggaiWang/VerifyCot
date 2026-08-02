"""Role-specific request contracts for pre-commit verification and experts.

The public role contracts intentionally expose only the information required by
that role.  In particular, a Grounder never receives the model's candidate box
or generated token trace, and learned verifier/Grounder requests never carry
oracle annotations through an unstructured context mapping.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .boxes import Box


VOCOT_PADDED_COORDINATE_SYSTEM = (
    'normalized_xyxy_on_center_padded_square'
)


@dataclass(frozen=True)
class VisualInput:
    """Immutable access to the source image used by an external visual role.

    Local backends consume ``image``; remote backends consume ``image_path``.
    Both are optional at the contract level so an oracle backend can operate
    without loading image pixels.  Concrete visual backends validate the input
    they require.
    """

    image: Any = None
    image_path: Optional[str] = None

    def __post_init__(self):
        if self.image_path is not None and not isinstance(self.image_path, str):
            raise TypeError('image_path must be a string or None')


@dataclass(frozen=True)
class CandidateAlignmentRequest:
    """Minimal candidate-aware input visible to a binary verifier."""

    sample_id: str
    grounding_step: int
    object_reference: str
    candidate_bbox: Box
    visual: VisualInput = VisualInput()
    coordinate_system: str = VOCOT_PADDED_COORDINATE_SYSTEM
    image_mode: Optional[str] = None


@dataclass(frozen=True)
class GroundingRequest:
    """Minimal input for locating an object independently from a candidate.

    Deliberately absent: candidate bbox/text, generated token ids/spans,
    verifier output, task question, and oracle annotations.
    """

    sample_id: str
    grounding_step: int
    object_reference: str
    visual: VisualInput = VisualInput()


@dataclass(frozen=True)
class CandidateGenerationTrace:
    """Controller-internal token trace; never passed to learned visual roles."""

    candidate_coordinate_text: str
    generated_ids: tuple
    candidate_span: tuple


__all__ = [
    'CandidateAlignmentRequest',
    'CandidateGenerationTrace',
    'GroundingRequest',
    'VOCOT_PADDED_COORDINATE_SYSTEM',
    'VisualInput',
]
