"""Canonical binary object--coordinate alignment verifier contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from .requests import CandidateAlignmentRequest


ALIGNMENT_OUTPUT_SCHEMA = 'vocot_alignment_score_v1'

AlignmentScoreKind = Literal[
    'calibrated_probability',
    'self_reported_probability',
    'iou_proxy',
    'hard_oracle_label',
]
ALIGNMENT_SCORE_KINDS: Tuple[AlignmentScoreKind, ...] = (
    'calibrated_probability',
    'self_reported_probability',
    'iou_proxy',
    'hard_oracle_label',
)

# ``score_semantics`` was the only scale identifier in the v1 wire payload
# before ``alignment_score_kind`` was introduced.  Keep exact aliases for
# payloads already emitted by the repository, but never guess from arbitrary
# free-form text: a new scored output must state its kind explicitly.
_LEGACY_SCORE_SEMANTICS_TO_KIND: Dict[str, AlignmentScoreKind] = {
    'calibrated_alignment_probability': 'calibrated_probability',
    'qwen_self_reported_label_confidence_transformed_uncalibrated': (
        'self_reported_probability'
    ),
    'candidate_selected_grounding_iou_proxy_uncalibrated': 'iou_proxy',
    'oracle_hard_binary_alignment_label': 'hard_oracle_label',
}


def alignment_score_kind_from_legacy_semantics(
        score_semantics: str) -> Optional[AlignmentScoreKind]:
    """Return the exact score kind for one historical wire identifier."""

    if not isinstance(score_semantics, str):
        return None
    return _LEGACY_SCORE_SEMANTICS_TO_KIND.get(score_semantics.strip())


@dataclass(frozen=True)
class AlignmentVerifierOutput:
    """One candidate-alignment score produced before coordinate commitment.

    ``alignment_score`` has one invariant direction across all backends: a
    larger value means that the candidate region is more likely to support the
    current object reference.  ``score_kind`` is the validated scale consumed
    by routing policy; ``score_semantics`` remains a backend-specific audit
    identifier and must not be used to silently reinterpret a new score.

    Abstention represents an unavailable verifier decision, rather than a
    third alignment class.  Consequently an abstained output carries no
    score, while every non-abstained output carries one finite value in
    ``[0, 1]``.
    """

    alignment_score: Optional[float]
    score_semantics: str
    score_kind: Optional[AlignmentScoreKind] = None
    abstained: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.score_semantics, str)
            or not self.score_semantics.strip()
        ):
            raise ValueError('score_semantics must be a non-empty string')
        object.__setattr__(self, 'score_semantics', self.score_semantics.strip())

        score_kind = self.score_kind
        legacy_score_kind = alignment_score_kind_from_legacy_semantics(
            self.score_semantics
        )
        if score_kind is None:
            score_kind = legacy_score_kind
        elif score_kind not in ALIGNMENT_SCORE_KINDS:
            raise ValueError(
                'score_kind must be one of '
                f'{ALIGNMENT_SCORE_KINDS}, got {score_kind!r}'
            )
        elif legacy_score_kind is not None and score_kind != legacy_score_kind:
            raise ValueError(
                'score_kind conflicts with the registered historical '
                f'score_semantics: {score_kind!r} != {legacy_score_kind!r}'
            )
        object.__setattr__(self, 'score_kind', score_kind)

        if not isinstance(self.abstained, bool):
            raise ValueError('abstained must be a boolean')
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError('error must be a string or None')
        if not isinstance(self.metadata, Mapping):
            raise ValueError('metadata must be a mapping')
        object.__setattr__(self, 'metadata', dict(self.metadata))

        score = self.alignment_score
        if self.abstained:
            if score is not None:
                raise ValueError(
                    'an abstained alignment verifier output cannot carry a score'
                )
            return

        if score is None:
            raise ValueError(
                'a non-abstained alignment verifier output must carry a score'
            )
        if score_kind is None:
            raise ValueError(
                'a non-abstained alignment verifier output must declare '
                'score_kind; only exact historical score_semantics aliases '
                'are inferred for wire compatibility'
            )
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError('alignment_score must be finite and in [0, 1]')
        if (
                score_kind == 'hard_oracle_label'
                and float(score) not in {0.0, 1.0}
        ):
            raise ValueError(
                'hard_oracle_label scores must be exactly 0.0 or 1.0'
            )
        if self.error is not None:
            raise ValueError(
                'a non-abstained alignment verifier output cannot carry an error'
            )
        object.__setattr__(self, 'alignment_score', float(score))

    @classmethod
    def unknown(
            cls,
            error: Optional[str] = None,
            *,
            score_semantics: str = 'unavailable',
            score_kind: Optional[AlignmentScoreKind] = None,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> 'AlignmentVerifierOutput':
        """Construct an explicit verifier abstention/failure."""

        return cls(
            alignment_score=None,
            score_semantics=score_semantics,
            score_kind=score_kind,
            abstained=True,
            error=error,
            metadata=dict(metadata or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the stable alignment-output wire schema."""

        return {
            'verifier_output_schema': ALIGNMENT_OUTPUT_SCHEMA,
            'alignment_score': self.alignment_score,
            'alignment_score_kind': self.score_kind,
            'score_semantics': self.score_semantics,
            'abstained': self.abstained,
            'error': self.error,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'AlignmentVerifierOutput':
        """Validate and deserialize one alignment-output payload."""

        if not isinstance(payload, Mapping):
            raise ValueError('alignment verifier payload must be a mapping')
        schema = payload.get('verifier_output_schema')
        if schema is not None and schema != ALIGNMENT_OUTPUT_SCHEMA:
            raise ValueError(
                f'unsupported alignment verifier schema: {schema!r}'
            )
        return cls(
            alignment_score=payload.get('alignment_score'),
            score_semantics=payload.get('score_semantics', ''),
            score_kind=payload.get('alignment_score_kind'),
            abstained=payload.get('abstained', False),
            error=payload.get('error'),
            metadata=dict(payload.get('metadata') or {}),
        )


class AlignmentVerifierBackend(ABC):
    """Score whether an uncommitted region supports its object reference."""

    @abstractmethod
    def verify_alignment(
            self,
            request: CandidateAlignmentRequest) -> AlignmentVerifierOutput:
        raise NotImplementedError
