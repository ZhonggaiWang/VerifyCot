"""Shared verdict types used by legacy and routing verifier backends."""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


Verdict = Literal['aligned', 'misaligned', 'uncertain']
Reason = Literal[
    'none',
    'wrong_object',
    'partial_coverage',
    'ambiguous',
    'unsupported',
]
Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    reason: Reason
    confidence: float

    def __post_init__(self):
        valid_pairs = {
            ('aligned', 'none'),
            ('misaligned', 'wrong_object'),
            ('misaligned', 'partial_coverage'),
            ('uncertain', 'ambiguous'),
            ('misaligned', 'unsupported'),
        }
        if (self.verdict, self.reason) not in valid_pairs:
            raise ValueError(
                'illegal verifier result combination: '
                f'{self.verdict!r} / {self.reason!r}'
            )
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ValueError('verifier confidence must be numeric')
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError('verifier confidence must be in [0, 1]')

    @classmethod
    def uncertain(cls) -> 'VerificationResult':
        return cls(verdict='uncertain', reason='ambiguous', confidence=0.0)

    @classmethod
    def unsupported(cls, confidence: float) -> 'VerificationResult':
        return cls(
            verdict='misaligned',
            reason='unsupported',
            confidence=confidence,
        )


@dataclass(frozen=True)
class VerificationLookup:
    """A verifier result plus backend diagnostics exposed in event logs.

    The two oracle flags remain for backward compatibility with the archived
    prompt-repair experiments.  New backends should put backend-specific
    diagnostics in ``metadata``.
    """

    result: VerificationResult
    missing_oracle_record: bool = False
    oracle_candidate_mismatch: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
