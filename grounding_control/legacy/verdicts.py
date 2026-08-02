"""Historical verdict/reason records used by archived repair experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


Verdict = Literal['aligned', 'misaligned', 'unknown', 'uncertain']
Reason = Literal[
    'none',
    'wrong_object',
    'partial_coverage',
    'ambiguous',
    'unsupported',
]


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
            ('misaligned', 'ambiguous'),
            ('misaligned', 'unsupported'),
            ('unknown', 'none'),
            # Legacy compatibility: archived oracle and parse-failure records
            # used ``uncertain/ambiguous`` before region ambiguity was
            # separated from verifier uncertainty.
            ('uncertain', 'ambiguous'),
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
        """Return the legacy fail-open value used by archived experiments."""

        return cls(verdict='uncertain', reason='ambiguous', confidence=0.0)

    @classmethod
    def unknown(cls, confidence: float = 0.0) -> 'VerificationResult':
        """Return a fail-open decision when the verifier cannot judge."""

        return cls(verdict='unknown', reason='none', confidence=confidence)

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


__all__ = [
    'Reason',
    'Verdict',
    'VerificationLookup',
    'VerificationResult',
]
