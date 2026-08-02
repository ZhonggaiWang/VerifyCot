"""Dispatch binary verifier rejections to a Grounder backend."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..contracts import GrounderBackend, GroundingRequest, validate_normalized_box
from ..contracts.boxes import Box
from ..contracts.errors import (
    ExpertNotConfiguredError,
    ExpertUnavailableError,
)


@dataclass(frozen=True)
class ExpertDispatchResult:
    """Role-neutral correction result consumed by the VoCoT controller."""

    bbox: Box
    source: str
    confidence: float
    expert_role: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExpertDispatcher:
    """Call the Grounder selected by binary alignment routing."""

    def __init__(
            self,
            grounder: Optional[GrounderBackend] = None):
        self.grounder = grounder

    @staticmethod
    def _finalize_result(result, *, role: str, action: str) \
            -> ExpertDispatchResult:
        confidence = float(result.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('expert confidence must be in [0, 1]')
        return ExpertDispatchResult(
            bbox=validate_normalized_box(result.bbox),
            source=str(result.source),
            confidence=confidence,
            expert_role=role,
            action=action,
            metadata=dict(result.metadata),
        )

    def route_grounder(
            self,
            request: GroundingRequest,
            *,
            action: str = 'relocate') -> ExpertDispatchResult:
        """Call the Grounder without requiring a verifier output object."""
        if self.grounder is None:
            raise ExpertNotConfiguredError(
                'routing requested grounder but no grounder is configured'
            )
        result = self.grounder.ground(request)
        return self._finalize_result(
            result,
            role='grounder',
            action=str(action),
        )

    # Canonical spelling for new orchestration code.  ``route_grounder`` is
    # retained because archived controllers and third-party test doubles use
    # that method name.
    dispatch_grounder = route_grounder

__all__ = [
    'ExpertDispatchResult',
    'ExpertDispatcher',
    'ExpertNotConfiguredError',
    'ExpertUnavailableError',
]
