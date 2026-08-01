"""Dispatch routed corrections to role-specific expert backends."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .contracts import (
    BoxRefinerBackend,
    ActionVerifierOutput,
    GrounderBackend,
    RefinementRequest,
    VerificationRequest,
    validate_normalized_box,
)
from .routing_policy import RoutingDecision
from .types import Box


class ExpertNotConfiguredError(RuntimeError):
    """Raised when a policy requests an expert that was not configured."""


class ExpertUnavailableError(RuntimeError):
    """A configured expert cannot act on this particular request."""


@dataclass(frozen=True)
class ExpertRouteResult:
    """Role-neutral correction result consumed by the VoCoT controller."""

    bbox: Box
    source: str
    confidence: float
    expert_role: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExpertRouter:
    """Call a grounder for relocation and a box refiner for geometry fixes."""

    def __init__(
            self,
            grounder: Optional[GrounderBackend] = None,
            box_refiner: Optional[BoxRefinerBackend] = None):
        self.grounder = grounder
        self.box_refiner = box_refiner

    def route(
            self,
            decision: RoutingDecision,
            request: VerificationRequest,
            verification: ActionVerifierOutput) -> ExpertRouteResult:
        if not decision.requires_expert:
            raise ValueError(
                f'routing action {decision.action!r} does not call an expert'
            )

        if decision.action == 'relocate':
            if self.grounder is None:
                raise ExpertNotConfiguredError(
                    'routing requested relocate but no grounder is configured'
                )
            result = self.grounder.ground(request, verification)
            role = 'grounder'
        else:
            if self.box_refiner is None:
                raise ExpertNotConfiguredError(
                    f'routing requested {decision.action} but no box refiner '
                    'is configured'
                )
            result = self.box_refiner.refine(RefinementRequest(
                verification_request=request,
                verification=verification,
                mode=decision.action,
            ))
            role = 'box_refiner'

        confidence = float(result.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('expert confidence must be in [0, 1]')
        return ExpertRouteResult(
            bbox=validate_normalized_box(result.bbox),
            source=str(result.source),
            confidence=confidence,
            expert_role=role,
            action=decision.action,
            metadata=dict(result.metadata),
        )
