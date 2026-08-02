"""Dispatch four-action corrections to Grounder and BoxRefiner experts."""

from typing import Optional

from ..contracts import GrounderBackend
from ..contracts.errors import ExpertNotConfiguredError
from ..contracts.verifier import VerificationRequest
from ..core.expert_dispatch import ExpertDispatcher, ExpertDispatchResult
from .contracts import (
    ActionVerifierOutput,
    BoxRefinerBackend,
    RefinementRequest,
)
from .routing_policy import RoutingDecision


class FourWayExpertDispatcher(ExpertDispatcher):
    """Route relocate to Grounder and expand/tighten to BoxRefiner."""

    def __init__(
            self,
            grounder: Optional[GrounderBackend] = None,
            box_refiner: Optional[BoxRefinerBackend] = None):
        super().__init__(grounder=grounder)
        self.box_refiner = box_refiner

    def dispatch(
            self,
            decision: RoutingDecision,
            request: VerificationRequest,
            verification: ActionVerifierOutput) -> ExpertDispatchResult:
        if not decision.requires_expert:
            raise ValueError(
                f'routing action {decision.action!r} does not call an expert'
            )
        if decision.action == 'relocate':
            return self.dispatch_grounder(
                request.grounding_request(),
                action=decision.action,
            )
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
        return self._finalize_result(
            result,
            role='box_refiner',
            action=decision.action,
        )

    # Historical method spelling used by the archived controller.
    route = dispatch


FourWayExpertDispatchResult = ExpertDispatchResult


__all__ = [
    'FourWayExpertDispatchResult',
    'FourWayExpertDispatcher',
]
