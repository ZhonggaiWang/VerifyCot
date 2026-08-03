"""Dispatch every archived correction action to the configured Grounder."""

from typing import Optional

from ..contracts import GrounderBackend
from ..contracts.verifier import VerificationRequest
from ..core.expert_dispatch import ExpertDispatcher, ExpertDispatchResult
from .contracts import ActionVerifierOutput
from .routing_policy import RoutingDecision


class FourWayExpertDispatcher(ExpertDispatcher):
    """Collapse every misalignment action into one Grounder invocation.

    The archived verifier may still emit ``relocate``, ``expand``, or
    ``tighten`` for diagnostic reproduction.  The active correction design no
    longer has a separate refinement role, so all three actions request an
    independent localization from the same Grounder contract.
    """

    def __init__(
            self,
            grounder: Optional[GrounderBackend] = None):
        super().__init__(grounder=grounder)

    def dispatch(
            self,
            decision: RoutingDecision,
            request: VerificationRequest,
            verification: ActionVerifierOutput) -> ExpertDispatchResult:
        if not decision.requires_expert:
            raise ValueError(
                f'routing action {decision.action!r} does not call an expert'
            )
        return self.dispatch_grounder(
            request.grounding_request(),
            action=decision.action,
        )

    # Historical method spelling used by the archived controller.
    route = dispatch


FourWayExpertDispatchResult = ExpertDispatchResult


__all__ = [
    'FourWayExpertDispatchResult',
    'FourWayExpertDispatcher',
]
