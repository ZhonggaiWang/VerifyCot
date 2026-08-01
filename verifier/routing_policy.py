"""Pure four-action verifier output to system routing policy."""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

from .adapters import legacy_lookup_to_action_output
from .contracts import ActionVerifierOutput, VerifierAction
from .types import VerificationLookup


RoutingAction = Literal[
    'no_action',
    'relocate',
    'expand',
    'tighten',
    'abstain',
]
UnsupportedAction = Literal['no_action', 'relocate', 'abstain']
UnknownAction = Literal['no_action', 'abstain']
VerifierPolicyInput = Union[ActionVerifierOutput, VerificationLookup]


@dataclass(frozen=True)
class RoutingDecision:
    """One auditable system action selected from canonical verifier output."""

    action: RoutingAction
    router_action: str
    verifier_action: Optional[VerifierAction]
    verifier_abstained: bool
    confidence: float
    # Compatibility fields for archived summary code.
    verifier_verdict: str
    verifier_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def requires_expert(self) -> bool:
        return self.action in {'relocate', 'expand', 'tighten'}

    @property
    def expert_role(self):
        if self.action == 'relocate':
            return 'grounder'
        if self.action in {'expand', 'tighten'}:
            return 'box_refiner'
        return None


class RoutingPolicy:
    """Apply confidence and abstention policy after four-way prediction.

    ``unsupported_action`` exists only for archived ``verdict/reason``
    outputs. New action verifiers never predict ``unsupported``; they abstain
    instead.
    """

    _ACTION_TO_LEGACY = {
        'no_action': ('aligned', 'none'),
        'relocate': ('misaligned', 'wrong_object'),
        'expand': ('misaligned', 'partial_coverage'),
        'tighten': ('misaligned', 'ambiguous'),
    }

    def __init__(
            self,
            confidence_threshold: float = 0.8,
            unsupported_action: UnsupportedAction = 'no_action',
            unknown_action: UnknownAction = 'no_action'):
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError('confidence_threshold must be in [0, 1]')
        if unsupported_action not in {'no_action', 'relocate', 'abstain'}:
            raise ValueError(
                'unsupported_action must be no_action, relocate, or abstain'
            )
        if unknown_action not in {'no_action', 'abstain'}:
            raise ValueError('unknown_action must be no_action or abstain')
        self.confidence_threshold = float(confidence_threshold)
        self.unsupported_action = unsupported_action
        self.unknown_action = unknown_action

    @staticmethod
    def _router_action(action: RoutingAction, detail: str) -> str:
        if action == 'relocate':
            return 'routed_to_grounder'
        if action == 'expand':
            return 'routed_to_box_refiner_expand'
        if action == 'tighten':
            return 'routed_to_box_refiner_tighten'
        if action == 'abstain':
            return 'routing_abstained'
        return detail

    def _canonical_output(
            self,
            value: VerifierPolicyInput) -> ActionVerifierOutput:
        if isinstance(value, ActionVerifierOutput):
            return value
        if isinstance(value, VerificationLookup):
            return legacy_lookup_to_action_output(
                value,
                unsupported_action=self.unsupported_action,
            )
        raise TypeError(
            'routing policy requires ActionVerifierOutput or '
            'VerificationLookup'
        )


    #根据verifier的output，确定router应该是路由到 no action,还是expect
    def decide(self, value: VerifierPolicyInput) -> RoutingDecision:
        output = self._canonical_output(value)
        confidence = float(output.confidence)
        legacy_verdict = output.metadata.get('legacy_verdict')
        legacy_reason = output.metadata.get('legacy_reason')
        if legacy_verdict is None:
            if output.predicted_action is None:
                legacy_verdict, legacy_reason = 'unknown', 'none'
            else:
                legacy_verdict, legacy_reason = self._ACTION_TO_LEGACY[
                    output.predicted_action
                ]

        policy_abstained = bool(output.abstained)
        abstention_source = None
        if output.abstained:
            abstention_source = 'verifier'
        elif confidence < self.confidence_threshold:
            policy_abstained = True
            abstention_source = 'low_confidence'

        force_unsupported_abstain = bool(
            output.metadata.get('legacy_reason') == 'unsupported'
            and self.unsupported_action == 'abstain'
        )
        if policy_abstained:
            action: RoutingAction = (
                'abstain'
                if force_unsupported_abstain
                or self.unknown_action == 'abstain'
                else 'no_action'
            )
            if action == 'no_action':
                detail = (
                    'low_confidence_fail_open'
                    if abstention_source == 'low_confidence'
                    else 'verifier_abstained_fail_open'
                )
            else:
                detail = 'routing_abstained'
        else:
            action = output.predicted_action  # type: ignore[assignment]
            detail = (
                'verified_accept'
                if action == 'no_action'
                else 'routed_to_specialist'
            )

        metadata: Dict[str, Any] = {
            'confidence_threshold': self.confidence_threshold,
            'unsupported_action': self.unsupported_action,
            'unknown_action': self.unknown_action,
            'verifier_predicted_action': output.predicted_action,
            'verifier_abstained': output.abstained,
            'policy_abstained': policy_abstained,
            'abstention_source': abstention_source,
            'action_probabilities': (
                None
                if output.action_probabilities is None
                else dict(output.action_probabilities)
            ),
        }
        return RoutingDecision(
            action=action,
            router_action=self._router_action(action, detail),
            verifier_action=output.predicted_action,
            verifier_abstained=policy_abstained,
            confidence=confidence,
            verifier_verdict=str(legacy_verdict),
            verifier_reason=str(legacy_reason),
            metadata=metadata,
        )
