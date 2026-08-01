"""Explicit migration boundary for historical ``verdict/reason`` outputs."""

from typing import Dict

from ..contracts.action_verifier import (
    ActionVerifierBackend,
    ActionVerifierOutput,
)
from ..contracts.verifier import VerificationRequest, VerifierBackend
from ..types import VerificationLookup, VerificationResult


_LEGACY_PAIR_TO_ACTION = {
    ('aligned', 'none'): 'no_action',
    ('misaligned', 'wrong_object'): 'relocate',
    ('misaligned', 'partial_coverage'): 'expand',
    ('misaligned', 'ambiguous'): 'tighten',
}
_ACTION_TO_LEGACY_PAIR = {
    'no_action': ('aligned', 'none'),
    'relocate': ('misaligned', 'wrong_object'),
    'expand': ('misaligned', 'partial_coverage'),
    'tighten': ('misaligned', 'ambiguous'),
}


def legacy_lookup_to_action_output(
        lookup: VerificationLookup,
        unsupported_action: str = 'unknown',
) -> ActionVerifierOutput:
    """Convert a legacy hard label without inventing fake probabilities."""

    if unsupported_action not in {
        'unknown', 'no_action', 'relocate', 'abstain'
    }:
        raise ValueError(
            'unsupported_action must be unknown, no_action, relocate, '
            'or abstain'
        )
    result = lookup.result
    pair = (result.verdict, result.reason)
    metadata: Dict = {
        **dict(lookup.metadata),
        'legacy_verdict': result.verdict,
        'legacy_reason': result.reason,
        'probability_source': 'unavailable_legacy_hard_label',
        'legacy_missing_oracle_record': lookup.missing_oracle_record,
        'legacy_oracle_candidate_mismatch': (
            lookup.oracle_candidate_mismatch
        ),
    }
    action = _LEGACY_PAIR_TO_ACTION.get(pair)
    if action is not None:
        return ActionVerifierOutput(
            predicted_action=action,
            action_probabilities=None,
            confidence=float(result.confidence),
            abstained=False,
            error=lookup.error,
            metadata=metadata,
        )

    if pair == ('misaligned', 'unsupported'):
        metadata['legacy_unsupported_policy'] = unsupported_action
        if unsupported_action in {'no_action', 'relocate'}:
            return ActionVerifierOutput(
                predicted_action=unsupported_action,
                action_probabilities=None,
                confidence=float(result.confidence),
                abstained=False,
                error=lookup.error,
                metadata=metadata,
            )
        # ``abstain`` and ``unknown`` are equivalent at the verifier boundary.
        return ActionVerifierOutput.unknown(
            error=lookup.error,
            confidence=0.0,
            metadata=metadata,
        )

    # Legacy uncertain/ambiguous and new unknown/none both mean that the
    # verifier did not produce one of the four visual actions.
    return ActionVerifierOutput.unknown(
        error=lookup.error,
        confidence=0.0,
        metadata=metadata,
    )


def action_output_to_legacy_lookup(
        output: ActionVerifierOutput,
) -> VerificationLookup:
    """Expose canonical output to archived consumers requiring old fields."""

    metadata = {
        **dict(output.metadata),
        'predicted_action': output.predicted_action,
        'action_probabilities': (
            None
            if output.action_probabilities is None
            else dict(output.action_probabilities)
        ),
        'action_verifier_abstained': output.abstained,
    }
    if output.abstained or output.predicted_action is None:
        result = VerificationResult.unknown(confidence=0.0)
    else:
        verdict, reason = _ACTION_TO_LEGACY_PAIR[output.predicted_action]
        result = VerificationResult(
            verdict=verdict,
            reason=reason,
            confidence=float(output.confidence),
        )
    return VerificationLookup(
        result=result,
        error=output.error,
        metadata=metadata,
    )


class LegacyVerifierActionAdapter(ActionVerifierBackend):
    """Make an archived ``VerifierBackend`` usable by the new controller."""

    def __init__(
            self,
            backend: VerifierBackend,
            unsupported_action: str = 'unknown'):
        self.backend = backend
        self.unsupported_action = unsupported_action

    def verify_action(
            self,
            request: VerificationRequest) -> ActionVerifierOutput:
        return legacy_lookup_to_action_output(
            self.backend.verify(request),
            unsupported_action=self.unsupported_action,
        )


class ActionVerifierLegacyAdapter(VerifierBackend):
    """Expose a new action verifier to archived evaluation code."""

    def __init__(self, backend: ActionVerifierBackend):
        self.backend = backend

    def verify(self, request: VerificationRequest) -> VerificationLookup:
        return action_output_to_legacy_lookup(
            self.backend.verify_action(request)
        )
