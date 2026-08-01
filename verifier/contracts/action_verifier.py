"""Canonical four-action output contract for learned and heuristic verifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from .verifier import VerificationRequest


VerifierAction = Literal[
    'no_action',
    'relocate',
    'expand',
    'tighten',
]
ACTION_NAMES: Tuple[VerifierAction, ...] = (
    'no_action',
    'relocate',
    'expand',
    'tighten',
)
ACTION_OUTPUT_SCHEMA = 'vocot_four_action_v1'


@dataclass(frozen=True)
class ActionVerifierOutput:
    """One verifier decision expressed directly in routing-action space.

    A learned four-way verifier should always populate
    ``action_probabilities``. Compatibility adapters for historical hard-label
    verifiers may set it to ``None`` and must identify that limitation in
    ``metadata["probability_source"]``.

    ``abstained`` represents verifier uncertainty or failure; it is not a
    fifth visual-error class. The routing policy independently decides whether
    an abstention fails open, stops, or invokes another verifier.
    """

    predicted_action: Optional[VerifierAction]
    action_probabilities: Optional[Dict[str, float]]
    confidence: float
    abstained: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError('action verifier confidence must be in [0, 1]')
        if self.abstained:
            if self.predicted_action is not None:
                raise ValueError(
                    'an abstained verifier output cannot predict an action'
                )
        elif self.predicted_action not in ACTION_NAMES:
            raise ValueError(
                'a non-abstained verifier output must predict one of '
                f'{ACTION_NAMES}'
            )

        probabilities = self.action_probabilities
        if probabilities is None:
            return
        if not isinstance(probabilities, Mapping):
            raise ValueError('action_probabilities must be a mapping or None')
        if set(probabilities) != set(ACTION_NAMES):
            raise ValueError(
                'action_probabilities must contain exactly '
                f'{ACTION_NAMES}'
            )
        normalized = {
            action: float(probabilities[action])
            for action in ACTION_NAMES
        }
        if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in normalized.values()):
            raise ValueError('action probabilities must be finite and in [0, 1]')
        if not math.isclose(
                sum(normalized.values()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5):
            raise ValueError('action probabilities must sum to 1')
        if self.predicted_action is not None:
            maximum = max(normalized.values())
            if normalized[self.predicted_action] < maximum - 1e-8:
                raise ValueError(
                    'predicted_action must attain the maximum probability'
                )
        if not math.isclose(
                float(self.confidence),
                max(normalized.values()),
                rel_tol=0.0,
                abs_tol=1e-5):
            raise ValueError(
                'confidence must equal the maximum action probability'
            )
        object.__setattr__(self, 'action_probabilities', normalized)

    @classmethod
    def unknown(
            cls,
            error: Optional[str] = None,
            action_probabilities: Optional[Dict[str, float]] = None,
            confidence: float = 0.0,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> 'ActionVerifierOutput':
        return cls(
            predicted_action=None,
            action_probabilities=action_probabilities,
            confidence=confidence,
            abstained=True,
            error=error,
            metadata=dict(metadata or {}),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            'verifier_output_schema': ACTION_OUTPUT_SCHEMA,
            'predicted_action': self.predicted_action,
            'action_probabilities': (
                None
                if self.action_probabilities is None
                else dict(self.action_probabilities)
            ),
            'confidence': float(self.confidence),
            'abstained': bool(self.abstained),
            'error': self.error,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'ActionVerifierOutput':
        if not isinstance(payload, Mapping):
            raise ValueError('action verifier payload must be a mapping')
        schema = payload.get('verifier_output_schema')
        if schema is not None and schema != ACTION_OUTPUT_SCHEMA:
            raise ValueError(
                f'unsupported action verifier schema: {schema!r}'
            )
        return cls(
            predicted_action=payload.get('predicted_action'),
            action_probabilities=payload.get('action_probabilities'),
            confidence=payload.get('confidence', 0.0),
            abstained=bool(payload.get('abstained', False)),
            error=payload.get('error'),
            metadata=dict(payload.get('metadata') or {}),
        )


class ActionVerifierBackend(ABC):
    """Judge a candidate and return the canonical four-action schema."""

    @abstractmethod
    def verify_action(
            self,
            request: VerificationRequest) -> ActionVerifierOutput:
        raise NotImplementedError
