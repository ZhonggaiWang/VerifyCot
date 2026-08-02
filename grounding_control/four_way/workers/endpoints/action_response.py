"""Canonical JSON serialization for four-way verifier workers."""

from typing import Any, Dict

from ...adapters.action_output import action_output_to_legacy_lookup
from ...contracts.action_verifier import (
    ACTION_OUTPUT_SCHEMA,
    ActionVerifierOutput,
)


def serialize_action_output(
        output: ActionVerifierOutput,
        verifier_mode: str,
) -> Dict[str, Any]:
    """Return the versioned action schema plus read-only legacy mirrors."""

    legacy = action_output_to_legacy_lookup(output)
    verdict = output.metadata.get(
        'legacy_verdict',
        legacy.result.verdict,
    )
    reason = output.metadata.get(
        'legacy_reason',
        legacy.result.reason,
    )
    return {
        'verifier_mode': verifier_mode,
        'verifier_output_schema': ACTION_OUTPUT_SCHEMA,
        'predicted_action': output.predicted_action,
        'action_probabilities': (
            None
            if output.action_probabilities is None
            else dict(output.action_probabilities)
        ),
        'confidence': float(output.confidence),
        'abstained': bool(output.abstained),
        'error': output.error,
        'metadata': dict(output.metadata),
        # Compatibility mirrors for archived JSONL readers only.
        'routing_action': output.predicted_action,
        'verdict': verdict,
        'reason': reason,
    }


__all__ = ['serialize_action_output']
