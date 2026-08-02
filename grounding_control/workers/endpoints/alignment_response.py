"""Versioned JSON response serialization for binary alignment verifiers."""

from typing import Any, Dict, Optional

from ...contracts.alignment_verifier import (
    ALIGNMENT_OUTPUT_SCHEMA,
    AlignmentVerifierOutput,
)


def serialize_alignment_response(
        output: AlignmentVerifierOutput,
        verifier_mode: str = 'binary_alignment',
        *,
        legacy_aligned_label: Optional[bool] = None,
        legacy_label_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Return the binary schema with optional read-only legacy mirrors.

    The mirrors keep archived worker clients inspectable, but policy code must
    consume ``alignment_score`` rather than infer an action inside the worker.
    """

    response = {
        **output.as_dict(),
        'verifier_mode': verifier_mode,
    }
    if (
            not output.abstained
            and legacy_aligned_label is not None
            and legacy_label_confidence is not None):
        response.update({
            'aligned': bool(legacy_aligned_label),
            'confidence': legacy_label_confidence,
            'routing_action': (
                'no_action' if legacy_aligned_label else 'relocate'
            ),
            'verdict': (
                'aligned' if legacy_aligned_label else 'misaligned'
            ),
            'reason': 'none' if legacy_aligned_label else 'wrong_object',
        })
    else:
        response.update({
            'aligned': None,
            'confidence': None,
            'routing_action': None,
            'verdict': 'unknown' if output.abstained else None,
            'reason': 'none',
        })
    # Make the schema value explicit even if ``as_dict`` is later extended.
    response['verifier_output_schema'] = ALIGNMENT_OUTPUT_SCHEMA
    return response


# Short-term source compatibility for the phase-one function name.  It is an
# alias, not a forwarding function, so identity-sensitive consumers still see
# one serializer implementation.
serialize_alignment_output = serialize_alignment_response


__all__ = [
    'serialize_alignment_output',
    'serialize_alignment_response',
]
