"""Compatibility adapters between legacy verdicts and action outputs."""

from .action_output import (
    ActionVerifierLegacyAdapter,
    LegacyVerifierActionAdapter,
    action_output_to_legacy_lookup,
    legacy_lookup_to_action_output,
)

__all__ = [
    'ActionVerifierLegacyAdapter',
    'LegacyVerifierActionAdapter',
    'action_output_to_legacy_lookup',
    'legacy_lookup_to_action_output',
]
