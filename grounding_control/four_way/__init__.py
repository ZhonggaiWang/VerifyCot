"""Archived four-action verifier, routing, and specialist interfaces.

The active grounding-control mainline uses binary alignment scores.  This
namespace keeps the completed no_action/relocate/expand/tighten experiments
importable without making them part of the default package surface.
"""

from .adapters import (
    ActionVerifierLegacyAdapter,
    LegacyVerifierActionAdapter,
    action_output_to_legacy_lookup,
    legacy_lookup_to_action_output,
)
from .contracts import (
    ACTION_NAMES,
    ACTION_OUTPUT_SCHEMA,
    ActionVerifierBackend,
    ActionVerifierOutput,
    BoxRefinerBackend,
    RefinementMode,
    RefinementRequest,
    RefinementResult,
    VerifierAction,
)
from .controller import FourWayPrecommitGroundingController
from .experts import OracleBoxRefinerBackend
from .expert_dispatch import (
    FourWayExpertDispatchResult,
    FourWayExpertDispatcher,
)
from .routing_policy import RoutingDecision, RoutingPolicy
from .verifiers import (
    GroundingDinoGeometryClassifier,
    GroundingDinoGeometryVerifierBackend,
    OracleIoUVerifierBackend,
    Qwen25VLGroundingGeometryClassifier,
    Qwen25VLVerifierBackend,
    RemoteActionVerifierBackend,
)

__all__ = [
    'ACTION_NAMES',
    'ACTION_OUTPUT_SCHEMA',
    'ActionVerifierBackend',
    'ActionVerifierLegacyAdapter',
    'ActionVerifierOutput',
    'BoxRefinerBackend',
    'FourWayExpertDispatchResult',
    'FourWayExpertDispatcher',
    'FourWayPrecommitGroundingController',
    'GroundingDinoGeometryClassifier',
    'GroundingDinoGeometryVerifierBackend',
    'LegacyVerifierActionAdapter',
    'OracleBoxRefinerBackend',
    'OracleIoUVerifierBackend',
    'Qwen25VLGroundingGeometryClassifier',
    'Qwen25VLVerifierBackend',
    'RefinementMode',
    'RefinementRequest',
    'RefinementResult',
    'RemoteActionVerifierBackend',
    'RoutingDecision',
    'RoutingPolicy',
    'VerifierAction',
    'action_output_to_legacy_lookup',
    'legacy_lookup_to_action_output',
]
