"""Object--coordinate verification and expert routing for VoCoT inference.

The active controller consumes the canonical four-action verifier output and
routes accepted errors to a grounder or box refiner. Historical prompt-repair
controllers remain under ``verifier.legacy`` solely for reproducing earlier
experiments.
"""

from .contracts import (
    ACTION_NAMES,
    ACTION_OUTPUT_SCHEMA,
    ActionVerifierBackend,
    ActionVerifierOutput,
    BoxRefinerBackend,
    GrounderBackend,
    GroundingResult,
    RefinementRequest,
    RefinementResult,
    VerificationRequest,
    VerifierBackend,
    VerifierAction,
)
from .adapters import (
    ActionVerifierLegacyAdapter,
    LegacyVerifierActionAdapter,
)
from .experts import (
    OracleBoxRefinerBackend,
    OracleGrounderBackend,
)
from .oracle_targets import OracleTargetResolver
from .expert_router import (
    ExpertNotConfiguredError,
    ExpertUnavailableError,
    ExpertRouteResult,
    ExpertRouter,
)
from .natural_grounding import audit_natural_coordinates
from .routing_policy import RoutingDecision, RoutingPolicy
from .routing_controller import RoutingController, RoutingInferenceResult
from .verifier_backends import (
    GroundingDinoGeometryVerifierBackend,
    OracleIoUVerifierBackend,
    Qwen25VLVerifierBackend,
    RemoteActionVerifierBackend,
)
from .types import VerificationResult

__all__ = [
    'audit_natural_coordinates',
    'ACTION_NAMES',
    'ACTION_OUTPUT_SCHEMA',
    'ActionVerifierBackend',
    'ActionVerifierLegacyAdapter',
    'ActionVerifierOutput',
    'BoxRefinerBackend',
    'ExpertNotConfiguredError',
    'ExpertUnavailableError',
    'ExpertRouteResult',
    'ExpertRouter',
    'GrounderBackend',
    'GroundingResult',
    'GroundingDinoGeometryVerifierBackend',
    'LegacyVerifierActionAdapter',
    'OracleGrounderBackend',
    'OracleIoUVerifierBackend',
    'OracleBoxRefinerBackend',
    'OracleTargetResolver',
    'Qwen25VLVerifierBackend',
    'RefinementRequest',
    'RefinementResult',
    'RemoteActionVerifierBackend',
    'RoutingController',
    'RoutingDecision',
    'RoutingInferenceResult',
    'RoutingPolicy',
    'VerificationRequest',
    'VerificationResult',
    'VerifierBackend',
    'VerifierAction',
]
