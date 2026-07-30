"""Object--coordinate verification and routing for VoCoT inference.

The active controller routes verifier rejections to a grounding backend.
Historical prompt-repair controllers remain available under ``verifier.legacy``
solely for reproducing earlier experiments.
"""

from .backend import (
    GrounderBackend,
    GroundingResult,
    VerificationRequest,
    VerifierBackend,
)
from .backends.oracle import OracleGrounderBackend, OracleIoUVerifierBackend
from .backends.qwen25_vl import Qwen25VLVerifierBackend
from .natural_grounding import audit_natural_coordinates
from .routing_controller import RoutingController, RoutingInferenceResult
from .types import VerificationResult

__all__ = [
    'audit_natural_coordinates',
    'GrounderBackend',
    'GroundingResult',
    'OracleGrounderBackend',
    'OracleIoUVerifierBackend',
    'Qwen25VLVerifierBackend',
    'RoutingController',
    'RoutingInferenceResult',
    'VerificationRequest',
    'VerificationResult',
    'VerifierBackend',
]
