"""Verifier implementations composed from reusable model capabilities."""

from .geometry import (
    GeometryVerificationInput,
    GeometryVerificationLookup,
    GeometryVerifier,
    GroundingGeometryDecision,
    PaddedGeometryVerificationInput,
    PaddedGeometryVerifier,
    route_from_grounding_geometry,
)
from .grounding_dino import (
    GroundingDinoGeometryClassifier,
    GroundingDinoGeometryVerifierBackend,
)
from .oracle import OracleIoUVerifierBackend
from .qwen25_vl import Qwen25VLVerifierBackend
from .remote import RemoteActionVerifierBackend

__all__ = [
    'GeometryVerificationInput',
    'GeometryVerificationLookup',
    'GeometryVerifier',
    'GroundingGeometryDecision',
    'GroundingDinoGeometryClassifier',
    'GroundingDinoGeometryVerifierBackend',
    'OracleIoUVerifierBackend',
    'PaddedGeometryVerificationInput',
    'PaddedGeometryVerifier',
    'Qwen25VLVerifierBackend',
    'RemoteActionVerifierBackend',
    'route_from_grounding_geometry',
]
