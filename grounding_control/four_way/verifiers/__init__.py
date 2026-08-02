"""Verifier backends retained for four-action appendix experiments."""

from .dino_geometry import (
    GroundingDinoGeometryClassifier,
    GroundingDinoGeometryVerifierBackend,
    PaddedGeometryActionClassifier,
)
from .geometry import (
    GeometryVerifier,
    GroundingGeometryDecision,
    PaddedGeometryVerifier,
    route_from_grounding_geometry,
)
from .oracle_iou import OracleIoUVerifierBackend
from .qwen25_vl import (
    Qwen25VLGroundingGeometryClassifier,
    Qwen25VLVerifierBackend,
)
from .remote import RemoteActionVerifierBackend

__all__ = [
    'GeometryVerifier',
    'GroundingDinoGeometryClassifier',
    'GroundingDinoGeometryVerifierBackend',
    'GroundingGeometryDecision',
    'OracleIoUVerifierBackend',
    'PaddedGeometryActionClassifier',
    'PaddedGeometryVerifier',
    'Qwen25VLGroundingGeometryClassifier',
    'Qwen25VLVerifierBackend',
    'RemoteActionVerifierBackend',
    'route_from_grounding_geometry',
]
