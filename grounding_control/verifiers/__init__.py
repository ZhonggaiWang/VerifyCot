"""Verifier implementations composed from reusable model capabilities."""

from .box_geometry import (
    BoxGeometryMeasurement,
    GeometryVerificationInput,
    GeometryVerificationLookup,
    PaddedGeometryComparator,
    PaddedGeometryComparison,
    PaddedGeometryVerificationInput,
    measure_box_geometry,
)
from .dino import (
    DINO_ALIGNMENT_SCORE_SEMANTICS,
    GroundingDinoAlignmentClassifier,
    GroundingDinoAlignmentScorer,
    GroundingDinoAlignmentVerifierBackend,
)
from .oracle_iou import (
    ORACLE_ALIGNMENT_SCORE_SEMANTICS,
    OracleAlignmentVerifierBackend,
)
from .qwen25_vl import (
    QWEN_ALIGNMENT_SCORE_SEMANTICS,
    Qwen25VLAlignmentVerifierBackend,
)
from .remote import RemoteAlignmentVerifierBackend

__all__ = [
    'BoxGeometryMeasurement',
    'DINO_ALIGNMENT_SCORE_SEMANTICS',
    'GeometryVerificationInput',
    'GeometryVerificationLookup',
    'GroundingDinoAlignmentClassifier',
    'GroundingDinoAlignmentScorer',
    'GroundingDinoAlignmentVerifierBackend',
    'ORACLE_ALIGNMENT_SCORE_SEMANTICS',
    'OracleAlignmentVerifierBackend',
    'PaddedGeometryComparator',
    'PaddedGeometryComparison',
    'PaddedGeometryVerificationInput',
    'QWEN_ALIGNMENT_SCORE_SEMANTICS',
    'Qwen25VLAlignmentVerifierBackend',
    'RemoteAlignmentVerifierBackend',
    'measure_box_geometry',
]
