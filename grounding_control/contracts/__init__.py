"""Binary-mainline contracts for verification and Grounder dispatch."""
from .alignment_verifier import (
    ALIGNMENT_SCORE_KINDS,
    ALIGNMENT_OUTPUT_SCHEMA,
    AlignmentScoreKind,
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
)
from .boxes import Box, validate_normalized_box
from .errors import (
    ExpertNotConfiguredError,
    ExpertUnavailableError,
    VerifierFailClosedError,
)
from .grounder import GrounderBackend, GroundingResult
from .requests import (
    CandidateAlignmentRequest,
    CandidateGenerationTrace,
    GroundingRequest,
    VOCOT_PADDED_COORDINATE_SYSTEM,
    VisualInput,
)
from .verifier import VerificationRequest

__all__ = [
    'Box',
    'ALIGNMENT_OUTPUT_SCHEMA',
    'ALIGNMENT_SCORE_KINDS',
    'AlignmentVerifierBackend',
    'AlignmentVerifierOutput',
    'AlignmentScoreKind',
    'CandidateAlignmentRequest',
    'CandidateGenerationTrace',
    'ExpertNotConfiguredError',
    'ExpertUnavailableError',
    'VerifierFailClosedError',
    'GrounderBackend',
    'GroundingRequest',
    'GroundingResult',
    'VerificationRequest',
    'VOCOT_PADDED_COORDINATE_SYSTEM',
    'VisualInput',
    'validate_normalized_box',
]
