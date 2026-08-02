"""Binary pre-commit grounding verification and selective correction.

The default package surface is intentionally binary-only.  Historical
four-action routing is available explicitly from
``grounding_control.four_way``; prompt-based repair experiments live under
``grounding_control.legacy``.
"""

from .contracts import (
    ALIGNMENT_OUTPUT_SCHEMA,
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
    Box,
    GrounderBackend,
    GroundingResult,
    VerifierFailClosedError,
    VerificationRequest,
)
from .core import (
    ALIGNMENT_EVENT_SCHEMA,
    AlignmentDecisionBand,
    AlignmentRoutingDecision,
    AlignmentRoutingPolicy,
    AlignmentScoreCalibrationError,
    AlignmentScoreCalibrator,
    AlignmentSystemAction,
    ExpertDispatchResult,
    ExpertDispatcher,
    ExpertNotConfiguredError,
    ExpertUnavailableError,
    PrecommitGroundingController,
    PrecommitInferenceResult,
)
from .experts.grounders import OracleGrounderBackend
from .oracle_targets import OracleTargetResolver
from .verifiers import (
    GroundingDinoAlignmentScorer,
    GroundingDinoAlignmentVerifierBackend,
    OracleAlignmentVerifierBackend,
    Qwen25VLAlignmentVerifierBackend,
    RemoteAlignmentVerifierBackend,
)

__all__ = [
    'ALIGNMENT_EVENT_SCHEMA',
    'ALIGNMENT_OUTPUT_SCHEMA',
    'AlignmentDecisionBand',
    'AlignmentRoutingDecision',
    'AlignmentRoutingPolicy',
    'AlignmentScoreCalibrationError',
    'AlignmentScoreCalibrator',
    'AlignmentSystemAction',
    'AlignmentVerifierBackend',
    'AlignmentVerifierOutput',
    'Box',
    'ExpertDispatchResult',
    'ExpertDispatcher',
    'ExpertNotConfiguredError',
    'ExpertUnavailableError',
    'GrounderBackend',
    'GroundingDinoAlignmentScorer',
    'GroundingDinoAlignmentVerifierBackend',
    'GroundingResult',
    'OracleAlignmentVerifierBackend',
    'OracleGrounderBackend',
    'OracleTargetResolver',
    'PrecommitGroundingController',
    'PrecommitInferenceResult',
    'Qwen25VLAlignmentVerifierBackend',
    'RemoteAlignmentVerifierBackend',
    'VerifierFailClosedError',
    'VerificationRequest',
]
