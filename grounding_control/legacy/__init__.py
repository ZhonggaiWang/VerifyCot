"""Archived controllers retained for reproducing earlier repair experiments."""

from .contracts import VerifierBackend
from .natural_grounding import audit_natural_coordinates
from .prompts import RepairMode, build_repair_prompt
from .repair_controller import LegacyRepairController, VerifierInferenceResult
from .oracle_backends import (
    SingleCandidateOracleVerifier,
    StoredOracleVerifier,
)
from .verdicts import (
    Reason,
    Verdict,
    VerificationLookup,
    VerificationResult,
)

__all__ = [
    'Reason',
    'RepairMode',
    'LegacyRepairController',
    'SingleCandidateOracleVerifier',
    'StoredOracleVerifier',
    'Verdict',
    'VerificationLookup',
    'VerificationResult',
    'VerifierBackend',
    'VerifierInferenceResult',
    'audit_natural_coordinates',
    'build_repair_prompt',
]
