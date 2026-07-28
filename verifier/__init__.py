"""Pre-commit object--coordinate verification for VoCoT inference.

The package is deliberately model-agnostic.  The only model-specific hook is
the small binding guard installed in ``VolCanoMistralForCausalLM``.
"""

from .controller import VerifierController, VerifierInferenceResult
from .natural_grounding import audit_natural_coordinates
from .single_candidate_oracle import SingleCandidateOracleVerifier
from .stored_oracle import StoredOracleVerifier
from .types import VerificationResult

__all__ = [
    'audit_natural_coordinates',
    'SingleCandidateOracleVerifier',
    'StoredOracleVerifier',
    'VerificationResult',
    'VerifierController',
    'VerifierInferenceResult',
]
