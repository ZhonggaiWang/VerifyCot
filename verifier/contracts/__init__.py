"""Public contracts for verifier decisions and routed expert backends.

The controller depends only on these role-oriented interfaces.  Model-specific
code (Qwen, Grounding DINO, future SAM/refiners) may implement more reusable
internal capabilities, but those capabilities are deliberately not exposed as
controller roles.
"""

from .box_refiner import (
    BoxRefinerBackend,
    RefinementMode,
    RefinementRequest,
    RefinementResult,
)
from .action_verifier import (
    ACTION_NAMES,
    ACTION_OUTPUT_SCHEMA,
    ActionVerifierBackend,
    ActionVerifierOutput,
    VerifierAction,
)
from .boxes import validate_normalized_box
from .grounder import GrounderBackend, GroundingResult
from .verifier import VerificationRequest, VerifierBackend

__all__ = [
    'BoxRefinerBackend',
    'ACTION_NAMES',
    'ACTION_OUTPUT_SCHEMA',
    'ActionVerifierBackend',
    'ActionVerifierOutput',
    'GrounderBackend',
    'GroundingResult',
    'RefinementMode',
    'RefinementRequest',
    'RefinementResult',
    'VerificationRequest',
    'VerifierBackend',
    'VerifierAction',
    'validate_normalized_box',
]
