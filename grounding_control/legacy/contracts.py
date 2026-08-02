"""Historical hard-verdict verifier contract.

The binary mainline uses ``AlignmentVerifierBackend``.  This interface is
retained only for archived verdict/reason backends and compatibility adapters.
"""

from abc import ABC, abstractmethod

from ..contracts.verifier import VerificationRequest
from .verdicts import VerificationLookup


class VerifierBackend(ABC):
    """Return one archived ``VerificationLookup`` for a candidate claim."""

    @abstractmethod
    def verify(self, request: VerificationRequest) -> VerificationLookup:
        raise NotImplementedError


__all__ = ['VerifierBackend']
