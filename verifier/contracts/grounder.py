"""Contract for a relocation expert."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from ..types import Box
from .action_verifier import ActionVerifierOutput
from .verifier import VerificationRequest


@dataclass(frozen=True)
class GroundingResult:
    """A replacement region returned by a relocation expert."""

    bbox: Box
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GrounderBackend(ABC):
    """Relocate the referenced object after a ``wrong_object`` rejection."""

    @abstractmethod
    def ground(
            self,
            request: VerificationRequest,
            verification: ActionVerifierOutput) -> GroundingResult:
        raise NotImplementedError
