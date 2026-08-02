"""Contract for a relocation expert."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from .boxes import Box
from .requests import GroundingRequest


@dataclass(frozen=True)
class GroundingResult:
    """A replacement region returned by a relocation expert."""

    bbox: Box
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GrounderBackend(ABC):
    """Locate an object reference independently of verifier decisions."""

    @abstractmethod
    def ground(self, request: GroundingRequest) -> GroundingResult:
        """Locate ``request.object_reference`` independently of a verifier."""
        raise NotImplementedError
