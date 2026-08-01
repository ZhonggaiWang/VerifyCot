"""Contract for geometry-preserving box correction experts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from ..types import Box
from .action_verifier import ActionVerifierOutput
from .verifier import VerificationRequest


RefinementMode = Literal['expand', 'tighten']


@dataclass(frozen=True)
class RefinementRequest:
    """A candidate box plus the verifier-requested geometric correction."""

    verification_request: VerificationRequest
    verification: ActionVerifierOutput
    mode: RefinementMode


@dataclass(frozen=True)
class RefinementResult:
    """A replacement region returned by a box-refinement expert."""

    bbox: Box
    source: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BoxRefinerBackend(ABC):
    """Expand or tighten a candidate without changing its object identity."""

    @abstractmethod
    def refine(self, request: RefinementRequest) -> RefinementResult:
        raise NotImplementedError
