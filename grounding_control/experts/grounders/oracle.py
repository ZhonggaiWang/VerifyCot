"""Perfect GT relocation expert for oracle upper-bound experiments."""

from ...contracts import (
    GrounderBackend,
    GroundingRequest,
    GroundingResult,
    validate_normalized_box,
)
from ...contracts.errors import ExpertUnavailableError
from ...oracle_targets import OracleTargetResolver


class OracleGrounderBackend(GrounderBackend):
    """Replace a rejected region with the uniquely matched full GT box."""

    def __init__(self, resolver: OracleTargetResolver):
        if resolver is None:
            raise ValueError('resolver is required')
        self.resolver = resolver

    def ground(self, request: GroundingRequest) -> GroundingResult:
        if not isinstance(request, GroundingRequest):
            raise TypeError('Oracle Grounder requires a GroundingRequest')
        # Resolve from the request every time.  In particular, never trust an
        # ``oracle_target_box`` supplied by another component: a learned/DINO
        # verifier must not become an implicit source of GT expert identity.
        resolution = self.resolver.resolve(request)
        if not resolution.matched or resolution.bbox is None:
            raise ExpertUnavailableError(
                'oracle target could not be resolved: '
                f'{resolution.reason}'
            )
        validated_box = validate_normalized_box(resolution.bbox)
        metadata = resolution.as_metadata()
        return GroundingResult(
            bbox=validated_box,
            source='oracle_gt_box',
            confidence=1.0,
            metadata={
                **metadata,
                'router_action': 'routed_to_oracle_grounder',
                'evaluation_reference_box': list(validated_box),
            },
        )


__all__ = ['OracleGrounderBackend']
