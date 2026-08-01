"""Perfect GT relocation expert for oracle upper-bound experiments."""

from utils.coordinate_intervention import box_iou

from ...contracts import (
    ActionVerifierOutput,
    GrounderBackend,
    GroundingResult,
    VerificationRequest,
    validate_normalized_box,
)
from ...expert_router import ExpertUnavailableError
from ...oracle_targets import OracleTargetResolver


class OracleGrounderBackend(GrounderBackend):
    """Replace a rejected region with the uniquely matched full GT box."""

    def __init__(self, resolver: OracleTargetResolver):
        if resolver is None:
            raise ValueError('resolver is required')
        self.resolver = resolver

    def ground(
            self,
            request: VerificationRequest,
            verification: ActionVerifierOutput) -> GroundingResult:
        # Resolve from the request every time.  In particular, never trust an
        # ``oracle_target_box`` supplied by verifier metadata: a learned/DINO
        # verifier must not become an implicit source of GT expert identity.
        resolution = self.resolver.resolve(request)
        if not resolution.matched or resolution.bbox is None:
            raise ExpertUnavailableError(
                'oracle target could not be resolved: '
                f'{resolution.reason}'
            )
        validated_box = validate_normalized_box(resolution.bbox)
        metadata = {
            **dict(verification.metadata),
            **resolution.as_metadata(),
            'candidate_iou_to_gt': box_iou(
                request.candidate_bbox,
                validated_box,
            ),
        }
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
