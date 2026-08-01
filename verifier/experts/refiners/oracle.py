"""Perfect GT box-refinement expert for oracle upper-bound experiments."""

from utils.coordinate_intervention import box_iou

from ...contracts import (
    BoxRefinerBackend,
    RefinementRequest,
    RefinementResult,
)
from ...expert_router import ExpertUnavailableError
from ...oracle_targets import OracleTargetResolver


class OracleBoxRefinerBackend(BoxRefinerBackend):
    """Return the uniquely matched full GT box for expand or tighten."""

    def __init__(self, resolver: OracleTargetResolver):
        self.resolver = resolver

    def refine(self, request: RefinementRequest) -> RefinementResult:
        resolution = self.resolver.resolve(
            request.verification_request
        )
        if not resolution.matched or resolution.bbox is None:
            raise ExpertUnavailableError(
                'oracle box refiner could not resolve a GT target: '
                f'{resolution.reason}'
            )
        metadata = resolution.as_metadata()
        metadata.update({
            'candidate_iou_to_gt': box_iou(
                request.verification_request.candidate_bbox,
                resolution.bbox,
            ),
            'router_action': (
                f'routed_to_oracle_box_refiner_{request.mode}'
            ),
            'evaluation_reference_box': list(resolution.bbox),
            'oracle_refinement_mode': request.mode,
        })
        return RefinementResult(
            bbox=resolution.bbox,
            source=f'oracle_gt_box_refiner_{request.mode}',
            confidence=1.0,
            metadata=metadata,
        )


__all__ = ['OracleBoxRefinerBackend']
