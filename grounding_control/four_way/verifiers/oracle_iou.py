"""GT-backed four-action verifier for archived selective-routing bounds."""

from utils.coordinate_intervention import box_iou

from ..contracts import ActionVerifierBackend, ActionVerifierOutput
from ...contracts.verifier import VerificationRequest
from ...oracle_targets import OracleTargetResolver


class OracleIoUVerifierBackend(ActionVerifierBackend):
    """Map a uniquely matched candidate into no-action/relocate space."""

    def __init__(
            self,
            resolver: OracleTargetResolver,
            iou_threshold: float):
        if resolver is None:
            raise ValueError('resolver is required')
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError('iou_threshold must be in [0, 1]')
        self.iou_threshold = float(iou_threshold)
        self.resolver = resolver

    def verify_action(
            self,
            request: VerificationRequest) -> ActionVerifierOutput:
        resolution = self.resolver.resolve(request)
        metadata = resolution.as_metadata()
        metadata.update({
            'match_status': (
                'matched_unique_explicit_target'
                if resolution.matched
                else 'unverifiable_accept'
            ),
            'match_reason': resolution.reason,
            'match_context': resolution.context,
            'candidate_iou_to_gt': None,
            'iou_threshold': self.iou_threshold,
            'probability_source': 'unavailable_oracle_hard_label',
        })
        if not resolution.matched or resolution.bbox is None:
            metadata.update({
                'accept_router_action': 'unverifiable_accept',
                'legacy_verdict': 'uncertain',
                'legacy_reason': 'ambiguous',
            })
            return ActionVerifierOutput.unknown(metadata=metadata)

        candidate_iou = box_iou(
            request.candidate_bbox,
            resolution.bbox,
        )
        metadata['candidate_iou_to_gt'] = candidate_iou
        should_relocate = candidate_iou < self.iou_threshold
        action = 'relocate' if should_relocate else 'no_action'
        metadata.update({
            'legacy_verdict': (
                'misaligned' if should_relocate else 'aligned'
            ),
            'legacy_reason': (
                'wrong_object' if should_relocate else 'none'
            ),
        })
        return ActionVerifierOutput(
            predicted_action=action,
            action_probabilities=None,
            confidence=1.0,
            metadata=metadata,
        )


__all__ = ['OracleIoUVerifierBackend']
