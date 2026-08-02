"""GT-backed action verifier for selective-routing upper bounds."""

from utils.coordinate_intervention import box_iou

from ..contracts import (
    CandidateAlignmentRequest,
)
from ..contracts.alignment_verifier import (
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
)
from ..oracle_targets import OracleTargetResolver


ORACLE_ALIGNMENT_SCORE_SEMANTICS = 'oracle_hard_binary_alignment_label'


class OracleAlignmentVerifierBackend(AlignmentVerifierBackend):
    """Return a hard binary upper-bound score for uniquely resolved targets.

    ``gt_iou_threshold`` defines the benchmark label and is deliberately
    independent of the controller's acceptance/rejection thresholds.
    Unmatched or ambiguous references abstain because partial annotations do
    not justify either an aligned or a misaligned label.
    """

    def __init__(
            self,
            resolver: OracleTargetResolver,
            gt_iou_threshold: float = 0.5):
        if resolver is None:
            raise ValueError('resolver is required')
        if not 0.0 <= float(gt_iou_threshold) <= 1.0:
            raise ValueError('gt_iou_threshold must be in [0, 1]')
        self.resolver = resolver
        self.gt_iou_threshold = float(gt_iou_threshold)

    def verify_alignment(
            self,
            request: CandidateAlignmentRequest,
    ) -> AlignmentVerifierOutput:
        if not isinstance(request, CandidateAlignmentRequest):
            raise TypeError(
                'Oracle alignment verifier requires CandidateAlignmentRequest'
            )
        resolution = self.resolver.resolve(request)
        metadata = resolution.as_metadata()
        metadata.update({
            'alignment_backend': 'oracle_binary_iou_label',
            'match_reason': resolution.reason,
            'match_context': resolution.context,
            'candidate_iou_to_gt': None,
            'gt_iou_threshold': self.gt_iou_threshold,
            'alignment_score_calibrated': False,
        })
        if not resolution.matched or resolution.bbox is None:
            metadata['match_status'] = 'unverifiable_abstain'
            return AlignmentVerifierOutput.unknown(
                error='oracle_target_not_uniquely_resolved',
                score_kind='hard_oracle_label',
                score_semantics=ORACLE_ALIGNMENT_SCORE_SEMANTICS,
                metadata=metadata,
            )

        candidate_iou = box_iou(
            request.candidate_bbox,
            resolution.bbox,
        )
        aligned = candidate_iou >= self.gt_iou_threshold
        metadata.update({
            'match_status': 'matched_unique_explicit_target',
            'candidate_iou_to_gt': candidate_iou,
            'oracle_binary_aligned_label': aligned,
        })
        return AlignmentVerifierOutput(
            alignment_score=1.0 if aligned else 0.0,
            score_kind='hard_oracle_label',
            score_semantics=ORACLE_ALIGNMENT_SCORE_SEMANTICS,
            abstained=False,
            error=None,
            metadata=metadata,
        )


__all__ = [
    'ORACLE_ALIGNMENT_SCORE_SEMANTICS',
    'OracleAlignmentVerifierBackend',
]
