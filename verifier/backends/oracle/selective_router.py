"""GT-backed verifier and grounder for the selective-router upper bound."""

from typing import Any, Dict, Sequence

from utils.coordinate_intervention import (
    OnlineOracleCoordinateLogitsProcessor,
    box_iou,
)

from ...backend import (
    GrounderBackend,
    GroundingResult,
    VerificationRequest,
    VerifierBackend,
    validate_normalized_box,
)
from ...types import VerificationLookup, VerificationResult


class OracleIoUVerifierBackend(VerifierBackend):
    """Verify uniquely matched candidates by comparing them with a GT box."""

    def __init__(
            self,
            tokenizer,
            oracle_targets: Sequence[Dict[str, Any]],
            iou_threshold: float,
            context_window_tokens: int = 48):
        if not oracle_targets:
            raise ValueError('oracle_targets must contain at least one target')
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError('iou_threshold must be in [0, 1]')
        if int(context_window_tokens) <= 0:
            raise ValueError('context_window_tokens must be positive')
        self.iou_threshold = float(iou_threshold)
        self.matcher = OnlineOracleCoordinateLogitsProcessor(
            tokenizer,
            prompt_length=0,
            oracle_targets=oracle_targets,
            context_window_tokens=int(context_window_tokens),
        )

    def verify(self, request: VerificationRequest) -> VerificationLookup:
        match = self.matcher._new_decision(
            request.generated_ids,
            request.candidate_span[0],
            request.grounding_step,
        )
        matched = match['decision'] == 'forced_gt_box'
        metadata = {
            'match_status': (
                'matched_unique_explicit_target' if matched else 'unverifiable_accept'
            ),
            'match_reason': match['reason'],
            'match_context': match['context'],
            'target_object': match['target_object'],
            'matched_alias': match['matched_alias'],
            'oracle_target_box': match['oracle_box'],
            'candidate_iou_to_gt': None,
            'iou_threshold': self.iou_threshold,
        }
        if not matched:
            return VerificationLookup(
                result=VerificationResult.uncertain(),
                metadata=metadata,
            )

        oracle_box = validate_normalized_box(match['oracle_box'])
        candidate_iou = box_iou(request.candidate_bbox, oracle_box)
        metadata['candidate_iou_to_gt'] = candidate_iou
        if candidate_iou < self.iou_threshold:
            result = VerificationResult(
                verdict='misaligned',
                reason='wrong_object',
                confidence=1.0,
            )
        else:
            result = VerificationResult(
                verdict='aligned',
                reason='none',
                confidence=1.0,
            )
        return VerificationLookup(result=result, metadata=metadata)


class OracleGrounderBackend(GrounderBackend):
    """Return the GT region selected by :class:`OracleIoUVerifierBackend`."""

    def ground(
            self,
            request: VerificationRequest,
            verification: VerificationLookup) -> GroundingResult:
        oracle_box = verification.metadata.get('oracle_target_box')
        if oracle_box is None:
            raise RuntimeError(
                'oracle grounder was called without a uniquely matched oracle target'
            )
        return GroundingResult(
            bbox=validate_normalized_box(oracle_box),
            source='oracle_gt_box',
            confidence=1.0,
            metadata={
                'target_object': verification.metadata.get('target_object'),
                'matched_alias': verification.metadata.get('matched_alias'),
                'router_action': 'routed_to_oracle_grounder',
                'evaluation_reference_box': list(
                    validate_normalized_box(oracle_box)
                ),
            },
        )
