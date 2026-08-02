"""Pure dual-threshold routing policy for binary alignment scores."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Literal, Optional

from .calibration import AlignmentScoreCalibrator
from ..contracts.alignment_verifier import (
    ALIGNMENT_SCORE_KINDS,
    AlignmentScoreKind,
    AlignmentVerifierOutput,
)


AlignmentDecisionBand = Literal[
    'accept',
    'reject',
    'uncertain',
    'verifier_failure',
]
AlignmentSystemAction = Literal[
    'accept_candidate',
    'call_grounder',
]


@dataclass(frozen=True)
class AlignmentRoutingDecision:
    """One auditable system decision derived from an alignment score."""

    band: AlignmentDecisionBand
    system_action: AlignmentSystemAction
    alignment_score: Optional[float]
    score_kind: Optional[AlignmentScoreKind]
    score_semantics: str
    reject_threshold: float
    accept_threshold: float
    routing_reason: str
    verifier_abstained: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def requires_grounder(self) -> bool:
        return self.system_action == 'call_grounder'

    @property
    def candidate_accepted(self) -> bool:
        return self.system_action == 'accept_candidate'


class AlignmentRoutingPolicy:
    """Map a binary alignment score into three bands and two actions.

    The first version deliberately sends both the explicit-reject and
    uncertain bands to the grounder.  Keeping those bands distinct in the
    decision and logs preserves the information required for a future strong
    checker or abstention policy.  Verifier failure is not uncertainty: it is
    recorded separately and fails open by accepting the original candidate.
    """

    def __init__(
            self,
            reject_threshold: float,
            accept_threshold: float,
            *,
            required_score_kind: AlignmentScoreKind = (
                'calibrated_probability'
            ),
            calibrator: Optional[AlignmentScoreCalibrator] = None):
        for name, value in (
            ('reject_threshold', reject_threshold),
            ('accept_threshold', accept_threshold),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f'{name} must be a finite number')

        reject = float(reject_threshold)
        accept = float(accept_threshold)
        if not 0.0 <= reject < accept <= 1.0:
            raise ValueError(
                'thresholds must satisfy '
                '0 <= reject_threshold < accept_threshold <= 1'
            )
        if required_score_kind not in ALIGNMENT_SCORE_KINDS:
            raise ValueError(
                'required_score_kind must be one of '
                f'{ALIGNMENT_SCORE_KINDS}'
            )
        if calibrator is not None:
            if not isinstance(calibrator, AlignmentScoreCalibrator):
                raise TypeError(
                    'calibrator must implement AlignmentScoreCalibrator'
                )
            if required_score_kind != 'calibrated_probability':
                raise ValueError(
                    'calibrator is only valid when required_score_kind is '
                    'calibrated_probability'
                )
        self.reject_threshold = reject
        self.accept_threshold = accept
        self.required_score_kind = required_score_kind
        self.calibrator = calibrator

    @classmethod
    def explicit_raw(
            cls,
            reject_threshold: float,
            accept_threshold: float,
            *,
            score_kind: AlignmentScoreKind) -> 'AlignmentRoutingPolicy':
        """Construct an auditable raw/proxy-threshold experiment policy.

        This path is intentionally explicit.  It permits DINO IoU, Qwen
        self-reported confidence, and hard oracle labels to reproduce baseline
        experiments without presenting those thresholds as calibrated
        probability thresholds.
        """

        if score_kind == 'calibrated_probability':
            raise ValueError(
                'explicit_raw requires a non-calibrated score kind'
            )
        return cls(
            reject_threshold,
            accept_threshold,
            required_score_kind=score_kind,
        )

    def _policy_output(
            self,
            output: AlignmentVerifierOutput) -> AlignmentVerifierOutput:
        if output.abstained:
            return output
        if output.score_kind == self.required_score_kind:
            return output
        if self.calibrator is not None:
            return self.calibrator.calibrate(output)
        raise ValueError(
            'alignment score kind mismatch: policy requires '
            f'{self.required_score_kind!r}, verifier produced '
            f'{output.score_kind!r}; configure an explicit raw policy or a '
            'fitted calibrator'
        )

    def decide(
            self,
            output: AlignmentVerifierOutput) -> AlignmentRoutingDecision:
        if not isinstance(output, AlignmentVerifierOutput):
            raise TypeError(
                'alignment routing policy requires AlignmentVerifierOutput'
            )

        policy_output = self._policy_output(output)
        score = policy_output.alignment_score
        if output.abstained:
            band: AlignmentDecisionBand = 'verifier_failure'
            action: AlignmentSystemAction = 'accept_candidate'
            reason = 'verifier_failure_fail_open'
        elif score >= self.accept_threshold:  # type: ignore[operator]
            band = 'accept'
            action = 'accept_candidate'
            reason = 'high_alignment_accept'
        elif score <= self.reject_threshold:  # type: ignore[operator]
            band = 'reject'
            action = 'call_grounder'
            reason = 'low_alignment_reject'
        else:
            band = 'uncertain'
            action = 'call_grounder'
            reason = 'uncertainty_grounder_fallback'

        calibration_record = policy_output.metadata.get(
            'alignment_calibration'
        )
        metadata: Dict[str, Any] = {
            'policy': 'binary_dual_threshold_grounder_v1',
            'score_policy_mode': (
                'formal_probability'
                if self.required_score_kind == 'calibrated_probability'
                else 'explicit_raw_threshold'
            ),
            'required_score_kind': self.required_score_kind,
            'input_score_kind': output.score_kind,
            'effective_score_kind': policy_output.score_kind,
            'input_alignment_score': output.alignment_score,
            'effective_alignment_score': policy_output.alignment_score,
            'calibration_applied': calibration_record is not None,
            'calibrator_id': (
                None
                if not isinstance(calibration_record, dict)
                else calibration_record.get('calibrator_id')
            ),
            'uncertain_action': 'call_grounder',
            'verifier_failure_action': 'accept_candidate',
        }
        return AlignmentRoutingDecision(
            band=band,
            system_action=action,
            alignment_score=score,
            score_kind=policy_output.score_kind,
            score_semantics=policy_output.score_semantics,
            reject_threshold=self.reject_threshold,
            accept_threshold=self.accept_threshold,
            routing_reason=reason,
            verifier_abstained=output.abstained,
            metadata=metadata,
        )


__all__ = [
    'AlignmentDecisionBand',
    'AlignmentRoutingDecision',
    'AlignmentRoutingPolicy',
    'AlignmentSystemAction',
]
