"""Explicit conversion from backend-specific scores to ``P(aligned)``.

Calibration is intentionally a separate boundary from both the verifier and
the routing policy.  A backend may expose an auditable raw score (for example,
candidate--DINO IoU), while a fitted calibrator turns that score into the one
scale accepted by the formal probability policy.
"""

from abc import ABC, abstractmethod
import math
from typing import Any, Mapping

from ..contracts.alignment_verifier import (
    ALIGNMENT_SCORE_KINDS,
    AlignmentScoreKind,
    AlignmentVerifierOutput,
)


class AlignmentScoreCalibrationError(ValueError):
    """A score cannot be converted by the configured calibrator."""


class AlignmentScoreCalibrator(ABC):
    """Convert one declared raw score kind into calibrated ``P(aligned)``.

    Implementations may use ``metadata`` for a pre-fitted conditional
    calibration model, but must not inspect labels or mutate model state during
    evaluation.  The public :meth:`calibrate` wrapper validates both ends of
    the conversion and preserves the original evidence in metadata.
    """

    def __init__(
            self,
            source_score_kind: AlignmentScoreKind,
            calibrator_id: str):
        if source_score_kind not in ALIGNMENT_SCORE_KINDS:
            raise ValueError(
                'source_score_kind must be one of '
                f'{ALIGNMENT_SCORE_KINDS}'
            )
        if source_score_kind == 'calibrated_probability':
            raise ValueError(
                'a calibrator source must be an uncalibrated or proxy score'
            )
        if not isinstance(calibrator_id, str) or not calibrator_id.strip():
            raise ValueError('calibrator_id must be a non-empty string')
        self.source_score_kind = source_score_kind
        self.calibrator_id = calibrator_id.strip()

    @abstractmethod
    def calibrate_score(
            self,
            score: float,
            metadata: Mapping[str, Any]) -> float:
        """Return a fitted estimate of ``P(aligned)`` in ``[0, 1]``."""

        raise NotImplementedError

    def calibrate(
            self,
            output: AlignmentVerifierOutput) -> AlignmentVerifierOutput:
        """Validate and convert one available verifier output."""

        if not isinstance(output, AlignmentVerifierOutput):
            raise TypeError('calibrator requires AlignmentVerifierOutput')
        if output.abstained or output.alignment_score is None:
            raise AlignmentScoreCalibrationError(
                'an unavailable verifier output cannot be calibrated'
            )
        if output.score_kind != self.source_score_kind:
            raise AlignmentScoreCalibrationError(
                'calibrator source kind mismatch: expected '
                f'{self.source_score_kind!r}, got {output.score_kind!r}'
            )

        calibrated = self.calibrate_score(
            float(output.alignment_score),
            output.metadata,
        )
        if (
                not isinstance(calibrated, (int, float))
                or isinstance(calibrated, bool)
                or not math.isfinite(float(calibrated))
                or not 0.0 <= float(calibrated) <= 1.0
        ):
            raise AlignmentScoreCalibrationError(
                'calibrator must return one finite probability in [0, 1]'
            )

        source_record = {
            'calibrator_id': self.calibrator_id,
            'source_score_kind': output.score_kind,
            'source_alignment_score': float(output.alignment_score),
            'source_score_semantics': output.score_semantics,
        }
        return AlignmentVerifierOutput(
            alignment_score=float(calibrated),
            score_kind='calibrated_probability',
            score_semantics=(
                f'calibrated_alignment_probability:{self.calibrator_id}'
            ),
            metadata={
                **dict(output.metadata),
                'alignment_calibration': source_record,
                'alignment_score_calibrated': True,
            },
        )


__all__ = [
    'AlignmentScoreCalibrationError',
    'AlignmentScoreCalibrator',
]
