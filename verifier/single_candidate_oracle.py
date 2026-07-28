"""In-memory oracle output for one controlled pre-commit candidate.

This backend is intentionally narrow: it exposes one already-known checker
decision without requiring a temporary JSONL file.  It is used by natural
baseline-error experiments after a strict GT audit has selected one candidate.
"""

import math
from typing import Sequence

from .types import Box, VerificationLookup, VerificationResult


def _as_box(value: Sequence[float]) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('candidate_bbox must be a four-element list')
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        raise ValueError('candidate_bbox must contain finite values')
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'invalid normalized candidate_bbox: {box}')
    return box  # type: ignore[return-value]


class SingleCandidateOracleVerifier:
    """Return one fixed verifier decision only for its exact candidate."""

    def __init__(
            self, sample_id: str, grounding_step: int, candidate_bbox: Sequence[float],
            result: VerificationResult, box_tolerance: float = 1e-3):
        if not sample_id:
            raise ValueError('sample_id is required')
        if int(grounding_step) <= 0:
            raise ValueError('grounding_step must be positive')
        if float(box_tolerance) < 0:
            raise ValueError('box_tolerance must be non-negative')
        self.sample_id = str(sample_id)
        self.grounding_step = int(grounding_step)
        self.candidate_bbox = _as_box(candidate_bbox)
        self.result = result
        self.box_tolerance = float(box_tolerance)

    def verify(self, sample_id: str, grounding_step: int, attempt_index: int,
               candidate_bbox: Sequence[float]) -> VerificationLookup:
        key = (str(sample_id), int(grounding_step), int(attempt_index))
        expected_key = (self.sample_id, self.grounding_step, 0)
        if key != expected_key:
            return VerificationLookup(
                result=VerificationResult.uncertain(),
                missing_oracle_record=True,
                error=f'missing in-memory oracle record for {key}; expected {expected_key}',
            )
        candidate = _as_box(candidate_bbox)
        if any(abs(actual - expected) > self.box_tolerance
               for actual, expected in zip(candidate, self.candidate_bbox)):
            return VerificationLookup(
                result=VerificationResult.uncertain(),
                oracle_candidate_mismatch=True,
                error=(
                    'in-memory oracle candidate_bbox mismatch: '
                    f'generated={candidate}, expected={self.candidate_bbox}'
                ),
            )
        return VerificationLookup(result=self.result)
