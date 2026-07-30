"""JSON/JSONL-backed oracle verifier used by the first repair experiments."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from ...types import Box, VerificationLookup, VerificationResult


def _as_box(value: Sequence[float]) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('candidate_bbox must be a four-element list')
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        raise ValueError('candidate_bbox must contain finite values')
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'invalid normalized candidate_bbox: {box}')
    return box  # type: ignore[return-value]


class StoredOracleVerifier:
    """Look up fixed verification outcomes without invoking any VLM.

    Records are uniquely indexed by ``(sample_id, grounding_step,
    attempt_index)``.  A missing record deliberately yields ``uncertain``;
    it can never silently accept a candidate.
    """

    def __init__(self, oracle_file: str, strict: bool = True, box_tolerance: float = 1e-3):
        self.oracle_file = str(oracle_file)
        self.strict = strict
        self.box_tolerance = float(box_tolerance)
        if self.box_tolerance < 0:
            raise ValueError('box_tolerance must be non-negative')
        self._records: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self._load(Path(oracle_file))

    @staticmethod
    def _key(record: Dict[str, Any]) -> Tuple[str, int, int]:
        try:
            return (
                str(record['sample_id']),
                int(record['grounding_step']),
                int(record['attempt_index']),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('oracle record requires sample_id, grounding_step, attempt_index') from exc

    @staticmethod
    def _read_records(path: Path) -> Iterable[Dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f'oracle file not found: {path}')
        with path.open('r', encoding='utf-8') as handle:
            if path.suffix.lower() == '.jsonl':
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'invalid JSONL at {path}:{line_number}') from exc
                    if not isinstance(record, dict):
                        raise ValueError(f'oracle JSONL record {line_number} is not an object')
                    yield record
                return
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get('records', [payload])
        if not isinstance(payload, list):
            raise ValueError('oracle JSON must be an object, a records object, or a list')
        for record in payload:
            if not isinstance(record, dict):
                raise ValueError('oracle JSON record is not an object')
            yield record

    def _load(self, path: Path) -> None:
        for record in self._read_records(path):
            key = self._key(record)
            if key in self._records:
                raise ValueError(f'duplicate oracle key: {key}')
            # Validate at load time so an invalid file cannot turn into an ACCEPT.
            output = record.get('verifier_output')
            if not isinstance(output, dict):
                raise ValueError(f'oracle record {key} lacks verifier_output')
            VerificationResult(
                verdict=output.get('verdict'),
                reason=output.get('reason'),
                confidence=output.get('confidence'),
            )
            if 'candidate_bbox' in record:
                _as_box(record['candidate_bbox'])
            self._records[key] = record

    def verify(self, sample_id: str, grounding_step: int, attempt_index: int,
               candidate_bbox: Sequence[float]) -> VerificationLookup:
        candidate = _as_box(candidate_bbox)
        key = (str(sample_id), int(grounding_step), int(attempt_index))
        record = self._records.get(key)
        if record is None:
            return VerificationLookup(
                result=VerificationResult.uncertain(),
                missing_oracle_record=True,
                error=f'missing oracle record for {key}',
            )
        expected = record.get('candidate_bbox')
        if expected is not None:
            expected_box = _as_box(expected)
            if any(abs(actual - stored) > self.box_tolerance
                   for actual, stored in zip(candidate, expected_box)):
                return VerificationLookup(
                    result=VerificationResult.uncertain(),
                    oracle_candidate_mismatch=True,
                    error=(f'oracle candidate_bbox mismatch for {key}: '
                           f'generated={candidate}, stored={expected_box}'),
                )
        output = record['verifier_output']
        return VerificationLookup(
            result=VerificationResult(
                verdict=output['verdict'],
                reason=output['reason'],
                confidence=output['confidence'],
            )
        )
