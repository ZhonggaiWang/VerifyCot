"""Leakage-safe adapter from GQA benchmark rows to Qwen verifier inputs."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Tuple

from PIL import Image

from ...verifier_backends.qwen25_vl import (
    CandidateVerificationInput,
    GroundingActionInput,
)
from ...coordinates import original_pixel_box_to_normalized_square_box
from ...verifier_backends import GeometryVerificationInput
from .labels import CONTROLLED_STATUSES


PixelBox = Tuple[float, float, float, float]


def expected_status_from_record(record: Mapping[str, Any]) -> str:
    """Map ``verdict/reason`` to the benchmark construction subtype."""

    verdict = str(record.get('verdict', '')).strip()
    reason = str(record.get('reason', '')).strip()
    status = 'aligned' if verdict == 'aligned' else reason
    if status not in CONTROLLED_STATUSES:
        raise ValueError(
            f'invalid benchmark label verdict={verdict!r}, reason={reason!r}'
        )
    return status


def _pixel_box(values: object) -> PixelBox:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('candidate_box_pixel_xyxy must have four elements')
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f'candidate pixel box contains non-finite values: {box}')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f'candidate pixel box is empty: {box}')
    return box  # type: ignore[return-value]


def _resolve_source_image(value: object, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('source_image must be a non-empty path string')
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


@dataclass(frozen=True)
class GQAControlledExample:
    """One benchmark example with labels kept outside model-visible input."""

    event_id: str
    sample_index: int
    split: str
    image_id: str
    source_image: Path
    object_reference: str
    candidate_box_pixel_xyxy: PixelBox
    expected_status: str
    expected_verdict: str
    expected_reason: str

    @classmethod
    def from_record(
            cls,
            record: Mapping[str, Any],
            manifest_path: Path,
    ) -> 'GQAControlledExample':
        split = str(record.get('split', '')).strip()
        if split not in ('dev', 'test'):
            raise ValueError(f'benchmark split must be dev or test, got {split!r}')
        reference = str(record.get('object_reference', '')).strip()
        if not reference:
            raise ValueError('object_reference must not be empty')
        event_id = str(record.get('event_id', '')).strip()
        if not event_id:
            raise ValueError('event_id must not be empty')
        return cls(
            event_id=event_id,
            sample_index=int(record['sample_index']),
            split=split,
            image_id=str(record['image_id']),
            source_image=_resolve_source_image(
                record.get('source_image'),
                manifest_path,
            ),
            object_reference=reference,
            candidate_box_pixel_xyxy=_pixel_box(
                record.get('candidate_box_pixel_xyxy')
            ),
            expected_status=expected_status_from_record(record),
            expected_verdict=str(record.get('verdict', '')).strip(),
            expected_reason=str(record.get('reason', '')).strip(),
        )

    def to_candidate_input(self) -> CandidateVerificationInput:
        """Load the source image and create the exact production Qwen input.

        Neither the expected label nor the target bounding box is represented
        in :class:`CandidateVerificationInput`.
        """

        if not self.source_image.is_file():
            raise FileNotFoundError(f'source image not found: {self.source_image}')
        with Image.open(self.source_image) as source:
            image = source.convert('RGB')
        padded_bbox = original_pixel_box_to_normalized_square_box(
            self.candidate_box_pixel_xyxy,
            image.width,
            image.height,
        )
        return CandidateVerificationInput(
            image=image,
            object_reference=self.object_reference,
            candidate_bbox=padded_bbox,
            sample_id=self.event_id,
        )

    def to_grounding_action_input(self) -> GroundingActionInput:
        """Load a clean image and preserve its original absolute bbox frame."""

        if not self.source_image.is_file():
            raise FileNotFoundError(f'source image not found: {self.source_image}')
        with Image.open(self.source_image) as source:
            image = source.convert('RGB')
        return GroundingActionInput(
            image=image,
            object_reference=self.object_reference,
            candidate_bbox_pixel_xyxy=self.candidate_box_pixel_xyxy,
            sample_id=self.event_id,
        )

    def to_geometry_verification_input(self) -> GeometryVerificationInput:
        """Create the model-independent input used by geometry verifiers."""

        if not self.source_image.is_file():
            raise FileNotFoundError(
                f'source image not found: {self.source_image}'
            )
        with Image.open(self.source_image) as source:
            image = source.convert('RGB')
        return GeometryVerificationInput(
            image=image,
            object_reference=self.object_reference,
            candidate_bbox_pixel_xyxy=self.candidate_box_pixel_xyxy,
            sample_id=self.event_id,
        )


def load_examples(
        manifest_path: Path,
        split: str,
) -> List[GQAControlledExample]:
    """Read and validate examples while preserving manifest order."""

    manifest_path = Path(manifest_path)
    if split not in ('dev', 'test', 'all'):
        raise ValueError(f'split must be dev, test, or all, got {split!r}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'benchmark manifest not found: {manifest_path}')

    examples: List[GQAControlledExample] = []
    seen = set()
    with manifest_path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                example = GQAControlledExample.from_record(
                    record,
                    manifest_path,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f'invalid benchmark row at {manifest_path}:{line_number}: '
                    f'{error}'
                ) from error
            if example.event_id in seen:
                raise ValueError(f'duplicate event_id: {example.event_id}')
            seen.add(example.event_id)
            if split == 'all' or example.split == split:
                examples.append(example)
    return examples
