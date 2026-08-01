"""Strict parser for one Qwen2.5-VL absolute-coordinate grounding box."""

from dataclasses import dataclass
import math
import re
from typing import Sequence, Tuple


GroundingBox = Tuple[float, float, float, float]
DEFAULT_BOUNDARY_TOLERANCE_PIXELS = 1.0
_NUMBER = r'-?(?:\d+(?:\.\d*)?|\.\d+)'
_JSON_BOX_PATTERN = re.compile(
    rf'["\']bbox_2d["\']\s*:\s*\[\s*'
    rf'({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*'
    rf'({_NUMBER})\s*,\s*({_NUMBER})\s*\]',
    re.IGNORECASE,
)
_SPECIAL_BOX_PATTERN = re.compile(
    rf'(?:<\|box_start\|>|<box>)?\s*\(\s*({_NUMBER})\s*,\s*'
    rf'({_NUMBER})\s*\)\s*,\s*\(\s*({_NUMBER})\s*,\s*'
    rf'({_NUMBER})\s*\)\s*(?:<\|box_end\|>|</box>)?',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedReferenceGroundingBox:
    """One parsed box plus an audit trail for bounded edge clipping."""

    box: GroundingBox
    raw_box: GroundingBox
    boundary_clipped: bool
    clipped_sides: Tuple[str, ...]
    boundary_tolerance_pixels: float


def parse_reference_grounding_box_details(
        text: str,
        image_size: Sequence[int],
        boundary_tolerance_pixels: float = DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
) -> ParsedReferenceGroundingBox:
    """Parse one box and clip only a bounded one-pixel edge excursion."""

    if isinstance(boundary_tolerance_pixels, bool):
        raise TypeError('boundary_tolerance_pixels must be numeric')
    tolerance = float(boundary_tolerance_pixels)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            'boundary_tolerance_pixels must be finite and non-negative'
        )
    if not isinstance(text, str) or not text.strip():
        raise ValueError('grounding response is empty')
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError('image_size must be (width, height)')
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')

    matches = list(_JSON_BOX_PATTERN.findall(text))
    if not matches:
        matches = list(_SPECIAL_BOX_PATTERN.findall(text))
    boxes = []
    for match in matches:
        box = tuple(float(value) for value in match)
        if box not in boxes:
            boxes.append(box)
    if len(boxes) != 1:
        raise ValueError(
            f'expected exactly one grounding bbox, found {len(boxes)}'
        )
    raw_box = boxes[0]
    if not all(math.isfinite(value) for value in raw_box):
        raise ValueError(f'grounding bbox is non-finite: {raw_box}')
    if not (
        -tolerance <= raw_box[0] < raw_box[2] <= width + tolerance
        and -tolerance <= raw_box[1] < raw_box[3] <= height + tolerance
    ):
        raise ValueError(
            f'grounding bbox {raw_box} is outside resized image '
            f'{width}x{height} beyond the {tolerance:g}-pixel tolerance'
        )

    clipped_box = (
        max(0.0, raw_box[0]),
        max(0.0, raw_box[1]),
        min(float(width), raw_box[2]),
        min(float(height), raw_box[3]),
    )
    if not (
        0 <= clipped_box[0] < clipped_box[2] <= width
        and 0 <= clipped_box[1] < clipped_box[3] <= height
    ):
        raise ValueError(
            f'grounding bbox {raw_box} is empty after boundary clipping'
        )
    side_values = (
        ('left', raw_box[0], clipped_box[0]),
        ('top', raw_box[1], clipped_box[1]),
        ('right', raw_box[2], clipped_box[2]),
        ('bottom', raw_box[3], clipped_box[3]),
    )
    clipped_sides = tuple(
        side for side, raw_value, clipped_value in side_values
        if raw_value != clipped_value
    )
    return ParsedReferenceGroundingBox(
        box=clipped_box,
        raw_box=raw_box,
        boundary_clipped=bool(clipped_sides),
        clipped_sides=clipped_sides,
        boundary_tolerance_pixels=tolerance,
    )


def parse_reference_grounding_box(
        text: str,
        image_size: Sequence[int],
        boundary_tolerance_pixels: float = DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
) -> GroundingBox:
    """Return the usable box while retaining a details API for audit logging."""

    return parse_reference_grounding_box_details(
        text,
        image_size,
        boundary_tolerance_pixels=boundary_tolerance_pixels,
    ).box
