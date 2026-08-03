"""Shared Qwen image preparation for box-producing four-way routing.

This module contains only input geometry and rendering so the retained
grounding-geometry verifier depends on no archived classifier implementation.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ....models.qwen25_vl.preprocessing import qwen_smart_resize_size


PixelBox = Tuple[float, float, float, float]
IntegerPixelBox = Tuple[int, int, int, int]
COORDINATE_ROUNDING_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GroundingActionInput:
    """Clean image, object reference, and original-image pixel candidate box."""

    image: Image.Image
    object_reference: str
    candidate_bbox_pixel_xyxy: PixelBox
    sample_id: str = ''


@dataclass(frozen=True)
class PreparedGroundingActionImage:
    clean_image: Image.Image
    marked_image: Image.Image
    original_size: Tuple[int, int]
    model_size: Tuple[int, int]
    candidate_bbox_model_xyxy: IntegerPixelBox


def _validate_original_pixel_box(
        values: Sequence[float],
        image_size: Tuple[int, int],
) -> PixelBox:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('candidate pixel bbox must have four coordinates')
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f'candidate pixel bbox is non-finite: {box}')
    width, height = image_size
    if not (
        0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height
    ):
        raise ValueError(
            f'candidate pixel bbox {box} is outside {width}x{height}'
        )
    return box  # type: ignore[return-value]


def prepare_grounding_action_image(
        image: Image.Image,
        candidate_bbox_pixel_xyxy: Sequence[float],
        min_pixels: int,
        max_pixels: Optional[int],
        box_color: Tuple[int, int, int] = (255, 0, 0),
        line_width: int = 4,
) -> PreparedGroundingActionImage:
    """Resize a clean source image and candidate box into one Qwen frame."""

    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    if not isinstance(line_width, int) or isinstance(line_width, bool):
        raise TypeError('line_width must be an integer')
    if line_width <= 0:
        raise ValueError('line_width must be positive')
    source = image.convert('RGB')
    original_size = source.size
    box = _validate_original_pixel_box(
        candidate_bbox_pixel_xyxy,
        original_size,
    )
    model_size = qwen_smart_resize_size(
        original_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    clean = source.resize(model_size, Image.Resampling.BICUBIC)
    scale_x = model_size[0] / original_size[0]
    scale_y = model_size[1] / original_size[1]
    x_min = max(0, min(
        model_size[0] - 1,
        int(math.floor(box[0] * scale_x)),
    ))
    y_min = max(0, min(
        model_size[1] - 1,
        int(math.floor(box[1] * scale_y)),
    ))
    x_max = max(x_min + 1, min(
        model_size[0],
        int(math.ceil(
            box[2] * scale_x - COORDINATE_ROUNDING_TOLERANCE
        )),
    ))
    y_max = max(y_min + 1, min(
        model_size[1],
        int(math.ceil(
            box[3] * scale_y - COORDINATE_ROUNDING_TOLERANCE
        )),
    ))
    model_box = (x_min, y_min, x_max, y_max)
    marked = clean.copy()
    ImageDraw.Draw(marked).rectangle(
        (x_min, y_min, x_max - 1, y_max - 1),
        outline=box_color,
        width=line_width,
    )
    return PreparedGroundingActionImage(
        clean_image=clean,
        marked_image=marked,
        original_size=original_size,
        model_size=model_size,
        candidate_bbox_model_xyxy=model_box,
    )


__all__ = [
    'GroundingActionInput',
    'PreparedGroundingActionImage',
    'prepare_grounding_action_image',
    'qwen_smart_resize_size',
]
