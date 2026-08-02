"""Candidate-region rendering in VoCoT's center-padded coordinate system."""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from PIL import Image, ImageDraw

from ...coordinates import (
    COORDINATE_SYSTEM,
    VOCOT_PADDING_COLOR,
    center_pad_image,
    normalized_square_box_to_pixel_box,
    original_pixel_box_to_normalized_square_box,
)


RGBColor = Tuple[int, int, int]
PixelBox = Tuple[int, int, int, int]

DEFAULT_BOX_COLOR: RGBColor = (255, 0, 0)
DEFAULT_QWEN_CROP_MIN_SIDE = 56


@dataclass(frozen=True)
class RenderedCandidate:
    """Images and geometry presented to the visual verifier."""

    original_image: Image.Image
    annotated_image: Image.Image
    crop_image: Image.Image
    original_size: Tuple[int, int]
    padded_size: int
    padding_offset: Tuple[int, int]
    pixel_bbox_xyxy: PixelBox


def resize_crop_for_qwen(
        crop: Image.Image,
        minimum_side: int = DEFAULT_QWEN_CROP_MIN_SIDE,
) -> Image.Image:
    """Upscale a tiny candidate crop without changing its aspect ratio.

    Qwen2.5-VL's merged visual patch factor is 28 and its processor rejects an
    image when either side is not larger than that factor.  A 56px minimum
    clears the constraint while retaining the original crop field of view.
    Crops already large enough are copied without resampling.
    """

    if not isinstance(crop, Image.Image):
        raise TypeError('crop must be a PIL.Image.Image')
    if (
        not isinstance(minimum_side, int)
        or isinstance(minimum_side, bool)
        or minimum_side <= 28
    ):
        raise ValueError('minimum_side must be an integer greater than 28')
    source = crop.convert('RGB')
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError(f'crop has invalid size: {source.size}')
    shortest = min(width, height)
    if shortest >= minimum_side:
        return source.copy()
    scale = minimum_side / float(shortest)
    resized = (
        max(minimum_side, int(math.ceil(width * scale))),
        max(minimum_side, int(math.ceil(height * scale))),
    )
    return source.resize(resized, Image.Resampling.BICUBIC)


def render_candidate_box(
        image: Image.Image,
        candidate_bbox: Sequence[float],
        box_color: RGBColor = DEFAULT_BOX_COLOR,
        padding_color: RGBColor = VOCOT_PADDING_COLOR,
        line_width: int = 4,
) -> RenderedCandidate:
    """Draw a candidate already expressed on the padded-square canvas.

    ``candidate_bbox`` must not be converted with
    ``normalized_box_to_square_padding`` here.  Generated VoCoT coordinates
    already use the padded-square coordinate system.
    """

    if not isinstance(line_width, int) or isinstance(line_width, bool):
        raise TypeError('line_width must be an integer')
    if line_width <= 0:
        raise ValueError('line_width must be positive')

    original = image.convert('RGB').copy()
    padded, offset = center_pad_image(original, padding_color)
    pixel_box = normalized_square_box_to_pixel_box(
        candidate_bbox,
        padded.size[0],
    )
    left, top, right, bottom = pixel_box
    # Crop before drawing so the close-up contains only candidate evidence,
    # not red border pixels that can dominate very small regions.
    crop = padded.crop((left, top, right + 1, bottom + 1))
    annotated = padded.copy()
    ImageDraw.Draw(annotated).rectangle(
        pixel_box,
        outline=box_color,
        width=line_width,
    )
    return RenderedCandidate(
        original_image=original,
        annotated_image=annotated,
        crop_image=crop,
        original_size=original.size,
        padded_size=padded.size[0],
        padding_offset=offset,
        pixel_bbox_xyxy=pixel_box,
    )
