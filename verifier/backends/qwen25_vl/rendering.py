"""Candidate-region rendering in VoCoT's center-padded coordinate system."""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from PIL import Image, ImageDraw

from ...backend import validate_normalized_box


RGBColor = Tuple[int, int, int]
PixelBox = Tuple[int, int, int, int]

# This is the integer CLIP-mean background used by VoCoT_InputProcessor.
VOCOT_PADDING_COLOR: RGBColor = (122, 116, 104)
DEFAULT_BOX_COLOR: RGBColor = (255, 0, 0)
COORDINATE_SYSTEM = 'normalized_xyxy_on_center_padded_square'
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


def center_pad_image(
        image: Image.Image,
        background_color: RGBColor = VOCOT_PADDING_COLOR,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Reproduce ``VoCoT_InputProcessor.expand2square_fn`` exactly.

    The short-axis offset intentionally uses integer ``// 2``.  Odd padding
    therefore leaves the extra pixel on the bottom or right, matching the
    generator's input and REFbind coordinate convention.
    """

    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    source = image.convert('RGB')
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError(f'image has invalid size: {source.size}')
    if width == height:
        return source.copy(), (0, 0)

    square_size = max(width, height)
    if width > height:
        offset = (0, (width - height) // 2)
    else:
        offset = ((height - width) // 2, 0)
    canvas = Image.new('RGB', (square_size, square_size), background_color)
    canvas.paste(source, offset)
    return canvas, offset


def normalized_square_box_to_pixel_box(
        bbox: Sequence[float],
        square_size: int,
) -> PixelBox:
    """Convert normalized padded-square ``xyxy`` to an inclusive PIL box."""

    normalized = validate_normalized_box(bbox)
    if (
        not isinstance(square_size, int)
        or isinstance(square_size, bool)
        or square_size <= 0
    ):
        raise ValueError(f'square_size must be a positive integer: {square_size}')

    x_min, y_min, x_max, y_max = normalized
    left = int(math.floor(x_min * square_size))
    top = int(math.floor(y_min * square_size))
    right = int(math.ceil(x_max * square_size)) - 1
    bottom = int(math.ceil(y_max * square_size)) - 1
    limit = square_size - 1
    pixel_box = (
        min(max(left, 0), limit),
        min(max(top, 0), limit),
        min(max(right, 0), limit),
        min(max(bottom, 0), limit),
    )
    if pixel_box[0] > pixel_box[2] or pixel_box[1] > pixel_box[3]:
        raise ValueError(
            f'normalized bbox becomes empty at {square_size}px: '
            f'{normalized} -> {pixel_box}'
        )
    return pixel_box


def original_pixel_box_to_normalized_square_box(
        bbox: Sequence[float],
        image_width: int,
        image_height: int,
) -> Tuple[float, float, float, float]:
    """Map original-image pixel ``xyxy`` into VoCoT's padded unit square.

    Benchmark annotations are usually expressed on the unpadded source image,
    while generated VoCoT coordinates already live on the center-padded square.
    This helper is the single conversion boundary between those conventions.
    It uses the same integer short-axis offset as :func:`center_pad_image`.
    """

    if (
        not isinstance(image_width, int)
        or isinstance(image_width, bool)
        or image_width <= 0
        or not isinstance(image_height, int)
        or isinstance(image_height, bool)
        or image_height <= 0
    ):
        raise ValueError(
            'image_width and image_height must be positive integers, got '
            f'{image_width!r} x {image_height!r}'
        )
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError('pixel bbox must be a four-element list or tuple')
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'pixel bbox must contain finite values: {values}')
    x_min, y_min, x_max, y_max = values
    if not (
        0 <= x_min < x_max <= image_width
        and 0 <= y_min < y_max <= image_height
    ):
        raise ValueError(
            f'invalid pixel bbox {values} for image '
            f'{image_width}x{image_height}'
        )

    square_size = max(image_width, image_height)
    if image_width > image_height:
        offset_x, offset_y = 0, (image_width - image_height) // 2
    else:
        offset_x, offset_y = (image_height - image_width) // 2, 0
    normalized = (
        (x_min + offset_x) / square_size,
        (y_min + offset_y) / square_size,
        (x_max + offset_x) / square_size,
        (y_max + offset_y) / square_size,
    )
    return validate_normalized_box(normalized)


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
