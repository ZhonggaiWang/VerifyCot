"""VoCoT coordinate conversions shared by verifier and expert adapters."""

import math
from typing import Sequence, Tuple

from PIL import Image

from .contracts import validate_normalized_box


RGBColor = Tuple[int, int, int]
PixelBox = Tuple[int, int, int, int]
VOCOT_PADDING_COLOR: RGBColor = (122, 116, 104)
COORDINATE_SYSTEM = 'normalized_xyxy_on_center_padded_square'


def center_pad_image(
        image: Image.Image,
        background_color: RGBColor = VOCOT_PADDING_COLOR,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Reproduce ``VoCoT_InputProcessor.expand2square_fn`` exactly."""

    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    source = image.convert('RGB')
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError(f'image has invalid size: {source.size}')
    if width == height:
        return source.copy(), (0, 0)
    square_size = max(width, height)
    offset = (
        (0, (width - height) // 2)
        if width > height
        else ((height - width) // 2, 0)
    )
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
    limit = square_size - 1
    result = (
        min(max(int(math.floor(x_min * square_size)), 0), limit),
        min(max(int(math.floor(y_min * square_size)), 0), limit),
        min(max(int(math.ceil(x_max * square_size)) - 1, 0), limit),
        min(max(int(math.ceil(y_max * square_size)) - 1, 0), limit),
    )
    if result[0] > result[2] or result[1] > result[3]:
        raise ValueError(
            f'normalized bbox becomes empty at {square_size}px: '
            f'{normalized} -> {result}'
        )
    return result


def original_pixel_box_to_normalized_square_box(
        bbox: Sequence[float],
        image_width: int,
        image_height: int,
) -> Tuple[float, float, float, float]:
    """Map original-image pixel ``xyxy`` into VoCoT's padded unit square."""

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
    offset_x, offset_y = (
        (0, (image_width - image_height) // 2)
        if image_width > image_height
        else ((image_height - image_width) // 2, 0)
    )
    return validate_normalized_box((
        (x_min + offset_x) / square_size,
        (y_min + offset_y) / square_size,
        (x_max + offset_x) / square_size,
        (y_max + offset_y) / square_size,
    ))
