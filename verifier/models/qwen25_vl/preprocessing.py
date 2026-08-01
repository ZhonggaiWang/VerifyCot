"""Qwen image preparation shared by box prediction adapters."""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from PIL import Image


QWEN_IMAGE_FACTOR = 28
QWEN_MAX_ASPECT_RATIO = 200
GROUNDING_ACTION_IMAGE_MODES = ('raw_image', 'bbox_image')


@dataclass(frozen=True)
class PreparedReferenceImage:
    image: Image.Image
    original_size: Tuple[int, int]
    model_size: Tuple[int, int]


def qwen_smart_resize_size(
        image_size: Sequence[int],
        min_pixels: int,
        max_pixels: int,
        factor: int = QWEN_IMAGE_FACTOR,
) -> Tuple[int, int]:
    """Reproduce Qwen2.5-VL smart-resize dimensions as ``(width, height)``."""

    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError('image_size must be (width, height)')
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError('invalid Qwen min/max pixel bounds')
    if factor <= 0:
        raise ValueError('resize factor must be positive')
    if max(width, height) / min(width, height) > QWEN_MAX_ASPECT_RATIO:
        raise ValueError(
            f'image aspect ratio must be at most {QWEN_MAX_ASPECT_RATIO}'
        )
    resized_width = max(factor, round(width / factor) * factor)
    resized_height = max(factor, round(height / factor) * factor)
    if resized_width * resized_height > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        resized_width = math.floor((width / beta) / factor) * factor
        resized_height = math.floor((height / beta) / factor) * factor
    elif resized_width * resized_height < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        resized_width = math.ceil((width * beta) / factor) * factor
        resized_height = math.ceil((height * beta) / factor) * factor
    if resized_width <= 0 or resized_height <= 0:
        raise ValueError('Qwen smart resize produced an empty image')
    return resized_width, resized_height


def prepare_reference_image(
        image: Image.Image,
        min_pixels: int,
        max_pixels: int,
) -> PreparedReferenceImage:
    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    source = image.convert('RGB')
    model_size = qwen_smart_resize_size(
        source.size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return PreparedReferenceImage(
        image=source.resize(model_size, Image.Resampling.BICUBIC),
        original_size=source.size,
        model_size=model_size,
    )
