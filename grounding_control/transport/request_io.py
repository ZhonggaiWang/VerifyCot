"""Validation helpers shared by JSONL model endpoints."""

import math
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from PIL import Image

from .jsonl_protocol import WorkerRequestError


def required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkerRequestError(f'{key} must be a non-empty string')
    return value.strip()


def load_image(path_value: Any) -> Image.Image:
    if not isinstance(path_value, str) or not path_value.strip():
        raise WorkerRequestError('image_path must be a non-empty string')
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise WorkerRequestError(f'image_path is not a file: {path}')
    try:
        with Image.open(path) as image:
            return image.convert('RGB').copy()
    except (OSError, ValueError) as error:
        raise WorkerRequestError(
            f'could not load image_path {path}: {error}'
        ) from error


def finite_pixel_box(
        values: Any,
        image_size: Sequence[int],
        field_name: str,
) -> List[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise WorkerRequestError(
            f'{field_name} must contain four coordinates'
        )
    try:
        box = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise WorkerRequestError(
            f'{field_name} coordinates must be numeric'
        ) from error
    if not all(math.isfinite(value) for value in box):
        raise WorkerRequestError(f'{field_name} coordinates must be finite')
    width, height = (int(value) for value in image_size)
    if not (
        0.0 <= box[0] < box[2] <= width
        and 0.0 <= box[1] < box[3] <= height
    ):
        raise WorkerRequestError(
            f'{field_name} {box} is outside image {width}x{height}'
        )
    return box


__all__ = [
    'finite_pixel_box',
    'load_image',
    'required_string',
]
