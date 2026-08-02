"""Shared coordinate types and validation at the VoCoT boundary."""

from typing import Sequence, Tuple


Box = Tuple[float, float, float, float]


def validate_normalized_box(values: Sequence[float]) -> Box:
    """Validate an ``xyxy`` box in the normalized padded-image unit square."""

    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('bbox must be a four-element list or tuple')
    box = tuple(float(value) for value in values)
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f'invalid normalized bbox: {box}')
    return box  # type: ignore[return-value]


__all__ = ['Box', 'validate_normalized_box']
