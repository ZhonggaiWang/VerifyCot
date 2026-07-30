"""Generate candidates that cover only a controlled fraction of the target."""

import random
from typing import Dict, Optional

from .common import base_candidate


def generate_partial_coverage(
        target: Dict[str, object],
        image_width: int,
        image_height: int,
        rng: random.Random,
        minimum_fraction: float = 0.25,
        maximum_fraction: float = 0.50,
        minimum_candidate_side: float = 12.0) -> Optional[Dict[str, object]]:
    """Crop one side or the centre of the target to 25--50% target area."""
    x1, y1, x2, y2 = (float(value) for value in target['pixel_box_xyxy'])
    width, height = x2 - x1, y2 - y1
    mode = rng.choice(('left', 'right', 'top', 'bottom', 'center'))
    fraction = rng.uniform(minimum_fraction, maximum_fraction)
    if mode in ('left', 'right'):
        candidate_width = width * fraction
        if candidate_width < minimum_candidate_side or height < minimum_candidate_side:
            return None
        candidate = (
            (x1, y1, x1 + candidate_width, y2)
            if mode == 'left' else
            (x2 - candidate_width, y1, x2, y2)
        )
    elif mode in ('top', 'bottom'):
        candidate_height = height * fraction
        if width < minimum_candidate_side or candidate_height < minimum_candidate_side:
            return None
        candidate = (
            (x1, y1, x2, y1 + candidate_height)
            if mode == 'top' else
            (x1, y2 - candidate_height, x2, y2)
        )
    else:
        # Equal scaling in both dimensions yields the requested area fraction.
        scale = fraction ** 0.5
        candidate_width, candidate_height = width * scale, height * scale
        if candidate_width < minimum_candidate_side or candidate_height < minimum_candidate_side:
            return None
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        candidate = (
            center_x - candidate_width / 2.0,
            center_y - candidate_height / 2.0,
            center_x + candidate_width / 2.0,
            center_y + candidate_height / 2.0,
        )
    result = base_candidate(
        candidate, target, image_width, image_height,
        verdict='misaligned', reason='partial_coverage',
        construction=f'target_{mode}_crop',
    )
    result['requested_target_area_fraction'] = fraction
    return result
