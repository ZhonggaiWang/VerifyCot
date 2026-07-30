"""Generate high-confidence aligned verifier examples."""

import random
from typing import Dict, Optional

from .common import base_candidate, clip_box, geometry


def generate_aligned(
        target: Dict[str, object],
        image_width: int,
        image_height: int,
        rng: random.Random,
        minimum_iou: float = 0.70,
        minimum_target_coverage: float = 0.85,
        max_trials: int = 100) -> Optional[Dict[str, object]]:
    """Jitter a GT box slightly while retaining a high-confidence alignment."""
    target_box = target['pixel_box_xyxy']
    x1, y1, x2, y2 = target_box
    width, height = x2 - x1, y2 - y1
    for _ in range(max_trials):
        # Independent edge jitter avoids making every positive an exact GT copy.
        jittered = clip_box(
            (
                x1 + rng.uniform(-0.08, 0.08) * width,
                y1 + rng.uniform(-0.08, 0.08) * height,
                x2 + rng.uniform(-0.08, 0.08) * width,
                y2 + rng.uniform(-0.08, 0.08) * height,
            ),
            image_width,
            image_height,
        )
        if jittered is None:
            continue
        overlap = geometry(jittered, target_box)
        if (
            overlap['iou'] >= minimum_iou
            and overlap['reference_coverage'] >= minimum_target_coverage
        ):
            return base_candidate(
                jittered, target, image_width, image_height,
                verdict='aligned', reason='aligned', construction='jittered_gt_box',
            )
    # Exact GT remains a valid deterministic fallback for border-touching objects.
    return base_candidate(
        target_box, target, image_width, image_height,
        verdict='aligned', reason='aligned', construction='exact_gt_box_fallback',
    )
