"""Generate background candidates unsupported by annotated foreground objects."""

import random
from typing import Dict, List, Optional

from .common import base_candidate, box_area, box_iou, coverage


def generate_unsupported(
        target: Dict[str, object],
        objects: List[Dict[str, object]],
        image_width: int,
        image_height: int,
        rng: random.Random,
        maximum_object_iou: float = 0.02,
        maximum_object_coverage: float = 0.05,
        maximum_occupancy_area_fraction: float = 0.50,
        minimum_side: float = 16.0,
        max_trials: int = 500) -> Optional[Dict[str, object]]:
    """Sample a box that avoids all non-global annotated scene objects.

    GQA scene graphs contain global regions such as ``air`` or ``ground`` that
    can cover almost the full image.  Objects occupying more than
    ``maximum_occupancy_area_fraction`` are ignored only for this background
    occupancy test and remain present in the source scene graph.
    """
    occupancy_objects = [
        item for item in objects
        if item['area_fraction'] <= maximum_occupancy_area_fraction
    ]
    target_box = target['pixel_box_xyxy']
    target_width = target_box[2] - target_box[0]
    target_height = target_box[3] - target_box[1]
    for _ in range(max_trials):
        scale = rng.uniform(0.75, 1.25)
        candidate_width = min(
            image_width * 0.30,
            max(minimum_side, target_width * scale),
        )
        candidate_height = min(
            image_height * 0.30,
            max(minimum_side, target_height * scale),
        )
        if candidate_width >= image_width or candidate_height >= image_height:
            continue
        x1 = rng.uniform(0.0, image_width - candidate_width)
        y1 = rng.uniform(0.0, image_height - candidate_height)
        candidate = (x1, y1, x1 + candidate_width, y1 + candidate_height)
        if any(
            box_iou(candidate, item['pixel_box_xyxy']) > maximum_object_iou
            or coverage(candidate, item['pixel_box_xyxy']) > maximum_object_coverage
            or coverage(item['pixel_box_xyxy'], candidate) > maximum_object_coverage
            for item in occupancy_objects
        ):
            continue
        result = base_candidate(
            candidate, target, image_width, image_height,
            verdict='misaligned', reason='unsupported',
            construction='random_unannotated_background_box',
        )
        result.update({
            'occupancy_object_count': len(occupancy_objects),
            'candidate_area_fraction': box_area(candidate) / float(image_width * image_height),
            'unsupported_annotation_caveat': (
                'Automatic label assumes retained GQA scene-graph foreground boxes are exhaustive.'
            ),
        })
        return result
    return None
