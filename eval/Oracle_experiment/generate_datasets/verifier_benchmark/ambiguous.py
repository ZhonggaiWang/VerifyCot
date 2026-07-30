"""Generate one candidate region that contains two different objects."""

import random
from typing import Dict, List, Optional

from .common import (
    base_candidate,
    box_area,
    box_iou,
    canonical_name,
    center_distance,
    geometry,
    union_box,
)


def generate_ambiguous(
        target: Dict[str, object],
        objects: List[Dict[str, object]],
        image_width: int,
        image_height: int,
        rng: random.Random,
        maximum_union_area_fraction: float = 0.60,
        maximum_union_to_object_area_ratio: float = 8.0,
        maximum_pair_iou: float = 0.20,
        minimum_union_expansion: float = 1.05) -> Optional[Dict[str, object]]:
    """Enclose the target and one nearby, differently named object in one box.

    This follows the benchmark definition requested for V1: ambiguity means a
    single highlighted region contains the referenced target and a different
    object, rather than requiring two instances with the same class name.
    """
    target_box = target['pixel_box_xyxy']
    target_name = canonical_name(target['name'])
    image_area = float(image_width * image_height)
    candidates = []
    for distractor in objects:
        if (
            distractor['object_id'] == target['object_id']
            or distractor['canonical_name'] == target_name
        ):
            continue
        distractor_box = distractor['pixel_box_xyxy']
        if box_iou(target_box, distractor_box) > maximum_pair_iou:
            continue
        combined = union_box(target_box, distractor_box)
        combined_area = box_area(combined)
        occupied_area = box_area(target_box) + box_area(distractor_box)
        if combined_area / image_area > maximum_union_area_fraction:
            continue
        if occupied_area <= 0 or combined_area / occupied_area > maximum_union_to_object_area_ratio:
            continue
        if combined_area < minimum_union_expansion * max(
            box_area(target_box), box_area(distractor_box)
        ):
            # Reject part/whole annotations whose union is effectively just one
            # object's original box.
            continue
        candidates.append((
            center_distance(target_box, distractor_box, image_width, image_height),
            rng.random(),
            distractor,
            combined,
        ))
    if not candidates:
        return None
    # Prefer a nearby second object, but randomize ties and near-ties.
    candidates.sort(key=lambda item: (item[0], item[1]))
    shortlist = candidates[:min(5, len(candidates))]
    _, _, distractor, combined = rng.choice(shortlist)
    result = base_candidate(
        combined, target, image_width, image_height,
        verdict='misaligned', reason='ambiguous',
        construction='union_of_target_and_different_object',
    )
    result.update({
        'distractor_object_id': distractor['object_id'],
        'distractor_object_name': distractor['name'],
        'distractor_box_pixel_xyxy': list(distractor['pixel_box_xyxy']),
        'distractor_box_normalized_xyxy': list(distractor['normalized_box_xyxy']),
        'candidate_distractor_geometry': geometry(
            combined, distractor['pixel_box_xyxy']
        ),
        'union_area_fraction': box_area(combined) / image_area,
    })
    return result
