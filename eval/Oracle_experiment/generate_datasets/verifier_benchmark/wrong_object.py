"""Generate candidates that tightly localize a different object."""

import random
from typing import Dict, List, Optional

from .common import base_candidate, box_iou, canonical_name, coverage, geometry


def generate_wrong_object(
        target: Dict[str, object],
        objects: List[Dict[str, object]],
        image_width: int,
        image_height: int,
        rng: random.Random,
        maximum_target_iou: float = 0.05,
        maximum_target_coverage: float = 0.10,
        maximum_candidate_purity: float = 0.10,
        maximum_object_area_fraction: float = 0.60) -> Optional[Dict[str, object]]:
    """Use a spatially separate, differently named object's GT box.

    Both asymmetric overlap directions are constrained.  This rejects nested
    part/whole pairs such as ``pizza`` and ``pepper``, which would otherwise be
    indistinguishable from partial coverage of the target.
    """
    target_box = target['pixel_box_xyxy']
    target_name = canonical_name(target['name'])
    distractors = [
        item for item in objects
        if item['object_id'] != target['object_id']
        and item['canonical_name'] != target_name
        and item['area_fraction'] <= maximum_object_area_fraction
        and box_iou(item['pixel_box_xyxy'], target_box) <= maximum_target_iou
        and coverage(item['pixel_box_xyxy'], target_box) <= maximum_target_coverage
        and coverage(target_box, item['pixel_box_xyxy']) <= maximum_candidate_purity
    ]
    if not distractors:
        return None
    distractor = rng.choice(distractors)
    candidate = base_candidate(
        distractor['pixel_box_xyxy'], target, image_width, image_height,
        verdict='misaligned', reason='wrong_object',
        construction='different_object_gt_box',
    )
    candidate.update({
        'distractor_object_id': distractor['object_id'],
        'distractor_object_name': distractor['name'],
        'distractor_box_pixel_xyxy': list(distractor['pixel_box_xyxy']),
        'distractor_box_normalized_xyxy': list(distractor['normalized_box_xyxy']),
        'candidate_distractor_geometry': geometry(
            candidate['candidate_box_pixel_xyxy'], distractor['pixel_box_xyxy']
        ),
    })
    return candidate
