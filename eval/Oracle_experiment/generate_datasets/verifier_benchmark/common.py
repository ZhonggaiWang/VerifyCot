"""Shared geometry and scene-graph utilities for verifier benchmark builders."""

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


def canonical_name(name: object) -> str:
    """Conservatively normalize a GQA object name for identity comparisons."""
    if not isinstance(name, str):
        return ''
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', name.lower())).strip()


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = intersection_area(first, second)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def coverage(candidate: Sequence[float], reference: Sequence[float]) -> float:
    """Return the fraction of ``reference`` covered by ``candidate``."""
    reference_area = box_area(reference)
    return intersection_area(candidate, reference) / reference_area if reference_area > 0 else 0.0


def geometry(candidate: Sequence[float], reference: Sequence[float]) -> Dict[str, float]:
    """Return all asymmetric overlap measures used to audit an example."""
    return {
        'iou': box_iou(candidate, reference),
        'reference_coverage': coverage(candidate, reference),
        'candidate_purity': coverage(reference, candidate),
        'candidate_area': box_area(candidate),
        'reference_area': box_area(reference),
    }


def union_box(first: Sequence[float], second: Sequence[float]) -> Box:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


def clip_box(box: Sequence[float], width: float, height: float) -> Optional[Box]:
    x1, y1, x2, y2 = (float(value) for value in box)
    clipped = (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def pixel_to_normalized(box: Sequence[float], width: float, height: float) -> List[float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def valid_scene_object(
        object_id: str,
        record: Dict[str, object],
        image_width: int,
        image_height: int,
        min_side: float = 2.0) -> Optional[Dict[str, object]]:
    """Convert one official GQA scene-graph XYWH object to a validated XYXY record."""
    try:
        x = float(record['x'])
        y = float(record['y'])
        width = float(record['w'])
        height = float(record['h'])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width < min_side or height < min_side:
        return None
    box = clip_box((x, y, x + width, y + height), image_width, image_height)
    if box is None:
        return None
    name = canonical_name(record.get('name'))
    if not name:
        return None
    return {
        'object_id': str(object_id),
        'name': str(record.get('name')).strip(),
        'canonical_name': name,
        'pixel_box_xyxy': list(box),
        'normalized_box_xyxy': pixel_to_normalized(box, image_width, image_height),
        'area_fraction': box_area(box) / float(image_width * image_height),
    }


def scene_objects(
        scene_graph: Dict[str, object],
        min_side: float = 2.0) -> List[Dict[str, object]]:
    width = int(scene_graph['width'])
    height = int(scene_graph['height'])
    return [
        converted
        for object_id, record in scene_graph.get('objects', {}).items()
        for converted in [valid_scene_object(object_id, record, width, height, min_side)]
        if converted is not None
    ]


def find_object(objects: Iterable[Dict[str, object]], object_id: str) -> Optional[Dict[str, object]]:
    object_id = str(object_id)
    return next((item for item in objects if item['object_id'] == object_id), None)


def center_distance(first: Sequence[float], second: Sequence[float], width: int, height: int) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    dx = ((ax1 + ax2) - (bx1 + bx2)) / (2.0 * width)
    dy = ((ay1 + ay2) - (by1 + by2)) / (2.0 * height)
    return math.sqrt(dx * dx + dy * dy)


def base_candidate(
        candidate_box: Sequence[float],
        target: Dict[str, object],
        image_width: int,
        image_height: int,
        verdict: str,
        reason: str,
        construction: str) -> Dict[str, object]:
    """Create the common, fully auditable portion of one generated candidate."""
    candidate_box = [float(value) for value in candidate_box]
    target_box = target['pixel_box_xyxy']
    return {
        'object_reference': target['name'],
        'target_object_id': target['object_id'],
        'target_object_name': target['name'],
        'target_box_pixel_xyxy': list(target_box),
        'target_box_normalized_xyxy': pixel_to_normalized(
            target_box, image_width, image_height
        ),
        'candidate_box_pixel_xyxy': candidate_box,
        'candidate_box_normalized_xyxy': pixel_to_normalized(
            candidate_box, image_width, image_height
        ),
        'verdict': verdict,
        'reason': reason,
        'construction': construction,
        'candidate_target_geometry': geometry(candidate_box, target_box),
    }
