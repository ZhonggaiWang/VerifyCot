"""Model-independent routing from a predicted reference box and candidate box."""

from dataclasses import dataclass
from typing import Sequence, Tuple


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundingGeometryDecision:
    """One auditable four-action decision derived only from box geometry."""

    action: str
    reason: str
    intersection_area: float
    candidate_area: float
    grounding_area: float
    iou: float
    candidate_coverage_by_grounding: float
    grounding_coverage_by_candidate: float
    accept_iou_threshold: float
    containment_threshold: float


def _box(values: Sequence[float], name: str) -> Box:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f'{name} must contain four coordinates')
    box = tuple(float(value) for value in values)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f'{name} is empty: {box}')
    return box  # type: ignore[return-value]


def route_from_grounding_geometry(
        candidate_bbox: Sequence[float],
        grounding_bbox: Sequence[float],
        accept_iou_threshold: float = 0.5,
        containment_threshold: float = 0.7,
) -> GroundingGeometryDecision:
    """Map candidate-versus-grounder geometry to one routing action.

    ``expand`` requires the candidate to lie mostly inside a larger grounding
    box. ``tighten`` is the inverse. Remaining low-IoU relations are routed to
    ``relocate`` instead of guessing a size-only correction.
    """

    if not 0.0 < accept_iou_threshold <= 1.0:
        raise ValueError('accept_iou_threshold must be in (0, 1]')
    if not 0.0 < containment_threshold <= 1.0:
        raise ValueError('containment_threshold must be in (0, 1]')
    candidate = _box(candidate_bbox, 'candidate_bbox')
    grounding = _box(grounding_bbox, 'grounding_bbox')
    intersection_width = max(
        0.0,
        min(candidate[2], grounding[2]) - max(candidate[0], grounding[0]),
    )
    intersection_height = max(
        0.0,
        min(candidate[3], grounding[3]) - max(candidate[1], grounding[1]),
    )
    intersection = intersection_width * intersection_height
    candidate_area = (
        (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
    )
    grounding_area = (
        (grounding[2] - grounding[0]) * (grounding[3] - grounding[1])
    )
    union = candidate_area + grounding_area - intersection
    iou = intersection / union
    candidate_coverage = intersection / candidate_area
    grounding_coverage = intersection / grounding_area

    if iou >= accept_iou_threshold:
        action = 'no_action'
        reason = 'candidate_grounding_iou_meets_accept_threshold'
    elif (
        candidate_coverage >= containment_threshold
        and grounding_coverage < containment_threshold
    ):
        action = 'expand'
        reason = 'candidate_is_mostly_inside_larger_grounding_box'
    elif (
        grounding_coverage >= containment_threshold
        and candidate_coverage < containment_threshold
    ):
        action = 'tighten'
        reason = 'grounding_box_is_mostly_inside_larger_candidate'
    else:
        action = 'relocate'
        reason = 'low_iou_without_directional_containment'

    return GroundingGeometryDecision(
        action=action,
        reason=reason,
        intersection_area=intersection,
        candidate_area=candidate_area,
        grounding_area=grounding_area,
        iou=iou,
        candidate_coverage_by_grounding=candidate_coverage,
        grounding_coverage_by_candidate=grounding_coverage,
        accept_iou_threshold=float(accept_iou_threshold),
        containment_threshold=float(containment_threshold),
    )
