"""Strictly audit naturally generated CoT coordinates against annotated GT."""

from typing import Any, Dict, Optional, Sequence

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from utils.coordinate_intervention import (
    OnlineOracleCoordinateLogitsProcessor,
    box_iou,
    find_coordinate_spans,
)


def _intersection_over_reference(candidate: Sequence[float], reference: Sequence[float]) -> float:
    x1 = max(float(candidate[0]), float(reference[0]))
    y1 = max(float(candidate[1]), float(reference[1]))
    x2 = min(float(candidate[2]), float(reference[2]))
    y2 = min(float(candidate[3]), float(reference[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    reference_area = max(0.0, float(reference[2]) - float(reference[0])) * max(
        0.0, float(reference[3]) - float(reference[1])
    )
    return 0.0 if reference_area <= 0 else intersection / reference_area


def audit_natural_coordinates(
        tokenizer,
        generated_ids: Sequence[int],
        generated_boxes: Sequence[Optional[Sequence[float]]],
        oracle_targets: Sequence[Dict[str, Any]],
        iou_threshold: float = 0.5,
        context_window_tokens: int = 48) -> Dict[str, Any]:
    """Match baseline coordinates to unique explicit targets and flag errors.

    Only the conservative matcher shared with the online GT oracle is used.
    Unmatched or ambiguous references remain ineligible rather than being
    treated as grounding errors.
    """
    if not 0 <= float(iou_threshold) <= 1:
        raise ValueError('iou_threshold must be in [0, 1]')
    boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
    eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
    spans = find_coordinate_spans(generated_ids, boc_token_id, eoc_token_id)
    if len(spans) != len(generated_boxes):
        raise ValueError(
            f'baseline has {len(spans)} coordinate spans but {len(generated_boxes)} boxes'
        )
    matcher = OnlineOracleCoordinateLogitsProcessor(
        tokenizer,
        prompt_length=0,
        oracle_targets=oracle_targets,
        context_window_tokens=context_window_tokens,
    )
    events = []
    for coordinate_index, ((boc_offset, _), candidate_box) in enumerate(
            zip(spans, generated_boxes), 1):
        decision = matcher._new_decision(generated_ids, boc_offset, coordinate_index)
        event = dict(decision)
        event.update({
            'eligible': False,
            'baseline_box': None,
            'baseline_iou_to_gt': None,
            'baseline_gt_coverage': None,
            'gt_box_area': None,
            'is_natural_error': False,
            'iou_threshold': float(iou_threshold),
        })
        if decision['decision'] == 'forced_gt_box' and candidate_box is not None:
            candidate = [float(value) for value in candidate_box]
            gt_box = [float(value) for value in decision['oracle_box']]
            iou = box_iou(candidate, gt_box)
            gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            event.update({
                'eligible': True,
                'baseline_box': candidate,
                'baseline_iou_to_gt': iou,
                'baseline_gt_coverage': _intersection_over_reference(candidate, gt_box),
                'gt_box_area': gt_area,
                'is_natural_error': iou < float(iou_threshold),
            })
        elif decision['decision'] == 'forced_gt_box':
            event['reason'] = 'malformed_or_missing_baseline_box'
        events.append(event)

    eligible = [event for event in events if event['eligible']]
    errors = [event for event in eligible if event['is_natural_error']]
    return {
        'coordinate_count': len(events),
        'eligible_coordinate_count': len(eligible),
        'ineligible_coordinate_count': len(events) - len(eligible),
        'natural_error_coordinate_count': len(errors),
        'has_eligible_coordinate': bool(eligible),
        'has_natural_error': bool(errors),
        'selected_first_natural_error': None if not errors else errors[0],
        'events': events,
        'matching_policy': 'latest_unique_longest_explicit_alias',
        'error_rule': f'IoU < {float(iou_threshold):g}',
    }
