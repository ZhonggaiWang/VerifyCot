"""CPU tests for the post-hoc VStar DINO routing audit."""

from copy import deepcopy

import pytest

from eval.Oracle_experiment.vstar.evaluate_dino_geometry_oracle_experts import (
    _posthoc_oracle_audit_events,
    _posthoc_oracle_metrics,
)


GT_BOX = [0.2, 0.2, 0.6, 0.6]
TARGETS = [{
    'object': 'cup',
    'aliases': ['cup'],
    'box': GT_BOX,
}]


def _event(
        candidate,
        committed,
        predicted_action,
        *,
        reference='Find the cup',
        dino_box=GT_BOX):
    return {
        'object_reference': reference,
        'candidate_box': list(candidate),
        'committed_box': list(committed),
        'predicted_action': predicted_action,
        'routing_decision': predicted_action or 'no_action',
        'verifier_metadata': {
            'selected_grounding_padded_normalized_bbox_xyxy': dino_box,
        },
    }


def _audit(events):
    return _posthoc_oracle_audit_events(
        events,
        TARGETS,
        context_window_tokens=48,
        accept_iou_threshold=0.5,
        containment_threshold=0.7,
    )


def test_posthoc_audit_records_all_three_ious_without_mutating_events():
    source = _event(
        candidate=[0.7, 0.7, 0.9, 0.9],
        committed=GT_BOX,
        predicted_action='relocate',
        dino_box=[0.2, 0.2, 0.4, 0.4],
    )
    before = deepcopy(source)

    audited = _audit([source])[0]
    audit = audited['posthoc_oracle_audit']

    assert source == before
    assert audited is not source
    assert audit['matchable'] is True
    assert audit['target_object'] == 'cup'
    assert audit['candidate_iou_to_gt'] == 0.0
    assert audit['committed_iou_to_gt'] == 1.0
    assert audit['dino_iou_to_gt'] == pytest.approx(0.25)
    assert audit['oracle_geometry_action'] == 'relocate'
    assert audit['predicted_action_correct'] is True
    assert audit['binary_route_correct'] is True


def test_posthoc_audit_excludes_unmatchable_reference_from_gt_metrics():
    audited = _audit([_event(
        candidate=GT_BOX,
        committed=GT_BOX,
        predicted_action='no_action',
        reference='it',
    )])[0]['posthoc_oracle_audit']

    assert audited['matchable'] is False
    assert audited['match_reason'] == 'no_explicit_target_alias'
    assert audited['oracle_target_box'] is None
    assert audited['candidate_iou_to_gt'] is None
    assert audited['oracle_geometry_action'] is None
    assert audited['predicted_action_correct'] is None


def test_posthoc_summary_reports_four_way_binary_and_miou_metrics():
    # Truth/prediction pairs:
    # no_action/no_action, relocate/relocate, expand/expand,
    # tighten/relocate, no_action/relocate, relocate/no_action.
    events = _audit([
        _event(GT_BOX, GT_BOX, 'no_action', dino_box=GT_BOX),
        _event(
            [0.7, 0.7, 0.9, 0.9], GT_BOX, 'relocate',
            dino_box=[0.7, 0.7, 0.9, 0.9],
        ),
        _event(
            [0.3, 0.3, 0.5, 0.5], GT_BOX, 'expand',
            dino_box=GT_BOX,
        ),
        _event(
            [0.1, 0.1, 0.7, 0.7], GT_BOX, 'relocate',
            dino_box=[0.1, 0.1, 0.7, 0.7],
        ),
        _event(GT_BOX, GT_BOX, 'relocate', dino_box=GT_BOX),
        _event(
            [0.7, 0.7, 0.9, 0.9],
            [0.7, 0.7, 0.9, 0.9],
            'no_action',
            dino_box=None,
        ),
        _event(
            GT_BOX,
            GT_BOX,
            'no_action',
            reference='the dog',
            dino_box=None,
        ),
    ])
    metrics = _posthoc_oracle_metrics([{
        'intervention': {'events': events},
    }])

    assert metrics['coordinate_event_count'] == 7
    assert metrics['audited_event_count'] == 7
    assert metrics['missing_audit_event_count'] == 0
    assert metrics['matchable_event_count'] == 6
    assert metrics['unmatchable_event_count'] == 1
    assert metrics['samples_with_matchable_event'] == 1

    four_way = metrics['four_way']
    assert four_way['correct_count'] == 3
    assert four_way['accuracy'] == pytest.approx(0.5)
    assert (
        four_way['confusion_true_by_predicted']['tighten']['relocate']
        == 1
    )
    assert (
        four_way['confusion_true_by_predicted']['no_action']['relocate']
        == 1
    )

    binary = metrics['binary_route']
    assert binary == {
        'positive_definition': 'action != no_action',
        'prediction_basis': (
            'raw_verifier_predicted_action_before_confidence_policy'
        ),
        'true_positive': 3,
        'false_positive': 1,
        'true_negative': 1,
        'false_negative': 1,
        'precision': pytest.approx(0.75),
        'recall': pytest.approx(0.75),
        'f1': pytest.approx(0.75),
        'accuracy': pytest.approx(4 / 6),
    }

    iou = metrics['iou']
    assert iou['candidate_to_gt_count'] == 6
    assert iou['committed_to_gt_count'] == 6
    assert iou['dino_to_gt_count'] == 5
    assert iou['committed_to_gt_miou'] == pytest.approx(5 / 6)
    assert iou['dino_box_coverage_rate'] == pytest.approx(5 / 6)
    assert iou['committed_minus_candidate_miou'] > 0


def test_posthoc_audit_records_malformed_dino_diagnostic_without_crashing():
    event = _event(
        GT_BOX,
        GT_BOX,
        'no_action',
        dino_box=[0.2, 0.2, 0.1, 0.1],
    )
    audit = _audit([event])[0]['posthoc_oracle_audit']

    assert audit['matchable'] is True
    assert audit['dino_box_available'] is False
    assert audit['dino_iou_to_gt'] is None
    assert audit['dino_box_error'].startswith('ValueError:')
