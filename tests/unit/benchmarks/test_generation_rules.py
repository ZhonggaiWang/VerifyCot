"""Unit tests for the five controlled verifier benchmark constructions."""

import random

from eval.Oracle_experiment.generate_datasets.verifier_benchmark import (
    generate_aligned,
    generate_ambiguous,
    generate_partial_coverage,
    generate_unsupported,
    generate_wrong_object,
)


IMAGE_WIDTH = 200
IMAGE_HEIGHT = 160


def _object(object_id, name, box):
    x1, y1, x2, y2 = box
    return {
        'object_id': object_id,
        'name': name,
        'canonical_name': name,
        'pixel_box_xyxy': list(box),
        'normalized_box_xyxy': [
            x1 / IMAGE_WIDTH, y1 / IMAGE_HEIGHT,
            x2 / IMAGE_WIDTH, y2 / IMAGE_HEIGHT,
        ],
        'area_fraction': ((x2 - x1) * (y2 - y1)) / (IMAGE_WIDTH * IMAGE_HEIGHT),
    }


def test_five_generation_rules_have_expected_labels_and_geometry():
    target = _object('target', 'cat', (20, 30, 80, 110))
    distractor = _object('other', 'bed', (110, 60, 180, 130))
    objects = [target, distractor]

    aligned = generate_aligned(
        target, IMAGE_WIDTH, IMAGE_HEIGHT, random.Random(1)
    )
    assert aligned['verdict'] == 'aligned'
    assert aligned['reason'] == 'aligned'
    assert aligned['candidate_target_geometry']['iou'] >= 0.70

    wrong = generate_wrong_object(
        target, objects, IMAGE_WIDTH, IMAGE_HEIGHT, random.Random(2)
    )
    assert wrong['verdict'] == 'misaligned'
    assert wrong['reason'] == 'wrong_object'
    assert wrong['distractor_object_id'] == 'other'
    assert wrong['candidate_target_geometry']['candidate_purity'] == 0.0

    partial = generate_partial_coverage(
        target, IMAGE_WIDTH, IMAGE_HEIGHT, random.Random(3)
    )
    assert partial['reason'] == 'partial_coverage'
    assert 0.25 <= partial['candidate_target_geometry']['reference_coverage'] <= 0.50
    assert partial['candidate_target_geometry']['candidate_purity'] == 1.0

    ambiguous = generate_ambiguous(
        target, objects, IMAGE_WIDTH, IMAGE_HEIGHT, random.Random(4)
    )
    assert ambiguous['reason'] == 'ambiguous'
    assert ambiguous['distractor_object_name'] == 'bed'
    assert ambiguous['candidate_target_geometry']['reference_coverage'] == 1.0
    assert ambiguous['candidate_distractor_geometry']['reference_coverage'] == 1.0

    unsupported = generate_unsupported(
        target, objects, IMAGE_WIDTH, IMAGE_HEIGHT, random.Random(5)
    )
    assert unsupported['reason'] == 'unsupported'
    assert unsupported['candidate_target_geometry']['iou'] <= 0.02


def test_wrong_object_rejects_nested_part_whole_pair():
    target = _object('pizza', 'pizza', (20, 20, 180, 140))
    nested_pepper = _object('pepper', 'pepper', (70, 60, 90, 80))
    generated = generate_wrong_object(
        target, [target, nested_pepper], IMAGE_WIDTH, IMAGE_HEIGHT, random.Random(6)
    )
    assert generated is None
