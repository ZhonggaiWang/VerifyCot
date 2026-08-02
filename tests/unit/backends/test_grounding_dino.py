"""CPU-only tests for Grounding DINO localization plus geometry routing."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from grounding_control.contracts import VerificationRequest
from grounding_control.models.grounding_dino import (
    GroundingDinoDetection,
    normalize_grounding_query,
)
from grounding_control.verifiers import (
    GeometryVerificationInput,
    PaddedGeometryVerificationInput,
)
from grounding_control.four_way.verifiers import (
    GroundingDinoGeometryClassifier,
    GroundingDinoGeometryVerifierBackend,
)


class _FakeGroundingDinoRunner:
    model_path = 'fake-grounding-dino'
    box_threshold = 0.3
    text_threshold = 0.25

    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = []
        self.last_run_metadata = {}

    def detect(self, image, object_reference):
        self.calls.append((image, object_reference))
        self.last_run_metadata = {
            'grounding_query': normalize_grounding_query(object_reference),
            'timing_ms': {'total': 2.5},
        }
        return list(self.detections)


def _detection(box, score=0.9, label='object'):
    return GroundingDinoDetection(tuple(box), score, label)


def _candidate(box):
    return GeometryVerificationInput(
        image=Image.new('RGB', (100, 80), 'white'),
        object_reference='The Object',
        candidate_bbox_pixel_xyxy=tuple(box),
        sample_id='sample-1',
    )


class GroundingDinoGeometryTests(unittest.TestCase):
    def test_query_normalization_is_conservative(self):
        self.assertEqual(
            normalize_grounding_query('  The   Red Car  '),
            'the red car.',
        )
        with self.assertRaises(ValueError):
            normalize_grounding_query('   ')

    def test_classifier_distinguishes_all_four_actions(self):
        cases = (
            ((10, 10, 30, 30), (10, 10, 30, 30), 'no_action'),
            ((20, 20, 30, 30), (10, 10, 40, 40), 'expand'),
            ((10, 10, 40, 40), (20, 20, 30, 30), 'tighten'),
            ((10, 10, 30, 30), (60, 50, 90, 75), 'relocate'),
        )
        for candidate_box, grounding_box, expected in cases:
            with self.subTest(expected=expected):
                lookup = GroundingDinoGeometryClassifier(
                    _FakeGroundingDinoRunner([_detection(grounding_box)])
                ).classify(_candidate(candidate_box))
                self.assertEqual(lookup.status, expected)
                self.assertEqual(
                    lookup.metadata['geometry']['action'],
                    expected,
                )
                action_output = GroundingDinoGeometryClassifier(
                    _FakeGroundingDinoRunner([_detection(grounding_box)])
                ).classify_action(_candidate(candidate_box))
                self.assertEqual(action_output.predicted_action, expected)
                self.assertIsNone(action_output.action_probabilities)
                self.assertFalse(action_output.abstained)

    def test_detector_never_receives_candidate_or_supervision(self):
        runner = _FakeGroundingDinoRunner([
            _detection((10, 10, 30, 30)),
        ])
        candidate = _candidate((10, 10, 30, 30))
        GroundingDinoGeometryClassifier(runner).classify(candidate)

        self.assertEqual(len(runner.calls), 1)
        image, reference = runner.calls[0]
        self.assertIs(image, candidate.image)
        self.assertEqual(reference, candidate.object_reference)

    def test_highest_score_is_selected_without_candidate_bias(self):
        runner = _FakeGroundingDinoRunner([
            _detection((10, 10, 30, 30), score=0.4, label='near'),
            _detection((60, 50, 90, 75), score=0.9, label='high'),
        ])
        lookup = GroundingDinoGeometryClassifier(runner).classify(
            _candidate((10, 10, 30, 30))
        )
        self.assertEqual(lookup.status, 'relocate')
        self.assertEqual(
            lookup.metadata['selected_detection_index'],
            1,
        )
        self.assertEqual(
            lookup.metadata['selected_grounding_label'],
            'high',
        )
        self.assertEqual(lookup.confidence, 0.9)
        self.assertIn(
            'not_calibrated',
            lookup.metadata['confidence_semantics'],
        )

    def test_equal_scores_preserve_detector_order(self):
        runner = _FakeGroundingDinoRunner([
            _detection((10, 10, 30, 30), score=0.8, label='first'),
            _detection((60, 50, 90, 75), score=0.8, label='second'),
        ])
        lookup = GroundingDinoGeometryClassifier(runner).classify(
            _candidate((10, 10, 30, 30))
        )
        self.assertEqual(lookup.status, 'no_action')
        self.assertEqual(lookup.metadata['selected_detection_index'], 0)

    def test_no_detection_is_an_end_to_end_failure(self):
        lookup = GroundingDinoGeometryClassifier(
            _FakeGroundingDinoRunner([])
        ).classify(_candidate((10, 10, 30, 30)))
        self.assertIsNone(lookup.status)
        self.assertIsNone(lookup.confidence)
        self.assertEqual(lookup.error, 'no_valid_grounding_detection')
        self.assertTrue(lookup.metadata['localization_failed'])
        self.assertTrue(lookup.metadata['parse_failed'])
        action_output = GroundingDinoGeometryClassifier(
            _FakeGroundingDinoRunner([])
        ).classify_action(_candidate((10, 10, 30, 30)))
        self.assertTrue(action_output.abstained)
        self.assertIsNone(action_output.predicted_action)

    def test_invalid_detection_is_rejected_and_valid_box_is_clipped(self):
        runner = _FakeGroundingDinoRunner([
            _detection((math.nan, 0, 10, 10), score=1.0, label='invalid'),
            _detection((-2, 5, 105, 70), score=0.8, label='clipped'),
        ])
        lookup = GroundingDinoGeometryClassifier(runner).classify(
            _candidate((0, 5, 100, 70))
        )
        self.assertEqual(lookup.status, 'no_action')
        self.assertEqual(lookup.metadata['invalid_detection_count'], 1)
        self.assertEqual(
            lookup.metadata[
                'selected_grounding_box_original_pixel_xyxy'
            ],
            [0.0, 5.0, 100.0, 70.0],
        )
        self.assertEqual(
            lookup.metadata['detections'][1]['boundary_clipped_sides'],
            ['x1', 'x2'],
        )

    def test_non_square_image_stays_in_original_pixel_coordinates(self):
        runner = _FakeGroundingDinoRunner([
            _detection((70, 10, 150, 40)),
        ])
        candidate = GeometryVerificationInput(
            image=Image.new('RGB', (200, 50), 'white'),
            object_reference='object',
            candidate_bbox_pixel_xyxy=(70, 10, 150, 40),
        )
        lookup = GroundingDinoGeometryClassifier(runner).classify(candidate)
        self.assertEqual(lookup.status, 'no_action')
        self.assertEqual(
            lookup.metadata['coordinate_system'],
            'absolute_xyxy_on_original_image',
        )
        self.assertEqual(
            lookup.metadata['original_image_size'],
            [200, 50],
        )

    def test_online_padded_candidate_uses_vocot_coordinate_frame(self):
        # A 100x50 source is padded by 25px above and below. The original
        # pixel box (20, 10, 80, 40) therefore becomes
        # (0.2, 0.35, 0.8, 0.65) in VoCoT's padded unit square.
        runner = _FakeGroundingDinoRunner([
            _detection((20, 10, 80, 40), score=0.9),
        ])
        output = GroundingDinoGeometryClassifier(
            runner
        ).classify_padded_action(PaddedGeometryVerificationInput(
            image=Image.new('RGB', (100, 50), 'white'),
            object_reference='object',
            candidate_bbox_padded_normalized_xyxy=(
                0.2, 0.35, 0.8, 0.65,
            ),
            sample_id='online',
        ))
        self.assertEqual(output.predicted_action, 'no_action')
        self.assertEqual(
            output.metadata[
                'selected_grounding_padded_normalized_bbox_xyxy'
            ],
            [0.2, 0.35, 0.8, 0.65],
        )
        self.assertEqual(
            output.metadata['candidate_padded_normalized_bbox_xyxy'],
            [0.2, 0.35, 0.8, 0.65],
        )

    def test_online_backend_consumes_verification_request(self):
        classifier = GroundingDinoGeometryClassifier(
            _FakeGroundingDinoRunner([
                _detection((20, 10, 80, 40), score=0.9),
            ])
        )
        output = GroundingDinoGeometryVerifierBackend(
            classifier
        ).verify_action(VerificationRequest(
            sample_id='online',
            grounding_step=1,
            object_reference='object',
            candidate_bbox=(0.2, 0.35, 0.8, 0.65),
            candidate_coordinate_text='',
            generated_ids=(),
            candidate_span=(0, 0),
            sample_context={
                'image': Image.new('RGB', (100, 50), 'white'),
            },
        ))
        self.assertEqual(output.predicted_action, 'no_action')

    def test_evaluator_dino_mode_does_not_construct_qwen_runner(self):
        from grounding_control.benchmarks.gqa_controlled import evaluator

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / 'empty.jsonl'
            benchmark.write_text('', encoding='utf-8')
            output = root / 'results.jsonl'
            argv = [
                'evaluator',
                '--benchmark', str(benchmark),
                '--model-path', 'unused',
                '--output', str(output),
                '--task-mode', 'routing_grounding_geometry',
                '--geometry-backend', 'grounding_dino',
                '--split', 'test',
            ]
            with mock.patch.object(sys, 'argv', argv):
                with mock.patch.object(
                    evaluator,
                    'LocalQwen25VLRunner',
                    side_effect=AssertionError('Qwen must not be constructed'),
                ):
                    evaluator.main()

            summary = json.loads(
                output.with_suffix('.summary.json').read_text(
                    encoding='utf-8'
                )
            )
            self.assertEqual(
                summary['backend'],
                'grounding_dino_geometry_router_raw_image',
            )
            self.assertEqual(
                summary['geometry_backend'],
                'grounding_dino',
            )
            self.assertIn(
                'no Qwen resize',
                summary['input_protocol']['coordinate_conversion'],
            )


if __name__ == '__main__':
    unittest.main()
