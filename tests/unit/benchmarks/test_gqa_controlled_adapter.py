"""CPU-only tests for the production-faithful GQA benchmark adapter."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from grounding_control.verifiers.qwen25_vl import (
    COORDINATE_SYSTEM,
    CandidateVerificationInput,
)
from grounding_control.four_way.verifiers.qwen25_vl import GroundingActionInput
from grounding_control.benchmarks.gqa_controlled import (
    compute_binary_alignment_metrics,
    compute_routing_metrics,
    expected_status_from_record,
    load_examples,
)
from grounding_control.verifiers.box_geometry import GeometryVerificationInput


class GQAControlledAdapterTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(
            expected_status_from_record({
                'verdict': 'aligned',
                'reason': 'aligned',
            }),
            'aligned',
        )
        self.assertEqual(
            expected_status_from_record({
                'verdict': 'misaligned',
                'reason': 'unsupported',
            }),
            'unsupported',
        )
        with self.assertRaises(ValueError):
            expected_status_from_record({
                'verdict': 'misaligned',
                'reason': 'other',
            })

    def test_adapter_converts_pixel_box_without_label_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'landscape.png'
            Image.new('RGB', (7, 4), 'white').save(image_path)
            manifest = root / 'benchmark.jsonl'
            row = {
                'event_id': 'gqa:test:1:aligned:0',
                'sample_index': 0,
                'split': 'test',
                'image_id': 'test',
                'source_image': str(image_path),
                'object_reference': 'lamp',
                'candidate_box_pixel_xyxy': [0.0, 0.0, 7.0, 4.0],
                'target_box_pixel_xyxy': [1.0, 1.0, 2.0, 2.0],
                'candidate_target_geometry': {'iou': 0.1},
                'construction': 'hidden_supervision',
                'verdict': 'aligned',
                'reason': 'aligned',
            }
            manifest.write_text(json.dumps(row) + '\n', encoding='utf-8')

            example = load_examples(manifest, 'test')[0]
            candidate = example.to_candidate_input()

            self.assertIsInstance(candidate, CandidateVerificationInput)
            self.assertEqual(candidate.sample_id, row['event_id'])
            self.assertEqual(candidate.object_reference, 'lamp')
            self.assertEqual(candidate.candidate_bbox, (0.0, 1 / 7, 1.0, 5 / 7))
            self.assertEqual(candidate.coordinate_system, COORDINATE_SYSTEM)
            self.assertNotIn('expected_status', candidate.__dict__)
            self.assertNotIn('target_box', candidate.__dict__)
            self.assertNotIn('construction', candidate.__dict__)

            action_input = example.to_grounding_action_input()
            self.assertIsInstance(action_input, GroundingActionInput)
            self.assertEqual(action_input.image.size, (7, 4))
            self.assertEqual(
                action_input.candidate_bbox_pixel_xyxy,
                (0.0, 0.0, 7.0, 4.0),
            )
            self.assertNotIn('expected_status', action_input.__dict__)
            self.assertNotIn('target_box', action_input.__dict__)
            self.assertNotIn('construction', action_input.__dict__)

            geometry_input = example.to_geometry_verification_input()
            self.assertIsInstance(
                geometry_input,
                GeometryVerificationInput,
            )
            self.assertEqual(geometry_input.image.size, (7, 4))
            self.assertEqual(
                geometry_input.candidate_bbox_pixel_xyxy,
                (0.0, 0.0, 7.0, 4.0),
            )
            self.assertNotIn('expected_status', geometry_input.__dict__)
            self.assertNotIn('target_box', geometry_input.__dict__)
            self.assertNotIn('construction', geometry_input.__dict__)

    def test_split_filtering_preserves_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / 'image.png'
            Image.new('RGB', (8, 8), 'white').save(image_path)
            rows = []
            for index, split in enumerate(('test', 'dev', 'test')):
                rows.append({
                    'event_id': f'event-{index}',
                    'sample_index': index,
                    'split': split,
                    'image_id': str(index),
                    'source_image': str(image_path),
                    'object_reference': 'object',
                    'candidate_box_pixel_xyxy': [1, 1, 4, 4],
                    'verdict': 'aligned',
                    'reason': 'aligned',
                })
            manifest = root / 'benchmark.jsonl'
            manifest.write_text(
                ''.join(json.dumps(row) + '\n' for row in rows),
                encoding='utf-8',
            )
            self.assertEqual(
                [item.event_id for item in load_examples(manifest, 'test')],
                ['event-0', 'event-2'],
            )


class GQAControlledMetricsTests(unittest.TestCase):
    def test_binary_alignment_metrics(self):
        metrics = compute_binary_alignment_metrics([
            {
                'expected_status': 'aligned',
                'expected_alignment': True,
                'predicted_alignment': True,
            },
            {
                'expected_status': 'wrong_object',
                'expected_alignment': False,
                'predicted_alignment': False,
            },
            {
                'expected_status': 'partial_coverage',
                'expected_alignment': False,
                'predicted_alignment': True,
            },
        ])
        self.assertAlmostEqual(metrics['end_to_end_accuracy'], 2 / 3)
        self.assertAlmostEqual(metrics['recall'], 0.5)
        self.assertEqual(
            metrics['by_expected_status']['wrong_object']['correct'],
            1,
        )

    def test_four_way_parse_failures_are_end_to_end_errors(self):
        rows = [
            {
                'expected_routing_status': 'no_action',
                'predicted_routing_status': 'no_action',
                'confidence': 0.9,
            },
            {
                'expected_routing_status': 'relocate',
                'predicted_routing_status': 'relocate',
                'confidence': 0.8,
            },
            {
                'expected_routing_status': 'expand',
                'predicted_routing_status': 'relocate',
                'confidence': 0.7,
            },
            {
                'expected_routing_status': 'tighten',
                'predicted_routing_status': 'tighten',
                'confidence': 0.6,
            },
            {
                'expected_routing_status': 'relocate',
                'predicted_routing_status': None,
                'confidence': None,
            },
        ]
        metrics = compute_routing_metrics(rows)
        self.assertEqual(metrics['total'], 5)
        self.assertEqual(metrics['parsed_count'], 4)
        self.assertAlmostEqual(metrics['parse_success_rate'], 0.8)
        self.assertAlmostEqual(metrics['four_way']['accuracy'], 0.6)
        self.assertEqual(
            metrics['four_way']['confusion_matrix']['relocate']['parse_failure'],
            1,
        )


if __name__ == '__main__':
    unittest.main()
