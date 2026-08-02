"""CPU-only tests for isolated binary and four-way DINO workers."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from grounding_control.coordinates import COORDINATE_SYSTEM
from grounding_control.models.grounding_dino import GroundingDinoDetection
from grounding_control.four_way.workers.dino_geometry_verifier import (
    DinoGeometryWorkerEngine,
)
from grounding_control.workers.dino_verifier import (
    DinoVerifierWorkerEngine,
    PROTOCOL_NAME,
)


class _Runner:
    model_path = 'fake-dino'
    box_threshold = 0.3
    text_threshold = 0.25
    last_run_metadata = {}

    def detect(self, image, object_reference):
        self.last_run_metadata = {'timing_ms': {'total': 1.0}}
        return [
            GroundingDinoDetection(
                box_original_pixel_xyxy=(20, 10, 80, 40),
                score=0.91,
                label=object_reference,
            )
        ]


class DinoWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / 'image.png'
        Image.new('RGB', (100, 50), 'white').save(self.image_path)

    def test_endpoint_returns_canonical_action_schema(self):
        engine = DinoGeometryWorkerEngine(runner=_Runner())
        response = engine.handle({
            'protocol': PROTOCOL_NAME,
            'operation': 'verify',
            'image_path': str(self.image_path),
            'sample_id': 'sample',
            'object_reference': 'cup',
            'candidate_bbox': [0.2, 0.35, 0.8, 0.65],
            'coordinate_system': COORDINATE_SYSTEM,
        })
        self.assertEqual(
            response['verifier_output_schema'],
            'vocot_four_action_v1',
        )
        self.assertEqual(response['predicted_action'], 'no_action')
        self.assertEqual(response['confidence'], 0.91)
        self.assertIsNone(response['action_probabilities'])
        self.assertFalse(response['abstained'])

    def test_binary_mode_returns_iou_score_not_detector_confidence(self):
        engine = DinoVerifierWorkerEngine(runner=_Runner())
        response = engine.handle({
            'protocol': PROTOCOL_NAME,
            'operation': 'verify',
            'verifier_mode': 'binary_alignment',
            'image_path': str(self.image_path),
            'sample_id': 'sample',
            'object_reference': 'cup',
            'candidate_bbox': [0.2, 0.35, 0.8, 0.65],
            'coordinate_system': COORDINATE_SYSTEM,
        })
        self.assertEqual(
            response['verifier_output_schema'],
            'vocot_alignment_score_v1',
        )
        self.assertEqual(response['alignment_score'], 1.0)
        self.assertEqual(response['metadata']['detector_confidence'], 0.91)
        self.assertFalse(
            response['metadata'][
                'detector_confidence_used_as_alignment_score'
            ]
        )

    def test_ping_without_model_is_explicit(self):
        response = DinoVerifierWorkerEngine().handle({
            'protocol': PROTOCOL_NAME,
            'operation': 'ping',
        })
        self.assertEqual(response['worker'], 'dino_geometry_verifier')
        self.assertFalse(response['configured'])
        self.assertEqual(response['verifier_mode'], 'binary_alignment')

        four_way = DinoGeometryWorkerEngine().handle({
            'protocol': PROTOCOL_NAME,
            'operation': 'ping',
        })
        self.assertEqual(four_way['worker'], 'dino_geometry_verifier')
        self.assertFalse(four_way['configured'])
        self.assertEqual(four_way['accept_iou_threshold'], 0.4)

    def test_binary_worker_rejects_four_way_mode(self):
        engine = DinoVerifierWorkerEngine(runner=_Runner())
        with self.assertRaisesRegex(ValueError, 'binary DINO verifier_mode'):
            engine.handle({
                'protocol': PROTOCOL_NAME,
                'operation': 'verify',
                'verifier_mode': 'grounding_dino_geometry',
                'image_path': str(self.image_path),
                'object_reference': 'cup',
                'candidate_bbox': [0.2, 0.35, 0.8, 0.65],
                'coordinate_system': COORDINATE_SYSTEM,
            })


if __name__ == '__main__':
    unittest.main()
