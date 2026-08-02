"""Golden tests for policy-free DINO geometry and the legacy adapter."""

from dataclasses import asdict
import unittest

from PIL import Image

from grounding_control.models.grounding_dino import GroundingDinoDetection
from grounding_control.verifiers.box_geometry import (
    PaddedGeometryVerificationInput,
    measure_box_geometry,
)
from grounding_control.four_way.verifiers.geometry import (
    route_from_grounding_geometry,
)
from grounding_control.verifiers.dino import (
    GroundingDinoAlignmentScorer,
)


class _DinoRunner:
    model_path = 'fake-dino'
    box_threshold = 0.3
    text_threshold = 0.25

    def __init__(self, detections):
        self.detections = list(detections)
        self.last_run_metadata = {}

    def detect(self, image, object_reference):
        self.last_run_metadata = {'timing_ms': {'total': 1.0}}
        return list(self.detections)


class GeometryBinaryDecouplingTests(unittest.TestCase):
    def test_policy_free_measurement_and_legacy_route_are_golden(self):
        candidate = (10.0, 10.0, 30.0, 30.0)
        grounding = (20.0, 20.0, 40.0, 40.0)

        self.assertEqual(asdict(measure_box_geometry(candidate, grounding)), {
            'intersection_area': 100.0,
            'candidate_area': 400.0,
            'grounding_area': 400.0,
            'iou': 1.0 / 7.0,
            'candidate_coverage_by_grounding': 0.25,
            'grounding_coverage_by_candidate': 0.25,
        })
        self.assertEqual(asdict(route_from_grounding_geometry(
            candidate,
            grounding,
            accept_iou_threshold=0.5,
            containment_threshold=0.7,
        )), {
            'action': 'relocate',
            'reason': 'low_iou_without_directional_containment',
            'intersection_area': 100.0,
            'candidate_area': 400.0,
            'grounding_area': 400.0,
            'iou': 1.0 / 7.0,
            'candidate_coverage_by_grounding': 0.25,
            'grounding_coverage_by_candidate': 0.25,
            'accept_iou_threshold': 0.5,
            'containment_threshold': 0.7,
        })

    def test_binary_dino_path_never_calls_four_way_classifier(self):
        classifier = GroundingDinoAlignmentScorer(
            _DinoRunner([
                GroundingDinoDetection(
                    (20.0, 20.0, 40.0, 40.0),
                    0.99,
                    'cup',
                )
            ]),
        )

        # A binary-only classifier cannot accidentally enter either legacy
        # action entry point and does not even own their thresholds.
        self.assertFalse(hasattr(classifier, 'classify_action'))
        self.assertFalse(hasattr(classifier, 'classify_padded_action'))

        output = classifier.classify_padded_alignment(
            PaddedGeometryVerificationInput(
                image=Image.new('RGB', (100, 100), 'white'),
                object_reference='the cup',
                candidate_bbox_padded_normalized_xyxy=(
                    0.1, 0.1, 0.3, 0.3,
                ),
                sample_id='binary-does-not-route',
            )
        )

        self.assertAlmostEqual(output.alignment_score, 1.0 / 7.0)
        self.assertFalse(output.abstained)
        self.assertFalse(hasattr(
            classifier.padded_comparator,
            'accept_iou_threshold',
        ))
        self.assertFalse(hasattr(
            classifier.padded_comparator,
            'containment_threshold',
        ))
        self.assertEqual(
            set(output.metadata['geometry_measurement']),
            {
                'intersection_area',
                'candidate_area',
                'grounding_area',
                'iou',
                'candidate_coverage_by_grounding',
                'grounding_coverage_by_candidate',
            },
        )


if __name__ == '__main__':
    unittest.main()
