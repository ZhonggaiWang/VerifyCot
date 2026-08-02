"""CPU-only tests for binary alignment backend adapters."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from grounding_control.contracts import (
    VerificationRequest,
    VerifierFailClosedError,
)
from grounding_control.models.grounding_dino import GroundingDinoDetection
from grounding_control.oracle_targets import OracleTargetResolution
from grounding_control.verifiers import (
    GroundingDinoAlignmentVerifierBackend,
    OracleAlignmentVerifierBackend,
    Qwen25VLAlignmentVerifierBackend,
    RemoteAlignmentVerifierBackend,
)
from grounding_control.verifiers.dino import (
    GroundingDinoAlignmentScorer,
)


class _QwenRunner:
    min_pixels = 4 * 28 * 28
    max_pixels = 512 * 28 * 28

    def __init__(self, response):
        self.response = response

    def generate(self, messages):
        return self.response


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


class _Resolver:
    def __init__(self, resolution):
        self.resolution = resolution

    def resolve(self, request):
        return self.resolution


class _RemoteClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def request(self, payload, timeout=None):
        self.requests.append((dict(payload), timeout))
        if self.error is not None:
            raise self.error
        return dict(self.response)


def _request(*, box=(0.1, 0.1, 0.3, 0.3), context=None):
    return VerificationRequest(
        sample_id='sample',
        grounding_step=1,
        object_reference='the cup',
        candidate_bbox=tuple(box),
        candidate_coordinate_text='',
        generated_ids=(1, 2, 3),
        candidate_span=(1, 2),
        sample_context=dict(context or {}),
    ).alignment_request()


class AlignmentBackendTests(unittest.TestCase):
    def test_qwen_misaligned_label_confidence_becomes_low_signed_score(self):
        output = Qwen25VLAlignmentVerifierBackend(
            runner=_QwenRunner(
                '{"aligned":false,"confidence":0.9}'
            ),
            image_mode='crop_only',
        ).verify_alignment(_request(context={
            'image': Image.new('RGB', (100, 100), 'white'),
        }))
        self.assertAlmostEqual(output.alignment_score, 0.1)
        self.assertFalse(output.abstained)
        self.assertFalse(output.metadata['binary_aligned_label'])
        self.assertFalse(output.metadata['alignment_score_calibrated'])

    def test_qwen_aligned_label_confidence_keeps_high_signed_score(self):
        output = Qwen25VLAlignmentVerifierBackend(
            runner=_QwenRunner(
                '{"aligned":true,"confidence":0.8}'
            ),
            image_mode='bbox_image_only',
        ).verify_alignment(_request(context={
            'image': Image.new('RGB', (100, 100), 'white'),
        }))
        self.assertAlmostEqual(output.alignment_score, 0.8)

    def test_qwen_inconsistent_label_confidence_abstains(self):
        output = Qwen25VLAlignmentVerifierBackend(
            runner=_QwenRunner(
                '{"aligned":false,"confidence":0.2}'
            ),
            image_mode='crop_only',
        ).verify_alignment(_request(context={
            'image': Image.new('RGB', (100, 100), 'white'),
        }))
        self.assertTrue(output.abstained)
        self.assertIsNone(output.alignment_score)
        self.assertTrue(
            output.metadata['binary_label_confidence_inconsistent']
        )

    def test_dino_uses_candidate_grounding_iou_not_detector_score(self):
        classifier = GroundingDinoAlignmentScorer(
            _DinoRunner([
                GroundingDinoDetection(
                    (20, 20, 40, 40),
                    0.99,
                    'cup',
                )
            ])
        )
        output = GroundingDinoAlignmentVerifierBackend(
            classifier
        ).verify_alignment(_request(context={
            'image': Image.new('RGB', (100, 100), 'white'),
        }))
        # Candidate (10,10,30,30) and DINO (20,20,40,40): 100 / 700.
        self.assertAlmostEqual(output.alignment_score, 1.0 / 7.0)
        self.assertEqual(output.metadata['detector_confidence'], 0.99)
        self.assertFalse(
            output.metadata['detector_confidence_used_as_alignment_score']
        )

    def test_dino_without_detection_abstains(self):
        output = GroundingDinoAlignmentVerifierBackend(
            GroundingDinoAlignmentScorer(_DinoRunner([]))
        ).verify_alignment(_request(context={
            'image': Image.new('RGB', (100, 100), 'white'),
        }))
        self.assertTrue(output.abstained)
        self.assertIsNone(output.alignment_score)
        self.assertEqual(output.error, 'no_valid_grounding_detection')

    def test_oracle_returns_hard_score_and_unmatched_abstains(self):
        matched = OracleTargetResolution(
            matched=True,
            reason='explicit_alias',
            target_object='cup',
            matched_alias='cup',
            bbox=(0.1, 0.1, 0.3, 0.3),
            target_index=0,
        )
        backend = OracleAlignmentVerifierBackend(
            _Resolver(matched),
            gt_iou_threshold=0.5,
        )
        self.assertEqual(
            backend.verify_alignment(_request()).alignment_score,
            1.0,
        )
        self.assertEqual(
            backend.verify_alignment(
                _request(box=(0.6, 0.6, 0.8, 0.8))
            ).alignment_score,
            0.0,
        )

        unavailable = OracleAlignmentVerifierBackend(_Resolver(
            OracleTargetResolution(
                matched=False,
                reason='no_explicit_target_alias',
            )
        )).verify_alignment(_request())
        self.assertTrue(unavailable.abstained)
        self.assertIsNone(unavailable.alignment_score)

    def test_remote_explicitly_requests_binary_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (20, 10), 'white').save(path)
            client = _RemoteClient({
                'request_id': 'request-1',
                'verifier_mode': 'binary_alignment',
                'verifier_output_schema': 'vocot_alignment_score_v1',
                'alignment_score': 0.72,
                'score_semantics': 'calibrated_alignment_probability',
                'abstained': False,
                'error': None,
                'metadata': {'backend': 'fake'},
            })
            output = RemoteAlignmentVerifierBackend(
                client,
                image_mode='crop_only',
            ).verify_alignment(_request(context={'image_path': str(path)}))

        self.assertEqual(output.alignment_score, 0.72)
        payload, timeout = client.requests[0]
        self.assertEqual(payload['verifier_mode'], 'binary_alignment')
        self.assertEqual(payload['image_mode'], 'crop_only')
        self.assertEqual(timeout, 300.0)

    def test_remote_failure_is_an_explicit_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (20, 10), 'white').save(path)
            output = RemoteAlignmentVerifierBackend(
                _RemoteClient(error=RuntimeError('worker died')),
                fail_open=True,
            ).verify_alignment(_request(context={'image_path': str(path)}))
        self.assertTrue(output.abstained)
        self.assertTrue(output.metadata['remote_failure'])

    def test_remote_failure_raises_typed_error_when_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (20, 10), 'white').save(path)
            backend = RemoteAlignmentVerifierBackend(
                _RemoteClient(error=RuntimeError('worker died')),
                fail_open=False,
            )
            with self.assertRaises(VerifierFailClosedError) as raised:
                backend.verify_alignment(
                    _request(context={'image_path': str(path)})
                )

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIn('RuntimeError: worker died', str(raised.exception))
        self.assertTrue(raised.exception.metadata['remote_failure'])
        self.assertEqual(
            raised.exception.metadata['requested_verifier_mode'],
            'binary_alignment',
        )

    def test_remote_rejects_missing_binary_wire_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            Image.new('RGB', (20, 10), 'white').save(path)
            output = RemoteAlignmentVerifierBackend(
                _RemoteClient({
                    'alignment_score': 0.9,
                    'score_semantics': 'calibrated_alignment_probability',
                    'abstained': False,
                    'error': None,
                    'metadata': {},
                }),
                fail_open=True,
            ).verify_alignment(_request(context={'image_path': str(path)}))
        self.assertTrue(output.abstained)
        self.assertTrue(output.metadata['remote_failure'])
        self.assertIn('verifier_mode', output.error)


if __name__ == '__main__':
    unittest.main()
