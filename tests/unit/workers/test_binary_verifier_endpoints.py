"""Binary worker endpoints must stay independent of four-way code."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from grounding_control.verifiers.qwen25_vl.classifier import (
    Qwen25VLBinaryAlignmentClassifier,
)
from grounding_control.verifiers.qwen25_vl.rendering import COORDINATE_SYSTEM
from grounding_control.workers.endpoints.qwen_alignment_verifier import (
    QwenAlignmentVerifierEndpoint,
)
from grounding_control.workers.qwen_verifier import QwenVerifierWorkerEngine


class _FakeQwenRunner:
    min_pixels = 4 * 28 * 28
    max_pixels = 512 * 28 * 28

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.response


class BinaryVerifierEndpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / 'image.png'
        Image.new('RGB', (100, 80), 'white').save(self.image_path)

    def test_qwen_endpoint_returns_only_binary_alignment_schema(self):
        runner = _FakeQwenRunner(
            '{"aligned":false,"confidence":0.8}'
        )
        endpoint = QwenAlignmentVerifierEndpoint(
            Qwen25VLBinaryAlignmentClassifier(runner),
            default_image_mode='crop_only',
        )
        response = endpoint.handle({
            'image_path': str(self.image_path),
            'object_reference': 'object',
            'candidate_bbox': [0.1, 0.2, 0.4, 0.6],
            'coordinate_system': COORDINATE_SYSTEM,
        })
        self.assertEqual(
            response['verifier_output_schema'],
            'vocot_alignment_score_v1',
        )
        self.assertAlmostEqual(response['alignment_score'], 0.2)
        self.assertEqual(response['aligned'], False)
        self.assertEqual(len(runner.calls), 1)

    def test_qwen_endpoint_rejects_four_way_mode(self):
        endpoint = QwenAlignmentVerifierEndpoint(
            Qwen25VLBinaryAlignmentClassifier(
                _FakeQwenRunner('{"aligned":true,"confidence":0.9}')
            )
        )
        with self.assertRaisesRegex(ValueError, 'binary Qwen verifier_mode'):
            endpoint.handle({
                'verifier_mode': 'routing_four_way',
                'image_path': str(self.image_path),
                'object_reference': 'object',
                'candidate_bbox': [0.1, 0.2, 0.4, 0.6],
            })

    def test_qwen_binary_worker_ping_without_model(self):
        response = QwenVerifierWorkerEngine().handle({
            'protocol': 'vocot_worker_v1',
            'operation': 'ping',
        })
        self.assertEqual(response['worker'], 'qwen25_vl_alignment_verifier')
        self.assertFalse(response['configured'])
        self.assertEqual(response['verifier_mode'], 'binary_alignment')

    def test_importing_binary_workers_does_not_import_four_way(self):
        project_root = Path(__file__).resolve().parents[3]
        script = (
            'import sys; '
            'import grounding_control.workers.dino_verifier; '
            'import grounding_control.workers.qwen_verifier; '
            'import grounding_control.workers.endpoints.dino_verifier; '
            'import grounding_control.workers.endpoints.qwen_alignment_verifier; '
            'bad=[name for name in sys.modules '
            'if name.startswith("grounding_control.four_way")]; '
            'assert not bad, bad'
        )
        completed = subprocess.run(
            [sys.executable, '-c', script],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == '__main__':
    unittest.main()
