"""CPU-only tests for persistent worker transport and remote verifier."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from grounding_control.contracts import (
    VerificationRequest,
    VerifierFailClosedError,
)
from grounding_control.core.precommit_controller import (
    PrecommitGroundingController,
)
from grounding_control.transport import PersistentJsonlWorkerClient
from grounding_control.verifiers.remote import RemoteAlignmentVerifierBackend
from grounding_control.four_way.verifiers.remote import (
    RemoteActionVerifierBackend,
)


class _Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def request(self, payload, timeout=None):
        self.requests.append((dict(payload), timeout))
        if self.error is not None:
            raise self.error
        return dict(self.response)


def _request(image_path):
    return VerificationRequest(
        sample_id='sample',
        grounding_step=2,
        object_reference='the cup',
        candidate_bbox=(0.1, 0.2, 0.3, 0.4),
        candidate_coordinate_text='',
        generated_ids=(1, 2),
        candidate_span=(0, 1),
        sample_context={'image_path': image_path},
    )


class RemoteVerifierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / 'image.png'
        Image.new('RGB', (20, 10), 'white').save(self.image_path)

    def test_remote_backend_parses_versioned_output(self):
        client = _Client({
            'request_id': 'remote-1',
            'verifier_mode': 'grounding_dino_geometry',
            'verifier_output_schema': 'vocot_four_action_v1',
            'predicted_action': 'relocate',
            'action_probabilities': None,
            'confidence': 0.9,
            'abstained': False,
            'error': None,
            'metadata': {'probability_source': 'hard'},
        })
        output = RemoteActionVerifierBackend(client).verify_action(
            _request(self.image_path)
        )
        self.assertEqual(output.predicted_action, 'relocate')
        self.assertEqual(output.metadata['remote_request_id'], 'remote-1')
        payload, timeout = client.requests[0]
        self.assertEqual(payload['operation'], 'verify')
        self.assertEqual(payload['grounding_step'], 2)
        self.assertEqual(timeout, 300.0)

    def test_remote_failure_can_fail_open(self):
        output = RemoteActionVerifierBackend(
            _Client(error=RuntimeError('worker died')),
            fail_open=True,
        ).verify_action(_request(self.image_path))
        self.assertTrue(output.abstained)
        self.assertTrue(output.metadata['remote_failure'])

    def test_binary_remote_fail_closed_reaches_controller_caller(self):
        controller = PrecommitGroundingController.__new__(
            PrecommitGroundingController
        )
        controller.verifier = RemoteAlignmentVerifierBackend(
            _Client(error=RuntimeError('worker died')),
            fail_open=False,
        )

        with self.assertRaises(VerifierFailClosedError) as raised:
            controller._verify_alignment(_request(str(self.image_path)))

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertTrue(raised.exception.metadata['remote_failure'])

    def test_real_persistent_client_ping_and_shutdown(self):
        project_root = Path(__file__).resolve().parents[3]
        client = PersistentJsonlWorkerClient(
            [
                sys.executable,
                '-u',
                '-m',
                'grounding_control.workers.dino_verifier',
            ],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
        process = client.process
        try:
            response = client.ping()
            self.assertTrue(response['ok'])
            self.assertFalse(response['configured'])
        finally:
            client.close()
        self.assertIsNotNone(process)
        self.assertEqual(process.returncode, 0)

    def test_binary_qwen_worker_ping_and_shutdown(self):
        project_root = Path(__file__).resolve().parents[3]
        client = PersistentJsonlWorkerClient(
            [
                sys.executable,
                '-u',
                '-m',
                'grounding_control.workers.qwen_verifier',
            ],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
        process = client.process
        try:
            response = client.ping()
            self.assertTrue(response['ok'])
            self.assertEqual(
                response['worker'],
                'qwen25_vl_alignment_verifier',
            )
            self.assertFalse(response['configured'])
        finally:
            client.close()
        self.assertIsNotNone(process)
        self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
