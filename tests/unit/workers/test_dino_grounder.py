"""CPU-only protocol tests for the dedicated Grounding DINO Grounder."""

from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from grounding_control.models.grounding_dino import GroundingDinoDetection
from grounding_control.transport import (
    GROUNDER_OUTPUT_SCHEMA,
    ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
    PersistentJsonlWorkerClient,
    process_request_line,
    serve_jsonl,
)
from grounding_control.workers.dino_grounder import (
    PROTOCOL_NAME,
    RESPONSE_PREFIX,
    DinoGrounderWorkerEngine,
)


class _FakeDinoRunner:
    model_path = 'fake-dino'
    box_threshold = 0.3
    text_threshold = 0.25

    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = []
        self.last_run_metadata = {}

    def detect(self, image, object_reference):
        self.calls.append({
            'image_size': image.size,
            'object_reference': object_reference,
        })
        self.last_run_metadata = {
            'grounding_query': object_reference + '.',
            'timing_ms': {'total': 1.0},
        }
        return list(self.detections)


class DinoGrounderWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / 'image.png'
        Image.new('RGB', (100, 80), 'white').save(self.image_path)

    @staticmethod
    def _request(operation, request_id='request-1'):
        return {
            'protocol': PROTOCOL_NAME,
            'request_id': request_id,
            'operation': operation,
        }

    @staticmethod
    def _detection():
        return GroundingDinoDetection(
            box_original_pixel_xyxy=(20.0, 10.0, 80.0, 60.0),
            score=0.91,
            label='cup',
        )

    def test_ping_is_available_without_loading_a_model(self):
        response = DinoGrounderWorkerEngine().handle(
            self._request('ping')
        )

        self.assertEqual(response['worker'], 'grounding_dino_grounder')
        self.assertFalse(response['configured'])
        self.assertEqual(response['box_threshold'], 0.3)
        self.assertEqual(response['text_threshold'], 0.25)
        self.assertEqual(
            response['output_coordinate_system'],
            ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
        )

    def test_ground_preserves_existing_original_pixel_wire_output(self):
        runner = _FakeDinoRunner([self._detection()])
        engine = DinoGrounderWorkerEngine(runner=runner, top_k_log=7)

        response = engine.handle({
            **self._request('ground'),
            'image_path': str(self.image_path),
            'sample_id': 'sample-1',
            'grounding_step': 2,
            'object_reference': 'the cup',
        })

        self.assertTrue(response['available'])
        self.assertEqual(
            response['grounder_output_schema'],
            GROUNDER_OUTPUT_SCHEMA,
        )
        self.assertEqual(response['source'], 'grounding_dino_grounder')
        self.assertEqual(
            response['coordinate_system'],
            ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
        )
        self.assertEqual(response['bbox'], [20.0, 10.0, 80.0, 60.0])
        self.assertEqual(response['image_size'], [100, 80])
        self.assertEqual(response['confidence'], 0.91)
        self.assertIsNone(response['error'])
        self.assertEqual(runner.calls, [{
            'image_size': (100, 80),
            'object_reference': 'the cup',
        }])
        legacy = response['metadata']['legacy_dino_endpoint_v0']
        self.assertEqual(legacy['selected_detection_index'], 0)
        self.assertEqual(legacy['selection_policy'], 'highest_detector_score')
        self.assertEqual(legacy['label'], 'cup')

    def test_no_detection_is_explicit_expert_unavailability(self):
        engine = DinoGrounderWorkerEngine(
            runner=_FakeDinoRunner([])
        )
        response = process_request_line(
            json.dumps({
                **self._request('ground'),
                'image_path': str(self.image_path),
                'object_reference': 'the missing cup',
            }),
            engine,
            StringIO(),
        )

        self.assertTrue(response['ok'])
        self.assertFalse(response['available'])
        self.assertIsNone(response['bbox'])
        self.assertIsNone(response['confidence'])
        self.assertEqual(response['error'], 'no_valid_grounding_detection')
        self.assertTrue(response['metadata']['localization_failed'])

    def test_malformed_request_is_protocol_error(self):
        engine = DinoGrounderWorkerEngine(
            runner=_FakeDinoRunner([self._detection()])
        )
        response = process_request_line(
            json.dumps({
                **self._request('ground'),
                'image_path': str(self.image_path),
            }),
            engine,
            StringIO(),
        )

        self.assertFalse(response['ok'])
        self.assertIn('object_reference', response['error'])

    def test_jsonl_server_prefixes_output_and_honors_shutdown(self):
        lines = [
            json.dumps(self._request('ping', 'ping-1')) + '\n',
            json.dumps(self._request('shutdown', 'stop-1')) + '\n',
            json.dumps(self._request('ping', 'never-read')) + '\n',
        ]
        stdout = StringIO()

        exit_code = serve_jsonl(
            DinoGrounderWorkerEngine(),
            stdin=lines,
            stdout=stdout,
            stderr=StringIO(),
        )

        self.assertEqual(exit_code, 0)
        output_lines = stdout.getvalue().splitlines()
        self.assertEqual(len(output_lines), 2)
        self.assertTrue(all(
            line.startswith(RESPONSE_PREFIX)
            for line in output_lines
        ))
        payloads = [
            json.loads(line[len(RESPONSE_PREFIX):])
            for line in output_lines
        ]
        self.assertEqual(payloads[0]['request_id'], 'ping-1')
        self.assertTrue(payloads[1]['shutdown'])

    def test_canonical_module_starts_as_persistent_worker(self):
        project_root = Path(__file__).resolve().parents[3]
        client = PersistentJsonlWorkerClient(
            [
                sys.executable,
                '-u',
                '-m',
                'grounding_control.workers.dino_grounder',
            ],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
        process = client.process
        try:
            response = client.ping()
            self.assertTrue(response['ok'])
            self.assertEqual(response['worker'], 'grounding_dino_grounder')
            self.assertFalse(response['configured'])
        finally:
            client.close()
        self.assertIsNotNone(process)
        self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
