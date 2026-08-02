"""CPU-only protocol tests for the dedicated Qwen2.5-VL Grounder."""

from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from grounding_control.workers.endpoints.qwen_grounder import (
    ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
)
from grounding_control.workers.qwen_grounder import (
    PROTOCOL_NAME,
    RESPONSE_PREFIX,
    QwenGrounderWorkerEngine,
)
from grounding_control.transport import (
    GROUNDER_OUTPUT_SCHEMA,
    process_request_line,
    serve_jsonl,
)


class _FakeQwenRunner:
    min_pixels = 4 * 28 * 28
    max_pixels = 512 * 28 * 28

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages):
        # Model libraries sometimes print diagnostics.  The runtime must keep
        # them away from the machine-readable stdout stream.
        print('fake qwen model stdout')
        self.calls.append(messages)
        return self.response


class QwenGrounderWorkerTests(unittest.TestCase):
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

    def test_ping_is_available_without_loading_a_model(self):
        response = QwenGrounderWorkerEngine().handle(
            self._request('ping')
        )

        self.assertEqual(response['worker'], 'qwen25_vl_grounder')
        self.assertFalse(response['configured'])
        self.assertEqual(
            response['output_coordinate_system'],
            ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
        )

    def test_ground_returns_original_image_pixel_box_without_vocot_padding(self):
        # A 100x80 source is smart-resized to 112x84.  A full resized-image
        # prediction must come back as a full 100x80 original-image box, not
        # as normalized square-padding coordinates.
        runner = _FakeQwenRunner(
            '{"bbox_2d":[0,0,112,84],"label":"tissue box"}'
        )
        engine = QwenGrounderWorkerEngine(
            runner=runner,
            prompt_protocol='single_object_json_v2',
        )

        response = engine.handle({
            **self._request('ground'),
            'image_path': str(self.image_path),
            'sample_id': 'main:9',
            'grounding_step': 1,
            'object_reference': 'the tissue box',
        })

        self.assertTrue(response['available'])
        self.assertEqual(
            response['grounder_output_schema'],
            GROUNDER_OUTPUT_SCHEMA,
        )
        self.assertEqual(response['source'], 'qwen25_vl_grounder')
        self.assertEqual(
            response['coordinate_system'],
            ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
        )
        self.assertEqual(
            response['bbox'],
            [0.0, 0.0, 100.0, 80.0],
        )
        self.assertEqual(response['image_size'], [100, 80])
        self.assertIsNone(response['confidence'])
        self.assertIsNone(response['error'])
        self.assertEqual(
            response['metadata']['model_image_size'],
            [112, 84],
        )
        self.assertEqual(
            response['metadata']['prompt_protocol'],
            'single_object_json_v2',
        )
        self.assertEqual(
            response['metadata']['raw_response'],
            runner.response,
        )
        self.assertEqual(
            response['metadata']['predicted_box_original_pixel_xyxy'],
            [0.0, 0.0, 100.0, 80.0],
        )
        self.assertNotIn('candidate_bbox', str(runner.calls[0]))
        self.assertIn('the tissue box', str(runner.calls[0]))

    def test_parse_failure_is_available_false_and_preserves_raw_metadata(self):
        runner = _FakeQwenRunner('I cannot find a single object.')
        engine = QwenGrounderWorkerEngine(runner=runner)
        request = {
            **self._request('ground'),
            'image_path': str(self.image_path),
            'sample_id': 'sample',
            'grounding_step': 2,
            'object_reference': 'the current object',
        }

        response = process_request_line(
            json.dumps(request),
            engine,
            StringIO(),
        )

        # The transport request succeeded; the expert is explicitly
        # unavailable so a remote backend can choose fail-open or fail-closed.
        self.assertTrue(response['ok'])
        self.assertFalse(response['available'])
        self.assertIsNone(response['bbox'])
        self.assertEqual(
            response['coordinate_system'],
            ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
        )
        self.assertEqual(
            response['metadata']['raw_response'],
            runner.response,
        )
        self.assertTrue(response['metadata']['parse_failed'])
        self.assertIn('expected exactly one grounding bbox', response['error'])

    def test_malformed_request_is_protocol_error_not_model_unavailable(self):
        engine = QwenGrounderWorkerEngine(
            runner=_FakeQwenRunner('{"bbox_2d":[0,0,10,10]}')
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
            QwenGrounderWorkerEngine(),
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


if __name__ == '__main__':
    unittest.main()
