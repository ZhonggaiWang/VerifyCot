"""CPU-only protocol tests for the persistent Qwen/DINO worker."""

from io import StringIO
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from grounding_control.models.grounding_dino import GroundingDinoDetection
from grounding_control.four_way.workers.qwen_verifier_dino_grounder import (
    PROTOCOL_NAME,
    RESPONSE_PREFIX,
    QwenFourWayVerifierDinoGrounderWorkerEngine,
    process_request_line,
    serve_jsonl,
)


QwenVerifierDinoGrounderWorkerEngine = (
    QwenFourWayVerifierDinoGrounderWorkerEngine
)


class _FakeQwenRunner:
    min_pixels = 4 * 28 * 28
    max_pixels = 512 * 28 * 28

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages):
        # This output must be redirected away from the JSON protocol stream.
        print('fake model stdout')
        self.calls.append(messages)
        return self.response


class _FakeDinoRunner:
    model_path = 'fake-dino'
    box_threshold = 0.3
    text_threshold = 0.25

    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = []
        self.last_run_metadata = {}

    def detect(self, image, object_reference):
        self.calls.append((image.size, object_reference))
        self.last_run_metadata = {'timing_ms': {'total': 2.5}}
        return list(self.detections)


class QwenDinoWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / 'image.png'
        Image.new('RGB', (100, 80), 'white').save(self.image_path)

    @staticmethod
    def _base_request(operation, request_id='request-1'):
        return {
            'protocol': PROTOCOL_NAME,
            'request_id': request_id,
            'operation': operation,
        }

    def test_ping_does_not_require_models(self):
        engine = QwenVerifierDinoGrounderWorkerEngine()
        response = engine.handle(self._base_request('ping'))
        self.assertEqual(response['worker'], 'qwen_dino')
        self.assertFalse(response['qwen_configured'])
        self.assertFalse(response['dino_configured'])

    def test_binary_and_routing_modes_map_to_actions(self):
        binary = QwenVerifierDinoGrounderWorkerEngine(
            qwen_runner=_FakeQwenRunner(
                '{"aligned":false,"confidence":0.8}'
            )
        )
        request = {
            **self._base_request('verify'),
            'image_path': str(self.image_path),
            'object_reference': 'object',
            'candidate_bbox': [0.1, 0.2, 0.4, 0.6],
            'verifier_mode': 'binary_alignment',
            'image_mode': 'crop_only',
        }
        response = binary.handle(request)
        self.assertEqual(response['routing_action'], 'relocate')
        self.assertEqual(response['verdict'], 'misaligned')
        self.assertEqual(
            response['verifier_output_schema'],
            'vocot_alignment_score_v1',
        )
        self.assertAlmostEqual(response['alignment_score'], 0.2)
        self.assertEqual(response['aligned'], False)

        routing = QwenVerifierDinoGrounderWorkerEngine(
            qwen_runner=_FakeQwenRunner(
                '{"status":"expand","confidence":0.85}'
            )
        )
        request['verifier_mode'] = 'routing_four_way'
        response = routing.handle(request)
        self.assertEqual(response['routing_action'], 'expand')
        self.assertEqual(response['predicted_action'], 'expand')
        self.assertEqual(response['reason'], 'partial_coverage')

    def test_missing_verifier_mode_uses_routing_four_way_default(self):
        runner = _FakeQwenRunner(
            '{"status":"tighten","confidence":0.86}'
        )
        engine = QwenVerifierDinoGrounderWorkerEngine(qwen_runner=runner)

        response = engine.handle({
            **self._base_request('verify'),
            'image_path': str(self.image_path),
            'object_reference': 'object',
            'candidate_bbox': [0.1, 0.2, 0.4, 0.6],
        })

        self.assertEqual(response['verifier_mode'], 'routing_four_way')
        self.assertEqual(response['predicted_action'], 'tighten')
        self.assertEqual(response['routing_action'], 'tighten')
        self.assertEqual(
            response['metadata']['backend'],
            'qwen25_vl_routing_four_way_bbox_image_only',
        )
        self.assertEqual(response['metadata']['image_mode'], 'bbox_image_only')
        self.assertEqual(len(runner.calls), 1)

    def test_qwen_verifier_rejects_wrong_candidate_coordinate_system(self):
        engine = QwenVerifierDinoGrounderWorkerEngine(
            qwen_runner=_FakeQwenRunner(
                '{"aligned":true,"confidence":0.9}'
            )
        )
        with self.assertRaisesRegex(
                ValueError,
                'requires candidate_bbox'):
            engine.handle({
                **self._base_request('verify'),
                'image_path': str(self.image_path),
                'object_reference': 'object',
                'candidate_bbox': [0.1, 0.2, 0.4, 0.6],
                'coordinate_system': 'normalized_xyxy_on_raw_image',
                'verifier_mode': 'binary_alignment',
            })

    def test_binary_abstention_has_no_contradictory_legacy_action(self):
        engine = QwenVerifierDinoGrounderWorkerEngine(
            qwen_runner=_FakeQwenRunner(
                '{"aligned":false,"confidence":0.2}'
            )
        )
        response = engine.handle({
            **self._base_request('verify'),
            'image_path': str(self.image_path),
            'object_reference': 'object',
            'candidate_bbox': [0.1, 0.2, 0.4, 0.6],
            'verifier_mode': 'binary_alignment',
            'image_mode': 'crop_only',
        })
        self.assertTrue(response['abstained'])
        self.assertIsNone(response['aligned'])
        self.assertIsNone(response['routing_action'])
        self.assertIsNone(response['confidence'])

    def test_ground_selects_highest_score_and_clips_to_original_image(self):
        dino = _FakeDinoRunner([
            GroundingDinoDetection(
                (10.0, 10.0, 30.0, 30.0), 0.4, 'low'
            ),
            GroundingDinoDetection(
                (-2.0, 5.0, 105.0, 70.0), 0.9, 'high'
            ),
        ])
        engine = QwenVerifierDinoGrounderWorkerEngine(dino_runner=dino)
        response = engine.handle({
            **self._base_request('ground'),
            'image_path': str(self.image_path),
            'object_reference': 'object',
        })
        self.assertEqual(
            response['bbox'],
            [0.0, 5.0, 100.0, 70.0],
        )
        self.assertTrue(response['available'])
        self.assertEqual(response['source'], 'grounding_dino_grounder')
        self.assertEqual(response['confidence'], 0.9)
        legacy = response['metadata']['legacy_dino_endpoint_v0']
        self.assertEqual(legacy['score'], 0.9)
        self.assertEqual(legacy['label'], 'high')
        self.assertEqual(legacy['selected_detection_index'], 1)
        self.assertEqual(legacy['detection_count'], 2)
        self.assertEqual(dino.calls, [((100, 80), 'object')])

    def test_no_detection_returns_error_without_stopping_worker(self):
        engine = QwenVerifierDinoGrounderWorkerEngine(
            dino_runner=_FakeDinoRunner([])
        )
        request = {
            **self._base_request('ground'),
            'image_path': str(self.image_path),
            'object_reference': 'object',
        }
        response = process_request_line(
            json.dumps(request),
            engine,
            StringIO(),
        )
        self.assertTrue(response['ok'])
        self.assertFalse(response['available'])
        self.assertIsNone(response['bbox'])
        self.assertEqual(
            response['error'],
            'no_valid_grounding_detection',
        )

        ping = process_request_line(
            json.dumps(self._base_request('ping', 'after-error')),
            engine,
            StringIO(),
        )
        self.assertTrue(ping['ok'])
        self.assertEqual(ping['request_id'], 'after-error')

    def test_jsonl_server_prefixes_responses_and_honors_shutdown(self):
        lines = [
            json.dumps(self._base_request('ping', 'ping-1')) + '\n',
            json.dumps(self._base_request('shutdown', 'stop-1')) + '\n',
            json.dumps(self._base_request('ping', 'never-read')) + '\n',
        ]
        stdout = StringIO()
        exit_code = serve_jsonl(
            QwenVerifierDinoGrounderWorkerEngine(),
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

    def test_invalid_json_is_a_structured_error(self):
        response = process_request_line(
            '{not json}',
            QwenVerifierDinoGrounderWorkerEngine(),
            StringIO(),
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error_type'], 'JSONDecodeError')
        self.assertIsNone(response['request_id'])

    def test_refiner_operation_is_reserved_but_not_faked(self):
        response = process_request_line(
            json.dumps(self._base_request('refine')),
            QwenVerifierDinoGrounderWorkerEngine(),
            StringIO(),
        )
        self.assertFalse(response['ok'])
        self.assertEqual(response['error'], 'box_refiner_not_configured')


if __name__ == '__main__':
    unittest.main()
