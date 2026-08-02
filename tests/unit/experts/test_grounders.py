"""CPU-only tests for reusable box predictors and Grounder adapters."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from grounding_control.contracts import GroundingRequest, VisualInput
from grounding_control.coordinates import (
    original_pixel_box_to_normalized_square_box,
)
from grounding_control.core import ExpertUnavailableError
from grounding_control.experts.grounders import (
    GroundingDinoGrounderBackend,
    RemoteGrounderBackend,
)
from grounding_control.models import BoxPredictionRequest
from grounding_control.models.grounding_dino import (
    GroundingDinoBoxPredictor,
    GroundingDinoDetection,
)
from grounding_control.models.qwen25_vl import Qwen25VLBoxPredictor
from grounding_control.transport import (
    GROUNDER_OUTPUT_SCHEMA,
    serialize_grounder_output,
)
from grounding_control.workers.endpoints import DinoGrounderEndpoint


class _DinoRunner:
    model_path = 'fake'
    box_threshold = 0.3
    text_threshold = 0.25
    last_run_metadata = {}

    def detect(self, image, object_reference):
        return [
            GroundingDinoDetection(
                box_original_pixel_xyxy=(20, 10, 80, 40),
                score=0.9,
                label='cup',
            )
        ]


class _EmptyDinoRunner(_DinoRunner):
    def detect(self, image, object_reference):
        del image, object_reference
        return []


class _QwenRunner:
    min_pixels = 1
    max_pixels = 100000

    def generate(self, messages):
        return (
            '{"bbox_2d":[11.2,5.6,56,28],'
            '"label":"the target"}'
        )


class _RemoteClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.payload = None
        self.timeout = None

    def request(self, payload, *, timeout=None):
        self.payload = dict(payload)
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return dict(self.response)


class _EndpointClient:
    """Exercise the real endpoint response through the remote adapter."""

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.payload = None

    def request(self, payload, *, timeout=None):
        del timeout
        self.payload = dict(payload)
        return {
            'request_id': 'endpoint-request-1',
            **self.endpoint.handle(payload),
        }


def _remote_request(image_path):
    return GroundingRequest(
        sample_id='sample-1',
        grounding_step=3,
        object_reference='the red cup',
        visual=VisualInput(image_path=str(image_path)),
    )


def _wire_response(**overrides):
    values = {
        'available': True,
        'source': 'qwen25_vl_grounder',
        'bbox': (20.0, 10.0, 80.0, 40.0),
        'image_size': (100, 50),
        'confidence': None,
        'error': None,
        'metadata': {'model': 'fake-qwen'},
    }
    values.update(overrides)
    return serialize_grounder_output(**values)


class ExpertGrounderTests(unittest.TestCase):
    def test_dino_grounder_converts_original_pixels_to_vocot_padding(self):
        image = Image.new('RGB', (100, 50), 'white')
        request = GroundingRequest(
            sample_id='s',
            grounding_step=1,
            object_reference='cup',
            visual=VisualInput(image=image),
        )
        result = GroundingDinoGrounderBackend(_DinoRunner()).ground(request)
        expected = original_pixel_box_to_normalized_square_box(
            (20, 10, 80, 40),
            100,
            50,
        )
        self.assertEqual(result.bbox, expected)
        self.assertEqual(result.source, 'grounding_dino_grounder')
        self.assertEqual(result.confidence, 0.9)

    def test_dino_no_detection_is_reported_as_expert_unavailable(self):
        request = GroundingRequest(
            sample_id='s',
            grounding_step=1,
            object_reference='cup',
            visual=VisualInput(
                image=Image.new('RGB', (100, 50), 'white'),
            ),
        )
        with self.assertRaisesRegex(
                ExpertUnavailableError,
                'no_valid_grounding_detection') as raised:
            GroundingDinoGrounderBackend(
                _EmptyDinoRunner()
            ).ground(request)
        self.assertEqual(
            raised.exception.metadata['prediction_error'],
            'no_valid_grounding_detection',
        )

    def test_qwen_predictor_returns_original_image_pixel_frame(self):
        prediction = Qwen25VLBoxPredictor(_QwenRunner()).predict(
            BoxPredictionRequest(
                image=Image.new('RGB', (100, 50), 'white'),
                object_reference='the target',
            )
        )
        self.assertIsNone(prediction.error)
        self.assertEqual(
            tuple(round(value, 6) for value in prediction.bbox_pixel_xyxy),
            (10.0, 5.0, 50.0, 25.0),
        )

    def test_remote_grounder_converts_validated_original_pixel_box(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            response = _wire_response(metadata={
                'model': 'fake-qwen',
                'grounder_mode': 'qwen25_vl',
                'raw_response': '{"bbox_2d":[20,10,80,40]}',
            })
            response.update({
                'request_id': 'remote-1',
            })
            client = _RemoteClient(response)
            result = RemoteGrounderBackend(
                client,
                timeout=12.5,
                source='qwen25_vl_grounder',
            ).ground(_remote_request(image_path))

        self.assertEqual(
            result.bbox,
            original_pixel_box_to_normalized_square_box(
                (20.0, 10.0, 80.0, 40.0),
                100,
                50,
            ),
        )
        self.assertEqual(result.source, 'qwen25_vl_grounder')
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(
            result.metadata['prediction_confidence_available']
        )
        self.assertEqual(result.metadata['remote_request_id'], 'remote-1')
        self.assertEqual(
            result.metadata['grounder_output_schema'],
            GROUNDER_OUTPUT_SCHEMA,
        )
        self.assertEqual(client.timeout, 12.5)
        self.assertEqual(client.payload, {
            'operation': 'ground',
            'image_path': str(image_path.resolve()),
            'sample_id': 'sample-1',
            'grounding_step': 3,
            'object_reference': 'the red cup',
        })
        self.assertNotIn('candidate_bbox', client.payload)

    def test_remote_grounder_rejects_worker_level_unavailability(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            client = _RemoteClient(_wire_response(
                available=False,
                bbox=None,
                confidence=None,
                error='coordinate_parse_failed',
                metadata={
                    'parse_failed': True,
                    'raw_response': 'not valid coordinate JSON',
                },
            ))
            with self.assertRaisesRegex(
                    ExpertUnavailableError,
                    'coordinate_parse_failed') as raised:
                RemoteGrounderBackend(client).ground(
                    _remote_request(image_path),
                )
        self.assertEqual(
            raised.exception.metadata['remote_raw_response'],
            'not valid coordinate JSON',
        )
        self.assertTrue(
            raised.exception.metadata['remote_metadata']['parse_failed']
        )

    def test_remote_grounder_wraps_transport_failure_as_unavailable(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            client = _RemoteClient(error=RuntimeError('worker died'))
            with self.assertRaisesRegex(
                    ExpertUnavailableError,
                    'worker died'):
                RemoteGrounderBackend(client).ground(
                    _remote_request(image_path),
                )

    def test_remote_grounder_rejects_coordinate_system_mismatch(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            client = _RemoteClient({
                'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
                'available': True,
                'source': 'qwen25_vl_grounder',
                'coordinate_system': 'normalized_xyxy',
                'bbox': [20.0, 10.0, 80.0, 40.0],
                'image_size': [100, 50],
                'confidence': 0.8,
                'error': None,
                'metadata': {},
            })
            with self.assertRaisesRegex(
                    ExpertUnavailableError,
                    'unsupported coordinate_system'):
                RemoteGrounderBackend(client).ground(
                    _remote_request(image_path),
                )

    def test_remote_grounder_rejects_remote_image_size_mismatch(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            client = _RemoteClient(_wire_response(
                image_size=(99, 50),
                bbox=(20.0, 10.0, 80.0, 40.0),
                confidence=0.8,
            ))
            with self.assertRaisesRegex(
                    ExpertUnavailableError,
                    'image_size mismatch'):
                RemoteGrounderBackend(client).ground(
                    _remote_request(image_path),
                )

    def test_remote_grounder_rejects_out_of_bounds_pixel_box(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            client = _RemoteClient({
                'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
                'available': True,
                'source': 'qwen25_vl_grounder',
                'coordinate_system': 'absolute_xyxy_on_original_image',
                'bbox': [-1.0, 10.0, 80.0, 40.0],
                'image_size': [100, 50],
                'confidence': 0.8,
                'error': None,
                'metadata': {},
            })
            with self.assertRaisesRegex(
                    ExpertUnavailableError,
                    'outside image'):
                RemoteGrounderBackend(client).ground(
                    _remote_request(image_path),
                )

    def test_actual_dino_endpoint_response_is_consumed_by_remote_backend(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            endpoint = DinoGrounderEndpoint(
                GroundingDinoBoxPredictor(_DinoRunner())
            )
            client = _EndpointClient(endpoint)
            result = RemoteGrounderBackend(client).ground(
                _remote_request(image_path)
            )

        self.assertEqual(result.source, 'grounding_dino_grounder')
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(
            result.bbox,
            original_pixel_box_to_normalized_square_box(
                (20.0, 10.0, 80.0, 40.0),
                100,
                50,
            ),
        )
        legacy = result.metadata['legacy_dino_endpoint_v0']
        self.assertEqual(legacy['selected_detection_index'], 0)
        self.assertEqual(legacy['label'], 'cup')

    def test_actual_dino_unavailable_response_is_consumed_by_remote_backend(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / 'source.png'
            Image.new('RGB', (100, 50), 'white').save(image_path)
            endpoint = DinoGrounderEndpoint(
                GroundingDinoBoxPredictor(_EmptyDinoRunner())
            )
            client = _EndpointClient(endpoint)
            with self.assertRaisesRegex(
                    ExpertUnavailableError,
                    'no_valid_grounding_detection') as raised:
                RemoteGrounderBackend(client).ground(
                    _remote_request(image_path)
                )

        self.assertNotIn('remote_failure', raised.exception.metadata)
        self.assertTrue(
            raised.exception.metadata['remote_metadata'][
                'localization_failed'
            ]
        )

if __name__ == '__main__':
    unittest.main()
