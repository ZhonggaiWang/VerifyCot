"""Remote relocation expert backed by a persistent JSONL worker."""

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple

from PIL import Image

from ...contracts import (
    GrounderBackend,
    GroundingRequest,
    GroundingResult,
)
from ...contracts.errors import ExpertUnavailableError
from ...models import BoxPrediction
from ...transport.grounder_wire import (
    GROUNDER_OUTPUT_SCHEMA,
    parse_grounder_output,
)
from .predictor import pixel_prediction_to_grounding_result


class WorkerClient(Protocol):
    def request(
            self,
            payload: Mapping[str, Any],
            *,
            timeout=None) -> Mapping[str, Any]:
        ...


class RemoteGrounderBackend(GrounderBackend):
    """Adapt original-image pixel boxes from a JSONL worker to VoCoT."""

    def __init__(
            self,
            client: WorkerClient,
            *,
            timeout: float = 300.0,
            source: Optional[str] = None):
        if float(timeout) <= 0:
            raise ValueError('timeout must be positive')
        normalized_source = None
        if source is not None:
            normalized_source = str(source).strip()
            if not normalized_source:
                raise ValueError('source must be a non-empty string')
        self.client = client
        self.timeout = float(timeout)
        self.expected_source = normalized_source
        # Retained for evaluator diagnostics.  When no expectation is set,
        # the canonical worker source becomes the GroundingResult source.
        self.source = normalized_source or 'remote_grounder'

    @staticmethod
    def _image_path_and_size(
            request: GroundingRequest) -> Tuple[str, Tuple[int, int]]:
        value = request.visual.image_path
        if not isinstance(value, (str, Path)):
            raise ValueError(
                'remote grounder requires request.visual.image_path'
            )
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            size = tuple(image.size)
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(f'image has invalid size: {size!r}')
        return str(path), (int(size[0]), int(size[1]))

    def ground(self, request: GroundingRequest) -> GroundingResult:
        if not isinstance(request, GroundingRequest):
            raise TypeError('Remote Grounder requires a GroundingRequest')
        try:
            image_path, local_image_size = self._image_path_and_size(request)
            response = self.client.request({
                'operation': 'ground',
                'image_path': image_path,
                'sample_id': request.sample_id,
                'grounding_step': request.grounding_step,
                'object_reference': request.object_reference,
            }, timeout=self.timeout)
            if not isinstance(response, Mapping):
                raise TypeError('remote grounder response must be a mapping')
            output = parse_grounder_output(response)
            if (
                    self.expected_source is not None
                    and output.source != self.expected_source):
                raise ValueError(
                    'remote grounder source mismatch: '
                    f'{output.source!r} != {self.expected_source!r}'
                )
            remote_metadata = dict(output.metadata)
            if not output.available:
                raise ExpertUnavailableError(
                    str(output.error),
                    metadata={
                        'transport': 'persistent_jsonl_worker',
                        'remote_request_id': response.get('request_id'),
                        'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
                        'remote_grounder_source': output.source,
                        'remote_error': output.error,
                        'remote_raw_response': remote_metadata.get(
                            'raw_response'
                        ),
                        'remote_metadata': remote_metadata,
                    },
                )
            remote_image_size = output.image_size
            if remote_image_size != local_image_size:
                raise ValueError(
                    'remote grounder image_size mismatch: '
                    f'{remote_image_size} != {local_image_size}'
                )

            assert output.bbox is not None
            prediction = BoxPrediction(
                bbox_pixel_xyxy=output.bbox,
                confidence=output.confidence,
                error=None,
                metadata=remote_metadata,
            )
            return pixel_prediction_to_grounding_result(
                prediction,
                image_width=local_image_size[0],
                image_height=local_image_size[1],
                source=output.source,
                extra_metadata={
                    'transport': 'persistent_jsonl_worker',
                    'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
                    'remote_request_id': response.get('request_id'),
                    'remote_grounder_source': output.source,
                    'remote_grounder_mode': remote_metadata.get(
                        'grounder_mode'
                    ),
                    'remote_image_size': list(remote_image_size),
                    'remote_raw_response': remote_metadata.get(
                        'raw_response'
                    ),
                },
            )
        except ExpertUnavailableError:
            raise
        except Exception as error:
            raise ExpertUnavailableError(
                f'{self.source} unavailable: '
                f'{type(error).__name__}: {error}',
                metadata={
                    'transport': 'persistent_jsonl_worker',
                    'remote_failure': True,
                    'error_type': type(error).__name__,
                },
            ) from error


__all__ = ['RemoteGrounderBackend']
