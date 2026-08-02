"""Remote binary alignment verifier backed by a persistent JSONL worker."""

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from ..contracts.alignment_verifier import (
    ALIGNMENT_OUTPUT_SCHEMA,
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
)
from ..contracts.errors import VerifierFailClosedError
from ..contracts.requests import CandidateAlignmentRequest
from ..coordinates import COORDINATE_SYSTEM


class WorkerClient(Protocol):
    """Minimal request interface shared by remote verifier adapters."""

    def request(
            self,
            payload: Mapping[str, Any],
            *,
            timeout=None) -> Mapping[str, Any]:
        ...


class RemoteAlignmentVerifierBackend(AlignmentVerifierBackend):
    """Request the binary wire schema without relying on worker defaults."""

    def __init__(
            self,
            client: WorkerClient,
            *,
            timeout: float = 300.0,
            fail_open: bool = True,
            image_mode: Optional[str] = None):
        if float(timeout) <= 0:
            raise ValueError('timeout must be positive')
        self.client = client
        self.timeout = float(timeout)
        self.fail_open = bool(fail_open)
        self.image_mode = None if image_mode is None else str(image_mode)

    @staticmethod
    def _image_path(request: CandidateAlignmentRequest) -> str:
        value = request.visual.image_path
        if not isinstance(value, (str, Path)):
            raise ValueError(
                'remote verifier requires visual.image_path'
            )
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    def verify_alignment(
            self,
            request: CandidateAlignmentRequest,
    ) -> AlignmentVerifierOutput:
        if not isinstance(request, CandidateAlignmentRequest):
            raise TypeError(
                'Remote alignment verifier requires CandidateAlignmentRequest'
            )
        try:
            payload = {
                'operation': 'verify',
                # Explicitly override the legacy four-way worker default.
                'verifier_mode': 'binary_alignment',
                'image_path': self._image_path(request),
                'sample_id': request.sample_id,
                'grounding_step': request.grounding_step,
                'object_reference': request.object_reference,
                'candidate_bbox': list(request.candidate_bbox),
                'coordinate_system': COORDINATE_SYSTEM,
            }
            image_mode = request.image_mode or self.image_mode
            if image_mode is not None:
                payload['image_mode'] = str(image_mode)
            response = self.client.request(
                payload,
                timeout=self.timeout,
            )
            response_mode = response.get('verifier_mode')
            if response_mode != 'binary_alignment':
                raise ValueError(
                    'remote worker must return verifier_mode '
                    '"binary_alignment", got '
                    f'{response_mode!r}'
                )
            response_schema = response.get('verifier_output_schema')
            if response_schema != ALIGNMENT_OUTPUT_SCHEMA:
                raise ValueError(
                    'remote worker must return verifier_output_schema '
                    f'{ALIGNMENT_OUTPUT_SCHEMA!r}, got {response_schema!r}'
                )
            output = AlignmentVerifierOutput.from_dict(response)
            metadata = {
                **dict(output.metadata),
                'transport': 'persistent_jsonl_worker',
                'remote_request_id': response.get('request_id'),
                'remote_verifier_mode': response_mode,
            }
            return AlignmentVerifierOutput(
                alignment_score=output.alignment_score,
                score_kind=output.score_kind,
                score_semantics=output.score_semantics,
                abstained=output.abstained,
                error=output.error,
                metadata=metadata,
            )
        except Exception as error:
            failure_metadata = {
                'transport': 'persistent_jsonl_worker',
                'remote_failure': True,
                'requested_verifier_mode': 'binary_alignment',
                'backend_exception_type': type(error).__name__,
            }
            if not self.fail_open:
                raise VerifierFailClosedError(
                    'remote alignment verifier failed closed: '
                    f'{type(error).__name__}: {error}',
                    metadata=failure_metadata,
                ) from error
            return AlignmentVerifierOutput.unknown(
                error=f'{type(error).__name__}: {error}',
                score_semantics='unavailable_remote_failure',
                score_kind=None,
                metadata=failure_metadata,
            )


__all__ = ['RemoteAlignmentVerifierBackend', 'WorkerClient']
