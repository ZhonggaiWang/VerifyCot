"""Remote canonical verifier backed by a persistent JSONL worker."""

from pathlib import Path
from typing import Any, Mapping, Protocol

from ..contracts import (
    ActionVerifierBackend,
    ActionVerifierOutput,
    VerificationRequest,
)
from ..coordinates import COORDINATE_SYSTEM


class WorkerClient(Protocol):
    def request(
            self,
            payload: Mapping[str, Any],
            *,
            timeout=None) -> Mapping[str, Any]:
        ...


class RemoteActionVerifierBackend(ActionVerifierBackend):
    """Send online candidates to a role-specific verifier worker."""

    def __init__(
            self,
            client: WorkerClient,
            *,
            timeout: float = 300.0,
            fail_open: bool = True):
        if float(timeout) <= 0:
            raise ValueError('timeout must be positive')
        self.client = client
        self.timeout = float(timeout)
        self.fail_open = bool(fail_open)

    @staticmethod
    def _image_path(request: VerificationRequest) -> str:
        value = request.sample_context.get('image_path')
        if not isinstance(value, (str, Path)):
            raise ValueError(
                'remote verifier requires '
                'sample_context["image_path"]'
            )
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    def verify_action(
            self,
            request: VerificationRequest,
    ) -> ActionVerifierOutput:
        try:
            response = self.client.request({
                'operation': 'verify',
                'image_path': self._image_path(request),
                'sample_id': request.sample_id,
                'grounding_step': request.grounding_step,
                'object_reference': request.object_reference,
                'candidate_bbox': list(request.candidate_bbox),
                'coordinate_system': COORDINATE_SYSTEM,
            }, timeout=self.timeout)
            output = ActionVerifierOutput.from_dict(response)
            metadata = {
                **dict(output.metadata),
                'transport': 'persistent_jsonl_worker',
                'remote_request_id': response.get('request_id'),
                'remote_verifier_mode': response.get('verifier_mode'),
            }
            return ActionVerifierOutput(
                predicted_action=output.predicted_action,
                action_probabilities=output.action_probabilities,
                confidence=output.confidence,
                abstained=output.abstained,
                error=output.error,
                metadata=metadata,
            )
        except Exception as error:
            if not self.fail_open:
                raise
            return ActionVerifierOutput.unknown(
                error=f'{type(error).__name__}: {error}',
                confidence=0.0,
                metadata={
                    'transport': 'persistent_jsonl_worker',
                    'probability_source': 'unavailable_remote_failure',
                    'remote_failure': True,
                },
            )
