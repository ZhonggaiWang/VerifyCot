"""Persistent binary Grounding DINO alignment-verifier worker."""

import argparse
from typing import Any, Dict, Mapping, Optional

from grounding_control.models.grounding_dino import (
    GroundingDinoRunner,
    LocalGroundingDinoRunner,
)
from grounding_control.transport import (
    DEFAULT_PROTOCOL_NAME,
    DEFAULT_RESPONSE_PREFIX,
    WorkerRequestError,
    serve_jsonl,
)
from grounding_control.transport.request_io import required_string
from grounding_control.verifiers.dino import GroundingDinoAlignmentScorer
from grounding_control.workers.endpoints.dino_verifier import (
    DinoVerifierEndpoint,
)


PROTOCOL_NAME = DEFAULT_PROTOCOL_NAME
RESPONSE_PREFIX = DEFAULT_RESPONSE_PREFIX


class DinoVerifierWorkerEngine:
    """Host one threshold-free DINO alignment scorer."""

    def __init__(
            self,
            runner: Optional[GroundingDinoRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'float32',
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
            top_k_log: int = 20):
        self.model_path = None if model_path is None else str(model_path)
        if runner is None and self.model_path is not None:
            runner = LocalGroundingDinoRunner(
                model_path=self.model_path,
                device=device,
                dtype=dtype,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        self.endpoint = (
            None
            if runner is None
            else DinoVerifierEndpoint(GroundingDinoAlignmentScorer(
                runner=runner,
                top_k_log=top_k_log,
            ))
        )
        self.device = str(device)
        self.dtype = str(dtype)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

    def _ping(self) -> Dict[str, Any]:
        return {
            # Preserve the established wire identifier; ``verifier_mode``
            # below disambiguates the now binary-only implementation.
            'worker': 'dino_geometry_verifier',
            'configured': self.endpoint is not None,
            'model_path': self.model_path,
            'device': self.device,
            'dtype': self.dtype,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold,
            'verifier_mode': 'binary_alignment',
        }

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise WorkerRequestError('request must be a JSON object')
        if payload.get('protocol') != PROTOCOL_NAME:
            raise WorkerRequestError(
                f'protocol must equal {PROTOCOL_NAME!r}'
            )
        operation = required_string(payload, 'operation')
        if operation == 'ping':
            return self._ping()
        if operation == 'verify':
            if self.endpoint is None:
                raise WorkerRequestError(
                    'Grounding DINO verifier is not configured'
                )
            return self.endpoint.handle(payload)
        if operation == 'shutdown':
            return {'shutdown': True}
        raise WorkerRequestError(f'unsupported operation: {operation!r}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--box-threshold', type=float, default=0.3)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--top-k-log', type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = DinoVerifierWorkerEngine(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        top_k_log=args.top_k_log,
    )
    return serve_jsonl(
        engine,
        protocol_name=PROTOCOL_NAME,
        response_prefix=RESPONSE_PREFIX,
    )


__all__ = [
    'DinoVerifierWorkerEngine',
    'PROTOCOL_NAME',
    'RESPONSE_PREFIX',
    'main',
    'parse_args',
]


if __name__ == '__main__':
    raise SystemExit(main())
