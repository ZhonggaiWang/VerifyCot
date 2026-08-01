"""Persistent role-specific Grounding DINO geometry verifier worker."""

import argparse
from typing import Any, Dict, Mapping, Optional

from verifier.models.grounding_dino import (
    GroundingDinoRunner,
    LocalGroundingDinoRunner,
)
from verifier.runtime import (
    DEFAULT_PROTOCOL_NAME,
    DEFAULT_RESPONSE_PREFIX,
    WorkerRequestError,
    serve_jsonl,
)
from verifier.runtime.request_io import required_string
from verifier.verifier_backends import GroundingDinoGeometryClassifier
from verifier.workers.endpoints import DinoGeometryVerifierEndpoint


PROTOCOL_NAME = DEFAULT_PROTOCOL_NAME
RESPONSE_PREFIX = DEFAULT_RESPONSE_PREFIX


class DinoGeometryWorkerEngine:
    """Host one DINO geometry verifier without any VoCoT dependency."""

    def __init__(
            self,
            runner: Optional[GroundingDinoRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'float32',
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
            accept_iou_threshold: float = 0.4,
            containment_threshold: float = 0.7,
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
        self.endpoint = None
        if runner is not None:
            self.endpoint = DinoGeometryVerifierEndpoint(
                GroundingDinoGeometryClassifier(
                    runner=runner,
                    accept_iou_threshold=accept_iou_threshold,
                    containment_threshold=containment_threshold,
                    top_k_log=top_k_log,
                )
            )
        self.device = str(device)
        self.dtype = str(dtype)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.accept_iou_threshold = float(accept_iou_threshold)
        self.containment_threshold = float(containment_threshold)

    def _ping(self) -> Dict[str, Any]:
        return {
            'worker': 'dino_geometry_verifier',
            'configured': self.endpoint is not None,
            'model_path': self.model_path,
            'device': self.device,
            'dtype': self.dtype,
            'box_threshold': self.box_threshold,
            'text_threshold': self.text_threshold,
            'accept_iou_threshold': self.accept_iou_threshold,
            'containment_threshold': self.containment_threshold,
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
    parser.add_argument('--accept-iou-threshold', type=float, default=0.4)
    parser.add_argument('--containment-threshold', type=float, default=0.7)
    parser.add_argument('--top-k-log', type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = DinoGeometryWorkerEngine(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        accept_iou_threshold=args.accept_iou_threshold,
        containment_threshold=args.containment_threshold,
        top_k_log=args.top_k_log,
    )
    return serve_jsonl(
        engine,
        protocol_name=PROTOCOL_NAME,
        response_prefix=RESPONSE_PREFIX,
    )


if __name__ == '__main__':
    raise SystemExit(main())
