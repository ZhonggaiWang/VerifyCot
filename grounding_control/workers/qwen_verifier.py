"""Persistent binary Qwen2.5-VL alignment-verifier worker."""

import argparse
from typing import Any, Dict, Mapping, Optional

from grounding_control.models.qwen25_vl import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLRunner,
)
from grounding_control.transport import (
    DEFAULT_PROTOCOL_NAME,
    DEFAULT_RESPONSE_PREFIX,
    WorkerRequestError,
    serve_jsonl,
)
from grounding_control.transport.request_io import required_string
from grounding_control.verifiers.qwen25_vl.classifier import (
    Qwen25VLBinaryAlignmentClassifier,
)
from grounding_control.workers.endpoints.qwen_alignment_verifier import (
    QwenAlignmentVerifierEndpoint,
)


PROTOCOL_NAME = DEFAULT_PROTOCOL_NAME
RESPONSE_PREFIX = DEFAULT_RESPONSE_PREFIX


class QwenVerifierWorkerEngine:
    """Host one Qwen binary alignment classifier without four-way imports."""

    def __init__(
            self,
            runner: Optional[Qwen25VLRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            max_new_tokens: int = 64,
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: int = DEFAULT_MAX_PIXELS,
            attn_implementation: str = 'sdpa',
            crop_min_side: int = 56,
            default_image_mode: str = 'bbox_image_only'):
        self.model_path = None if model_path is None else str(model_path)
        if runner is None and self.model_path is not None:
            runner = LocalQwen25VLRunner(
                model_path=self.model_path,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                attn_implementation=attn_implementation,
            )
        self.endpoint = (
            None
            if runner is None
            else QwenAlignmentVerifierEndpoint(
                Qwen25VLBinaryAlignmentClassifier(
                    runner,
                    crop_min_side=crop_min_side,
                    parse_fail_open=True,
                ),
                default_image_mode=default_image_mode,
            )
        )
        self.device = str(device)
        self.dtype = str(dtype)
        self.default_image_mode = str(default_image_mode)

    def _ping(self) -> Dict[str, Any]:
        return {
            'protocol': PROTOCOL_NAME,
            'worker': 'qwen25_vl_alignment_verifier',
            'configured': self.endpoint is not None,
            'model_path': self.model_path,
            'device': self.device,
            'dtype': self.dtype,
            'verifier_mode': 'binary_alignment',
            'default_image_mode': self.default_image_mode,
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
                    'Qwen alignment verifier is not configured'
                )
            return self.endpoint.handle(payload)
        if operation == 'shutdown':
            return {'shutdown': True}
        raise WorkerRequestError(f'unsupported operation: {operation!r}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--min-pixels', type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument('--attn-implementation', default='sdpa')
    parser.add_argument('--crop-min-side', type=int, default=56)
    parser.add_argument('--default-image-mode', default='bbox_image_only')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = QwenVerifierWorkerEngine(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        crop_min_side=args.crop_min_side,
        default_image_mode=args.default_image_mode,
    )
    return serve_jsonl(
        engine,
        protocol_name=PROTOCOL_NAME,
        response_prefix=RESPONSE_PREFIX,
    )


__all__ = [
    'PROTOCOL_NAME',
    'QwenVerifierWorkerEngine',
    'RESPONSE_PREFIX',
    'main',
    'parse_args',
]


if __name__ == '__main__':
    raise SystemExit(main())
