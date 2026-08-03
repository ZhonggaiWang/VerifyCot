"""Persistent role-specific Qwen2.5-VL Grounder JSONL worker."""

import argparse
from typing import Any, Dict, Mapping, Optional

from grounding_control.models.qwen25_vl import (
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLBoxPredictor,
    Qwen25VLRunner,
)
from grounding_control.models.qwen25_vl.grounding_parser import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
)
from grounding_control.models.qwen25_vl.grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
)
from grounding_control.transport import (
    DEFAULT_PROTOCOL_NAME,
    DEFAULT_RESPONSE_PREFIX,
    WorkerRequestError,
    serve_jsonl,
)
from grounding_control.transport.request_io import required_string
from grounding_control.workers.endpoints import QwenGrounderEndpoint


PROTOCOL_NAME = DEFAULT_PROTOCOL_NAME
RESPONSE_PREFIX = DEFAULT_RESPONSE_PREFIX
# Single-card Qwen7B Grounder budget.  This is deliberately much larger than
# the retired 401,408-pixel cap while bounding visual-encoder memory for
# arbitrarily large source images on a 24 GiB GPU.
DEFAULT_GROUNDER_MAX_PIXELS = 12_000_000


class QwenGrounderWorkerEngine:
    """Host one Qwen Grounder without any VoCoT coordinate dependency."""

    def __init__(
            self,
            runner: Optional[Qwen25VLRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            max_new_tokens: int = 64,
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: Optional[int] = DEFAULT_GROUNDER_MAX_PIXELS,
            attn_implementation: str = 'sdpa',
            prompt_protocol: str = DEFAULT_GROUNDING_PROMPT_PROTOCOL,
            boundary_tolerance_pixels: float = (
                DEFAULT_BOUNDARY_TOLERANCE_PIXELS
            )):
        if prompt_protocol not in GROUNDING_PROMPT_PROTOCOLS:
            raise ValueError(
                f'prompt_protocol must be one of {GROUNDING_PROMPT_PROTOCOLS}'
            )
        self.model_path = None if model_path is None else str(model_path)
        self.device = str(device)
        self.dtype = str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = None if max_pixels is None else int(max_pixels)
        self.attn_implementation = str(attn_implementation)
        self.prompt_protocol = str(prompt_protocol)
        self.boundary_tolerance_pixels = float(boundary_tolerance_pixels)

        if runner is None and self.model_path is not None:
            runner = LocalQwen25VLRunner(
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
                max_new_tokens=self.max_new_tokens,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                attn_implementation=self.attn_implementation,
            )
        self.endpoint = None
        if runner is not None:
            self.endpoint = QwenGrounderEndpoint(Qwen25VLBoxPredictor(
                runner=runner,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                boundary_tolerance_pixels=(
                    self.boundary_tolerance_pixels
                ),
                prompt_protocol=self.prompt_protocol,
            ))

    def _ping(self) -> Dict[str, Any]:
        return {
            'worker': 'qwen25_vl_grounder',
            'configured': self.endpoint is not None,
            'model_path': self.model_path,
            'device': self.device,
            'dtype': self.dtype,
            'max_new_tokens': self.max_new_tokens,
            'min_pixels': self.min_pixels,
            'max_pixels': self.max_pixels,
            'attn_implementation': self.attn_implementation,
            'prompt_protocol': self.prompt_protocol,
            'boundary_tolerance_pixels': (
                self.boundary_tolerance_pixels
            ),
            'output_coordinate_system': (
                'absolute_xyxy_on_original_image'
            ),
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
        if operation == 'ground':
            if self.endpoint is None:
                raise WorkerRequestError(
                    'Qwen2.5-VL Grounder is not configured'
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
    parser.add_argument(
        '--max-pixels', type=int, default=DEFAULT_GROUNDER_MAX_PIXELS,
        help=(
            'Grounder image-pixel budget; defaults to 12MP for reliable '
            'single-card Qwen7B inference.'
        ),
    )
    parser.add_argument('--attn-implementation', default='sdpa')
    parser.add_argument(
        '--prompt-protocol',
        choices=GROUNDING_PROMPT_PROTOCOLS,
        default=DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    )
    parser.add_argument(
        '--boundary-tolerance-pixels',
        type=float,
        default=DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = QwenGrounderWorkerEngine(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        prompt_protocol=args.prompt_protocol,
        boundary_tolerance_pixels=args.boundary_tolerance_pixels,
    )
    return serve_jsonl(
        engine,
        protocol_name=PROTOCOL_NAME,
        response_prefix=RESPONSE_PREFIX,
    )


__all__ = [
    'DEFAULT_GROUNDER_MAX_PIXELS',
    'PROTOCOL_NAME',
    'QwenGrounderWorkerEngine',
    'RESPONSE_PREFIX',
    'main',
    'parse_args',
]


if __name__ == '__main__':
    raise SystemExit(main())
