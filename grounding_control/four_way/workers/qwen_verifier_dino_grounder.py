"""Retained Qwen four-way verifier and DINO Grounder composition.

JSONL framing, exception isolation, and shutdown behavior live in
``grounding_control.transport``.  Model-role request handling lives in
``grounding_control.four_way.workers.endpoints``. This module is the archived
deployable composition and its command-line configuration. Its binary request
mode remains available only for exact compatibility with existing runs.
"""

import argparse
from typing import Any, Dict, Mapping, Optional

from grounding_control.models.grounding_dino import (
    GroundingDinoRunner,
    LocalGroundingDinoRunner,
)
from grounding_control.models.qwen25_vl.grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
)
from grounding_control.verifiers.qwen25_vl.classifier import (
    Qwen25VLBinaryAlignmentClassifier,
)
from grounding_control.four_way.verifiers.qwen25_vl.geometry import (
    Qwen25VLGroundingGeometryClassifier,
)
from grounding_control.four_way.verifiers.qwen25_vl.backend import (
    Qwen25VLVerifierBackend,
)
from grounding_control.models.qwen25_vl import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLRunner,
)
from grounding_control.models.grounding_dino import GroundingDinoBoxPredictor
from grounding_control.transport import (
    DEFAULT_PROTOCOL_NAME,
    DEFAULT_RESPONSE_PREFIX,
    WorkerRequestError,
    process_request_line,
    serve_jsonl,
)
from grounding_control.transport.request_io import required_string
from grounding_control.workers.endpoints.dino_grounder import (
    DinoGrounderEndpoint,
)
from grounding_control.four_way.workers.endpoints.qwen_verifier import (
    QwenFourWayVerifierEndpoint,
    VERIFIER_MODES,
)


PROTOCOL_NAME = DEFAULT_PROTOCOL_NAME
RESPONSE_PREFIX = DEFAULT_RESPONSE_PREFIX


class QwenFourWayVerifierDinoGrounderWorkerEngine:
    """Compose independent verifier and expert endpoints in one process."""

    def __init__(
            self,
            qwen_runner: Optional[Qwen25VLRunner] = None,
            dino_runner: Optional[GroundingDinoRunner] = None,
            qwen_model_path: Optional[str] = None,
            dino_model_path: Optional[str] = None,
            qwen_device: str = 'cuda:0',
            dino_device: str = 'cuda:0',
            qwen_dtype: str = 'bfloat16',
            dino_dtype: str = 'float32',
            qwen_max_new_tokens: int = 64,
            qwen_min_pixels: int = DEFAULT_MIN_PIXELS,
            qwen_max_pixels: int = DEFAULT_MAX_PIXELS,
            qwen_attn_implementation: str = 'sdpa',
            crop_min_side: int = 56,
            default_verifier_mode: str = 'routing_four_way',
            default_verifier_image_mode: str = 'bbox_image_only',
            grounding_accept_iou: float = 0.5,
            grounding_containment: float = 0.7,
            grounding_prompt_protocol: str = (
                DEFAULT_GROUNDING_PROMPT_PROTOCOL
            ),
            dino_box_threshold: float = 0.3,
            dino_text_threshold: float = 0.25,
            dino_top_k_log: int = 20):
        if default_verifier_mode not in VERIFIER_MODES:
            raise ValueError(
                f'default_verifier_mode must be one of {VERIFIER_MODES}'
            )
        if grounding_prompt_protocol not in GROUNDING_PROMPT_PROTOCOLS:
            raise ValueError(
                'invalid grounding_prompt_protocol: '
                f'{grounding_prompt_protocol!r}'
            )
        if (
            not isinstance(dino_top_k_log, int)
            or isinstance(dino_top_k_log, bool)
            or dino_top_k_log <= 0
        ):
            raise ValueError('dino_top_k_log must be a positive integer')

        self.default_verifier_mode = default_verifier_mode
        self.default_verifier_image_mode = default_verifier_image_mode
        self.qwen_model_path = (
            None if qwen_model_path is None else str(qwen_model_path)
        )
        self.dino_model_path = (
            None if dino_model_path is None else str(dino_model_path)
        )

        if qwen_runner is None and self.qwen_model_path is not None:
            qwen_runner = LocalQwen25VLRunner(
                model_path=self.qwen_model_path,
                device=qwen_device,
                dtype=qwen_dtype,
                max_new_tokens=qwen_max_new_tokens,
                min_pixels=qwen_min_pixels,
                max_pixels=qwen_max_pixels,
                attn_implementation=qwen_attn_implementation,
            )
        if dino_runner is None and self.dino_model_path is not None:
            dino_runner = LocalGroundingDinoRunner(
                model_path=self.dino_model_path,
                device=dino_device,
                dtype=dino_dtype,
                box_threshold=dino_box_threshold,
                text_threshold=dino_text_threshold,
            )

        self.qwen_runner = qwen_runner
        self.dino_runner = dino_runner
        self.qwen_backend = (
            None
            if qwen_runner is None
            else Qwen25VLVerifierBackend(
                runner=qwen_runner,
                crop_min_side=crop_min_side,
                parse_fail_open=True,
            )
        )
        self.qwen_alignment = (
            None
            if qwen_runner is None
            else Qwen25VLBinaryAlignmentClassifier(
                qwen_runner,
                crop_min_side=crop_min_side,
                parse_fail_open=True,
            )
        )
        self.qwen_geometry = (
            None
            if qwen_runner is None
            else Qwen25VLGroundingGeometryClassifier(
                runner=qwen_runner,
                accept_iou_threshold=grounding_accept_iou,
                containment_threshold=grounding_containment,
                prompt_protocol=grounding_prompt_protocol,
            )
        )
        self.verifier_endpoint = (
            None
            if self.qwen_backend is None or self.qwen_geometry is None
            else QwenFourWayVerifierEndpoint(
                backend=self.qwen_backend,
                geometry_classifier=self.qwen_geometry,
                alignment_classifier=self.qwen_alignment,
                default_mode=default_verifier_mode,
                default_image_mode=default_verifier_image_mode,
            )
        )
        self.grounder_endpoint = (
            None
            if dino_runner is None
            else DinoGrounderEndpoint(GroundingDinoBoxPredictor(
                dino_runner,
                top_k_log=dino_top_k_log,
            ))
        )
        # The operation is reserved in the protocol, but no implementation is
        # silently substituted for an explicit expand/tighten expert.
        self.refiner_endpoint = None

    def _ping(self) -> Dict[str, Any]:
        return {
            'protocol': PROTOCOL_NAME,
            'worker': 'qwen_dino',
            'qwen_configured': self.verifier_endpoint is not None,
            'dino_configured': self.grounder_endpoint is not None,
            'box_refiner_configured': self.refiner_endpoint is not None,
            'qwen_model_path': self.qwen_model_path,
            'dino_model_path': self.dino_model_path,
            'default_verifier_mode': self.default_verifier_mode,
            'default_verifier_image_mode': (
                self.default_verifier_image_mode
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
        if operation == 'verify':
            if self.verifier_endpoint is None:
                raise WorkerRequestError(
                    'Qwen verifier is not configured for this worker'
                )
            return self.verifier_endpoint.handle(payload)
        if operation == 'ground':
            if self.grounder_endpoint is None:
                raise WorkerRequestError(
                    'Grounding DINO is not configured for this worker'
                )
            return self.grounder_endpoint.handle(payload)
        if operation == 'refine':
            if self.refiner_endpoint is None:
                raise WorkerRequestError('box_refiner_not_configured')
            return self.refiner_endpoint.handle(payload)
        if operation == 'shutdown':
            return {'shutdown': True}
        raise WorkerRequestError(f'unsupported operation: {operation!r}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Persistent Qwen verifier and DINO grounder JSONL worker.'
    )
    parser.add_argument('--qwen-model-path')
    parser.add_argument('--dino-model-path')
    parser.add_argument('--qwen-device', default='cuda:0')
    parser.add_argument('--dino-device', default='cuda:0')
    parser.add_argument('--qwen-dtype', default='bfloat16')
    parser.add_argument('--dino-dtype', default='float32')
    parser.add_argument('--qwen-max-new-tokens', type=int, default=64)
    parser.add_argument('--qwen-min-pixels', type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument('--qwen-max-pixels', type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument('--qwen-attn-implementation', default='sdpa')
    parser.add_argument('--crop-min-side', type=int, default=56)
    parser.add_argument(
        '--default-verifier-mode',
        choices=VERIFIER_MODES,
        default='routing_four_way',
    )
    parser.add_argument(
        '--default-verifier-image-mode',
        default='bbox_image_only',
    )
    parser.add_argument('--grounding-accept-iou', type=float, default=0.5)
    parser.add_argument('--grounding-containment', type=float, default=0.7)
    parser.add_argument(
        '--grounding-prompt-protocol',
        choices=GROUNDING_PROMPT_PROTOCOLS,
        default=DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    )
    parser.add_argument('--dino-box-threshold', type=float, default=0.3)
    parser.add_argument('--dino-text-threshold', type=float, default=0.25)
    parser.add_argument('--dino-top-k-log', type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = QwenFourWayVerifierDinoGrounderWorkerEngine(
        qwen_model_path=args.qwen_model_path,
        dino_model_path=args.dino_model_path,
        qwen_device=args.qwen_device,
        dino_device=args.dino_device,
        qwen_dtype=args.qwen_dtype,
        dino_dtype=args.dino_dtype,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        qwen_min_pixels=args.qwen_min_pixels,
        qwen_max_pixels=args.qwen_max_pixels,
        qwen_attn_implementation=args.qwen_attn_implementation,
        crop_min_side=args.crop_min_side,
        default_verifier_mode=args.default_verifier_mode,
        default_verifier_image_mode=args.default_verifier_image_mode,
        grounding_accept_iou=args.grounding_accept_iou,
        grounding_containment=args.grounding_containment,
        grounding_prompt_protocol=args.grounding_prompt_protocol,
        dino_box_threshold=args.dino_box_threshold,
        dino_text_threshold=args.dino_text_threshold,
        dino_top_k_log=args.dino_top_k_log,
    )
    return serve_jsonl(
        engine,
        protocol_name=PROTOCOL_NAME,
        response_prefix=RESPONSE_PREFIX,
    )


# Compatibility names retained for old worker entry points.
QwenVerifierDinoGrounderWorkerEngine = (
    QwenFourWayVerifierDinoGrounderWorkerEngine
)
QwenDinoWorkerEngine = QwenFourWayVerifierDinoGrounderWorkerEngine


__all__ = [
    'PROTOCOL_NAME',
    'QwenFourWayVerifierDinoGrounderWorkerEngine',
    'QwenDinoWorkerEngine',
    'QwenVerifierDinoGrounderWorkerEngine',
    'RESPONSE_PREFIX',
    'main',
    'parse_args',
    # Retain the accidental phase-one re-export used by existing tests.
    'process_request_line',
    'serve_jsonl',
]


if __name__ == '__main__':
    raise SystemExit(main())
