"""Real-model smoke test for one remote verifier or Grounder role.

The script starts the production JSONL worker, calls it through the production
remote backend, validates the typed output, and shuts the worker down.  It is
manual because it loads multi-billion-parameter checkpoints on a real GPU.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grounding_control.contracts import (  # noqa: E402
    CandidateAlignmentRequest,
    GroundingRequest,
    VisualInput,
)
from grounding_control.experts.grounders import RemoteGrounderBackend  # noqa: E402
from grounding_control.transport import PersistentJsonlWorkerClient  # noqa: E402
from grounding_control.verifiers import RemoteAlignmentVerifierBackend  # noqa: E402


ROLES = (
    'dino_alignment',
    'dino_grounder',
    'qwen_alignment',
    'qwen_grounder',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=ROLES, required=True)
    parser.add_argument('--worker-python', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--reference', default='the glove')
    parser.add_argument(
        '--candidate-box',
        nargs=4,
        type=float,
        default=(0.282, 0.196, 0.3595, 0.2745),
    )
    parser.add_argument('--timeout', type=float, default=600.0)
    return parser.parse_args()


def worker_command(args: argparse.Namespace):
    common = [str(args.worker_python), '-u', '-m']
    if args.role == 'dino_alignment':
        return common + [
            'grounding_control.workers.dino_verifier',
            '--model-path', str(Path(args.model_path).resolve()),
            '--device', 'cuda:0',
        ]
    if args.role == 'dino_grounder':
        return common + [
            'grounding_control.workers.dino_grounder',
            '--model-path', str(Path(args.model_path).resolve()),
            '--device', 'cuda:0',
        ]
    if args.role == 'qwen_alignment':
        return common + [
            'grounding_control.workers.qwen_verifier',
            '--model-path', str(Path(args.model_path).resolve()),
            '--device', 'cuda:0',
            '--default-image-mode', 'bbox_image_only',
        ]
    return common + [
        'grounding_control.workers.qwen_grounder',
        '--model-path', str(Path(args.model_path).resolve()),
        '--device', 'cuda:0',
    ]


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    command = worker_command(args)
    with PersistentJsonlWorkerClient(
            command,
            cwd=str(ROOT),
            env={'CUDA_VISIBLE_DEVICES': str(args.gpu)},
            timeout=args.timeout,
    ) as client:
        ping = client.ping()
        if args.role.endswith('_alignment'):
            output = RemoteAlignmentVerifierBackend(
                client,
                timeout=args.timeout,
                fail_open=False,
                image_mode='bbox_image_only',
            ).verify_alignment(CandidateAlignmentRequest(
                sample_id='gpu-smoke',
                grounding_step=1,
                object_reference=args.reference,
                candidate_bbox=tuple(args.candidate_box),
                visual=VisualInput(image_path=str(image_path)),
            ))
            expected_kind = (
                'iou_proxy'
                if args.role == 'dino_alignment'
                else 'self_reported_probability'
            )
            if output.abstained:
                raise RuntimeError(
                    f'real alignment backend abstained: {output.error}'
                )
            if output.score_kind != expected_kind:
                raise RuntimeError(
                    f'unexpected score kind: {output.score_kind!r}'
                )
            result = output.as_dict()
        else:
            expected_source = (
                'grounding_dino_grounder'
                if args.role == 'dino_grounder'
                else 'qwen25_vl_grounder'
            )
            result = RemoteGrounderBackend(
                client,
                timeout=args.timeout,
                source=expected_source,
            ).ground(GroundingRequest(
                sample_id='gpu-smoke',
                grounding_step=1,
                object_reference=args.reference,
                visual=VisualInput(image_path=str(image_path)),
            ))
            result = {
                'bbox_vocot_padded_normalized_xyxy': list(result.bbox),
                'source': result.source,
                'confidence': result.confidence,
                'metadata': result.metadata,
            }
    print(json.dumps({
        'role': args.role,
        'worker_ping': ping,
        'result': result,
        'status': 'passed',
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
