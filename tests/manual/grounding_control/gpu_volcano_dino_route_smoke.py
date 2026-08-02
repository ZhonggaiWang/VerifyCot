"""Real Volcano + remote DINO verifier + oracle Grounder smoke test."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from model.load_model import alignment_routing_infer, load_model  # noqa: E402
from grounding_control import OracleGrounderBackend, OracleTargetResolver  # noqa: E402
from grounding_control.transport import PersistentJsonlWorkerClient  # noqa: E402
from grounding_control.verifiers import RemoteAlignmentVerifierBackend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--dino-model-path', required=True)
    parser.add_argument('--worker-python', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print compact route invariants instead of the complete event log.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image_path).resolve()
    with Image.open(image_path) as opened:
        image = opened.convert('RGB').copy()
    command = [
        str(args.worker_python),
        '-u',
        '-m',
        'grounding_control.workers.dino_verifier',
        '--model-path',
        str(Path(args.dino_model_path).resolve()),
        '--device',
        'cuda:0',
    ]
    with PersistentJsonlWorkerClient(
            command,
            cwd=str(ROOT),
            timeout=600.0,
    ) as client:
        ping = client.ping()
        model, preprocessor = load_model(
            str(Path(args.model_path).resolve()),
            precision='fp16',
        )
        resolver = OracleTargetResolver(
            preprocessor.tokenizer,
            oracle_targets=[{
                'object': 'glove',
                'aliases': ['glove'],
                'box': [0.282, 0.196, 0.3595, 0.2745],
            }],
        )
        result = alignment_routing_infer(
            model=model,
            preprocessor=preprocessor,
            image=image,
            verifier_backend=RemoteAlignmentVerifierBackend(
                client,
                timeout=600.0,
                fail_open=False,
            ),
            grounder_backend=OracleGrounderBackend(resolver),
            # Deliberately high raw-IoU thresholds make this smoke test enter
            # the Grounder branch; they are not proposed experiment values.
            reject_threshold=0.95,
            accept_threshold=0.99,
            alignment_score_kind='iou_proxy',
            query='What is the material of the glove?',
            cot=True,
            sample_id='gpu-volcano-dino-smoke',
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            sample_context={'image_path': str(image_path)},
            missing_expert_policy='error',
        )
    if not result['events']:
        raise RuntimeError('Volcano emitted no coordinate to verify')
    if not any(event.get('grounder_succeeded') for event in result['events']):
        raise RuntimeError('combined smoke test never entered the Grounder')
    for event in result['events']:
        if event.get('raw_alignment_score_kind') != 'iou_proxy':
            raise RuntimeError('DINO IoU proxy kind was not preserved')
        if event.get('verification') != (
                'text_and_refbind_match_committed_box'):
            raise RuntimeError('coordinate text/REFbind invariant failed')
        metadata = event.get('verifier_metadata') or {}
        if metadata.get('transport') != 'persistent_jsonl_worker':
            raise RuntimeError('event did not use the remote worker')
    payload = {
        'status': 'passed',
        'worker_ping': ping,
        'response': result['response'],
        'boxes': result['boxes'],
        'bound_boxes': result['bound_boxes'],
        'events': result['events'],
    }
    if args.summary_only:
        payload = {
            'status': payload['status'],
            'worker_ping': payload['worker_ping'],
            'response': payload['response'],
            'coordinate_count': len(payload['events']),
            'boxes_match_bound_boxes': payload['boxes'] == payload['bound_boxes'],
            'events': [{
                key: event.get(key)
                for key in (
                    'grounding_step',
                    'decision_band',
                    'system_action',
                    'grounder_succeeded',
                    'candidate_box',
                    'committed_box',
                    'alignment_score',
                    'alignment_score_kind',
                    'verification',
                )
            } for event in payload['events']],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
