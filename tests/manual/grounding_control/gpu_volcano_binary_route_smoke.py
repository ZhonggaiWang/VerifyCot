"""Real Volcano smoke test for binary pre-commit routing and REFbind."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from model.load_model import alignment_routing_infer, load_model  # noqa: E402
from grounding_control import (  # noqa: E402
    OracleAlignmentVerifierBackend,
    OracleGrounderBackend,
    OracleTargetResolver,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--gt-iou-threshold', type=float, default=0.5)
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
    model, preprocessor = load_model(
        str(Path(args.model_path).resolve()),
        precision='fp16',
    )
    targets = [{
        'object': 'glove',
        'aliases': ['glove'],
        'box': [0.282, 0.196, 0.3595, 0.2745],
    }]
    resolver = OracleTargetResolver(
        preprocessor.tokenizer,
        oracle_targets=targets,
    )
    result = alignment_routing_infer(
        model=model,
        preprocessor=preprocessor,
        image=image,
        verifier_backend=OracleAlignmentVerifierBackend(
            resolver,
            gt_iou_threshold=args.gt_iou_threshold,
        ),
        grounder_backend=OracleGrounderBackend(resolver),
        reject_threshold=0.25,
        accept_threshold=0.75,
        alignment_score_kind='hard_oracle_label',
        query='What is the material of the glove?',
        cot=True,
        sample_id='gpu-volcano-smoke',
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        sample_context={'image_path': str(image_path)},
        missing_expert_policy='error',
    )
    if not result['events']:
        raise RuntimeError('Volcano emitted no coordinate to verify')
    for event in result['events']:
        if event.get('event_schema') != 'vocot_precommit_alignment_event_v1':
            raise RuntimeError('binary event schema was not emitted')
        if event.get('verification') != (
                'text_and_refbind_match_committed_box'):
            raise RuntimeError('coordinate text/REFbind invariant failed')
        if event.get('alignment_score_kind') != 'hard_oracle_label':
            raise RuntimeError('routing did not consume the declared score kind')
        if event.get('raw_alignment_score_kind') != 'hard_oracle_label':
            raise RuntimeError('raw verifier score kind was not logged')
    payload = {
        'status': 'passed',
        'response': result['response'],
        'boxes': result['boxes'],
        'bound_boxes': result['bound_boxes'],
        'events': result['events'],
    }
    if args.summary_only:
        payload = {
            'status': payload['status'],
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
