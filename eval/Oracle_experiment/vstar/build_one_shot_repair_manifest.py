"""Build one-shot random-box repair inputs from saved VStar online-oracle CoTs.

No GPU is used.  The online-oracle rollout is treated as a reusable reference
trajectory.  One explicitly GT-forced coordinate is selected per sample and
replaced with a low-IoU random box; the matching StoredOracle JSONL contains
the sole ``misaligned/wrong_object`` verifier event.
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coordinate_intervention import box_iou, make_random_box_perturbation
from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN
from grounding_control.run_paths import resolve_run_output


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--online-oracle-results', required=True)
    parser.add_argument('--manifest-output', required=True)
    parser.add_argument('--verifier-output', required=True)
    parser.add_argument('--run-id', default=None,
                        help='Output subdirectory. Defaults to a YYYYMMDD_HHMMSS timestamp.')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--iou-max', type=float, default=0.05)
    parser.add_argument('--min-box-size', type=float, default=0.05)
    parser.add_argument('--max-box-size', type=float, default=0.5)
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.iou_max <= 1:
        raise ValueError('--iou-max must be in [0, 1]')
    manifest_path, run_id = resolve_run_output(args.manifest_output, args.run_id)
    verifier_path, _ = resolve_run_output(args.verifier_output, run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_path.parent.mkdir(parents=True, exist_ok=True)

    selected = []
    for source in read_jsonl(args.online_oracle_results):
        if source.get('status') != 'ok':
            continue
        oracle = source.get('oracle', {})
        events = source.get('intervention', {}).get('events', [])
        forced = [event for event in events if event.get('decision') == 'forced_gt_box']
        if not oracle.get('generated_ids') or not forced:
            continue
        sample_id = str(source.get('question_id', source['sample_index']))
        rng = random.Random(f'{args.seed}:{sample_id}')
        event = rng.choice(forced)
        reference_box = tuple(float(value) for value in event['oracle_box'])
        random_box = make_random_box_perturbation(
            reference_box, rng, iou_range=(0.0, args.iou_max),
            min_box_size=args.min_box_size, max_box_size=args.max_box_size,
        )
        selected.append({
            'sample_id': sample_id,
            'sample_index': source['sample_index'],
            'question_id': source.get('question_id'),
            'image': source['image'],
            'question': source['question'],
            'options': source.get('options'),
            'label': source.get('label'),
            'category': source.get('category'),
            'conversation': [{
                'from': 'human',
                'value': ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + source['question'] + ' '
                         + COT_ACTIVATION,
            }],
            'reference_generated_ids': oracle['generated_ids'],
            'reference_response': oracle.get('response'),
            'reference_boxes': oracle.get('boxes'),
            'selected_coordinate_index': int(event['coordinate_index']),
            'selected_object_reference': event.get('target_object'),
            'selected_event': event,
            'reference_box': list(reference_box),
            'random_box': list(random_box),
            'random_box_iou_to_reference': box_iou(reference_box, random_box),
            'selection_seed': args.seed,
        })
        if args.max_samples is not None and len(selected) >= args.max_samples:
            break

    with manifest_path.open('w', encoding='utf-8') as manifest, verifier_path.open('w', encoding='utf-8') as verifier:
        for item in selected:
            manifest.write(json.dumps(item, ensure_ascii=False) + '\n')
            verifier.write(json.dumps({
                'sample_id': item['sample_id'],
                'grounding_step': item['selected_coordinate_index'],
                'attempt_index': 0,
                'object_reference': item['selected_object_reference'],
                'candidate_bbox': item['random_box'],
                'verifier_output': {
                    'verdict': 'misaligned',
                    'reason': 'wrong_object',
                    'confidence': 1.0,
                },
            }, ensure_ascii=False) + '\n')
    print(f'Run id: {run_id}')
    print(f'One-shot manifest: {manifest_path} ({len(selected)} samples)')
    print(f'Stored verifier outputs: {verifier_path} ({len(selected)} records)')


if __name__ == '__main__':
    main()
