"""Run single-grounding counterfactual interventions on GQA-val manifests."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm

from model.load_model import _build_inference_batch, counterfactual_infer, load_model
from common import answer_is_correct, generate_gqa_final_answer, make_gqa_conversation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--final-max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--perturb-mode', choices=('random_box', 'remove_grounding'),
                        default='random_box')
    parser.add_argument('--perturb-index', type=int, default=None)
    parser.add_argument('--perturb-position', choices=('random', 'first', 'last'), default='random')
    parser.add_argument('--selection-seed', type=int, default=2026)
    parser.add_argument('--perturb-seed', type=int, default=2027)
    parser.add_argument('--iou-min', type=float, default=0.0)
    parser.add_argument('--iou-max', type=float, default=0.1)
    parser.add_argument('--perturb-box-mode', choices=('random', 'same_shape'), default='random')
    parser.add_argument('--random-box-min-size', type=float, default=0.05)
    parser.add_argument('--random-box-max-size', type=float, default=0.5)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_existing(path):
    return read_jsonl(path) if Path(path).exists() else []


def accuracy(records, prediction_key):
    eligible = [record for record in records if record.get(prediction_key) is not None]
    if not eligible:
        return None, 0
    return sum(record[prediction_key + '_correct'] for record in eligible) / len(eligible), len(eligible)


def transitions(records):
    result = Counter({
        'correct_to_wrong': 0, 'wrong_to_correct': 0,
        'correct_to_correct': 0, 'wrong_to_wrong': 0,
    })
    for record in records:
        baseline_correct = record['baseline_prediction_correct']
        counterfactual_correct = record['counterfactual_prediction_correct']
        if baseline_correct and not counterfactual_correct:
            result['correct_to_wrong'] += 1
        elif not baseline_correct and counterfactual_correct:
            result['wrong_to_correct'] += 1
        elif baseline_correct:
            result['correct_to_correct'] += 1
        else:
            result['wrong_to_wrong'] += 1
    return dict(result)


def paired_summary(records):
    paired = [
        record for record in records
        if record.get('baseline_prediction') is not None
        and record.get('counterfactual_prediction') is not None
    ]
    baseline_accuracy, baseline_count = accuracy(paired, 'baseline_prediction')
    counterfactual_accuracy, counterfactual_count = accuracy(paired, 'counterfactual_prediction')
    return {
        'samples': len(paired),
        'baseline_accuracy': baseline_accuracy,
        'baseline_accuracy_count': baseline_count,
        'counterfactual_accuracy': counterfactual_accuracy,
        'counterfactual_accuracy_count': counterfactual_count,
        'counterfactual_minus_baseline': (
            None if baseline_accuracy is None else counterfactual_accuracy - baseline_accuracy
        ),
        'answer_changed_count': sum(
            record['baseline_prediction'] != record['counterfactual_prediction'] for record in paired
        ),
        'correctness_transitions': transitions(paired),
    }


def make_summary(records, args):
    successful = [record for record in records if record.get('status') in ('ok', 'no_coordinate')]
    by_type = {}
    for type_name in sorted({record['types']['structural'] for record in successful}):
        by_type[type_name] = paired_summary([
            record for record in successful if record['types']['structural'] == type_name
        ])
    return {
        'total_records': len(records),
        'successful_records': len(successful),
        'errors': sum(record.get('status') == 'error' for record in records),
        'no_coordinate': sum(record.get('status') == 'no_coordinate' for record in records),
        'all_samples': paired_summary(successful),
        'by_structural_type': by_type,
        'settings': {
            'dataset': 'gqa_val_manifest',
            'max_new_tokens': args.max_new_tokens,
            'final_max_new_tokens': args.final_max_new_tokens,
            'temperature': args.temperature,
            'perturb_mode': args.perturb_mode,
            'perturb_position': args.perturb_position,
            'perturb_index': args.perturb_index,
            'selection_seed': args.selection_seed,
            'perturb_seed': args.perturb_seed,
            'iou_range': [args.iou_min, args.iou_max],
            'perturb_box_mode': args.perturb_box_mode,
        },
    }


def main():
    args = parse_args()
    if args.perturb_index is not None and args.perturb_position != 'random':
        raise ValueError('--perturb-index cannot be combined with --perturb-position')
    if args.perturb_position == 'first':
        resolved_index = 1
    elif args.perturb_position == 'last':
        resolved_index = 'last'
    else:
        resolved_index = args.perturb_index

    manifest = read_jsonl(args.manifest_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.no_resume else load_existing(output_path)
    completed_indices = {record['sample_index'] for record in existing if 'sample_index' in record}
    end_index = len(manifest) if args.max_samples is None else min(
        len(manifest), args.start_index + args.max_samples
    )
    indices = [index for index in range(args.start_index, end_index) if index not in completed_indices]
    print(f'GQA manifest samples: {len(manifest)}; evaluating: {len(indices)}; resumed: {len(completed_indices)}')
    if indices:
        model, preprocessor = load_model(args.model_path, precision='fp16')

    with output_path.open('a') as handle:
        for index in tqdm(indices, desc='GQA counterfactual evaluation'):
            item = manifest[index]
            record = {
                'sample_index': item['sample_index'],
                'question_id': item['question_id'],
                'image_id': item['image_id'],
                'question': item['question'],
                'answer': item['answer'],
                'types': item['types'],
                'target_objects': item['target_objects'],
            }
            try:
                with Image.open(item['image_path']) as source_image:
                    image = source_image.convert('RGB')
                conversation = make_gqa_conversation(item['question'])
                prompt_length = _build_inference_batch(
                    preprocessor, image, conversation=conversation
                )['input_ids'].shape[-1]
                selection_seed = args.selection_seed + 2 * index
                perturb_seed = args.perturb_seed + 2 * index + 1
                rollout = counterfactual_infer(
                    model, preprocessor, image, query=None, cot=True,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    perturb_index=resolved_index,
                    selection_seed=selection_seed,
                    perturb_seed=perturb_seed,
                    perturb_iou_range=(args.iou_min, args.iou_max),
                    perturb_mode=args.perturb_mode,
                    perturb_box_mode=args.perturb_box_mode,
                    random_box_min_size=args.random_box_min_size,
                    random_box_max_size=args.random_box_max_size,
                    conversation=conversation,
                    return_sequences=True,
                    allow_missing_coordinates=True,
                )
                baseline_sequences = rollout.pop('_baseline_sequences')
                counterfactual_sequences = rollout.pop('_counterfactual_sequences')
                baseline_final = generate_gqa_final_answer(
                    model, preprocessor, image, baseline_sequences, prompt_length,
                    args.final_max_new_tokens, args.temperature,
                )
                record['baseline_prediction'] = baseline_final['prediction']
                record['baseline_prediction_correct'] = answer_is_correct(
                    baseline_final['prediction'], item['answer']
                )
                record['baseline_final_response'] = baseline_final['response']
                if counterfactual_sequences is not None:
                    counterfactual_final = generate_gqa_final_answer(
                        model, preprocessor, image, counterfactual_sequences, prompt_length,
                        args.final_max_new_tokens, args.temperature,
                    )
                    record['counterfactual_prediction'] = counterfactual_final['prediction']
                    record['counterfactual_prediction_correct'] = answer_is_correct(
                        counterfactual_final['prediction'], item['answer']
                    )
                    record['counterfactual_final_response'] = counterfactual_final['response']
                    record['status'] = 'ok'
                else:
                    record['counterfactual_prediction'] = None
                    record['counterfactual_prediction_correct'] = None
                    record['status'] = 'no_coordinate'
                record.update(rollout)
            except Exception as error:
                record['status'] = 'error'
                record['error'] = f'{type(error).__name__}: {error}'
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    summary = make_summary(load_existing(output_path), args)
    summary_path = output_path.with_suffix('.summary.json')
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
