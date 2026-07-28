"""Run online GT target-grounding oracle evaluation on GQA-val manifests."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from tqdm import tqdm

from model.load_model import _build_inference_batch, load_model, online_oracle_infer
from utils.coordinate_intervention import (
    normalize_object_reference,
    normalized_box_to_square_padding,
)
from common import (
    answer_is_correct,
    generate_gqa_cot,
    generate_gqa_final_answer,
    make_gqa_conversation,
)

ORACLE_BOX_COORDINATE_SYSTEM = 'normalized_xyxy_on_center_padded_square'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--final-max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--context-window-tokens', type=int, default=48)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_existing(path):
    return read_jsonl(path) if Path(path).exists() else []


def build_unique_oracle_targets(target_objects, image_width, image_height):
    """Keep unique GQA targets and map their boxes to the model image canvas."""
    by_alias = {}
    invalid = []
    for object_record in target_objects:
        name = object_record.get('name')
        alias = normalize_object_reference(name)
        if not alias:
            invalid.append({'object_id': object_record['object_id'], 'reason': 'empty_object_name'})
            continue
        by_alias.setdefault(alias, []).append(object_record)
    targets = []
    for alias, records in by_alias.items():
        if len(records) != 1:
            for record in records:
                invalid.append({
                    'object_id': record['object_id'],
                    'reason': 'duplicate_target_name_in_question',
                    'normalized_name': ' '.join(alias),
                })
            continue
        record = records[0]
        targets.append({
            'object': record['name'],
            'box': normalized_box_to_square_padding(
                record['normalized_bbox_xyxy'], image_width, image_height
            ),
            'aliases': [record['name']],
            'object_id': record['object_id'],
        })
    return targets, invalid


def accuracy(records, key):
    records = [record for record in records if record.get(key) is not None]
    return (None, 0) if not records else (sum(record[key + '_correct'] for record in records) / len(records), len(records))


def transition_counts(records):
    counts = Counter({'correct_to_wrong': 0, 'wrong_to_correct': 0, 'correct_to_correct': 0, 'wrong_to_wrong': 0})
    for record in records:
        before, after = record['baseline_prediction_correct'], record['oracle_prediction_correct']
        counts['correct_to_wrong' if before and not after else 'wrong_to_correct' if not before and after else 'correct_to_correct' if before else 'wrong_to_wrong'] += 1
    return dict(counts)


def subset_summary(records):
    paired = [r for r in records if r.get('baseline_prediction') is not None and r.get('oracle_prediction') is not None]
    baseline, baseline_count = accuracy(paired, 'baseline_prediction')
    oracle, oracle_count = accuracy(paired, 'oracle_prediction')
    return {
        'samples': len(paired),
        'baseline_accuracy': baseline,
        'baseline_accuracy_count': baseline_count,
        'oracle_accuracy': oracle,
        'oracle_accuracy_count': oracle_count,
        'oracle_minus_baseline': None if baseline is None else oracle - baseline,
        'forced_sample_count': sum(r['intervention']['forced_coordinate_count'] > 0 for r in paired),
        'total_forced_coordinate_count': sum(r['intervention']['forced_coordinate_count'] for r in paired),
        'answer_changed_count': sum(r['baseline_prediction'] != r['oracle_prediction'] for r in paired),
        'correctness_transitions': transition_counts(paired),
    }


def main():
    args = parse_args()
    manifest = read_jsonl(args.manifest_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.no_resume else load_existing(output_path)
    incompatible = [
        record for record in existing
        if record.get('oracle_box_coordinate_system') != ORACLE_BOX_COORDINATE_SYSTEM
    ]
    if incompatible:
        raise ValueError(
            'existing output uses the old or unknown oracle-box coordinate system; '
            'choose a new --output path or rerun with --no-resume'
        )
    completed = {record['sample_index'] for record in existing if 'sample_index' in record}
    end = len(manifest) if args.max_samples is None else min(len(manifest), args.start_index + args.max_samples)
    indices = [index for index in range(args.start_index, end) if index not in completed]
    print(f'GQA manifest samples: {len(manifest)}; evaluating: {len(indices)}; resumed: {len(completed)}')
    if indices:
        model, preprocessor = load_model(args.model_path, precision='fp16')

    output_mode = 'w' if args.no_resume else 'a'
    with output_path.open(output_mode) as handle:
        for index in tqdm(indices, desc='GQA online oracle evaluation'):
            item = manifest[index]
            record = {
                'sample_index': item['sample_index'], 'question_id': item['question_id'],
                'image_id': item['image_id'], 'question': item['question'],
                'answer': item['answer'], 'types': item['types'],
                'target_objects': item['target_objects'],
                'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
            }
            try:
                with Image.open(item['image_path']) as source_image:
                    image = source_image.convert('RGB')
                image_width, image_height = image.size
                conversation = make_gqa_conversation(item['question'])
                prompt_length = _build_inference_batch(preprocessor, image, conversation=conversation)['input_ids'].shape[-1]
                oracle_targets, excluded_targets = build_unique_oracle_targets(
                    item['target_objects'], image_width, image_height
                )
                if not oracle_targets:
                    raise ValueError('no textually unique GT targets are available for this sample')
                baseline, baseline_sequences = generate_gqa_cot(
                    model, preprocessor, image, conversation, args.max_new_tokens, args.temperature
                )
                oracle_rollout = online_oracle_infer(
                    model, preprocessor, image, query=None, cot=True,
                    oracle_targets=oracle_targets, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, conversation=conversation,
                    return_sequences=True, context_window_tokens=args.context_window_tokens,
                )
                oracle_sequences = oracle_rollout.pop('_oracle_sequences')
                baseline_final = generate_gqa_final_answer(model, preprocessor, image, baseline_sequences, prompt_length, args.final_max_new_tokens, args.temperature)
                oracle_final = generate_gqa_final_answer(model, preprocessor, image, oracle_sequences, prompt_length, args.final_max_new_tokens, args.temperature)
                record.update(oracle_rollout)
                record.update({
                    'oracle_targets': oracle_targets,
                    'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
                    'source_image_size': {
                        'width': image_width,
                        'height': image_height,
                    },
                    'excluded_oracle_targets': excluded_targets,
                    'baseline': baseline,
                    'baseline_prediction': baseline_final['prediction'],
                    'baseline_prediction_correct': answer_is_correct(baseline_final['prediction'], item['answer']),
                    'baseline_final_response': baseline_final['response'],
                    'oracle_prediction': oracle_final['prediction'],
                    'oracle_prediction_correct': answer_is_correct(oracle_final['prediction'], item['answer']),
                    'oracle_final_response': oracle_final['response'],
                    'status': 'ok',
                })
            except Exception as error:
                record['status'] = 'error'
                record['error'] = f'{type(error).__name__}: {error}'
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    records = load_existing(output_path)
    successful = [record for record in records if record.get('status') == 'ok']
    by_type = {
        type_name: subset_summary([r for r in successful if r['types']['structural'] == type_name])
        for type_name in sorted({r['types']['structural'] for r in successful})
    }
    summary = {
        'total_records': len(records), 'successful_records': len(successful),
        'errors': sum(r.get('status') == 'error' for r in records),
        'all_samples': subset_summary(successful), 'by_structural_type': by_type,
        'settings': {
            'dataset': 'gqa_val_manifest', 'max_new_tokens': args.max_new_tokens,
            'final_max_new_tokens': args.final_max_new_tokens, 'temperature': args.temperature,
            'context_window_tokens': args.context_window_tokens,
            'oracle_mode': 'online_explicit_target_oracle', 'coreference': 'off',
            'alias_policy': 'normalized full GQA scene-graph object name only',
            'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
        },
    }
    summary_path = output_path.with_suffix('.summary.json')
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
