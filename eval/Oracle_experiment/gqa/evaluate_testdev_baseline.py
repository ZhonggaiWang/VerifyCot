"""Evaluate the untouched VoCoT baseline on official GQA Test-Dev balanced."""

import argparse
import fcntl
import json
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.load_model import _build_inference_batch, load_model
from common import (
    answer_is_correct,
    generate_gqa_cot,
    generate_gqa_final_answer,
    make_gqa_conversation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument(
        '--manifest-path',
        default='output/gqa/annotations/testdev_balanced/manifest.jsonl',
    )
    parser.add_argument(
        '--output',
        default='output/gqa/testdev_baseline/results.jsonl',
    )
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--final-max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-id', default=None)
    parser.add_argument('--model-load-lock', default=None)
    parser.add_argument('--model-load-lock-timeout-seconds', type=float, default=300.0)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model_with_optional_lock(args):
    """Serialize checkpoint loading while keeping inference parallel."""
    if args.model_load_lock is None:
        return load_model(args.model_path, precision='fp16')
    lock_path = Path(args.model_load_lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.model_load_lock_timeout_seconds
    with lock_path.open('a+', encoding='utf-8') as lock_handle:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f'timed out waiting for model-load lock: {lock_path}'
                    )
                time.sleep(1.0)
        print(f'Acquired model-load lock: {lock_path}', flush=True)
        try:
            result = load_model(args.model_path, precision='fp16')
            print('Model loaded; releasing model-load lock.', flush=True)
            return result
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def baseline_metrics(records):
    eligible = [
        record for record in records
        if record.get('baseline_prediction') is not None
    ]
    if not eligible:
        return {'samples': 0}
    correct = sum(record['baseline_prediction_correct'] for record in eligible)
    coordinate_counts = [
        len(record.get('baseline', {}).get('boxes') or []) for record in eligible
    ]
    return {
        'samples': len(eligible),
        'correct_count': correct,
        'accuracy': correct / len(eligible),
        'coordinate_count': sum(coordinate_counts),
        'samples_with_coordinate': sum(count > 0 for count in coordinate_counts),
        'mean_coordinates_per_sample': sum(coordinate_counts) / len(eligible),
    }


def grouped_metrics(records, type_key):
    values = sorted({
        str(record.get('types', {}).get(type_key, 'unknown'))
        for record in records
    })
    return {
        value: baseline_metrics([
            record for record in records
            if str(record.get('types', {}).get(type_key, 'unknown')) == value
        ])
        for value in values
    }


def make_summary(records, args):
    successful = [record for record in records if record.get('status') == 'ok']
    return {
        'total_records': len(records),
        'successful_records': len(successful),
        'error_records': len(records) - len(successful),
        'all_samples': baseline_metrics(successful),
        'by_structural_type': grouped_metrics(successful, 'structural'),
        'by_semantic_type': grouped_metrics(successful, 'semantic'),
        'settings': {
            'dataset': 'GQA',
            'split': 'testdev_balanced',
            'mode': 'untouched_vocot_baseline',
            'manifest_path': args.manifest_path,
            'max_new_tokens': args.max_new_tokens,
            'final_max_new_tokens': args.final_max_new_tokens,
            'temperature': args.temperature,
            'answer_metric': 'normalized_exact_match',
            'gt_object_boxes_used': False,
            'model_load_lock': args.model_load_lock,
        },
    }


def main():
    args = parse_args()
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    if args.model_load_lock_timeout_seconds <= 0:
        raise ValueError('--model-load-lock-timeout-seconds must be positive')

    manifest = read_jsonl(args.manifest_path)
    if any(record.get('source_split') != 'testdev_balanced' for record in manifest):
        raise ValueError('manifest contains records outside GQA testdev_balanced')
    if args.sample_id is not None:
        selected = [
            record for record in manifest
            if str(record['question_id']) == str(args.sample_id)
        ]
        if len(selected) != 1:
            raise ValueError(
                f'expected one record for --sample-id {args.sample_id!r}, '
                f'found {len(selected)}'
            )
    else:
        end = (
            len(manifest)
            if args.max_samples is None
            else min(len(manifest), args.start_index + args.max_samples)
        )
        selected = manifest[args.start_index:end]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        [] if args.no_resume or not output_path.exists()
        else read_jsonl(output_path)
    )
    incompatible = [
        record for record in existing
        if record.get('source_split') != 'testdev_balanced'
        or record.get('evaluation_mode') != 'untouched_vocot_baseline'
    ]
    if incompatible:
        raise ValueError(
            'existing output is not a compatible GQA Test-Dev baseline run; '
            'use another output or --no-resume'
        )
    completed = {
        int(record['sample_index'])
        for record in existing if record.get('sample_index') is not None
    }
    pending = [
        record for record in selected
        if int(record['sample_index']) not in completed
    ]
    print(
        f'GQA Test-Dev selected: {len(selected)}; evaluating: {len(pending)}; '
        f'resumed: {len(selected) - len(pending)}'
    )
    if pending:
        model, preprocessor = load_model_with_optional_lock(args)

    output_mode = 'w' if args.no_resume else 'a'
    with output_path.open(output_mode, encoding='utf-8') as handle:
        for item in tqdm(pending, desc='GQA Test-Dev baseline'):
            record = {
                key: item.get(key) for key in (
                    'sample_index', 'question_id', 'image_id', 'image_path',
                    'image_size', 'question', 'answer', 'full_answer', 'types',
                    'source_split',
                )
            }
            record['evaluation_mode'] = 'untouched_vocot_baseline'
            try:
                with Image.open(item['image_path']) as opened:
                    image = opened.convert('RGB')
                expected_size = item.get('image_size') or {}
                if image.size != (
                        expected_size.get('width'), expected_size.get('height')):
                    raise ValueError(
                        f'image size {image.size} differs from manifest {expected_size}'
                    )
                conversation = make_gqa_conversation(item['question'])
                prompt_length = _build_inference_batch(
                    preprocessor, image, conversation=conversation
                )['input_ids'].shape[-1]
                baseline, sequences = generate_gqa_cot(
                    model,
                    preprocessor,
                    image,
                    conversation,
                    args.max_new_tokens,
                    args.temperature,
                )
                final = generate_gqa_final_answer(
                    model,
                    preprocessor,
                    image,
                    sequences,
                    prompt_length,
                    args.final_max_new_tokens,
                    args.temperature,
                )
                record.update({
                    'baseline': baseline,
                    'baseline_prediction': final['prediction'],
                    'baseline_prediction_correct': answer_is_correct(
                        final['prediction'], item['answer']
                    ),
                    'baseline_final_response': final['response'],
                    'status': 'ok',
                })
                if args.verbose:
                    tqdm.write(
                        f'[{item["question_id"]}] prediction='
                        f'{final["prediction"]!r}; answer={item["answer"]!r}; '
                        f'correct={record["baseline_prediction_correct"]}'
                    )
            except Exception as error:
                record.update({
                    'status': 'error',
                    'error': f'{type(error).__name__}: {error}',
                })
                if args.verbose:
                    tqdm.write(f'[{item["question_id"]}] ERROR: {record["error"]}')
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    records = read_jsonl(output_path)
    summary = make_summary(records, args)
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
