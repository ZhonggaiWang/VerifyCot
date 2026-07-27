"""Paired no-repair control for one-shot random-box VStar interventions."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.load_model import (
    _build_inference_batch,
    load_model,
    one_shot_reference_corruption_infer,
)
from verifier.run_paths import resolve_run_output


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', default=None,
                        help='Output subdirectory. Defaults to a YYYYMMDD_HHMMSS timestamp.')
    parser.add_argument('--event-log', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def score_options(model, preprocessor, image, item, generated_ids, max_new_tokens, temperature):
    batch = _build_inference_batch(preprocessor, image, conversation=item['conversation'], options=item['options'])
    prompt = batch['input_ids']
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    thought_ids = torch.cat([
        prompt.to(model.device),
        torch.tensor(generated_ids, dtype=prompt.dtype, device=model.device).unsqueeze(0),
    ], dim=1)
    with torch.inference_mode():
        prediction, _ = model.calculate_options(
            batch, cot=True, further_instruct=True, temperature=temperature,
            max_new_tokens=max_new_tokens, likelihood_reduction='mean',
            thought_override_ids=thought_ids,
        )
    return int(prediction)


def make_summary(records, args):
    ok = [record for record in records if record.get('status') == 'ok']
    by_step = defaultdict(list)
    for record in ok:
        by_step[str(record['selected_coordinate_index'])].append(record)

    def metrics(items):
        correct = sum(bool(item.get('prediction_correct')) for item in items)
        return {
            'samples': len(items),
            'correct_count': correct,
            'accuracy': correct / len(items) if items else None,
        }
    return {
        'total_records': len(records),
        'successful_records': len(ok),
        'error_records': len(records) - len(ok),
        'all_samples': metrics(ok),
        'by_selected_coordinate_index': {step: metrics(items) for step, items in sorted(by_step.items())},
        'settings': {
            'mode': 'one_shot_reference_random_box_corruption',
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'verifier_called': False,
            'random_box_committed': True,
        },
    }


def main():
    args = parse_args()
    output_path, run_id = resolve_run_output(args.output, args.run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_log = Path(args.event_log) if args.event_log else output_path.with_name('corruption_events.jsonl')
    existing = [] if args.no_resume or not output_path.exists() else read_jsonl(output_path)
    completed = {record['sample_id'] for record in existing if record.get('status') == 'ok'}
    pending = [record for record in read_jsonl(args.manifest) if record['sample_id'] not in completed]
    if args.max_samples is not None:
        pending = pending[:args.max_samples]
    print(f'Run id: {run_id}; output: {output_path}')
    print(f'Manifest samples: {len(read_jsonl(args.manifest))}; running: {len(pending)}; resumed: {len(completed)}')
    if pending:
        model, preprocessor = load_model(args.model_path, precision='fp16')
        with output_path.open('a', encoding='utf-8') as handle:
            for item in tqdm(pending, desc='VStar one-shot random-box corruption'):
                record = {key: item[key] for key in (
                    'sample_id', 'sample_index', 'question_id', 'image', 'question', 'label', 'category',
                    'selected_coordinate_index', 'reference_box', 'random_box', 'random_box_iou_to_reference',
                )}
                try:
                    with Image.open(Path(args.image_dir) / item['image']) as opened:
                        image = opened.convert('RGB')
                    result = one_shot_reference_corruption_infer(
                        model, preprocessor, image, query=None, conversation=item['conversation'], options=item['options'],
                        reference_generated_ids=item['reference_generated_ids'],
                        selected_coordinate_index=item['selected_coordinate_index'], random_box=item['random_box'],
                        sample_id=item['sample_id'], max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, log_path=str(event_log),
                    )
                    prediction = score_options(model, preprocessor, image, item, result.generated_ids,
                                               args.max_new_tokens, args.temperature)
                    record.update({'status': 'ok', 'corruption': result.as_dict(), 'prediction': prediction,
                                   'prediction_correct': prediction == item['label']})
                    if args.verbose:
                        tqdm.write(f"[{item['sample_id']}] q={item['random_box']}; "
                                   f"pred={prediction}, correct={record['prediction_correct']}")
                except Exception as error:
                    record.update({'status': 'error', 'error': f'{type(error).__name__}: {error}'})
                    if args.verbose:
                        tqdm.write(f"[{item['sample_id']}] ERROR: {record['error']}")
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
    records = read_jsonl(output_path) if output_path.exists() else []
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(make_summary(records, args), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Event log: {event_log}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
