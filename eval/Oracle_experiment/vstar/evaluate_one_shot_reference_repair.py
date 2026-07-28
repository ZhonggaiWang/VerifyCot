"""Run one-shot random-box repair from saved VStar online-oracle trajectories."""

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
    one_shot_reference_repair_infer,
)
from utils.coordinate_intervention import box_iou
from verifier.run_paths import resolve_run_output


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--verifier-output', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', default=None,
                        help='Output subdirectory. Defaults to a YYYYMMDD_HHMMSS timestamp.')
    parser.add_argument('--verifier-log', default=None,
                        help='Optional JSONL event log. Defaults to verifier_events.jsonl next to --output.')
    parser.add_argument(
        '--repair-mode',
        choices=(
            'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
            'separated_reference_feedback', 'separated_reference_feedback_v2',
        ),
        default='typed_feedback',
    )
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument(
        '--no-sandbox-refbind', action='store_true',
        help='Backward-compatible alias for --sandbox-refbind-mode text_only.',
    )
    parser.add_argument(
        '--sandbox-refbind-mode', choices=('bind', 'text_only', 'skip_q_refbind'), default='bind',
        help=(
            'bind: normal q REFbind; text_only: q has no <coor> tag; '
            'skip_q_refbind: preserve <coor>q</coor> text but skip only V(q).'
        ),
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Print the single verifier decision and repair box for every sample.')
    return parser.parse_args()


def score_options(model, preprocessor, image, item, generated_ids, max_new_tokens, temperature):
    batch = _build_inference_batch(
        preprocessor, image, conversation=item['conversation'], options=item['options']
    )
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
    successful = [record for record in records if record.get('status') == 'ok']
    errors = [record for record in records if record.get('status') == 'error']
    by_step = defaultdict(list)
    for record in successful:
        by_step[int(record['selected_coordinate_index'])].append(record)

    def metrics(items):
        if not items:
            return {'samples': 0}
        correct = sum(bool(item.get('prediction_correct')) for item in items)
        exact_copy = 0
        repair_success = 0
        iou_gain = []
        for item in items:
            repair_event = item.get('repair', {}).get('events', [{}])[0]
            random_box, replacement_box, reference_box = (
                item.get('random_box'), repair_event.get('replacement_box'), item.get('reference_box')
            )
            if random_box and replacement_box:
                exact_copy += all(abs(float(left) - float(right)) <= 1e-9
                                  for left, right in zip(random_box, replacement_box))
            if replacement_box and reference_box:
                replacement_iou = box_iou(replacement_box, reference_box)
                repair_success += replacement_iou >= 0.5
                iou_gain.append(replacement_iou - box_iou(random_box, reference_box))
        return {
            'samples': len(items),
            'correct_count': correct,
            'accuracy': correct / len(items),
            'exact_copy_count': exact_copy,
            'exact_copy_rate': exact_copy / len(items),
            'replacement_iou_ge_0_5_count': repair_success,
            'replacement_iou_ge_0_5_rate': repair_success / len(items),
            'mean_replacement_iou_gain': sum(iou_gain) / len(iou_gain) if iou_gain else None,
        }

    return {
        'total_records': len(records),
        'successful_records': len(successful),
        'error_records': len(errors),
        'all_samples': metrics(successful),
        'by_selected_coordinate_index': {
            str(step): metrics(items) for step, items in sorted(by_step.items())
        },
        'settings': {
            'repair_mode': (
                f'{args.repair_mode}_text_only_q'
                if args.sandbox_refbind_mode == 'text_only' else args.repair_mode
            ),
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'mode': 'one_shot_reference_random_box_repair',
            'initial_verification_count': 1,
            'replacement_second_verification': False,
            'later_coordinate_verification': False,
            'sandbox_refbind_for_random_box': args.sandbox_refbind_mode == 'bind',
            'sandbox_refbind_mode': args.sandbox_refbind_mode,
        },
    }


def main():
    args = parse_args()
    if args.no_sandbox_refbind:
        if args.sandbox_refbind_mode != 'bind':
            raise ValueError('use either --no-sandbox-refbind or --sandbox-refbind-mode, not both')
        args.sandbox_refbind_mode = 'text_only'
    output_path, run_id = resolve_run_output(args.output, args.run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_log_path = Path(args.verifier_log) if args.verifier_log else output_path.with_name('verifier_events.jsonl')
    existing = [] if args.no_resume or not output_path.exists() else read_jsonl(output_path)
    completed = {record['sample_id'] for record in existing if record.get('status') == 'ok'}
    records = [record for record in read_jsonl(args.manifest) if record['sample_id'] not in completed]
    if args.max_samples is not None:
        records = records[:args.max_samples]
    print(f'Run id: {run_id}; output: {output_path}')
    print(f'Manifest samples: {len(read_jsonl(args.manifest))}; running: {len(records)}; resumed: {len(completed)}')
    if not records:
        summary_path = output_path.with_suffix('.summary.json')
        summary = make_summary(existing, args)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'Summary: {summary_path}')
        return
    model, preprocessor = load_model(args.model_path, precision='fp16')
    with output_path.open('a', encoding='utf-8') as handle:
        for item in tqdm(records, desc='VStar one-shot reference repair'):
            record = {key: item[key] for key in (
                'sample_id', 'sample_index', 'question_id', 'image', 'question',
                'label', 'category', 'selected_coordinate_index', 'reference_box',
                'random_box', 'random_box_iou_to_reference',
            )}
            try:
                with Image.open(Path(args.image_dir) / item['image']) as opened:
                    image = opened.convert('RGB')
                common_kwargs = dict(
                    model=model, preprocessor=preprocessor, image=image, query=None,
                    conversation=item['conversation'], options=item['options'],
                    reference_generated_ids=item['reference_generated_ids'],
                    selected_coordinate_index=item['selected_coordinate_index'],
                    random_box=item['random_box'], sample_id=item['sample_id'],
                    oracle_file=args.verifier_output,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    log_path=str(verifier_log_path),
                )
                result = one_shot_reference_repair_infer(
                    repair_mode=args.repair_mode,
                    sandbox_refbind_mode=args.sandbox_refbind_mode,
                    **common_kwargs,
                )
                event = result.events[0]
                repair_box = event['replacement_box']
                prediction = score_options(
                    model, preprocessor, image, item, result.generated_ids,
                    args.max_new_tokens, args.temperature,
                )
                record.update({
                    'status': 'ok',
                    'repair': result.as_dict(),
                    'replacement_iou_to_reference': box_iou(repair_box, item['reference_box']),
                    'prediction': prediction,
                    'prediction_correct': prediction == item['label'],
                })
                if args.verbose:
                    tqdm.write(
                        f"[{item['sample_id']}] step={item['selected_coordinate_index']} "
                        f"q={event['random_box']} -> {event['initial_verdict']}/"
                        f"{event['initial_reason']} -> r={repair_box}; "
                        f"IoU(r, ref)={record['replacement_iou_to_reference']:.3f}; "
                        f"pred={prediction}, correct={record['prediction_correct']}"
                    )
            except Exception as error:
                record.update({'status': 'error', 'error': f'{type(error).__name__}: {error}'})
                if args.verbose:
                    tqdm.write(f"[{item['sample_id']}] ERROR: {record['error']}")
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
    print(f'Verifier event log: {verifier_log_path}')
    all_records = read_jsonl(output_path)
    summary = make_summary(all_records, args)
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
