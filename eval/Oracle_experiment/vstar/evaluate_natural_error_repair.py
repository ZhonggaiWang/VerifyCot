"""Repair the first strictly verifiable natural baseline grounding error.

This evaluator consumes the normal baseline trajectories already saved by the
VStar online-oracle run.  It rematches *baseline* coordinates from scratch,
selects the first unique explicit target whose IoU is below the configured
threshold, supplies only ``misaligned/wrong_object`` feedback, generates one
text-only-q concise repair, and then releases all later CoT tokens.

Samples with no eligible natural error pass through unchanged and remain in
the full-dataset denominator.  GT coordinates are used only by the simulated
checker and evaluator; they are never included in the repair prompt.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import ALL_IMG_TOKENS_STR, COT_ACTIVATION, DEFAULT_GRD_TOKEN
from model.load_model import (
    _build_inference_batch,
    load_model,
    one_shot_natural_error_repair_infer,
)
from utils.coordinate_intervention import box_iou
from verifier import audit_natural_coordinates
from verifier.run_paths import resolve_run_output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument(
        '--baseline-results',
        default='output/vstar/online_oracle/full_238_padding_fix/results.jsonl',
        help='Saved online-oracle results containing an untouched baseline trajectory.',
    )
    parser.add_argument('--image-dir', default='/data/zhonggai/VStar')
    parser.add_argument(
        '--output',
        default='output/vstar/natural_error_repair/text_only_concise/results.jsonl',
    )
    parser.add_argument('--run-id', default=None,
                        help='Output subdirectory; defaults to a timestamp.')
    parser.add_argument('--verifier-log', default=None,
                        help='Defaults to verifier_events.jsonl next to the resolved output.')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--context-window-tokens', type=int, default=48)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--likelihood-reduction', choices=('mean', 'sum'), default='mean')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-id', default=None,
                        help='Optional single sample such as main:9 for a smoke test.')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_conversation(question):
    return [{
        'from': 'human',
        'value': (
            ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n'
            + question + ' ' + COT_ACTIVATION
        ),
    }]


def score_options(model, preprocessor, image, conversation, options, generated_ids,
                  max_new_tokens, temperature, likelihood_reduction):
    batch = _build_inference_batch(
        preprocessor, image, conversation=conversation, options=options
    )
    prompt = batch['input_ids']
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    thought_ids = torch.cat([
        prompt.to(model.device),
        torch.tensor(
            generated_ids, dtype=prompt.dtype, device=model.device
        ).unsqueeze(0),
    ], dim=1)
    with torch.inference_mode():
        prediction, _ = model.calculate_options(
            batch,
            cot=True,
            further_instruct=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            likelihood_reduction=likelihood_reduction,
            thought_override_ids=thought_ids,
        )
    return int(prediction)


def intersection_over_reference(candidate, reference):
    x1 = max(float(candidate[0]), float(reference[0]))
    y1 = max(float(candidate[1]), float(reference[1]))
    x2 = min(float(candidate[2]), float(reference[2]))
    y2 = min(float(candidate[3]), float(reference[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    reference_area = max(0.0, float(reference[2]) - float(reference[0])) * max(
        0.0, float(reference[3]) - float(reference[1])
    )
    return 0.0 if reference_area <= 0 else intersection / reference_area


def transition_counts(records):
    counts = Counter({
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'correct_to_correct': 0,
        'wrong_to_wrong': 0,
    })
    for record in records:
        before = bool(record['baseline_prediction_correct'])
        after = bool(record['repair_prediction_correct'])
        if before and not after:
            counts['correct_to_wrong'] += 1
        elif not before and after:
            counts['wrong_to_correct'] += 1
        elif before:
            counts['correct_to_correct'] += 1
        else:
            counts['wrong_to_wrong'] += 1
    return dict(counts)


def exact_mcnemar_pvalue(wrong_to_correct, correct_to_wrong):
    discordant = int(wrong_to_correct) + int(correct_to_wrong)
    if discordant == 0:
        return 1.0
    tail = min(int(wrong_to_correct), int(correct_to_wrong))
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / float(2 ** discordant))


def paired_metrics(records):
    if not records:
        return {'samples': 0}
    transitions = transition_counts(records)
    baseline_correct = sum(record['baseline_prediction_correct'] for record in records)
    repair_correct = sum(record['repair_prediction_correct'] for record in records)
    return {
        'samples': len(records),
        'baseline_correct_count': baseline_correct,
        'baseline_accuracy': baseline_correct / len(records),
        'repair_correct_count': repair_correct,
        'repair_accuracy': repair_correct / len(records),
        'repair_minus_baseline': (repair_correct - baseline_correct) / len(records),
        'answer_changed_count': sum(
            record['baseline_prediction'] != record['repair_prediction']
            for record in records
        ),
        'correctness_transitions': transitions,
        'mcnemar_exact_two_sided_pvalue': exact_mcnemar_pvalue(
            transitions['wrong_to_correct'], transitions['correct_to_wrong']
        ),
    }


def grounding_metrics(records):
    if not records:
        return {'samples': 0}
    exact_copy = sum(
        all(abs(float(left) - float(right)) <= 1e-9
            for left, right in zip(record['baseline_box'], record['replacement_box']))
        for record in records
    )
    initial_ious = [float(record['baseline_iou_to_gt']) for record in records]
    replacement_ious = [float(record['replacement_iou_to_gt']) for record in records]
    gains = [
        replacement - initial
        for initial, replacement in zip(initial_ious, replacement_ious)
    ]
    return {
        'samples': len(records),
        'exact_copy_count': exact_copy,
        'exact_copy_rate': exact_copy / len(records),
        'coordinate_changed_count': len(records) - exact_copy,
        'coordinate_changed_rate': 1.0 - exact_copy / len(records),
        'mean_baseline_iou_to_gt': sum(initial_ious) / len(records),
        'mean_replacement_iou_to_gt': sum(replacement_ious) / len(records),
        'mean_iou_gain': sum(gains) / len(records),
        'iou_improved_count': sum(gain > 0 for gain in gains),
        'iou_improved_rate': sum(gain > 0 for gain in gains) / len(records),
        'replacement_iou_ge_0_3_count': sum(value >= 0.3 for value in replacement_ious),
        'replacement_iou_ge_0_3_rate': sum(value >= 0.3 for value in replacement_ious) / len(records),
        'replacement_iou_ge_0_5_count': sum(value >= 0.5 for value in replacement_ious),
        'replacement_iou_ge_0_5_rate': sum(value >= 0.5 for value in replacement_ious) / len(records),
        'crossed_to_iou_ge_0_3_count': sum(
            initial < 0.3 <= replacement
            for initial, replacement in zip(initial_ious, replacement_ious)
        ),
        'crossed_to_iou_ge_0_5_count': sum(
            initial < 0.5 <= replacement
            for initial, replacement in zip(initial_ious, replacement_ious)
        ),
        'mean_baseline_gt_coverage': sum(
            record['baseline_gt_coverage'] for record in records
        ) / len(records),
        'mean_replacement_gt_coverage': sum(
            record['replacement_gt_coverage'] for record in records
        ) / len(records),
    }


def triggered_subset_metrics(records):
    metrics = paired_metrics(records)
    metrics['grounding'] = grounding_metrics(records)
    return metrics


def area_bin(record):
    area = float(record['gt_box_area'])
    if area < 0.001:
        return 'tiny_lt_0.001'
    if area < 0.01:
        return 'small_0.001_to_0.01'
    return 'medium_or_large_ge_0.01'


def make_summary(records, args, run_id):
    successful = [record for record in records if record.get('status') == 'ok']
    triggered = [record for record in successful if record.get('checker_triggered')]
    strict_matched = [
        record for record in successful
        if record.get('natural_grounding_audit', {}).get('has_eligible_coordinate')
    ]
    no_trigger = [record for record in successful if not record.get('checker_triggered')]

    by_step = defaultdict(list)
    by_area = defaultdict(list)
    for record in triggered:
        by_step[int(record['selected_coordinate_index'])].append(record)
        by_area[area_bin(record)].append(record)

    by_category = {}
    for category in sorted({record.get('category') for record in successful}):
        category_records = [
            record for record in successful if record.get('category') == category
        ]
        category_triggered = [
            record for record in category_records if record.get('checker_triggered')
        ]
        by_category[str(category)] = {
            'all_samples': paired_metrics(category_records),
            'checker_triggered_subset': triggered_subset_metrics(category_triggered),
        }

    total_coordinate_count = sum(
        record['natural_grounding_audit']['coordinate_count'] for record in successful
    )
    eligible_coordinate_count = sum(
        record['natural_grounding_audit']['eligible_coordinate_count']
        for record in successful
    )
    error_coordinate_count = sum(
        record['natural_grounding_audit']['natural_error_coordinate_count']
        for record in successful
    )
    return {
        'run_id': run_id,
        'total_records': len(records),
        'successful_records': len(successful),
        'error_records': len(records) - len(successful),
        'all_samples': paired_metrics(successful),
        'strict_matched_subset': paired_metrics(strict_matched),
        'checker_triggered_subset': triggered_subset_metrics(triggered),
        'no_trigger_passthrough_subset': paired_metrics(no_trigger),
        'checker_coverage': {
            'samples_with_eligible_coordinate': len(strict_matched),
            'samples_with_natural_error': len(triggered),
            'samples_without_natural_error': len(successful) - len(triggered),
            'total_coordinate_count': total_coordinate_count,
            'eligible_coordinate_count': eligible_coordinate_count,
            'natural_error_coordinate_count': error_coordinate_count,
        },
        'by_selected_coordinate_index': {
            str(step): triggered_subset_metrics(items)
            for step, items in sorted(by_step.items())
        },
        'by_selected_gt_box_area': {
            name: triggered_subset_metrics(items)
            for name, items in sorted(by_area.items())
        },
        'by_category': by_category,
        'settings': {
            'mode': 'first_natural_baseline_error_one_shot_repair',
            'baseline_results': args.baseline_results,
            'repair_mode': 'concise_typed_feedback',
            'sandbox_refbind_mode': 'text_only',
            'checker_verdict': 'misaligned',
            'checker_reason': 'wrong_object',
            'checker_confidence': 1.0,
            'iou_threshold': args.iou_threshold,
            'selection_policy': 'first_strictly_matched_coordinate_below_threshold',
            'unmatched_policy': 'not_applicable_and_unchanged',
            'no_trigger_policy': 'baseline_passthrough',
            'replacement_second_verification': False,
            'later_coordinate_verification': False,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'likelihood_reduction': args.likelihood_reduction,
            'context_window_tokens': args.context_window_tokens,
        },
    }


def main():
    args = parse_args()
    if not 0 <= args.iou_threshold <= 1:
        raise ValueError('--iou-threshold must be in [0, 1]')
    if args.context_window_tokens <= 0:
        raise ValueError('--context-window-tokens must be positive')
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')

    output_path, run_id = resolve_run_output(args.output, args.run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verifier_log_path = (
        Path(args.verifier_log)
        if args.verifier_log else output_path.with_name('verifier_events.jsonl')
    )
    existing = [] if args.no_resume or not output_path.exists() else read_jsonl(output_path)
    completed = {
        record['question_id'] for record in existing if record.get('status') == 'ok'
    }
    sources = [
        record for record in read_jsonl(args.baseline_results)
        if record.get('status') == 'ok' and record.get('baseline')
    ]
    if args.sample_id is not None:
        sources = [
            record for record in sources if record.get('question_id') == args.sample_id
        ]
        if len(sources) != 1:
            raise ValueError(
                f'expected one baseline record for --sample-id {args.sample_id!r}, '
                f'found {len(sources)}'
            )
    else:
        end = len(sources) if args.max_samples is None else min(
            len(sources), args.start_index + args.max_samples
        )
        sources = sources[args.start_index:end]
    pending = [
        record for record in sources if record['question_id'] not in completed
    ]
    print(f'Run id: {run_id}; output: {output_path}')
    print(
        f'Baseline records selected: {len(sources)}; '
        f'running: {len(pending)}; resumed: {len(completed)}'
    )

    if pending:
        model, preprocessor = load_model(args.model_path, precision='fp16')
    with output_path.open('a', encoding='utf-8') as handle:
        for source in tqdm(pending, desc='VStar natural-error oracle repair'):
            baseline = source['baseline']
            record = {
                key: source.get(key) for key in (
                    'sample_index', 'question_id', 'image', 'category', 'question',
                    'options', 'label', 'source_jsonl_label', 'oracle_targets',
                    'has_complete_question_target_coverage',
                )
            }
            record.update({
                'baseline': baseline,
                'baseline_prediction': source['baseline_prediction'],
                'baseline_answer': source.get('baseline_answer'),
                'baseline_prediction_correct': (
                    source['baseline_prediction'] == source['label']
                ),
            })
            try:
                audit = audit_natural_coordinates(
                    preprocessor.tokenizer,
                    baseline['generated_ids'],
                    baseline['boxes'],
                    source['oracle_targets'],
                    iou_threshold=args.iou_threshold,
                    context_window_tokens=args.context_window_tokens,
                )
                record['natural_grounding_audit'] = audit
                selected = audit['selected_first_natural_error']
                if selected is None:
                    record.update({
                        'status': 'ok',
                        'checker_triggered': False,
                        'checker_action': (
                            'no_eligible_coordinate_passthrough'
                            if not audit['has_eligible_coordinate']
                            else 'all_eligible_coordinates_aligned_passthrough'
                        ),
                        'repair': None,
                        'repair_prediction': source['baseline_prediction'],
                        'repair_answer': source.get('baseline_answer'),
                        'repair_prediction_correct': (
                            source['baseline_prediction'] == source['label']
                        ),
                    })
                else:
                    sample_id = source['question_id']
                    image_path = Path(args.image_dir) / source['image']
                    with Image.open(image_path) as opened:
                        image = opened.convert('RGB')
                    conversation = make_conversation(source['question'])
                    result = one_shot_natural_error_repair_infer(
                        model=model,
                        preprocessor=preprocessor,
                        image=image,
                        query=None,
                        conversation=conversation,
                        options=source['options'],
                        baseline_generated_ids=baseline['generated_ids'],
                        selected_coordinate_index=selected['coordinate_index'],
                        baseline_box=selected['baseline_box'],
                        oracle_target_box=selected['oracle_box'],
                        target_object=selected['target_object'],
                        sample_id=sample_id,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        repair_mode='concise_typed_feedback',
                        log_path=str(verifier_log_path),
                    )
                    repair_event = result.events[0]
                    replacement_box = repair_event['replacement_box']
                    prediction = score_options(
                        model,
                        preprocessor,
                        image,
                        conversation,
                        source['options'],
                        result.generated_ids,
                        args.max_new_tokens,
                        args.temperature,
                        args.likelihood_reduction,
                    )
                    gt_box = selected['oracle_box']
                    record.update({
                        'status': 'ok',
                        'checker_triggered': True,
                        'checker_action': 'misaligned_wrong_object_then_one_shot_repair',
                        'selected_coordinate_index': selected['coordinate_index'],
                        'selected_object_reference': selected['context'],
                        'target_object': selected['target_object'],
                        'matched_alias': selected['matched_alias'],
                        'baseline_box': selected['baseline_box'],
                        'gt_box': gt_box,
                        'gt_box_area': selected['gt_box_area'],
                        'baseline_iou_to_gt': selected['baseline_iou_to_gt'],
                        'baseline_gt_coverage': selected['baseline_gt_coverage'],
                        'repair': result.as_dict(),
                        'replacement_box': replacement_box,
                        'replacement_iou_to_gt': box_iou(replacement_box, gt_box),
                        'replacement_gt_coverage': intersection_over_reference(
                            replacement_box, gt_box
                        ),
                        'repair_prediction': prediction,
                        'repair_answer': source['options'][prediction],
                        'repair_prediction_correct': prediction == source['label'],
                    })
                    if args.verbose:
                        tqdm.write(
                            f'[{sample_id}] step={selected["coordinate_index"]} '
                            f'{selected["target_object"]}: '
                            f'IoU(q,GT)={record["baseline_iou_to_gt"]:.3f} -> '
                            f'IoU(r,GT)={record["replacement_iou_to_gt"]:.3f}; '
                            f'pred {record["baseline_prediction"]}->{prediction}, '
                            f'correct={record["repair_prediction_correct"]}'
                        )
            except Exception as error:
                record.update({
                    'status': 'error',
                    'error': f'{type(error).__name__}: {error}',
                })
                if args.verbose:
                    tqdm.write(
                        f'[{source.get("question_id")}] ERROR: {record["error"]}'
                    )
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    all_records = read_jsonl(output_path)
    summary = make_summary(all_records, args, run_id)
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'Verifier event log: {verifier_log_path}')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
