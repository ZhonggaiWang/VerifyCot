"""Evaluate a perfect selective verifier plus perfect grounding expert on VStar.

The model freely proposes every coordinate.  At each completed ``<coor>``, a
conservative explicit-alias matcher decides whether the local reference has a
unique annotated VStar target.  Only in that matchable case is candidate IoU
measured.  A candidate below ``--iou-threshold`` is replaced by the target GT
box before REFbind; matched candidates above threshold and unmatchable
candidates are committed unchanged.

This is intentionally an oracle *routing* upper bound, not the existing
online oracle that forces every matchable coordinate to GT.
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
    selective_oracle_router_infer,
)
from grounding_control.run_paths import (
    create_run_layout,
    write_run_config,
    write_run_status,
)


ORACLE_BOX_COORDINATE_SYSTEM = 'normalized_xyxy_on_center_padded_square'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument(
        '--baseline-results',
        default='output/vstar/online_oracle/full_238_padding_fix/results.jsonl',
        help='Formal padding-fixed online-oracle run providing untouched baseline CoTs and GT targets.',
    )
    parser.add_argument('--image-dir', default='/data/zhonggai/VStar')
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Legacy output filename. Omit it to use the canonical VStar '
            'routing run layout.'
        ),
    )
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-id', default=None,
                        help='Output subdirectory; defaults to a timestamp.')
    parser.add_argument('--verifier-log', default=None,
                        help='Defaults to verifier_events.jsonl next to the resolved output.')
    parser.add_argument(
        '--iou-threshold', type=float, default=0.1,
        help='Route a uniquely matchable candidate to GT when candidate IoU is below this value.',
    )
    parser.add_argument('--context-window-tokens', type=int, default=48)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--likelihood-reduction', choices=('mean', 'sum'), default='mean')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-id', default=None,
                        help='Optional single ID, for example main:9.')
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


def transition_counts(records):
    counts = Counter({
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'correct_to_correct': 0,
        'wrong_to_wrong': 0,
    })
    for record in records:
        before = bool(record['baseline_prediction_correct'])
        after = bool(record['router_prediction_correct'])
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
    router_correct = sum(record['router_prediction_correct'] for record in records)
    return {
        'samples': len(records),
        'baseline_correct_count': baseline_correct,
        'baseline_accuracy': baseline_correct / len(records),
        'router_correct_count': router_correct,
        'router_accuracy': router_correct / len(records),
        'router_minus_baseline': (router_correct - baseline_correct) / len(records),
        'answer_changed_count': sum(
            record['baseline_prediction'] != record['router_prediction']
            for record in records
        ),
        'correctness_transitions': transitions,
        'mcnemar_exact_two_sided_pvalue': exact_mcnemar_pvalue(
            transitions['wrong_to_correct'], transitions['correct_to_wrong']
        ),
    }


def _mean(values):
    return None if not values else sum(values) / len(values)


def routing_metrics(records):
    events = [
        event
        for record in records
        for event in record['intervention']['events']
    ]
    matched = [
        event for event in events
        if event['match_status'] == 'matched_unique_explicit_target'
    ]
    routed = [
        event for event in events
        if event['router_action'] == 'routed_to_oracle_grounder'
    ]
    verified = [
        event for event in events
        if event['router_action'] == 'verified_accept'
    ]
    unverified = [
        event for event in events
        if event['router_action'] == 'unverifiable_accept'
    ]
    candidate_ious = [float(event['candidate_iou_to_gt']) for event in matched]
    committed_ious = [float(event['committed_iou_to_gt']) for event in matched]
    first_route_positions = Counter(
        next(
            (
                event['grounding_step']
                for event in record['intervention']['events']
                if event['router_action'] == 'routed_to_oracle_grounder'
            ),
            None,
        )
        for record in records
    )
    first_route_positions.pop(None, None)
    return {
        'samples': len(records),
        'samples_with_matchable_coordinate': sum(
            any(
                event['match_status'] == 'matched_unique_explicit_target'
                for event in record['intervention']['events']
            )
            for record in records
        ),
        'samples_routed_to_oracle_grounder': sum(
            any(
                event['router_action'] == 'routed_to_oracle_grounder'
                for event in record['intervention']['events']
            )
            for record in records
        ),
        'coordinate_event_count': len(events),
        'matchable_coordinate_count': len(matched),
        'routed_coordinate_count': len(routed),
        'verified_accepted_coordinate_count': len(verified),
        'unverifiable_accepted_coordinate_count': len(unverified),
        'route_rate_among_matchable_coordinates': (
            None if not matched else len(routed) / len(matched)
        ),
        'mean_candidate_iou_to_gt_on_matchable_coordinates': _mean(candidate_ious),
        'mean_committed_iou_to_gt_on_matchable_coordinates': _mean(committed_ious),
        'committed_iou_ge_0_5_count': sum(value >= 0.5 for value in committed_ious),
        'committed_iou_ge_0_5_rate': (
            None if not committed_ious
            else sum(value >= 0.5 for value in committed_ious) / len(committed_ious)
        ),
        'first_route_position_counts': {
            str(position): count
            for position, count in sorted(first_route_positions.items())
        },
    }


def subset_summary(records):
    metrics = paired_metrics(records)
    metrics['routing'] = routing_metrics(records)
    return metrics


def make_summary(records, args, run_id):
    successful = [record for record in records if record.get('status') == 'ok']
    complete = [
        record for record in successful
        if record.get('has_complete_question_target_coverage')
    ]
    by_category = {
        str(category): subset_summary([
            record for record in successful if record.get('category') == category
        ])
        for category in sorted({record.get('category') for record in successful})
    }
    return {
        'run_id': run_id,
        'total_records': len(records),
        'successful_records': len(successful),
        'error_records': len(records) - len(successful),
        'all_samples': subset_summary(successful),
        'complete_target_coverage_subset': subset_summary(complete),
        'by_category': by_category,
        'settings': {
            'mode': 'online_selective_oracle_router_grounder',
            'baseline_results': args.baseline_results,
            'iou_threshold': args.iou_threshold,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'likelihood_reduction': args.likelihood_reduction,
            'context_window_tokens': args.context_window_tokens,
            'alias_policy': 'latest_unique_longest_explicit_alias',
            'unmatched_policy': 'unverifiable_accept',
            'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
            'kv_cache': False,
        },
    }


def main():
    args = parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError('--iou-threshold must be in [0, 1]')
    if args.context_window_tokens <= 0:
        raise ValueError('--context-window-tokens must be positive')
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')

    iou_setting = 'iou_' + format(args.iou_threshold, 'g').replace('.', 'p')
    layout = create_run_layout(
        dataset='vstar',
        split='full_238',
        study='routing',
        method='oracle_verifier__oracle_experts',
        setting=iou_setting,
        run_id=args.run_id,
        output=args.output,
        output_root=args.output_root,
    )
    layout.ensure_run_directories()
    output_path = layout.results_path
    run_id = layout.run_id
    verifier_log_path = (
        Path(args.verifier_log)
        if args.verifier_log else layout.events_path
    )
    write_run_config(layout, {
        'command': list(sys.argv),
        'arguments': vars(args),
        'inputs': {
            'baseline_results': args.baseline_results,
            'image_dir': args.image_dir,
        },
        'components': {
            'generator': args.model_path,
            'verifier': 'oracle_iou',
            'grounder': 'oracle',
        },
        'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
    })
    write_run_status(layout, 'running', completed_records=0)
    existing = [] if args.no_resume or not output_path.exists() else read_jsonl(output_path)
    incompatible = [
        record for record in existing
        if record.get('oracle_box_coordinate_system') != ORACLE_BOX_COORDINATE_SYSTEM
        or record.get('intervention', {}).get('mode') != 'online_selective_oracle_router_grounder'
        or float(record.get('intervention', {}).get('iou_threshold', -1)) != args.iou_threshold
    ]
    if incompatible:
        raise ValueError(
            'existing output has a different coordinate system, routing mode, or IoU threshold; '
            'choose a new --run-id/output or use --no-resume'
        )
    completed = {
        record['question_id'] for record in existing if record.get('status') == 'ok'
    }

    sources = [
        record for record in read_jsonl(args.baseline_results)
        if record.get('status') == 'ok' and record.get('baseline')
    ]
    incompatible_sources = [
        record for record in sources
        if record.get('oracle_box_coordinate_system') != ORACLE_BOX_COORDINATE_SYSTEM
    ]
    if incompatible_sources:
        raise ValueError(
            'baseline-results contains old/unknown coordinate system; use the padding-fixed run'
        )
    if args.sample_id is not None:
        sources = [
            record for record in sources if record.get('question_id') == args.sample_id
        ]
        if len(sources) != 1:
            raise ValueError(
                f'expected exactly one baseline record for --sample-id {args.sample_id!r}, '
                f'found {len(sources)}'
            )
    else:
        end = len(sources) if args.max_samples is None else min(
            len(sources), args.start_index + args.max_samples
        )
        sources = sources[args.start_index:end]
    pending = [record for record in sources if record['question_id'] not in completed]

    print(f'Run id: {run_id}; output: {output_path}')
    print(
        f'Baseline records selected: {len(sources)}; '
        f'running: {len(pending)}; resumed: {len(completed)}'
    )
    if pending:
        model, preprocessor = load_model(args.model_path, precision='fp16')

    output_mode = 'w' if args.no_resume else 'a'
    with output_path.open(output_mode, encoding='utf-8') as handle:
        for source in tqdm(pending, desc='VStar selective oracle router'):
            record = {
                key: source.get(key) for key in (
                    'sample_index', 'question_id', 'image', 'category', 'question',
                    'options', 'label', 'source_jsonl_label', 'oracle_targets',
                    'source_oracle_boxes', 'oracle_box_coordinate_system',
                    'source_image_size', 'has_complete_question_target_coverage',
                )
            }
            baseline = source['baseline']
            record.update({
                'baseline': baseline,
                'baseline_prediction': source['baseline_prediction'],
                'baseline_answer': source.get('baseline_answer'),
                'baseline_prediction_correct': source['baseline_prediction'] == source['label'],
            })
            try:
                image_path = Path(args.image_dir) / source['image']
                with Image.open(image_path) as opened:
                    image = opened.convert('RGB')
                expected_size = source.get('source_image_size', {})
                if image.size != (expected_size.get('width'), expected_size.get('height')):
                    raise ValueError(
                        f'image size {image.size} does not match audited size {expected_size}'
                    )
                conversation = make_conversation(source['question'])
                routed = selective_oracle_router_infer(
                    model=model,
                    preprocessor=preprocessor,
                    image=image,
                    query=None,
                    cot=True,
                    sample_id=source['question_id'],
                    oracle_targets=source['oracle_targets'],
                    iou_threshold=args.iou_threshold,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    conversation=conversation,
                    options=source['options'],
                    context_window_tokens=args.context_window_tokens,
                    log_path=str(verifier_log_path),
                )
                prediction = score_options(
                    model,
                    preprocessor,
                    image,
                    conversation,
                    source['options'],
                    routed['selective_router']['generated_ids'],
                    args.max_new_tokens,
                    args.temperature,
                    args.likelihood_reduction,
                )
                record.update(routed)
                record.update({
                    'router_prediction': prediction,
                    'router_answer': source['options'][prediction],
                    'router_prediction_correct': prediction == source['label'],
                    'status': 'ok',
                })
                if args.verbose:
                    actions = Counter(
                        event['router_action'] for event in routed['intervention']['events']
                    )
                    tqdm.write(
                        f'[{source["question_id"]}] actions={dict(actions)}; '
                        f'pred {source["baseline_prediction"]}->{prediction}; '
                        f'correct={record["router_prediction_correct"]}'
                    )
            except Exception as error:
                record.update({
                    'status': 'error',
                    'error': f'{type(error).__name__}: {error}',
                })
                if args.verbose:
                    tqdm.write(f'[{source["question_id"]}] ERROR: {record["error"]}')
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()

    records = read_jsonl(output_path)
    summary = make_summary(records, args, run_id)
    summary_path = layout.summary_path
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    write_run_status(
        layout,
        'completed' if summary['error_records'] == 0
        else 'completed_with_errors',
        completed_records=summary['successful_records'],
        error_records=summary['error_records'],
        summary_path=str(summary_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Verifier/router events: {verifier_log_path}')
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
