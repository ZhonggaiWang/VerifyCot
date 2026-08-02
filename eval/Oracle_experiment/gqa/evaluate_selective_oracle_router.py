"""Evaluate a perfect selective verifier plus perfect grounder on GQA.

This is the GQA counterpart of the VStar selective-oracle-router experiment.
The model freely proposes each coordinate.  When the local object reference
uniquely matches an annotated GQA target, the oracle verifier measures IoU:

* candidate IoU >= ``--iou-threshold``: accept the model coordinate;
* candidate IoU < ``--iou-threshold``: replace it with the target GT box;
* no unique explicit target match: accept without verification.

Every decision is made before REFbind.  The committed coordinate is then
replayed through Volcano's normal path, so its coordinate text and bound
visual feature are identical.  The routed CoT is evaluated with GQA's
original short-final-answer turn.
"""

import argparse
import fcntl
import json
import math
import sys
import time
from collections import Counter
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
    selective_oracle_router_infer,
)
from common import (
    answer_is_correct,
    generate_gqa_final_answer,
    make_gqa_conversation,
)
from grounding_control.run_paths import (
    create_exact_output_layout,
    create_run_layout,
    write_run_config,
    write_run_status,
)


ORACLE_BOX_COORDINATE_SYSTEM = 'normalized_xyxy_on_center_padded_square'
ROUTER_MODE = 'online_selective_oracle_router_grounder'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument(
        '--manifest-path',
        default='output/gqa/annotations/oracle_val_1000/manifest.jsonl',
        help='GQA manifest used to resolve image paths and sample indices.',
    )
    parser.add_argument(
        '--baseline-results',
        default='output/gqa/online_oracle/padding_fix_v1/results.jsonl',
        help='Padding-fixed run providing paired baseline predictions and GT targets.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Exact results JSONL path. Omit it to use the canonical '
            'output/<dataset>/runs/... layout.'
        ),
    )
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--run-split', default='val_1000_dev')
    parser.add_argument(
        '--verifier-log',
        default=None,
        help='Defaults to verifier_events.jsonl next to --output.',
    )
    parser.add_argument(
        '--model-load-lock',
        default=None,
        help='Optional shared lock file that serializes model loading across shards.',
    )
    parser.add_argument(
        '--model-load-lock-timeout-seconds',
        type=float,
        default=300.0,
        help='Maximum time to wait for another shard to finish loading its model.',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.1,
        help='Route a uniquely matchable candidate to GT when IoU is below this value.',
    )
    parser.add_argument('--context-window-tokens', type=int, default=48)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--final-max-new-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument(
        '--sample-id',
        default=None,
        help='Evaluate one GQA question_id, independently of start/max-samples.',
    )
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model_with_optional_lock(args):
    """Serialize only checkpoint loading; inference remains fully parallel."""
    if args.model_load_lock is None:
        return load_model(args.model_path, precision='fp16')

    lock_path = Path(args.model_load_lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.model_load_lock_timeout_seconds
    last_report = 0.0
    with lock_path.open('a+', encoding='utf-8') as lock_handle:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        'timed out waiting for the serialized model-load lock '
                        f'after {args.model_load_lock_timeout_seconds:g}s: {lock_path}'
                    )
                if now - last_report >= 10.0:
                    remaining = max(0.0, deadline - now)
                    print(
                        f'Waiting for model-load lock: {lock_path} '
                        f'({remaining:.0f}s remaining)',
                        flush=True,
                    )
                    last_report = now
                time.sleep(1.0)

        print(f'Acquired model-load lock: {lock_path}', flush=True)
        try:
            model_and_preprocessor = load_model(args.model_path, precision='fp16')
            print('Model loaded; releasing model-load lock.', flush=True)
            return model_and_preprocessor
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def exact_mcnemar_pvalue(wrong_to_correct, correct_to_wrong):
    discordant = int(wrong_to_correct) + int(correct_to_wrong)
    if discordant == 0:
        return 1.0
    tail = min(int(wrong_to_correct), int(correct_to_wrong))
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / float(2 ** discordant))


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
    unverifiable = [
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
        'unverifiable_accepted_coordinate_count': len(unverifiable),
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
    result = paired_metrics(records)
    result['routing'] = routing_metrics(records)
    return result


def make_summary(records, args):
    successful = [record for record in records if record.get('status') == 'ok']
    by_type = {
        type_name: subset_summary([
            record for record in successful
            if record.get('types', {}).get('structural') == type_name
        ])
        for type_name in sorted({
            record.get('types', {}).get('structural')
            for record in successful
        })
    }
    return {
        'total_records': len(records),
        'successful_records': len(successful),
        'error_records': len(records) - len(successful),
        'all_samples': subset_summary(successful),
        'by_structural_type': by_type,
        'settings': {
            'dataset': 'gqa_val_manifest',
            'mode': ROUTER_MODE,
            'manifest_path': args.manifest_path,
            'baseline_results': args.baseline_results,
            'iou_threshold': args.iou_threshold,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'final_max_new_tokens': args.final_max_new_tokens,
            'context_window_tokens': args.context_window_tokens,
            'alias_policy': 'latest_unique_longest_explicit_alias',
            'unmatched_policy': 'unverifiable_accept',
            'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
            'kv_cache': False,
            'model_load_lock': args.model_load_lock,
            'model_load_lock_timeout_seconds': args.model_load_lock_timeout_seconds,
        },
    }


def full_thought_sequences(model, preprocessor, image, conversation, generated_ids):
    """Reconstruct prompt+completion IDs required by GQA's final-answer turn."""
    batch = _build_inference_batch(
        preprocessor, image, conversation=conversation
    )
    prompt = batch['input_ids']
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    completion = torch.tensor(
        generated_ids, dtype=prompt.dtype, device=model.device
    ).unsqueeze(0)
    return torch.cat([prompt.to(model.device), completion], dim=1), prompt.shape[-1]


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
    if args.model_load_lock_timeout_seconds <= 0:
        raise ValueError('--model-load-lock-timeout-seconds must be positive')

    manifest = read_jsonl(args.manifest_path)
    manifest_by_question = {
        str(record['question_id']): record for record in manifest
    }
    if len(manifest_by_question) != len(manifest):
        raise ValueError('manifest contains duplicate question_id values')

    baseline_records = [
        record for record in read_jsonl(args.baseline_results)
        if record.get('status') == 'ok' and record.get('baseline')
    ]
    incompatible_sources = [
        record for record in baseline_records
        if record.get('oracle_box_coordinate_system') != ORACLE_BOX_COORDINATE_SYSTEM
    ]
    if incompatible_sources:
        raise ValueError(
            '--baseline-results contains old/unknown oracle coordinates; '
            'use the padding-fixed GQA run'
        )
    missing_manifest = [
        record['question_id'] for record in baseline_records
        if str(record['question_id']) not in manifest_by_question
    ]
    if missing_manifest:
        raise ValueError(
            f'{len(missing_manifest)} baseline records are absent from the manifest'
        )

    if args.sample_id is not None:
        selected = [
            record for record in baseline_records
            if str(record['question_id']) == str(args.sample_id)
        ]
        if len(selected) != 1:
            raise ValueError(
                f'expected one padding-fixed baseline for {args.sample_id!r}, '
                f'found {len(selected)}'
            )
    else:
        end = (
            len(manifest)
            if args.max_samples is None
            else min(len(manifest), args.start_index + args.max_samples)
        )
        selected = [
            record for record in baseline_records
            if args.start_index <= int(record['sample_index']) < end
        ]

    setting = 'iou_{}'.format(str(args.iou_threshold).replace('.', 'p'))
    if args.output is None:
        layout = create_run_layout(
            dataset='gqa',
            split=args.run_split,
            study='routing',
            method='oracle_verifier__oracle_experts',
            setting=setting,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    else:
        requested_output = Path(args.output)
        layout = create_exact_output_layout(
            dataset='gqa',
            split=args.run_split,
            study='routing',
            method='oracle_verifier__oracle_experts',
            setting=setting,
            run_id=args.run_id or requested_output.parent.name,
            output=requested_output,
        )
    layout.ensure_run_directories()
    output_path = layout.results_path
    verifier_log_path = (
        Path(args.verifier_log)
        if args.verifier_log else layout.events_path
    )
    write_run_config(layout, {
        'command': list(sys.argv),
        'arguments': vars(args),
        'inputs': {
            'manifest': args.manifest_path,
            'baseline_results': args.baseline_results,
        },
        'components': {
            'generator': args.model_path,
            'verifier': 'oracle_iou_threshold',
            'grounder': 'oracle_gt_box',
        },
    })
    write_run_status(layout, 'running', completed_records=0)
    existing = (
        [] if args.no_resume or not output_path.exists()
        else read_jsonl(output_path)
    )
    incompatible = [
        record for record in existing
        if record.get('oracle_box_coordinate_system') != ORACLE_BOX_COORDINATE_SYSTEM
        or record.get('intervention', {}).get('mode') != ROUTER_MODE
        or float(record.get('intervention', {}).get('iou_threshold', -1))
        != args.iou_threshold
    ]
    if incompatible:
        raise ValueError(
            'existing output has a different coordinate system, routing mode, '
            'or IoU threshold; choose another --output or use --no-resume'
        )
    completed = {
        str(record['question_id'])
        for record in existing if record.get('question_id') is not None
    }
    pending = [
        record for record in selected
        if str(record['question_id']) not in completed
    ]

    print(
        f'GQA padding-fixed baselines selected: {len(selected)}; '
        f'evaluating: {len(pending)}; resumed: {len(selected) - len(pending)}'
    )
    print(f'IoU threshold: {args.iou_threshold}; output: {output_path}')
    if pending:
        model, preprocessor = load_model_with_optional_lock(args)

    output_mode = 'w' if args.no_resume else 'a'
    with output_path.open(output_mode, encoding='utf-8') as handle:
        for source in tqdm(pending, desc='GQA selective oracle router'):
            item = manifest_by_question[str(source['question_id'])]
            record = {
                key: source.get(key) for key in (
                    'sample_index', 'question_id', 'image_id', 'question',
                    'answer', 'types', 'target_objects', 'oracle_targets',
                    'excluded_oracle_targets', 'source_image_size',
                    'oracle_box_coordinate_system', 'baseline',
                    'baseline_prediction', 'baseline_prediction_correct',
                    'baseline_final_response',
                )
            }
            try:
                with Image.open(item['image_path']) as opened:
                    image = opened.convert('RGB')
                expected_size = source.get('source_image_size', {})
                if image.size != (
                        expected_size.get('width'), expected_size.get('height')):
                    raise ValueError(
                        f'image size {image.size} does not match padding-fixed '
                        f'baseline size {expected_size}'
                    )
                oracle_targets = source.get('oracle_targets') or []
                if not oracle_targets:
                    raise ValueError('padding-fixed baseline has no unique oracle target')

                conversation = make_gqa_conversation(source['question'])
                routed = selective_oracle_router_infer(
                    model=model,
                    preprocessor=preprocessor,
                    image=image,
                    query=None,
                    cot=True,
                    sample_id=source['question_id'],
                    oracle_targets=oracle_targets,
                    iou_threshold=args.iou_threshold,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    conversation=conversation,
                    context_window_tokens=args.context_window_tokens,
                    log_path=str(verifier_log_path),
                )
                thought_sequences, prompt_length = full_thought_sequences(
                    model,
                    preprocessor,
                    image,
                    conversation,
                    routed['selective_router']['generated_ids'],
                )
                router_final = generate_gqa_final_answer(
                    model,
                    preprocessor,
                    image,
                    thought_sequences,
                    prompt_length,
                    args.final_max_new_tokens,
                    args.temperature,
                )
                record.update(routed)
                record.update({
                    'router_prediction': router_final['prediction'],
                    'router_prediction_correct': answer_is_correct(
                        router_final['prediction'], source['answer']
                    ),
                    'router_final_response': router_final['response'],
                    'status': 'ok',
                })
                if args.verbose:
                    actions = Counter(
                        event['router_action']
                        for event in routed['intervention']['events']
                    )
                    tqdm.write(
                        f'[{source["question_id"]}] actions={dict(actions)}; '
                        f'pred {source["baseline_prediction"]!r}'
                        f'->{router_final["prediction"]!r}; '
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
    summary = make_summary(records, args)
    summary_path = layout.summary_path
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    error_records = summary.get('error_records', 0)
    write_run_status(
        layout,
        'completed' if error_records == 0 else 'completed_with_errors',
        completed_records=summary.get('successful_records', 0),
        error_records=error_records,
        summary_path=str(summary_path),
        events_path=str(verifier_log_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Verifier/router events: {verifier_log_path}')
    print(f'Per-example results: {output_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
