"""Merge sharded GQA intervention results and recompute paired summaries."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grounding_control.run_paths import (
    create_exact_output_layout,
    write_run_config,
    write_run_status,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        choices=('counterfactual', 'oracle', 'selective_router', 'testdev_baseline'),
        required=True,
    )
    input_source = parser.add_mutually_exclusive_group(required=True)
    input_source.add_argument('--input-glob',
                              help='Glob for shard result JSONLs, for example run/shards/*/results.jsonl.')
    input_source.add_argument('--shards-dir',
                              help='Directory containing shard_*/results.jsonl files.')
    parser.add_argument(
        '--expected-shards',
        type=int,
        default=None,
        help=(
            'Required with --shards-dir. Enforces exactly shard_000 through '
            'shard_NNN and rejects missing or stale extra shard results.'
        ),
    )
    parser.add_argument('--output', required=True, help='Merged JSONL output path.')
    parser.add_argument(
        '--events-output',
        default=None,
        help=(
            'Optional merged verifier-event JSONL. For --shards-dir, each '
            'existing shard_*/verifier_events.jsonl is merged in shard order.'
        ),
    )
    parser.add_argument('--settings', default=None,
                        help='Optional JSON object recorded under summary.settings.')
    parser.add_argument(
        '--run-id',
        default=None,
        help='Logical run id shared by every shard; defaults to the output parent name.',
    )
    parser.add_argument(
        '--run-split',
        default=None,
        help='Canonical split identity; defaults from --experiment.',
    )
    return parser.parse_args()


def experiment_identity(experiment, settings):
    if experiment == 'counterfactual':
        perturb_index = settings.get('perturb_index')
        setting = (
            f'index_{perturb_index}'
            if perturb_index is not None
            else settings.get('perturb_position', 'random')
        )
        return 'counterfactual', settings.get('perturb_mode', 'random_box'), setting
    if experiment == 'oracle':
        return 'oracle', 'always_gt', 'default'
    if experiment == 'selective_router':
        threshold = settings.get('iou_threshold', 0.1)
        return (
            'routing',
            'oracle_verifier__oracle_experts',
            f'iou_{str(threshold).replace(".", "p")}',
        )
    if experiment == 'testdev_baseline':
        return 'baseline', 'volcano_7b', 'default'
    raise ValueError(f'unsupported GQA experiment: {experiment}')


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def shard_result_paths(shards_dir, expected_shards):
    shards_dir = Path(shards_dir)
    if expected_shards is None:
        raise ValueError('--expected-shards is required with --shards-dir')
    if expected_shards < 1:
        raise ValueError('--expected-shards must be a positive integer')

    expected = [
        shards_dir / f'shard_{index:03d}' / 'results.jsonl'
        for index in range(expected_shards)
    ]
    found = sorted(shards_dir.glob('shard_*/results.jsonl'))
    expected_set = set(expected)
    found_set = set(found)
    missing = [path for path in expected if path not in found_set]
    unexpected = [path for path in found if path not in expected_set]
    if missing or unexpected:
        details = []
        if missing:
            details.append('missing=' + ','.join(str(path) for path in missing))
        if unexpected:
            details.append(
                'unexpected=' + ','.join(str(path) for path in unexpected)
            )
        raise ValueError(
            f'shard result set does not match --expected-shards='
            f'{expected_shards}: ' + '; '.join(details)
        )
    return expected


def validate_sample_index_coverage(records, start_index, total_samples):
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise ValueError('settings.start_index must be a non-negative integer')
    if start_index < 0:
        raise ValueError('settings.start_index must be a non-negative integer')
    if isinstance(total_samples, bool) or not isinstance(total_samples, int):
        raise ValueError('settings.total_samples must be a positive integer')
    if total_samples < 1:
        raise ValueError('settings.total_samples must be a positive integer')

    indices = [record.get('sample_index') for record in records]
    invalid = [
        value for value in indices
        if isinstance(value, bool) or not isinstance(value, int)
    ]
    if invalid:
        raise ValueError(
            'all merged records must have an integer sample_index; '
            f'invalid={invalid[:10]}'
        )

    duplicates = [
        index for index, count in Counter(indices).items() if count > 1
    ]
    if duplicates:
        raise ValueError(
            f'duplicate sample_index values across shards: {duplicates[:10]}'
        )

    expected = set(range(start_index, start_index + total_samples))
    actual = set(indices)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                f'missing_count={len(missing)}, missing={missing[:10]}'
            )
        if unexpected:
            details.append(
                f'unexpected_count={len(unexpected)}, '
                f'unexpected={unexpected[:10]}'
            )
        raise ValueError(
            'merged sample_index coverage does not match expected interval '
            f'[{start_index}, {start_index + total_samples}): '
            + '; '.join(details)
        )


def accuracy(records, prediction_key):
    eligible = [record for record in records if record.get(prediction_key) is not None]
    if not eligible:
        return None, 0
    return sum(record[prediction_key + '_correct'] for record in eligible) / len(eligible), len(eligible)


def transitions(records, second_key):
    result = Counter({
        'correct_to_wrong': 0,
        'wrong_to_correct': 0,
        'correct_to_correct': 0,
        'wrong_to_wrong': 0,
    })
    for record in records:
        before = record['baseline_prediction_correct']
        after = record[second_key + '_correct']
        if before and not after:
            result['correct_to_wrong'] += 1
        elif not before and after:
            result['wrong_to_correct'] += 1
        elif before:
            result['correct_to_correct'] += 1
        else:
            result['wrong_to_wrong'] += 1
    return dict(result)


def exact_mcnemar_pvalue(wrong_to_correct, correct_to_wrong):
    discordant = int(wrong_to_correct) + int(correct_to_wrong)
    if discordant == 0:
        return 1.0
    tail = min(int(wrong_to_correct), int(correct_to_wrong))
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / float(2 ** discordant))


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


def baseline_summary(records):
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


def paired_summary(records, experiment):
    second_key = {
        'counterfactual': 'counterfactual_prediction',
        'oracle': 'oracle_prediction',
        'selective_router': 'router_prediction',
    }[experiment]
    paired = [
        record for record in records
        if record.get('baseline_prediction') is not None and record.get(second_key) is not None
    ]
    baseline_accuracy, baseline_count = accuracy(paired, 'baseline_prediction')
    second_accuracy, second_count = accuracy(paired, second_key)
    summary = {
        'samples': len(paired),
        'baseline_accuracy': baseline_accuracy,
        'baseline_accuracy_count': baseline_count,
        f'{second_key.removesuffix("_prediction")}_accuracy': second_accuracy,
        f'{second_key.removesuffix("_prediction")}_accuracy_count': second_count,
        f'{second_key.removesuffix("_prediction")}_minus_baseline': (
            None if baseline_accuracy is None else second_accuracy - baseline_accuracy
        ),
        'answer_changed_count': sum(
            record['baseline_prediction'] != record[second_key] for record in paired
        ),
        'correctness_transitions': transitions(paired, second_key),
    }
    if experiment == 'selective_router':
        transition = summary['correctness_transitions']
        summary.update({
            'mcnemar_exact_two_sided_pvalue': exact_mcnemar_pvalue(
                transition['wrong_to_correct'], transition['correct_to_wrong']
            ),
            'routing': routing_metrics(paired),
        })
    if experiment == 'oracle':
        summary.update({
            'forced_sample_count': sum(
                record.get('intervention', {}).get('forced_coordinate_count', 0) > 0
                for record in paired
            ),
            'total_forced_coordinate_count': sum(
                record.get('intervention', {}).get('forced_coordinate_count', 0)
                for record in paired
            ),
        })
    return summary


def aggregate(args, settings, layout):
    if args.shards_dir:
        shards_dir = Path(args.shards_dir)
        input_paths = shard_result_paths(shards_dir, args.expected_shards)
        input_description = f'{shards_dir}/shard_*/results.jsonl'
    else:
        if args.expected_shards is not None:
            raise ValueError('--expected-shards can only be used with --shards-dir')
        import glob
        input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
        input_description = args.input_glob
    if not input_paths:
        raise FileNotFoundError(f'no shard JSONLs match: {input_description}')
    records = []
    for path in input_paths:
        records.extend(read_jsonl(path))
    if args.shards_dir:
        required_coverage_settings = ('start_index', 'total_samples')
        missing_settings = [
            key for key in required_coverage_settings if key not in settings
        ]
        if missing_settings:
            raise ValueError(
                '--settings must include shard coverage fields: '
                + ', '.join(missing_settings)
            )
        validate_sample_index_coverage(
            records,
            settings['start_index'],
            settings['total_samples'],
        )
    records.sort(key=lambda record: record.get('sample_index', -1))

    output_path = layout.results_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    successful_statuses = (
        ('ok', 'no_coordinate') if args.experiment == 'counterfactual'
        else ('ok',)
    )
    successful = [record for record in records if record.get('status') in successful_statuses]
    summary_function = (
        baseline_summary if args.experiment == 'testdev_baseline'
        else lambda subset: paired_summary(subset, args.experiment)
    )
    by_type = {
        type_name: summary_function([
            record for record in successful
            if record['types']['structural'] == type_name
        ])
        for type_name in sorted({
            record['types']['structural'] for record in successful
        })
    }
    summary = {
        'shard_paths': [str(path) for path in input_paths],
        'total_records': len(records),
        'successful_records': len(successful),
        'errors': sum(record.get('status') == 'error' for record in records),
        'all_samples': summary_function(successful),
        'by_structural_type': by_type,
        'settings': settings,
    }
    if args.experiment == 'testdev_baseline':
        summary['by_semantic_type'] = {
            type_name: baseline_summary([
                record for record in successful
                if record['types']['semantic'] == type_name
            ])
            for type_name in sorted({
                record['types']['semantic'] for record in successful
            })
        }
    if args.experiment == 'counterfactual':
        summary['no_coordinate'] = sum(record.get('status') == 'no_coordinate' for record in records)
    if args.events_output:
        expected_event_paths = [
            path.with_name('verifier_events.jsonl') for path in input_paths
        ]
        event_paths = [path for path in expected_event_paths if path.is_file()]
        event_records = []
        for path in event_paths:
            event_records.extend(read_jsonl(path))
        events_output_path = Path(args.events_output)
        events_output_path.parent.mkdir(parents=True, exist_ok=True)
        with events_output_path.open('w', encoding='utf-8') as handle:
            for event in event_records:
                handle.write(json.dumps(event, ensure_ascii=False) + '\n')
        summary['verifier_events'] = {
            'shard_paths': [str(path) for path in event_paths],
            'missing_shard_paths': [
                str(path) for path in expected_event_paths if not path.is_file()
            ],
            'total_records': len(event_records),
            'output': str(events_output_path),
        }
    summary_path = layout.summary_path
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    error_records = summary['errors']
    write_run_status(
        layout,
        'completed' if error_records == 0 else 'completed_with_errors',
        phase='aggregation',
        total_records=summary['total_records'],
        completed_records=summary['successful_records'],
        error_records=error_records,
        shard_count=len(input_paths),
        results_path=str(output_path),
        summary_path=str(summary_path),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Merged per-example results: {output_path}')
    if args.events_output:
        print(f'Merged verifier events: {args.events_output}')
    print(f'Merged summary: {summary_path}')


def main():
    args = parse_args()
    settings = json.loads(args.settings) if args.settings else {}
    if not isinstance(settings, dict):
        raise ValueError('--settings must decode to a JSON object')

    output_path = Path(args.output)
    run_id = args.run_id or output_path.parent.name
    if not run_id:
        raise ValueError('--run-id is required when --output has no parent directory')
    run_split = args.run_split or (
        'testdev_12578'
        if args.experiment == 'testdev_baseline' else 'val_1000_dev'
    )
    study, method, setting = experiment_identity(args.experiment, settings)
    layout = create_exact_output_layout(
        dataset='gqa',
        split=run_split,
        study=study,
        method=method,
        setting=setting,
        run_id=run_id,
        output=output_path,
    )
    layout.ensure_run_directories()
    write_run_config(layout, {
        'command': list(sys.argv),
        'arguments': vars(args),
        'inputs': {
            'shards_dir': args.shards_dir,
            'input_glob': args.input_glob,
            'manifest': settings.get('manifest_path'),
        },
        'components': {
            'generator': settings.get('model_path'),
            'experiment': args.experiment,
        },
        'aggregation': {
            'experiment': args.experiment,
            'settings': settings,
        },
    })
    write_run_status(layout, 'running', phase='aggregation')
    try:
        aggregate(args, settings, layout)
    except Exception as error:
        write_run_status(
            layout,
            'failed',
            phase='aggregation',
            error=f'{type(error).__name__}: {error}',
        )
        raise


if __name__ == '__main__':
    main()
