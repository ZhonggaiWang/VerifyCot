"""Merge sharded GQA intervention results and recompute paired summaries."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


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
    parser.add_argument('--output', required=True, help='Merged JSONL output path.')
    parser.add_argument('--settings', default=None,
                        help='Optional JSON object recorded under summary.settings.')
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def main():
    args = parse_args()
    if args.shards_dir:
        shards_dir = Path(args.shards_dir)
        input_paths = sorted(shards_dir.glob('shard_*/results.jsonl'))
        input_description = f'{shards_dir}/shard_*/results.jsonl'
    else:
        import glob
        input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
        input_description = args.input_glob
    if not input_paths:
        raise FileNotFoundError(f'no shard JSONLs match: {input_description}')
    records = []
    for path in input_paths:
        records.extend(read_jsonl(path))
    records.sort(key=lambda record: record.get('sample_index', -1))
    indices = [record.get('sample_index') for record in records]
    duplicates = [index for index, count in Counter(indices).items() if count > 1]
    if duplicates:
        raise ValueError(f'duplicate sample_index values across shards: {duplicates[:10]}')

    output_path = Path(args.output)
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
        'settings': json.loads(args.settings) if args.settings else {},
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
    summary_path = output_path.with_suffix('.summary.json')
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Merged per-example results: {output_path}')
    print(f'Merged summary: {summary_path}')


if __name__ == '__main__':
    main()
