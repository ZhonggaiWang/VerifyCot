"""Merge sharded GQA intervention results and recompute paired summaries."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=('counterfactual', 'oracle'), required=True)
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


def paired_summary(records, experiment):
    second_key = 'counterfactual_prediction' if experiment == 'counterfactual' else 'oracle_prediction'
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

    successful_statuses = ('ok', 'no_coordinate') if args.experiment == 'counterfactual' else ('ok',)
    successful = [record for record in records if record.get('status') in successful_statuses]
    by_type = {
        type_name: paired_summary(
            [record for record in successful if record['types']['structural'] == type_name], args.experiment
        )
        for type_name in sorted({record['types']['structural'] for record in successful})
    }
    summary = {
        'shard_paths': [str(path) for path in input_paths],
        'total_records': len(records),
        'successful_records': len(successful),
        'errors': sum(record.get('status') == 'error' for record in records),
        'all_samples': paired_summary(successful, args.experiment),
        'by_structural_type': by_type,
        'settings': json.loads(args.settings) if args.settings else {},
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
