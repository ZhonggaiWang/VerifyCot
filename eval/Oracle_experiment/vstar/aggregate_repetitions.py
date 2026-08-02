"""Aggregate VStar repetition summaries without merging per-sample JSONLs."""

import argparse
import json
import statistics
import sys
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-split', default='full_238')
    parser.add_argument(
        '--method',
        choices=('random_box', 'remove_grounding'),
        required=True,
    )
    parser.add_argument(
        '--position',
        choices=('random', 'first', 'last'),
        required=True,
    )
    parser.add_argument(
        '--expected-repetitions',
        type=int,
        default=None,
        help='When set, require exactly repeat_01 through repeat_N.',
    )
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f'repetition summary must be a JSON object: {path}')
    return document


def repetition_summary_paths(repetitions_dir, expected_repetitions=None):
    repetitions_dir = Path(repetitions_dir)
    found = sorted(repetitions_dir.glob('repeat_*/results.summary.json'))
    if expected_repetitions is None:
        if not found:
            raise FileNotFoundError(
                f'no repetition summaries under: {repetitions_dir}'
            )
        return found
    if expected_repetitions < 1:
        raise ValueError('--expected-repetitions must be a positive integer')

    expected = [
        repetitions_dir / f'repeat_{index:02d}' / 'results.summary.json'
        for index in range(1, expected_repetitions + 1)
    ]
    missing = [path for path in expected if not path.is_file()]
    expected_set = set(expected)
    unexpected = [path for path in found if path not in expected_set]
    if missing or unexpected:
        details = []
        if missing:
            details.append('missing=' + ','.join(str(path) for path in missing))
        if unexpected:
            details.append(
                'unexpected=' + ','.join(str(path) for path in unexpected)
            )
        raise ValueError('repetition set does not match expectation: ' + '; '.join(details))
    return expected


def metric_leaves(document, prefix=()):
    for key, value in document.items():
        if not prefix and key == 'settings':
            continue
        path = prefix + (str(key),)
        if isinstance(value, dict):
            yield from metric_leaves(value, path)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if path[-1].endswith('_count'):
            continue
        if any(
            'accuracy' in component
            or component.endswith('_rate')
            or component.endswith('_rates')
            for component in path
        ):
            yield '.'.join(path), float(value)


def metric_statistics(repetition_summaries):
    values_by_metric = {}
    for summary in repetition_summaries:
        for metric, value in metric_leaves(summary):
            values_by_metric.setdefault(metric, []).append(value)

    repetition_count = len(repetition_summaries)
    return {
        metric: {
            'count': len(values),
            'missing_count': repetition_count - len(values),
            'mean': statistics.fmean(values),
            'std': statistics.pstdev(values),
            'min': min(values),
            'max': max(values),
        }
        for metric, values in sorted(values_by_metric.items())
    }


def build_summary(paths, args):
    repetition_summaries = [read_json(path) for path in paths]
    repetitions = [
        {
            'name': path.parent.name,
            'summary_path': str(path),
            'summary': summary,
        }
        for path, summary in zip(paths, repetition_summaries)
    ]
    error_records = sum(
        int(summary.get('errors', summary.get('error_records', 0)) or 0)
        for summary in repetition_summaries
    )
    repetitions_with_errors = sum(
        int(summary.get('errors', summary.get('error_records', 0)) or 0) > 0
        for summary in repetition_summaries
    )
    return {
        'aggregation': {
            'kind': 'repetition_summaries',
            'repetition_count': len(repetitions),
            'expected_repetition_count': args.expected_repetitions,
            'sample_jsonl_merged': False,
            'std_ddof': 0,
            'repetitions_with_errors': repetitions_with_errors,
            'total_error_records': error_records,
        },
        'metric_statistics': metric_statistics(repetition_summaries),
        'repetitions': repetitions,
        'settings': {
            'dataset': 'vstar',
            'split': args.run_split,
            'study': 'counterfactual',
            'method': args.method,
            'position': args.position,
            'run_id': args.run_id,
        },
    }


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    layout = create_exact_output_layout(
        dataset='vstar',
        split=args.run_split,
        study='counterfactual',
        method=args.method,
        setting=args.position,
        run_id=args.run_id,
        output=run_root / 'results.jsonl',
    )
    layout.ensure_run_directories()
    repetitions_dir = run_root / 'repetitions'
    repetition_config_paths = sorted(
        repetitions_dir.glob('repeat_*/run.config.json')
    )
    write_run_config(layout, {
        'command': list(sys.argv),
        'arguments': vars(args),
        'resolved_paths': {
            'run_dir': str(layout.run_dir),
            'results': None,
            'summary': str(layout.summary_path),
            'verifier_events': None,
        },
        'inputs': {
            'repetitions_dir': str(repetitions_dir),
            'summary_glob': str(
                repetitions_dir / 'repeat_*' / 'results.summary.json'
            ),
            'config_glob': str(
                repetitions_dir / 'repeat_*' / 'run.config.json'
            ),
            'repetition_config_paths': [
                str(path) for path in repetition_config_paths
            ],
        },
        'aggregation': {
            'kind': 'repetition_summaries',
            'sample_jsonl_merged': False,
            'std_ddof': 0,
        },
    })
    write_run_status(layout, 'running', phase='repetition_aggregation')

    try:
        paths = repetition_summary_paths(
            repetitions_dir,
            args.expected_repetitions,
        )
        summary = build_summary(paths, args)
        summary_path = layout.summary_path
        with summary_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')

        aggregation = summary['aggregation']
        status = (
            'completed'
            if aggregation['total_error_records'] == 0
            else 'completed_with_errors'
        )
        write_run_status(
            layout,
            status,
            phase='repetition_aggregation',
            completed_repetitions=aggregation['repetition_count'],
            repetitions_with_errors=aggregation['repetitions_with_errors'],
            error_records=aggregation['total_error_records'],
            summary_path=str(summary_path),
            sample_jsonl_merged=False,
        )
    except Exception as error:
        write_run_status(
            layout,
            'failed',
            phase='repetition_aggregation',
            error=f'{type(error).__name__}: {error}',
            sample_jsonl_merged=False,
        )
        raise

    print(f'Repetitions aggregated: {len(paths)}')
    print(f'Root summary: {summary_path}')
    print('Per-sample repetition JSONLs were not merged.')


if __name__ == '__main__':
    main()
