import argparse
import json

import pytest

from eval.Oracle_experiment.vstar import aggregate_repetitions


def _write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def test_repetition_aggregation_writes_summary_metadata_without_sample_merge(
        tmp_path, monkeypatch):
    run_root = tmp_path / 'custom_output' / 'random_box' / 'first' / 'run_a'
    summaries = [
        {
            'errors': 0,
            'baseline_accuracy_all': 0.6,
            'baseline_accuracy_all_count': 10,
            'counterfactual_accuracy_paired_subset': 0.5,
            'answer_changed_rate': 0.2,
            'correctness_transition_rates': {'correct_to_wrong': 0.1},
            'settings': {'selection_seed': 101},
        },
        {
            'errors': 1,
            'baseline_accuracy_all': 0.8,
            'baseline_accuracy_all_count': 9,
            'counterfactual_accuracy_paired_subset': 0.7,
            'answer_changed_rate': 0.4,
            'correctness_transition_rates': {'correct_to_wrong': 0.3},
            'settings': {'selection_seed': 202},
        },
    ]
    repetition_configs = []
    for index, summary in enumerate(summaries, 1):
        repeat_dir = run_root / 'repetitions' / f'repeat_{index:02d}'
        _write_json(repeat_dir / 'results.summary.json', summary)
        config_path = repeat_dir / 'run.config.json'
        _write_json(config_path, {'repeat': index})
        repetition_configs.append(str(config_path))

    args = argparse.Namespace(
        run_root=str(run_root),
        run_id='run_a',
        run_split='full_238',
        method='random_box',
        position='first',
        expected_repetitions=2,
    )
    monkeypatch.setattr(aggregate_repetitions, 'parse_args', lambda: args)

    aggregate_repetitions.main()

    root_summary = json.loads(
        (run_root / 'results.summary.json').read_text(encoding='utf-8')
    )
    stats = root_summary['metric_statistics']
    assert stats['baseline_accuracy_all'] == {
        'count': 2,
        'missing_count': 0,
        'mean': pytest.approx(0.7),
        'std': pytest.approx(0.1),
        'min': 0.6,
        'max': 0.8,
    }
    assert stats['answer_changed_rate']['mean'] == pytest.approx(0.3)
    assert stats['correctness_transition_rates.correct_to_wrong']['std'] \
        == pytest.approx(0.1)
    assert 'baseline_accuracy_all_count' not in stats
    assert [item['summary'] for item in root_summary['repetitions']] == summaries
    assert root_summary['aggregation']['sample_jsonl_merged'] is False

    config = json.loads(
        (run_root / 'run.config.json').read_text(encoding='utf-8')
    )
    status = json.loads(
        (run_root / 'run.status.json').read_text(encoding='utf-8')
    )
    assert config['inputs']['repetition_config_paths'] == repetition_configs
    assert config['inputs']['config_glob'].endswith(
        'repetitions/repeat_*/run.config.json'
    )
    assert config['resolved_paths']['results'] is None
    assert config['dataset'] == 'vstar'
    assert config['split'] == 'full_238'
    assert config['study'] == 'counterfactual'
    assert config['method'] == 'random_box'
    assert config['setting'] == 'first'
    assert status['status'] == 'completed_with_errors'
    assert status['error_records'] == 1
    assert not (run_root / 'results.jsonl').exists()
