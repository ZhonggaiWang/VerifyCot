import json
from pathlib import Path

import pytest

from grounding_control.benchmarks.gqa_controlled.evaluator import (
    benchmark_summary_path,
    benchmark_run_identity,
    parse_args as parse_evaluator_args,
    resolve_benchmark_run_layout,
)
from grounding_control.benchmarks.gqa_controlled.tune_grounding_dino_thresholds import (
    _run_dev_inference,
    _write_cache_reference,
    _write_outputs,
    parse_args as parse_threshold_args,
    resolve_inference_cache_dir,
    resolve_threshold_search_layout,
)


def test_evaluator_default_uses_exact_population_and_protocol(tmp_path):
    args = parse_evaluator_args([
        '--output-root', str(tmp_path),
        '--run-id', 'binary_run',
        '--split', 'dev',
        '--task-mode', 'binary_alignment',
        '--model-path', 'weights/Qwen2.5-VL-7B-Instruct',
    ])

    layout = resolve_benchmark_run_layout(args)

    assert layout.run_dir == (
        tmp_path
        / 'verifier_benchmark'
        / 'runs'
        / 'gqa_controlled_v1_dev'
        / 'binary'
        / 'qwen25_vl_7b'
        / 'crop_only'
        / 'binary_run'
    )
    assert layout.is_canonical
    assert not layout.run_dir.exists()


def test_dino_evaluator_identity_records_backend_and_threshold():
    args = parse_evaluator_args([
        '--task-mode', 'routing_grounding_geometry',
        '--geometry-backend', 'grounding_dino',
        '--model-path', 'weights/grounding-dino-base',
        '--split', 'test',
        '--dino-box-threshold', '0.3',
        '--grounding-accept-iou', '0.5',
    ])

    identity = benchmark_run_identity(args)

    assert identity == {
        'dataset': 'verifier_benchmark',
        'split': 'gqa_controlled_v1_test',
        'study': 'four_way',
        'method': 'grounding_dino_base',
        'setting': 'raw_image__box_0p3__iou_0p5',
    }


def test_evaluator_explicit_output_remains_exact(tmp_path):
    requested = tmp_path / 'legacy' / 'custom.jsonl'
    args = parse_evaluator_args([
        '--output', str(requested),
        '--run-id', 'legacy_run',
    ])

    layout = resolve_benchmark_run_layout(args)

    assert layout.is_exact_output
    assert layout.run_id == 'legacy_run'
    assert layout.results_path == requested
    assert layout.run_dir == requested.parent
    assert not requested.parent.exists()


def test_evaluator_explicit_non_jsonl_keeps_old_summary_suffix(tmp_path):
    requested = tmp_path / 'legacy' / 'custom.records'
    args = parse_evaluator_args(['--output', str(requested)])
    layout = resolve_benchmark_run_layout(args)

    assert benchmark_summary_path(layout) == Path(
        str(requested) + '.summary.json'
    )


def test_threshold_search_default_uses_canonical_run(tmp_path):
    args = parse_threshold_args([
        '--output-root', str(tmp_path),
        '--run-id', 'search_run',
    ])

    layout = resolve_threshold_search_layout(args)

    assert layout.run_dir == (
        tmp_path
        / 'verifier_benchmark'
        / 'runs'
        / 'gqa_controlled_v1_dev'
        / 'threshold_search'
        / 'grounding_dino_base'
        / 'top1_score_gating__macro_f1__iou_0p5'
        / 'search_run'
    )
    assert layout.is_canonical


def test_threshold_search_explicit_directory_remains_exact(tmp_path):
    requested = tmp_path / 'legacy_threshold_search'
    args = parse_threshold_args([
        '--output-dir', str(requested),
        '--run-id', 'legacy_search',
    ])

    layout = resolve_threshold_search_layout(args)

    assert layout.is_exact_output
    assert layout.run_dir == requested
    assert layout.results_path == requested / 'threshold_search.json'
    assert not requested.exists()


def test_canonical_threshold_cache_is_shared_across_run_ids(tmp_path):
    first_args = parse_threshold_args([
        '--output-root', str(tmp_path),
        '--run-id', 'search_one',
    ])
    second_args = parse_threshold_args([
        '--output-root', str(tmp_path),
        '--run-id', 'search_two',
    ])
    first_layout = resolve_threshold_search_layout(first_args)
    second_layout = resolve_threshold_search_layout(second_args)

    first_cache = resolve_inference_cache_dir(first_args, first_layout)
    second_cache = resolve_inference_cache_dir(second_args, second_layout)

    assert first_layout.run_dir != second_layout.run_dir
    assert first_cache == second_cache
    first_cache.relative_to(tmp_path / 'verifier_benchmark' / 'cache')
    assert 'search_one' not in str(first_cache)


def test_explicit_threshold_directory_keeps_legacy_local_cache(tmp_path):
    requested = tmp_path / 'legacy_threshold_search'
    args = parse_threshold_args(['--output-dir', str(requested)])
    layout = resolve_threshold_search_layout(args)

    assert resolve_inference_cache_dir(args, layout) == (
        requested / 'inference_cache'
    )


def test_explicit_cache_directory_overrides_default(tmp_path):
    requested = tmp_path / 'shared_cache'
    args = parse_threshold_args(['--cache-dir', str(requested)])
    layout = resolve_threshold_search_layout(args)

    assert resolve_inference_cache_dir(args, layout) == requested


def test_canonical_threshold_outputs_use_standard_result_paths(tmp_path):
    args = parse_threshold_args([
        '--output-root', str(tmp_path),
        '--run-id', 'search_outputs',
    ])
    layout = resolve_threshold_search_layout(args)
    evaluation = {
        'box_threshold': 0.3,
        'accuracy': 0.5,
        'macro_f1': 0.4,
        'macro_precision': 0.4,
        'macro_recall': 0.4,
        'localization_success_rate': 0.9,
        'localization_failure_count': 1,
        'correct': 5,
        'total': 10,
    }
    payload = {'best': evaluation, 'evaluations': [evaluation]}

    paths = _write_outputs(layout, payload, [evaluation])

    assert paths['results'] == layout.results_path
    assert paths['summary'] == layout.summary_path
    rows = [
        json.loads(line)
        for line in layout.results_path.read_text(encoding='utf-8').splitlines()
    ]
    assert rows == [evaluation]
    assert json.loads(
        layout.summary_path.read_text(encoding='utf-8')
    ) == payload
    assert paths['best_config'].parent == layout.artifacts_dir
    assert paths['csv'].parent == layout.artifacts_dir
    assert not layout.shards_dir.exists()
    assert not layout.repetitions_dir.exists()


def test_canonical_run_records_shared_cache_reference(tmp_path):
    args = parse_threshold_args([
        '--output-root', str(tmp_path),
        '--run-id', 'cache_reference',
    ])
    layout = resolve_threshold_search_layout(args)
    cache_dir = resolve_inference_cache_dir(args, layout)
    layout.ensure_run_directories(include_artifacts=True)

    _write_cache_reference(
        layout,
        cache_dir,
        cache_dir / 'results.summary.json',
        reused=True,
    )

    reference = json.loads((
        layout.artifacts_dir / 'inference_cache.reference.json'
    ).read_text(encoding='utf-8'))
    assert reference['cache_dir'] == str(cache_dir)
    assert reference['reused'] is True


def test_threshold_cache_inference_passes_exact_output_and_shared_run_id(
        tmp_path, monkeypatch):
    args = parse_threshold_args([])
    requested = tmp_path / 'inference_cache' / 'results.jsonl'
    observed = {}

    def fake_run(command, check):
        observed['command'] = command
        observed['check'] = check

    from grounding_control.benchmarks.gqa_controlled import (
        tune_grounding_dino_thresholds as threshold_module,
    )
    monkeypatch.setattr(threshold_module.subprocess, 'run', fake_run)

    _run_dev_inference(args, requested, 'shared_search_run')

    command = observed['command']
    assert observed['check'] is True
    assert command[command.index('--output') + 1] == str(requested)
    assert command[command.index('--run-id') + 1] == 'shared_search_run'


def test_evaluator_main_marks_started_run_failed(tmp_path, monkeypatch):
    from grounding_control.benchmarks.gqa_controlled import evaluator as module
    args = parse_evaluator_args([
        '--output-root', str(tmp_path),
        '--run-id', 'failed_eval',
    ])
    layout = resolve_benchmark_run_layout(args)

    def fail_after_start(parsed_args, resolved_layout, lifecycle):
        module.write_run_status(resolved_layout, 'running')
        lifecycle['started'] = True
        raise RuntimeError('backend load failed')

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, '_run_evaluation', fail_after_start)

    with pytest.raises(RuntimeError, match='backend load failed'):
        module.main()

    status = json.loads(layout.status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'failed'
    assert status['error_type'] == 'RuntimeError'
    assert status['error'] == 'backend load failed'


def test_threshold_main_marks_started_run_failed(tmp_path, monkeypatch):
    from grounding_control.benchmarks.gqa_controlled import (
        tune_grounding_dino_thresholds as module,
    )
    args = parse_threshold_args([
        '--output-root', str(tmp_path),
        '--run-id', 'failed_search',
    ])
    layout = resolve_threshold_search_layout(args)

    def fail_after_start(parsed_args, resolved_layout, lifecycle):
        module.write_run_status(resolved_layout, 'running')
        lifecycle['started'] = True
        raise RuntimeError('cache validation failed')

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, '_run_threshold_search', fail_after_start)

    with pytest.raises(RuntimeError, match='cache validation failed'):
        module.main()

    status = json.loads(layout.status_path.read_text(encoding='utf-8'))
    assert status['status'] == 'failed'
    assert status['error_type'] == 'RuntimeError'
    assert status['error'] == 'cache validation failed'
