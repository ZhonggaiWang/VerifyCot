from pathlib import Path

import pytest

from grounding_control.run_paths import (
    create_exact_output_layout,
    create_run_layout,
    resolve_run_output,
    write_run_config,
    write_run_status,
)


def test_canonical_layout_has_stable_run_files(tmp_path):
    layout = create_run_layout(
        dataset='vstar',
        split='full_238',
        study='routing',
        method='dino_geometry__oracle_experts',
        setting='iou_0p5',
        run_id='20260801_153000__main',
        output_root=tmp_path,
    )

    expected = (
        tmp_path
        / 'vstar'
        / 'runs'
        / 'full_238'
        / 'routing'
        / 'dino_geometry__oracle_experts'
        / 'iou_0p5'
        / '20260801_153000__main'
    )
    assert layout.is_canonical
    assert layout.run_dir == expected
    assert layout.results_path == expected / 'results.jsonl'
    assert layout.summary_path == expected / 'results.summary.json'
    assert layout.config_path == expected / 'run.config.json'
    assert layout.status_path == expected / 'run.status.json'
    assert layout.events_path == expected / 'verifier_events.jsonl'
    assert layout.log_path == expected / 'run.log'
    assert layout.artifacts_dir == expected / 'artifacts'
    assert layout.shards_dir == expected / 'shards'
    assert layout.repetitions_dir == expected / 'repetitions'
    assert not expected.exists()


def test_ensure_run_directories_does_not_create_parallel_subtrees(tmp_path):
    layout = create_run_layout(
        dataset='gqa',
        split='testdev_12578',
        study='baseline',
        method='volcano_7b',
        run_id='test_run',
        output_root=tmp_path,
    )

    layout.ensure_run_directories(include_artifacts=True)

    assert layout.run_dir.is_dir()
    assert layout.artifacts_dir.is_dir()
    assert not layout.shards_dir.exists()
    assert not layout.repetitions_dir.exists()


def test_explicit_output_preserves_legacy_placement_and_filename(tmp_path):
    requested = tmp_path / 'legacy_study' / 'custom_records.jsonl'
    layout = create_run_layout(
        dataset='vstar',
        split='full_238',
        study='repair',
        method='legacy_prompt_repair',
        run_id='manual_run',
        output=requested,
    )

    expected = tmp_path / 'legacy_study' / 'manual_run'
    assert not layout.is_canonical
    assert not layout.is_exact_output
    assert layout.layout_kind == 'timestamped_output'
    assert layout.run_dir == expected
    assert layout.results_path == expected / 'custom_records.jsonl'
    assert layout.summary_path == expected / 'custom_records.summary.json'
    assert layout.events_path == expected / 'verifier_events.jsonl'
    assert not expected.exists()


def test_exact_output_layout_does_not_insert_run_id(tmp_path):
    requested = (
        tmp_path
        / 'gqa'
        / 'runs'
        / 'testdev_12578'
        / 'baseline'
        / 'volcano_7b'
        / 'default'
        / 'logical_run'
        / 'shards'
        / 'shard_007'
        / 'results.jsonl'
    )
    layout = create_exact_output_layout(
        dataset='gqa',
        split='testdev_12578',
        study='baseline',
        method='volcano_7b',
        setting='default',
        run_id='logical_run',
        output=requested,
    )

    assert not layout.is_canonical
    assert layout.is_exact_output
    assert layout.layout_kind == 'exact_output'
    assert layout.run_id == 'logical_run'
    assert layout.results_path == requested
    assert layout.run_dir == requested.parent
    assert layout.summary_path == requested.with_suffix('.summary.json')
    assert layout.config_path == requested.parent / 'run.config.json'
    assert layout.status_path == requested.parent / 'run.status.json'
    assert not requested.parent.exists()


def test_exact_bare_output_can_generate_metadata_run_id(tmp_path, monkeypatch):
    from grounding_control import run_paths

    monkeypatch.setattr(run_paths, '_new_run_id', lambda: 'generated_run')
    requested = Path('results.jsonl')
    layout = create_exact_output_layout(
        dataset='gqa',
        split='val_1000_dev',
        study='counterfactual',
        method='random_box',
        setting='random',
        run_id=None,
        output=requested,
    )

    assert layout.run_id == 'generated_run'
    assert layout.results_path == requested
    assert layout.run_dir == Path('.')


def test_exact_output_metadata_is_written_next_to_shard_result(tmp_path):
    requested = tmp_path / 'shards' / 'shard_000' / 'records.jsonl'
    layout = create_exact_output_layout(
        dataset='gqa',
        split='val_1000_dev',
        study='routing',
        method='oracle_verifier__oracle_experts',
        setting='iou_0p1',
        run_id='shared_run',
        output=requested,
    )

    write_run_config(layout, {'shard_id': 0})
    write_run_status(layout, 'completed', completed_records=500)

    import json
    config = json.loads(layout.config_path.read_text(encoding='utf-8'))
    status = json.loads(layout.status_path.read_text(encoding='utf-8'))
    assert config['layout'] == 'exact_output'
    assert config['run_id'] == 'shared_run'
    assert config['shard_id'] == 0
    assert status['layout'] == 'exact_output'
    assert status['completed_records'] == 500
    assert not (requested.parent / 'shared_run').exists()


def test_resolve_run_output_retains_existing_contract():
    path, run_id = resolve_run_output(
        Path('output/vstar/online_oracle/results.jsonl'),
        'fixed_run',
    )

    assert run_id == 'fixed_run'
    assert path == Path(
        'output/vstar/online_oracle/fixed_run/results.jsonl'
    )


def test_metadata_writers_are_atomic_and_embed_run_identity(
        tmp_path, monkeypatch):
    layout = create_run_layout(
        dataset='vstar',
        split='full_238',
        study='routing',
        method='dino_geometry__oracle_experts',
        setting='iou_0p5',
        run_id='run_a',
        output_root=tmp_path,
    )
    replacements = []
    from grounding_control import run_paths
    real_replace = run_paths.os.replace

    def tracked_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(run_paths.os, 'replace', tracked_replace)

    write_run_config(
        layout,
        {
            'schema_version': 999,
            'run_id': 'copied_from_another_run',
            'generator': {'model': 'weights/Volcano-7b'},
        },
    )
    write_run_status(layout, 'running', completed_records=7)

    import json
    config = json.loads(layout.config_path.read_text(encoding='utf-8'))
    status = json.loads(layout.status_path.read_text(encoding='utf-8'))
    assert config['schema_version'] == 1
    assert config['run_id'] == 'run_a'
    assert config['dataset'] == 'vstar'
    assert config['generator']['model'] == 'weights/Volcano-7b'
    assert 'created_at' in config
    assert 'python_executable' in config['provenance']
    assert config['resolved_paths']['results'] == str(layout.results_path)
    assert status['status'] == 'running'
    assert status['completed_records'] == 7
    assert status['layout'] == 'canonical'
    assert 'updated_at' in status
    assert [destination for _, destination in replacements] == [
        layout.config_path,
        layout.status_path,
    ]
    assert not list(layout.run_dir.glob('*.tmp'))
    assert not layout.shards_dir.exists()
    assert not layout.repetitions_dir.exists()


def test_non_json_config_does_not_create_or_replace_metadata(tmp_path):
    layout = create_run_layout(
        dataset='vstar',
        split='full_238',
        study='baseline',
        method='volcano_7b',
        run_id='bad_config',
        output_root=tmp_path,
    )

    with pytest.raises(TypeError):
        write_run_config(layout, {'not_json': object()})

    assert not layout.run_dir.exists()


def test_non_finite_config_number_is_rejected(tmp_path):
    layout = create_run_layout(
        dataset='vstar',
        split='full_238',
        study='baseline',
        method='volcano_7b',
        run_id='nan_config',
        output_root=tmp_path,
    )

    with pytest.raises(ValueError):
        write_run_config(layout, {'threshold': float('nan')})

    assert not layout.run_dir.exists()


@pytest.mark.parametrize(
    'field,value',
    [
        ('dataset', ''),
        ('split', '..'),
        ('study', 'routing/oracle'),
        ('method', 'dino\\oracle'),
        ('setting', '.'),
        ('run_id', 'nested/run'),
    ],
)
def test_canonical_components_must_be_single_directories(
        tmp_path, field, value):
    kwargs = {
        'dataset': 'vstar',
        'split': 'full_238',
        'study': 'routing',
        'method': 'dino_geometry__oracle_experts',
        'setting': 'iou_0p5',
        'run_id': 'run',
        'output_root': tmp_path,
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError)):
        create_run_layout(**kwargs)
