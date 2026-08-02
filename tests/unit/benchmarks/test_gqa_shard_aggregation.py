import pytest

from eval.Oracle_experiment.gqa.aggregate_results import (
    shard_result_paths,
    validate_sample_index_coverage,
)


def _shard_result(shards_dir, index):
    path = shards_dir / f'shard_{index:03d}' / 'results.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}\n', encoding='utf-8')
    return path


def test_expected_shards_are_returned_in_scheduler_order(tmp_path):
    shards_dir = tmp_path / 'shards'
    expected = [_shard_result(shards_dir, index) for index in range(3)]

    assert shard_result_paths(shards_dir, 3) == expected


def test_missing_expected_shard_is_rejected(tmp_path):
    shards_dir = tmp_path / 'shards'
    _shard_result(shards_dir, 0)
    missing = shards_dir / 'shard_001' / 'results.jsonl'

    with pytest.raises(ValueError, match='missing=.*shard_001/results.jsonl'):
        shard_result_paths(shards_dir, 2)

    assert not missing.exists()


def test_stale_extra_shard_is_rejected(tmp_path):
    shards_dir = tmp_path / 'shards'
    _shard_result(shards_dir, 0)
    stale = _shard_result(shards_dir, 1)

    with pytest.raises(ValueError, match='unexpected=.*shard_001/results.jsonl'):
        shard_result_paths(shards_dir, 1)

    assert stale.exists()


def test_expected_shard_count_is_required_and_positive(tmp_path):
    with pytest.raises(ValueError, match='required'):
        shard_result_paths(tmp_path / 'shards', None)
    with pytest.raises(ValueError, match='positive integer'):
        shard_result_paths(tmp_path / 'shards', 0)


def test_sample_indices_must_exactly_cover_expected_interval():
    records = [{'sample_index': index} for index in range(10, 14)]

    validate_sample_index_coverage(records, start_index=10, total_samples=4)


@pytest.mark.parametrize(
    ('records', 'message'),
    [
        (
            [{'sample_index': 10}, {'sample_index': 12}],
            r'missing=\[11\]',
        ),
        (
            [{'sample_index': 10}, {'sample_index': 11}, {'sample_index': 12}],
            r'unexpected=\[12\]',
        ),
        (
            [{'sample_index': 10}, {'sample_index': 10}],
            'duplicate sample_index',
        ),
        (
            [{'sample_index': '10'}],
            'integer sample_index',
        ),
    ],
)
def test_incomplete_or_out_of_range_sample_coverage_is_rejected(
        records, message):
    with pytest.raises(ValueError, match=message):
        validate_sample_index_coverage(
            records,
            start_index=10,
            total_samples=2,
        )
