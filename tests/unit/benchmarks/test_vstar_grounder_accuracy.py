"""CPU-only contracts for the backend-neutral VStar Grounder benchmark.

The benchmark must measure the Grounder role itself: one canonical object
reference and one immutable source image go in, while no VoCoT candidate box,
question answer, or routing decision is exposed.  These tests deliberately use
the real :class:`RemoteGrounderBackend` around a fake worker response so the
pixel-to-VoCoT coordinate boundary is covered without loading a model.
"""

import json

import pytest
from PIL import Image

from eval.grounding_control.vstar import evaluate_grounder_accuracy as evaluator
from grounding_control.experts.grounders import RemoteGrounderBackend
from grounding_control.transport import serialize_grounder_output


class _FakeGrounderWorkerClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def request(self, payload, *, timeout=None):
        self.requests.append({
            'payload': dict(payload),
            'timeout': timeout,
        })
        response = dict(self.responses.pop(0))
        response.setdefault('request_id', 'fake-grounder-request')
        return response


def _write_oracle_fixture(tmp_path):
    image_dir = tmp_path / 'images'
    (image_dir / 'direct_attributes').mkdir(parents=True)
    (image_dir / 'relative_position').mkdir(parents=True)
    Image.new('RGB', (100, 50), 'white').save(
        image_dir / 'direct_attributes' / 'one.jpg'
    )
    Image.new('RGB', (40, 80), 'white').save(
        image_dir / 'relative_position' / 'two.jpg'
    )

    rows = [
        {
            'sample_index': 0,
            'question_id': 'main:0',
            'category': 'direct_attributes',
            'image': 'direct_attributes/one.jpg',
            # Deliberately unusable absolute source path: --image-dir must be
            # authoritative so a downloaded manifest remains relocatable.
            'image_path': '/old/machine/VStar/direct_attributes/one.jpg',
            'image_size': {'width': 100, 'height': 50},
            'target_objects': ['red cup'],
            'pixel_bboxes_xywh': [[20, 10, 40, 20]],
            'normalized_bboxes_xyxy': [[0.2, 0.2, 0.6, 0.6]],
        },
        {
            'sample_index': 1,
            'question_id': 'main:1',
            'category': 'relative_position',
            'image': 'relative_position/two.jpg',
            'image_path': '/old/machine/VStar/relative_position/two.jpg',
            'image_size': {'width': 40, 'height': 80},
            'target_objects': ['cat', 'bed'],
            'pixel_bboxes_xywh': [[4, 8, 12, 16], [20, 40, 16, 32]],
            'normalized_bboxes_xyxy': [
                [0.1, 0.1, 0.4, 0.3],
                [0.5, 0.5, 0.9, 0.9],
            ],
        },
    ]
    manifest = tmp_path / 'oracle_boxes.jsonl'
    manifest.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8',
    )
    return manifest, image_dir


def test_load_grounding_tasks_expands_every_annotated_target_and_xywh(tmp_path):
    manifest, image_dir = _write_oracle_fixture(tmp_path)

    tasks = evaluator.load_grounding_tasks(manifest, image_dir)

    assert len(tasks) == 3
    assert [task['task_id'] for task in tasks] == [
        'main:0:target:0',
        'main:1:target:0',
        'main:1:target:1',
    ]
    first = tasks[0]
    assert first['sample_id'] == 'main:0'
    assert first['target_index'] == 0
    assert first['object_reference'] == 'red cup'
    assert first['category'] == 'direct_attributes'
    assert first['image_path'] == str(
        (image_dir / 'direct_attributes' / 'one.jpg').resolve()
    )
    assert first['gt_bbox_original_pixel_xyxy'] == pytest.approx(
        [20.0, 10.0, 60.0, 30.0]
    )
    # 100x50 is vertically center padded to 100x100: y += 25.
    assert first['gt_bbox_vocot_normalized_padded_xyxy'] == pytest.approx(
        [0.2, 0.35, 0.6, 0.55]
    )
    assert tasks[2]['object_reference'] == 'bed'
    assert tasks[2]['gt_bbox_original_pixel_xyxy'] == pytest.approx(
        [20.0, 40.0, 36.0, 72.0]
    )


def test_load_grounding_tasks_rejects_target_box_count_mismatch(tmp_path):
    manifest, image_dir = _write_oracle_fixture(tmp_path)
    malformed = json.loads(manifest.read_text(encoding='utf-8').splitlines()[0])
    malformed['target_objects'].append('unpaired object')
    manifest.write_text(json.dumps(malformed) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='target.*box.*count|count.*mismatch'):
        evaluator.load_grounding_tasks(manifest, image_dir)


@pytest.mark.parametrize(
    ('first', 'second', 'expected'),
    [
        ((0, 0, 2, 2), (0, 0, 2, 2), 1.0),
        ((0, 0, 2, 2), (1, 1, 3, 3), 1.0 / 7.0),
        ((0, 0, 1, 1), (1, 0, 2, 1), 0.0),
    ],
)
def test_box_iou_xyxy_uses_continuous_half_open_geometry(
        first, second, expected):
    assert evaluator.box_iou_xyxy(first, second) == pytest.approx(expected)
    assert evaluator.box_iou_xyxy(second, first) == pytest.approx(expected)


def test_evaluate_task_exercises_real_remote_backend_without_candidate(tmp_path):
    manifest, image_dir = _write_oracle_fixture(tmp_path)
    task = evaluator.load_grounding_tasks(manifest, image_dir)[0]
    client = _FakeGrounderWorkerClient([
        serialize_grounder_output(
            available=True,
            source='qwen25_vl_grounder',
            bbox=(20.0, 10.0, 60.0, 30.0),
            image_size=(100, 50),
            confidence=None,
            error=None,
            metadata={
                'raw_response': '{"bbox_2d":[20,10,60,30]}',
                'prompt_protocol': 'single_object_json_v2',
            },
        ),
    ])
    backend = RemoteGrounderBackend(
        client,
        timeout=17.0,
        source='qwen25_vl_grounder',
    )

    record = evaluator.evaluate_task(task, backend)

    assert record['status'] == 'ok'
    assert record['iou'] == pytest.approx(1.0)
    assert record['prediction_bbox_original_pixel_xyxy'] == pytest.approx(
        [20.0, 10.0, 60.0, 30.0]
    )
    assert record[
        'prediction_bbox_vocot_normalized_padded_xyxy'
    ] == pytest.approx([0.2, 0.35, 0.6, 0.55])
    assert record['grounder_source'] == 'qwen25_vl_grounder'
    assert record['grounder_metadata']['remote_raw_response'] == (
        '{"bbox_2d":[20,10,60,30]}'
    )
    assert client.requests == [{
        'payload': {
            'operation': 'ground',
            'image_path': str(
                (image_dir / 'direct_attributes' / 'one.jpg').resolve()
            ),
            'sample_id': 'main:0',
            'grounding_step': 1,
            'object_reference': 'red cup',
        },
        'timeout': 17.0,
    }]
    assert 'candidate_bbox' not in client.requests[0]['payload']
    assert 'question' not in client.requests[0]['payload']
    assert 'answer' not in client.requests[0]['payload']


def test_unavailable_is_retained_as_zero_for_overall_but_not_success_only(
        tmp_path):
    manifest, image_dir = _write_oracle_fixture(tmp_path)
    task = evaluator.load_grounding_tasks(manifest, image_dir)[0]
    client = _FakeGrounderWorkerClient([
        serialize_grounder_output(
            available=False,
            source='qwen25_vl_grounder',
            bbox=None,
            image_size=(100, 50),
            confidence=None,
            error='coordinate_parse_failed',
            metadata={
                'parse_failed': True,
                'raw_response': 'not a coordinate',
            },
        ),
    ])
    backend = RemoteGrounderBackend(client, source='qwen25_vl_grounder')

    failed = evaluator.evaluate_task(task, backend)
    successful = {
        **task,
        'status': 'ok',
        'iou': 0.6,
        'grounder_source': 'qwen25_vl_grounder',
    }
    summary = evaluator.summarize_records([successful, failed])

    assert failed['status'] == 'grounder_unavailable'
    assert failed['iou'] == 0.0
    assert failed['error_type'] == 'ExpertUnavailableError'
    assert 'coordinate_parse_failed' in failed['error']
    assert summary['target_request_count'] == 2
    assert summary['successful_request_count'] == 1
    assert summary['failed_request_count'] == 1
    assert summary['availability_rate'] == pytest.approx(0.5)
    assert summary['overall_miou'] == pytest.approx(0.3)
    assert summary['successful_only_miou'] == pytest.approx(0.6)


def test_summary_reports_inclusive_iou_recall_01_through_09():
    records = [
        {'status': 'ok', 'iou': 1.0, 'category': 'a'},
        {'status': 'ok', 'iou': 0.5, 'category': 'a'},
        {'status': 'ok', 'iou': 0.1, 'category': 'b'},
        {'status': 'grounder_unavailable', 'iou': 0.0, 'category': 'b'},
    ]

    summary = evaluator.summarize_records(records)

    assert list(summary['iou_recall']) == [
        '0.1', '0.2', '0.3', '0.4', '0.5',
        '0.6', '0.7', '0.8', '0.9',
    ]
    assert summary['iou_recall']['0.1'] == pytest.approx(3 / 4)
    assert summary['iou_recall']['0.5'] == pytest.approx(2 / 4)
    assert summary['iou_recall']['0.6'] == pytest.approx(1 / 4)
    assert summary['iou_recall']['0.9'] == pytest.approx(1 / 4)
    assert summary['by_category']['a']['overall_miou'] == pytest.approx(0.75)
    # Category-local mIoU uses the two category-b targets: (0.1 + 0) / 2.
    assert summary['by_category']['b']['overall_miou'] == pytest.approx(0.05)


def test_resume_filters_completed_task_ids_without_duplicate_requests():
    tasks = [
        {'task_id': 'main:0:target:0'},
        {'task_id': 'main:1:target:0'},
        {'task_id': 'main:1:target:1'},
    ]
    existing = [
        {'task_id': 'main:0:target:0', 'status': 'ok'},
        {
            'task_id': 'main:1:target:0',
            'status': 'grounder_unavailable',
        },
    ]

    pending = evaluator.filter_pending_tasks(tasks, existing)

    # Failed requests are completed measurements too: resuming must not
    # silently re-sample them and inflate model availability.
    assert pending == [{'task_id': 'main:1:target:1'}]
