"""CPU tests for formal VStar DINO evaluator run mechanics."""

import argparse
import json

import pytest
from PIL import Image

from eval.Oracle_experiment.vstar import (
    evaluate_dino_geometry_oracle_experts as evaluator,
)
from eval.Oracle_experiment.vstar.evaluate_dino_geometry_oracle_experts import (
    _atomic_write_jsonl,
    _experiment_signature,
    _latest_records_by_question_id,
    _record_events,
    _validate_resume_signatures,
    _worker_warmup,
)


def _args(tmp_path):
    baseline = tmp_path / 'baseline.jsonl'
    baseline.write_text('{"question_id":"main:0"}\n', encoding='utf-8')
    dino_python = tmp_path / 'python'
    dino_python.touch()
    dino_model = tmp_path / 'dino'
    dino_model.mkdir()
    generator = tmp_path / 'volcano'
    generator.mkdir()
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    return argparse.Namespace(
        run_split='full_238',
        image_dir=str(image_dir),
        model_path=str(generator),
        baseline_results=str(baseline),
        dino_python=str(dino_python),
        dino_model_path=str(dino_model),
        dino_dtype='float32',
        dino_box_threshold=0.3,
        dino_text_threshold=0.25,
        dino_top_k_log=20,
        geometry_accept_iou=0.5,
        geometry_containment=0.7,
        verifier_confidence_threshold=0.0,
        max_new_tokens=2048,
        temperature=0.0,
        likelihood_reduction='mean',
        context_window_tokens=48,
        worker_timeout=300.0,
        worker_fail_open=False,
        missing_expert_policy='fail_open',
    )


def test_experiment_signature_is_stable_and_parameter_sensitive(tmp_path):
    args = _args(tmp_path)
    first = _experiment_signature(args)
    second = _experiment_signature(args)
    assert first == second
    assert first['parameters']['baseline']['results_path'] == str(
        (tmp_path / 'baseline.jsonl').resolve()
    )
    assert len(first['parameters']['baseline']['sha256']) == 64

    args.geometry_accept_iou = 0.6
    assert _experiment_signature(args)['sha256'] != first['sha256']


def test_resume_requires_exact_signature_and_latest_record_wins(tmp_path):
    signature = _experiment_signature(_args(tmp_path))
    records = [
        {
            'question_id': 'main:0',
            'status': 'error',
            'experiment_signature': signature,
        },
        {
            'question_id': 'main:1',
            'status': 'ok',
            'experiment_signature': signature,
        },
        {
            'question_id': 'main:0',
            'status': 'ok',
            'experiment_signature': signature,
        },
    ]
    latest = _latest_records_by_question_id(records)
    assert len(latest) == 2
    assert next(
        row for row in latest if row['question_id'] == 'main:0'
    )['status'] == 'ok'
    _validate_resume_signatures(latest, signature)

    changed = dict(signature)
    changed['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='different experiment signature'):
        _validate_resume_signatures(latest, changed)


def test_atomic_jsonl_rewrite_and_event_rebuild_use_deduplicated_records(
        tmp_path):
    path = tmp_path / 'results.jsonl'
    records = [{
        'question_id': 'main:0',
        'status': 'ok',
        'intervention': {'events': [{'sample_id': 'main:0', 'audit': True}]},
    }]
    _atomic_write_jsonl(path, records)
    assert [json.loads(line) for line in path.read_text().splitlines()] \
        == records
    assert _record_events(records) == [
        {'sample_id': 'main:0', 'audit': True}
    ]


def test_worker_warmup_makes_a_real_verify_request_without_routing(tmp_path):
    args = _args(tmp_path)

    class Client:
        payload = None

        def request(self, payload, timeout):
            self.payload = dict(payload)
            return {
                'verifier_output_schema': 'vocot_four_action_v1',
                'predicted_action': 'no_action',
                'action_probabilities': None,
                'confidence': 0.9,
                'abstained': False,
                'error': None,
                'metadata': {'forward_executed': True},
            }

    client = Client()
    source = {
        'question_id': 'main:0',
        'image': 'sample.jpg',
        'oracle_targets': [{
            'object': 'tissue box',
            'box': [0.1, 0.2, 0.3, 0.4],
        }],
    }
    result = _worker_warmup(client, source, args)

    assert client.payload['operation'] == 'verify'
    assert client.payload['object_reference'] == 'tissue box'
    assert client.payload['candidate_bbox'] == [0.1, 0.2, 0.3, 0.4]
    assert result['affects_routing'] is False
    assert result['validated_output']['predicted_action'] == 'no_action'


def _main_fixture(tmp_path, *, fail_fast=False):
    args = _args(tmp_path)
    args.output = None
    args.output_root = str(tmp_path / 'output')
    args.run_id = 'formal_run'
    args.verifier_log = None
    args.dino_gpu = '7'
    args.start_index = 0
    args.max_samples = 1
    args.sample_id = None
    args.fail_fast = fail_fast
    args.no_resume = True
    args.verbose = False
    image_path = tmp_path / 'images' / 'sample.jpg'
    Image.new('RGB', (20, 10)).save(image_path)
    source = {
        'sample_index': 0,
        'question_id': 'main:0',
        'image': 'sample.jpg',
        'category': 'direct_attributes',
        'question': 'What is the cup?',
        'options': ['red', 'blue'],
        'label': 0,
        'source_jsonl_label': 'A',
        'oracle_targets': [{
            'object': 'cup',
            'aliases': ['cup'],
            'box': [0.2, 0.3, 0.6, 0.7],
        }],
        'source_oracle_boxes': [[0.2, 0.1, 0.6, 0.5]],
        'oracle_box_coordinate_system': (
            'normalized_xyxy_on_center_padded_square'
        ),
        'source_image_size': {'width': 20, 'height': 10},
        'has_complete_question_target_coverage': True,
        'baseline': {'response': 'baseline'},
        'baseline_prediction': 0,
        'baseline_answer': 'red',
        'status': 'ok',
    }
    (tmp_path / 'baseline.jsonl').write_text(
        json.dumps(source) + '\n',
        encoding='utf-8',
    )
    return args


class _FakeWorkerClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def ping(self, timeout):
        return {'configured': True, 'worker': 'fake_dino'}

    def request(self, payload, timeout):
        return {
            'verifier_output_schema': 'vocot_four_action_v1',
            'predicted_action': 'no_action',
            'action_probabilities': None,
            'confidence': 0.9,
            'abstained': False,
            'error': None,
            'metadata': {'warmup_forward_executed': True},
        }

    def close(self):
        self.closed = True


def _fake_routed_result(**kwargs):
    assert kwargs['log_path'] is None
    assert kwargs['missing_expert_policy'] == 'fail_open'
    box = [0.2, 0.3, 0.6, 0.7]
    event = {
        'object_reference': 'Find the cup',
        'candidate_box': box,
        'committed_box': box,
        'predicted_action': 'no_action',
        'router_action': 'verified_accept',
        'grounder_invoked': False,
        'box_refiner_invoked': False,
        'missing_expert_error': None,
        'verifier_abstained': False,
        'verifier_metadata': {
            'selected_grounding_padded_normalized_bbox_xyxy': box,
        },
        'expert_metadata': None,
        'committed_iou_to_gt': None,
    }
    return {
        'response': 'Find the cup <coor> 0.2,0.3,0.6,0.7</coor>',
        'generated_ids': [1, 2, 3],
        'boxes': [box],
        'bound_boxes': [box],
        'finished_with_eos': True,
        'events': [event],
        'status': 'ok',
    }


def test_main_writes_signature_audited_events_and_closes_worker(
        tmp_path, monkeypatch):
    args = _main_fixture(tmp_path)
    _FakeWorkerClient.instances.clear()
    monkeypatch.setattr(evaluator, 'parse_args', lambda: args)
    monkeypatch.setattr(
        evaluator,
        'PersistentJsonlWorkerClient',
        _FakeWorkerClient,
    )
    monkeypatch.setattr(
        evaluator,
        'load_model',
        lambda *args, **kwargs: (object(), argparse.Namespace(tokenizer=object())),
    )
    monkeypatch.setattr(evaluator, 'routing_infer', _fake_routed_result)
    monkeypatch.setattr(evaluator, 'score_options', lambda *args: 0)

    evaluator.main()

    run_dir = (
        tmp_path / 'output' / 'vstar' / 'runs' / 'full_238'
        / 'routing' / 'dino_geometry__oracle_experts' / 'iou_0p5'
        / 'formal_run'
    )
    record = json.loads(
        (run_dir / 'results.jsonl').read_text(encoding='utf-8')
    )
    config = json.loads(
        (run_dir / 'run.config.json').read_text(encoding='utf-8')
    )
    summary = json.loads(
        (run_dir / 'results.summary.json').read_text(encoding='utf-8')
    )
    status = json.loads(
        (run_dir / 'run.status.json').read_text(encoding='utf-8')
    )
    event = json.loads(
        (run_dir / 'verifier_events.jsonl').read_text(encoding='utf-8')
    )
    assert record['experiment_signature'] == config['experiment_signature']
    assert summary['experiment_signature'] == record['experiment_signature']
    assert config['worker_warmup']['status'] == 'ok'
    assert event['posthoc_oracle_audit']['affects_routing'] is False
    assert status['status'] == 'completed'
    assert _FakeWorkerClient.instances[-1].closed is True


def test_fail_fast_records_sample_and_failed_lifecycle_then_closes_worker(
        tmp_path, monkeypatch):
    args = _main_fixture(tmp_path, fail_fast=True)
    _FakeWorkerClient.instances.clear()
    monkeypatch.setattr(evaluator, 'parse_args', lambda: args)
    monkeypatch.setattr(
        evaluator,
        'PersistentJsonlWorkerClient',
        _FakeWorkerClient,
    )
    monkeypatch.setattr(
        evaluator,
        'load_model',
        lambda *args, **kwargs: (object(), argparse.Namespace(tokenizer=object())),
    )
    monkeypatch.setattr(
        evaluator,
        'routing_infer',
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError('broken sample')),
    )

    with pytest.raises(RuntimeError, match='--fail-fast'):
        evaluator.main()

    run_dir = (
        tmp_path / 'output' / 'vstar' / 'runs' / 'full_238'
        / 'routing' / 'dino_geometry__oracle_experts' / 'iou_0p5'
        / 'formal_run'
    )
    record = json.loads(
        (run_dir / 'results.jsonl').read_text(encoding='utf-8')
    )
    status = json.loads(
        (run_dir / 'run.status.json').read_text(encoding='utf-8')
    )
    assert record['status'] == 'error'
    assert record['experiment_signature']['sha256']
    assert status['status'] == 'failed'
    assert _FakeWorkerClient.instances[-1].closed is True
