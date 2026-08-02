"""CPU-only runtime tests for the VStar DINO/Qwen expert evaluator."""

import argparse
import json
from pathlib import Path

import pytest
from PIL import Image

from eval.Oracle_experiment.vstar import (
    evaluate_dino_geometry_qwen_grounder_oracle_refiner as evaluator,
)
from grounding_control.contracts import VerificationRequest
from grounding_control.experts.grounders import RemoteGrounderBackend
from grounding_control.four_way import (
    OracleBoxRefinerBackend,
    RemoteActionVerifierBackend,
)


def _args(tmp_path, *, fail_fast=False):
    image_dir = tmp_path / 'images'
    image_dir.mkdir(parents=True)
    Image.new('RGB', (20, 10)).save(image_dir / 'sample.jpg')

    baseline = tmp_path / 'baseline.jsonl'
    baseline.write_text(json.dumps({
        'sample_index': 0,
        'question_id': 'main:0',
        'image': 'sample.jpg',
        'category': 'direct_attributes',
        'question': 'What color is the cup?',
        'options': ['red', 'blue'],
        'label': 0,
        'source_jsonl_label': 'A',
        'oracle_targets': [{
            'object': 'cup',
            'aliases': ['cup'],
            # The equivalent original-image pixel box is [4, 1, 12, 9].
            'box': [0.2, 0.3, 0.6, 0.7],
        }],
        'source_oracle_boxes': [[0.2, 0.3, 0.6, 0.7]],
        'oracle_box_coordinate_system': (
            'normalized_xyxy_on_center_padded_square'
        ),
        'source_image_size': {'width': 20, 'height': 10},
        'has_complete_question_target_coverage': True,
        'baseline': {'response': 'baseline response'},
        'baseline_prediction': 1,
        'baseline_answer': 'blue',
        'status': 'ok',
    }) + '\n', encoding='utf-8')

    dino_python = tmp_path / 'dino_python'
    dino_python.touch()
    qwen_python = tmp_path / 'qwen_python'
    qwen_python.touch()
    dino_model = tmp_path / 'dino_model'
    dino_model.mkdir()
    qwen_model = tmp_path / 'qwen_model'
    qwen_model.mkdir()
    generator = tmp_path / 'volcano'
    generator.mkdir()

    return argparse.Namespace(
        model_path=str(generator),
        baseline_results=str(baseline),
        image_dir=str(image_dir),
        output=None,
        output_root=str(tmp_path / 'output'),
        run_split='full_238',
        run_id='formal_run',
        verifier_log=None,
        dino_python=str(dino_python),
        dino_model_path=str(dino_model),
        dino_gpu='7',
        dino_dtype='float32',
        dino_box_threshold=0.3,
        dino_text_threshold=0.25,
        geometry_accept_iou=0.4,
        geometry_containment=0.7,
        dino_top_k_log=20,
        dino_worker_timeout=30.0,
        dino_worker_fail_open=False,
        qwen_python=str(qwen_python),
        qwen_model_path=str(qwen_model),
        qwen_gpu='5',
        qwen_dtype='bfloat16',
        qwen_max_new_tokens=64,
        qwen_min_pixels=3136,
        qwen_max_pixels=1003520,
        qwen_attn_implementation='sdpa',
        qwen_prompt_protocol='compact_json_v1',
        qwen_boundary_tolerance_pixels=1.0,
        qwen_worker_timeout=45.0,
        verifier_confidence_threshold=0.0,
        context_window_tokens=48,
        max_new_tokens=128,
        temperature=0.0,
        likelihood_reduction='mean',
        start_index=0,
        max_samples=1,
        sample_id=None,
        missing_expert_policy='fail_open',
        fail_fast=fail_fast,
        no_resume=True,
        verbose=False,
    )


def _run_dir(tmp_path):
    return (
        tmp_path / 'output' / 'vstar' / 'runs' / 'full_238'
        / 'routing'
        / 'dino_geometry__qwen25vl_grounder__oracle_refiner'
        / 'iou_0p4' / 'formal_run'
    )


class _FakeWorkerClient:
    """Role-aware substitute for two independent persistent workers."""

    instances = []

    def __init__(self, command, *args, **kwargs):
        del args, kwargs
        command_text = ' '.join(map(str, command))
        if 'grounding_control.workers.qwen_grounder' in command_text:
            self.role = 'qwen'
        elif (
            'grounding_control.four_way.workers.dino_geometry_verifier'
            in command_text
        ):
            self.role = 'dino'
        else:  # Make an evaluator wiring typo immediately visible.
            raise AssertionError(f'unknown worker command: {command_text}')
        self.command = list(command)
        self.started = False
        self.closed = False
        self.requests = []
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def ping(self, timeout):
        assert self.started
        assert timeout == 30.0
        return {'configured': True, 'worker': f'fake_{self.role}'}

    def request(self, payload, timeout):
        assert self.started
        self.requests.append(dict(payload))
        if self.role == 'dino':
            assert payload['operation'] == 'verify'
            assert timeout == 30.0
            return {
                'verifier_output_schema': 'vocot_four_action_v1',
                'predicted_action': 'relocate',
                'action_probabilities': None,
                'confidence': 0.9,
                'abstained': False,
                'error': None,
                'metadata': {'forward_executed': True},
            }

        assert payload['operation'] == 'ground'
        assert timeout == 45.0
        assert payload['object_reference'] == 'cup'
        return {
            'grounder_output_schema': 'vocot_grounder_output_v1',
            'available': True,
            'source': 'qwen25_vl_grounder',
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'bbox': [4.0, 1.0, 12.0, 9.0],
            'image_size': [20, 10],
            'confidence': None,
            'error': None,
            'metadata': {
                'prompt_protocol': 'compact_json_v1',
                'raw_response': '{"bbox_2d":[4,1,12,9]}',
            },
        }

    def close(self):
        self.closed = True


def _routed_result_with_real_remote_grounder(captured, **kwargs):
    """Exercise the injected remote adapter while replacing only Volcano."""

    captured.update(kwargs)
    assert isinstance(
        kwargs['verifier_backend'], RemoteActionVerifierBackend
    )
    grounder = kwargs['grounder_backend']
    assert isinstance(grounder, RemoteGrounderBackend)
    assert grounder.source == 'qwen25_vl_grounder'
    assert isinstance(
        kwargs['box_refiner_backend'], OracleBoxRefinerBackend
    )
    assert kwargs['missing_expert_policy'] == 'fail_open'
    assert kwargs['log_path'] is None
    assert kwargs['sample_context']['question'] == 'What color is the cup?'
    assert kwargs['sample_context']['oracle_targets'][0]['object'] == 'cup'

    request = VerificationRequest(
        sample_id=kwargs['sample_id'],
        grounding_step=1,
        object_reference='cup',
        candidate_bbox=(0.0, 0.0, 0.1, 0.1),
        candidate_coordinate_text='<coor>0,0,0.1,0.1</coor>',
        generated_ids=(1, 2),
        candidate_span=(0, 1),
        sample_context=kwargs['sample_context'],
    )
    grounded = grounder.ground(request.grounding_request())
    # This also proves the remote backend, not the worker/transport, owns the
    # original-pixel -> VoCoT center-padded coordinate conversion.
    assert grounded.bbox == pytest.approx((0.2, 0.3, 0.6, 0.7))
    assert grounded.metadata['bbox_original_pixel_xyxy'] == [
        4.0, 1.0, 12.0, 9.0
    ]

    committed = list(grounded.bbox)
    candidate = [0.0, 0.0, 0.1, 0.1]
    event = {
        'sample_id': kwargs['sample_id'],
        'grounding_step': 1,
        'object_reference': 'Find the cup',
        'candidate_box': candidate,
        'committed_box': committed,
        'predicted_action': 'relocate',
        'routing_decision': 'relocate',
        'router_action': 'routed_to_grounder',
        'expert_role': 'grounder',
        'expert_metadata': dict(grounded.metadata),
        'grounder_invoked': True,
        'box_refiner_invoked': False,
        'missing_expert_error': None,
        'verifier_abstained': False,
        'verifier_metadata': {
            'selected_grounding_padded_normalized_bbox_xyxy': candidate,
        },
        'committed_iou_to_gt': None,
    }
    return {
        'response': 'Find the cup <coor> 0.2,0.3,0.6,0.7</coor>',
        'generated_ids': [1, 2, 3],
        'boxes': [committed],
        'bound_boxes': [committed],
        'finished_with_eos': True,
        'events': [event],
        'status': 'ok',
    }


def test_signature_and_commands_name_qwen_without_changing_oracle_method(
        tmp_path):
    args = _args(tmp_path)
    signature = evaluator._experiment_signature(args)
    parameters = signature['parameters']

    assert evaluator.METHOD_NAME == (
        'dino_geometry__qwen25vl_grounder__oracle_refiner'
    )
    assert parameters['grounder'] == {
        'backend': 'qwen25_vl',
        'python': str((tmp_path / 'qwen_python').resolve()),
        'model_path': str((tmp_path / 'qwen_model').resolve()),
        'dtype': 'bfloat16',
        'max_new_tokens': 64,
        'min_pixels': 3136,
        'max_pixels': 1003520,
        'attn_implementation': 'sdpa',
        'prompt_protocol': 'compact_json_v1',
        'boundary_tolerance_pixels': 1.0,
        'input': 'clean_original_image_plus_local_object_reference',
        'candidate_box_exposed': False,
        'output_coordinate_system': 'absolute_xyxy_on_original_image',
    }
    assert parameters['experts'] == {
        'relocate': 'qwen25_vl_grounder',
        'expand': 'oracle_box_refiner',
        'tighten': 'oracle_box_refiner',
    }
    assert 'grounding_control.workers.qwen_grounder' in ' '.join(
        evaluator._qwen_worker_command(args)
    )
    assert (
        'grounding_control.four_way.workers.dino_geometry_verifier'
        in ' '.join(
            evaluator._dino_worker_command(args)
        )
    )

    changed = _args(tmp_path / 'changed')
    changed.qwen_dtype = 'float16'
    assert evaluator._experiment_signature(changed)['sha256'] != (
        signature['sha256']
    )


def test_main_starts_independent_workers_injects_experts_and_writes_outputs(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    _FakeWorkerClient.instances.clear()
    captured = {}
    monkeypatch.setattr(evaluator, 'parse_args', lambda: args)
    monkeypatch.setattr(
        evaluator,
        'PersistentJsonlWorkerClient',
        _FakeWorkerClient,
    )
    monkeypatch.setattr(
        evaluator,
        'load_model',
        lambda *args, **kwargs: (
            object(), argparse.Namespace(tokenizer=object())
        ),
    )
    monkeypatch.setattr(
        evaluator,
        'routing_infer',
        lambda **kwargs: _routed_result_with_real_remote_grounder(
            captured, **kwargs
        ),
    )
    monkeypatch.setattr(evaluator, 'score_options', lambda *args: 0)

    evaluator.main()

    assert [client.role for client in _FakeWorkerClient.instances] == [
        'dino', 'qwen'
    ]
    assert all(client.started for client in _FakeWorkerClient.instances)
    assert all(client.closed for client in _FakeWorkerClient.instances)
    dino_client, qwen_client = _FakeWorkerClient.instances
    assert captured['verifier_backend'].client is dino_client
    assert captured['grounder_backend'].client is qwen_client
    assert [request['operation'] for request in dino_client.requests] == [
        'verify'
    ]
    # One environment warm-up and one real GrounderBackend call.
    assert [request['operation'] for request in qwen_client.requests] == [
        'ground', 'ground'
    ]

    run_dir = _run_dir(tmp_path)
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

    assert record['status'] == 'ok'
    assert record['router_prediction'] == 0
    assert record['router_prediction_correct'] is True
    assert record['experiment_signature'] == config['experiment_signature']
    assert summary['experiment_signature'] == record['experiment_signature']
    assert config['method'] == evaluator.METHOD_NAME
    assert config['components'] == {
        'generator': args.model_path,
        'verifier': 'grounding_dino_geometry',
        'grounder': 'qwen25_vl',
        'box_refiner': 'oracle',
    }
    assert config['workers']['dino_verifier']['warmup']['status'] == 'ok'
    assert config['workers']['qwen_grounder']['warmup']['status'] == 'ok'
    assert summary['workers']['dino_verifier']['ping']['configured'] is True
    assert summary['workers']['qwen_grounder']['ping']['configured'] is True
    assert summary['all_samples']['qwen_grounder'] == {
        'relocate_event_count': 1,
        'successful_invocation_count': 1,
        'unavailable_fail_open_count': 0,
        'parse_failure_count': 0,
        'unavailable_reason_counts': {},
        'success_rate_on_relocate': 1.0,
        'matchable_successful_invocation_count': 1,
        'mean_committed_minus_candidate_iou': 1.0,
        'improved_iou_count': 1,
        'unchanged_iou_count': 0,
        'degraded_iou_count': 0,
    }
    assert event['posthoc_oracle_audit']['affects_routing'] is False
    assert event['posthoc_oracle_audit']['committed_iou_to_gt'] == 1.0
    assert status['status'] == 'completed'


def test_fail_fast_records_error_marks_run_failed_and_closes_both_workers(
        tmp_path, monkeypatch):
    args = _args(tmp_path, fail_fast=True)
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
        lambda *args, **kwargs: (
            object(), argparse.Namespace(tokenizer=object())
        ),
    )
    monkeypatch.setattr(
        evaluator,
        'routing_infer',
        lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError('broken routed sample')
        ),
    )

    with pytest.raises(RuntimeError, match='sample failed under --fail-fast'):
        evaluator.main()

    record = json.loads(
        (_run_dir(tmp_path) / 'results.jsonl').read_text(encoding='utf-8')
    )
    status = json.loads(
        (_run_dir(tmp_path) / 'run.status.json').read_text(encoding='utf-8')
    )
    assert record['status'] == 'error'
    assert record['error'] == 'RuntimeError: broken routed sample'
    assert record['experiment_signature']['sha256']
    assert status['status'] == 'failed'
    assert 'sample failed under --fail-fast' in status['error']
    assert [client.role for client in _FakeWorkerClient.instances] == [
        'dino', 'qwen'
    ]
    assert all(client.closed for client in _FakeWorkerClient.instances)
