"""CPU tests for the VStar oracle-verifier/Qwen-Grounder evaluator."""

import argparse
import json

import pytest
from PIL import Image

from eval.grounding_control.vstar import (
    evaluate_oracle_verifier_qwen_grounder as evaluator,
)
from grounding_control.core import AlignmentRoutingPolicy
from grounding_control.experts.grounders import RemoteGrounderBackend
from grounding_control.verifiers import OracleAlignmentVerifierBackend


def _args(tmp_path):
    baseline = tmp_path / 'baseline.jsonl'
    baseline.write_text('{"question_id":"main:0"}\n', encoding='utf-8')
    qwen_python = tmp_path / 'qwen-python'
    qwen_python.touch()
    qwen_model = tmp_path / 'qwen-model'
    qwen_model.mkdir()
    volcano = tmp_path / 'volcano'
    volcano.mkdir()
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    return argparse.Namespace(
        model_path=str(volcano),
        baseline_results=str(baseline),
        image_dir=str(image_dir),
        output=None,
        output_root=str(tmp_path / 'output'),
        run_split='full_238',
        run_id='formal_run',
        verifier_log=None,
        qwen_python=str(qwen_python),
        qwen_model_path=str(qwen_model),
        qwen_gpu='7',
        qwen_dtype='bfloat16',
        qwen_max_new_tokens=64,
        qwen_min_pixels=3136,
        qwen_max_pixels=12_000_000,
        qwen_attn_implementation='flash_attention_2',
        qwen_prompt_protocol='compact_json_v1',
        qwen_boundary_tolerance_pixels=1.0,
        worker_timeout=600.0,
        oracle_iou_threshold=0.5,
        reject_threshold=0.25,
        accept_threshold=0.75,
        context_window_tokens=48,
        missing_expert_policy='fail_open',
        max_new_tokens=2048,
        temperature=0.0,
        likelihood_reduction='mean',
        start_index=0,
        max_samples=1,
        sample_id=None,
        fail_fast=True,
        no_resume=True,
        verbose=False,
    )


def test_signature_and_worker_command_record_real_qwen_configuration(tmp_path):
    args = _args(tmp_path)
    first = evaluator._experiment_signature(args)
    second = evaluator._experiment_signature(args)
    assert first == second
    assert first['parameters']['verifier'] == {
        'backend': 'oracle_binary_iou_label',
        'gt_iou_threshold': 0.5,
        'unmatched_policy': 'abstain_then_controller_fail_open',
        'context_window_tokens': 48,
    }
    assert first['parameters']['grounder']['max_pixels'] == 12_000_000

    command = evaluator._worker_command(args)
    assert 'grounding_control.workers.qwen_grounder' in command
    assert command[command.index('--max-pixels') + 1] == '12000000'
    assert command[command.index('--attn-implementation') + 1] \
        == 'flash_attention_2'

    args.oracle_iou_threshold = 0.4
    assert evaluator._experiment_signature(args)['sha256'] != first['sha256']


def test_routing_metrics_measure_qwen_correction_not_only_call_count():
    records = [{
        'intervention': {'events': [
            {
                'grounding_step': 1,
                'match_status': 'matched_unique_explicit_target',
                'decision_band': 'reject',
                'grounder_requested': True,
                'grounder_attempted': True,
                'grounder_succeeded': True,
                'candidate_iou_to_gt': 0.1,
                'committed_iou_to_gt': 0.8,
                'verifier_abstained': False,
                'missing_expert_error': None,
            },
            {
                'grounding_step': 2,
                'match_status': 'unverifiable_abstain',
                'decision_band': 'verifier_failure',
                'grounder_requested': False,
                'grounder_attempted': False,
                'grounder_succeeded': False,
                'candidate_iou_to_gt': None,
                'committed_iou_to_gt': None,
                'verifier_abstained': True,
                'missing_expert_error': None,
            },
        ]},
    }]

    metrics = evaluator.routing_metrics(records)

    assert metrics['coordinate_event_count'] == 2
    assert metrics['grounder_requested_count'] == 1
    assert metrics['grounder_succeeded_count'] == 1
    assert metrics['grounder_committed_miou'] == 0.8
    assert metrics['mean_iou_gain_on_successful_grounder_calls'] == pytest.approx(0.7)
    assert metrics['candidate_lt_0p5_to_committed_ge_0p5_count'] == 1
    assert metrics['unverifiable_fail_open_count'] == 1


class _FakeWorkerClient:
    instances = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def ping(self, timeout):
        del timeout
        return {
            'configured': True,
            'worker': 'qwen25_vl_grounder',
            'max_pixels': 12_000_000,
        }

    def request(self, payload, timeout):
        del timeout
        assert payload['operation'] == 'ground'
        return {
            'grounder_output_schema': 'vocot_grounder_output_v1',
            'available': True,
            'source': 'qwen25_vl_grounder',
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'bbox': [2.0, 1.0, 12.0, 8.0],
            'image_size': [20, 10],
            'confidence': None,
            'error': None,
            'metadata': {'raw_response': '{"bbox_2d":[2,1,12,8]}'},
        }

    def close(self):
        self.closed = True


def test_main_composes_binary_oracle_verifier_and_remote_qwen_grounder(
        tmp_path, monkeypatch):
    args = _args(tmp_path)
    image_path = tmp_path / 'images' / 'sample.jpg'
    Image.new('RGB', (20, 10), 'white').save(image_path)
    source = {
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
            'box': [0.2, 0.3, 0.6, 0.7],
        }],
        'source_oracle_boxes': [[0.2, 0.1, 0.6, 0.5]],
        'oracle_box_coordinate_system': (
            'normalized_xyxy_on_center_padded_square'
        ),
        'source_image_size': {'width': 20, 'height': 10},
        'has_complete_question_target_coverage': True,
        'baseline': {'response': 'baseline'},
        'baseline_prediction': 1,
        'baseline_answer': 'blue',
        'status': 'ok',
    }
    (tmp_path / 'baseline.jsonl').write_text(
        json.dumps(source) + '\n',
        encoding='utf-8',
    )

    observed = {}

    def fake_routing_infer(**kwargs):
        observed.update(kwargs)
        assert isinstance(
            kwargs['verifier_backend'], OracleAlignmentVerifierBackend
        )
        assert isinstance(kwargs['grounder_backend'], RemoteGrounderBackend)
        assert isinstance(
            kwargs['alignment_routing_policy'], AlignmentRoutingPolicy
        )
        assert kwargs['sample_context'] == {
            'image_path': str(image_path.resolve())
        }
        box = [0.2, 0.3, 0.6, 0.7]
        return {
            'response': 'Find the cup <coor> 0.2,0.3,0.6,0.7</coor>',
            'generated_ids': [1, 2, 3],
            'boxes': [box],
            'bound_boxes': [box],
            'finished_with_eos': True,
            'status': 'ok',
            'events': [{
                'grounding_step': 1,
                'match_status': 'matched_unique_explicit_target',
                'decision_band': 'reject',
                'grounder_requested': True,
                'grounder_attempted': True,
                'grounder_succeeded': True,
                'candidate_iou_to_gt': 0.1,
                'committed_iou_to_gt': 0.8,
                'verifier_abstained': False,
                'missing_expert_error': None,
            }],
        }

    _FakeWorkerClient.instances.clear()
    monkeypatch.setattr(evaluator, 'parse_args', lambda: args)
    monkeypatch.setattr(
        evaluator, 'PersistentJsonlWorkerClient', _FakeWorkerClient
    )
    monkeypatch.setattr(
        evaluator,
        'load_model',
        lambda *args, **kwargs: (
            object(),
            argparse.Namespace(tokenizer=object()),
        ),
    )
    monkeypatch.setattr(evaluator, 'routing_infer', fake_routing_infer)
    monkeypatch.setattr(evaluator, 'score_options', lambda *args: 0)

    evaluator.main()

    run_dir = (
        tmp_path / 'output' / 'vstar' / 'runs' / 'full_238'
        / 'routing' / 'oracle_verifier__qwen25_vl_7b_grounder'
        / 'gt_iou_0p5__12mp' / 'formal_run'
    )
    record = json.loads((run_dir / 'results.jsonl').read_text())
    summary = json.loads((run_dir / 'results.summary.json').read_text())
    config = json.loads((run_dir / 'run.config.json').read_text())
    status = json.loads((run_dir / 'run.status.json').read_text())
    event = json.loads((run_dir / 'verifier_events.jsonl').read_text())

    assert record['router_prediction_correct'] is True
    assert record['baseline_prediction_correct'] is False
    assert summary['all_samples']['router_minus_baseline'] == 1.0
    assert summary['all_samples']['routing']['grounder_succeeded_count'] == 1
    assert config['components']['verifier'] == 'oracle_binary_iou_label'
    assert config['components']['grounder'] == 'qwen25_vl_grounder'
    assert status['status'] == 'completed'
    assert event['grounder_succeeded'] is True
    assert observed['missing_expert_policy'] == 'fail_open'
    assert _FakeWorkerClient.instances[-1].closed is True
