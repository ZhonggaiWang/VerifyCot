"""Evaluate DINO verification with Qwen grounding and oracle refinement.

This evaluator intentionally lives beside, rather than modifying,
``evaluate_dino_geometry_oracle_experts.py``.  The frozen oracle-expert
evaluator remains the exact ceiling experiment.  Here Grounding DINO judges
every uncommitted VoCoT coordinate, Qwen2.5-VL handles only ``relocate``, and
the oracle BoxRefiner handles ``expand``/``tighten``.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.load_model import load_model, routing_infer
from utils.coordinate_intervention import ExplicitOracleTargetMatcher
from grounding_control.experts.grounders import RemoteGrounderBackend
from grounding_control.four_way import (
    OracleBoxRefinerBackend,
    RemoteActionVerifierBackend,
    RoutingPolicy,
)
from grounding_control.models.qwen25_vl import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS
from grounding_control.models.qwen25_vl.grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
)
from grounding_control.oracle_targets import OracleTargetResolver
from grounding_control.run_paths import (
    create_run_layout,
    write_run_config,
    write_run_status,
)
from grounding_control.transport import (
    PersistentJsonlWorkerClient,
    parse_grounder_output,
)

from eval.Oracle_experiment.vstar import (
    evaluate_dino_geometry_oracle_experts as oracle_eval,
)
from eval.Oracle_experiment.vstar.evaluate_selective_oracle_router import (
    make_conversation,
    read_jsonl,
    score_options,
)


ORACLE_BOX_COORDINATE_SYSTEM = oracle_eval.ORACLE_BOX_COORDINATE_SYSTEM
DEFAULT_DINO_PYTHON = oracle_eval.DEFAULT_DINO_PYTHON
DEFAULT_DINO_MODEL = oracle_eval.DEFAULT_DINO_MODEL
DEFAULT_QWEN_PYTHON = '/home/zhonggai/miniconda3/envs/qwen25/bin/python'
DEFAULT_QWEN_MODEL = '/data/zhonggai/models/Qwen2.5-VL-7B-Instruct'
EXPERIMENT_MODE = 'remote_dino_geometry_qwen_grounder_oracle_refiner'
METHOD_NAME = 'dino_geometry__qwen25vl_grounder__oracle_refiner'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument(
        '--baseline-results',
        default=(
            'output/vstar/online_oracle/'
            'full_238_padding_fix/results.jsonl'
        ),
    )
    parser.add_argument('--image-dir', default='/data/zhonggai/VStar')
    parser.add_argument('--output', default=None)
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-split', default='full_238')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--verifier-log', default=None)

    parser.add_argument('--dino-python', default=DEFAULT_DINO_PYTHON)
    parser.add_argument('--dino-model-path', default=DEFAULT_DINO_MODEL)
    parser.add_argument('--dino-gpu', default='7')
    parser.add_argument('--dino-dtype', default='float32')
    parser.add_argument('--dino-box-threshold', type=float, default=0.3)
    parser.add_argument('--dino-text-threshold', type=float, default=0.25)
    parser.add_argument('--geometry-accept-iou', type=float, default=0.4)
    parser.add_argument('--geometry-containment', type=float, default=0.7)
    parser.add_argument('--dino-top-k-log', type=int, default=20)
    parser.add_argument('--dino-worker-timeout', type=float, default=300.0)
    parser.add_argument('--dino-worker-fail-open', action='store_true')

    parser.add_argument('--qwen-python', default=DEFAULT_QWEN_PYTHON)
    parser.add_argument('--qwen-model-path', default=DEFAULT_QWEN_MODEL)
    parser.add_argument('--qwen-gpu', default='5')
    parser.add_argument('--qwen-dtype', default='bfloat16')
    parser.add_argument('--qwen-max-new-tokens', type=int, default=64)
    parser.add_argument('--qwen-min-pixels', type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument('--qwen-max-pixels', type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument('--qwen-attn-implementation', default='sdpa')
    parser.add_argument(
        '--qwen-prompt-protocol',
        choices=GROUNDING_PROMPT_PROTOCOLS,
        default=DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    )
    parser.add_argument(
        '--qwen-boundary-tolerance-pixels',
        type=float,
        default=1.0,
    )
    parser.add_argument('--qwen-worker-timeout', type=float, default=300.0)

    parser.add_argument(
        '--verifier-confidence-threshold',
        type=float,
        default=0.0,
    )
    parser.add_argument('--context-window-tokens', type=int, default=48)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument(
        '--likelihood-reduction',
        choices=('mean', 'sum'),
        default='mean',
    )
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-id', default=None)
    parser.add_argument(
        '--missing-expert-policy',
        choices=('fail_open', 'error'),
        default='fail_open',
    )
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def _validate_args(args):
    for value, name in (
        (args.dino_box_threshold, 'dino-box-threshold'),
        (args.dino_text_threshold, 'dino-text-threshold'),
        (args.verifier_confidence_threshold,
         'verifier-confidence-threshold'),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f'--{name} must be in [0, 1]')
    for value, name in (
        (args.geometry_accept_iou, 'geometry-accept-iou'),
        (args.geometry_containment, 'geometry-containment'),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f'--{name} must be in (0, 1]')
    for value, name in (
        (args.dino_worker_timeout, 'dino-worker-timeout'),
        (args.qwen_worker_timeout, 'qwen-worker-timeout'),
    ):
        if float(value) <= 0:
            raise ValueError(f'--{name} must be positive')
    if args.context_window_tokens <= 0:
        raise ValueError('--context-window-tokens must be positive')
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    if args.dino_top_k_log <= 0:
        raise ValueError('--dino-top-k-log must be positive')
    if args.qwen_max_new_tokens <= 0:
        raise ValueError('--qwen-max-new-tokens must be positive')
    if args.qwen_min_pixels <= 0 or args.qwen_max_pixels <= 0:
        raise ValueError('Qwen pixel limits must be positive')
    if args.qwen_min_pixels > args.qwen_max_pixels:
        raise ValueError('--qwen-min-pixels must not exceed --qwen-max-pixels')
    if args.qwen_boundary_tolerance_pixels < 0:
        raise ValueError('--qwen-boundary-tolerance-pixels must be non-negative')
    for path, name in (
        (args.dino_python, 'dino-python'),
        (args.qwen_python, 'qwen-python'),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f'--{name}: {path}')
    for path, name in (
        (args.dino_model_path, 'dino-model-path'),
        (args.qwen_model_path, 'qwen-model-path'),
    ):
        if not Path(path).is_dir():
            raise FileNotFoundError(f'--{name}: {path}')


def _experiment_signature(args):
    baseline_path = Path(args.baseline_results).resolve()
    payload = {
        'schema': 'vstar_dino_qwen_grounder_oracle_refiner_signature_v1',
        'dataset': {
            'name': 'vstar',
            'split': args.run_split,
            'image_dir': str(Path(args.image_dir).resolve()),
        },
        'generator': {
            'model_path': str(Path(args.model_path).resolve()),
            'precision': 'fp16',
            'cot': True,
        },
        'baseline': {
            'results_path': str(baseline_path),
            'sha256': oracle_eval._sha256_file(baseline_path),
        },
        'verifier': {
            'backend': 'grounding_dino_geometry',
            'python': str(Path(args.dino_python).resolve()),
            'model_path': str(Path(args.dino_model_path).resolve()),
            'dtype': args.dino_dtype,
            'box_threshold': float(args.dino_box_threshold),
            'text_threshold': float(args.dino_text_threshold),
            'top_k_log': int(args.dino_top_k_log),
            'geometry_accept_iou': float(args.geometry_accept_iou),
            'geometry_containment': float(args.geometry_containment),
            'confidence_threshold': float(
                args.verifier_confidence_threshold
            ),
        },
        'grounder': {
            'backend': 'qwen25_vl',
            'python': str(Path(args.qwen_python).resolve()),
            'model_path': str(Path(args.qwen_model_path).resolve()),
            'dtype': args.qwen_dtype,
            'max_new_tokens': int(args.qwen_max_new_tokens),
            'min_pixels': int(args.qwen_min_pixels),
            'max_pixels': int(args.qwen_max_pixels),
            'attn_implementation': args.qwen_attn_implementation,
            'prompt_protocol': args.qwen_prompt_protocol,
            'boundary_tolerance_pixels': float(
                args.qwen_boundary_tolerance_pixels
            ),
            'input': 'clean_original_image_plus_local_object_reference',
            'candidate_box_exposed': False,
            'output_coordinate_system': 'absolute_xyxy_on_original_image',
        },
        'box_refiner': {
            'backend': 'oracle_box_refiner',
            'alias_policy': ExplicitOracleTargetMatcher.POLICY,
        },
        'generation': {
            'max_new_tokens': int(args.max_new_tokens),
            'temperature': float(args.temperature),
        },
        'routing_runtime': {
            'verify_every_coordinate': True,
            'cross_call_kv_cache': False,
            'expert_coordinate_commit': (
                'local_roundtrip_then_single_clean_replay'
            ),
        },
        'scoring': {
            'method': 'option_conditional_likelihood',
            'likelihood_reduction': args.likelihood_reduction,
            'further_instruct': True,
        },
        'context': {
            'context_window_tokens': int(args.context_window_tokens),
            'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
        },
        'worker_policy': {
            'dino_timeout_seconds': float(args.dino_worker_timeout),
            'dino_fail_open': bool(args.dino_worker_fail_open),
            'qwen_timeout_seconds': float(args.qwen_worker_timeout),
            'missing_expert_policy': args.missing_expert_policy,
        },
        'experts': {
            'relocate': 'qwen25_vl_grounder',
            'expand': 'oracle_box_refiner',
            'tighten': 'oracle_box_refiner',
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode('utf-8')
    return {
        'schema': payload['schema'],
        'sha256': hashlib.sha256(canonical).hexdigest(),
        'parameters': payload,
    }


def _dino_worker_command(args):
    proxy = argparse.Namespace(**vars(args))
    proxy.worker_timeout = args.dino_worker_timeout
    return oracle_eval._worker_command(proxy)


def _qwen_worker_command(args):
    return [
        str(Path(args.qwen_python).resolve()),
        '-u',
        '-m',
        'grounding_control.workers.qwen_grounder',
        '--model-path',
        str(Path(args.qwen_model_path).resolve()),
        '--device',
        'cuda:0',
        '--dtype',
        args.qwen_dtype,
        '--max-new-tokens',
        str(args.qwen_max_new_tokens),
        '--min-pixels',
        str(args.qwen_min_pixels),
        '--max-pixels',
        str(args.qwen_max_pixels),
        '--attn-implementation',
        args.qwen_attn_implementation,
        '--prompt-protocol',
        args.qwen_prompt_protocol,
        '--boundary-tolerance-pixels',
        str(args.qwen_boundary_tolerance_pixels),
    ]


def _warmup_reference(source):
    targets = source.get('oracle_targets') or []
    if not targets:
        raise ValueError('warm-up source has no oracle target')
    target = targets[0]
    reference = str(target.get('object') or '').strip()
    if not reference:
        aliases = target.get('aliases') or []
        reference = str(aliases[0]).strip() if aliases else ''
    if not reference:
        raise ValueError('warm-up target has no object reference')
    return reference


def _qwen_worker_warmup(client, source, args):
    image_path = (Path(args.image_dir) / source['image']).resolve()
    request = {
        'operation': 'ground',
        'image_path': str(image_path),
        'sample_id': f'{source["question_id"]}:environment_warmup',
        'grounding_step': 0,
        'object_reference': _warmup_reference(source),
    }
    response = client.request(request, timeout=args.qwen_worker_timeout)
    output = parse_grounder_output(response)
    if not output.available:
        raise RuntimeError(
            'Qwen grounder warm-up returned unavailable: '
            f'{output.error}'
        )
    if output.bbox is None:
        raise RuntimeError('Qwen grounder warm-up returned no bbox')
    return {
        'purpose': 'environment_check_only',
        'affects_routing': False,
        'source_question_id': source['question_id'],
        'request': request,
        'response': response,
        'status': 'ok',
    }


def _qwen_grounder_metrics(records):
    events = [
        event
        for record in records
        if record.get('status') == 'ok'
        for event in record['intervention']['events']
    ]
    relocate = [
        event for event in events
        if event.get('routing_decision') == 'relocate'
    ]
    invoked = [event for event in relocate if event.get('grounder_invoked')]
    unavailable = [
        event for event in relocate
        if event.get('missing_expert_error') is not None
    ]
    unavailable_reasons = Counter(
        event.get('missing_expert_error') or 'unknown'
        for event in unavailable
    )
    parse_failure_count = sum(
        bool(
            ((event.get('missing_expert_metadata') or {}).get(
                'remote_metadata'
            ) or {}).get('parse_failed')
        )
        for event in unavailable
    )
    matched_invoked = [
        event for event in invoked
        if (event.get('posthoc_oracle_audit') or {}).get('matchable')
    ]
    improvements = []
    for event in matched_invoked:
        audit = event['posthoc_oracle_audit']
        before = audit.get('candidate_iou_to_gt')
        after = audit.get('committed_iou_to_gt')
        if before is not None and after is not None:
            improvements.append(float(after) - float(before))
    return {
        'relocate_event_count': len(relocate),
        'successful_invocation_count': len(invoked),
        'unavailable_fail_open_count': len(unavailable),
        'parse_failure_count': parse_failure_count,
        'unavailable_reason_counts': dict(unavailable_reasons),
        'success_rate_on_relocate': oracle_eval._safe_ratio(
            len(invoked), len(relocate)
        ),
        'matchable_successful_invocation_count': len(matched_invoked),
        'mean_committed_minus_candidate_iou': (
            None if not improvements else sum(improvements) / len(improvements)
        ),
        'improved_iou_count': sum(value > 0 for value in improvements),
        'unchanged_iou_count': sum(value == 0 for value in improvements),
        'degraded_iou_count': sum(value < 0 for value in improvements),
    }


def _subset_summary(records):
    summary = oracle_eval._subset_summary(records)
    summary['qwen_grounder'] = _qwen_grounder_metrics(records)
    return summary


def _make_summary(
        records,
        args,
        run_id,
        dino_ping,
        dino_warmup,
        qwen_ping,
        qwen_warmup,
        experiment_signature):
    successful = [
        record for record in records if record.get('status') == 'ok'
    ]
    complete = [
        record for record in successful
        if record.get('has_complete_question_target_coverage')
    ]
    categories = sorted({
        record.get('category') for record in successful
    })
    return {
        'run_id': run_id,
        'total_records': len(records),
        'successful_records': len(successful),
        'error_records': len(records) - len(successful),
        'all_samples': _subset_summary(successful),
        'complete_target_coverage_subset': _subset_summary(complete),
        'by_category': {
            str(category): _subset_summary([
                record for record in successful
                if record.get('category') == category
            ])
            for category in categories
        },
        'workers': {
            'dino_verifier': {
                'ping': dino_ping,
                'warmup': dino_warmup,
            },
            'qwen_grounder': {
                'ping': qwen_ping,
                'warmup': qwen_warmup,
            },
        },
        'experiment_signature': experiment_signature,
        'settings': {
            'mode': EXPERIMENT_MODE,
            'baseline_results': args.baseline_results,
            'generator_cuda_visible_devices': os.environ.get(
                'CUDA_VISIBLE_DEVICES'
            ),
            'dino_physical_gpu': args.dino_gpu,
            'qwen_physical_gpu': args.qwen_gpu,
            'dino_model_path': args.dino_model_path,
            'qwen_model_path': args.qwen_model_path,
            'geometry_accept_iou': args.geometry_accept_iou,
            'geometry_containment': args.geometry_containment,
            'qwen_prompt_protocol': args.qwen_prompt_protocol,
            'qwen_candidate_box_exposed': False,
            'missing_expert_policy': args.missing_expert_policy,
            'experts': {
                'relocate': 'qwen25_vl_grounder',
                'expand': 'oracle_box_refiner',
                'tighten': 'oracle_box_refiner',
            },
            'posthoc_oracle_audit': {
                'enabled': True,
                'affects_routing': False,
                'alias_policy': ExplicitOracleTargetMatcher.POLICY,
            },
            'oracle_box_coordinate_system': (
                ORACLE_BOX_COORDINATE_SYSTEM
            ),
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'likelihood_reduction': args.likelihood_reduction,
            'kv_cache': False,
            'expert_coordinate_commit': (
                'local_roundtrip_then_single_clean_replay'
            ),
        },
    }


def _close_client(client, label):
    if client is None:
        return
    try:
        client.close()
    except Exception as error:
        print(
            f'Warning: failed to close {label} worker: '
            f'{type(error).__name__}: {error}',
            file=sys.stderr,
            flush=True,
        )


def main():
    args = parse_args()
    _validate_args(args)
    experiment_signature = _experiment_signature(args)
    setting = 'iou_' + format(
        args.geometry_accept_iou, 'g'
    ).replace('.', 'p')
    layout = create_run_layout(
        dataset='vstar',
        split=args.run_split,
        study='routing',
        method=METHOD_NAME,
        setting=setting,
        run_id=args.run_id,
        output=args.output,
        output_root=args.output_root,
    )
    layout.ensure_run_directories()
    output_path = layout.results_path
    verifier_log = (
        Path(args.verifier_log)
        if args.verifier_log
        else layout.events_path
    )
    if verifier_log.resolve() == output_path.resolve():
        raise ValueError('--verifier-log must differ from results path')

    config = {
        'command': list(sys.argv),
        'arguments': vars(args),
        'experiment_signature': experiment_signature,
        'inputs': {
            'baseline_results': str(Path(args.baseline_results).resolve()),
            'image_dir': str(Path(args.image_dir).resolve()),
        },
        'components': {
            'generator': args.model_path,
            'verifier': 'grounding_dino_geometry',
            'grounder': 'qwen25_vl',
            'box_refiner': 'oracle',
        },
        'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
        'workers': {
            'dino_verifier': {'ping': None, 'warmup': None},
            'qwen_grounder': {'ping': None, 'warmup': None},
        },
    }
    dino_client = None
    qwen_client = None
    dino_ping = None
    dino_warmup = None
    qwen_ping = None
    qwen_warmup = None
    successful_count = 0
    error_count = 0
    lifecycle_started = False

    try:
        sources = oracle_eval._selected_sources(args)
        existing = (
            []
            if args.no_resume or not output_path.exists()
            else oracle_eval._latest_records_by_question_id(
                read_jsonl(output_path)
            )
        )
        if not args.no_resume:
            oracle_eval._validate_resume_signatures(
                existing, experiment_signature
            )
        completed = {
            record['question_id']
            for record in existing
            if record.get('status') == 'ok'
        }
        pending = [
            source for source in sources
            if source['question_id'] not in completed
        ]
        pending_ids = {source['question_id'] for source in pending}
        retained = [
            record for record in existing
            if record['question_id'] not in pending_ids
        ]
        if not pending and layout.config_path.is_file():
            previous_config = json.loads(
                layout.config_path.read_text(encoding='utf-8')
            )
            if previous_config.get('experiment_signature') \
                    == experiment_signature:
                previous_workers = previous_config.get('workers') or {}
                dino_state = previous_workers.get('dino_verifier') or {}
                qwen_state = previous_workers.get('qwen_grounder') or {}
                dino_ping = dino_state.get('ping')
                dino_warmup = dino_state.get('warmup')
                qwen_ping = qwen_state.get('ping')
                qwen_warmup = qwen_state.get('warmup')
                config['workers'] = previous_workers
        successful_count = sum(
            record.get('status') == 'ok' for record in retained
        )
        error_count = len(retained) - successful_count
        oracle_eval._atomic_write_jsonl(output_path, retained)
        oracle_eval._atomic_write_jsonl(
            verifier_log,
            oracle_eval._record_events(retained),
        )

        write_run_config(layout, config)
        write_run_status(
            layout,
            'running',
            completed_records=len(completed),
            pending_records=len(pending),
            experiment_signature_sha256=experiment_signature['sha256'],
        )
        lifecycle_started = True
        print(f'Run id: {layout.run_id}; output: {output_path}', flush=True)
        print(
            'Generator CUDA_VISIBLE_DEVICES='
            f'{os.environ.get("CUDA_VISIBLE_DEVICES")!r}; '
            f'DINO GPU={args.dino_gpu!r}; Qwen GPU={args.qwen_gpu!r}',
            flush=True,
        )
        print(
            f'VStar selected={len(sources)}; pending={len(pending)}; '
            f'resumed={len(completed)}',
            flush=True,
        )

        if pending:
            dino_command = _dino_worker_command(args)
            print(
                'Starting DINO verifier worker: '
                + ' '.join(dino_command),
                flush=True,
            )
            dino_client = PersistentJsonlWorkerClient(
                dino_command,
                cwd=str(PROJECT_ROOT),
                env={'CUDA_VISIBLE_DEVICES': args.dino_gpu},
                timeout=args.dino_worker_timeout,
                stderr=None,
                start=False,
            )
            dino_client.start()
            dino_ping = dino_client.ping(timeout=30.0)
            config['workers']['dino_verifier']['ping'] = dino_ping
            write_run_config(layout, config)
            if not dino_ping.get('configured'):
                raise RuntimeError(
                    f'DINO verifier worker is not configured: {dino_ping}'
                )
            dino_proxy = argparse.Namespace(**vars(args))
            dino_proxy.worker_timeout = args.dino_worker_timeout
            dino_warmup = oracle_eval._worker_warmup(
                dino_client, pending[0], dino_proxy
            )
            dino_warmup['status'] = 'ok'
            config['workers']['dino_verifier']['warmup'] = dino_warmup
            write_run_config(layout, config)

            qwen_command = _qwen_worker_command(args)
            print(
                'Starting Qwen Grounder worker: '
                + ' '.join(qwen_command),
                flush=True,
            )
            qwen_client = PersistentJsonlWorkerClient(
                qwen_command,
                cwd=str(PROJECT_ROOT),
                env={'CUDA_VISIBLE_DEVICES': args.qwen_gpu},
                timeout=args.qwen_worker_timeout,
                stderr=None,
                start=False,
            )
            qwen_client.start()
            qwen_ping = qwen_client.ping(timeout=30.0)
            config['workers']['qwen_grounder']['ping'] = qwen_ping
            write_run_config(layout, config)
            if not qwen_ping.get('configured'):
                raise RuntimeError(
                    f'Qwen Grounder worker is not configured: {qwen_ping}'
                )
            qwen_warmup = _qwen_worker_warmup(
                qwen_client, pending[0], args
            )
            config['workers'] = {
                'dino_verifier': {
                    'ping': dino_ping,
                    'warmup': dino_warmup,
                },
                'qwen_grounder': {
                    'ping': qwen_ping,
                    'warmup': qwen_warmup,
                },
            }
            write_run_config(layout, config)
            print('Both remote workers passed real warm-up.', flush=True)

            model, preprocessor = load_model(
                args.model_path,
                precision='fp16',
            )
            verifier = RemoteActionVerifierBackend(
                dino_client,
                timeout=args.dino_worker_timeout,
                fail_open=args.dino_worker_fail_open,
            )
            grounder = RemoteGrounderBackend(
                qwen_client,
                timeout=args.qwen_worker_timeout,
                source='qwen25_vl_grounder',
            )
            resolver = OracleTargetResolver(
                preprocessor.tokenizer,
                oracle_targets_by_sample_id={
                    str(source['question_id']): source.get('oracle_targets') or []
                    for source in sources
                    if source.get('oracle_targets')
                },
                context_window_tokens=args.context_window_tokens,
            )
            box_refiner = OracleBoxRefinerBackend(resolver)
            routing_policy = RoutingPolicy(
                confidence_threshold=args.verifier_confidence_threshold,
                unsupported_action='no_action',
                unknown_action='no_action',
            )

        with output_path.open('a', encoding='utf-8') as handle:
            for source in tqdm(
                    pending,
                    desc='VStar DINO + Qwen Grounder + oracle refiner'):
                record = {
                    key: source.get(key) for key in (
                        'sample_index',
                        'question_id',
                        'image',
                        'category',
                        'question',
                        'options',
                        'label',
                        'source_jsonl_label',
                        'oracle_targets',
                        'source_oracle_boxes',
                        'oracle_box_coordinate_system',
                        'source_image_size',
                        'has_complete_question_target_coverage',
                    )
                }
                record.update({
                    'experiment_signature': experiment_signature,
                    'baseline': source['baseline'],
                    'baseline_prediction': source['baseline_prediction'],
                    'baseline_answer': source.get('baseline_answer'),
                    'baseline_prediction_correct': (
                        source['baseline_prediction'] == source['label']
                    ),
                })
                try:
                    image_path = (
                        Path(args.image_dir) / source['image']
                    ).resolve()
                    with Image.open(image_path) as opened:
                        image = opened.convert('RGB')
                    expected_size = source.get('source_image_size') or {}
                    if image.size != (
                        expected_size.get('width'),
                        expected_size.get('height'),
                    ):
                        raise ValueError(
                            f'image size {image.size} does not match '
                            f'{expected_size}'
                        )
                    conversation = make_conversation(source['question'])
                    routed = routing_infer(
                        model=model,
                        preprocessor=preprocessor,
                        image=image,
                        verifier_backend=verifier,
                        grounder_backend=grounder,
                        box_refiner_backend=box_refiner,
                        routing_policy=routing_policy,
                        missing_expert_policy=args.missing_expert_policy,
                        query=None,
                        cot=True,
                        sample_id=source['question_id'],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        conversation=conversation,
                        options=source['options'],
                        log_path=None,
                        verifier_confidence_threshold=(
                            args.verifier_confidence_threshold
                        ),
                        sample_context={
                            'image_path': str(image_path),
                            'question': source['question'],
                            'oracle_targets': source['oracle_targets'],
                        },
                    )
                    routed = dict(routed)
                    routed['events'] = (
                        oracle_eval._posthoc_oracle_audit_events(
                            routed['events'],
                            source['oracle_targets'],
                            context_window_tokens=(
                                args.context_window_tokens
                            ),
                            accept_iou_threshold=(
                                args.geometry_accept_iou
                            ),
                            containment_threshold=(
                                args.geometry_containment
                            ),
                        )
                    )
                    prediction = score_options(
                        model,
                        preprocessor,
                        image,
                        conversation,
                        source['options'],
                        routed['generated_ids'],
                        args.max_new_tokens,
                        args.temperature,
                        args.likelihood_reduction,
                    )
                    record.update({
                        'dino_geometry_qwen_grounder_router': routed,
                        'intervention': {
                            'mode': EXPERIMENT_MODE,
                            'events': routed['events'],
                        },
                        'router_prediction': prediction,
                        'router_answer': source['options'][prediction],
                        'router_prediction_correct': (
                            prediction == source['label']
                        ),
                        'status': 'ok',
                    })
                    if args.verbose:
                        actions = Counter(
                            event.get('predicted_action') or 'abstained'
                            for event in routed['events']
                        )
                        experts = Counter(
                            event.get('expert_role') or 'none'
                            for event in routed['events']
                        )
                        tqdm.write(
                            f'[{source["question_id"]}] '
                            f'actions={dict(actions)} '
                            f'experts={dict(experts)} '
                            f'pred={source["baseline_prediction"]}'
                            f'->{prediction}'
                        )
                except Exception as error:
                    record.update({
                        'status': 'error',
                        'error': f'{type(error).__name__}: {error}',
                    })
                    if args.verbose:
                        tqdm.write(
                            f'[{source["question_id"]}] '
                            f'ERROR: {record["error"]}'
                        )
                handle.write(json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + '\n')
                handle.flush()
                if record.get('status') == 'ok':
                    oracle_eval._append_events(
                        verifier_log,
                        record['intervention']['events'],
                    )
                    successful_count += 1
                else:
                    error_count += 1
                    if args.fail_fast:
                        raise RuntimeError(
                            'sample failed under --fail-fast: '
                            f'{source["question_id"]}: {record["error"]}'
                        )

        records = oracle_eval._latest_records_by_question_id(
            read_jsonl(output_path)
        )
        oracle_eval._atomic_write_jsonl(output_path, records)
        oracle_eval._atomic_write_jsonl(
            verifier_log,
            oracle_eval._record_events(records),
        )
        summary = _make_summary(
            records,
            args,
            layout.run_id,
            dino_ping,
            dino_warmup,
            qwen_ping,
            qwen_warmup,
            experiment_signature,
        )
        layout.summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + '\n',
            encoding='utf-8',
        )
        write_run_status(
            layout,
            'completed' if summary['error_records'] == 0
            else 'completed_with_errors',
            completed_records=summary['successful_records'],
            error_records=summary['error_records'],
            summary_path=str(layout.summary_path),
            experiment_signature_sha256=experiment_signature['sha256'],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'Verifier events: {verifier_log}')
        print(f'Per-example results: {output_path}')
        print(f'Summary: {layout.summary_path}')
    except KeyboardInterrupt:
        if lifecycle_started:
            write_run_status(
                layout,
                'interrupted',
                completed_records=successful_count,
                error_records=error_count,
                experiment_signature_sha256=(
                    experiment_signature['sha256']
                ),
            )
        raise
    except BaseException as error:
        if lifecycle_started:
            write_run_status(
                layout,
                'failed',
                completed_records=successful_count,
                error_records=error_count,
                error=f'{type(error).__name__}: {error}',
                experiment_signature_sha256=(
                    experiment_signature['sha256']
                ),
            )
        raise
    finally:
        _close_client(qwen_client, 'Qwen Grounder')
        _close_client(dino_client, 'DINO verifier')


if __name__ == '__main__':
    main()
