"""Evaluate an oracle binary verifier with a real Qwen7B Grounder on VStar.

Every natural VoCoT coordinate is intercepted before REFbind.  References
that uniquely match an annotated VStar target receive a hard oracle alignment
label based on candidate-to-GT IoU.  Rejected candidates are relocated by a
persistent Qwen2.5-VL Grounder worker; accepted and unverifiable candidates
are committed unchanged.  The selected coordinate then follows normal
Volcano REFbind and all later CoT tokens remain freely generated.

This experiment isolates the correction ability of the Grounder under a
perfect verifier.  GT boxes are owned only by the evaluator-side verifier and
are never included in the Grounder's request.
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

from model.load_model import load_model, routing_infer  # noqa: E402
from grounding_control.core import AlignmentRoutingPolicy  # noqa: E402
from grounding_control.experts.grounders import (  # noqa: E402
    RemoteGrounderBackend,
)
from grounding_control.oracle_targets import OracleTargetResolver  # noqa: E402
from grounding_control.run_paths import (  # noqa: E402
    create_run_layout,
    write_run_config,
    write_run_status,
)
from grounding_control.transport import (  # noqa: E402
    PersistentJsonlWorkerClient,
    parse_grounder_output,
)
from grounding_control.verifiers import (  # noqa: E402
    OracleAlignmentVerifierBackend,
)
from eval.grounding_control.vstar.routing_support import (  # noqa: E402
    ORACLE_BOX_COORDINATE_SYSTEM,
    append_events,
    atomic_write_jsonl,
    latest_records_by_question_id,
    make_conversation,
    paired_metrics,
    read_jsonl,
    record_events,
    score_options,
)


DEFAULT_QWEN_PYTHON = '/home/zhonggai/miniconda3/envs/qwen25/bin/python'
DEFAULT_QWEN_MODEL = '/data/zhonggai/models/Qwen2.5-VL-7B-Instruct'
DEFAULT_BASELINE_RESULTS = (
    'output/vstar/online_oracle/full_238_padding_fix/results.jsonl'
)
DEFAULT_IMAGE_DIR = '/data/zhonggai/VStar'
EXPECTED_GROUNDER_SOURCE = 'qwen25_vl_grounder'
EXPERIMENT_SIGNATURE_SCHEMA = (
    'vstar_oracle_verifier_qwen_grounder_signature_v1'
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='weights/Volcano-7b')
    parser.add_argument(
        '--baseline-results', default=DEFAULT_BASELINE_RESULTS,
        help='Padding-fixed VStar run providing baseline CoTs and GT targets.',
    )
    parser.add_argument('--image-dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output', default=None)
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-split', default='full_238')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--verifier-log', default=None)

    parser.add_argument('--qwen-python', default=DEFAULT_QWEN_PYTHON)
    parser.add_argument('--qwen-model-path', default=DEFAULT_QWEN_MODEL)
    parser.add_argument('--qwen-gpu', default='7')
    parser.add_argument('--qwen-dtype', default='bfloat16')
    parser.add_argument('--qwen-max-new-tokens', type=int, default=64)
    parser.add_argument('--qwen-min-pixels', type=int, default=3136)
    parser.add_argument('--qwen-max-pixels', type=int, default=12_000_000)
    parser.add_argument(
        '--qwen-attn-implementation', default='flash_attention_2'
    )
    parser.add_argument(
        '--qwen-prompt-protocol',
        choices=('compact_json_v1', 'single_object_json_v2'),
        default='compact_json_v1',
    )
    parser.add_argument(
        '--qwen-boundary-tolerance-pixels', type=float, default=1.0
    )
    parser.add_argument('--worker-timeout', type=float, default=600.0)

    parser.add_argument(
        '--oracle-iou-threshold', type=float, default=0.5,
        help='A uniquely matched natural candidate below this IoU is rejected.',
    )
    parser.add_argument('--reject-threshold', type=float, default=0.25)
    parser.add_argument('--accept-threshold', type=float, default=0.75)
    parser.add_argument('--context-window-tokens', type=int, default=48)
    parser.add_argument(
        '--missing-expert-policy',
        choices=('fail_open', 'error'),
        default='fail_open',
    )

    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument(
        '--likelihood-reduction', choices=('mean', 'sum'), default='mean'
    )
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--sample-id', default=None)
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def _validate_args(args):
    for name, value in (
        ('oracle-iou-threshold', args.oracle_iou_threshold),
        ('reject-threshold', args.reject_threshold),
        ('accept-threshold', args.accept_threshold),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f'--{name} must be in [0, 1]')
    if not float(args.reject_threshold) < float(args.accept_threshold):
        raise ValueError('--reject-threshold must be below --accept-threshold')
    if args.context_window_tokens <= 0:
        raise ValueError('--context-window-tokens must be positive')
    if args.max_new_tokens <= 0 or args.qwen_max_new_tokens <= 0:
        raise ValueError('generation token limits must be positive')
    if args.qwen_min_pixels <= 0 or args.qwen_max_pixels <= 0:
        raise ValueError('Qwen pixel limits must be positive')
    if args.qwen_min_pixels > args.qwen_max_pixels:
        raise ValueError('Qwen min pixels must not exceed max pixels')
    if args.qwen_boundary_tolerance_pixels < 0:
        raise ValueError('Qwen boundary tolerance must be non-negative')
    if args.worker_timeout <= 0:
        raise ValueError('--worker-timeout must be positive')
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    if args.sample_id is not None and (
            args.start_index != 0 or args.max_samples is not None):
        raise ValueError(
            '--sample-id cannot be combined with start/max sample selection'
        )
    if not str(args.qwen_gpu).isdigit():
        raise ValueError('--qwen-gpu must be one physical GPU index')
    for value, label, kind in (
        (args.qwen_python, 'Qwen Python', 'file'),
        (args.qwen_model_path, 'Qwen model', 'dir'),
        (args.model_path, 'VoCoT model', 'exists'),
        (args.baseline_results, 'baseline results', 'file'),
        (args.image_dir, 'image directory', 'dir'),
    ):
        path = Path(value)
        valid = (
            path.is_file() if kind == 'file'
            else path.is_dir() if kind == 'dir'
            else path.exists()
        )
        if not valid:
            raise FileNotFoundError(f'{label} not found: {path}')


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_signature(args):
    baseline_path = Path(args.baseline_results).resolve()
    payload = {
        'schema': EXPERIMENT_SIGNATURE_SCHEMA,
        'dataset': {
            'name': 'vstar',
            'split': args.run_split,
            'image_dir': str(Path(args.image_dir).resolve()),
        },
        'baseline': {
            'results_path': str(baseline_path),
            'sha256': _sha256_file(baseline_path),
        },
        'generator': {
            'model_path': str(Path(args.model_path).resolve()),
            'precision': 'fp16',
            'cot': True,
        },
        'verifier': {
            'backend': 'oracle_binary_iou_label',
            'gt_iou_threshold': float(args.oracle_iou_threshold),
            'unmatched_policy': 'abstain_then_controller_fail_open',
            'context_window_tokens': int(args.context_window_tokens),
        },
        'routing_policy': {
            'score_kind': 'hard_oracle_label',
            'reject_threshold': float(args.reject_threshold),
            'accept_threshold': float(args.accept_threshold),
            'uncertain_action': 'call_grounder',
            'verifier_failure_action': 'accept_candidate',
            'missing_expert_policy': args.missing_expert_policy,
        },
        'grounder': {
            'backend': EXPECTED_GROUNDER_SOURCE,
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
        },
        'generation': {
            'max_new_tokens': int(args.max_new_tokens),
            'temperature': float(args.temperature),
        },
        'scoring': {
            'method': 'option_conditional_likelihood',
            'likelihood_reduction': args.likelihood_reduction,
            'further_instruct': True,
        },
        'runtime': {
            'verify_every_coordinate': True,
            'cross_call_kv_cache': False,
            'expert_coordinate_commit': (
                'local_roundtrip_then_single_clean_replay'
            ),
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
        'schema': EXPERIMENT_SIGNATURE_SCHEMA,
        'sha256': hashlib.sha256(canonical).hexdigest(),
        'parameters': payload,
    }


def _validate_resume_signatures(records, signature):
    mismatches = [
        record.get('question_id')
        for record in records
        if record.get('experiment_signature') != signature
    ]
    if mismatches:
        preview = ', '.join(map(str, mismatches[:5]))
        raise ValueError(
            'existing results use a different experiment signature '
            f'({preview}); select a new --run-id or use --no-resume'
        )


def _selected_sources(args):
    records = [
        record for record in read_jsonl(args.baseline_results)
        if record.get('status') == 'ok' and record.get('baseline')
    ]
    incompatible = [
        record.get('question_id') for record in records
        if record.get('oracle_box_coordinate_system')
        != ORACLE_BOX_COORDINATE_SYSTEM
    ]
    if incompatible:
        raise ValueError(
            'baseline results contain pre-padding-fix oracle boxes; '
            f'first incompatible sample: {incompatible[0]}'
        )
    if args.sample_id is not None:
        selected = [
            record for record in records
            if record.get('question_id') == args.sample_id
        ]
        if len(selected) != 1:
            raise ValueError(
                f'expected one --sample-id match, found {len(selected)}'
            )
        return selected
    end = len(records) if args.max_samples is None else min(
        len(records), args.start_index + args.max_samples
    )
    return records[args.start_index:end]


def _worker_command(args):
    return [
        str(Path(args.qwen_python).resolve()),
        '-u',
        '-m',
        'grounding_control.workers.qwen_grounder',
        '--model-path', str(Path(args.qwen_model_path).resolve()),
        '--device', 'cuda:0',
        '--dtype', args.qwen_dtype,
        '--max-new-tokens', str(args.qwen_max_new_tokens),
        '--min-pixels', str(args.qwen_min_pixels),
        '--max-pixels', str(args.qwen_max_pixels),
        '--attn-implementation', args.qwen_attn_implementation,
        '--prompt-protocol', args.qwen_prompt_protocol,
        '--boundary-tolerance-pixels', str(
            args.qwen_boundary_tolerance_pixels
        ),
    ]


def _worker_warmup(client, source, args):
    """Run one real Grounder forward without affecting evaluation metrics."""
    targets = source.get('oracle_targets') or []
    if not targets:
        raise ValueError('warm-up sample has no oracle target')
    target = targets[0]
    reference = str(target.get('object') or '').strip()
    if not reference:
        aliases = target.get('aliases') or []
        reference = str(aliases[0]).strip() if aliases else ''
    if not reference:
        raise ValueError('warm-up target has no object reference')
    image_path = (Path(args.image_dir) / source['image']).resolve()
    request = {
        'operation': 'ground',
        'image_path': str(image_path),
        'sample_id': f'{source["question_id"]}:environment_warmup',
        'grounding_step': 0,
        'object_reference': reference,
    }
    response = client.request(request, timeout=args.worker_timeout)
    output = parse_grounder_output(response)
    if not output.available:
        raise RuntimeError(f'Qwen Grounder warm-up unavailable: {output.error}')
    return {
        'purpose': 'environment_check_only',
        'affects_routing': False,
        'source_question_id': source['question_id'],
        'request': request,
        'response': output.as_dict(),
    }


def _mean(values):
    return None if not values else sum(values) / len(values)


def _recall(values, threshold):
    return None if not values else sum(
        value >= threshold for value in values
    ) / len(values)


def routing_metrics(records):
    events = [
        event
        for record in records
        for event in record['intervention']['events']
    ]
    matched = [
        event for event in events
        if event.get('match_status') == 'matched_unique_explicit_target'
    ]
    requested = [event for event in events if event.get('grounder_requested')]
    succeeded = [event for event in events if event.get('grounder_succeeded')]
    candidate_ious = [
        float(event['candidate_iou_to_gt'])
        for event in matched
        if event.get('candidate_iou_to_gt') is not None
    ]
    committed_ious = [
        float(event['committed_iou_to_gt'])
        for event in matched
        if event.get('committed_iou_to_gt') is not None
    ]
    succeeded_iou_pairs = [
        (
            float(event['candidate_iou_to_gt']),
            float(event['committed_iou_to_gt']),
        )
        for event in succeeded
        if event.get('candidate_iou_to_gt') is not None
        and event.get('committed_iou_to_gt') is not None
    ]
    succeeded_candidate_ious = [
        candidate for candidate, _ in succeeded_iou_pairs
    ]
    succeeded_committed_ious = [
        committed for _, committed in succeeded_iou_pairs
    ]
    deltas = [
        after - before
        for before, after in zip(
            succeeded_candidate_ious,
            succeeded_committed_ious,
        )
    ]
    decision_counts = Counter(
        event.get('decision_band') or 'unknown' for event in events
    )
    first_grounder_positions = Counter()
    for record in records:
        first = next((
            event['grounding_step']
            for event in record['intervention']['events']
            if event.get('grounder_succeeded')
        ), None)
        if first is not None:
            first_grounder_positions[int(first)] += 1
    return {
        'coordinate_event_count': len(events),
        'matchable_coordinate_count': len(matched),
        'unverifiable_fail_open_count': sum(
            bool(event.get('verifier_abstained')) for event in events
        ),
        'decision_band_counts': dict(decision_counts),
        'grounder_requested_count': len(requested),
        'grounder_attempted_count': sum(
            bool(event.get('grounder_attempted')) for event in events
        ),
        'grounder_succeeded_count': len(succeeded),
        'grounder_unavailable_fail_open_count': sum(
            event.get('missing_expert_error') is not None for event in events
        ),
        'grounder_call_rate_per_coordinate': (
            None if not events else len(requested) / len(events)
        ),
        'samples_with_grounder_request': sum(
            any(
                event.get('grounder_requested')
                for event in record['intervention']['events']
            ) for record in records
        ),
        'samples_with_grounder_success': sum(
            any(
                event.get('grounder_succeeded')
                for event in record['intervention']['events']
            ) for record in records
        ),
        'candidate_miou_on_matchable_coordinates': _mean(candidate_ious),
        'committed_miou_on_matchable_coordinates': _mean(committed_ious),
        'candidate_miou_on_successful_grounder_calls': _mean(
            succeeded_candidate_ious
        ),
        'grounder_committed_miou': _mean(succeeded_committed_ious),
        'grounder_committed_recall': {
            str(threshold): _recall(succeeded_committed_ious, threshold)
            for threshold in (0.1, 0.3, 0.5, 0.7)
        },
        'mean_iou_gain_on_successful_grounder_calls': _mean(deltas),
        'grounder_improved_iou_count': sum(delta > 0 for delta in deltas),
        'grounder_unchanged_iou_count': sum(delta == 0 for delta in deltas),
        'grounder_degraded_iou_count': sum(delta < 0 for delta in deltas),
        'candidate_lt_0p5_to_committed_ge_0p5_count': sum(
            before < 0.5 <= after
            for before, after in zip(
                succeeded_candidate_ious,
                succeeded_committed_ious,
            )
        ),
        'first_successful_grounder_position_counts': {
            str(position): count
            for position, count in sorted(first_grounder_positions.items())
        },
    }


def _subset_summary(records):
    summary = paired_metrics(records)
    summary['routing'] = routing_metrics(records)
    return summary


def _make_summary(
        records, args, run_id, worker_ping, worker_warmup,
        experiment_signature):
    successful = [record for record in records if record.get('status') == 'ok']
    complete = [
        record for record in successful
        if record.get('has_complete_question_target_coverage')
    ]
    categories = sorted({record.get('category') for record in successful})
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
        'worker_ping': worker_ping,
        'worker_warmup': worker_warmup,
        'experiment_signature': experiment_signature,
        'settings': {
            'mode': 'oracle_binary_verifier_qwen25_vl_7b_grounder',
            'generator_cuda_visible_devices': os.environ.get(
                'CUDA_VISIBLE_DEVICES'
            ),
            'qwen_physical_gpu': args.qwen_gpu,
            'oracle_iou_threshold': args.oracle_iou_threshold,
            'reject_threshold': args.reject_threshold,
            'accept_threshold': args.accept_threshold,
            'unmatched_reference_policy': 'verifier_abstain_fail_open',
            'missing_expert_policy': args.missing_expert_policy,
            'qwen_model_path': args.qwen_model_path,
            'qwen_max_pixels': args.qwen_max_pixels,
            'qwen_attn_implementation': args.qwen_attn_implementation,
            'qwen_prompt_protocol': args.qwen_prompt_protocol,
            'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens,
            'likelihood_reduction': args.likelihood_reduction,
            'oracle_box_coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
            'verify_every_coordinate': True,
            'kv_cache': False,
        },
    }


def main():
    args = parse_args()
    _validate_args(args)
    signature = _experiment_signature(args)
    setting = (
        'gt_iou_' + format(args.oracle_iou_threshold, 'g').replace('.', 'p')
        + '__' + str(args.qwen_max_pixels // 1_000_000) + 'mp'
    )
    layout = create_run_layout(
        dataset='vstar',
        split=args.run_split,
        study='routing',
        method='oracle_verifier__qwen25_vl_7b_grounder',
        setting=setting,
        run_id=args.run_id,
        output=args.output,
        output_root=args.output_root,
    )
    layout.ensure_run_directories()
    output_path = layout.results_path
    verifier_log = (
        Path(args.verifier_log) if args.verifier_log else layout.events_path
    )
    if verifier_log.resolve() == output_path.resolve():
        raise ValueError('--verifier-log must differ from results output')

    worker_command = _worker_command(args)
    config = {
        'command': list(sys.argv),
        'arguments': vars(args),
        'experiment_signature': signature,
        'inputs': {
            'baseline_results': str(Path(args.baseline_results).resolve()),
            'image_dir': str(Path(args.image_dir).resolve()),
        },
        'components': {
            'generator': args.model_path,
            'verifier': 'oracle_binary_iou_label',
            'routing_policy': 'binary_dual_threshold_grounder_v1',
            'grounder': EXPECTED_GROUNDER_SOURCE,
        },
        'worker': {
            'command': worker_command,
            'physical_gpu': args.qwen_gpu,
            'ping': None,
            'warmup': None,
        },
        'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
    }
    client = None
    successful_count = 0
    error_count = 0
    lifecycle_started = False
    worker_ping = None
    worker_warmup = None

    try:
        sources = _selected_sources(args)
        existing = (
            [] if args.no_resume or not output_path.exists()
            else latest_records_by_question_id(read_jsonl(output_path))
        )
        if existing:
            _validate_resume_signatures(existing, signature)
        completed = {
            record['question_id']
            for record in existing if record.get('status') == 'ok'
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
        successful_count = sum(
            record.get('status') == 'ok' for record in retained
        )
        error_count = len(retained) - successful_count
        atomic_write_jsonl(output_path, retained)
        atomic_write_jsonl(verifier_log, record_events(retained))

        if not pending and layout.config_path.is_file():
            previous = json.loads(
                layout.config_path.read_text(encoding='utf-8')
            )
            if previous.get('experiment_signature') == signature:
                config['worker'] = previous.get('worker', config['worker'])
                worker_ping = config['worker'].get('ping')
                worker_warmup = config['worker'].get('warmup')

        write_run_config(layout, config)
        write_run_status(
            layout,
            'running',
            completed_records=len(completed),
            pending_records=len(pending),
            experiment_signature_sha256=signature['sha256'],
        )
        lifecycle_started = True
        print(f'Run id: {layout.run_id}; output: {output_path}', flush=True)
        print(
            f'Generator CUDA_VISIBLE_DEVICES='
            f'{os.environ.get("CUDA_VISIBLE_DEVICES")!r}; '
            f'Qwen Grounder physical GPU={args.qwen_gpu!r}',
            flush=True,
        )
        print(
            f'VStar selected={len(sources)}; pending={len(pending)}; '
            f'resumed={len(completed)}',
            flush=True,
        )

        if pending:
            print(
                'Starting Qwen Grounder worker: '
                + ' '.join(worker_command),
                flush=True,
            )
            client = PersistentJsonlWorkerClient(
                worker_command,
                cwd=str(PROJECT_ROOT),
                env={'CUDA_VISIBLE_DEVICES': str(args.qwen_gpu)},
                timeout=args.worker_timeout,
                stderr=None,
                start=False,
            )
            client.start()
            worker_ping = client.ping(timeout=30.0)
            config['worker']['ping'] = worker_ping
            write_run_config(layout, config)
            if not worker_ping.get('configured'):
                raise RuntimeError(
                    f'Qwen Grounder worker is not configured: {worker_ping}'
                )
            worker_warmup = _worker_warmup(client, pending[0], args)
            worker_warmup['status'] = 'ok'
            config['worker']['warmup'] = worker_warmup
            write_run_config(layout, config)
            print(
                'Qwen Grounder ready after real warm-up: '
                + json.dumps({
                    'worker': worker_ping.get('worker'),
                    'max_pixels': worker_ping.get('max_pixels'),
                    'warmup_available': worker_warmup['response']['available'],
                }, ensure_ascii=False),
                flush=True,
            )

            model, preprocessor = load_model(
                args.model_path,
                precision='fp16',
            )
            resolver = OracleTargetResolver(
                preprocessor.tokenizer,
                oracle_targets_by_sample_id={
                    str(source['question_id']): (
                        source.get('oracle_targets') or []
                    )
                    for source in sources
                    if source.get('oracle_targets')
                },
                context_window_tokens=args.context_window_tokens,
            )
            verifier = OracleAlignmentVerifierBackend(
                resolver,
                gt_iou_threshold=args.oracle_iou_threshold,
            )
            grounder = RemoteGrounderBackend(
                client,
                timeout=args.worker_timeout,
                source=EXPECTED_GROUNDER_SOURCE,
            )
            policy = AlignmentRoutingPolicy.explicit_raw(
                reject_threshold=args.reject_threshold,
                accept_threshold=args.accept_threshold,
                score_kind='hard_oracle_label',
            )

        with output_path.open('a', encoding='utf-8') as handle:
            for source in tqdm(
                    pending,
                    desc='VStar oracle verifier + Qwen7B Grounder'):
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
                    'experiment_signature': signature,
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
                        alignment_routing_policy=policy,
                        query=None,
                        cot=True,
                        sample_id=source['question_id'],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        conversation=conversation,
                        options=source['options'],
                        log_path=None,
                        missing_expert_policy=args.missing_expert_policy,
                        sample_context={
                            'image_path': str(image_path),
                        },
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
                        'oracle_verifier_qwen_grounder': routed,
                        'intervention': {
                            'mode': (
                                'oracle_binary_verifier_'
                                'qwen25_vl_7b_grounder'
                            ),
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
                        bands = Counter(
                            event.get('decision_band') or 'unknown'
                            for event in routed['events']
                        )
                        tqdm.write(
                            f'[{source["question_id"]}] '
                            f'bands={dict(bands)} '
                            f'grounder={sum(bool(event.get("grounder_succeeded")) for event in routed["events"])} '
                            f'pred={source["baseline_prediction"]}->{prediction}'
                        )
                except Exception as error:
                    record.update({
                        'status': 'error',
                        'error': f'{type(error).__name__}: {error}',
                    })
                    if args.verbose:
                        tqdm.write(
                            f'[{source["question_id"]}] ERROR: '
                            f'{record["error"]}'
                        )
                handle.write(json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + '\n')
                handle.flush()
                if record.get('status') == 'ok':
                    append_events(
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

        records = latest_records_by_question_id(read_jsonl(output_path))
        atomic_write_jsonl(output_path, records)
        atomic_write_jsonl(verifier_log, record_events(records))
        summary = _make_summary(
            records,
            args,
            layout.run_id,
            worker_ping,
            worker_warmup,
            signature,
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
            experiment_signature_sha256=signature['sha256'],
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
                experiment_signature_sha256=signature['sha256'],
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
                experiment_signature_sha256=signature['sha256'],
            )
        raise
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as close_error:
                print(
                    'Warning: failed to close Qwen worker: '
                    f'{type(close_error).__name__}: {close_error}',
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == '__main__':
    main()
