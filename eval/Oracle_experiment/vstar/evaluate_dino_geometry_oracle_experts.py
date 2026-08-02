"""Evaluate remote DINO geometry verification with oracle correction experts.

The VoCoT generator runs in this process. A persistent Grounding DINO worker
runs in a separate process/GPU and judges every pre-commit coordinate as one
of no_action/relocate/expand/tighten. Relocate is routed to an oracle
Grounder; expand/tighten are routed to an oracle BoxRefiner. Both experts use
the same conservative explicit-alias resolver and fail open when no unique GT
target can be resolved.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.load_model import load_model, routing_infer
from utils.coordinate_intervention import (
    ExplicitOracleTargetMatcher,
    box_iou,
    normalize_object_reference,
)
from grounding_control.contracts import validate_normalized_box
from grounding_control.four_way import (
    ACTION_NAMES,
    ActionVerifierOutput,
    OracleBoxRefinerBackend,
    RemoteActionVerifierBackend,
    RoutingPolicy,
)
from grounding_control.experts.grounders import OracleGrounderBackend
from grounding_control.oracle_targets import OracleTargetResolver
from grounding_control.run_paths import (
    create_run_layout,
    write_run_config,
    write_run_status,
)
from grounding_control.transport import PersistentJsonlWorkerClient
from grounding_control.four_way.verifiers.geometry import (
    route_from_grounding_geometry,
)

# Reuse the established VStar prompt, option-likelihood scoring, and paired
# statistics rather than creating a second evaluation definition.
from eval.Oracle_experiment.vstar.evaluate_selective_oracle_router import (
    make_conversation,
    paired_metrics,
    read_jsonl,
    score_options,
)


ORACLE_BOX_COORDINATE_SYSTEM = (
    'normalized_xyxy_on_center_padded_square'
)
DEFAULT_DINO_PYTHON = (
    '/home/zhonggai/miniconda3/envs/qwen25/bin/python'
)
DEFAULT_DINO_MODEL = '/data/zhonggai/models/grounding-dino-base'


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
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Legacy output filename. When omitted, use the canonical '
            'output/vstar/runs/<run-split>/routing/... layout.'
        ),
    )
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--run-split', default='full_238')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--verifier-log', default=None)

    parser.add_argument('--dino-python', default=DEFAULT_DINO_PYTHON)
    parser.add_argument('--dino-model-path', default=DEFAULT_DINO_MODEL)
    parser.add_argument(
        '--dino-gpu',
        default='7',
        help=(
            'Physical GPU exposed only to the worker. The worker itself uses '
            'logical cuda:0.'
        ),
    )
    parser.add_argument('--dino-dtype', default='float32')
    parser.add_argument('--dino-box-threshold', type=float, default=0.3)
    parser.add_argument('--dino-text-threshold', type=float, default=0.25)
    parser.add_argument(
        '--geometry-accept-iou',
        type=float,
        default=0.4,
    )
    parser.add_argument(
        '--geometry-containment',
        type=float,
        default=0.7,
    )
    parser.add_argument('--dino-top-k-log', type=int, default=20)
    parser.add_argument('--worker-timeout', type=float, default=300.0)
    parser.add_argument(
        '--worker-fail-open',
        action='store_true',
        help='Turn worker failures into verifier abstentions instead of errors.',
    )

    parser.add_argument(
        '--verifier-confidence-threshold',
        type=float,
        default=0.0,
        help=(
            'DINO detector scores are not calibrated action probabilities; '
            '0.0 measures the raw geometry-routing upper bound.'
        ),
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
    if args.worker_timeout <= 0:
        raise ValueError('--worker-timeout must be positive')
    if args.context_window_tokens <= 0:
        raise ValueError('--context-window-tokens must be positive')
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError('--max-samples must be positive')
    if args.dino_top_k_log <= 0:
        raise ValueError('--dino-top-k-log must be positive')
    if not Path(args.dino_python).is_file():
        raise FileNotFoundError(args.dino_python)
    if not Path(args.dino_model_path).is_dir():
        raise FileNotFoundError(args.dino_model_path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _experiment_signature(args):
    """Return a stable semantic experiment identity for strict resume."""

    baseline_path = Path(args.baseline_results).resolve()
    payload = {
        'schema': 'vstar_dino_oracle_experts_signature_v1',
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
            'sha256': _sha256_file(baseline_path),
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
            'alias_policy': ExplicitOracleTargetMatcher.POLICY,
        },
        'worker_policy': {
            'timeout_seconds': float(args.worker_timeout),
            'fail_open': bool(args.worker_fail_open),
            'missing_expert_policy': args.missing_expert_policy,
        },
        'experts': {
            'relocate': 'oracle_grounder',
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


def _latest_records_by_question_id(records):
    """Keep the last JSONL record for every question without reordering."""

    latest = {}
    for record in records:
        question_id = record.get('question_id')
        if not isinstance(question_id, str) or not question_id:
            raise ValueError('existing result record lacks question_id')
        latest[question_id] = record
    return list(latest.values())


def _validate_resume_signatures(records, signature):
    mismatches = [
        record.get('question_id')
        for record in records
        if record.get('experiment_signature') != signature
    ]
    if mismatches:
        preview = ', '.join(map(str, mismatches[:5]))
        raise ValueError(
            'existing results use a missing or different experiment '
            f'signature ({preview}); choose a new --run-id or use '
            '--no-resume'
        )


def _atomic_write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=str(path.parent),
        prefix='.' + path.name + '.',
        suffix='.tmp',
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            for record in records:
                handle.write(json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _record_events(records):
    return [
        event
        for record in records
        if record.get('status') == 'ok'
        for event in (record.get('intervention') or {}).get('events', [])
    ]


def _append_events(path, events):
    if not events:
        return
    with Path(path).open('a', encoding='utf-8') as handle:
        for event in events:
            handle.write(json.dumps(
                event,
                ensure_ascii=False,
                allow_nan=False,
            ) + '\n')
        handle.flush()


def _worker_warmup(client, source, args):
    """Force one real DINO forward without affecting model routing."""

    targets = source.get('oracle_targets') or []
    if not targets:
        raise ValueError('warm-up source has no oracle target')
    target = targets[0]
    reference = str(target.get('object') or '').strip()
    if not reference:
        aliases = target.get('aliases') or []
        reference = str(aliases[0]).strip() if aliases else ''
    if not reference:
        raise ValueError('warm-up oracle target has no object reference')
    candidate_box = validate_normalized_box(target['box'])
    image_path = (
        Path(args.image_dir) / source['image']
    ).resolve()
    request = {
        'operation': 'verify',
        'image_path': str(image_path),
        'sample_id': f'{source["question_id"]}:environment_warmup',
        'grounding_step': 0,
        'object_reference': reference,
        'candidate_bbox': list(candidate_box),
        'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
    }
    response = client.request(request, timeout=args.worker_timeout)
    output = ActionVerifierOutput.from_dict(response)
    return {
        'purpose': 'environment_check_only',
        'affects_routing': False,
        'source_question_id': source['question_id'],
        'request': request,
        'response': response,
        'validated_output': output.as_dict(),
    }


def _worker_command(args):
    return [
        str(Path(args.dino_python).resolve()),
        '-u',
        '-m',
        'grounding_control.four_way.workers.dino_geometry_verifier',
        '--model-path',
        str(Path(args.dino_model_path).resolve()),
        '--device',
        'cuda:0',
        '--dtype',
        args.dino_dtype,
        '--box-threshold',
        str(args.dino_box_threshold),
        '--text-threshold',
        str(args.dino_text_threshold),
        '--accept-iou-threshold',
        str(args.geometry_accept_iou),
        '--containment-threshold',
        str(args.geometry_containment),
        '--top-k-log',
        str(args.dino_top_k_log),
    ]


def _selected_sources(args):
    records = [
        record for record in read_jsonl(args.baseline_results)
        if record.get('status') == 'ok' and record.get('baseline')
    ]
    incompatible = [
        record for record in records
        if record.get('oracle_box_coordinate_system')
        != ORACLE_BOX_COORDINATE_SYSTEM
    ]
    if incompatible:
        raise ValueError(
            '--baseline-results contains old/unknown oracle coordinates; '
            'use the padding-fixed run'
        )
    if args.sample_id is not None:
        selected = [
            record for record in records
            if record.get('question_id') == args.sample_id
        ]
        if len(selected) != 1:
            raise ValueError(
                f'expected one record for --sample-id {args.sample_id!r}, '
                f'found {len(selected)}'
            )
        return selected
    end = len(records) if args.max_samples is None else min(
        len(records),
        args.start_index + args.max_samples,
    )
    return records[args.start_index:end]


def _posthoc_oracle_audit_event(
        event,
        matcher,
        *,
        context_window_tokens,
        accept_iou_threshold,
        containment_threshold):
    """Attach read-only GT diagnostics to one completed routing event.

    This function runs only after the controller has selected and committed a
    coordinate.  Its output is never passed back to the verifier, router, or
    expert, so the oracle annotation cannot influence the generated trajectory.
    """

    audited_event = dict(event)
    reference = str(event.get('object_reference') or '')
    context_tokens = normalize_object_reference(reference)[
        -int(context_window_tokens):
    ]
    matched, match_reason = matcher.match(context_tokens)
    predicted_action = event.get('predicted_action')
    audit = {
        'schema': 'vstar_posthoc_oracle_event_v1',
        'affects_routing': False,
        'matchable': matched is not None,
        'match_reason': str(match_reason),
        'match_policy': ExplicitOracleTargetMatcher.POLICY,
        'context_normalized_tokens': list(context_tokens),
        'target_object': None,
        'matched_alias': None,
        'oracle_target_box': None,
        'candidate_iou_to_gt': None,
        'committed_iou_to_gt': None,
        'dino_iou_to_gt': None,
        'dino_box_available': False,
        'oracle_geometry_action': None,
        'predicted_action': predicted_action,
        'predicted_action_correct': None,
        'oracle_should_route': None,
        'predicted_should_route': (
            predicted_action in {'relocate', 'expand', 'tighten'}
        ),
        'binary_route_correct': None,
    }
    if matched is None:
        audited_event['posthoc_oracle_audit'] = audit
        return audited_event

    gt_box = validate_normalized_box(matched['box'])
    candidate_box = validate_normalized_box(event['candidate_box'])
    committed_box = validate_normalized_box(event['committed_box'])
    oracle_geometry = route_from_grounding_geometry(
        candidate_box,
        gt_box,
        accept_iou_threshold=float(accept_iou_threshold),
        containment_threshold=float(containment_threshold),
    )
    oracle_action = oracle_geometry.action
    oracle_should_route = oracle_action != 'no_action'
    predicted_should_route = bool(audit['predicted_should_route'])

    verifier_metadata = event.get('verifier_metadata') or {}
    dino_box_value = verifier_metadata.get(
        'selected_grounding_padded_normalized_bbox_xyxy'
    )
    dino_box = None
    dino_box_error = None
    if dino_box_value is not None:
        try:
            dino_box = validate_normalized_box(dino_box_value)
        except (TypeError, ValueError) as error:
            # A malformed diagnostic must remain visible, but it must not
            # change the already completed routing trajectory.
            dino_box_error = f'{type(error).__name__}: {error}'

    audit.update({
        'target_object': str(matched['object']),
        'matched_alias': ' '.join(matched['alias_tokens']),
        'oracle_target_box': list(gt_box),
        'candidate_iou_to_gt': box_iou(candidate_box, gt_box),
        'committed_iou_to_gt': box_iou(committed_box, gt_box),
        'dino_iou_to_gt': (
            None if dino_box is None else box_iou(dino_box, gt_box)
        ),
        'dino_box_available': dino_box is not None,
        'dino_box_error': dino_box_error,
        'oracle_geometry_action': oracle_action,
        'oracle_geometry_reason': oracle_geometry.reason,
        'predicted_action_correct': predicted_action == oracle_action,
        'oracle_should_route': oracle_should_route,
        'predicted_should_route': predicted_should_route,
        'binary_route_correct': (
            predicted_should_route == oracle_should_route
        ),
    })
    audited_event['posthoc_oracle_audit'] = audit
    return audited_event


def _posthoc_oracle_audit_events(
        events,
        oracle_targets,
        *,
        context_window_tokens,
        accept_iou_threshold,
        containment_threshold):
    """Return audited event copies without mutating controller events."""

    matcher = ExplicitOracleTargetMatcher(oracle_targets, precision=3)
    return [
        _posthoc_oracle_audit_event(
            event,
            matcher,
            context_window_tokens=context_window_tokens,
            accept_iou_threshold=accept_iou_threshold,
            containment_threshold=containment_threshold,
        )
        for event in events
    ]


def _safe_ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def _posthoc_oracle_metrics(records):
    """Summarize natural-trajectory verifier quality on matchable events."""

    all_events = [
        event
        for record in records
        for event in record['intervention']['events']
    ]
    audits = [
        event['posthoc_oracle_audit']
        for event in all_events
        if isinstance(event.get('posthoc_oracle_audit'), dict)
    ]
    missing_audit_count = len(all_events) - len(audits)
    matched = [audit for audit in audits if audit.get('matchable')]
    unmatched_reasons = Counter(
        audit.get('match_reason') or 'missing_audit'
        for audit in audits
        if not audit.get('matchable')
    )

    prediction_labels = tuple(ACTION_NAMES) + ('abstained',)
    confusion = {
        truth: {prediction: 0 for prediction in prediction_labels}
        for truth in ACTION_NAMES
    }
    correct = 0
    for audit in matched:
        truth = audit['oracle_geometry_action']
        prediction = audit.get('predicted_action')
        if prediction not in ACTION_NAMES:
            prediction = 'abstained'
        confusion[truth][prediction] += 1
        correct += int(prediction == truth)

    true_positive = sum(
        bool(audit['oracle_should_route'])
        and bool(audit['predicted_should_route'])
        for audit in matched
    )
    false_positive = sum(
        not bool(audit['oracle_should_route'])
        and bool(audit['predicted_should_route'])
        for audit in matched
    )
    true_negative = sum(
        not bool(audit['oracle_should_route'])
        and not bool(audit['predicted_should_route'])
        for audit in matched
    )
    false_negative = sum(
        bool(audit['oracle_should_route'])
        and not bool(audit['predicted_should_route'])
        for audit in matched
    )
    precision = _safe_ratio(
        true_positive,
        true_positive + false_positive,
    )
    recall = _safe_ratio(
        true_positive,
        true_positive + false_negative,
    )
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    def mean_for(field):
        values = [
            float(audit[field])
            for audit in matched
            if audit.get(field) is not None
        ]
        if not values:
            return None, 0
        return sum(values) / len(values), len(values)

    candidate_miou, candidate_count = mean_for('candidate_iou_to_gt')
    committed_miou, committed_count = mean_for('committed_iou_to_gt')
    dino_miou, dino_count = mean_for('dino_iou_to_gt')
    matchable_samples = sum(
        any(
            (event.get('posthoc_oracle_audit') or {}).get('matchable')
            for event in record['intervention']['events']
        )
        for record in records
    )
    return {
        'coordinate_event_count': len(all_events),
        'audited_event_count': len(audits),
        'missing_audit_event_count': missing_audit_count,
        'matchable_event_count': len(matched),
        'unmatchable_event_count': len(audits) - len(matched),
        'matchable_event_rate': _safe_ratio(len(matched), len(audits)),
        'samples_with_matchable_event': matchable_samples,
        'unmatchable_reason_counts': dict(unmatched_reasons),
        'four_way': {
            'labels': list(ACTION_NAMES),
            'prediction_labels': list(prediction_labels),
            'confusion_true_by_predicted': confusion,
            'correct_count': correct,
            'accuracy': _safe_ratio(correct, len(matched)),
            'oracle_action_counts': dict(Counter(
                audit['oracle_geometry_action'] for audit in matched
            )),
            'predicted_action_counts': dict(Counter(
                audit.get('predicted_action') or 'abstained'
                for audit in matched
            )),
        },
        'binary_route': {
            'positive_definition': 'action != no_action',
            'prediction_basis': (
                'raw_verifier_predicted_action_before_confidence_policy'
            ),
            'true_positive': true_positive,
            'false_positive': false_positive,
            'true_negative': true_negative,
            'false_negative': false_negative,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': _safe_ratio(
                true_positive + true_negative,
                len(matched),
            ),
        },
        'iou': {
            'candidate_to_gt_miou': candidate_miou,
            'candidate_to_gt_count': candidate_count,
            'committed_to_gt_miou': committed_miou,
            'committed_to_gt_count': committed_count,
            'dino_to_gt_miou': dino_miou,
            'dino_to_gt_count': dino_count,
            'dino_box_coverage_rate': _safe_ratio(
                dino_count,
                len(matched),
            ),
            'committed_minus_candidate_miou': (
                None
                if committed_miou is None or candidate_miou is None
                else committed_miou - candidate_miou
            ),
        },
    }


def _event_metrics(records):
    events = [
        event
        for record in records
        for event in record['intervention']['events']
    ]
    action_counts = Counter(
        event.get('predicted_action') or 'abstained'
        for event in events
    )
    router_counts = Counter(
        event.get('router_action') or 'none'
        for event in events
    )
    candidate_ious = []
    committed_ious = []
    for event in events:
        expert_metadata = event.get('expert_metadata') or {}
        candidate_iou = expert_metadata.get('candidate_iou_to_gt')
        committed_iou = event.get('committed_iou_to_gt')
        if candidate_iou is not None:
            candidate_ious.append(float(candidate_iou))
        if committed_iou is not None:
            committed_ious.append(float(committed_iou))
    return {
        'coordinate_event_count': len(events),
        'predicted_action_counts': dict(action_counts),
        'router_action_counts': dict(router_counts),
        'grounder_invocation_count': sum(
            bool(event.get('grounder_invoked')) for event in events
        ),
        'box_refiner_invocation_count': sum(
            bool(event.get('box_refiner_invoked')) for event in events
        ),
        'expert_unavailable_fail_open_count': sum(
            event.get('missing_expert_error') is not None for event in events
        ),
        'verifier_abstention_count': sum(
            bool(event.get('verifier_abstained')) for event in events
        ),
        'remote_worker_failure_count': sum(
            bool(
                (event.get('verifier_metadata') or {}).get(
                    'remote_failure'
                )
            )
            for event in events
        ),
        'mean_candidate_iou_to_gt_on_corrected_events': (
            None if not candidate_ious
            else sum(candidate_ious) / len(candidate_ious)
        ),
        'mean_committed_iou_to_gt_on_corrected_events': (
            None if not committed_ious
            else sum(committed_ious) / len(committed_ious)
        ),
    }


def _subset_summary(records):
    summary = paired_metrics(records)
    summary['routing'] = _event_metrics(records)
    summary['posthoc_oracle_audit'] = _posthoc_oracle_metrics(records)
    return summary


def _make_summary(
        records,
        args,
        run_id,
        worker_ping,
        worker_warmup,
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
        'worker_ping': worker_ping,
        'worker_warmup': worker_warmup,
        'experiment_signature': experiment_signature,
        'settings': {
            'mode': 'remote_dino_geometry_oracle_experts',
            'baseline_results': args.baseline_results,
            'generator_cuda_visible_devices': os.environ.get(
                'CUDA_VISIBLE_DEVICES'
            ),
            'dino_physical_gpu': args.dino_gpu,
            'dino_python': args.dino_python,
            'dino_model_path': args.dino_model_path,
            'dino_box_threshold': args.dino_box_threshold,
            'dino_text_threshold': args.dino_text_threshold,
            'geometry_accept_iou': args.geometry_accept_iou,
            'geometry_containment': args.geometry_containment,
            'verifier_confidence_threshold': (
                args.verifier_confidence_threshold
            ),
            'worker_fail_open': args.worker_fail_open,
            'missing_expert_policy': args.missing_expert_policy,
            'fail_fast': args.fail_fast,
            'oracle_experts': {
                'relocate': 'oracle_grounder',
                'expand': 'oracle_box_refiner',
                'tighten': 'oracle_box_refiner',
                'unmatched_policy': 'fail_open_keep_candidate',
                'alias_policy': (
                    'latest_unique_longest_explicit_alias'
                ),
            },
            'posthoc_oracle_audit': {
                'enabled': True,
                'affects_routing': False,
                'geometry_accept_iou': args.geometry_accept_iou,
                'geometry_containment': args.geometry_containment,
                'alias_policy': (
                    ExplicitOracleTargetMatcher.POLICY
                ),
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


def main():
    args = parse_args()
    _validate_args(args)
    experiment_signature = _experiment_signature(args)
    iou_setting = 'iou_' + format(
        args.geometry_accept_iou, 'g'
    ).replace('.', 'p')
    layout = create_run_layout(
        dataset='vstar',
        split=args.run_split,
        study='routing',
        method='dino_geometry__oracle_experts',
        setting=iou_setting,
        run_id=args.run_id,
        output=args.output,
        output_root=args.output_root,
    )
    layout.ensure_run_directories()
    output_path = layout.results_path
    run_id = layout.run_id
    verifier_log = (
        Path(args.verifier_log)
        if args.verifier_log
        else layout.events_path
    )
    if verifier_log.resolve() == output_path.resolve():
        raise ValueError('--verifier-log must differ from the results path')

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
            'grounder': 'oracle',
            'box_refiner': 'oracle',
        },
        'coordinate_system': ORACLE_BOX_COORDINATE_SYSTEM,
        'worker_ping': None,
        'worker_warmup': None,
    }
    client = None
    worker_ping = None
    worker_warmup = None
    successful_count = 0
    error_count = 0
    lifecycle_started = False

    try:
        sources = _selected_sources(args)
        existing = (
            []
            if args.no_resume or not output_path.exists()
            else _latest_records_by_question_id(read_jsonl(output_path))
        )
        if not args.no_resume:
            _validate_resume_signatures(existing, experiment_signature)
        completed = {
            record['question_id']
            for record in existing
            if record.get('status') == 'ok'
        }
        pending = [
            source for source in sources
            if source['question_id'] not in completed
        ]
        # Remove an earlier error for every sample that is about to be
        # retried.  The following append therefore cannot create duplicate
        # question IDs, even if the previous process was interrupted.
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
                worker_ping = previous_config.get('worker_ping')
                worker_warmup = previous_config.get('worker_warmup')
                config['worker_ping'] = worker_ping
                config['worker_warmup'] = worker_warmup
        successful_count = sum(
            record.get('status') == 'ok' for record in retained
        )
        error_count = len(retained) - successful_count
        _atomic_write_jsonl(output_path, retained)
        _atomic_write_jsonl(verifier_log, _record_events(retained))

        write_run_config(layout, config)
        write_run_status(
            layout,
            'running',
            completed_records=len(completed),
            pending_records=len(pending),
            experiment_signature_sha256=experiment_signature['sha256'],
        )
        lifecycle_started = True
        print(f'Run id: {run_id}; output: {output_path}', flush=True)
        print(
            'Generator CUDA_VISIBLE_DEVICES='
            f'{os.environ.get("CUDA_VISIBLE_DEVICES")!r}; '
            f'DINO worker physical GPU={args.dino_gpu!r}',
            flush=True,
        )
        print(
            f'VStar selected={len(sources)}; pending={len(pending)}; '
            f'resumed={len(completed)}',
            flush=True,
        )

        if pending:
            worker_command = _worker_command(args)
            print(
                'Starting DINO verifier worker: '
                + ' '.join(worker_command),
                flush=True,
            )
            client = PersistentJsonlWorkerClient(
                worker_command,
                cwd=str(PROJECT_ROOT),
                env={'CUDA_VISIBLE_DEVICES': args.dino_gpu},
                timeout=args.worker_timeout,
                stderr=None,
                start=False,
            )
            client.start()
            worker_ping = client.ping(timeout=30.0)
            config['worker_ping'] = worker_ping
            write_run_config(layout, config)
            if not worker_ping.get('configured'):
                raise RuntimeError(
                    f'DINO worker is not configured: {worker_ping}'
                )
            try:
                worker_warmup = _worker_warmup(client, pending[0], args)
            except Exception as error:
                worker_warmup = {
                    'purpose': 'environment_check_only',
                    'affects_routing': False,
                    'status': 'error',
                    'error': f'{type(error).__name__}: {error}',
                }
                config['worker_ping'] = worker_ping
                config['worker_warmup'] = worker_warmup
                write_run_config(layout, config)
                raise
            worker_warmup['status'] = 'ok'
            print(
                'DINO worker ready after real warm-up: '
                + json.dumps({
                    'ping': worker_ping,
                    'warmup_action': (
                        worker_warmup['validated_output'][
                            'predicted_action'
                        ]
                    ),
                }, ensure_ascii=False),
                flush=True,
            )
            config['worker_ping'] = worker_ping
            config['worker_warmup'] = worker_warmup
            write_run_config(layout, config)

            model, preprocessor = load_model(
                args.model_path,
                precision='fp16',
            )
            verifier = RemoteActionVerifierBackend(
                client,
                timeout=args.worker_timeout,
                fail_open=args.worker_fail_open,
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
            grounder = OracleGrounderBackend(resolver)
            box_refiner = OracleBoxRefinerBackend(resolver)
            routing_policy = RoutingPolicy(
                confidence_threshold=args.verifier_confidence_threshold,
                unsupported_action='no_action',
                unknown_action='no_action',
            )

        with output_path.open('a', encoding='utf-8') as handle:
            for source in tqdm(
                    pending,
                    desc='VStar DINO verifier + oracle experts'):
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
                    'baseline_prediction': source[
                        'baseline_prediction'
                    ],
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
                        # Audited copies are written below, after routing.
                        log_path=None,
                        verifier_confidence_threshold=(
                            args.verifier_confidence_threshold
                        ),
                        sample_context={
                            'image_path': str(image_path),
                            'oracle_targets': source['oracle_targets'],
                        },
                    )
                    # Attach GT diagnostics only after the complete routed CoT
                    # has been generated.  These copied events are used solely
                    # for reporting and cannot change any routing decision.
                    routed = dict(routed)
                    routed['events'] = _posthoc_oracle_audit_events(
                        routed['events'],
                        source['oracle_targets'],
                        context_window_tokens=args.context_window_tokens,
                        accept_iou_threshold=args.geometry_accept_iou,
                        containment_threshold=args.geometry_containment,
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
                        'dino_geometry_oracle_router': routed,
                        'intervention': {
                            'mode': (
                                'remote_dino_geometry_oracle_experts'
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
                    _append_events(
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

        records = _latest_records_by_question_id(read_jsonl(output_path))
        # A final canonical rewrite guarantees one row per sample even if a
        # manually edited or partially recovered run reached this point.
        _atomic_write_jsonl(output_path, records)
        _atomic_write_jsonl(verifier_log, _record_events(records))
        summary = _make_summary(
            records,
            args,
            run_id,
            worker_ping,
            worker_warmup,
            experiment_signature,
        )
        summary_path = layout.summary_path
        summary_path.write_text(
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
            summary_path=str(summary_path),
            experiment_signature_sha256=experiment_signature['sha256'],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'Verifier events: {verifier_log}')
        print(f'Per-example results: {output_path}')
        print(f'Summary: {summary_path}')
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
        if client is not None:
            try:
                client.close()
            except Exception as close_error:
                print(
                    'Warning: failed to close DINO worker cleanly: '
                    f'{type(close_error).__name__}: {close_error}',
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == '__main__':
    main()
