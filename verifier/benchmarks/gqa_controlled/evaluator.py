"""Run selectable verifier backends on the controlled GQA benchmark."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ...backends.grounding_dino import (
    GroundingDinoGeometryClassifier,
    LocalGroundingDinoRunner,
)
from ...backends.qwen25_vl import (
    BINARY_IMAGE_MODES,
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    DEFAULT_QWEN_CROP_MIN_SIDE,
    GROUNDING_ACTION_IMAGE_MODES,
    GROUNDING_PROMPT_PROTOCOLS,
    LocalQwen25VLRunner,
    Qwen25VLGroundingActionClassifier,
    Qwen25VLGroundingGeometryClassifier,
    Qwen25VLVerifierBackend,
)
from .adapter import GQAControlledExample, load_examples
from .metrics import (
    compute_binary_alignment_metrics,
    compute_routing_metrics,
    compute_verifier_metrics,
)


DEFAULT_BENCHMARK = Path(
    'output/verifier_benchmark/gqa_controlled/v1/benchmark.jsonl'
)
DEFAULT_MODEL = Path('weights/Qwen2.5-VL-7B-Instruct')
FIVE_WAY_TO_ROUTING = {
    'aligned': 'no_action',
    'wrong_object': 'relocate',
    'unsupported': 'relocate',
    'partial_coverage': 'expand',
    'ambiguous': 'tighten',
}
ROUTING_TASK_MODES = (
    'routing_four_way',
    'routing_option_likelihood',
    'routing_grounding_geometry',
)
GEOMETRY_BACKENDS = ('qwen25_vl', 'grounding_dino')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate selectable verifier and reference-localizer protocols '
            'on the controlled GQA benchmark.'
        )
    )
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test', 'all'), default='test')
    parser.add_argument(
        '--task-mode',
        choices=(
            'five_way',
            'routing_four_way',
            'routing_option_likelihood',
            'routing_grounding_geometry',
            'binary_alignment',
        ),
        default='five_way',
    )
    parser.add_argument(
        '--binary-image-mode',
        choices=BINARY_IMAGE_MODES,
        default='crop_only',
        help=(
            'binary ablation input: crop only, marked full scene only, or '
            'marked full scene followed by the same crop'
        ),
    )
    parser.add_argument(
        '--five-way-image-mode',
        choices=BINARY_IMAGE_MODES,
        default='marked_plus_crop',
        help=(
            'five-way input: crop only, marked full scene only, or marked '
            'full scene followed by the same crop'
        ),
    )
    parser.add_argument(
        '--routing-image-mode',
        choices=BINARY_IMAGE_MODES,
        default='bbox_image_only',
        help=(
            'routing four-way input: crop only, marked full scene only, or '
            'marked full scene followed by the same crop'
        ),
    )
    parser.add_argument(
        '--option-image-mode',
        choices=GROUNDING_ACTION_IMAGE_MODES,
        default='raw_image',
        help=(
            'option-likelihood routing input: clean source image with bbox '
            'coordinates in text, or the same image with a red bbox overlay'
        ),
    )
    parser.add_argument(
        '--grounding-image-mode',
        choices=GROUNDING_ACTION_IMAGE_MODES,
        default='raw_image',
        help=(
            'geometry routing input: clean source image or source image with '
            'the candidate outlined in red'
        ),
    )
    parser.add_argument(
        '--geometry-backend',
        choices=GEOMETRY_BACKENDS,
        default='qwen25_vl',
        help=(
            'reference localizer used by routing_grounding_geometry; the '
            'default preserves the existing Qwen behavior'
        ),
    )
    parser.add_argument(
        '--grounding-accept-iou',
        type=float,
        default=0.5,
        help='accept the candidate when candidate/grounder IoU reaches this',
    )
    parser.add_argument(
        '--grounding-containment',
        type=float,
        default=0.7,
        help=(
            'directed coverage required to classify a low-IoU candidate as '
            'expand or tighten instead of relocate'
        ),
    )
    parser.add_argument(
        '--grounding-boundary-tolerance',
        type=float,
        default=DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
        help=(
            'maximum per-side pixel excursion that is clipped and audited '
            'instead of treated as a grounding parse failure'
        ),
    )
    parser.add_argument(
        '--grounding-prompt-protocol',
        choices=GROUNDING_PROMPT_PROTOCOLS,
        default=DEFAULT_GROUNDING_PROMPT_PROTOCOL,
        help=(
            'compact original localization prompt or strict one-object JSON '
            'protocol'
        ),
    )
    parser.add_argument(
        '--dino-box-threshold',
        type=float,
        default=0.3,
        help='Grounding DINO box confidence threshold, selected on dev only',
    )
    parser.add_argument(
        '--dino-text-threshold',
        type=float,
        default=0.25,
        help='Grounding DINO text confidence threshold, selected on dev only',
    )
    parser.add_argument(
        '--dino-dtype',
        choices=(
            'auto',
            'float32',
            'fp32',
            'float16',
            'fp16',
            'bfloat16',
            'bf16',
        ),
        default='float32',
        help='Grounding DINO inference dtype; independent of Qwen --dtype',
    )
    parser.add_argument(
        '--dino-top-k-log',
        type=int,
        default=20,
        help='maximum number of raw Grounding DINO detections logged per item',
    )
    parser.add_argument(
        '--crop-min-side',
        type=int,
        default=DEFAULT_QWEN_CROP_MIN_SIDE,
        help='upscale candidate crops whose shortest side is below this value',
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument(
        '--min-pixels',
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help='minimum Qwen image pixels per image',
    )
    parser.add_argument(
        '--max-pixels',
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=(
            'maximum Qwen image pixels per image; default corresponds to '
            'about 512 merged visual tokens per image'
        ),
    )
    parser.add_argument('--attn-implementation', default='sdpa')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def _summary_path(output: Path) -> Path:
    if output.suffix == '.jsonl':
        return output.with_suffix('.summary.json')
    return Path(str(output) + '.summary.json')


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'invalid existing result at {path}:{line_number}: {error}'
                ) from error
    return rows


def _progress(examples: Iterable[GQAControlledExample], total: int):
    try:
        from tqdm import tqdm
    except ImportError:
        return examples
    return tqdm(examples, total=total, desc='GQA verifier benchmark')


def _percentile(values: List[float], quantile: float):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def _result_record(
        example: GQAControlledExample,
        candidate_bbox,
        lookup,
) -> Dict[str, Any]:
    status = lookup.metadata.get('status')
    result = lookup.result
    return {
        'event_id': example.event_id,
        'sample_index': example.sample_index,
        'split': example.split,
        'image_id': example.image_id,
        'source_image': str(example.source_image),
        'object_reference': example.object_reference,
        'candidate_box_pixel_xyxy': list(example.candidate_box_pixel_xyxy),
        'candidate_box_padded_normalized_xyxy': list(candidate_bbox),
        'expected_status': example.expected_status,
        'expected_verdict': example.expected_verdict,
        'expected_reason': example.expected_reason,
        'predicted_status': status,
        'predicted_verdict': result.verdict,
        'predicted_reason': result.reason,
        'confidence': result.confidence,
        'correct': status == example.expected_status,
        'parse_failed': bool(lookup.metadata.get('parse_failed')),
        'error': lookup.error,
        'verifier_metadata': lookup.metadata,
    }


def _binary_result_record(
        example: GQAControlledExample,
        candidate_bbox,
        lookup,
) -> Dict[str, Any]:
    expected_alignment = example.expected_status == 'aligned'
    return {
        'event_id': example.event_id,
        'sample_index': example.sample_index,
        'split': example.split,
        'image_id': example.image_id,
        'source_image': str(example.source_image),
        'object_reference': example.object_reference,
        'candidate_box_pixel_xyxy': list(example.candidate_box_pixel_xyxy),
        'candidate_box_padded_normalized_xyxy': list(candidate_bbox),
        'expected_status': example.expected_status,
        'expected_alignment': expected_alignment,
        'predicted_alignment': lookup.aligned,
        'confidence': lookup.confidence,
        'correct': lookup.aligned == expected_alignment,
        'parse_failed': bool(lookup.metadata.get('parse_failed')),
        'error': lookup.error,
        'verifier_metadata': lookup.metadata,
    }


def _routing_result_record(
        example: GQAControlledExample,
        candidate_bbox,
        lookup,
) -> Dict[str, Any]:
    expected_routing_status = FIVE_WAY_TO_ROUTING[example.expected_status]
    return {
        'event_id': example.event_id,
        'sample_index': example.sample_index,
        'split': example.split,
        'image_id': example.image_id,
        'source_image': str(example.source_image),
        'object_reference': example.object_reference,
        'candidate_box_pixel_xyxy': list(example.candidate_box_pixel_xyxy),
        'candidate_box_padded_normalized_xyxy': list(candidate_bbox),
        'expected_status': example.expected_status,
        'expected_routing_status': expected_routing_status,
        'predicted_routing_status': lookup.status,
        'confidence': lookup.confidence,
        'correct': lookup.status == expected_routing_status,
        'parse_failed': bool(lookup.metadata.get('parse_failed')),
        'error': lookup.error,
        'verifier_metadata': lookup.metadata,
    }


def _routing_option_result_record(
        example: GQAControlledExample,
        lookup,
) -> Dict[str, Any]:
    expected_routing_status = FIVE_WAY_TO_ROUTING[example.expected_status]
    return {
        'event_id': example.event_id,
        'sample_index': example.sample_index,
        'split': example.split,
        'image_id': example.image_id,
        'source_image': str(example.source_image),
        'object_reference': example.object_reference,
        'candidate_box_pixel_xyxy': list(example.candidate_box_pixel_xyxy),
        'candidate_box_padded_normalized_xyxy': None,
        'expected_status': example.expected_status,
        'expected_routing_status': expected_routing_status,
        'predicted_routing_status': lookup.status,
        'confidence': lookup.confidence,
        'correct': lookup.status == expected_routing_status,
        'parse_failed': bool(lookup.metadata.get('parse_failed')),
        'error': lookup.error,
        'verifier_metadata': lookup.metadata,
    }


def main() -> None:
    args = parse_args()
    if args.start_index < 0:
        raise ValueError('--start-index must be non-negative')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    if args.max_new_tokens <= 0:
        raise ValueError('--max-new-tokens must be positive')
    if args.min_pixels <= 0 or args.max_pixels <= 0:
        raise ValueError('--min-pixels and --max-pixels must be positive')
    if args.min_pixels > args.max_pixels:
        raise ValueError('--min-pixels must not exceed --max-pixels')
    if args.crop_min_side <= 28:
        raise ValueError('--crop-min-side must be greater than 28')
    if not 0.0 < args.grounding_accept_iou <= 1.0:
        raise ValueError('--grounding-accept-iou must be in (0, 1]')
    if not 0.0 < args.grounding_containment <= 1.0:
        raise ValueError('--grounding-containment must be in (0, 1]')
    if args.grounding_boundary_tolerance < 0:
        raise ValueError(
            '--grounding-boundary-tolerance must be non-negative'
        )
    for value, name in (
        (args.dino_box_threshold, '--dino-box-threshold'),
        (args.dino_text_threshold, '--dino-text-threshold'),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')
    if args.dino_top_k_log <= 0:
        raise ValueError('--dino-top-k-log must be positive')
    if args.resume and args.overwrite:
        raise ValueError('--resume and --overwrite are mutually exclusive')
    if (
        args.task_mode != 'routing_grounding_geometry'
        and args.geometry_backend != 'qwen25_vl'
    ):
        raise ValueError(
            '--geometry-backend only applies with '
            '--task-mode routing_grounding_geometry'
        )
    if (
        args.geometry_backend == 'grounding_dino'
        and args.grounding_image_mode != 'raw_image'
    ):
        raise ValueError(
            'Grounding DINO supports only --grounding-image-mode raw_image '
            'so the candidate remains hidden from the localizer'
        )
    if args.task_mode != 'binary_alignment' and args.binary_image_mode != 'crop_only':
        raise ValueError(
            '--binary-image-mode only applies with --task-mode binary_alignment'
        )
    if (
        args.task_mode != 'five_way'
        and args.five_way_image_mode != 'marked_plus_crop'
    ):
        raise ValueError(
            '--five-way-image-mode only applies with --task-mode five_way'
        )
    if (
        args.task_mode != 'routing_four_way'
        and args.routing_image_mode != 'bbox_image_only'
    ):
        raise ValueError(
            '--routing-image-mode only applies with '
            '--task-mode routing_four_way'
        )
    if (
        args.task_mode != 'routing_option_likelihood'
        and args.option_image_mode != 'raw_image'
    ):
        raise ValueError(
            '--option-image-mode only applies with '
            '--task-mode routing_option_likelihood'
        )
    if (
        args.task_mode != 'routing_grounding_geometry'
        and args.grounding_image_mode != 'raw_image'
    ):
        raise ValueError(
            '--grounding-image-mode only applies with '
            '--task-mode routing_grounding_geometry'
        )

    examples = load_examples(args.benchmark, args.split)
    examples = examples[args.start_index:]
    if args.limit is not None:
        examples = examples[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(args.output)
    if existing and not args.resume and not args.overwrite:
        raise FileExistsError(
            f'output already contains results: {args.output}; use --resume '
            'or choose a new output path'
        )
    if args.overwrite:
        existing = []
    completed_ids = {str(row.get('event_id')) for row in existing}
    pending = [
        example for example in examples
        if example.event_id not in completed_ids
    ]

    backend = None
    action_classifier = None
    geometry_classifier = None
    if (
        args.task_mode == 'routing_grounding_geometry'
        and args.geometry_backend == 'grounding_dino'
    ):
        dino_runner = LocalGroundingDinoRunner(
            model_path=str(args.model_path),
            device=args.device,
            dtype=args.dino_dtype,
            box_threshold=args.dino_box_threshold,
            text_threshold=args.dino_text_threshold,
        )
        geometry_classifier = GroundingDinoGeometryClassifier(
            runner=dino_runner,
            accept_iou_threshold=args.grounding_accept_iou,
            containment_threshold=args.grounding_containment,
            top_k_log=args.dino_top_k_log,
        )
    else:
        runner = LocalQwen25VLRunner(
            model_path=str(args.model_path),
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attn_implementation=args.attn_implementation,
        )
        backend = Qwen25VLVerifierBackend(
            runner=runner,
            crop_min_side=args.crop_min_side,
            parse_fail_open=True,
        )
        action_classifier = Qwen25VLGroundingActionClassifier(runner=runner)
        geometry_classifier = Qwen25VLGroundingGeometryClassifier(
            runner=runner,
            accept_iou_threshold=args.grounding_accept_iou,
            containment_threshold=args.grounding_containment,
            boundary_tolerance_pixels=args.grounding_boundary_tolerance,
            prompt_protocol=args.grounding_prompt_protocol,
        )

    output_mode = 'a' if existing else 'w'
    with args.output.open(output_mode, encoding='utf-8') as handle:
        for example in _progress(pending, len(pending)):
            try:
                if args.task_mode == 'routing_option_likelihood':
                    action_input = example.to_grounding_action_input()
                    assert action_classifier is not None
                    lookup = action_classifier.classify(
                        action_input,
                        image_mode=args.option_image_mode,
                    )
                    record = _routing_option_result_record(
                        example,
                        lookup,
                    )
                elif args.task_mode == 'routing_grounding_geometry':
                    action_input = example.to_grounding_action_input()
                    assert geometry_classifier is not None
                    lookup = geometry_classifier.classify(
                        action_input,
                        image_mode=args.grounding_image_mode,
                    )
                    record = _routing_option_result_record(
                        example,
                        lookup,
                    )
                else:
                    candidate = example.to_candidate_input()
                    assert backend is not None
                    if args.task_mode == 'binary_alignment':
                        lookup = backend.verify_binary_alignment_candidate(
                            candidate,
                            image_mode=args.binary_image_mode,
                        )
                        record = _binary_result_record(
                            example,
                            candidate.candidate_bbox,
                            lookup,
                        )
                    elif args.task_mode == 'routing_four_way':
                        lookup = backend.classify_routing_candidate(
                            candidate,
                            image_mode=args.routing_image_mode,
                        )
                        record = _routing_result_record(
                            example,
                            candidate.candidate_bbox,
                            lookup,
                        )
                    else:
                        lookup = backend.verify_candidate(
                            candidate,
                            image_mode=args.five_way_image_mode,
                        )
                        record = _result_record(
                            example,
                            candidate.candidate_bbox,
                            lookup,
                        )
            except Exception as error:
                if args.fail_fast:
                    raise
                record = {
                    'event_id': example.event_id,
                    'sample_index': example.sample_index,
                    'split': example.split,
                    'image_id': example.image_id,
                    'source_image': str(example.source_image),
                    'object_reference': example.object_reference,
                    'candidate_box_pixel_xyxy': list(
                        example.candidate_box_pixel_xyxy
                    ),
                    'candidate_box_padded_normalized_xyxy': None,
                    'expected_status': example.expected_status,
                    'expected_verdict': example.expected_verdict,
                    'expected_reason': example.expected_reason,
                    'expected_routing_status': (
                        FIVE_WAY_TO_ROUTING[example.expected_status]
                        if args.task_mode in ROUTING_TASK_MODES
                        else None
                    ),
                    'expected_alignment': (
                        example.expected_status == 'aligned'
                        if args.task_mode == 'binary_alignment'
                        else None
                    ),
                    'predicted_alignment': None,
                    'predicted_status': None,
                    'predicted_routing_status': None,
                    'predicted_verdict': None,
                    'predicted_reason': None,
                    'confidence': None,
                    'correct': False,
                    'parse_failed': False,
                    'error': f'{type(error).__name__}: {error}',
                    'verifier_metadata': {},
                }
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            if args.verbose:
                predicted = (
                    record.get('predicted_alignment')
                    if args.task_mode == 'binary_alignment'
                    else (
                        record.get('predicted_routing_status')
                        if args.task_mode in ROUTING_TASK_MODES
                        else record.get('predicted_status')
                    )
                )
                expected = (
                    record.get('expected_routing_status')
                    if args.task_mode in ROUTING_TASK_MODES
                    else example.expected_status
                )
                print(
                    f"[{example.event_id}] expected={expected} "
                    f"predicted={predicted} "
                    f"confidence={record['confidence']} "
                    f"correct={record['correct']}",
                    flush=True,
                )
                raw_response = record['verifier_metadata'].get('raw_response')
                if raw_response is not None:
                    print(f'  raw: {raw_response}', flush=True)
                option_losses = record['verifier_metadata'].get(
                    'option_negative_log_likelihoods'
                )
                if option_losses is not None:
                    print(f'  option NLL: {option_losses}', flush=True)

    results = _read_jsonl(args.output)
    selected_ids = {example.event_id for example in examples}
    selected_results = [
        record for record in results
        if str(record.get('event_id')) in selected_ids
    ]
    is_dino_geometry = (
        args.task_mode == 'routing_grounding_geometry'
        and args.geometry_backend == 'grounding_dino'
    )
    selected_image_mode = (
        args.binary_image_mode
        if args.task_mode == 'binary_alignment'
        else (
            (
                args.option_image_mode
                if args.task_mode == 'routing_option_likelihood'
                else args.grounding_image_mode
            )
            if args.task_mode in (
                'routing_option_likelihood',
                'routing_grounding_geometry',
            )
            else (
                args.routing_image_mode
                if args.task_mode == 'routing_four_way'
                else args.five_way_image_mode
            )
        )
    )
    model_images_by_mode = {
        'raw_image': 'clean source image with candidate bbox supplied as text',
        'bbox_image': (
            'source image with red candidate rectangle and the same bbox '
            'supplied as text'
        ),
        'crop_only': 'border-free crop from inside the candidate rectangle',
        'bbox_image_only': 'marked full scene only',
        'marked_plus_crop': (
            'marked full scene followed by border-free candidate crop'
        ),
    }
    if args.task_mode == 'routing_grounding_geometry':
        if is_dino_geometry:
            model_image_description = (
                'clean source image and object reference only; candidate '
                'coordinates are hidden from Grounding DINO'
            )
        else:
            model_image_description = (
                'clean source image; candidate coordinates are hidden from '
                'Qwen'
                if selected_image_mode == 'raw_image'
                else (
                    'source image with red candidate rectangle; candidate '
                    'coordinates are hidden from Qwen'
                )
            )
    else:
        model_image_description = model_images_by_mode[selected_image_mode]
    if is_dino_geometry:
        dino_metadata = [
            record.get('verifier_metadata', {})
            for record in selected_results
        ]
        detection_counts = [
            metadata.get('detection_count')
            for metadata in dino_metadata
            if isinstance(metadata.get('detection_count'), int)
        ]
        total_latencies = [
            metadata.get('timing_ms', {}).get('total')
            for metadata in dino_metadata
            if isinstance(metadata.get('timing_ms'), dict)
            and isinstance(
                metadata.get('timing_ms', {}).get('total'),
                (int, float),
            )
        ]
        dino_diagnostics = {
            'localization_failure_count': sum(
                bool(metadata.get('localization_failed'))
                for metadata in dino_metadata
            ),
            'mean_detection_count': (
                sum(detection_counts) / len(detection_counts)
                if detection_counts else None
            ),
            'mean_latency_ms': (
                sum(total_latencies) / len(total_latencies)
                if total_latencies else None
            ),
            'p50_latency_ms': _percentile(total_latencies, 0.5),
            'p95_latency_ms': _percentile(total_latencies, 0.95),
        }
    else:
        dino_diagnostics = None
    backend_name = {
        'binary_alignment': (
            f'qwen25_vl_binary_alignment_{args.binary_image_mode}'
        ),
        'routing_four_way': (
            f'qwen25_vl_routing_four_way_{args.routing_image_mode}'
        ),
        'routing_option_likelihood': (
            'qwen25_vl_grounding_action_option_likelihood_'
            f'{args.option_image_mode}'
        ),
        'routing_grounding_geometry': (
            (
                'grounding_dino_geometry_router_raw_image'
                if is_dino_geometry
                else (
                    'qwen25_vl_grounding_geometry_router_'
                    f'{args.grounding_image_mode}'
                )
            )
        ),
        'five_way': (
            f'qwen25_vl_zero_shot_five_way_{args.five_way_image_mode}'
        ),
    }[args.task_mode]
    summary = {
        'benchmark': str(args.benchmark),
        'model_path': str(args.model_path),
        'backend': backend_name,
        'task_mode': args.task_mode,
        'geometry_backend': (
            args.geometry_backend
            if args.task_mode == 'routing_grounding_geometry'
            else None
        ),
        'binary_image_mode': (
            args.binary_image_mode
            if args.task_mode == 'binary_alignment'
            else None
        ),
        'five_way_image_mode': (
            args.five_way_image_mode
            if args.task_mode == 'five_way'
            else None
        ),
        'routing_image_mode': (
            args.routing_image_mode
            if args.task_mode == 'routing_four_way'
            else None
        ),
        'option_image_mode': (
            args.option_image_mode
            if args.task_mode == 'routing_option_likelihood'
            else None
        ),
        'grounding_image_mode': (
            args.grounding_image_mode
            if args.task_mode == 'routing_grounding_geometry'
            else None
        ),
        'grounding_accept_iou': (
            args.grounding_accept_iou
            if args.task_mode == 'routing_grounding_geometry'
            else None
        ),
        'grounding_containment': (
            args.grounding_containment
            if args.task_mode == 'routing_grounding_geometry'
            else None
        ),
        'grounding_boundary_tolerance_pixels': (
            args.grounding_boundary_tolerance
            if (
                args.task_mode == 'routing_grounding_geometry'
                and not is_dino_geometry
            )
            else None
        ),
        'grounding_prompt_protocol': (
            args.grounding_prompt_protocol
            if (
                args.task_mode == 'routing_grounding_geometry'
                and not is_dino_geometry
            )
            else None
        ),
        'dino_box_threshold': (
            args.dino_box_threshold if is_dino_geometry else None
        ),
        'dino_text_threshold': (
            args.dino_text_threshold if is_dino_geometry else None
        ),
        'dino_dtype': args.dino_dtype if is_dino_geometry else None,
        'dino_top_k_log': (
            args.dino_top_k_log if is_dino_geometry else None
        ),
        'dino_selection_policy': (
            'highest_detector_score_candidate_hidden'
            if is_dino_geometry else None
        ),
        'dino_diagnostics': dino_diagnostics,
        'split': args.split,
        'start_index': args.start_index,
        'limit': args.limit,
        'min_pixels_per_image': (
            None if is_dino_geometry else args.min_pixels
        ),
        'max_pixels_per_image': (
            None if is_dino_geometry else args.max_pixels
        ),
        'crop_min_side': None if is_dino_geometry else args.crop_min_side,
        'selected_example_count': len(examples),
        'completed_result_count': len(selected_results),
        'input_protocol': {
            'source_fields': [
                'source_image',
                'object_reference',
                'candidate_box_pixel_xyxy',
            ],
            'coordinate_conversion': (
                (
                    'candidate and post-processed grounding box are compared '
                    'directly as original-image absolute pixel xyxy; no Qwen '
                    'resize or VoCoT padding conversion'
                )
                if is_dino_geometry
                else (
                    (
                        'original-image pixel xyxy and clean image are jointly '
                        'scaled to the exact Qwen smart-resized image frame'
                    )
                    if args.task_mode in (
                        'routing_option_likelihood',
                        'routing_grounding_geometry',
                    )
                    else (
                        'original-image pixel xyxy to normalized xyxy on '
                        'VoCoT center-padded square'
                    )
                )
            ),
            'image_mode': selected_image_mode,
            'model_images': [model_image_description],
            'detector_visible_fields': (
                ['source_image', 'object_reference']
                if is_dino_geometry else None
            ),
            'supervision_not_visible_to_model': [
                'target_box_pixel_xyxy',
                'target_box_normalized_xyxy',
                'candidate_target_geometry',
                'verdict',
                'reason',
                'construction',
            ],
        },
        'metrics': (
            compute_binary_alignment_metrics(selected_results)
            if args.task_mode == 'binary_alignment'
            else (
                compute_routing_metrics(selected_results)
                if args.task_mode in ROUTING_TASK_MODES
                else compute_verifier_metrics(selected_results)
            )
        ),
    }
    summary_path = _summary_path(args.output)
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    metrics = summary['metrics']
    print(f'Results: {args.output}')
    print(f'Summary: {summary_path}')
    if args.task_mode == 'binary_alignment':
        print(
            'Binary alignment accuracy: '
            f"{metrics['end_to_end_accuracy'] * 100:.2f}% "
            f"({metrics['correct']}/{metrics['total']})"
        )
        print(f"Misalignment recall: {metrics['recall'] * 100:.2f}%")
    elif args.task_mode in ROUTING_TASK_MODES:
        print(
            'Routing four-way accuracy: '
            f"{metrics['four_way']['accuracy'] * 100:.2f}% "
            f"({metrics['four_way']['correct']}/{metrics['total']})"
        )
        print(
            'Macro-F1: '
            f"{metrics['four_way']['macro_f1'] * 100:.2f}%"
        )
    else:
        print(
            'Five-way accuracy: '
            f"{metrics['five_way']['accuracy'] * 100:.2f}% "
            f"({metrics['five_way']['correct']}/{metrics['total']})"
        )
        print(
            'Macro-F1: '
            f"{metrics['five_way']['macro_f1'] * 100:.2f}%"
        )
        print(
            'Binary aligned-vs-invalid accuracy: '
            f"{metrics['binary_aligned_vs_invalid']['end_to_end_accuracy'] * 100:.2f}%"
        )
    print(
        'Parse success rate: '
        f"{metrics['parse_success_rate'] * 100:.2f}%"
    )


if __name__ == '__main__':
    main()
