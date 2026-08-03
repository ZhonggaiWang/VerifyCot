"""Evaluate a Grounder backend independently on VStar object annotations.

The benchmark deliberately excludes VoCoT candidates, verifier decisions, the
question answer, and oracle data from the Grounder request.  Every annotated
target contributes one request containing only the immutable source image and
the canonical VStar ``target_object`` string.  Predictions and ground truth
are compared in the original-image pixel coordinate frame.

The evaluator is backend-neutral at the Grounder contract boundary.  The
first production configuration is Qwen2.5-VL-7B; Grounding DINO is also wired
through the same versioned worker protocol so later comparisons use identical
data loading and metrics.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grounding_control.contracts import (  # noqa: E402
    GroundingRequest,
    VisualInput,
)
from grounding_control.contracts.errors import (  # noqa: E402
    ExpertUnavailableError,
)
from grounding_control.coordinates import (  # noqa: E402
    original_pixel_box_to_normalized_square_box,
)
from grounding_control.experts.grounders import (  # noqa: E402
    RemoteGrounderBackend,
)
from grounding_control.models.qwen25_vl import (  # noqa: E402
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
)
from grounding_control.models.qwen25_vl.grounding_prompt import (  # noqa: E402
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
)
from grounding_control.run_paths import (  # noqa: E402
    RunLayout,
    create_run_layout,
    write_run_config,
    write_run_status,
)
from grounding_control.transport import (  # noqa: E402
    PersistentJsonlWorkerClient,
)


BENCHMARK_SCHEMA = 'vstar_grounder_accuracy_v1'
EXPERIMENT_SIGNATURE_SCHEMA = 'vstar_grounder_accuracy_signature_v1'
IOU_THRESHOLDS = tuple(index / 10.0 for index in range(1, 10))
COMPLETED_STATUSES = frozenset({'ok', 'grounder_unavailable'})
SUPPORTED_BACKENDS = ('qwen25_vl', 'grounding_dino')
EXPECTED_SOURCES = {
    'qwen25_vl': 'qwen25_vl_grounder',
    'grounding_dino': 'grounding_dino_grounder',
}
DEFAULT_WORKER_PYTHON = (
    '/home/zhonggai/miniconda3/envs/qwen25/bin/python'
)
DEFAULT_QWEN_MODEL = (
    '/data/zhonggai/models/Qwen2.5-VL-7B-Instruct'
)
DEFAULT_DINO_MODEL = '/data/zhonggai/models/grounding-dino-base'
DEFAULT_ORACLE_BOXES = (
    'output/vstar/annotations/oracle_boxes/full_238.jsonl'
)
DEFAULT_IMAGE_DIR = '/data/zhonggai/VStar'
_CATEGORY_NAMES = frozenset({
    'direct_attributes',
    'relative_position',
    'OCR',
    'GPT4V-hard',
})


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'{path}:{line_number}: invalid JSON: {error}'
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f'{path}:{line_number}: record must be an object'
                )
            records.append(record)
    return records


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field_name} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{field_name} must be finite')
    return result


def _validate_xyxy(
        box: Sequence[float],
        *,
        width: Optional[float] = None,
        height: Optional[float] = None,
        field_name: str = 'bbox') -> Tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f'{field_name} must contain four coordinates')
    values = tuple(
        _finite_number(value, f'{field_name}[{index}]')
        for index, value in enumerate(box)
    )
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f'{field_name} must have positive extent')
    if width is not None and height is not None:
        if not (
                0.0 <= values[0] < values[2] <= float(width)
                and 0.0 <= values[1] < values[3] <= float(height)):
            raise ValueError(
                f'{field_name} {values} is outside image '
                f'{width:g}x{height:g}'
            )
    return values


def _xywh_to_xyxy(
        box: Sequence[float],
        *,
        width: float,
        height: float,
        field_name: str) -> Tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f'{field_name} must contain [x, y, width, height]')
    x, y, box_width, box_height = tuple(
        _finite_number(value, f'{field_name}[{index}]')
        for index, value in enumerate(box)
    )
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f'{field_name} must have positive width and height')
    return _validate_xyxy(
        (x, y, x + box_width, y + box_height),
        width=width,
        height=height,
        field_name=field_name,
    )


def _relative_image_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'{field_name} must be a safe relative path')
    return path


def _size_bucket(area_fraction: float) -> str:
    # VStar targets are unusually small.  These relative-area bins retain
    # useful resolution where COCO's absolute-pixel bins would not.
    if area_fraction < 0.0005:
        return 'tiny_lt_0p05pct'
    if area_fraction < 0.002:
        return 'small_0p05_to_0p2pct'
    if area_fraction < 0.01:
        return 'medium_0p2_to_1pct'
    return 'large_ge_1pct'


def load_grounding_tasks(
        oracle_boxes_path: Any,
        image_dir: Any) -> List[Dict[str, Any]]:
    """Expand VStar rows into one strictly validated task per target box."""

    annotations_path = Path(oracle_boxes_path).resolve()
    image_root = Path(image_dir).resolve()
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)
    if not image_root.is_dir():
        raise NotADirectoryError(image_root)

    rows = _read_jsonl(annotations_path)
    tasks: List[Dict[str, Any]] = []
    seen_sample_ids = set()
    for row_index, row in enumerate(rows):
        sample_index = row.get('sample_index')
        if sample_index != row_index:
            raise ValueError(
                'sample_index must be contiguous and match JSONL order: '
                f'row {row_index} has {sample_index!r}'
            )
        sample_id = row.get('question_id')
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f'row {row_index}: invalid question_id')
        if sample_id in seen_sample_ids:
            raise ValueError(f'duplicate question_id: {sample_id!r}')
        seen_sample_ids.add(sample_id)

        category = row.get('category')
        if category not in _CATEGORY_NAMES:
            raise ValueError(
                f'{sample_id}: unsupported category {category!r}'
            )
        relative_path = _relative_image_path(
            row.get('image'), f'{sample_id}.image'
        )
        image_path = (image_root / relative_path).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError as error:
            raise ValueError(
                f'{sample_id}: image escapes the dataset root'
            ) from error
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        image_size = row.get('image_size')
        if not isinstance(image_size, dict):
            raise ValueError(f'{sample_id}: image_size must be an object')
        width = image_size.get('width')
        height = image_size.get('height')
        if (
                isinstance(width, bool) or not isinstance(width, int)
                or isinstance(height, bool) or not isinstance(height, int)
                or width <= 0 or height <= 0):
            raise ValueError(f'{sample_id}: invalid image_size')
        with Image.open(image_path) as image:
            actual_size = tuple(image.size)
        if actual_size != (width, height):
            raise ValueError(
                f'{sample_id}: image size {actual_size} does not match '
                f'annotation {(width, height)}'
            )

        references = row.get('target_objects')
        pixel_boxes = row.get('pixel_bboxes_xywh')
        normalized_boxes = row.get('normalized_bboxes_xyxy')
        if not isinstance(references, list) or not references:
            raise ValueError(f'{sample_id}: target_objects must be non-empty')
        if not isinstance(pixel_boxes, list) or not pixel_boxes:
            raise ValueError(
                f'{sample_id}: pixel_bboxes_xywh must be non-empty'
            )
        if len(references) != len(pixel_boxes):
            raise ValueError(
                f'{sample_id}: target and box count mismatch: '
                f'{len(references)} != {len(pixel_boxes)}'
            )
        if normalized_boxes is not None and (
                not isinstance(normalized_boxes, list)
                or len(normalized_boxes) != len(pixel_boxes)):
            raise ValueError(
                f'{sample_id}: normalized target box count mismatch'
            )

        for target_index, (reference, xywh) in enumerate(
                zip(references, pixel_boxes)):
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f'{sample_id}:target:{target_index}: invalid reference'
                )
            # Preserve the canonical annotation exactly apart from removing
            # accidental surrounding whitespace.
            reference = reference.strip()
            field = f'{sample_id}:target:{target_index}.bbox'
            gt_pixel = _xywh_to_xyxy(
                xywh,
                width=width,
                height=height,
                field_name=field,
            )
            original_normalized = (
                gt_pixel[0] / width,
                gt_pixel[1] / height,
                gt_pixel[2] / width,
                gt_pixel[3] / height,
            )
            if normalized_boxes is not None:
                supplied = tuple(
                    _finite_number(value, f'{field}.normalized[{index}]')
                    for index, value in enumerate(
                        normalized_boxes[target_index]
                    )
                )
                if len(supplied) != 4 or any(
                        abs(first - second) > 1e-9
                        for first, second in zip(
                            supplied, original_normalized
                        )):
                    raise ValueError(
                        f'{sample_id}:target:{target_index}: normalized '
                        'bbox does not match pixel bbox'
                    )
            padded = original_pixel_box_to_normalized_square_box(
                gt_pixel, width, height
            )
            area_fraction = (
                (gt_pixel[2] - gt_pixel[0])
                * (gt_pixel[3] - gt_pixel[1])
                / float(width * height)
            )
            tasks.append({
                'benchmark_schema': BENCHMARK_SCHEMA,
                'task_id': f'{sample_id}:target:{target_index}',
                'target_ordinal': len(tasks),
                'sample_index': sample_index,
                'sample_id': sample_id,
                'target_index': target_index,
                'category': category,
                'image': str(relative_path),
                'image_path': str(image_path),
                'image_size': [width, height],
                'question': row.get('question'),
                'object_reference': reference,
                'reference_protocol': 'canonical_vstar_target_object',
                'gt_bbox_original_pixel_xyxy': list(gt_pixel),
                'gt_bbox_original_normalized_xyxy': list(
                    original_normalized
                ),
                'gt_bbox_vocot_normalized_padded_xyxy': list(padded),
                'gt_area_fraction': area_fraction,
                'gt_size_bucket': _size_bucket(area_fraction),
                'has_complete_question_target_coverage': row.get(
                    'has_complete_question_target_coverage'
                ),
            })
    return tasks


def _intersection_area(
        first: Sequence[float],
        second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    return max(0.0, right - left) * max(0.0, bottom - top)


def box_iou_xyxy(
        first: Sequence[float],
        second: Sequence[float]) -> float:
    """Continuous-coordinate IoU; pixel boxes deliberately use no ``+1``."""

    first_box = _validate_xyxy(first, field_name='first_bbox')
    second_box = _validate_xyxy(second, field_name='second_bbox')
    intersection = _intersection_area(first_box, second_box)
    first_area = (
        (first_box[2] - first_box[0])
        * (first_box[3] - first_box[1])
    )
    second_area = (
        (second_box[2] - second_box[0])
        * (second_box[3] - second_box[1])
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _overlap_metrics(
        prediction: Sequence[float],
        target: Sequence[float]) -> Dict[str, float]:
    prediction_box = _validate_xyxy(prediction, field_name='prediction')
    target_box = _validate_xyxy(target, field_name='target')
    intersection = _intersection_area(prediction_box, target_box)
    prediction_area = (
        (prediction_box[2] - prediction_box[0])
        * (prediction_box[3] - prediction_box[1])
    )
    target_area = (
        (target_box[2] - target_box[0])
        * (target_box[3] - target_box[1])
    )
    union = prediction_area + target_area - intersection
    return {
        'iou': 0.0 if union <= 0 else intersection / union,
        'gt_coverage_by_prediction': intersection / target_area,
        'prediction_coverage_by_gt': intersection / prediction_area,
    }


def _parse_failure_from_metadata(metadata: Mapping[str, Any]) -> bool:
    if metadata.get('parse_failed'):
        return True
    remote = metadata.get('remote_metadata')
    return isinstance(remote, Mapping) and bool(remote.get('parse_failed'))


def evaluate_task(
        task: Mapping[str, Any],
        grounder_backend: Any) -> Dict[str, Any]:
    """Evaluate one canonical object query through a Grounder backend."""

    base = dict(task)
    started = time.perf_counter()
    try:
        result = grounder_backend.ground(GroundingRequest(
            sample_id=str(task['sample_id']),
            grounding_step=int(task['target_index']) + 1,
            object_reference=str(task['object_reference']),
            visual=VisualInput(image_path=str(task['image_path'])),
        ))
    except ExpertUnavailableError as error:
        metadata = dict(error.metadata)
        # Transport/schema/image failures are infrastructure failures.  They
        # must abort rather than make a Grounder look less accurate.  A normal
        # worker response with available=false is a measured model failure.
        if metadata.get('remote_failure'):
            raise
        return {
            **base,
            'status': 'grounder_unavailable',
            'iou': 0.0,
            'gt_coverage_by_prediction': 0.0,
            'prediction_coverage_by_gt': 0.0,
            'prediction_bbox_original_pixel_xyxy': None,
            'prediction_bbox_vocot_normalized_padded_xyxy': None,
            'grounder_source': metadata.get('remote_grounder_source'),
            'grounder_confidence': None,
            'prediction_confidence_available': False,
            'grounder_metadata': metadata,
            'parse_failed': _parse_failure_from_metadata(metadata),
            'error_type': type(error).__name__,
            'error': str(error),
            'latency_seconds': time.perf_counter() - started,
        }

    metadata = dict(result.metadata)
    prediction_pixel = metadata.get('bbox_original_pixel_xyxy')
    width, height = (int(value) for value in task['image_size'])
    prediction_pixel = _validate_xyxy(
        prediction_pixel,
        width=width,
        height=height,
        field_name='prediction_bbox_original_pixel_xyxy',
    )
    overlap = _overlap_metrics(
        prediction_pixel,
        task['gt_bbox_original_pixel_xyxy'],
    )
    return {
        **base,
        'status': 'ok',
        **overlap,
        'prediction_bbox_original_pixel_xyxy': list(prediction_pixel),
        'prediction_bbox_vocot_normalized_padded_xyxy': list(result.bbox),
        'grounder_source': result.source,
        'grounder_confidence': float(result.confidence),
        'prediction_confidence_available': bool(
            metadata.get('prediction_confidence_available')
        ),
        'grounder_metadata': metadata,
        'parse_failed': False,
        'error_type': None,
        'error': None,
        'latency_seconds': time.perf_counter() - started,
    }


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else sum(values) / len(values)


def _basic_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    successful = [record for record in records if record.get('status') == 'ok']
    overall_ious = [float(record.get('iou') or 0.0) for record in records]
    successful_ious = [float(record['iou']) for record in successful]
    latency = [
        float(record['latency_seconds']) for record in records
        if record.get('latency_seconds') is not None
    ]
    return {
        'target_request_count': total,
        'successful_request_count': len(successful),
        'failed_request_count': total - len(successful),
        'availability_rate': (
            None if total == 0 else len(successful) / total
        ),
        'overall_miou': _safe_mean(overall_ious),
        'overall_median_iou': (
            None if not overall_ious else statistics.median(overall_ious)
        ),
        'successful_only_miou': _safe_mean(successful_ious),
        'successful_only_median_iou': (
            None if not successful_ious
            else statistics.median(successful_ious)
        ),
        'iou_recall': {
            f'{threshold:.1f}': (
                None if total == 0 else sum(
                    iou >= threshold for iou in overall_ious
                ) / total
            )
            for threshold in IOU_THRESHOLDS
        },
        'mean_gt_coverage_by_prediction': _safe_mean([
            float(record.get('gt_coverage_by_prediction') or 0.0)
            for record in records
        ]),
        'mean_prediction_coverage_by_gt': _safe_mean([
            float(record.get('prediction_coverage_by_gt') or 0.0)
            for record in records
        ]),
        'mean_latency_seconds': _safe_mean(latency),
    }


def summarize_records(
        records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize target-micro accuracy without dropping failed requests."""

    records = list(records)
    summary = _basic_metrics(records)
    status_counts = Counter(
        str(record.get('status') or 'unknown') for record in records
    )
    error_counts = Counter(
        str(record.get('error_type') or 'unknown')
        for record in records
        if record.get('status') != 'ok'
    )
    summary.update({
        'metric_unit': 'annotated_object_target',
        'failure_iou_policy': 'zero',
        'status_counts': dict(sorted(status_counts.items())),
        'parse_failure_count': sum(
            bool(record.get('parse_failed')) for record in records
        ),
        'error_type_counts': dict(sorted(error_counts.items())),
    })

    by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_size: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_coverage: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get('sample_id') is not None:
            by_sample[str(record['sample_id'])].append(record)
        if record.get('category') is not None:
            by_category[str(record['category'])].append(record)
        if record.get('gt_size_bucket') is not None:
            by_size[str(record['gt_size_bucket'])].append(record)
        complete = record.get('has_complete_question_target_coverage')
        if complete is not None:
            by_coverage['complete' if complete else 'partial'].append(record)

    sample_ious = [
        sum(float(record.get('iou') or 0.0) for record in sample_records)
        / len(sample_records)
        for sample_records in by_sample.values()
    ]
    summary['sample_macro'] = {
        'sample_count': len(by_sample),
        'mean_sample_iou': _safe_mean(sample_ious),
        'median_sample_iou': (
            None if not sample_ious else statistics.median(sample_ious)
        ),
    }
    summary['by_category'] = {
        key: _basic_metrics(value)
        for key, value in sorted(by_category.items())
    }
    category_metrics = list(summary['by_category'].values())
    summary['category_macro'] = {
        'category_count': len(category_metrics),
        'mean_category_miou': _safe_mean([
            metric['overall_miou'] for metric in category_metrics
            if metric['overall_miou'] is not None
        ]),
        'mean_category_availability_rate': _safe_mean([
            metric['availability_rate'] for metric in category_metrics
            if metric['availability_rate'] is not None
        ]),
        'mean_category_iou_recall': {
            f'{threshold:.1f}': _safe_mean([
                metric['iou_recall'][f'{threshold:.1f}']
                for metric in category_metrics
                if metric['iou_recall'][f'{threshold:.1f}'] is not None
            ])
            for threshold in IOU_THRESHOLDS
        },
    }
    summary['by_gt_size'] = {
        key: _basic_metrics(value)
        for key, value in sorted(by_size.items())
    }
    summary['by_question_target_coverage'] = {
        key: _basic_metrics(value)
        for key, value in sorted(by_coverage.items())
    }
    return summary


def filter_pending_tasks(
        tasks: Sequence[Mapping[str, Any]],
        existing_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    completed = {
        str(record['task_id'])
        for record in existing_records
        if record.get('task_id') is not None
        and record.get('status') in COMPLETED_STATUSES
    }
    return [dict(task) for task in tasks if str(task['task_id']) not in completed]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _method_and_setting(args: argparse.Namespace) -> Tuple[str, str]:
    if args.method_name:
        method = args.method_name
    elif args.backend == 'qwen25_vl':
        method = 'qwen25_vl_7b'
    else:
        method = 'grounding_dino_base'
    if args.setting:
        setting = args.setting
    elif args.backend == 'qwen25_vl':
        setting = args.prompt_protocol
    else:
        setting = (
            f'box_{args.dino_box_threshold:g}'
            f'__text_{args.dino_text_threshold:g}'
        ).replace('.', 'p')
    return method, setting


def resolve_run_layout(args: argparse.Namespace) -> RunLayout:
    method, setting = _method_and_setting(args)
    return create_run_layout(
        dataset='vstar',
        split=args.run_split,
        study='grounder_accuracy',
        method=method,
        setting=setting,
        run_id=args.run_id,
        output=args.output,
        output_root=args.output_root,
    )


def _worker_command(args: argparse.Namespace) -> List[str]:
    common = [
        str(Path(args.worker_python).resolve()),
        '-u',
        '-m',
    ]
    if args.backend == 'qwen25_vl':
        command = common + [
            'grounding_control.workers.qwen_grounder',
            '--model-path', str(Path(args.model_path).resolve()),
            '--device', 'cuda:0',
            '--dtype', args.dtype,
            '--max-new-tokens', str(args.max_new_tokens),
            '--min-pixels', str(args.min_pixels),
            '--attn-implementation', args.attn_implementation,
            '--prompt-protocol', args.prompt_protocol,
            '--boundary-tolerance-pixels', str(
                args.boundary_tolerance_pixels
            ),
        ]
        if args.max_pixels is not None:
            command.extend(['--max-pixels', str(args.max_pixels)])
        return command
    return common + [
        'grounding_control.workers.dino_grounder',
        '--model-path', str(Path(args.model_path).resolve()),
        '--device', 'cuda:0',
        '--dtype', args.dtype,
        '--box-threshold', str(args.dino_box_threshold),
        '--text-threshold', str(args.dino_text_threshold),
        '--top-k-log', str(args.dino_top_k_log),
    ]


def _experiment_signature(
        args: argparse.Namespace,
        *,
        selected_tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    method, setting = _method_and_setting(args)
    payload = {
        'schema': EXPERIMENT_SIGNATURE_SCHEMA,
        'benchmark': {
            'dataset': 'vstar',
            'split': args.run_split,
            'oracle_boxes_path': str(Path(args.oracle_boxes_path).resolve()),
            'oracle_boxes_sha256': _sha256_file(
                Path(args.oracle_boxes_path).resolve()
            ),
            'image_dir': str(Path(args.image_dir).resolve()),
            'reference_protocol': 'canonical_vstar_target_object',
            'target_request_count': len(selected_tasks),
            'first_task_id': (
                None if not selected_tasks else selected_tasks[0]['task_id']
            ),
            'last_task_id': (
                None if not selected_tasks else selected_tasks[-1]['task_id']
            ),
        },
        'grounder': {
            'backend': args.backend,
            'method': method,
            'setting': setting,
            'worker_python': str(Path(args.worker_python).resolve()),
            'model_path': str(Path(args.model_path).resolve()),
            'source': EXPECTED_SOURCES[args.backend],
            'device': 'cuda:0',
            'dtype': args.dtype,
            'timeout_seconds': float(args.worker_timeout),
            'qwen': {
                'max_new_tokens': int(args.max_new_tokens),
                'min_pixels': int(args.min_pixels),
                'max_pixels': (
                    None if args.max_pixels is None else int(args.max_pixels)
                ),
                'attn_implementation': args.attn_implementation,
                'prompt_protocol': args.prompt_protocol,
                'boundary_tolerance_pixels': float(
                    args.boundary_tolerance_pixels
                ),
            } if args.backend == 'qwen25_vl' else None,
            'grounding_dino': {
                'box_threshold': float(args.dino_box_threshold),
                'text_threshold': float(args.dino_text_threshold),
                'top_k_log': int(args.dino_top_k_log),
            } if args.backend == 'grounding_dino' else None,
        },
        'selection': {
            'start_target_index': int(args.start_target_index),
            'max_targets': args.max_targets,
            'target_id': args.target_id,
        },
        'metrics': {
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'iou_geometry': 'continuous_no_plus_one',
            'failed_request_iou': 0.0,
            'thresholds': list(IOU_THRESHOLDS),
        },
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    ).encode('utf-8')
    return {
        'schema': EXPERIMENT_SIGNATURE_SCHEMA,
        'sha256': hashlib.sha256(serialized).hexdigest(),
        'parameters': payload,
    }


def _atomic_write_jsonl(
        path: Path,
        records: Iterable[Mapping[str, Any]]) -> None:
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


def _latest_by_task_id(
        records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for record in records:
        task_id = str(record.get('task_id') or '')
        if not task_id:
            raise ValueError('result record is missing task_id')
        if task_id not in latest:
            order.append(task_id)
        latest[task_id] = dict(record)
    return [latest[task_id] for task_id in order]


def _validate_resume_signature(
        records: Sequence[Mapping[str, Any]],
        signature: Mapping[str, Any]) -> None:
    expected = signature['sha256']
    mismatches = {
        record.get('experiment_signature_sha256')
        for record in records
        if record.get('experiment_signature_sha256') != expected
    }
    if mismatches:
        raise ValueError(
            'existing results use a different experiment signature; '
            'choose a new --run-id or use --no-resume'
        )


def _select_tasks(
        tasks: Sequence[Mapping[str, Any]],
        args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.target_id is not None:
        selected = [
            dict(task) for task in tasks
            if task['task_id'] == args.target_id
        ]
        if not selected:
            raise ValueError(f'unknown --target-id: {args.target_id!r}')
        return selected
    selected = list(tasks[args.start_target_index:])
    if args.max_targets is not None:
        selected = selected[:args.max_targets]
    return [dict(task) for task in selected]


def _warmup_grounder(
        backend: RemoteGrounderBackend,
        task: Mapping[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        result = backend.ground(GroundingRequest(
            sample_id=f'{task["sample_id"]}:environment_warmup',
            grounding_step=0,
            object_reference=str(task['object_reference']),
            visual=VisualInput(image_path=str(task['image_path'])),
        ))
        return {
            'purpose': 'environment_check_only',
            'affects_metrics': False,
            'task_id': task['task_id'],
            'status': 'available',
            'source': result.source,
            'bbox_original_pixel_xyxy': result.metadata.get(
                'bbox_original_pixel_xyxy'
            ),
            'latency_seconds': time.perf_counter() - started,
        }
    except ExpertUnavailableError as error:
        if error.metadata.get('remote_failure'):
            raise
        return {
            'purpose': 'environment_check_only',
            'affects_metrics': False,
            'task_id': task['task_id'],
            'status': 'model_unavailable',
            'error': str(error),
            'metadata': dict(error.metadata),
            'latency_seconds': time.perf_counter() - started,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=SUPPORTED_BACKENDS,
                        default='qwen25_vl')
    parser.add_argument('--oracle-boxes-path', default=DEFAULT_ORACLE_BOXES)
    parser.add_argument('--image-dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--worker-python', default=DEFAULT_WORKER_PYTHON)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--dtype', default=None)
    parser.add_argument('--worker-timeout', type=float, default=600.0)

    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--min-pixels', type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument(
        '--max-pixels', type=int, default=DEFAULT_MAX_PIXELS,
        help='Optional explicit Qwen pixel cap; default keeps source resolution.',
    )
    parser.add_argument('--attn-implementation', default='sdpa')
    parser.add_argument('--prompt-protocol',
                        choices=GROUNDING_PROMPT_PROTOCOLS,
                        default=DEFAULT_GROUNDING_PROMPT_PROTOCOL)
    parser.add_argument('--boundary-tolerance-pixels', type=float, default=1.0)

    parser.add_argument('--dino-box-threshold', type=float, default=0.3)
    parser.add_argument('--dino-text-threshold', type=float, default=0.25)
    parser.add_argument('--dino-top-k-log', type=int, default=20)

    parser.add_argument('--run-split', default='full_238')
    parser.add_argument('--method-name', default=None)
    parser.add_argument('--setting', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--start-target-index', type=int, default=0)
    parser.add_argument('--max-targets', type=int, default=None)
    parser.add_argument('--target-id', default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def _normalize_and_validate_args(args: argparse.Namespace) -> None:
    if args.model_path is None:
        args.model_path = (
            DEFAULT_QWEN_MODEL
            if args.backend == 'qwen25_vl'
            else DEFAULT_DINO_MODEL
        )
    if args.dtype is None:
        args.dtype = 'bfloat16' if args.backend == 'qwen25_vl' else 'float32'
    if not str(args.gpu).isdigit():
        raise ValueError('--gpu must be one physical GPU index')
    if args.worker_timeout <= 0:
        raise ValueError('--worker-timeout must be positive')
    if args.start_target_index < 0:
        raise ValueError('--start-target-index must be non-negative')
    if args.max_targets is not None and args.max_targets <= 0:
        raise ValueError('--max-targets must be positive')
    if args.target_id is not None and (
            args.start_target_index != 0 or args.max_targets is not None):
        raise ValueError(
            '--target-id cannot be combined with start/max target selection'
        )
    if args.max_new_tokens <= 0:
        raise ValueError('--max-new-tokens must be positive')
    if args.min_pixels <= 0:
        raise ValueError('--min-pixels must be positive')
    if args.max_pixels is not None and args.max_pixels <= 0:
        raise ValueError('--max-pixels must be positive when provided')
    if args.max_pixels is not None and args.min_pixels > args.max_pixels:
        raise ValueError('--min-pixels must not exceed --max-pixels')
    if args.boundary_tolerance_pixels < 0:
        raise ValueError('--boundary-tolerance-pixels must be non-negative')
    for value, name in (
        (args.dino_box_threshold, '--dino-box-threshold'),
        (args.dino_text_threshold, '--dino-text-threshold'),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be in [0, 1]')
    if args.dino_top_k_log <= 0:
        raise ValueError('--dino-top-k-log must be positive')
    if not Path(args.worker_python).is_file():
        raise FileNotFoundError(args.worker_python)
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(args.model_path)


def main() -> None:
    args = parse_args()
    _normalize_and_validate_args(args)
    all_tasks = load_grounding_tasks(
        args.oracle_boxes_path,
        args.image_dir,
    )
    selected_tasks = _select_tasks(all_tasks, args)
    if not selected_tasks:
        raise ValueError('target selection is empty')

    layout = resolve_run_layout(args)
    layout.ensure_run_directories()
    signature = _experiment_signature(args, selected_tasks=selected_tasks)
    output_path = layout.results_path
    existing = (
        []
        if args.no_resume or not output_path.is_file()
        else _latest_by_task_id(_read_jsonl(output_path))
    )
    if existing:
        _validate_resume_signature(existing, signature)
    pending = filter_pending_tasks(selected_tasks, existing)
    selected_ids = {task['task_id'] for task in selected_tasks}
    retained = [
        record for record in existing
        if record.get('task_id') in selected_ids
    ]
    _atomic_write_jsonl(output_path, retained)

    command = _worker_command(args)
    previous_worker = None
    if not pending and layout.config_path.is_file():
        try:
            previous_config = json.loads(
                layout.config_path.read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            previous_config = {}
        previous_signature = previous_config.get('experiment_signature') or {}
        if previous_signature.get('sha256') == signature['sha256']:
            previous_worker = previous_config.get('worker')
    config = {
        'benchmark_schema': BENCHMARK_SCHEMA,
        'arguments': vars(args),
        'experiment_signature': signature,
        'inputs': {
            'oracle_boxes_path': str(
                Path(args.oracle_boxes_path).resolve()
            ),
            'image_dir': str(Path(args.image_dir).resolve()),
            'full_sample_count': len({task['sample_id'] for task in all_tasks}),
            'full_target_count': len(all_tasks),
            'selected_sample_count': len({
                task['sample_id'] for task in selected_tasks
            }),
            'selected_target_count': len(selected_tasks),
        },
        'grounder': signature['parameters']['grounder'],
        'worker': previous_worker or {
            'command': command,
            'physical_gpu': str(args.gpu),
            'ping': None,
            'warmup': None,
        },
    }
    write_run_config(layout, config)
    write_run_status(
        layout,
        'running',
        completed_targets=len(selected_tasks) - len(pending),
        pending_targets=len(pending),
        experiment_signature_sha256=signature['sha256'],
    )
    print(
        f'Run id: {layout.run_id}; backend={args.backend}; '
        f'targets={len(selected_tasks)}; pending={len(pending)}',
        flush=True,
    )
    print(f'Output: {output_path}', flush=True)

    client = None
    runtime_error_count = 0
    try:
        if pending:
            client = PersistentJsonlWorkerClient(
                command,
                cwd=str(PROJECT_ROOT),
                env={'CUDA_VISIBLE_DEVICES': str(args.gpu)},
                timeout=args.worker_timeout,
                stderr=None,
                start=False,
            )
            client.start()
            ping = client.ping(timeout=min(30.0, args.worker_timeout))
            config['worker']['ping'] = ping
            if not ping.get('configured'):
                raise RuntimeError(f'Grounder worker is not configured: {ping}')
            backend = RemoteGrounderBackend(
                client,
                timeout=args.worker_timeout,
                source=EXPECTED_SOURCES[args.backend],
            )
            warmup = _warmup_grounder(backend, pending[0])
            config['worker']['warmup'] = warmup
            write_run_config(layout, config)
            print(
                f'Worker warm-up: {warmup["status"]}; '
                f'GPU={args.gpu}',
                flush=True,
            )

            with output_path.open('a', encoding='utf-8') as handle:
                for task in tqdm(pending, desc='VStar Grounder accuracy'):
                    caught_error = None
                    try:
                        record = evaluate_task(task, backend)
                    except Exception as error:
                        caught_error = error
                        record = {
                            **task,
                            'status': 'runtime_error',
                            'iou': 0.0,
                            'gt_coverage_by_prediction': 0.0,
                            'prediction_coverage_by_gt': 0.0,
                            'prediction_bbox_original_pixel_xyxy': None,
                            'prediction_bbox_vocot_normalized_padded_xyxy': None,
                            'grounder_source': EXPECTED_SOURCES[args.backend],
                            'parse_failed': False,
                            'error_type': type(error).__name__,
                            'error': str(error),
                        }
                        runtime_error_count += 1
                    record['experiment_signature_sha256'] = signature['sha256']
                    handle.write(json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + '\n')
                    handle.flush()
                    if args.verbose:
                        tqdm.write(
                            f'[{record["task_id"]}] '
                            f'{record["object_reference"]!r} '
                            f'status={record["status"]} '
                            f'iou={record["iou"]:.4f}'
                        )
                    if record['status'] == 'runtime_error':
                        # Remote transport/schema failures invalidate every
                        # subsequent request, so never silently turn them into
                        # localization errors.  --fail-fast also stops on any
                        # unexpected per-target runtime error.
                        if args.fail_fast or isinstance(
                                caught_error, ExpertUnavailableError):
                            raise RuntimeError(
                                f'Grounder runtime failed for '
                                f'{record["task_id"]}: {record["error"]}'
                            ) from caught_error

        final_records = _latest_by_task_id(_read_jsonl(output_path))
        final_records = [
            record for record in final_records
            if record.get('task_id') in selected_ids
        ]
        order = {
            task['task_id']: index
            for index, task in enumerate(selected_tasks)
        }
        final_records.sort(key=lambda record: order[record['task_id']])
        _atomic_write_jsonl(output_path, final_records)
        summary = summarize_records(final_records)
        summary.update({
            'benchmark_schema': BENCHMARK_SCHEMA,
            'run_id': layout.run_id,
            'backend': args.backend,
            'method': _method_and_setting(args)[0],
            'setting': _method_and_setting(args)[1],
            'experiment_signature_sha256': signature['sha256'],
            'selected_target_count': len(selected_tasks),
            'selected_sample_count': len({
                task['sample_id'] for task in selected_tasks
            }),
            'complete_measurement_count': sum(
                record.get('status') in COMPLETED_STATUSES
                for record in final_records
            ),
            'runtime_error_count': sum(
                record.get('status') == 'runtime_error'
                for record in final_records
            ),
            'worker': config['worker'],
        })
        layout.summary_path.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + '\n',
            encoding='utf-8',
        )
        terminal_status = (
            'completed'
            if summary['runtime_error_count'] == 0
            else 'completed_with_errors'
        )
        write_run_status(
            layout,
            terminal_status,
            completed_targets=summary['complete_measurement_count'],
            runtime_error_targets=summary['runtime_error_count'],
            summary_path=str(layout.summary_path),
            experiment_signature_sha256=signature['sha256'],
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f'Per-target results: {output_path}', flush=True)
        print(f'Summary: {layout.summary_path}', flush=True)
    except KeyboardInterrupt:
        write_run_status(
            layout,
            'interrupted',
            runtime_error_targets=runtime_error_count,
            experiment_signature_sha256=signature['sha256'],
        )
        raise
    except BaseException as error:
        write_run_status(
            layout,
            'failed',
            runtime_error_targets=runtime_error_count,
            error=f'{type(error).__name__}: {error}',
            experiment_signature_sha256=signature['sha256'],
        )
        raise
    finally:
        if client is not None:
            client.close()


if __name__ == '__main__':
    main()
