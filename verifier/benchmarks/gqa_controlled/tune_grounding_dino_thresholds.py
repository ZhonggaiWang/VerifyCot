"""Tune Grounding DINO's box threshold on the controlled GQA dev split.

The current geometry backend selects the highest-score detector box without
using the candidate.  Therefore increasing ``box_threshold`` can only turn a
sample's fixed top-1 localization into a no-detection failure; it cannot
change which box wins.  We exploit that property by running Grounding DINO
once at the smallest requested threshold and evaluating all larger thresholds
offline.

``text_threshold`` only changes the phrase label returned by the Hugging Face
post-processor.  It does not filter boxes or enter the geometry router, so it
is fixed and audited rather than searched in this benchmark.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .metrics import compute_routing_metrics


DEFAULT_BENCHMARK = Path(
    'output/verifier_benchmark/gqa_controlled/v1/benchmark.jsonl'
)
DEFAULT_MODEL = Path('weights/grounding-dino-base')
DEFAULT_OUTPUT_DIR = Path(
    'output/verifier_benchmark/gqa_controlled/'
    'routing_grounding_geometry/grounding_dino_base/dev_threshold_search'
)
DEFAULT_BOX_THRESHOLDS = (
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
)
SELECTION_METRICS = ('macro_f1', 'accuracy')


def parse_box_thresholds(value: str) -> List[float]:
    """Parse a deterministic, increasing threshold grid."""

    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError(
            'box thresholds must be a comma-separated list'
        )
    thresholds: List[float] = []
    for item in value.split(','):
        try:
            threshold = float(item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f'invalid box threshold: {item!r}'
            ) from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(
                f'box threshold must be finite and in [0, 1]: {threshold}'
            )
        thresholds.append(threshold)
    return sorted(set(thresholds))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'invalid JSONL row at {path}:{line_number}: {error}'
                ) from error
    return rows


def _summary_path(results_path: Path) -> Path:
    return results_path.with_suffix('.summary.json')


def _selected_score_and_action(
        row: Mapping[str, Any],
) -> tuple:
    metadata = row.get('verifier_metadata')
    if not isinstance(metadata, Mapping):
        return None, None
    score = metadata.get('selected_grounding_score')
    geometry = metadata.get('geometry')
    action = geometry.get('action') if isinstance(geometry, Mapping) else None
    if not (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
    ):
        return None, None
    return float(score), action


def records_at_box_threshold(
        cached_rows: Iterable[Mapping[str, Any]],
        box_threshold: float,
) -> List[Dict[str, Any]]:
    """Replay top-1 detection gating at one larger box threshold."""

    replayed: List[Dict[str, Any]] = []
    for source in cached_rows:
        row = dict(source)
        score, action = _selected_score_and_action(source)
        # HF Grounding DINO uses ``scores > threshold`` in post-processing.
        predicted = action if score is not None and score > box_threshold else None
        row['predicted_routing_status'] = predicted
        row['confidence'] = score if predicted is not None else None
        row['correct'] = predicted == row.get('expected_routing_status')
        row['parse_failed'] = predicted is None
        replayed.append(row)
    return replayed


def evaluate_box_thresholds(
        cached_rows: Sequence[Mapping[str, Any]],
        box_thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """Compute the existing end-to-end routing metrics for every threshold."""

    evaluations: List[Dict[str, Any]] = []
    for threshold in box_thresholds:
        metrics = compute_routing_metrics(
            records_at_box_threshold(cached_rows, float(threshold))
        )
        four_way = metrics['four_way']
        evaluations.append({
            'box_threshold': float(threshold),
            'accuracy': four_way['accuracy'],
            'macro_precision': four_way['macro_precision'],
            'macro_recall': four_way['macro_recall'],
            'macro_f1': four_way['macro_f1'],
            'localization_success_rate': metrics['parse_success_rate'],
            'localization_failure_count': (
                metrics['runtime_or_parse_failure_count']
            ),
            'correct': four_way['correct'],
            'total': metrics['total'],
            'per_class': four_way['per_class'],
            'confusion_matrix': four_way['confusion_matrix'],
        })
    return evaluations


def select_best_evaluation(
        evaluations: Sequence[Mapping[str, Any]],
        selection_metric: str = 'macro_f1',
) -> Dict[str, Any]:
    """Select by metric, then accuracy, coverage, and lower threshold."""

    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f'selection_metric must be one of {SELECTION_METRICS}'
        )
    if not evaluations:
        raise ValueError('at least one threshold evaluation is required')
    return dict(max(
        evaluations,
        key=lambda row: (
            float(row[selection_metric]),
            float(row['accuracy']),
            float(row['localization_success_rate']),
            -float(row['box_threshold']),
        ),
    ))


def _validate_cached_run(
        summary: Mapping[str, Any],
        benchmark: Path,
        model_path: Path,
        cache_threshold: float,
        text_threshold: float,
) -> None:
    expected = {
        'backend': 'grounding_dino_geometry_router_raw_image',
        'geometry_backend': 'grounding_dino',
        'split': 'dev',
        'limit': None,
        'dino_box_threshold': cache_threshold,
        'dino_text_threshold': text_threshold,
    }
    for key, value in expected.items():
        observed = summary.get(key)
        if isinstance(value, float):
            matches = (
                isinstance(observed, (int, float))
                and math.isclose(float(observed), value, abs_tol=1e-12)
            )
        else:
            matches = observed == value
        if not matches:
            raise ValueError(
                f'incompatible cached inference: {key}={observed!r}, '
                f'expected {value!r}; use --force-inference'
            )
    if Path(str(summary.get('benchmark'))) != benchmark:
        raise ValueError(
            'cached benchmark path differs; use --force-inference'
        )
    if Path(str(summary.get('model_path'))) != model_path:
        raise ValueError(
            'cached model path differs; use --force-inference'
        )


def _run_dev_inference(args: argparse.Namespace, results_path: Path) -> None:
    command = [
        sys.executable,
        '-u',
        '-m',
        'verifier.benchmarks.gqa_controlled.evaluator',
        '--benchmark',
        str(args.benchmark),
        '--model-path',
        str(args.model_path),
        '--task-mode',
        'routing_grounding_geometry',
        '--geometry-backend',
        'grounding_dino',
        '--grounding-image-mode',
        'raw_image',
        '--grounding-accept-iou',
        str(args.grounding_accept_iou),
        '--grounding-containment',
        str(args.grounding_containment),
        '--dino-box-threshold',
        str(min(args.box_thresholds)),
        '--dino-text-threshold',
        str(args.text_threshold),
        '--dino-dtype',
        args.dtype,
        '--device',
        args.device,
        '--split',
        'dev',
        '--output',
        str(results_path),
        '--fail-fast',
    ]
    if args.force_inference:
        command.append('--overwrite')
    print('Running one low-threshold dev inference:', flush=True)
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)


def _write_outputs(
        output_dir: Path,
        payload: Mapping[str, Any],
        evaluations: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'threshold_search.json'
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    best_path = output_dir / 'best_config.json'
    with best_path.open('w', encoding='utf-8') as handle:
        json.dump(payload['best'], handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    csv_path = output_dir / 'threshold_search.csv'
    fields = (
        'box_threshold',
        'accuracy',
        'macro_f1',
        'macro_precision',
        'macro_recall',
        'localization_success_rate',
        'localization_failure_count',
        'correct',
        'total',
    )
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for evaluation in evaluations:
            writer.writerow({
                field: evaluation[field] for field in fields
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run Grounding DINO once on controlled GQA dev and select the '
            'best top-1 box confidence threshold offline.'
        )
    )
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--box-thresholds',
        type=parse_box_thresholds,
        default=list(DEFAULT_BOX_THRESHOLDS),
        help='comma-separated dev grid; default: 0.10,...,0.50',
    )
    parser.add_argument(
        '--text-threshold',
        type=float,
        default=0.25,
        help=(
            'fixed phrase-label threshold; it does not affect box selection '
            'or routing in the current backend'
        ),
    )
    parser.add_argument(
        '--selection-metric',
        choices=SELECTION_METRICS,
        default='macro_f1',
    )
    parser.add_argument('--grounding-accept-iou', type=float, default=0.5)
    parser.add_argument('--grounding-containment', type=float, default=0.7)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dtype',
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
    )
    parser.add_argument(
        '--force-inference',
        action='store_true',
        help='discard and regenerate the low-threshold dev cache',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.box_thresholds:
        raise ValueError('--box-thresholds must not be empty')
    if not 0.0 <= args.text_threshold <= 1.0:
        raise ValueError('--text-threshold must be in [0, 1]')
    if not 0.0 < args.grounding_accept_iou <= 1.0:
        raise ValueError('--grounding-accept-iou must be in (0, 1]')
    if not 0.0 < args.grounding_containment <= 1.0:
        raise ValueError('--grounding-containment must be in (0, 1]')

    cache_dir = args.output_dir / 'inference_cache'
    results_path = cache_dir / 'results.jsonl'
    summary_path = _summary_path(results_path)
    if args.force_inference or not results_path.exists():
        if summary_path.exists() and not args.force_inference:
            raise FileExistsError(
                f'cache summary exists without results: {summary_path}; '
                'use --force-inference'
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        _run_dev_inference(args, results_path)
    elif not summary_path.exists():
        raise FileNotFoundError(
            f'incomplete inference cache lacks summary: {summary_path}; '
            'use --force-inference'
        )

    with summary_path.open('r', encoding='utf-8') as handle:
        inference_summary = json.load(handle)
    _validate_cached_run(
        inference_summary,
        args.benchmark,
        args.model_path,
        min(args.box_thresholds),
        args.text_threshold,
    )
    cached_rows = _read_jsonl(results_path)
    expected_count = inference_summary.get('selected_example_count')
    if len(cached_rows) != expected_count:
        raise ValueError(
            f'inference cache has {len(cached_rows)} rows, expected '
            f'{expected_count}; use --force-inference'
        )

    evaluations = evaluate_box_thresholds(
        cached_rows,
        args.box_thresholds,
    )
    best_metrics = select_best_evaluation(
        evaluations,
        args.selection_metric,
    )
    best = {
        'box_threshold': best_metrics['box_threshold'],
        'text_threshold': args.text_threshold,
        'selection_metric': args.selection_metric,
        'selection_metric_value': best_metrics[args.selection_metric],
        'accuracy': best_metrics['accuracy'],
        'macro_f1': best_metrics['macro_f1'],
        'localization_success_rate': (
            best_metrics['localization_success_rate']
        ),
        'grounding_accept_iou': args.grounding_accept_iou,
        'grounding_containment': args.grounding_containment,
        'model_path': str(args.model_path),
        'benchmark': str(args.benchmark),
        'split': 'dev',
    }
    payload = {
        'method': (
            'single_low_threshold_inference_then_offline_top1_score_gating'
        ),
        'selection_rule': (
            f'maximize {args.selection_metric}, then accuracy, localization '
            'success, then prefer the lower threshold'
        ),
        'text_threshold_note': (
            'text_threshold affects decoded phrase labels only; the current '
            'top-1 geometry router does not use labels, so it is fixed'
        ),
        'inference_cache': str(results_path),
        'cache_box_threshold': min(args.box_thresholds),
        'box_thresholds': list(args.box_thresholds),
        'best': best,
        'evaluations': evaluations,
    }
    _write_outputs(args.output_dir, payload, evaluations)

    print('\nDev threshold search:', flush=True)
    print(
        'threshold  accuracy  macro_f1  localization_success',
        flush=True,
    )
    for evaluation in evaluations:
        marker = (
            '  <-- best'
            if evaluation['box_threshold'] == best['box_threshold']
            else ''
        )
        print(
            f"{evaluation['box_threshold']:>9.2f}  "
            f"{evaluation['accuracy'] * 100:>7.2f}%  "
            f"{evaluation['macro_f1'] * 100:>7.2f}%  "
            f"{evaluation['localization_success_rate'] * 100:>19.2f}%"
            f'{marker}',
            flush=True,
        )
    print(f"\nBest config: {args.output_dir / 'best_config.json'}")
    print(f"Full search: {args.output_dir / 'threshold_search.json'}")
    print(f"CSV table: {args.output_dir / 'threshold_search.csv'}")


if __name__ == '__main__':
    main()
