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
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ...run_paths import (
    RunLayout,
    create_exact_output_layout,
    create_run_layout,
    write_run_config,
    write_run_status,
)
from .metrics import compute_routing_metrics


DEFAULT_BENCHMARK = Path(
    'output/verifier_benchmark/gqa_controlled/v1/benchmark.jsonl'
)
DEFAULT_MODEL = Path('weights/grounding-dino-base')
DEFAULT_OUTPUT_ROOT = Path('output')
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


def _path_token(value: Any) -> str:
    token = str(value).strip().lower().replace('.', 'p')
    token = re.sub(r'[^a-z0-9]+', '_', token).strip('_')
    if not token:
        raise ValueError('threshold-search path token must not be empty')
    return token


def _float_token(value: float) -> str:
    return _path_token(format(float(value), 'g'))


def _grounding_dino_method(model_path: Path) -> str:
    model_slug = _path_token(Path(model_path).name)
    for variant in ('tiny', 'base', 'large'):
        if re.search(r'(?:^|_){}(?:_|$)'.format(variant), model_slug):
            return 'grounding_dino_{}'.format(variant)
    return 'grounding_dino__{}'.format(model_slug)


def threshold_search_run_identity(
        args: argparse.Namespace) -> Dict[str, str]:
    return {
        'dataset': 'verifier_benchmark',
        'split': 'gqa_controlled_v1_dev',
        'study': 'threshold_search',
        'method': _grounding_dino_method(args.model_path),
        'setting': 'top1_score_gating__{}__iou_{}'.format(
            args.selection_metric,
            _float_token(args.grounding_accept_iou),
        ),
    }


def resolve_threshold_search_layout(args: argparse.Namespace) -> RunLayout:
    """Use canonical defaults or preserve an exact legacy output directory."""
    identity = threshold_search_run_identity(args)
    canonical = create_run_layout(
        **identity,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    if args.output_dir is None:
        return canonical
    return create_exact_output_layout(
        **identity,
        run_id=canonical.run_id,
        output=args.output_dir / 'threshold_search.json',
    )


def resolve_inference_cache_dir(
        args: argparse.Namespace,
        layout: RunLayout) -> Path:
    """Resolve a reusable inference cache without changing legacy layout."""
    if args.cache_dir is not None:
        return args.cache_dir
    if args.output_dir is not None:
        return layout.run_dir / 'inference_cache'
    cache_parameters = {
        'benchmark': str(args.benchmark),
        'model_path': str(args.model_path),
        'cache_box_threshold': min(args.box_thresholds),
        'text_threshold': args.text_threshold,
        'grounding_accept_iou': args.grounding_accept_iou,
        'grounding_containment': args.grounding_containment,
        'dtype': args.dtype,
        'protocol': 'top1_geometry_candidate_hidden',
    }
    digest = hashlib.sha256(json.dumps(
        cache_parameters,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')).hexdigest()[:12]
    setting = 'box_{}__text_{}__{}'.format(
        _float_token(min(args.box_thresholds)),
        _float_token(args.text_threshold),
        digest,
    )
    return (
        Path(args.output_root)
        / 'verifier_benchmark'
        / 'cache'
        / 'gqa_controlled_v1_dev'
        / _grounding_dino_method(args.model_path)
        / setting
    )


def _jsonable_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


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


def _run_dev_inference(
        args: argparse.Namespace,
        results_path: Path,
        run_id: str) -> None:
    command = [
        sys.executable,
        '-u',
        '-m',
        'grounding_control.benchmarks.gqa_controlled.evaluator',
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
        '--run-id',
        run_id,
        '--fail-fast',
    ]
    if args.force_inference:
        command.append('--overwrite')
    print('Running one low-threshold dev inference:', flush=True)
    print(' '.join(command), flush=True)
    subprocess.run(command, check=True)


def _write_outputs(
        layout: RunLayout,
        payload: Mapping[str, Any],
        evaluations: Sequence[Mapping[str, Any]],
) -> Dict[str, Path]:
    if layout.is_canonical:
        layout.ensure_run_directories(include_artifacts=True)
        json_path = layout.summary_path
        best_path = layout.artifacts_dir / 'best_config.json'
        csv_path = layout.artifacts_dir / 'threshold_search.csv'
        with layout.results_path.open('w', encoding='utf-8') as handle:
            for evaluation in evaluations:
                handle.write(json.dumps(evaluation, ensure_ascii=False) + '\n')
    else:
        layout.ensure_run_directories()
        json_path = layout.results_path
        best_path = layout.run_dir / 'best_config.json'
        csv_path = layout.run_dir / 'threshold_search.csv'

    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    with best_path.open('w', encoding='utf-8') as handle:
        json.dump(payload['best'], handle, ensure_ascii=False, indent=2)
        handle.write('\n')

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
    return {
        'results': layout.results_path,
        'summary': json_path,
        'best_config': best_path,
        'csv': csv_path,
    }


def _write_cache_reference(
        layout: RunLayout,
        cache_dir: Path,
        summary_path: Path,
        reused: bool) -> None:
    if not layout.is_canonical:
        return
    reference_path = layout.artifacts_dir / 'inference_cache.reference.json'
    payload = {
        'schema_version': 1,
        'cache_dir': str(cache_dir),
        'results_path': str(cache_dir / 'results.jsonl'),
        'summary_path': str(summary_path),
        'reused': bool(reused),
    }
    with reference_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run Grounding DINO once on controlled GQA dev and select the '
            'best top-1 box confidence threshold offline.'
        )
    )
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=(
            'Exact legacy output directory. Omit it to use the canonical '
            'verifier benchmark run hierarchy.'
        ),
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument('--run-id', default=None)
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=None,
        help=(
            'Optional exact reusable inference cache. Canonical runs default '
            'to a parameter-addressed shared cache below --output-root.'
        ),
    )
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
    return parser.parse_args(argv)


def _run_threshold_search(
        args: argparse.Namespace,
        layout: RunLayout,
        lifecycle: Dict[str, bool]) -> None:
    if not args.box_thresholds:
        raise ValueError('--box-thresholds must not be empty')
    if not 0.0 <= args.text_threshold <= 1.0:
        raise ValueError('--text-threshold must be in [0, 1]')
    if not 0.0 < args.grounding_accept_iou <= 1.0:
        raise ValueError('--grounding-accept-iou must be in (0, 1]')
    if not 0.0 < args.grounding_containment <= 1.0:
        raise ValueError('--grounding-containment must be in (0, 1]')

    cache_dir = resolve_inference_cache_dir(args, layout)
    layout.ensure_run_directories(include_artifacts=layout.is_canonical)
    write_run_config(layout, {
        'command': list(sys.argv),
        'arguments': _jsonable_arguments(args),
        'resolved_paths': {
            'run_dir': str(layout.run_dir),
            'results': str(layout.results_path),
            'summary': str(
                layout.summary_path
                if layout.is_canonical else layout.results_path
            ),
            'verifier_events': str(layout.events_path),
        },
        'inputs': {
            'benchmark': str(args.benchmark),
            'inference_cache': str(cache_dir),
        },
        'components': {
            'backend': 'grounding_dino_geometry',
            'model': str(args.model_path),
            'selection_metric': args.selection_metric,
        },
    })
    write_run_status(layout, 'running', threshold_count=len(args.box_thresholds))
    lifecycle['started'] = True
    results_path = cache_dir / 'results.jsonl'
    summary_path = _summary_path(results_path)
    reused_cache = not args.force_inference and results_path.exists()
    if args.force_inference or not results_path.exists():
        if summary_path.exists() and not args.force_inference:
            raise FileExistsError(
                f'cache summary exists without results: {summary_path}; '
                'use --force-inference'
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        _run_dev_inference(args, results_path, layout.run_id)
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
    _write_cache_reference(
        layout,
        cache_dir,
        summary_path,
        reused=reused_cache,
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
        'run_id': layout.run_id,
        'run_layout': layout.layout_kind,
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
    output_paths = _write_outputs(layout, payload, evaluations)
    write_run_status(
        layout,
        'completed',
        threshold_count=len(evaluations),
        best_box_threshold=best['box_threshold'],
        results_path=str(output_paths['results']),
        summary_path=str(output_paths['summary']),
        best_config_path=str(output_paths['best_config']),
        csv_path=str(output_paths['csv']),
    )

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
    print(f"\nRun id: {layout.run_id}")
    print(f"Best config: {output_paths['best_config']}")
    print(f"Full search: {output_paths['summary']}")
    print(f"CSV table: {output_paths['csv']}")


def main() -> None:
    args = parse_args()
    layout = resolve_threshold_search_layout(args)
    lifecycle = {'started': False}
    try:
        _run_threshold_search(args, layout, lifecycle)
    except BaseException as error:
        if lifecycle['started']:
            try:
                write_run_status(
                    layout,
                    'failed',
                    error_type=type(error).__name__,
                    error=str(error)[:2000],
                )
            except Exception:
                # Preserve the search/inference exception if status cleanup
                # itself cannot be written.
                pass
        raise


if __name__ == '__main__':
    main()
