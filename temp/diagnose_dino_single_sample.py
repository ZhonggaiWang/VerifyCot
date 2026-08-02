#!/usr/bin/env python3
"""Diagnose one VStar Grounding DINO event without touching main code.

The diagnostic separates three common failure sources:

1. reference formulation: a canonical GT object name works but the online ref does not;
2. score ranking: a good box exists in DINO's top-k, but top-1 selects another box;
3. proposal/localization: no sufficiently overlapping box is produced, even with a
   canonical reference and permissive post-processing thresholds.

Run this script with the environment that already hosts Grounding DINO, normally
``qwen25`` in this project. All artifacts are disposable and stay under ``temp/``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from grounding_control.models.grounding_dino import LocalGroundingDinoRunner  # noqa: E402


DEFAULT_EVENTS = (
    PROJECT_ROOT
    / 'output/vstar/runs/full_238/routing/dino_geometry__oracle_experts'
    / 'iou_0p6/full_238_iou06_v1/verifier_events.jsonl'
)
DEFAULT_SAMPLES = (
    PROJECT_ROOT
    / 'output/vstar/online_oracle/full_238_padding_fix/results.jsonl'
)
DEFAULT_IMAGE_DIR = Path('/data/zhonggai/VStar')
DEFAULT_MODEL = Path('/data/zhonggai/models/grounding-dino-base')
DEFAULT_OUTPUT = PROJECT_ROOT / 'temp/dino_single_sample_diagnosis'

PixelBox = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-id', default='main:0')
    parser.add_argument('--grounding-step', type=int, default=1)
    parser.add_argument('--events', type=Path, default=DEFAULT_EVENTS)
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument('--image-dir', type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--box-threshold', type=float, default=0.3)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--low-box-threshold', type=float, default=0.05)
    parser.add_argument('--low-text-threshold', type=float, default=0.05)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--success-iou', type=float, default=0.5)
    parser.add_argument(
        '--extra-reference',
        action='append',
        default=[],
        help='Optional additional query; may be passed more than once.',
    )
    args = parser.parse_args()
    if args.grounding_step <= 0:
        parser.error('--grounding-step must be positive')
    if args.top_k <= 0:
        parser.error('--top-k must be positive')
    for name in (
        'box_threshold', 'text_threshold', 'low_box_threshold',
        'low_text_threshold', 'success_iou',
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            parser.error(f'--{name.replace("_", "-")} must be in [0, 1]')
    return args


def read_matching_jsonl(
        path: Path,
        predicate,
        description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'{description} file not found: {path}')
    matches = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if predicate(record):
                matches.append((line_number, record))
    if not matches:
        raise LookupError(f'no {description} matched in {path}')
    if len(matches) > 1:
        lines = ', '.join(str(item[0]) for item in matches)
        raise LookupError(
            f'multiple {description} records matched at lines {lines}'
        )
    return matches[0][1]


def padded_normalized_to_original_pixel(
        box: Sequence[float], width: int, height: int) -> PixelBox:
    if len(box) != 4:
        raise ValueError(f'expected four box coordinates, got {box!r}')
    square = float(max(width, height))
    offset_x = 0.0 if width >= height else (height - width) // 2
    offset_y = (width - height) // 2 if width > height else 0.0
    x1, y1, x2, y2 = (float(value) for value in box)
    converted = (
        min(max(x1 * square - offset_x, 0.0), float(width)),
        min(max(y1 * square - offset_y, 0.0), float(height)),
        min(max(x2 * square - offset_x, 0.0), float(width)),
        min(max(y2 * square - offset_y, 0.0), float(height)),
    )
    if converted[0] >= converted[2] or converted[1] >= converted[3]:
        raise ValueError(
            f'padded box falls outside original image: {box!r} -> {converted!r}'
        )
    return converted


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def simple_reference_cleanup(reference: str) -> str:
    """Create a deliberately transparent, non-learned query variant."""

    text = re.sub(r'<coor>.*?</coor>', ' ', reference, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s-]', ' ', text.lower())
    text = ' '.join(text.split())
    prefixes = (
        'find the ', 'find a ', 'find an ', 'find ',
        'locate the ', 'locate a ', 'locate an ', 'locate ',
        'identify the ', 'identify a ', 'identify an ', 'identify ',
        'look at the ', 'look at a ', 'look at an ', 'look at ',
        'check the ', 'check a ', 'check an ', 'check ',
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def unique_references(items: Iterable[Tuple[str, Optional[str]]]) -> List[Dict[str, str]]:
    result = []
    seen = set()
    for name, value in items:
        if not isinstance(value, str):
            continue
        normalized = ' '.join(value.strip().split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        result.append({'name': name, 'reference': normalized})
    return result


def summarize_detections(
        detections,
        gt_box: PixelBox,
        candidate_box: PixelBox,
        top_k: int,
        success_iou: float) -> Dict[str, Any]:
    ranked = sorted(detections, key=lambda item: -float(item.score))[:top_k]
    records = []
    for rank, detection in enumerate(ranked, start=1):
        box = tuple(float(value) for value in detection.box_original_pixel_xyxy)
        records.append({
            'rank': rank,
            'score': float(detection.score),
            'label': str(detection.label),
            'box_original_pixel_xyxy': list(box),
            'iou_to_gt': box_iou(box, gt_box),
            'iou_to_vocot_candidate': box_iou(box, candidate_box),
        })
    top1 = records[0] if records else None
    best = max(records, key=lambda item: item['iou_to_gt'], default=None)
    recall_at = {}
    for cutoff in (1, 3, 5, 10, top_k):
        effective = min(cutoff, top_k)
        recall_at[str(effective)] = any(
            item['iou_to_gt'] >= success_iou
            for item in records[:effective]
        )
    return {
        'detection_count_before_top_k': len(detections),
        'detections_logged': len(records),
        'top1': top1,
        'best_iou_candidate': best,
        'top1_success': bool(top1 and top1['iou_to_gt'] >= success_iou),
        'top_k_contains_success': bool(
            best and best['iou_to_gt'] >= success_iou
        ),
        'ranking_failure': bool(
            best
            and best['iou_to_gt'] >= success_iou
            and (top1 is None or top1['iou_to_gt'] < success_iou)
        ),
        'proposal_failure': bool(
            best is None or best['iou_to_gt'] < success_iou
        ),
        'recall_at_k': recall_at,
        'detections': records,
    }


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '_', value).strip('_')
    return cleaned or 'query'


def draw_diagnostic(
        image: Image.Image,
        output: Path,
        title: str,
        gt_box: PixelBox,
        candidate_box: PixelBox,
        result: Dict[str, Any]) -> None:
    rendered = image.convert('RGB').copy()
    draw = ImageDraw.Draw(rendered)
    line_width = max(2, round(max(rendered.size) / 500))
    colors = {
        'GT': (180, 0, 255),
        'VoCoT': (255, 150, 0),
        'DINO top1': (255, 40, 40),
        'DINO best': (0, 210, 80),
    }

    def box_with_label(box: Sequence[float], label: str, color) -> None:
        draw.rectangle(tuple(box), outline=color, width=line_width)
        x, y = float(box[0]), max(0.0, float(box[1]) - 14)
        draw.text((x + 2, y), label, fill=color, stroke_width=2,
                  stroke_fill=(0, 0, 0))

    box_with_label(gt_box, 'GT', colors['GT'])
    box_with_label(candidate_box, 'VoCoT', colors['VoCoT'])
    top1 = result.get('top1')
    best = result.get('best_iou_candidate')
    if top1:
        box_with_label(
            top1['box_original_pixel_xyxy'],
            f"DINO top1 s={top1['score']:.3f} IoU={top1['iou_to_gt']:.3f}",
            colors['DINO top1'],
        )
    if best and (top1 is None or best['rank'] != top1['rank']):
        box_with_label(
            best['box_original_pixel_xyxy'],
            f"DINO best rank={best['rank']} IoU={best['iou_to_gt']:.3f}",
            colors['DINO best'],
        )
    draw.text(
        (8, 8), title, fill=(255, 255, 255), stroke_width=3,
        stroke_fill=(0, 0, 0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output)


def diagnose(report: Dict[str, Any], success_iou: float) -> Dict[str, Any]:
    by_key = {
        (run['reference_name'], run['profile']): run
        for run in report['runs']
    }
    raw_production = by_key.get(('raw_online_reference', 'production'))
    canonical_names = ('matched_alias', 'target_object')
    canonical_production = next(
        (by_key.get((name, 'production')) for name in canonical_names
         if by_key.get((name, 'production')) is not None),
        None,
    )
    canonical_low = next(
        (by_key.get((name, 'low_threshold')) for name in canonical_names
         if by_key.get((name, 'low_threshold')) is not None),
        None,
    )

    def best_iou(run: Optional[Dict[str, Any]]) -> Optional[float]:
        if not run or not run['analysis']['best_iou_candidate']:
            return None
        return float(run['analysis']['best_iou_candidate']['iou_to_gt'])

    def top1_iou(run: Optional[Dict[str, Any]]) -> Optional[float]:
        if not run or not run['analysis']['top1']:
            return None
        return float(run['analysis']['top1']['iou_to_gt'])

    raw_best = best_iou(raw_production)
    canonical_best = best_iou(canonical_production)
    canonical_low_best = best_iou(canonical_low)
    reference_failure = bool(
        (raw_best is None or raw_best < success_iou)
        and canonical_best is not None
        and canonical_best >= success_iou
    )
    threshold_failure = bool(
        (canonical_best is None or canonical_best < success_iou)
        and canonical_low_best is not None
        and canonical_low_best >= success_iou
    )
    canonical_ranking_failure = bool(
        canonical_production
        and canonical_production['analysis']['ranking_failure']
    )
    proposal_failure = bool(
        canonical_low_best is None or canonical_low_best < success_iou
    )

    causes = []
    if reference_failure:
        causes.append('reference_formulation_failure')
    if threshold_failure:
        causes.append('threshold_filtering_failure')
    if canonical_ranking_failure:
        causes.append('ranking_failure')
    if proposal_failure:
        causes.append('proposal_or_localization_failure')
    if not causes:
        causes.append('no_failure_under_canonical_query')
    return {
        'success_iou': success_iou,
        'primary_diagnosis': causes[0],
        'all_supported_diagnoses': causes,
        'reference_formulation_failure': reference_failure,
        'threshold_filtering_failure': threshold_failure,
        'canonical_ranking_failure': canonical_ranking_failure,
        'proposal_or_localization_failure': proposal_failure,
        'raw_production_best_iou': raw_best,
        'raw_production_top1_iou': top1_iou(raw_production),
        'canonical_production_best_iou': canonical_best,
        'canonical_production_top1_iou': top1_iou(canonical_production),
        'canonical_low_threshold_best_iou': canonical_low_best,
        'interpretation': {
            'reference_formulation_failure': (
                'The production ref fails, while the canonical GT name yields '
                'a successful DINO proposal.'
            ),
            'threshold_filtering_failure': (
                'A successful proposal appears only after lowering DINO '
                'post-processing thresholds.'
            ),
            'ranking_failure': (
                'A successful proposal exists in top-k, but detector-score '
                'top-1 selects a different box.'
            ),
            'proposal_or_localization_failure': (
                'No successful proposal is found even with a canonical name '
                'and permissive thresholds.'
            ),
        },
    }


def write_text_report(report: Dict[str, Any], path: Path) -> None:
    sample = report['sample']
    diagnosis = report['diagnosis']
    lines = [
        'Grounding DINO single-sample diagnosis',
        f"sample: {sample['sample_id']} step={sample['grounding_step']}",
        f"image: {sample['image_path']}",
        f"question: {sample['question']}",
        f"GT object: {sample['target_object']}",
        f"GT box (original pixels): {sample['gt_box_original_pixel_xyxy']}",
        f"VoCoT candidate IoU to GT: {sample['vocot_candidate_iou_to_gt']:.4f}",
        '',
        f"primary diagnosis: {diagnosis['primary_diagnosis']}",
        'all diagnoses: ' + ', '.join(diagnosis['all_supported_diagnoses']),
        '',
        'Runs:',
    ]
    for run in report['runs']:
        analysis = run['analysis']
        top1 = analysis['top1']
        best = analysis['best_iou_candidate']
        lines.extend([
            (
                f"- {run['reference_name']} / {run['profile']}: "
                f"ref={run['reference']!r}, detections="
                f"{analysis['detection_count_before_top_k']}"
            ),
            (
                '  top1: none' if top1 is None else
                f"  top1: score={top1['score']:.4f}, "
                f"IoU={top1['iou_to_gt']:.4f}, label={top1['label']!r}"
            ),
            (
                '  best: none' if best is None else
                f"  best: rank={best['rank']}, score={best['score']:.4f}, "
                f"IoU={best['iou_to_gt']:.4f}, label={best['label']!r}"
            ),
            (
                f"  ranking_failure={analysis['ranking_failure']}, "
                f"proposal_failure={analysis['proposal_failure']}"
            ),
        ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    event = read_matching_jsonl(
        args.events,
        lambda item: (
            item.get('sample_id') == args.sample_id
            and int(item.get('grounding_step', -1)) == args.grounding_step
        ),
        'verifier event',
    )
    sample = read_matching_jsonl(
        args.samples,
        lambda item: str(item.get('question_id')) == args.sample_id,
        'VStar sample',
    )
    image_path = args.image_dir / str(sample['image'])
    if not image_path.is_file():
        raise FileNotFoundError(f'image not found: {image_path}')
    with Image.open(image_path) as opened:
        image = opened.convert('RGB').copy()

    target_object = event.get('target_object')
    matched_alias = event.get('matched_alias')
    raw_reference = str(event['object_reference'])
    gt_padded = event.get('oracle_target_box')
    if gt_padded is None:
        gt_padded = event.get('posthoc_oracle_audit', {}).get(
            'oracle_target_box'
        )
    if gt_padded is None:
        raise ValueError('selected event has no uniquely matched oracle target')
    candidate_padded = event['candidate_box']
    gt_pixel = padded_normalized_to_original_pixel(
        gt_padded, image.width, image.height
    )
    candidate_pixel = padded_normalized_to_original_pixel(
        candidate_padded, image.width, image.height
    )

    references = unique_references([
        ('raw_online_reference', raw_reference),
        ('matched_alias', matched_alias),
        ('target_object', target_object),
        ('simple_cleanup', simple_reference_cleanup(raw_reference)),
        *[(f'extra_{index}', value)
          for index, value in enumerate(args.extra_reference, start=1)],
    ])
    profiles = [
        {
            'name': 'production',
            'box_threshold': args.box_threshold,
            'text_threshold': args.text_threshold,
        },
        {
            'name': 'low_threshold',
            'box_threshold': args.low_box_threshold,
            'text_threshold': args.low_text_threshold,
        },
    ]

    runner = LocalGroundingDinoRunner(
        model_path=str(args.model_path),
        device=args.device,
        dtype=args.dtype,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        local_files_only=True,
    )
    runs = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for reference in references:
        for profile in profiles:
            runner.box_threshold = float(profile['box_threshold'])
            runner.text_threshold = float(profile['text_threshold'])
            detections = runner.detect(image, reference['reference'])
            analysis = summarize_detections(
                detections,
                gt_pixel,
                candidate_pixel,
                args.top_k,
                args.success_iou,
            )
            run = {
                'reference_name': reference['name'],
                'reference': reference['reference'],
                'profile': profile['name'],
                'box_threshold': profile['box_threshold'],
                'text_threshold': profile['text_threshold'],
                'runner_metadata': dict(runner.last_run_metadata),
                'analysis': analysis,
            }
            runs.append(run)
            image_name = (
                f"{safe_name(reference['name'])}__{profile['name']}.png"
            )
            draw_diagnostic(
                image,
                args.output_dir / image_name,
                f"{reference['name']} | {profile['name']} | "
                f"{reference['reference']}",
                gt_pixel,
                candidate_pixel,
                analysis,
            )

    report = {
        'schema': 'dino_single_sample_reference_proposal_diagnosis_v1',
        'disposable_output': True,
        'sample': {
            'sample_id': args.sample_id,
            'grounding_step': args.grounding_step,
            'question': sample.get('question'),
            'image_path': str(image_path),
            'image_size': [image.width, image.height],
            'raw_online_reference': raw_reference,
            'matched_alias': matched_alias,
            'target_object': target_object,
            'gt_box_padded_normalized_xyxy': list(gt_padded),
            'gt_box_original_pixel_xyxy': list(gt_pixel),
            'vocot_candidate_box_padded_normalized_xyxy': list(
                candidate_padded
            ),
            'vocot_candidate_box_original_pixel_xyxy': list(candidate_pixel),
            'vocot_candidate_iou_to_gt': box_iou(candidate_pixel, gt_pixel),
        },
        'settings': {
            'model_path': str(args.model_path),
            'device': args.device,
            'dtype': args.dtype,
            'top_k': args.top_k,
            'success_iou': args.success_iou,
            'events': str(args.events),
            'samples': str(args.samples),
        },
        'runs': runs,
    }
    report['diagnosis'] = diagnose(report, args.success_iou)
    json_path = args.output_dir / 'report.json'
    text_path = args.output_dir / 'report.txt'
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    write_text_report(report, text_path)

    print(text_path.read_text(encoding='utf-8'), end='')
    print(f'JSON report: {json_path}')
    print(f'Visualizations: {args.output_dir}/*.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
