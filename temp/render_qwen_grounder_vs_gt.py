#!/usr/bin/env python3
"""Render one Qwen Grounder prediction together with its VStar GT box.

The script reads the intrinsic Grounder benchmark ``results.jsonl``.  Both
boxes stored by that evaluator are already expressed in the original-image
pixel coordinate frame, so this renderer intentionally performs no VoCoT
padding conversion.

Examples:

    python temp/render_qwen_grounder_vs_gt.py --sample-id main:9

    python temp/render_qwen_grounder_vs_gt.py \
        --sample-id main:147 --target-index 1

    python temp/render_qwen_grounder_vs_gt.py \
        --task-id main:174:target:0 --output temp/main174_slide.png

    python temp/render_qwen_grounder_vs_gt.py --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / (
    'output/vstar/runs/full_238/grounder_accuracy/qwen25_vl_7b/'
    'compact_json_v1/qwen7b_full_v1/results.jsonl'
)
DEFAULT_IMAGE_DIR = Path('/data/zhonggai/VStar')
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'temp/qwen_grounder_visualizations'

GT_COLOR = (235, 48, 48, 255)
QWEN_COLOR = (22, 135, 255, 255)
TEXT_COLOR = (255, 255, 255, 255)
PANEL_COLOR = (12, 16, 24, 235)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {error}') from error
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{line_number}: record must be a JSON object')
            records.append(record)
    if not records:
        raise ValueError(f'no records found in {path}')
    return records


def select_record(
        records: Iterable[Dict[str, Any]], *, task_id: Optional[str],
        sample_id: Optional[str], target_index: int) -> Dict[str, Any]:
    if task_id:
        matches = [record for record in records if record.get('task_id') == task_id]
        selector = f'task_id={task_id!r}'
    else:
        matches = [
            record for record in records
            if record.get('sample_id') == sample_id
            and record.get('target_index') == target_index
        ]
        selector = f'sample_id={sample_id!r}, target_index={target_index}'
    if len(matches) != 1:
        raise ValueError(f'expected exactly one record for {selector}; found {len(matches)}')
    return matches[0]


def validate_box(box: Any, field: str) -> Tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f'{field} must contain four coordinates, got {box!r}')
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{field} contains a non-finite coordinate')
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f'{field} has non-positive extent: {values}')
    return values


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except OSError:
        return ImageFont.load_default()


def draw_dashed_line(
        draw: ImageDraw.ImageDraw, start: Tuple[float, float],
        end: Tuple[float, float], color: Tuple[int, int, int, int],
        width: int, dash: int) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    offset = 0.0
    while offset < length:
        segment_end = min(offset + dash, length)
        draw.line(
            [
                (x1 + dx * offset, y1 + dy * offset),
                (x1 + dx * segment_end, y1 + dy * segment_end),
            ],
            fill=color,
            width=width,
        )
        offset += dash * 2


def draw_dashed_rectangle(
        draw: ImageDraw.ImageDraw, box: Sequence[float],
        color: Tuple[int, int, int, int], width: int, dash: int) -> None:
    x1, y1, x2, y2 = box
    draw_dashed_line(draw, (x1, y1), (x2, y1), color, width, dash)
    draw_dashed_line(draw, (x2, y1), (x2, y2), color, width, dash)
    draw_dashed_line(draw, (x2, y2), (x1, y2), color, width, dash)
    draw_dashed_line(draw, (x1, y2), (x1, y1), color, width, dash)


def draw_box_label(
        draw: ImageDraw.ImageDraw, box: Sequence[float], label: str,
        color: Tuple[int, int, int, int], font: ImageFont.ImageFont,
        image_size: Tuple[int, int], *, below: bool) -> None:
    image_width, image_height = image_size
    text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    left = max(0, min(int(round(box[0])), image_width - text_width - 12))
    if below:
        top = int(round(box[3])) + 4
    else:
        top = int(round(box[1])) - text_height - 12
    top = max(0, min(top, image_height - text_height - 10))
    draw.rounded_rectangle(
        [left, top, left + text_width + 10, top + text_height + 8],
        radius=3,
        fill=color,
    )
    draw.text(
        (left + 5, top + 3), label, font=font, fill=TEXT_COLOR,
        stroke_width=1, stroke_fill=(0, 0, 0, 180),
    )


def format_box(box: Optional[Sequence[float]]) -> str:
    if box is None:
        return 'unavailable'
    return '[' + ', '.join(f'{float(value):.1f}' for value in box) + ']'


def resolve_image_path(record: Dict[str, Any], image_dir: Path) -> Path:
    stored_path = record.get('image_path')
    if isinstance(stored_path, str) and Path(stored_path).is_file():
        return Path(stored_path)
    relative_path = record.get('image')
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError('record contains neither a valid image_path nor image')
    path = image_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def render_record(
        record: Dict[str, Any], *, image_dir: Path,
        output_path: Path) -> Dict[str, Any]:
    image_path = resolve_image_path(record, image_dir)
    with Image.open(image_path) as source:
        image = source.convert('RGBA')

    stored_size = record.get('image_size')
    if isinstance(stored_size, list) and tuple(stored_size) != image.size:
        raise ValueError(
            f'record image_size={stored_size} does not match source image={image.size}'
        )

    gt_box = validate_box(
        record.get('gt_bbox_original_pixel_xyxy'),
        'gt_bbox_original_pixel_xyxy',
    )
    raw_prediction = record.get('prediction_bbox_original_pixel_xyxy')
    prediction_box = None
    if raw_prediction is not None:
        prediction_box = validate_box(
            raw_prediction, 'prediction_bbox_original_pixel_xyxy'
        )

    short_side = min(image.size)
    line_width = max(4, round(short_side / 220))
    font = load_font(max(18, round(short_side / 52)))

    # Translucent fills make the overlap visibly purple while the exact box
    # boundaries remain distinguishable: GT is solid red; Qwen is dashed blue.
    fill_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    fill_draw.rectangle(gt_box, fill=(235, 48, 48, 42))
    if prediction_box is not None:
        fill_draw.rectangle(prediction_box, fill=(22, 135, 255, 42))
    image = Image.alpha_composite(image, fill_layer)

    draw = ImageDraw.Draw(image)
    draw.rectangle(gt_box, outline=GT_COLOR, width=line_width)
    draw_box_label(
        draw, gt_box, 'GT', GT_COLOR, font, image.size, below=False
    )
    if prediction_box is not None:
        draw_dashed_rectangle(
            draw, prediction_box, QWEN_COLOR,
            width=line_width, dash=max(10, line_width * 3),
        )
        draw_box_label(
            draw, prediction_box, 'Qwen Grounder', QWEN_COLOR,
            font, image.size, below=True,
        )

    header_font = load_font(max(20, round(short_side / 45)))
    small_font = load_font(max(16, round(short_side / 60)))
    iou = float(record.get('iou', 0.0))
    status = str(record.get('status', 'unknown'))
    header_lines = [
        f'{record.get("task_id")} | object: {record.get("object_reference")} | '
        f'IoU: {iou:.4f} | status: {status}',
        f'GT {format_box(gt_box)}   Qwen {format_box(prediction_box)}',
        'red solid = GT    blue dashed = Qwen Grounder',
    ]
    padding = 14
    line_gap = 6
    heights = []
    for index, line in enumerate(header_lines):
        selected_font = header_font if index == 0 else small_font
        bbox = draw.textbbox((0, 0), line, font=selected_font)
        heights.append(bbox[3] - bbox[1])
    header_height = padding * 2 + sum(heights) + line_gap * (len(heights) - 1)
    canvas = Image.new('RGBA', (image.width, image.height + header_height), PANEL_COLOR)
    canvas.alpha_composite(image, (0, header_height))
    canvas_draw = ImageDraw.Draw(canvas)
    text_y = padding
    for index, (line, height) in enumerate(zip(header_lines, heights)):
        selected_font = header_font if index == 0 else small_font
        canvas_draw.text((padding, text_y), line, font=selected_font, fill=TEXT_COLOR)
        text_y += height + line_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert('RGB').save(output_path, quality=95)
    return {
        'task_id': record.get('task_id'),
        'sample_id': record.get('sample_id'),
        'target_index': record.get('target_index'),
        'object_reference': record.get('object_reference'),
        'status': status,
        'iou': iou,
        'gt_bbox_original_pixel_xyxy': list(gt_box),
        'prediction_bbox_original_pixel_xyxy': (
            None if prediction_box is None else list(prediction_box)
        ),
        'image_path': str(image_path),
        'output_path': str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--task-id', help='For example: main:174:target:0')
    selector.add_argument('--sample-id', help='For example: main:9')
    selector.add_argument(
        '--all', action='store_true',
        help='Render every target record (292 targets for VStar full-238).',
    )
    parser.add_argument(
        '--target-index', type=int, default=0,
        help='Target index used together with --sample-id (default: 0).',
    )
    parser.add_argument('--image-dir', type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        '--output', type=Path,
        help='Output PNG. Default: temp/qwen_grounder_visualizations/<task-id>.png',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
        help='Directory used by --all (default: temp/qwen_grounder_visualizations).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.results)
    if args.all:
        if args.output is not None:
            raise ValueError('--output is only valid for one record; use --output-dir with --all')
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for record in tqdm(records, desc='Rendering Qwen Grounder vs GT'):
            ordinal = int(record.get('target_ordinal', len(reports)))
            safe_task_id = str(record['task_id']).replace(':', '_').replace('/', '_')
            # JPEG keeps the complete 292-target visualization set compact.
            output_path = output_dir / f'{ordinal:03d}_{safe_task_id}.jpg'
            reports.append(render_record(
                record, image_dir=args.image_dir, output_path=output_path,
            ))
        index_path = output_dir / 'render_index.jsonl'
        with index_path.open('w', encoding='utf-8') as handle:
            for report in reports:
                handle.write(json.dumps(report, ensure_ascii=False) + '\n')
        summary = {
            'results': str(args.results.resolve()),
            'rendered_target_count': len(reports),
            'unique_sample_count': len({report['sample_id'] for report in reports}),
            'output_dir': str(output_dir),
            'index': str(index_path),
        }
        summary_path = output_dir / 'render_summary.json'
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    record = select_record(
        records, task_id=args.task_id,
        sample_id=args.sample_id, target_index=args.target_index,
    )
    output_path = args.output
    if output_path is None:
        safe_task_id = str(record['task_id']).replace(':', '_').replace('/', '_')
        output_path = DEFAULT_OUTPUT_DIR / f'{safe_task_id}.png'
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    report = render_record(
        record, image_dir=args.image_dir, output_path=output_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
