"""Render original VStar GT boxes and baseline's explicitly matched boxes.

The input is one online-oracle result record (JSON) or a JSONL file containing
such records.  Ground-truth objects and boxes are *always* read from the
original VStar annotation next to the image, never from ``oracle_targets`` in
the input result.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import DEFAULT_BOC_TOKEN
from utils.coordinate_intervention import OnlineOracleCoordinateLogitsProcessor, box_iou


def read_records(path):
    """Read either one JSON object or a JSONL collection of result records."""
    content = Path(path).read_text(encoding='utf-8').strip()
    if not content:
        raise ValueError(f'input file is empty: {path}')
    if content.startswith('{'):
        try:
            return [json.loads(content)]
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def select_record(records, sample_id):
    if sample_id is None:
        if len(records) != 1:
            raise ValueError('--sample-id is required when --input contains multiple JSONL records')
        return records[0]
    matched = [record for record in records if record.get('question_id') == sample_id]
    if len(matched) != 1:
        raise ValueError(f'expected exactly one record with question_id={sample_id!r}, found {len(matched)}')
    return matched[0]


def original_targets(annotation, image_size):
    """Convert untouched VStar pixel XYWH boxes to normalized XYXY targets."""
    names = annotation.get('target_object')
    boxes = annotation.get('bbox')
    if not isinstance(names, list) or not isinstance(boxes, list) or len(names) != len(boxes):
        raise ValueError('original VStar annotation must contain equally sized target_object and bbox lists')
    width, height = image_size
    targets = []
    for index, (name, xywh) in enumerate(zip(names, boxes)):
        if not isinstance(name, str) or not name.strip() or not isinstance(xywh, list) or len(xywh) != 4:
            raise ValueError(f'invalid original target at index {index}')
        x, y, box_width, box_height = (float(value) for value in xywh)
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f'non-positive original bbox at index {index}: {xywh}')
        targets.append({
            'target_index': index,
            'object': name,
            'pixel_xywh': [x, y, box_width, box_height],
            'normalized_xyxy': [x / width, y / height, (x + box_width) / width, (y + box_height) / height],
        })
    return targets


def match_baseline_coordinates(record, tokenizer, targets):
    """Apply the original oracle's local-text alias matcher to baseline IDs."""
    baseline = record.get('baseline')
    if not isinstance(baseline, dict):
        raise ValueError('input record has no baseline section')
    generated_ids = baseline.get('generated_ids')
    baseline_boxes = baseline.get('boxes')
    if not isinstance(generated_ids, list) or not isinstance(baseline_boxes, list):
        raise ValueError('baseline.generated_ids and baseline.boxes are required')

    boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
    boc_offsets = [offset for offset, token_id in enumerate(generated_ids) if token_id == boc_token_id]
    if len(boc_offsets) != len(baseline_boxes):
        raise RuntimeError(
            f'baseline has {len(boc_offsets)} <coor> tokens but {len(baseline_boxes)} parseable boxes'
        )

    # ``OnlineOracleCoordinateLogitsProcessor`` is reused only for matching.
    # Its target boxes are derived above from the original annotation.
    matcher_targets = [
        {'object': target['object'], 'aliases': [target['object']], 'box': target['normalized_xyxy']}
        for target in targets
    ]
    matcher = OnlineOracleCoordinateLogitsProcessor(
        tokenizer, prompt_length=0, oracle_targets=matcher_targets, context_window_tokens=48
    )
    target_by_name = {target['object']: target for target in targets}
    if len(target_by_name) != len(targets):
        raise ValueError('duplicate original target_object names are ambiguous for strict rendering')

    matches = []
    decisions = []
    for coordinate_index, (boc_offset, baseline_box) in enumerate(zip(boc_offsets, baseline_boxes), 1):
        decision = matcher._new_decision(generated_ids, boc_offset, coordinate_index)
        decisions.append(decision)
        if decision['decision'] != 'forced_gt_box':
            continue
        target = target_by_name[decision['target_object']]
        matches.append({
            'coordinate_index': coordinate_index,
            'context': decision['context'],
            'object': target['object'],
            'baseline_box': [float(value) for value in baseline_box],
            'gt_box': target['normalized_xyxy'],
            'iou': box_iou(baseline_box, target['normalized_xyxy']),
        })
    return matches, decisions


def normalized_to_pixel_xyxy(box, image_size):
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in box)
    return [
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(0, min(width - 1, round(x2 * width))),
        max(0, min(height - 1, round(y2 * height))),
    ]


def draw_labeled_box(draw, xyxy, label, color, width, image_size, label_above):
    draw.rectangle(xyxy, outline=color, width=width)
    font = ImageFont.load_default()
    left, top, right, bottom = xyxy
    text_left = max(0, left)
    text_box = draw.textbbox((text_left, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    image_width, image_height = image_size
    text_left = min(text_left, max(0, image_width - text_width - 2))
    text_top = top - text_height - 5 if label_above else bottom + 3
    text_top = max(0, min(text_top, max(0, image_height - text_height - 2)))
    draw.rectangle(
        [text_left, text_top, text_left + text_width + 2, text_top + text_height + 2],
        fill=color,
    )
    draw.text((text_left + 1, text_top + 1), label, fill='white', font=font)


def draw_legend(draw):
    font = ImageFont.load_default()
    entries = [
        ('red', 'GT only'),
        ('blue', 'CoT only'),
        ('purple', 'strict text-matched GT + CoT'),
    ]
    left, top = 12, 12
    for color, label in entries:
        draw.rectangle([left, top, left + 14, top + 14], fill=color)
        draw.text((left + 20, top + 1), label, fill='white', stroke_width=1, stroke_fill='black', font=font)
        top += 19


def render(record, dataset_root, tokenizer_path, output_path):
    image_relpath = record.get('image')
    if not isinstance(image_relpath, str):
        raise ValueError('input record has no image path')
    image_path = Path(dataset_root) / image_relpath
    annotation_path = image_path.with_suffix('.json')
    if not image_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError(f'expected image and original annotation: {image_path}')

    with Image.open(image_path) as source_image:
        image = source_image.convert('RGB')
    annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
    if annotation.get('question') != record.get('question'):
        raise ValueError('input record question does not match the untouched VStar annotation')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    targets = original_targets(annotation, image.size)
    matches, decisions = match_baseline_coordinates(record, tokenizer, targets)

    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 250))
    # Every original target is drawn.  A target is purple when at least one
    # baseline CoT coordinate explicitly and uniquely matched that target.
    # Other original GT targets remain red.
    matched_target_names = {match['object'] for match in matches}
    for target in targets:
        gt_xyxy = [
            target['pixel_xywh'][0], target['pixel_xywh'][1],
            target['pixel_xywh'][0] + target['pixel_xywh'][2],
            target['pixel_xywh'][1] + target['pixel_xywh'][3],
        ]
        matched = target['object'] in matched_target_names
        draw_labeled_box(
            draw, gt_xyxy, f'GT {target["object"]}',
            'purple' if matched else 'red', line_width, image.size, label_above=True,
        )

    # Draw every baseline coordinate.  Exact duplicate boxes are consolidated
    # only visually, with all coordinate indices kept in the label.
    match_by_coordinate_index = {match['coordinate_index']: match for match in matches}
    baseline_boxes = record['baseline']['boxes']
    grouped = {}
    for coordinate_index, baseline_box in enumerate(baseline_boxes, 1):
        match = match_by_coordinate_index.get(coordinate_index)
        key = (
            match['object'] if match else None,
            tuple(round(float(value), 6) for value in baseline_box),
        )
        grouped.setdefault(key, []).append((coordinate_index, match, baseline_box))
    for grouped_coordinates in grouped.values():
        first_index, first_match, first_box = grouped_coordinates[0]
        baseline_xyxy = normalized_to_pixel_xyxy(first_box, image.size)
        indices = ','.join(str(index) for index, _, _ in grouped_coordinates)
        if first_match:
            label = f'CoT #{indices} -> {first_match["object"]} IoU={first_match["iou"]:.3f}'
            color = 'purple'
        else:
            label = f'CoT #{indices} (unmatched)'
            color = 'blue'
        draw_labeled_box(
            draw, baseline_xyxy,
            label, color, line_width, image.size,
            label_above=False,
        )
    draw_legend(draw)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    report = {
        'sample_id': record.get('question_id'),
        'image': str(image_path),
        'original_annotation': str(annotation_path),
        'question': record.get('question'),
        'baseline_answer': record.get('baseline_answer'),
        'oracle_answer': record.get('oracle_answer'),
        'matches': matches,
        'all_gt_target_count': len(targets),
        'all_baseline_coordinate_count': len(baseline_boxes),
        'unmatched_or_ambiguous_coordinate_count': sum(
            decision['decision'] != 'forced_gt_box' for decision in decisions
        ),
        'output_image': str(output_path),
    }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input', default='output/vstar/online_oracle/full_238_padding_fix/results.jsonl',
        help='Online-oracle result JSON/JSONL. Defaults to the full-238 run.',
    )
    parser.add_argument(
        '--sample-id', required=True,
        help='VStar question id, for example main:9. Required because the default is JSONL.',
    )
    parser.add_argument('--dataset-root', default='/data/zhonggai/VStar')
    parser.add_argument('--tokenizer-path', default='weights/Volcano-7b')
    parser.add_argument('--output', required=True, help='Output PNG path.')
    return parser.parse_args()


def main():
    args = parse_args()
    record = select_record(read_records(args.input), args.sample_id)
    report = render(record, args.dataset_root, args.tokenizer_path, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
