"""Visualize one VStar one-shot corruption/repair result on its source image.

The GT boxes are always read from the untouched VStar ``.json`` annotation.
This intentionally does not trust copied ``oracle_targets`` fields in an
evaluation result.  The selected oracle target, injected random box ``q``,
and the first repaired box ``r`` are rendered together so that a repair can
be inspected spatially rather than only through aggregate IoU.

Typical use::

    python eval/Oracle_experiment/vstar/render_one_shot_reference_repair.py \
      --results output/vstar/one_shot_reference_repair/text_only_sandbox/typed_feedback/20260728_113623/results.jsonl \
      --manifest output/vstar/one_shot_reference_repair/full_238_padding_fix/manifest.jsonl \
      --sample-id main:6 \
      --dataset-root /data/zhonggai/VStar \
      --output output/vstar/repair_renders/main_6.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coordinate_intervention import box_iou


Box = Tuple[float, float, float, float]
COORDINATE_PATTERN = re.compile(
    r'<coor>\s*([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*,\s*'
    r'([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*</coor>'
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read one JSON object or a non-empty JSONL file."""
    content = Path(path).read_text(encoding='utf-8').strip()
    if not content:
        raise ValueError(f'input file is empty: {path}')
    if content.startswith('{'):
        try:
            return [json.loads(content)]
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def select_by_sample_id(records: Iterable[Dict[str, Any]], sample_id: str, source: str) -> Dict[str, Any]:
    matches = [record for record in records if record.get('sample_id') == sample_id]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one {source} record for sample_id={sample_id!r}, found {len(matches)}')
    return matches[0]


def original_targets(annotation: Dict[str, Any], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Read original VStar target boxes from untouched pixel-XYWH metadata."""
    names, boxes = annotation.get('target_object'), annotation.get('bbox')
    if not isinstance(names, list) or not isinstance(boxes, list) or len(names) != len(boxes):
        raise ValueError('original VStar annotation needs equally sized target_object and bbox lists')
    width, height = image_size
    targets = []
    for index, (name, xywh) in enumerate(zip(names, boxes)):
        if not isinstance(name, str) or not isinstance(xywh, list) or len(xywh) != 4:
            raise ValueError(f'invalid original target at index {index}')
        x, y, box_width, box_height = (float(value) for value in xywh)
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f'non-positive original bbox at index {index}: {xywh}')
        targets.append({
            'object': name,
            'pixel_xywh': [x, y, box_width, box_height],
            'box': [x / width, y / height, (x + box_width) / width, (y + box_height) / height],
        })
    return targets


def normalized_to_pixel_xyxy(box: Sequence[float], image_size: Tuple[int, int]) -> List[int]:
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in box)
    return [
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(0, min(width - 1, round(x2 * width))),
        max(0, min(height - 1, round(y2 * height))),
    ]


def draw_labeled_box(draw: ImageDraw.ImageDraw, xyxy: Sequence[int], label: str, color: str,
                     line_width: int, image_size: Tuple[int, int], label_above: bool) -> None:
    draw.rectangle(xyxy, outline=color, width=line_width)
    font = ImageFont.load_default()
    left, top, right, bottom = xyxy
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    image_width, image_height = image_size
    text_left = min(max(0, left), max(0, image_width - text_width - 2))
    text_top = top - text_height - 5 if label_above else bottom + 3
    text_top = min(max(0, text_top), max(0, image_height - text_height - 2))
    draw.rectangle([text_left, text_top, text_left + text_width + 2, text_top + text_height + 2], fill=color)
    draw.text((text_left + 1, text_top + 1), label, fill='white', font=font)


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    font = ImageFont.load_default()
    entries = [
        ('red', 'other GT target'),
        ('purple', 'selected GT / oracle reference'),
        ('orange', 'injected wrong box q'),
        ('green', 'first repaired box r'),
        ('blue', 'later free CoT box'),
    ]
    left, top = 12, 12
    for color, label in entries:
        draw.rectangle([left, top, left + 14, top + 14], fill=color)
        draw.text((left + 20, top + 1), label, fill='white', stroke_width=1, stroke_fill='black', font=font)
        top += 19


def parse_response_boxes(response: Any) -> List[Box]:
    if not isinstance(response, str):
        return []
    return [tuple(float(value) for value in match.groups()) for match in COORDINATE_PATTERN.finditer(response)]  # type: ignore[return-value]


def replacement_event(result: Dict[str, Any]) -> Dict[str, Any]:
    events = result.get('repair', {}).get('events', [])
    if not isinstance(events, list) or not events or not isinstance(events[0], dict):
        raise ValueError('result has no repair.events[0] record')
    return events[0]


def selected_target_name(manifest: Dict[str, Any], event: Dict[str, Any]) -> str:
    selected = manifest.get('selected_event', {})
    name = selected.get('target_object') if isinstance(selected, dict) else None
    name = name or manifest.get('selected_object_reference') or event.get('object_reference')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('could not identify the selected object from manifest/event')
    return name.strip()


def render(result: Dict[str, Any], manifest: Dict[str, Any], dataset_root: str,
           output: str, draw_later_coordinates: bool = True) -> Dict[str, Any]:
    if result.get('status') != 'ok':
        raise ValueError(f"cannot render non-successful result: {result.get('status')!r}")
    if result.get('image') != manifest.get('image'):
        raise ValueError('result and manifest image paths differ')
    image_relpath = result.get('image')
    if not isinstance(image_relpath, str):
        raise ValueError('result has no image path')
    image_path = Path(dataset_root) / image_relpath
    annotation_path = image_path.with_suffix('.json')
    if not image_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError(f'expected original image and annotation under {image_path}')

    with Image.open(image_path) as source:
        image = source.convert('RGB')
    annotation = json.loads(annotation_path.read_text(encoding='utf-8'))
    if annotation.get('question') != result.get('question'):
        raise ValueError('result question does not match untouched VStar annotation')

    event = replacement_event(result)
    target_name = selected_target_name(manifest, event)
    targets = original_targets(annotation, image.size)
    # Use the manifest/oracle reference box as the canonical selected GT box.
    # If the object name occurs more than once in an annotation, this avoids
    # guessing which identically named instance was selected.
    reference_box = tuple(float(value) for value in result.get('reference_box', manifest['reference_box']))
    random_box = tuple(float(value) for value in result['random_box'])
    repair_box = tuple(float(value) for value in event['replacement_box'])

    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 250))
    # Oracle boxes are emitted at three decimal places whereas untouched VStar
    # boxes keep full pixel precision.  Select the same-named GT instance with
    # the highest overlap instead of doing a brittle float-equality check.
    same_name_targets = [target for target in targets if target['object'].strip().lower() == target_name.lower()]
    selected_target = max(same_name_targets, key=lambda target: box_iou(target['box'], reference_box), default=None)
    selected_gt_count = int(selected_target is not None)
    for target in targets:
        is_selected = target is selected_target
        pixel_box = [
            target['pixel_xywh'][0], target['pixel_xywh'][1],
            target['pixel_xywh'][0] + target['pixel_xywh'][2],
            target['pixel_xywh'][1] + target['pixel_xywh'][3],
        ]
        draw_labeled_box(
            draw, pixel_box,
            f'GT {target["object"]}' + (' (selected)' if is_selected else ''),
            'purple' if is_selected else 'red', line_width, image.size, label_above=True,
        )
    # In case rounding differs between the online trace and raw annotation,
    # always render the exact reference box used by this experiment.
    draw_labeled_box(
        draw, normalized_to_pixel_xyxy(reference_box, image.size),
        f'oracle ref: {target_name}', 'purple', line_width, image.size, label_above=True,
    )
    draw_labeled_box(
        draw, normalized_to_pixel_xyxy(random_box, image.size),
        f'q injected IoU={box_iou(random_box, reference_box):.3f}',
        'orange', line_width, image.size, label_above=False,
    )
    repair_iou = box_iou(repair_box, reference_box)
    draw_labeled_box(
        draw, normalized_to_pixel_xyxy(repair_box, image.size),
        f'r repaired IoU={repair_iou:.3f}',
        'green', line_width, image.size, label_above=False,
    )

    repair_response = result.get('repair', {}).get('response', '')
    later_boxes = parse_response_boxes(repair_response)[1:]
    if draw_later_coordinates:
        for index, box in enumerate(later_boxes, 2):
            # Repeated output boxes are informative in the text trace but do
            # not need to obscure the image; only draw a new geometry once.
            if any(all(abs(a - b) <= 1e-9 for a, b in zip(box, earlier)) for earlier in [repair_box] + later_boxes[:index - 2]):
                continue
            draw_labeled_box(
                draw, normalized_to_pixel_xyxy(box, image.size),
                f'free CoT #{index}', 'blue', line_width, image.size, label_above=False,
            )
    draw_legend(draw)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        'sample_id': result.get('sample_id'),
        'image': str(image_path),
        'original_annotation': str(annotation_path),
        'question': result.get('question'),
        'selected_object': target_name,
        'original_gt_target_count': len(targets),
        'raw_annotation_selected_box_count': selected_gt_count,
        'reference_box': reference_box,
        'random_box': random_box,
        'replacement_box': repair_box,
        'random_box_iou_to_reference': box_iou(random_box, reference_box),
        'replacement_box_iou_to_reference': repair_iou,
        'prediction': result.get('prediction'),
        'prediction_correct': result.get('prediction_correct'),
        'source_coordinate_index': event.get('source_coordinate_index'),
        'repair_mode': event.get('mode'),
        'later_free_coordinate_count': len(later_boxes),
        'output_image': str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results',
        default=(
            'output/vstar/one_shot_reference_repair/text_only_sandbox/'
            'concise_typed_feedback/20260728_113623/results.jsonl'
        ),
        help='One-shot repair results JSONL (or one JSON record). Defaults to text-only concise feedback.',
    )
    parser.add_argument(
        '--manifest',
        default='output/vstar/one_shot_reference_repair/full_238_padding_fix/manifest.jsonl',
        help='Manifest paired with --results. Defaults to the full-238 one-shot manifest.',
    )
    parser.add_argument('--sample-id', required=True, help='For example: main:6.')
    parser.add_argument('--dataset-root', default='/data/zhonggai/VStar')
    parser.add_argument('--output', required=True, help='Rendered PNG path.')
    parser.add_argument('--hide-later-coordinates', action='store_true',
                        help='Draw only GT, oracle reference, q, and first repair r.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = select_by_sample_id(read_jsonl(args.results), args.sample_id, 'result')
    manifest = select_by_sample_id(read_jsonl(args.manifest), args.sample_id, 'manifest')
    report = render(
        result, manifest, args.dataset_root, args.output,
        draw_later_coordinates=not args.hide_later_coordinates,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
