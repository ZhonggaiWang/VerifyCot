"""Audit VStar target-object boxes and export normalized oracle boxes.

VStar's paired per-image annotations provide ``target_object`` and ``bbox``.
This script retains the official ``test_questions.jsonl`` sample order, checks
that each paired annotation and image is usable, and converts pixel boxes from
``[x, y, width, height]`` to VoCoT's normalized ``[x1, y1, x2, y2]`` format.

It deliberately performs no model inference and does not alter the dataset.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions-path', required=True,
                        help='Official VStar test_questions.jsonl path.')
    parser.add_argument('--dataset-root', required=True,
                        help='VStar root containing images and paired JSON annotations.')
    parser.add_argument('--output', required=True,
                        help='Output JSONL for valid per-sample oracle-box records.')
    parser.add_argument('--summary-output', default=None,
                        help='Optional audit-summary JSON path; defaults beside --output.')
    return parser.parse_args()


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_box(box, image_width, image_height):
    """Convert one VStar [x, y, w, h] box to normalized xyxy coordinates."""
    x, y, width, height = (float(value) for value in box)
    x2 = x + width
    y2 = y + height
    return [
        max(0.0, min(1.0, x / image_width)),
        max(0.0, min(1.0, y / image_height)),
        max(0.0, min(1.0, x2 / image_width)),
        max(0.0, min(1.0, y2 / image_height)),
    ]


def validate_box(box, image_width, image_height):
    if not isinstance(box, list) or len(box) != 4:
        return 'invalid_bbox_format'
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in box):
        return 'non_numeric_bbox'
    x, y, width, height = box
    if width <= 0 or height <= 0:
        return 'non_positive_bbox_extent'
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        return 'bbox_outside_image'
    return None


def audit_sample(index, record, dataset_root):
    image_relative_path = record.get('image')
    if not isinstance(image_relative_path, str):
        return None, 'missing_image_path'
    image_path = dataset_root / image_relative_path
    annotation_path = image_path.with_suffix('.json')
    if not image_path.is_file():
        return None, 'missing_image'
    if not annotation_path.is_file():
        return None, 'missing_annotation'
    try:
        with Image.open(image_path) as image:
            image_width, image_height = image.size
    except Exception:
        return None, 'unreadable_image'
    try:
        with annotation_path.open() as handle:
            annotation = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None, 'unreadable_annotation'

    objects = annotation.get('target_object')
    boxes = annotation.get('bbox')
    # The original 191-row JSONL stores the question inside ``text``.  The
    # normalized full-238 manifest additionally carries a canonical
    # ``question`` field, which is authoritative and avoids parsing options
    # appended to the single-line text prompt.
    source_question = record.get('question')
    if not isinstance(source_question, str):
        source_text = record.get('text')
        source_question = source_text.splitlines()[0] if isinstance(source_text, str) else None
    if not isinstance(annotation.get('question'), str) or not annotation['question'].strip():
        return None, 'missing_annotation_question'
    if source_question != annotation['question']:
        return None, 'jsonl_annotation_question_mismatch'
    source_options = record.get('options')
    if source_options is not None and source_options != annotation.get('options'):
        return None, 'jsonl_annotation_options_mismatch'
    if not isinstance(objects, list) or not objects:
        return None, 'missing_target_objects'
    if not isinstance(boxes, list) or not boxes:
        return None, 'missing_bboxes'
    if len(objects) != len(boxes):
        return None, 'target_bbox_count_mismatch'
    if not all(isinstance(name, str) and name.strip() for name in objects):
        return None, 'invalid_target_object_name'

    for box in boxes:
        error = validate_box(box, image_width, image_height)
        if error:
            return None, error

    normalized_boxes = [normalize_box(box, image_width, image_height) for box in boxes]
    category = record.get('category')
    expected_target_count = 2 if category == 'relative_position' else 1
    return {
        'sample_index': index,
        'question_id': record.get('question_id', str(index)),
        'category': category,
        'image': image_relative_path,
        'image_path': str(image_path),
        'annotation_path': str(annotation_path),
        'image_size': {'width': image_width, 'height': image_height},
        'question': annotation.get('question'),
        'target_objects': objects,
        'pixel_bboxes_xywh': boxes,
        'normalized_bboxes_xyxy': normalized_boxes,
        'expected_question_target_count': expected_target_count,
        'has_complete_question_target_coverage': len(objects) >= expected_target_count,
    }, None


def main():
    args = parse_args()
    questions_path = Path(args.questions_path)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_suffix('.summary.json')
    )
    records = read_jsonl(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    category_counts = Counter()
    valid_category_counts = Counter()
    target_count_counts = Counter()
    target_coverage_counts = Counter()
    error_counts = Counter()
    valid_records = []
    error_records = []
    for index, record in enumerate(records):
        category_counts[record.get('category', 'unknown')] += 1
        oracle_record, error = audit_sample(index, record, dataset_root)
        if error is not None:
            error_counts[error] += 1
            error_records.append({
                'sample_index': index,
                'question_id': record.get('question_id', str(index)),
                'image': record.get('image'),
                'error': error,
            })
            continue
        valid_records.append(oracle_record)
        valid_category_counts[oracle_record['category']] += 1
        target_count_counts[str(len(oracle_record['target_objects']))] += 1
        target_coverage_counts[
            'complete' if oracle_record['has_complete_question_target_coverage'] else 'partial'
        ] += 1

    with output_path.open('w') as handle:
        for record in valid_records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = {
        'questions_path': str(questions_path),
        'dataset_root': str(dataset_root),
        'total_samples': len(records),
        'valid_oracle_samples': len(valid_records),
        'invalid_samples': len(error_records),
        'category_counts': dict(sorted(category_counts.items())),
        'valid_category_counts': dict(sorted(valid_category_counts.items())),
        'target_object_count_distribution': dict(sorted(target_count_counts.items())),
        'question_target_coverage': dict(sorted(target_coverage_counts.items())),
        'error_counts': dict(sorted(error_counts.items())),
        'errors': error_records,
        'coordinate_convention': {
            'input_bbox': '[x, y, width, height] in original image pixels',
            'output_bbox': '[x1, y1, x2, y2] normalized to [0, 1]',
            'normalization': 'x1=x/W, y1=y/H, x2=(x+w)/W, y2=(y+h)/H',
        },
    }
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

    print(
        f'Audited {len(records)} VStar samples: {len(valid_records)} valid, '
        f'{len(error_records)} invalid.\n'
        f'Oracle records: {output_path}\n'
        f'Summary: {summary_path}'
    )
    if error_records:
        print('Error counts:', dict(sorted(error_counts.items())))
        sys.exit(1)


if __name__ == '__main__':
    main()
