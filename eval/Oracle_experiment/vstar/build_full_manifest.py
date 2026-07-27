"""Build a unified, evaluator-compatible manifest for all official V* resources.

The published V* repository stores its main 191-question benchmark in
``test_questions.jsonl`` and its two supplementary subsets as per-image JSON
annotations.  This script normalizes all four annotation categories into one
238-row JSONL without modifying the source dataset.

Each output row is compatible with ``VStarJsonlDataset``: ``image`` is a path
relative to the V* root and ``label`` is ``A`` because the paired annotation's
canonical ``options[0]`` is the ground-truth answer convention used by the
repository's existing VStar evaluation path.  Extra provenance fields keep the
original category, annotation, targets, and pixel boxes auditable.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
MAIN_CATEGORIES = ('direct_attributes', 'relative_position')
EXTRA_CATEGORIES = ('OCR', 'GPT4V-hard')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vstar-root', default='/data/zhonggai/VStar')
    parser.add_argument('--main-questions-path', default=None,
                        help='Defaults to <vstar-root>/test_questions.jsonl.')
    parser.add_argument('--output', default='output/vstar/annotations/full_238_manifest.jsonl')
    return parser.parse_args()


def natural_key(path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def image_for_annotation(annotation_path):
    candidates = sorted(
        (
            path for path in annotation_path.parent.iterdir()
            if path.stem == annotation_path.stem and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.suffix.lower(),
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'expected exactly one sibling image for {annotation_path}, found {candidates}'
        )
    return candidates[0]


def read_annotation(annotation_path):
    with annotation_path.open() as handle:
        annotation = json.load(handle)
    if not isinstance(annotation.get('question'), str) or not annotation['question'].strip():
        raise ValueError(f'annotation has no question: {annotation_path}')
    options = annotation.get('options')
    if not isinstance(options, list) or not options or not all(isinstance(item, str) for item in options):
        raise ValueError(f'annotation has invalid options: {annotation_path}')
    return annotation


def canonical_text(question, options):
    choices = ' '.join(f'({chr(ord("A") + index)}) {option}' for index, option in enumerate(options))
    return f'{question} {choices} Answer with the option\'s letter from the given choices directly.'


def manifest_record(root, image_path, annotation_path, annotation, category, subset, question_id,
                    source_jsonl_label=None):
    # PIL verification catches a corrupt image before it enters a long evaluation.
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    options = annotation['options']
    return {
        # Fields consumed by the existing VStar evaluator.
        'image': str(image_path.relative_to(root)),
        'text': canonical_text(annotation['question'], options),
        'category': category,
        'question_id': question_id,
        'label': 'A',
        # Canonical, non-shuffled data used for every row.
        'canonical_label': 'A',
        'canonical_label_index': 0,
        'label_source': 'official_paired_annotation_options[0]',
        'question': annotation['question'],
        'options': options,
        # Provenance and grounding annotations retained for oracle auditing.
        'subset': subset,
        'annotation_path': str(annotation_path.relative_to(root)),
        'target_object': annotation.get('target_object'),
        'bbox_xywh': annotation.get('bbox'),
        'image_size': {'width': image_width, 'height': image_height},
        'source_jsonl_label': source_jsonl_label,
    }


def build_main_records(root, main_questions_path):
    with main_questions_path.open() as handle:
        source_records = [json.loads(line) for line in handle if line.strip()]
    records = []
    for source_index, source_record in enumerate(source_records):
        image_path = root / source_record['image']
        annotation_path = image_path.with_suffix('.json')
        if not image_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError(
                f'main record {source_index} requires image={image_path} and annotation={annotation_path}'
            )
        annotation = read_annotation(annotation_path)
        records.append(manifest_record(
            root, image_path, annotation_path, annotation,
            category=source_record['category'], subset='main',
            question_id=f'main:{source_record["question_id"]}',
            source_jsonl_label=source_record['label'],
        ))
    return records


def build_extra_records(root, category):
    records = []
    for annotation_path in sorted((root / category).glob('*.json'), key=natural_key):
        image_path = image_for_annotation(annotation_path)
        annotation = read_annotation(annotation_path)
        records.append(manifest_record(
            root, image_path, annotation_path, annotation,
            category=category, subset=category,
            question_id=f'{category}:{annotation_path.stem}',
        ))
    return records


def main():
    args = parse_args()
    root = Path(args.vstar_root).resolve()
    main_questions_path = (
        Path(args.main_questions_path).resolve()
        if args.main_questions_path else root / 'test_questions.jsonl'
    )
    output_path = Path(args.output)
    if not root.is_dir():
        raise FileNotFoundError(f'VStar root not found: {root}')
    if not main_questions_path.is_file():
        raise FileNotFoundError(f'main test JSONL not found: {main_questions_path}')

    records = build_main_records(root, main_questions_path)
    for category in EXTRA_CATEGORIES:
        records.extend(build_extra_records(root, category))

    expected_counts = {
        'direct_attributes': 115,
        'relative_position': 76,
        'OCR': 30,
        'GPT4V-hard': 17,
    }
    category_counts = Counter(record['category'] for record in records)
    if category_counts != expected_counts:
        raise RuntimeError(f'unexpected VStar category counts: {dict(category_counts)}')
    if len(records) != 238:
        raise RuntimeError(f'expected 238 records, found {len(records)}')
    question_ids = [record['question_id'] for record in records]
    if len(set(question_ids)) != len(question_ids):
        raise RuntimeError('duplicate question_id in generated manifest')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary_path = output_path.with_suffix('.summary.json')
    summary = {
        'manifest_path': str(output_path),
        'vstar_root': str(root),
        'main_questions_path': str(main_questions_path),
        'total_records': len(records),
        'category_counts': dict(sorted(category_counts.items())),
        'subset_counts': dict(sorted(Counter(record['subset'] for record in records).items())),
        'label_convention': 'all rows are canonical option index 0 / letter A',
        'main_source_label_note': (
            'source_jsonl_label preserves the shuffled multiple-choice label in the official 191-row JSONL; '
            'label is canonicalized to A to match the paired annotation option order.'
        ),
    }
    with summary_path.open('w') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
