"""Build a JSONL manifest from the official GQA Test-Dev balanced questions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--questions-path',
        default='/data/zhonggai/GQA/testdev_balanced_questions.json',
    )
    parser.add_argument('--image-dir', default='/data/zhonggai/GQA/images')
    parser.add_argument(
        '--output',
        default='output/gqa/annotations/testdev_balanced/manifest.jsonl',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    questions_path = Path(args.questions_path)
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    with questions_path.open(encoding='utf-8') as handle:
        questions = json.load(handle)
    if not isinstance(questions, dict) or not questions:
        raise ValueError('questions file must contain a non-empty question dictionary')

    image_sizes = {}
    records = []
    type_counts = Counter()
    missing_images = []
    missing_answers = []
    for sample_index, (question_id, source) in enumerate(questions.items()):
        image_id = str(source['imageId'])
        image_path = image_dir / f'{image_id}.jpg'
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        answer = source.get('answer')
        if not isinstance(answer, str) or not answer.strip():
            missing_answers.append(str(question_id))
            continue
        if image_id not in image_sizes:
            with Image.open(image_path) as image:
                image_sizes[image_id] = {
                    'width': int(image.width),
                    'height': int(image.height),
                }
        types = source.get('types') or {}
        type_counts[str(types.get('structural', 'unknown'))] += 1
        records.append({
            'sample_index': sample_index,
            'question_id': str(question_id),
            'image_id': image_id,
            'image_path': str(image_path),
            'image_size': image_sizes[image_id],
            'question': source['question'],
            'answer': answer,
            'full_answer': source.get('fullAnswer'),
            'types': types,
            'semantic': source.get('semantic', []),
            'semantic_str': source.get('semanticStr'),
            'annotations': source.get('annotations', {}),
            'is_balanced': bool(source.get('isBalanced')),
            'source_split': 'testdev_balanced',
        })

    if missing_images:
        raise FileNotFoundError(
            f'{len(missing_images)} Test-Dev images are missing; first: {missing_images[0]}'
        )
    if missing_answers:
        raise ValueError(
            f'{len(missing_answers)} Test-Dev questions have no answer; '
            f'first: {missing_answers[0]}'
        )
    if len(records) != len(questions):
        raise RuntimeError(
            f'manifest contains {len(records)} of {len(questions)} source questions'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary = {
        'dataset': 'GQA',
        'split': 'testdev_balanced',
        'source_questions_path': str(questions_path),
        'image_dir': str(image_dir),
        'source_question_count': len(questions),
        'manifest_record_count': len(records),
        'unique_image_count': len(image_sizes),
        'all_answers_present': True,
        'all_images_present': True,
        'has_public_gt_object_boxes': False,
        'structural_type_counts': dict(sorted(type_counts.items())),
    }
    summary_path = output_path.with_suffix('.summary.json')
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Test-Dev manifest: {output_path}')
    print(f'Manifest summary: {summary_path}')


if __name__ == '__main__':
    main()
